"""Environment-driven agent configuration and validation."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping

from dotenv import load_dotenv

DEFAULT_MODEL = "gemini-3.5-flash-lite"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_STEPS = 8
DEFAULT_MAX_PLAN_STEPS = 8
DEFAULT_MAX_REPAIR_ATTEMPTS = 2
DEFAULT_VERIFICATION_COMMAND = ""
DEFAULT_COMMAND_TIMEOUT_SECONDS = 60
DEFAULT_MAX_OBSERVATION_CHARS = 6000
DEFAULT_ENABLE_CHECKPOINTS = True
DEFAULT_ROLLBACK_ON_REPAIR_LIMIT = True
# Sized for an 8,000-token provider request limit with headroom (the estimator
# in context_budget.py over-counts, and tool schemas are counted inside it).
DEFAULT_CONTEXT_TOKEN_BUDGET = 25000
MIN_CONTEXT_TOKEN_BUDGET = 1000
MAX_CONTEXT_TOKEN_BUDGET = 1_000_000
# Must match RepoService.MAX_COMMAND_TIMEOUT (service.py). Duplicated here
# rather than imported so this module keeps working without the rest of the
# package's (stdlib-only, but heavier) dependencies - see the module docstring.
MAX_COMMAND_TIMEOUT_SECONDS = 300
MAX_AGENT_STEPS = 100
MAX_PLAN_STEPS = 50
MAX_REPAIR_ATTEMPTS = 10
MAX_OBSERVATION_CHARS = 100_000


class MissingApiKeyError(RuntimeError):
    """GEMINI_API_KEY is not configured."""


class ConfigError(ValueError):
    """An environment variable has an invalid value."""


@dataclass(frozen=True)
class AgentConfig:
    api_key: str | None
    model: str = DEFAULT_MODEL
    temperature: float = DEFAULT_TEMPERATURE
    max_agent_steps: int = DEFAULT_MAX_STEPS
    max_plan_steps: int = DEFAULT_MAX_PLAN_STEPS
    max_observation_chars: int = DEFAULT_MAX_OBSERVATION_CHARS
    max_repair_attempts: int = DEFAULT_MAX_REPAIR_ATTEMPTS
    verification_command: str = DEFAULT_VERIFICATION_COMMAND
    command_timeout_seconds: int = DEFAULT_COMMAND_TIMEOUT_SECONDS
    enable_checkpoints: bool = DEFAULT_ENABLE_CHECKPOINTS
    rollback_on_repair_limit: bool = DEFAULT_ROLLBACK_ON_REPAIR_LIMIT

    def __post_init__(self):
        bounds = {
            "max_agent_steps": (1, MAX_AGENT_STEPS),
            "max_plan_steps": (1, MAX_PLAN_STEPS),
            "max_observation_chars": (1, MAX_OBSERVATION_CHARS),
            "max_repair_attempts": (1, MAX_REPAIR_ATTEMPTS),
            "command_timeout_seconds": (1, MAX_COMMAND_TIMEOUT_SECONDS),
        }
        for name, (low, high) in bounds.items():
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or not low <= value <= high:
                raise ConfigError(f"{name.upper()} must be between {low} and {high}, got {value!r}")


def load_config(require_api_key: bool = True, env: Mapping[str, str] | None = None) -> AgentConfig:
    """Build an AgentConfig. With env=None, reads a .env file (if any) and os.environ."""
    if env is None:
        load_dotenv()  # does not override variables already set in the environment
        env = os.environ

    # Load Gemini API key (GEMINI_API_KEY or GOOGLE_API_KEY).
    api_key = (
        env.get("GEMINI_API_KEY")
        or env.get("GOOGLE_API_KEY")
        or ""
    ).strip() or None
    if require_api_key and api_key is None:
        raise MissingApiKeyError(
            "GEMINI_API_KEY is not set. Copy .env.example to .env, add your Gemini API key "
            "(https://aistudio.google.com/apikey), or export GEMINI_API_KEY in your shell. "
            "GOOGLE_API_KEY is also accepted."
        )

    model = (env.get("GEMINI_MODEL") or "").strip() or DEFAULT_MODEL
    temperature_raw = (env.get("GEMINI_TEMPERATURE") or "").strip()
    try:
        temperature = float(temperature_raw) if temperature_raw else DEFAULT_TEMPERATURE
    except ValueError:
        raise ConfigError(f"GEMINI_TEMPERATURE must be a float, got {temperature_raw!r}") from None
    max_steps = _parse(env, "MAX_AGENT_STEPS", int, DEFAULT_MAX_STEPS)
    max_plan_steps = _parse(env, "MAX_PLAN_STEPS", int, DEFAULT_MAX_PLAN_STEPS)
    max_observation_chars = _parse(env, "MAX_OBSERVATION_CHARS", int, DEFAULT_MAX_OBSERVATION_CHARS)
    max_repair_attempts = _parse(env, "MAX_REPAIR_ATTEMPTS", int, DEFAULT_MAX_REPAIR_ATTEMPTS)
    verification_command = (env.get("VERIFICATION_COMMAND") or "").strip() or DEFAULT_VERIFICATION_COMMAND
    command_timeout_seconds = _parse(env, "COMMAND_TIMEOUT_SECONDS", int, DEFAULT_COMMAND_TIMEOUT_SECONDS)
    enable_checkpoints = _parse_bool(env, "ENABLE_CHECKPOINTS", DEFAULT_ENABLE_CHECKPOINTS)
    rollback_on_repair_limit = _parse_bool(env, "ROLLBACK_ON_REPAIR_LIMIT", DEFAULT_ROLLBACK_ON_REPAIR_LIMIT)
    if not 0.0 <= temperature <= 2.0:
        raise ConfigError(f"GEMINI_TEMPERATURE must be between 0 and 2, got {temperature}")
    load_context_token_budget(env)  # fail fast on an invalid value; used at request time
    return AgentConfig(api_key=api_key, model=model, temperature=temperature,
                       max_agent_steps=max_steps, max_plan_steps=max_plan_steps,
                       max_observation_chars=max_observation_chars,
                       max_repair_attempts=max_repair_attempts, verification_command=verification_command,
                       command_timeout_seconds=command_timeout_seconds,
                       enable_checkpoints=enable_checkpoints,
                       rollback_on_repair_limit=rollback_on_repair_limit)


def load_context_token_budget(env: Mapping[str, str] | None = None) -> int:
    """MODEL_CONTEXT_TOKEN_BUDGET, validated (default DEFAULT_CONTEXT_TOKEN_BUDGET)."""
    env = os.environ if env is None else env
    value = _parse(env, "MODEL_CONTEXT_TOKEN_BUDGET", int, DEFAULT_CONTEXT_TOKEN_BUDGET)
    if not MIN_CONTEXT_TOKEN_BUDGET <= value <= MAX_CONTEXT_TOKEN_BUDGET:
        raise ConfigError(f"MODEL_CONTEXT_TOKEN_BUDGET must be between {MIN_CONTEXT_TOKEN_BUDGET} "
                          f"and {MAX_CONTEXT_TOKEN_BUDGET}, got {value!r}")
    return value


def _parse(env: Mapping[str, str], name: str, cast, default):
    raw = (env.get(name) or "").strip()
    if not raw:
        return default
    try:
        return cast(raw)
    except ValueError:
        raise ConfigError(f"{name} must be a {cast.__name__}, got {raw!r}") from None


_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def _parse_bool(env: Mapping[str, str], name: str, default: bool) -> bool:
    raw = (env.get(name) or "").strip().lower()
    if not raw:
        return default
    if raw in _TRUE:
        return True
    if raw in _FALSE:
        return False
    raise ConfigError(f"{name} must be a boolean (true/false), got {raw!r}")
