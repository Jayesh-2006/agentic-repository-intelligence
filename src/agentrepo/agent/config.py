"""
Agent configuration, read from environment variables (and a local .env file).

    GROQ_API_KEY       required to run the real agent
    GROQ_MODEL         default: openai/gpt-oss-120b (a Groq model with tool calling)
    GROQ_TEMPERATURE   default: 0
    MAX_AGENT_STEPS       default: 8  (max tool-using LLM turns before a forced final answer)
    MAX_PLAN_STEPS        default: 8  (max steps the Deep Agent's upfront planning call may
                                       propose, and max PENDING steps a revise_plan call may set)
    MAX_EXPLORER_STEPS    default: 6  (max tool-using turns for the Explorer sub-agent)
    MAX_CODER_STEPS       default: 6  (max tool-using turns for the Coder sub-agent)
    MAX_DEBUGGER_STEPS    default: 6  (max tool-using turns for the Debugger sub-agent)
    MAX_REPAIR_ATTEMPTS   default: 2  (max Debugger invocations in the verification/repair loop)
    Hard ceilings: agent/sub-agent iterations 100, plan steps 50, repair attempts 10,
                   observation characters 100,000, command timeout 300 seconds.
    VERIFICATION_COMMAND  optional repository-specific verification command
    COMMAND_TIMEOUT_SECONDS default: 60  (timeout for the automatic verification command)
    ENABLE_CHECKPOINTS    default: true  (roll the repository back after repair limit;
                                          a baseline snapshot is always taken to detect command mutations)
    ROLLBACK_ON_REPAIR_LIMIT default: true  (when the repair loop exhausts
                                          MAX_REPAIR_ATTEMPTS and verification is still
                                          failing, restore the repository to its
                                          pre-task checkpoint)

This module only requires python-dotenv; it can load configuration without
LangChain, LangGraph, or an API key.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping

from dotenv import load_dotenv

DEFAULT_MODEL = "openai/gpt-oss-120b"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_STEPS = 8
DEFAULT_MAX_PLAN_STEPS = 8
DEFAULT_MAX_EXPLORER_STEPS = 6
DEFAULT_MAX_CODER_STEPS = 6
DEFAULT_MAX_DEBUGGER_STEPS = 6
DEFAULT_MAX_REPAIR_ATTEMPTS = 2
DEFAULT_VERIFICATION_COMMAND = ""
DEFAULT_COMMAND_TIMEOUT_SECONDS = 60
DEFAULT_MAX_OBSERVATION_CHARS = 6000
DEFAULT_ENABLE_CHECKPOINTS = True
DEFAULT_ROLLBACK_ON_REPAIR_LIMIT = True
# Must match RepoService.MAX_COMMAND_TIMEOUT (service.py). Duplicated here
# rather than imported so this module keeps working without the rest of the
# package's (stdlib-only, but heavier) dependencies - see the module docstring.
MAX_COMMAND_TIMEOUT_SECONDS = 300
MAX_AGENT_STEPS = 100
MAX_PLAN_STEPS = 50
MAX_SUBAGENT_STEPS = 100
MAX_REPAIR_ATTEMPTS = 10
MAX_OBSERVATION_CHARS = 100_000


class MissingApiKeyError(RuntimeError):
    """GROQ_API_KEY is not configured."""


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
    max_explorer_steps: int = DEFAULT_MAX_EXPLORER_STEPS
    max_coder_steps: int = DEFAULT_MAX_CODER_STEPS
    max_debugger_steps: int = DEFAULT_MAX_DEBUGGER_STEPS
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
            "max_explorer_steps": (1, MAX_SUBAGENT_STEPS),
            "max_coder_steps": (1, MAX_SUBAGENT_STEPS),
            "max_debugger_steps": (1, MAX_SUBAGENT_STEPS),
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

    api_key = (env.get("GROQ_API_KEY") or "").strip() or None
    if require_api_key and api_key is None:
        raise MissingApiKeyError(
            "GROQ_API_KEY is not set. Copy .env.example to .env, put your Groq API key in it "
            "(https://console.groq.com/keys), or export GROQ_API_KEY in your shell."
        )

    model = (env.get("GROQ_MODEL") or "").strip() or DEFAULT_MODEL
    temperature = _parse(env, "GROQ_TEMPERATURE", float, DEFAULT_TEMPERATURE)
    max_steps = _parse(env, "MAX_AGENT_STEPS", int, DEFAULT_MAX_STEPS)
    max_plan_steps = _parse(env, "MAX_PLAN_STEPS", int, DEFAULT_MAX_PLAN_STEPS)
    max_observation_chars = _parse(env, "MAX_OBSERVATION_CHARS", int, DEFAULT_MAX_OBSERVATION_CHARS)
    max_explorer_steps = _parse(env, "MAX_EXPLORER_STEPS", int, DEFAULT_MAX_EXPLORER_STEPS)
    max_coder_steps = _parse(env, "MAX_CODER_STEPS", int, DEFAULT_MAX_CODER_STEPS)
    max_debugger_steps = _parse(env, "MAX_DEBUGGER_STEPS", int, DEFAULT_MAX_DEBUGGER_STEPS)
    max_repair_attempts = _parse(env, "MAX_REPAIR_ATTEMPTS", int, DEFAULT_MAX_REPAIR_ATTEMPTS)
    verification_command = (env.get("VERIFICATION_COMMAND") or "").strip() or DEFAULT_VERIFICATION_COMMAND
    command_timeout_seconds = _parse(env, "COMMAND_TIMEOUT_SECONDS", int, DEFAULT_COMMAND_TIMEOUT_SECONDS)
    enable_checkpoints = _parse_bool(env, "ENABLE_CHECKPOINTS", DEFAULT_ENABLE_CHECKPOINTS)
    rollback_on_repair_limit = _parse_bool(env, "ROLLBACK_ON_REPAIR_LIMIT", DEFAULT_ROLLBACK_ON_REPAIR_LIMIT)
    if not 0.0 <= temperature <= 2.0:
        raise ConfigError(f"GROQ_TEMPERATURE must be between 0 and 2, got {temperature}")
    return AgentConfig(api_key=api_key, model=model, temperature=temperature,
                       max_agent_steps=max_steps, max_plan_steps=max_plan_steps,
                       max_observation_chars=max_observation_chars,
                       max_explorer_steps=max_explorer_steps,
                       max_coder_steps=max_coder_steps, max_debugger_steps=max_debugger_steps,
                       max_repair_attempts=max_repair_attempts, verification_command=verification_command,
                       command_timeout_seconds=command_timeout_seconds,
                       enable_checkpoints=enable_checkpoints,
                       rollback_on_repair_limit=rollback_on_repair_limit)


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
