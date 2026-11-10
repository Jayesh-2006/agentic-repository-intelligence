"""Agent construction and public helpers."""

from .config import AgentConfig, ConfigError, MissingApiKeyError, load_config
from .utils import final_answer


def build_deep_agent(service, config=None, llm=None, **_legacy_kwargs):
    """Build the Main Agent: a bounded, plan-driven single-agent orchestration
    graph with direct access to all read-only repository tools, plan revision
    (revise_plan), and capability-gated modification and verification workflows
    (write_file, verify_repository, run_command). See deep_agent.build_deep_agent."""
    from .deep_agent import build_deep_agent as _build_deep_agent
    return _build_deep_agent(service, config=config, llm=llm)


__all__ = [
    "build_deep_agent",
    "final_answer",
    "load_config",
    "AgentConfig",
    "MissingApiKeyError",
    "ConfigError",
]
