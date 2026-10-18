"""Agent construction and public helpers.

The application runs a single primary LangGraph agent (`build_deep_agent`)
with direct access to repository search, structural analysis, file-reading,
plan-revision, and guarded modification/verification tools.

    from agentrepo import RepoService
    from agentrepo.agent import build_deep_agent, final_answer

    agent = build_deep_agent(RepoService.from_repository("path/to/repo"))
    result = agent.invoke({"messages": [{"role": "user", "content": "Who calls save?"}]})
    print(final_answer(result["messages"]))

Configuration can be loaded without LangChain; graph dependencies are
imported when `build_deep_agent` is called.
"""

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
