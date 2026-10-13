"""Agent construction and public helpers.

The application runs the Deep Agent, which coordinates repository
investigation, code changes, and verification through delegation tools.

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


def build_explorer(service, config=None, llm=None):
    """Build the read-only Explorer sub-agent. See explorer.build_explorer."""
    from .explorer import build_explorer as _build_explorer
    return _build_explorer(service, config=config, llm=llm)


def build_coder(service, config=None, llm=None):
    """Build the read/write Coder sub-agent. See coder.build_coder."""
    from .coder import build_coder as _build_coder
    return _build_coder(service, config=config, llm=llm)


def build_debugger(service, config=None, llm=None):
    """Build the Debugger sub-agent (read/write + run_command). See debugger.build_debugger."""
    from .debugger import build_debugger as _build_debugger
    return _build_debugger(service, config=config, llm=llm)


def build_deep_agent(service, config=None, llm=None, explorer_llm=None, coder_llm=None, debugger_llm=None):
    """Build the Deep Agent: a bounded, plan-driven autonomous orchestration
    graph that coordinates the existing Explorer/Coder/Debugger via the same
    delegation tools the Main Agent uses (investigate_repository,
    code_repository, verify_repository), plus its own plan-revision tool
    (revise_plan). See deep_agent.build_deep_agent."""
    from .deep_agent import build_deep_agent as _build_deep_agent
    return _build_deep_agent(service, config=config, llm=llm, explorer_llm=explorer_llm, coder_llm=coder_llm,
                              debugger_llm=debugger_llm)


__all__ = [
    "build_explorer", "build_coder", "build_debugger", "build_deep_agent", "final_answer",
    "load_config", "AgentConfig", "MissingApiKeyError", "ConfigError",
]
