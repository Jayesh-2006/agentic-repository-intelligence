"""
The LangGraph agent: an explicit  agent -> tools -> agent  loop.

    START -> agent --(tool calls?)--> tools --+
               ^          |                   |
               |          no                  |
               +----------|------------------+
                          v
                         END

- `agent` calls the Groq LLM (bound to the ten repository tools and the
  Explorer delegation tool). The model
  decides whether to call tools (AIMessage.tool_calls) or to answer.
- `tools` (LangGraph's ToolNode) runs the requested tools and appends the
  results as ToolMessages. The graph then goes back to `agent`, so the model
  sees every observation and chooses the next step.
- Step limit: after `max_agent_steps` tool-using turns the agent node makes one
  last call WITHOUT tools and asks for the best answer from the evidence so far.
  LangGraph's own recursion_limit is set slightly above that as a hard backstop.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from ..service import RepoService
from .config import AgentConfig, load_config
from .delegation import build_delegation_tools
from .model import build_llm
from .prompts import SYSTEM_PROMPT
from .state import AgentState
from .tools import build_repo_tools
from .utils import route_after_agent, wrap_up_prompt


def build_agent(service: RepoService, config: AgentConfig | None = None, llm=None, explorer_llm=None,
                 coder_llm=None, debugger_llm=None):
    """
    Return a compiled agent. Use it like any LangGraph graph:
        agent.invoke({"messages": [{"role": "user", "content": "..."}]})
        agent.stream(..., stream_mode="updates")

    llm, explorer_llm, coder_llm and debugger_llm are optional and mainly for
    tests (any LangChain chat model with bind_tools). Without them, the Groq
    model from the configuration is used. explorer_llm/coder_llm/debugger_llm
    let deterministic tests script a sub-agent independently from the Main
    Agent; debugger_llm defaults to coder_llm (or llm) since the Debugger is
    a variant of the same "inspect and write" role as the Coder.
    """
    if config is None:
        config = load_config(require_api_key=llm is None)
    if llm is None:
        llm = build_llm(config)

    tools = build_repo_tools(service, config.max_observation_chars)
    tools += build_delegation_tools(service, config, explorer_llm=explorer_llm or llm, coder_llm=coder_llm or llm,
                                     debugger_llm=debugger_llm or coder_llm or llm)
    llm_with_tools = llm.bind_tools(tools)

    def agent_node(state: AgentState) -> dict:
        steps = state.get("step_count", 0)
        if steps >= config.max_agent_steps:
            # Out of budget: no tools, plain-text evidence, ask for the best answer.
            reply = llm.invoke([SystemMessage(SYSTEM_PROMPT),
                                HumanMessage(wrap_up_prompt(state["messages"], config.max_agent_steps))])
            reply = AIMessage(content=reply.content)  # guarantee: no tool calls, so the graph ends
        else:
            reply = llm_with_tools.invoke([SystemMessage(SYSTEM_PROMPT)] + state["messages"])
        return {"messages": [reply], "step_count": steps + 1}

    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    # handle_tool_errors=False: expected problems already arrive as {"error": ...}
    # observations, and bad LLM arguments are handled in tools.py, so any other
    # exception is a real bug and should fail loudly.
    builder.add_node("tools", ToolNode(tools, handle_tool_errors=False))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", route_after_agent, {"tools": "tools", "end": END})
    builder.add_edge("tools", "agent")

    # agent + tools = 2 graph steps per turn; +1 for the wrap-up call; +margin.
    return builder.compile().with_config(recursion_limit=2 * config.max_agent_steps + 5)
