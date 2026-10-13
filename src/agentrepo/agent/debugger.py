"""A bounded repository-repair agent.

The verification and repair loop invokes the Debugger when a Coder change
fails verification. It can inspect and modify files, then run a command to
check its repair.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from ..service import RepoService
from .config import AgentConfig, load_config
from .model import build_llm
from .prompts import DEBUGGER_STEP_LIMIT_NOTICE, DEBUGGER_SYSTEM_PROMPT
from .state import AgentState
from .tools import build_debugger_tools
from .utils import bounded_model_history, final_answer, route_after_agent, tool_evidence


def debugger_report(messages: list) -> str:
    """Return the Debugger's final report from a completed Debugger graph."""
    return final_answer(messages)


def build_debugger(service: RepoService, config: AgentConfig | None = None, llm=None):
    """Build the Debugger LangGraph with repository and verification tools."""
    if config is None:
        config = load_config(require_api_key=llm is None)
    if llm is None:
        llm = build_llm(config)

    tools = build_debugger_tools(service, config.max_observation_chars)
    llm_with_tools = llm.bind_tools(tools)

    def debugger_node(state: AgentState) -> dict:
        steps = state.get("step_count", 0)
        try:
            if steps >= config.max_debugger_steps:
                question = next((str(m.content) for m in state["messages"] if getattr(m, "type", "") == "human"), "")
                reply = llm.invoke([
                    SystemMessage(DEBUGGER_SYSTEM_PROMPT),
                    HumanMessage(DEBUGGER_STEP_LIMIT_NOTICE.format(
                        max_steps=config.max_debugger_steps,
                        question=question,
                        evidence=tool_evidence(state["messages"]),
                    )),
                ])
                reply = AIMessage(content=reply.content)
            else:
                model_history = bounded_model_history(
                    state["messages"], max_turns=min(config.max_debugger_steps, 6))
                reply = llm_with_tools.invoke([SystemMessage(DEBUGGER_SYSTEM_PROMPT)] + model_history)
            return {"messages": [reply], "step_count": steps + 1}
        except Exception as exc:
            return {"messages": [AIMessage(content=f"Debugger model call failed ({type(exc).__name__}).")],
                    "step_count": steps + 1}

    builder = StateGraph(AgentState)
    builder.add_node("debugger", debugger_node)
    builder.add_node("tools", ToolNode(tools, handle_tool_errors=False))
    builder.add_edge(START, "debugger")
    builder.add_conditional_edges("debugger", route_after_agent, {"tools": "tools", "end": END})
    builder.add_edge("tools", "debugger")
    return builder.compile().with_config(recursion_limit=2 * config.max_debugger_steps + 5)
