"""A bounded repository-modification agent.

The Coder uses a bounded tool loop with read and write repository tools.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from ..service import RepoService
from .config import AgentConfig, load_config
from .model import build_llm
from .prompts import CODER_STEP_LIMIT_NOTICE, CODER_SYSTEM_PROMPT
from .state import AgentState
from .tools import build_coder_tools
from .utils import bounded_model_history, final_answer, route_after_agent, tool_evidence


def coder_report(messages: list) -> str:
    """Return the Coder's final report from a completed Coder graph."""
    return final_answer(messages)


def build_coder(service: RepoService, config: AgentConfig | None = None, llm=None):
    """Build the Coder LangGraph with read and write repository tools."""
    if config is None:
        config = load_config(require_api_key=llm is None)
    if llm is None:
        llm = build_llm(config)

    tools = build_coder_tools(service, config.max_observation_chars)
    llm_with_tools = llm.bind_tools(tools)

    def coder_node(state: AgentState) -> dict:
        steps = state.get("step_count", 0)
        try:
            if steps >= config.max_coder_steps:
                question = next((str(m.content) for m in state["messages"] if getattr(m, "type", "") == "human"), "")
                reply = llm.invoke([
                    SystemMessage(CODER_SYSTEM_PROMPT),
                    HumanMessage(CODER_STEP_LIMIT_NOTICE.format(
                        max_steps=config.max_coder_steps,
                        question=question,
                        evidence=tool_evidence(state["messages"]),
                    )),
                ])
                reply = AIMessage(content=reply.content)
            else:
                model_history = bounded_model_history(
                    state["messages"], max_turns=min(config.max_coder_steps, 6))
                reply = llm_with_tools.invoke([SystemMessage(CODER_SYSTEM_PROMPT)] + model_history)
            return {"messages": [reply], "step_count": steps + 1}
        except Exception as exc:
            return {"messages": [AIMessage(content=f"Coder model call failed ({type(exc).__name__})." )],
                    "step_count": steps + 1}

    builder = StateGraph(AgentState)
    builder.add_node("coder", coder_node)
    builder.add_node("tools", ToolNode(tools, handle_tool_errors=False))
    builder.add_edge(START, "coder")
    builder.add_conditional_edges("coder", route_after_agent, {"tools": "tools", "end": END})
    builder.add_edge("tools", "coder")
    return builder.compile().with_config(recursion_limit=2 * config.max_coder_steps + 5)
