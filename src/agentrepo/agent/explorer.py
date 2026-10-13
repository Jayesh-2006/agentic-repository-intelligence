"""A bounded, read-only repository investigation agent."""

from __future__ import annotations

import json

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from ..service import RepoService
from .config import AgentConfig, load_config
from .model import build_llm
from .prompts import EXPLORER_STEP_LIMIT_NOTICE, EXPLORER_SYSTEM_PROMPT
from .state import AgentState
from .tools import build_repo_tools
from .utils import bounded_model_history, final_answer, route_after_agent, tool_evidence


def explorer_report(messages: list, *, max_report_chars: int = 12000,
                    max_raw_observations: int = 6, max_raw_chars: int = 500) -> str:
    """Return a bounded, provenance-preserving Explorer result.

    The top-level report is model-generated.  The bounded `raw_observations`
    list retains representative tool outputs so the parent agent/grader can
    distinguish what the Explorer concluded from what repository tools actually
    returned.
    """
    report = final_answer(messages)
    if len(report) > max_report_chars:
        report = report[: max_report_chars - 80] + " ...[report truncated]"
    calls = {}
    observations = []
    for message in messages:
        for call in getattr(message, "tool_calls", None) or []:
            calls[call.get("id")] = call
        if getattr(message, "type", "") != "tool":
            continue
        call = calls.get(getattr(message, "tool_call_id", None), {})
        content = str(getattr(message, "content", ""))
        if len(content) > max_raw_chars:
            content = content[:max_raw_chars] + " ...[observation truncated]"
        args_text = json.dumps(call.get("args") or {}, ensure_ascii=False, separators=(",", ":"))
        if len(args_text) > 300:
            args_text = args_text[:300] + " ...[args truncated]"
        observations.append({
            "tool": call.get("name") or getattr(message, "name", None) or "tool",
            "args": args_text,
            "observation": content,
        })
    observations = observations[-max_raw_observations:]
    return json.dumps({
        "report": report,
        "report_kind": "model_generated_summary",
        "raw_observations": observations,
    }, ensure_ascii=False)


def build_explorer(service: RepoService, config: AgentConfig | None = None, llm=None):
    """Build the Explorer LangGraph with read-only repository tools."""
    if config is None:
        config = load_config(require_api_key=llm is None)
    if llm is None:
        llm = build_llm(config)

    tools = build_repo_tools(service, config.max_observation_chars)
    llm_with_tools = llm.bind_tools(tools)

    def explorer_node(state: AgentState) -> dict:
        steps = state.get("step_count", 0)
        try:
            if steps >= config.max_explorer_steps:
                question = next((str(m.content) for m in state["messages"] if getattr(m, "type", "") == "human"), "")
                reply = llm.invoke([
                    SystemMessage(EXPLORER_SYSTEM_PROMPT),
                    HumanMessage(EXPLORER_STEP_LIMIT_NOTICE.format(
                        max_steps=config.max_explorer_steps,
                        question=question,
                        evidence=tool_evidence(state["messages"]),
                    )),
                ])
                reply = AIMessage(content=reply.content)
            else:
                model_history = bounded_model_history(
                    state["messages"], max_turns=min(config.max_explorer_steps, 6))
                reply = llm_with_tools.invoke([SystemMessage(EXPLORER_SYSTEM_PROMPT)] + model_history)
            return {"messages": [reply], "step_count": steps + 1}
        except Exception as exc:
            code = getattr(exc, "status_code", None)
            detail = type(exc).__name__ + (f", HTTP {code}" if code else "")
            return {"messages": [AIMessage(content=f"Explorer model call failed ({detail}).")],
                    "step_count": steps + 1}

    builder = StateGraph(AgentState)
    builder.add_node("explorer", explorer_node)
    builder.add_node("tools", ToolNode(tools, handle_tool_errors=False))
    builder.add_edge(START, "explorer")
    builder.add_conditional_edges("explorer", route_after_agent, {"tools": "tools", "end": END})
    builder.add_edge("tools", "explorer")
    return builder.compile().with_config(recursion_limit=2 * config.max_explorer_steps + 5)
