"""
Small pure helpers for the agent layer. No LangChain / LangGraph imports:
they only read attributes (`type`, `content`, `tool_calls`, ...) from message
objects, so they are easy to test and easy to explain.
"""

from __future__ import annotations

import json

from .plan import describe_step
from .prompts import STEP_LIMIT_NOTICE


# A collection this long inside a single tool result is bounded
# before the result is ever serialized (see _bound_collections below), so a
# repository discovery/investigation call that surfaces thousands of paths
# or other repetitive entries cannot blow up the request on its own. This is
# generic - it looks only at list length, never at directory names, file
# extensions or any other repository-specific content.
DEFAULT_MAX_LIST_ITEMS = 40


def _bound_collections(value, max_items: int = DEFAULT_MAX_LIST_ITEMS):
    """Recursively cap long lists anywhere inside a tool result dict.

    A list at or under `max_items` is returned unchanged (in place and
    order) - small/normal results are byte-for-byte what they always were.
    A longer list keeps its first `max_items` entries (the same ones that
    would otherwise have appeared first, so ordering/ranking done by the
    repository tool - e.g. find_files' name-match-first ordering - is
    preserved) and is replaced by a small dict that says how many entries
    were kept, how many were left out and the true total, so the omission is
    explicit rather than a result that quietly looks complete. This runs
    before JSON serialization, so it bounds the data itself rather than
    chopping the final text at an arbitrary character offset.
    """
    if isinstance(value, list):
        kept = [_bound_collections(item, max_items) for item in value[:max_items]]
        if len(value) <= max_items:
            return kept
        return {
            "items": kept,
            "shown": len(kept),
            "omitted": len(value) - len(kept),
            "total": len(value),
            "note": (f"{len(value) - len(kept)} more entries omitted (showing "
                     f"{len(kept)} of {len(value)}); narrow the query/pattern "
                     "or ask about a specific entry to see more."),
        }
    if isinstance(value, dict):
        return {key: _bound_collections(item, max_items) for key, item in value.items()}
    return value


def to_observation(result: dict, max_chars: int, max_list_items: int = DEFAULT_MAX_LIST_ITEMS) -> str:
    """Turn a repository tool result (a dict, possibly {"error": ...}) into the
    text the LLM sees. Unusually large collections inside the result (e.g. a
    repository discovery call returning thousands of paths or other
    repetitive entries) are bounded to representative, clearly-labelled
    entries first (`_bound_collections`), then the result is compacted to
    JSON and, only if it is still oversized (e.g. a single very large text
    field), cut at max_chars as a last-resort safety net so one result cannot
    flood the model's context."""
    bounded = _bound_collections(result, max_list_items) if isinstance(result, dict) else result
    text = json.dumps(bounded, ensure_ascii=False, separators=(",", ":"))
    if len(text) <= max_chars:
        return text
    if _is_get_file_result(result):
        file_result = _bound_get_file_result(result, max_chars)
        text = json.dumps(file_result, ensure_ascii=False, separators=(",", ":"))
        if len(text) <= max_chars:
            return text
    notice = (f"... [truncated: showing {max_chars} of {len(text)} characters; "
              "ask for a narrower line range, symbol or query]")
    if len(notice) >= max_chars:
        return notice[:max_chars]
    return text[:max_chars - len(notice)] + notice


def _is_get_file_result(result) -> bool:
    return (isinstance(result, dict) and isinstance(result.get("file"), str)
            and isinstance(result.get("content"), str) and "line_start" in result)


def _bound_get_file_result(result: dict, max_chars: int) -> dict:
    """Keep get_file observations valid JSON and align ranges with delivered lines.

    Character truncation after JSON serialization used to hide part of a
    successful service read while leaving its original line_end/next_start_line
    visible. That could make the model skip lines it never received and made
    range provenance an estimate. Bound at whole-line boundaries instead.
    """
    data = dict(result)
    lines = str(result.get("content", "")).splitlines()
    start = result.get("line_start") or 1
    data["truncated"] = True
    data["observation_truncated"] = True

    def candidate(count: int) -> dict:
        bounded = dict(data)
        bounded["content"] = "\n".join(lines[:count])
        bounded["line_end"] = start + count - 1 if count else start - 1
        bounded["next_start_line"] = start + count
        return bounded

    low, high = 0, len(lines)
    best = candidate(0)
    while low <= high:
        middle = (low + high) // 2
        current = candidate(middle)
        size = len(json.dumps(current, ensure_ascii=False, separators=(",", ":")))
        if size <= max_chars:
            best = current
            low = middle + 1
        else:
            high = middle - 1
    return best


def bounded_model_history(messages: list, task: str | None = None, max_turns: int = 6) -> list:
    """Return a bounded recent conversation view for an LLM call.

    The full state history remains intact for evidence accounting and tool-result
    correlation.  Only the model-facing view is reduced: the original task is
    retained, then the most recent AI-turn groups (AI tool request + following
    tool/check messages). This prevents context growth from being proportional
    to the total number of completed investigation turns.
    """
    if not messages:
        return []
    max_turns = max(1, int(max_turns))
    first_human = None
    if task:
        for message in messages:
            if getattr(message, "type", "") == "human":
                first_human = message
                break
    if first_human is None:
        first_human = next((m for m in messages if getattr(m, "type", "") == "human"), None)

    groups = []
    current = None
    for message in messages:
        kind = getattr(message, "type", "")
        if kind == "ai":
            if current:
                groups.append(current)
            current = [message]
        elif current is not None:
            current.append(message)
    if current:
        groups.append(current)

    selected = groups[-max_turns:]
    flattened = [message for group in selected for message in group]
    if first_human is not None and all(first_human is not m for m in flattened):
        return [first_human, *flattened]
    return flattened


def route_after_agent(state: dict) -> str:
    """Conditional edge: if the last AI message asked for tools, run them;
    otherwise the model has given its final answer."""
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else "end"


def wrap_up_prompt(messages: list, max_steps: int, obs_chars: int = 1200, max_items: int = 12,
                   task: str | None = None) -> str:
    """Prompt used when the step limit is hit: the question plus a plain-text
    digest of the tool evidence, so the model can still answer.

    `task`, when given, is used as "the question" instead of
    scanning `messages` for the last HumanMessage. The Deep Agent now
    appends its own system-authored HumanMessage notes mid-run (hold-back
    feedback, blind-repeat/redundant-investigation skips) - once one of
    those is the most recent human-typed message, "last human message"
    silently stops meaning "the user's original question". Callers that
    track the original task separately (deep_agent.py) should pass it;
    callers that don't (the Main Agent's graph.py) keep the old
    last-HumanMessage behaviour by simply omitting it."""
    question = task if task is not None else next(
        (str(m.content) for m in reversed(messages) if getattr(m, "type", "") == "human"), "")
    evidence = tool_evidence(messages, obs_chars, max_items)
    return STEP_LIMIT_NOTICE.format(max_steps=max_steps, question=question, evidence=evidence)


def tool_evidence(messages: list, obs_chars: int = 1200, max_items: int = 12) -> str:
    """A bounded plain-text digest of tool results, for a final no-tools LLM call."""
    calls: dict[str, tuple[str, dict]] = {}
    lines: list[str] = []
    for m in messages:
        for call in getattr(m, "tool_calls", None) or []:
            calls[call["id"]] = (call["name"], call["args"])
        if getattr(m, "type", "") == "tool":
            name, args = calls.get(getattr(m, "tool_call_id", None), (getattr(m, "name", None) or "tool", {}))
            lines.append(f"- {name}({json.dumps(args, ensure_ascii=False)}) -> {str(m.content)[:obs_chars]}")
    return "\n".join(lines[-max_items:]) or "(no tool results)"


def final_answer(messages: list) -> str:
    """Text of the last AI message that is not a tool request.

    Accepts either real BaseMessage objects (``.type``/``.content``, the
    production shape returned by the compiled graph) or plain
    ``{"role": ..., "content": ...}`` dicts, so a caller that hands back
    dict-shaped messages does not silently resolve to an empty answer.
    """
    for m in reversed(messages):
        if isinstance(m, dict):
            is_ai = m.get("role") in ("assistant", "ai")
            tool_calls = m.get("tool_calls")
            content = m.get("content")
        else:
            is_ai = getattr(m, "type", "") == "ai"
            tool_calls = getattr(m, "tool_calls", None)
            content = m.content
        if is_ai and not tool_calls:
            if isinstance(content, list):  # some providers return content blocks
                content = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
            return str(content)
    return ""


def summarize_observation(content) -> str:
    """One short line describing a tool result, for the console trace."""
    try:
        data = json.loads(content)
    except (TypeError, ValueError):
        return str(content).replace("\n", " ")[:80] + ("..." if len(str(content)) > 80 else "")
    if isinstance(data, dict):
        error = data.get("error")
        if isinstance(error, dict):
            return f"ERROR {error.get('code')}: {str(error.get('message', ''))[:70]}"
        counts = [f"{k}: {len(v)}" for k, v in data.items() if isinstance(v, list)]
        if counts:
            return ", ".join(counts)
        return str(data.get("symbol") or data.get("file") or "ok")
    return "ok"


def describe_update(node: str, update: dict) -> list[str]:
    """Console-trace lines for one LangGraph `updates` chunk ({node: state_update})."""
    lines: list[str] = []
    messages = update.get("messages", [])
    # A Deep Agent turn that holds an answer back or skips a blind
    # repeat pairs the AI reply with a system-check human message; that reply
    # is not a final answer, and the check message is what the trace shows.
    held_back = any(getattr(m, "type", "") == "human" for m in messages)
    for m in messages:
        kind = getattr(m, "type", "")
        if kind == "ai":
            calls = getattr(m, "tool_calls", None) or []
            for call in calls:
                args = json.dumps(call["args"], ensure_ascii=False)
                lines.append(f"[Agent] Calling {call['name']}({args if len(args) <= 110 else args[:107] + '...'})")
            if not calls and not held_back:
                lines.append("[Agent] Final answer")
        elif kind == "human":
            lines.append(f"[Check] {str(m.content).splitlines()[0][:110]}")
        elif kind == "tool":
            lines.append(f"[Tool] {getattr(m, 'name', 'tool')} -> {summarize_observation(m.content)}")
    plan = update.get("plan")
    if plan and plan.get("steps"):
        # Only the Deep Agent's `decide`/`update_state` updates
        # ever carry a "plan" key (see deep_agent.py) - the Main Agent's
        # plain agent_node (graph.py) never does, so this is a no-op for it.
        lines.append(f"[Plan] {describe_step(plan['steps'][-1])}")
        # The most recently recorded step evidence (`order` is its
        # execution sequence number, see evidence.record_step_evidence).
        recorded = [s for s in plan["steps"] if s.get("result")]
        if node == "update_state" and recorded:
            latest = max(recorded, key=lambda s: s["order"])
            lines.append(f"[Evidence] {latest['step_id']} {latest['action'][:90]} -> {latest['result']}")
    if update.get("model_error"):
        lines.append(f"[Error] {str(update['model_error'])[:300]}")
    completion = update.get("completion")
    if completion:
        lines.append(f"[Completion] {completion['support']}")
    return lines


def written_files(messages: list) -> list[str]:
    """Repository-relative paths that a completed Coder/Debugger sub-agent
    run (delegation.py's code_repository repair loop) actually wrote -
    determined from each write_file call's own tool RESULT (its
    ToolMessage), not merely from the arguments the LLM asked for.

    write_file can reject a call (invalid_input, path_outside_repository,
    protected_path, a directory path, ...), in which case the repository is
    left unchanged. The AI message's tool_calls only show the requested write;
    the matching ToolMessage confirms whether the repository actually changed.
    A rejected write must not be reported as changed or sent to scoped
    verification.

    Deliberately not Checkpoint.diff(): that depends on a checkpoint having
    been taken, and would go silently empty (so scoped verification would
    silently check nothing) whenever config.enable_checkpoints is False -
    checkpointing and verification are meant to be independent safety
    mechanisms.

    Only a genuine LangGraph/LangChain write_file (AIMessage tool_call ->
    matching ToolMessage) is considered; a coincidental ToolMessage from a
    different tool is never mistaken for a write. A tool call left without a
    matching ToolMessage (e.g. the run stopped before the tool executed) is
    not counted, and neither is a result whose content is not the expected
    JSON object. The same file written more than once - including a later
    Debugger attempt confirming an earlier Coder write - is reported once,
    in the order it was first successfully written.
    """
    write_calls: dict[str, dict] = {}
    for message in messages:
        for call in getattr(message, "tool_calls", None) or []:
            if call.get("name") == "write_file":
                call_id = call.get("id")
                if call_id is not None:
                    write_calls[call_id] = call.get("args") or {}

    seen: set[str] = set()
    paths: list[str] = []
    for message in messages:
        if getattr(message, "type", "") != "tool":
            continue
        call_id = getattr(message, "tool_call_id", None)
        if call_id not in write_calls:
            # Not a write_file result (or no matching call survived) - some
            # other tool's ToolMessage, or an id we don't recognize.
            continue
        try:
            result = json.loads(message.content)
        except (TypeError, ValueError):
            continue  # not the JSON object write_file always returns
        if not isinstance(result, dict) or "error" in result:
            continue  # rejected write: the repository was never touched
        path = result.get("file") or write_calls[call_id].get("path")
        if path and path not in seen:
            seen.add(path)
            paths.append(path)
    return paths
