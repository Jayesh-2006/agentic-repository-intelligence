"""
Small pure helpers for the agent layer. No LangChain / LangGraph imports:
they only read attributes (`type`, `content`, `tool_calls`, ...) from message
objects, so they are easy to test and easy to explain.
"""

from __future__ import annotations

import json

from .observability import format_tool_args, get_run_context, redact_secrets
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


def _dumps(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


# Fields worth keeping legible when an oversized result is reduced: outcome,
# error, identity and provenance. Strings under them get a larger floor.
_PROTECTED_KEYS = frozenset({
    "error", "message", "status", "state", "code", "reason", "file", "path", "symbol", "name", "id",
    "tool", "kind", "type", "match_type", "provenance", "line_start", "line_end", "total_lines",
    "next_start_line",
})
_PROTECTED_MIN_CHARS = 200
_TRUNCATION_NOTE = ("This observation was reduced to fit the size limit and is INCOMPLETE; omitted "
                    "entries/text are marked in place. Ask for a narrower line range, symbol or query.")


def _shrink(value, str_cap: int, list_cap: int, protected: bool = False):
    """Structure-aware reduction into a NEW value (the input is never mutated):
    long strings keep their head plus an explicit omitted-characters marker,
    long lists keep their first entries plus a count of what was omitted."""
    if isinstance(value, str):
        cap = max(str_cap, _PROTECTED_MIN_CHARS) if protected else str_cap
        if len(value) <= cap:
            return value
        return value[:cap] + f"...[{len(value) - cap} more characters omitted]"
    if isinstance(value, list):
        kept = [_shrink(item, str_cap, list_cap, protected) for item in value[:list_cap]]
        if len(value) <= list_cap:
            return kept
        return {"items": kept, "shown": len(kept), "omitted": len(value) - len(kept), "total": len(value),
                "note": f"{len(value) - len(kept)} more entries omitted (showing {len(kept)} of {len(value)})"}
    if isinstance(value, dict):
        return {key: _shrink(item, str_cap, list_cap, protected or key in _PROTECTED_KEYS)
                for key, item in value.items()}
    return value


def _mark_incomplete(value):
    """Explicit, machine-readable incompleteness markers on the top level
    (`truncated` / `observation_truncated` are what downstream code reads)."""
    marked = dict(value) if isinstance(value, dict) else {"result": value}
    marked.update(truncated=True, observation_truncated=True, observation_note=_TRUNCATION_NOTE)
    return marked


def _incomplete_envelope(result, max_chars: int, drop: frozenset = frozenset()) -> str:
    """Last resort: valid JSON naming the top-level keys and keeping only short
    protected scalars; `drop` removes fields whose partial delivery would be
    misleading (e.g. get_file content and its line range)."""
    envelope: dict = {"truncated": True, "observation_truncated": True, "observation_note": _TRUNCATION_NOTE}
    if isinstance(result, dict):
        envelope["keys"] = [str(k) for k in result][:20]
        for key, item in result.items():
            if key in _PROTECTED_KEYS and key not in drop and isinstance(item, (str, int, float, bool)):
                envelope[key] = item[:120] if isinstance(item, str) else item
            if key == "error" and isinstance(item, dict):
                envelope["error"] = _shrink(item, 120, 3, True)
    text = _dumps(envelope)
    if len(text) > max_chars:
        envelope = {"truncated": True, "observation_truncated": True,
                    "observation_note": "result too large; ask for a narrower query"}
        text = _dumps(envelope)  # valid JSON takes precedence over the character bound
    return text


def to_observation(result: dict, max_chars: int, max_list_items: int = DEFAULT_MAX_LIST_ITEMS) -> str:
    """Turn a repository tool result (a dict, possibly {"error": ...}) into the
    text the LLM sees. The result is never mutated and the output is ALWAYS
    valid JSON. Unusually large collections are bounded first
    (`_bound_collections`); if the JSON is still oversized, get_file and
    get_symbol results are cut at whole-line boundaries, and any other result
    is reduced structure-aware (`_shrink`: long strings and lists shrink
    progressively, protected status/error/identity/provenance fields keep more
    text) and marked `truncated`/`observation_truncated` so it cannot be read as
    complete. If nothing fits, a small incomplete-envelope JSON is returned."""
    bounded = _bound_collections(result, max_list_items) if isinstance(result, dict) else result
    text = _dumps(bounded)
    if len(text) <= max_chars:
        return text
    if _is_get_file_result(result):
        text = _dumps(_bound_get_file_result(result, max_chars))
        if len(text) <= max_chars:
            return text
        return _incomplete_envelope(
            result, max_chars, drop=frozenset({"content", "line_start", "line_end", "next_start_line"}))
    if _is_get_symbol_result(result):
        text = _dumps(_bound_symbol_result(result, max_chars))
        if len(text) <= max_chars:
            return text
        return _incomplete_envelope(
            result, max_chars, drop=frozenset({"source", "line_start", "line_end"}))
    str_cap, list_cap = max(64, max_chars // 4), max(1, max_list_items)
    for _ in range(8):
        text = _dumps(_mark_incomplete(_shrink(bounded, str_cap, list_cap)))
        if len(text) <= max_chars:
            return text
        str_cap, list_cap = max(16, str_cap // 2), max(1, list_cap // 2)
    return _incomplete_envelope(result, max_chars)


def _is_get_file_result(result) -> bool:
    return (isinstance(result, dict) and isinstance(result.get("file"), str)
            and isinstance(result.get("content"), str) and "line_start" in result)


def _is_get_symbol_result(result) -> bool:
    return (isinstance(result, dict) and isinstance(result.get("symbol"), str)
            and isinstance(result.get("source"), str) and "line_start" in result)


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


def _bound_symbol_result(result: dict, max_chars: int) -> dict:
    """Keep get_symbol observations valid JSON and align line_end with delivered lines."""
    data = dict(result)
    lines = str(result.get("source", "")).splitlines()
    start = result.get("line_start") or 1
    data["truncated"] = True
    data["observation_truncated"] = True

    def candidate(count: int) -> dict:
        bounded = dict(data)
        bounded["source"] = "\n".join(lines[:count])
        bounded["line_end"] = start + count - 1 if count else start - 1
        return bounded

    low, high = 0, len(lines)
    best = candidate(0)
    while low <= high:
        mid = (low + high) // 2
        cur = candidate(mid)
        if len(json.dumps(cur, ensure_ascii=False, separators=(",", ":"))) <= max_chars:
            best = cur
            low = mid + 1
        else:
            high = mid - 1
    return best


def bounded_model_history(messages: list, task: str | None = None, max_turns: int = 6) -> list:
    """Return a bounded recent conversation view for an LLM call.

    The full state history remains intact for evidence accounting and tool-result
    correlation. Only the model-facing view is reduced: all initial leading
    human messages before the first AI turn are retained, followed by
    the most recent AI-turn groups (AI tool request + following tool/check
    messages).
    """
    if not messages:
        return []
    max_turns = max(1, int(max_turns))
    leading_humans = []
    for message in messages:
        kind = getattr(message, "type", "")
        if kind == "ai":
            break
        if kind == "human":
            leading_humans.append(message)
    if not leading_humans:
        first_human = next((m for m in messages if getattr(m, "type", "") == "human"), None)
        if first_human is not None:
            leading_humans = [first_human]

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
    if len(selected) > 1:
        selected = _compact_older_search_groups(selected)
    flattened = [message for group in selected for message in group]
    prefix = [m for m in leading_humans if all(m is not f for f in flattened)]
    return [*prefix, *flattened]


def _compact_older_search_groups(groups: list[list]) -> list[list]:
    """Compact redundant search_repository observations in older turn groups
    while preserving all evidence provenance and unread code snippets."""
    read_intervals: dict[str, list[tuple[int, int]]] = {}
    for group in groups:
        for msg in group:
            if getattr(msg, "type", "") != "tool":
                continue
            try:
                data = json.loads(getattr(msg, "content", "") or "")
            except (TypeError, ValueError):
                continue
            if not isinstance(data, dict):
                continue
            f_path = data.get("file")
            l_start, l_end = data.get("line_start"), data.get("line_end")
            if isinstance(f_path, str) and isinstance(l_start, int) and isinstance(l_end, int) and ("content" in data or "source" in data):
                key = f_path.replace("\\", "/").casefold()
                read_intervals.setdefault(key, []).append((l_start, l_end))

    out_groups: list[list] = []
    for g_idx, group in enumerate(groups):
        if g_idx == len(groups) - 1:
            out_groups.append(group)
            continue
        new_group = []
        for msg in group:
            if getattr(msg, "type", "") != "tool":
                new_group.append(msg)
                continue
            raw_content = getattr(msg, "content", "")
            try:
                data = json.loads(raw_content or "")
            except (TypeError, ValueError):
                new_group.append(msg)
                continue
            results = data.get("results") if isinstance(data, dict) else None
            if not isinstance(data, dict) or "query" not in data or not isinstance(results, list):
                new_group.append(msg)
                continue
            changed = False
            compacted_results = []
            for hit in results[:5]:
                if not isinstance(hit, dict):
                    compacted_results.append(hit)
                    continue
                hit_copy = dict(hit)
                h_file = str(hit_copy.get("file") or "").replace("\\", "/").casefold()
                s_start = hit_copy.get("snippet_line_start") or hit_copy.get("line_start")
                s_end = hit_copy.get("snippet_line_end") or hit_copy.get("line_end")
                covered = False
                if h_file in read_intervals and isinstance(s_start, int) and isinstance(s_end, int):
                    covered = any(r_start <= s_start and s_end <= r_end for r_start, r_end in read_intervals[h_file])
                snippet = str(hit_copy.get("snippet") or "")
                if covered and len(snippet) > 48:
                    hit_copy["snippet"] = "[covered by recorded get_file/get_symbol read]"
                    changed = True
                elif len(snippet) > 220:
                    hit_copy["snippet"] = snippet[:220] + "...[clipped]"
                    changed = True
                compacted_results.append(hit_copy)
            if len(results) > 5:
                changed = True
            if changed:
                new_data = dict(data, results=compacted_results, count=len(compacted_results))
                new_content = _dumps(new_data)
                try:
                    msg = msg.model_copy(update={"content": new_content})
                except AttributeError:
                    import copy as _copy
                    msg = _copy.copy(msg)
                    msg.content = new_content
            new_group.append(msg)
        out_groups.append(new_group)
    return out_groups


def route_after_agent(state: dict) -> str:
    """Conditional edge: if the last AI message asked for tools, run them;
    otherwise the model has given its final answer."""
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else "end"


def wrap_up_prompt(messages: list, max_steps: int, obs_chars: int = 1200, max_items: int = 12,
                   task: str | None = None) -> str:
    """Prompt used when the step limit is hit: the question plus a plain-text
    digest of the tool evidence, so the model can still answer."""
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
    """Text of the last AI message that is not a tool request."""
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


def describe_update(node: str, update: dict, agent: str = "Main") -> list[str]:
    """Console-trace lines for one LangGraph `updates` chunk ({node: state_update})."""
    ctx = get_run_context()
    if ctx is not None and ctx.on_line is not None:
        return []
    lines: list[str] = []
    api_key = ctx.api_key if ctx else None
    messages = update.get("messages", [])
    for m in messages:
        kind = getattr(m, "type", "")
        if kind == "tool":
            lines.append(
                f"[{agent}] {getattr(m, 'name', 'tool')} -> "
                f"{redact_secrets(summarize_observation(m.content), api_key)}"
            )
    if update.get("model_error"):
        lines.append(f"[{agent}] model_error -> FAILED: {redact_secrets(str(update['model_error']), api_key)[:240]}")
    completion = update.get("completion")
    if completion:
        lines.append(f"[{agent}] run completed -> {completion['support']}")
    return lines


def written_files(messages: list) -> list[str]:
    """Repository-relative paths that write_file tool calls actually wrote."""
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
            continue
        try:
            result = json.loads(message.content)
        except (TypeError, ValueError):
            continue
        if not isinstance(result, dict) or "error" in result:
            continue
        path = result.get("file") or write_calls[call_id].get("path")
        if path and path not in seen:
            seen.add(path)
            paths.append(path)
    return paths
