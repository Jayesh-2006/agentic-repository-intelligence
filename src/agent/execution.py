"""Execution ledger and retry/reassessment state tracking."""

from __future__ import annotations

from typing import Optional

READ_ONLY_TOOLS = frozenset({
    "search_repository",
    "find_files",
    "find_symbol",
    "get_file",
    "get_symbol",
    "find_callers",
    "find_callees",
    "find_importers",
    "get_dependencies",
    "get_ast",
})
STRUCTURAL_TOOLS = frozenset({
    "find_symbol",
    "get_symbol",
    "find_callers",
    "find_callees",
    "find_importers",
    "get_dependencies",
    "get_ast",
})
DIRECT_TOOLS = READ_ONLY_TOOLS
MUTATION_TOOLS = frozenset({"write_file"})
EXECUTION_TOOLS = frozenset({"verify_repository", "run_command"})

# Times a repeated read-only action may trigger a strategy reassessment instead
# of ending the run, per invocation.
MAX_STRATEGY_REASSESSMENTS = 1
# Remaining decide turns needed to act on a reassessment.
MIN_TURNS_FOR_REASSESSMENT = 2

STARTED = "started"


def empty_execution() -> dict:
    return {"attempts": []}


def _attempts(execution: Optional[dict]) -> list[dict]:
    return list((execution or {}).get("attempts") or [])


def mode_for(tool: str) -> str:
    if tool in MUTATION_TOOLS:
        return "mutation"
    if tool in EXECUTION_TOOLS:
        return "execution"
    return "direct"


def start_attempt(
    execution: Optional[dict],
    *,
    step_id: Optional[str],
    tool: str,
    action_key: str,
    binding: str,
    generation: int,
    tool_call_id: Optional[str] = None,
) -> dict:
    """Record that `tool` is being started for `step_id`."""
    attempts = _attempts(execution)
    previous = next(
        (a["attempt_id"] for a in reversed(attempts) if a.get("action_key") == action_key),
        None,
    )
    attempts.append({
        "attempt_id": f"attempt_{len(attempts) + 1}",
        "step_id": step_id,
        "tool_call_id": tool_call_id,
        "binding": binding,
        "tool": tool,
        "mode": mode_for(tool),
        "action_key": action_key,
        "retry_of": previous,
        "generation": generation,
        "status": STARTED,
        "productive": None,
        "succeeded": None,
        "interrupted": False,
    })
    return {"attempts": attempts}


def finish_attempt(
    execution: Optional[dict],
    *,
    step_id: Optional[str],
    tool: str,
    result: str,
    generation: int,
    succeeded: bool,
    interrupted: bool = False,
    baseline_generation: Optional[int] = None,
    tool_call_id: Optional[str] = None,
) -> dict:
    """Close the in-flight attempt for `tool_call_id`/`step_id`/`tool` with its outcome."""
    attempts = _attempts(execution)
    index = None
    if tool_call_id is not None:
        index = next(
            (i for i in range(len(attempts) - 1, -1, -1)
             if attempts[i].get("status") == STARTED and attempts[i].get("tool_call_id") == tool_call_id),
            None,
        )
    if index is None:
        index = next(
            (i for i in range(len(attempts) - 1, -1, -1)
             if attempts[i].get("status") == STARTED and attempts[i].get("tool") == tool
             and (step_id is None or attempts[i].get("step_id") in (step_id, None))),
            None,
        )
    if index is None:
        attempts.append({
            "attempt_id": f"attempt_{len(attempts) + 1}",
            "step_id": step_id,
            "tool_call_id": tool_call_id,
            "binding": "automatic",
            "tool": tool,
            "mode": mode_for(tool),
            "action_key": None,
            "retry_of": None,
            "generation": generation if baseline_generation is None else baseline_generation,
            "status": STARTED,
            "productive": None,
            "succeeded": None,
            "interrupted": False,
        })
        index = len(attempts) - 1
    record = dict(attempts[index])
    start_gen = int(record.get("generation") if baseline_generation is None else baseline_generation or 0)
    productive = generation > start_gen
    record.update(
        status=result,
        productive=productive,
        succeeded=bool(succeeded),
        interrupted=bool(interrupted),
    )
    attempts[index] = record
    return {"attempts": attempts}


def finished_attempts(execution: Optional[dict]) -> list[dict]:
    return [a for a in _attempts(execution) if a.get("status") != STARTED]


def has_used_structural_tools(execution: Optional[dict]) -> bool:
    """True when at least one structural tool (symbol, caller/callee, importer,
    dependency, or AST) has been executed."""
    return any(a.get("tool") in STRUCTURAL_TOOLS for a in finished_attempts(execution))


def unproductive_streak(execution: Optional[dict]) -> int:
    """Consecutive finished attempts since the last productive attempt."""
    done = finished_attempts(execution)
    count = 0
    for a in reversed(done):
        if a.get("productive"):
            break
        count += 1
    return count


def reassessment_strategy(
    execution: Optional[dict],
    metrics: Optional[dict],
    *,
    sources_restricted: bool = False,
    repeated_tool: str,
    steps_left: int,
) -> Optional[str]:
    """When a read-only action is repeated, offer one bounded strategy change
    before stopping on a subsequent repeat:
      - "structural": switch from keyword/file search to structural tools
        (`find_symbol`, `get_symbol`, `find_callers`, `find_callees`,
        `find_importers`, `get_dependencies`, `get_ast`) when no hard source
        restriction applies and structural tools have not yet been used.
      - "direct": choose a different unread file/range or symbol target.
      - None: reassessment budget or step budget exhausted -> stop.
    """
    if (repeated_tool not in READ_ONLY_TOOLS
            or int((metrics or {}).get("strategy_reassessments", 0)) >= MAX_STRATEGY_REASSESSMENTS
            or steps_left < MIN_TURNS_FOR_REASSESSMENT):
        return None
    if not sources_restricted and not has_used_structural_tools(execution):
        return "structural"
    return "direct"


def can_reassess_before_stop(
    execution: Optional[dict],
    metrics: Optional[dict],
    *,
    sources_restricted: bool = False,
    repeated_tool: str,
    steps_left: int,
) -> bool:
    return reassessment_strategy(
        execution,
        metrics,
        sources_restricted=sources_restricted,
        repeated_tool=repeated_tool,
        steps_left=steps_left,
    ) is not None


def strategy_hint(
    execution: Optional[dict],
    *,
    sources_restricted: bool = False,
    unmet_criteria: int,
    steps_left: Optional[int] = None,
) -> str:
    """One concise policy sentence for a retry/feedback message; guidance only."""
    if not unmet_criteria:
        return ""
    if sources_restricted:
        return "Policy hint: explicit source restriction is active; use get_file on the permitted path(s) only."
    if steps_left is not None and steps_left < MIN_TURNS_FOR_REASSESSMENT:
        return (
            "Policy hint: little step budget remains; issue targeted read-only call(s) for the specific "
            "unmet criteria or state what could not be established."
        )
    if not has_used_structural_tools(execution):
        return (
            "Policy hint: use structural tools (find_symbol, get_symbol, find_callers, find_callees, "
            "find_importers, get_dependencies, get_ast) to trace definitions and call flow directly "
            "rather than repeating keyword searches."
        )
    if unproductive_streak(execution) >= 2:
        return (
            "Policy hint: recent tool calls added no new information; read a discovered symbol/file "
            "range that has not yet been inspected, or state what could not be established."
        )
    return ""


def read_coverage_summary(plan: Optional[dict], limit: int = 8) -> list[str]:
    """Summarize already-covered files/line ranges and symbols from recorded plan steps
    so the agent avoids rereading covered ranges or repeating empty searches."""
    if not plan:
        return []
    steps = [s for s in plan.get("steps", []) if s.get("result")]
    if not steps:
        return []

    file_intervals: dict[str, list[tuple[int, int]]] = {}
    file_totals: dict[str, int] = {}
    symbols_read: list[str] = []
    empty_queries: list[str] = []

    for s in steps:
        tool = s.get("tool")
        res = s.get("result")
        prov = s.get("provenance") or {}
        if tool == "get_file" and res in ("ok", "inconclusive"):
            path = prov.get("file")
            start, end, total = prov.get("line_start"), prov.get("line_end"), prov.get("total_lines")
            if isinstance(path, str) and path and isinstance(start, int) and isinstance(end, int):
                file_intervals.setdefault(path, []).append((start, end))
                if isinstance(total, int):
                    file_totals[path] = total
        elif tool == "get_symbol" and res == "ok":
            sym = prov.get("symbol")
            path = prov.get("file")
            start, end = prov.get("line_start"), prov.get("line_end")
            label = f"{sym} ({path}:{start}-{end})" if path and start is not None else str(sym or s.get("action"))
            if label and label not in symbols_read:
                symbols_read.append(label)
        elif res == "empty":
            act = str(s.get("action") or "")
            if act and act not in empty_queries:
                empty_queries.append(act)

    notes: list[str] = []
    for path in sorted(file_intervals)[:limit]:
        merged: list[tuple[int, int]] = []
        for a, b in sorted(file_intervals[path]):
            if merged and a <= merged[-1][1] + 1:
                merged[-1] = (merged[-1][0], max(merged[-1][1], b))
            else:
                merged.append((a, b))
        total = file_totals.get(path)
        range_str = ", ".join(f"{a}-{b}" for a, b in merged)
        if total is not None:
            full = len(merged) == 1 and merged[0][0] == 1 and merged[0][1] >= total
            notes.append(f"{path}:{range_str}/{total}" + (" (complete)" if full else ""))
        else:
            notes.append(f"{path}:{range_str}")
    if symbols_read:
        notes.append("symbols read: " + ", ".join(symbols_read[:6]))
    if empty_queries:
        notes.append("0-match lookups (do not repeat): " + "; ".join(empty_queries[-4:]))
    return notes


def digest(execution: Optional[dict], plan: Optional[dict], criteria: Optional[list], limit: int = 6) -> str:
    """A concise, factual summary of attempted actions, read coverage, and remaining work."""
    done = finished_attempts(execution)
    if not done:
        return ""
    parts = []
    for a in done[-limit:]:
        flags = [
            a.get("mode", "direct"),
            "retry of " + a["retry_of"] if a.get("retry_of") else None,
            "new information" if a.get("productive") else "nothing new",
        ]
        parts.append(f"{a['attempt_id']} {a['tool']} [{', '.join(f for f in flags if f)}] -> {a['status']}")
    pending = [s for s in (plan or {}).get("steps", []) if s.get("status") == "pending"]
    unmet = [c for c in (criteria or []) if not c.get("satisfied")]
    coverage = read_coverage_summary(plan)
    cov_text = f" Covered so far: {'; '.join(coverage)}." if coverage else ""
    return (
        "Execution state (system data): actions so far: "
        + "; ".join(parts)
        + f". Total tool attempts: {len(done)}.{cov_text} "
        + f"Remaining: {len(pending)} planned step(s) pending, {len(unmet)} unmet criteria."
    )
