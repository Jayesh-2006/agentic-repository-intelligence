"""
Evidence-grounded completion for the Deep Agent.

    objective -> acceptance criteria -> steps with evidence
              -> which criteria the evidence supports -> completion state

The pieces (all plain dicts, same style as plan.py; no LangChain imports):

- Criterion: {"id", "description", "evidence_needed", "satisfied",
  "evidence_ids", "note"}. Seeded once per invocation by `plan_task`
  (deep_agent.py) from the objective; kept in `DeepAgentState.criteria`.
- Step evidence: extra keys on a plan step, written by `record_step_evidence`
  from the ACTUAL delegation result (never from the model's own words):
  `tool`, `action`, `action_key`, `result` (ok/failed/inconclusive/empty),
  `evidence` (a clipped excerpt), `order` (1, 2, 3 ... in execution order) and,
  once graded, `criteria` (ids of the criteria this step supports). Plan
  functions in plan.py copy step dicts, so these keys survive plan updates.
- Grading: an LLM proposes, per criterion, satisfied + cited step ids
  (deep_agent.py's `_assess`). `apply_assessment` then enforces the rules
  deterministically: a criterion only counts as satisfied if it cites at
  least one REAL step whose result is `ok`. The model cannot satisfy a
  criterion by assertion alone.
- Completion state: `classify_support` maps criteria + failures to one of
  SUPPORTED / PARTIAL / UNSUPPORTED / BLOCKED.
- Final response: `build_final_response` builds the answer from the
  recorded state - Conclusion / Supporting evidence / Unresolved - citing
  only real, gate-accepted steps and never contradicting the completion state.
"""

from __future__ import annotations

import json
import re
from typing import Optional

from ..retrieval import tokenize
from .plan import _CODE_FENCE_RE, _normalize, pending_steps

# Completion states (what the gathered evidence supports).
SUPPORTED = "SUPPORTED"      # every acceptance criterion has usable evidence
PARTIAL = "PARTIAL"          # some criteria supported, or work still outstanding
UNSUPPORTED = "UNSUPPORTED"  # no criterion is supported by usable evidence
BLOCKED = "BLOCKED"          # the latest change/verification failed, or nothing usable could be gathered

# Result of a single step's delegation.
RESULT_OK = "ok"
RESULT_FAILED = "failed"
RESULT_INCONCLUSIVE = "inconclusive"  # ran, but proves nothing (NO_TESTS / UNAVAILABLE)
RESULT_EMPTY = "empty"

# Bounds. Criteria are capped so grading stays a small, fixed-size call; the
# completion-retry cap is on top of (never instead of) config.max_agent_steps.
MAX_CRITERIA = 4
MAX_COMPLETION_RETRIES = 2
# Per-step evidence excerpt kept for the grader. 600 was too small: a README
# report is easily longer, so the grader saw only its opening and answered
# "no excerpt provides X" even when the full report did.
MAX_EVIDENCE_CHARS = 1500
# Criterion note prefix used when the grading call itself failed (rate limit,
# network, unparseable reply) - so "grader unavailable" is never confused with
# "the evidence does not support it".
GRADER_FAILED_NOTE = "grading call failed"
_ACTION_LABEL_CHARS = 160
_NOTE_CHARS = 200


def _clip(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


# ---------------------------------------------------------------- criteria


def new_criterion(index: int, description: str, evidence_needed: str) -> dict:
    return {"id": f"c{index}", "description": description, "evidence_needed": evidence_needed,
            "satisfied": False, "evidence_ids": [], "note": ""}


def default_criteria(task: str) -> list[dict]:
    """Generic fallback when the planning call gave no usable criteria: one
    criterion parameterised by the task text, not tailored to any task kind.

    Also the ONLY criterion used for lookup/explain-style tasks (see
    intent.is_lookup_task / plan_task in deep_agent.py, which skips the
    planning call - and therefore this fallback - entirely for those). The
    evidence_needed wording must therefore read naturally for a descriptive
    "what does X do" question, not only an audit/change task: "reports,
    changes, verification" previously implied the kind of artifact a
    change/audit task produces, which the grading call (CRITERIA_CHECK_PROMPT)
    could read as requiring more than a plain description needs, and send a
    perfectly good README/source read back as UNSUPPORTED."""
    return [new_criterion(1, f"The objective is fully addressed: {task}",
                          "Concrete repository evidence (file/document contents, code, structure, or "
                          "verification results, as applicable) that directly shows it")]


def parse_criteria(text: str, max_criteria: int = MAX_CRITERIA) -> list[dict]:
    """Extract acceptance criteria from the planning reply. Expects a JSON
    object with a "criteria" list of {"criterion", "evidence_needed"} objects
    (plain strings are accepted too). Returns [] - never raises - when there
    is nothing usable; the caller then uses `default_criteria`."""
    cleaned = _CODE_FENCE_RE.sub("", (text or "").strip()).strip()
    try:
        data = json.loads(cleaned)
    except (TypeError, ValueError):
        return []
    items = data.get("criteria") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return []
    criteria: list[dict] = []
    for item in items:
        if isinstance(item, str):
            description, needed = item, ""
        elif isinstance(item, dict):
            description = item.get("criterion") or item.get("description") or ""
            needed = item.get("evidence_needed") or item.get("evidence") or ""
        else:
            continue
        if isinstance(description, str) and description.strip():
            criteria.append(new_criterion(len(criteria) + 1, _clip(description, 240),
                                          _clip(needed, 240) if isinstance(needed, str) else ""))
        if len(criteria) >= max(1, max_criteria):
            break
    return criteria


# ------------------------------------------------------------ step evidence


def _action_key(tool: str, arg: str) -> str:
    return f"{tool}|{_normalize(arg)}"[:300]


def classify_result(tool: str, report: str, payload: dict, code_tools: tuple) -> str:
    """How much did this delegation actually establish? Deterministic, from
    the tool's own result: a code/verify run is `ok` only on a real SUCCESS
    state (NO_TESTS / UNAVAILABLE ran but prove nothing); an investigation is
    `ok` only when it contains raw repository tool observations (the Explorer's
    model-generated summary alone is never source evidence)."""
    text = (report or "").strip()
    if tool in code_tools:
        state = payload.get("state")
        if state == "SUCCESS":
            return RESULT_OK
        if state in ("NO_TESTS", "UNAVAILABLE"):
            return RESULT_INCONCLUSIVE
        return RESULT_FAILED
    if not text:
        return RESULT_EMPTY
    if text.startswith("Explorer model call failed"):
        return RESULT_FAILED
    if tool == "investigate_repository":
        observations = payload.get("raw_observations")
        if not isinstance(observations, list):
            return RESULT_EMPTY
        for item in observations:
            if not isinstance(item, dict):
                continue
            observation = item.get("observation")
            if not isinstance(observation, str) or not observation.strip():
                continue
            try:
                parsed = json.loads(observation)
            except (TypeError, ValueError):
                parsed = None
            if not (isinstance(parsed, dict) and "error" in parsed):
                return RESULT_OK
        return RESULT_EMPTY
    return RESULT_OK


# get_file's own result shape (see service.get_file / tools.py) is
# a plain dict - {"file", "line_start", "line_end", "total_lines",
# "truncated", "content"} on success, {"error": {"code", "message"}} on a
# rejected/failed read - never the {"state": ..., "report": ...} shape the
# code/verify delegation tools use, so it needs its own classifier/
# summarizer rather than sharing classify_result/summarize_evidence.
_GET_FILE_ERROR_PREFIX = '{"error":'
_GET_FILE_SUCCESS_PREFIX = '{"file":'


def classify_get_file_result(report: str) -> str:
    """Dedicated, conservative classifier for a direct get_file
    result. The generic classify_result is not right for it: an
    {"error": ...} dict would fall into classify_result's "any non-empty
    text that is not the Explorer's own failure message" branch and come
    back RESULT_OK, which would wrongly count a rejected read (unknown path,
    protected path, invalid range, ...) as usable evidence.

    utils.to_observation keeps oversized get_file results valid JSON and
    updates the delivered line range. If an unusually small character bound
    still cuts the JSON, parsing fails and the result is inconclusive (never
    OK), because the delivered content and range cannot be established
    safely."""
    text = (report or "").strip()
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        data = None
    if isinstance(data, dict):
        if "error" in data:
            return RESULT_FAILED
        content = data.get("content")
        if not (isinstance(content, str) and content):
            return RESULT_INCONCLUSIVE if data.get("truncated") else RESULT_EMPTY
        # A bounded read (service max_lines) is only part of the file. It is
        # real but incomplete evidence, so it is INCONCLUSIVE on its own; a
        # file whose reads jointly cover every line is promoted back to
        # usable evidence by `_fully_read_files` (see apply_assessment).
        return RESULT_INCONCLUSIVE if data.get("truncated") else RESULT_OK
    if text.startswith(_GET_FILE_ERROR_PREFIX):
        return RESULT_FAILED
    # Cut mid-JSON by utils.to_observation: the content is partial by
    # definition, so it is never RESULT_OK.
    return RESULT_INCONCLUSIVE


def classify_find_files_result(report: str) -> str:
    """Classify both ordinary and list-bounded find_files observations.

    A bounded positive result remains usable for questions about a returned
    path; omitted entries are retained in provenance so completeness checks
    can prevent it from supporting an exhaustive listing.
    """
    try:
        data = json.loads((report or "").strip())
    except (TypeError, ValueError):
        return RESULT_INCONCLUSIVE
    if not isinstance(data, dict):
        return RESULT_INCONCLUSIVE
    if "error" in data:
        return RESULT_FAILED
    files = data.get("files")
    if isinstance(files, list):
        return RESULT_OK if files else RESULT_EMPTY
    if isinstance(files, dict):
        items = files.get("items")
        if isinstance(items, list):
            return RESULT_OK if items else RESULT_EMPTY
    return RESULT_INCONCLUSIVE


def classify_find_symbol_result(report: str) -> str:
    """Only a structured symbol match is usable lookup evidence."""
    try:
        data = json.loads((report or "").strip())
    except (TypeError, ValueError):
        return RESULT_INCONCLUSIVE
    if not isinstance(data, dict):
        return RESULT_INCONCLUSIVE
    if "error" in data:
        return RESULT_FAILED
    matches = data.get("matches")
    return RESULT_OK if isinstance(matches, list) and bool(matches) else RESULT_EMPTY


def _find_files_provenance(payload: dict) -> dict:
    files = payload.get("files")
    bounded = files if isinstance(files, dict) else {}
    return {
        "pattern": payload.get("pattern"),
        "total_matches": payload.get("total_matches"),
        "shown": bounded.get("shown", len(files) if isinstance(files, list) else None),
        "omitted": bounded.get("omitted", 0),
        "truncated": bool(bounded.get("omitted", 0) or payload.get("truncated")),
    }


def get_file_provenance(report: str) -> dict:
    """What a get_file result actually covered: file, line range, total lines
    and whether the read was cut short (by the service's max_lines bound OR by
    utils.to_observation cutting the JSON string). Never raises."""
    text = (report or "").strip()
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        data = None
    if isinstance(data, dict) and "error" not in data:
        return {"file": data.get("file"), "line_start": data.get("line_start"),
                "line_end": data.get("line_end"), "total_lines": data.get("total_lines"),
                "truncated": bool(data.get("truncated"))}
    if text.startswith(_GET_FILE_SUCCESS_PREFIX):
        # Observation was cut mid-JSON: the leading keys survive, the content is partial.
        def num(key):
            m = re.search(rf'"{key}":(\d+)', text)
            return int(m.group(1)) if m else None
        f = re.search(r'"file":"([^"]*)"', text)
        start, total = num("line_start"), num("total_lines")
        shown = text.count("\\n")  # JSON-escaped newlines that survived the cut
        end = (start + shown) if start is not None else None
        return {"file": f.group(1) if f else None, "line_start": start, "line_end": end,
                "total_lines": total, "truncated": True}
    return {}


_COMPLETENESS_RE = re.compile(
    r"\b(how\s+many|all|every|list|different|count|number\s+of|entire|whole|complete|total)\b", re.IGNORECASE)


def _file_coverage(plan: Optional[dict]) -> tuple[dict, dict, set]:
    """Per-file line ranges read by get_file steps (OK or INCONCLUSIVE - a
    truncated read still covers the lines it returned), the file's total line
    count, and the files that had at least one truncated read."""
    covered: dict[str, list[tuple[int, int]]] = {}
    totals: dict[str, int] = {}
    partial: set[str] = set()
    for s in evidence_steps(plan):
        prov = s.get("provenance") or {}
        if s.get("tool") != "get_file" or s.get("result") not in (RESULT_OK, RESULT_INCONCLUSIVE) \
                or not prov.get("file"):
            continue
        start, end, total = prov.get("line_start"), prov.get("line_end"), prov.get("total_lines")
        if start is None or end is None or total is None:
            continue
        covered.setdefault(prov["file"], []).append((int(start), int(end)))
        totals[prov["file"]] = int(total)
        if prov.get("truncated"):
            partial.add(prov["file"])
    return covered, totals, partial


def _reach(ranges: list[tuple[int, int]]) -> int:
    """Highest line reached by contiguous coverage starting at line 1."""
    reach = 0
    for a, b in sorted(ranges):
        if a <= reach + 1:
            reach = max(reach, b)
    return reach


def _read_gaps(plan: Optional[dict]) -> list[str]:
    """Files that were read only in part (per recorded get_file provenance) and
    whose unread remainder was never read by a later/other get_file step."""
    covered, totals, partial = _file_coverage(plan)
    gaps = []
    for path in sorted(partial):
        reach = _reach(covered[path])
        if reach < totals[path]:
            gaps.append(f"{path}: read only up to line {reach} of {totals[path]} - read the rest with "
                        f"get_file start_line={reach + 1} before answering anything that needs the whole file")
    return gaps


def _fully_read_files(plan: Optional[dict]) -> set[str]:
    """Files whose truncated reads together cover every line."""
    covered, totals, partial = _file_coverage(plan)
    return {path for path in partial if _reach(covered[path]) >= totals[path]}


def incomplete_reasons(criteria: list[dict], plan: Optional[dict]) -> list[str]:
    """Deterministic reasons a fully-graded result must not be called SUPPORTED:
    (1) the task needs completeness (count/all/list/...) but a file was only
    partly read; (2) an investigation/read FAILED after the last usable evidence."""
    reasons = []
    text = " ".join(f"{c.get('description', '')} {c.get('evidence_needed', '')}" for c in criteria)
    if _COMPLETENESS_RE.search(text):
        reasons.extend(_read_gaps(plan))
    if re.search(r"\b(all|every|list|entire|whole|complete)\b", text, re.IGNORECASE):
        for step in evidence_steps(plan):
            provenance = step.get("provenance") or {}
            if step.get("tool") == "find_files" and int(provenance.get("omitted") or 0) > 0:
                reasons.append(
                    f"find_files omitted {provenance['omitted']} of {provenance.get('total_matches')} matches; "
                    "the returned paths are not a complete listing"
                )
    steps = evidence_steps(plan)
    ok_orders = [s["order"] for s in steps if s["result"] == RESULT_OK]
    failed_after = [s for s in steps if s["result"] == RESULT_FAILED
                    and (not ok_orders or s["order"] > max(ok_orders))]
    if ok_orders and failed_after:
        reasons.append(f"the latest attempt failed ({failed_after[-1]['step_id']}) after the last usable evidence")
    return reasons


def summarize_get_file_evidence(report: str) -> str:
    """get_file's own evidence excerpt - the actual file content
    (what the read established), with its path/line-range metadata, rather
    than the raw JSON wrapper summarize_evidence would otherwise show. Notes
    when the returned portion was truncated (either the service's own
    max_lines bound, or utils.to_observation cutting a large JSON string)
    so nothing implies a whole file was observed when only a bounded/cut
    portion came through. Still clipped to MAX_EVIDENCE_CHARS like every
    other step's evidence excerpt (the evidence-size limit itself is
    unchanged)."""
    text = (report or "").strip()
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        data = None
    if not isinstance(data, dict) or "error" in data:
        return _clip(text, MAX_EVIDENCE_CHARS)
    path = data.get("file", "?")
    start, end, total = data.get("line_start"), data.get("line_end"), data.get("total_lines")
    if start is not None and end is not None:
        range_note = f"lines {start}-{end} of {total}" if total is not None else f"lines {start}-{end}"
    else:
        range_note = f"{total} line(s)" if total is not None else "unknown range"
    bounded_note = " (truncated - not the full file)" if data.get("truncated") else ""
    header = f"{path} [{range_note}{bounded_note}]:\n"
    return _clip(header + str(data.get("content", "")), MAX_EVIDENCE_CHARS)


def summarize_evidence(tool: str, report: str, payload: dict, code_tools: tuple) -> str:
    """A concise, explicitly labelled evidence excerpt.

    Model-generated Explorer summaries are excluded from the evidence excerpt.
    Bounded raw tool observations and their provenance are what the grader
    receives for an investigation.
    """
    text = " ".join(str(report).split())
    if len(text) > MAX_EVIDENCE_CHARS:
        text = text[: MAX_EVIDENCE_CHARS - 190] + " [...] " + text[-180:]
    if tool in code_tools:
        status = (payload.get("verification") or {}).get("status")
        return (
            f"MODEL SUMMARY: {text}\n"
            f"RAW VERIFICATION: state={payload.get('state')}; status={status}; "
            f"exit_code={(payload.get('verification') or {}).get('exit_code')}"
        )
    if tool == "investigate_repository":
        raw_lines = []
        for item in payload.get("raw_observations", [])[:3]:
            if not isinstance(item, dict):
                continue
            observation = _raw_observation_excerpt(item.get("observation", ""))
            raw_lines.append(
                f"{item.get('tool', 'tool')}: {_clip(observation, 350)}"
            )
        return "RAW REPOSITORY TOOL OBSERVATIONS (Explorer summary excluded): " + " | ".join(raw_lines)
    if tool == "find_files":
        data = _parse_report_object(report)
        files = data.get("files") if isinstance(data, dict) else None
        if isinstance(files, dict):
            files = files.get("items")
        paths = [str(item) for item in files or [] if isinstance(item, str)] if isinstance(files, list) else []
        provenance = _find_files_provenance(payload)
        return ("RAW FIND_FILES RESULTS: " + ", ".join(paths[:30]) +
                f"; total_matches={provenance.get('total_matches')}; omitted={provenance.get('omitted', 0)}")
    if tool == "find_symbol":
        data = _parse_report_object(report)
        matches = data.get("matches") if isinstance(data, dict) else None
        return "RAW FIND_SYMBOL RESULTS: " + json.dumps(matches or [], ensure_ascii=False, separators=(",", ":"))
    return f"RAW TOOL OBSERVATION: {text}"


_REQUEST_ECHO_KEYS = {"query", "pattern", "args", "objective", "question", "request", "task"}
_REQUEST_ECHO_FIELD_RE = re.compile(
    r'"(?:query|pattern|args|objective|question|request|task)"\s*:\s*"(?:\\.|[^"\\])*"\s*,?',
    re.IGNORECASE,
)


def _without_request_echo(value):
    if isinstance(value, dict):
        return {key: _without_request_echo(item) for key, item in value.items()
                if str(key).lower() not in _REQUEST_ECHO_KEYS}
    if isinstance(value, list):
        return [_without_request_echo(item) for item in value]
    return value


def _raw_observation_excerpt(observation) -> str:
    """Remove request-shaped fields before raw Explorer observations reach grading."""
    raw = str(observation or "")
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return _REQUEST_ECHO_FIELD_RE.sub("", raw)
    return json.dumps(_without_request_echo(parsed), ensure_ascii=False, separators=(",", ":"))


def _parse_report_object(report: str) -> dict:
    try:
        parsed = json.loads(report or "")
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def evidence_provenance(tool: str, payload: dict, code_tools: tuple) -> dict:
    """Structured provenance kept beside the evidence excerpt."""
    if tool in code_tools:
        verification = payload.get("verification") or {}
        return {
            "report_kind": "model_generated_summary",
            "source_tool": tool,
            "raw_verification": {
                "status": verification.get("status"),
                "exit_code": verification.get("exit_code"),
                "timed_out": verification.get("timed_out"),
            },
        }
    if tool == "investigate_repository":
        return {
            "report_kind": "raw_repository_observations",
            "source_tool": "explorer",
            "model_summary_excluded_from_evidence": True,
            "raw_observation_count": len(payload.get("raw_observations") or []),
            "raw_observations": [
                {
                    "tool": item.get("tool"),
                    "observation": _clip(_raw_observation_excerpt(item.get("observation", "")), 500),
                }
                for item in (payload.get("raw_observations") or [])[:4]
            ],
        }
    if tool == "find_files":
        return {"report_kind": "raw_tool_observation", "source_tool": tool,
                **_find_files_provenance(payload)}
    return {"report_kind": "raw_tool_observation", "source_tool": tool}


def record_step_evidence(plan: dict, tool: str, arg: str, result: str, evidence: str,
                         provenance: dict | None = None) -> dict:
    """Attach what the just-finished delegation established to the plan's
    current step. Returns a new plan (plan.py style). A no-op when there is
    no current step. If one turn produced several results for the same step,
    the evidence is concatenated and the WORST result wins."""
    current_id = plan.get("current_step_id")
    if current_id is None:
        return plan
    order = 1 + max([s.get("order", 0) for s in plan["steps"]] + [0])
    rank = {RESULT_OK: 0, RESULT_INCONCLUSIVE: 1, RESULT_EMPTY: 2, RESULT_FAILED: 3}
    steps = []
    for step in plan["steps"]:
        if step["step_id"] == current_id:
            if step.get("result"):  # a second result for the same step
                evidence = f"{step['evidence']} | {evidence}"
                if rank[step["result"]] > rank[result]:
                    result = step["result"]
                order = step["order"]
            step = dict(step, tool=tool, action=f"{tool}: {_clip(arg, _ACTION_LABEL_CHARS)}",
                        action_key=_action_key(tool, arg), result=result,
                        evidence=evidence, provenance=provenance or {}, order=order)
        steps.append(step)
    return dict(plan, steps=steps)


def evidence_steps(plan: Optional[dict]) -> list[dict]:
    """The plan steps that carry evidence, in execution order."""
    if not plan:
        return []
    return sorted((s for s in plan["steps"] if s.get("result")), key=lambda s: s["order"])


def is_blind_repeat(plan: Optional[dict], tool: str, arg: str, mutating_tool: str) -> bool:
    """True if this exact action already ran WITHOUT usable evidence and
    nothing has changed since: repeating it can only fail the same way. The
    repository counts as changed once a later `mutating_tool` step ran."""
    steps = evidence_steps(plan)
    key = _action_key(tool, arg)
    bad = [s["order"] for s in steps if s["action_key"] == key and s["result"] != RESULT_OK]
    if not bad:
        return False
    return not any(s["order"] > max(bad) and s["tool"] == mutating_tool for s in steps)


# ------------------------------------------------------------------ grading


def evidence_for_grader(plan: Optional[dict], criteria: list[dict] | None = None) -> list[dict]:
    """The evidence steps shown to the grading LLM for this call.

    Grader-payload optimization only - does not touch plan/state, and does
    not change what `apply_assessment` is able to grant. While any criterion
    remains unresolved, every step is kept (there is no reliable way to know
    in advance which recorded evidence an unresolved criterion needs, so
    nothing is withheld from it). Only once EVERY criterion is already
    satisfied is the list narrowed, and then only to the evidence those
    satisfied criteria actually cite (`criterion["evidence_ids"]`, set by
    `apply_assessment`) - this is exactly the evidence the grader needs to
    reconfirm (or, per apply_assessment's own "recomputed from scratch"
    contract, withdraw) each satisfied verdict. Steps cited by no satisfied
    criterion are only ever dropped in this all-satisfied case, where no
    unresolved criterion exists that could have needed them.
    """
    steps = evidence_steps(plan)
    if criteria and all(c["satisfied"] for c in criteria):
        cited_ids = {step_id for c in criteria for step_id in (c.get("evidence_ids") or [])}
        steps = [s for s in steps if s["step_id"] in cited_ids]
    return [
        {"id": s["step_id"], "action": s["action"], "result": s["result"],
         "evidence": s["evidence"], "provenance": s.get("provenance", {})}
        for s in steps
    ]


def parse_assessment(text: str) -> dict[str, dict]:
    """Parse the grader's reply into {criterion_id: verdict}. Expects
    {"criteria": [{"id", "satisfied", "evidence_ids", "reason"}]}. Anything
    malformed yields {} (= nothing satisfied)."""
    cleaned = _CODE_FENCE_RE.sub("", (text or "").strip()).strip()
    try:
        data = json.loads(cleaned)
    except (TypeError, ValueError):
        return {}
    items = data.get("criteria") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return {}
    return {item["id"]: item for item in items if isinstance(item, dict) and isinstance(item.get("id"), str)}


_RELEVANCE_STOP = {"that", "this", "with", "from", "have", "been", "shown", "show", "true", "must", "which",
                   "objective", "fully", "addressed", "concrete", "results", "repository", "report", "reports",
                   "directly", "evidence", "file", "files", "about", "their", "there", "what", "where", "does"}
_RELEVANCE_STOP.update({"the", "and", "or", "but", "for", "from", "into", "with", "used", "use", "using",
                        "obtain", "obtained", "achieve", "achieved", "find", "found", "showing", "whether",
                        "answer", "answers", "task", "requested", "verification", "changes", "change"})
_FACET_SPLIT_RE = re.compile(r"\s*(?:,|;|\band\b|\bor\b|\bversus\b|\bvs\.?\b)\s*", re.IGNORECASE)


def _content_words(text: str) -> set[str]:
    return {w for w in tokenize(text or "") if len(w) >= 4 and w not in _RELEVANCE_STOP}


def _evidence_is_relevant(criterion: dict, plan: dict, cited: list[str]) -> bool:
    """Require evidence to cover the criterion's requested concepts.

    A single incidental overlap no longer makes a multi-term criterion
    relevant. Conjoined facets (for example, "strict and soft accuracy")
    must each occur in the cited evidence set. Identifier tokenization also
    splits CamelCase names so `ImageEncoder` can match `image encoder`.
    """
    description = str(criterion.get("description", ""))
    wanted = _content_words(description)
    if not wanted:
        return False
    by_id = {s["step_id"]: s for s in plan["steps"]}
    observed: set[str] = set()
    criterion_text = " ".join((str(criterion.get("description", "")),
                               str(criterion.get("evidence_needed", ""))))
    identifies_file = bool(re.search(r"\b(?:file|files|filename|path|module)\b", criterion_text, re.IGNORECASE))
    for step_id in cited:
        step = by_id.get(step_id) or {}
        evidence = str(step.get("evidence", ""))
        # get_file's path and line-range header is provenance, not evidence
        # for ordinary content claims. File-identification criteria are the
        # exception: use the actual retrieved path alongside source content.
        if step.get("tool") == "get_file" and ":\n" in evidence:
            evidence = evidence.split(":\n", 1)[1]
        observed |= _content_words(evidence)
        if identifies_file and step.get("tool") == "get_file":
            path = str((step.get("provenance") or {}).get("file") or "")
            if path:
                observed |= _content_words(path)
                # The extension is path evidence for a Python-file criterion;
                # its two-letter spelling would otherwise be discarded by
                # _content_words' minimum token length.
                if path.casefold().endswith(".py"):
                    observed.add("python")

    facets = [_content_words(part) for part in _FACET_SPLIT_RE.split(description)]
    facets = [facet for facet in facets if facet]
    if len(facets) > 1:
        return all(facet & observed for facet in facets)
    # A one-concept fact can be established by one matching term; multi-term
    # criteria need at least two independent content matches.
    return len(wanted & observed) >= min(2, len(wanted))


def apply_assessment(criteria: list[dict], plan: dict, verdicts: dict[str, dict]) -> tuple[list[dict], dict]:
    """The deterministic gate between the grader's opinion and the state.

    A criterion is satisfied only if the grader said `satisfied` is exactly
    True AND it cites at least one real step id whose result is `ok`. Cited
    ids that do not exist, or whose step produced no usable evidence, are
    dropped. Everything is recomputed from scratch on every call, so a later
    assessment can withdraw an earlier one. Returns (new criteria, new plan
    whose steps list the criteria they support)."""
    fully_read = _fully_read_files(plan)
    ok_steps = {s["step_id"] for s in evidence_steps(plan)
                if s["result"] == RESULT_OK
                or (s["result"] == RESULT_INCONCLUSIVE and s.get("tool") == "get_file"
                    and (s.get("provenance") or {}).get("file") in fully_read)}
    supports: dict[str, list[str]] = {}
    new_criteria = []
    for criterion in criteria:
        verdict = verdicts.get(criterion["id"], {})
        cited = verdict.get("evidence_ids")
        valid = [i for i in cited if i in ok_steps] if isinstance(cited, list) else []
        if valid and not _evidence_is_relevant(criterion, plan, valid):
            valid = []
        satisfied = verdict.get("satisfied") is True and bool(valid)
        reason = verdict.get("reason") if isinstance(verdict.get("reason"), str) else ""
        if verdict.get("satisfied") is True and not valid:
            reason = "Claimed satisfied but cited no step with usable, relevant evidence."
        new_criteria.append(dict(criterion, satisfied=satisfied, evidence_ids=valid if satisfied else [],
                                 note=_clip(reason, _NOTE_CHARS)))
        if satisfied:
            for step_id in valid:
                supports.setdefault(step_id, []).append(criterion["id"])
    steps = [dict(s, criteria=supports.get(s["step_id"], [])) if s.get("result") else s for s in plan["steps"]]
    return new_criteria, dict(plan, steps=steps)


def classify_support(criteria: list[dict], plan: Optional[dict], failure_info: Optional[dict],
                     verification_info: Optional[dict]) -> str:
    """What the gathered evidence supports - never stronger than that.

    BLOCKED     the latest code/verify run failed, or steps ran but none
                produced usable evidence.
    SUPPORTED   every criterion satisfied, no planned work left untouched,
                and verification (if run) was not unavailable.
    PARTIAL     some criteria satisfied, or all satisfied with work outstanding.
    UNSUPPORTED nothing satisfied (and nothing outright failed)."""
    steps = evidence_steps(plan)
    unavailable = bool(verification_info and (verification_info.get("status") == "unavailable"
                                              or verification_info.get("state") == "UNAVAILABLE"))
    # An unavailable verification is inconclusive, not a failure (plan.py's
    # classify_outcome ranks it INCOMPLETE even alongside a failure_info), so
    # it never makes the state BLOCKED.
    if not unavailable:
        if failure_info is not None:
            return BLOCKED
        if steps and not any(s["result"] == RESULT_OK for s in steps) \
                and any(s["result"] == RESULT_FAILED for s in steps):
            return BLOCKED
    satisfied = [c for c in criteria if c["satisfied"]]
    if not satisfied:
        return UNSUPPORTED
    if len(satisfied) < len(criteria) or pending_steps(plan) or unavailable:
        return PARTIAL
    if incomplete_reasons(criteria, plan):
        return PARTIAL
    return SUPPORTED


def can_drop_pending_readonly_steps(support: str, criteria: list[dict], plan: Optional[dict], *,
                                   light_readonly: bool, failure_info: Optional[dict]) -> bool:
    """Whether satisfied evidence makes leftover read-only plan steps unnecessary.

    A failed search/read does not invalidate a later successful result that
    supports every criterion. A current code/verification failure still blocks
    this shortcut through `failure_info`, and non-read-only tasks never use it.
    """
    return (
        support == PARTIAL
        and bool(criteria)
        and all(c["satisfied"] for c in criteria)
        and bool(pending_steps(plan))
        and light_readonly
        and failure_info is None
    )


# ------------------------------------------------------------- text for LLM


def describe_criteria(criteria: list[dict], plan: Optional[dict]) -> str:
    """Digest shown to `decide` (as a HumanMessage - it is data): what must be
    shown, whether the last grading supported it, and which attempts gave no
    usable evidence so they are not repeated unchanged."""
    if not criteria:
        return ""
    lines = ["Acceptance criteria (your final answer is checked against recorded evidence):"]
    for c in criteria:
        mark = "supported" if c["satisfied"] else "not yet supported"
        lines.append(f"  - [{c['id']}] {c['description']} -- {mark}"
                     + (f" (needs: {c['evidence_needed']})" if c["evidence_needed"] and not c["satisfied"] else ""))
    steps = evidence_steps(plan)
    if steps:
        lines.append("Evidence recorded so far:")
        for s in steps:
            flag = "" if s["result"] == RESULT_OK else " -- no usable evidence, do not repeat unchanged"
            lines.append(f"  - {s['step_id']} {s['action']} -> {s['result']}{flag}")
    return "\n".join(lines)


def completion_feedback(support: str, criteria: list[dict], plan: Optional[dict]) -> str:
    """Message that sends `decide` back to work instead of accepting a final
    answer the evidence does not support. Built from the ACTUAL blockers:
    unmet criteria (with what would show each), a grader that could not run,
    and/or planned steps never started - and it only asks for more evidence
    when a criterion is really unmet."""
    lines = [f"Completion check (system data): your answer was not accepted yet - evidence support is {support}."]
    unmet = [c for c in criteria if not c["satisfied"]]
    grader_down = [c for c in unmet if c["note"].startswith(GRADER_FAILED_NOTE)]
    really_unmet = [c for c in unmet if c not in grader_down]
    for c in really_unmet:
        line = f"  - [{c['id']}] not supported: {c['description']}"
        if c["note"]:
            line += f" (grader: {c['note']})"
        if c["evidence_needed"]:
            line += f" -- evidence that would show it: {c['evidence_needed']}"
        lines.append(line)
    if grader_down:
        lines.append("  - The grading call itself failed for: " + ", ".join(f"[{c['id']}]" for c in grader_down)
                     + ". That says nothing about your evidence.")
    left = pending_steps(plan)
    if left:
        lines.append("  - Planned steps never started: " + "; ".join(_clip(s["description"], 80) for s in left)
                     + ". Untouched planned steps keep the result PARTIAL until they are done or dropped.")
    done = [s for s in evidence_steps(plan) if s["result"] == RESULT_OK]
    if done:
        lines.append("Evidence already recorded with usable results (do not re-investigate what these cover): "
                     + "; ".join(f"{s['step_id']} {s['action']}" for s in done[-6:]))
    if really_unmet:
        lines.append("Gather ONLY the evidence for the unmet criteria above, with one targeted delegation aimed at "
                     "what each criterion needs (a report that confirms something is absent from the repository "
                     "counts as evidence). Do not repeat an attempt that gave no usable evidence, unless the "
                     "repository changed. If it truly cannot be obtained, give a final answer that states plainly "
                     "what could not be established."
                     + (" If a planned step is no longer needed, call revise_plan with only the steps you still "
                        "intend to do." if left else ""))
    elif left:
        lines.append("Every criterion is supported by recorded evidence; the only thing blocking acceptance is the "
                     "planned steps above. Do NOT delegate more investigation. Drop the steps you no longer need with "
                     "revise_plan (only the steps you genuinely still need), then give your final answer.")
    elif grader_down:
        lines.append("Do not gather more evidence because of this. Give your final answer again; grading will be retried.")
    for reason in incomplete_reasons(criteria, plan):
        lines.append(f"  - Not complete: {reason}.")
    return "\n".join(lines)


def completion_retry_allowed(step_count: int, step_budget: int, retries: int) -> bool:
    """Whether another tools-enabled decide turn remains for evidence recovery.

    `step_count` is the count before the current decide call. The current call
    consumes one turn, so a retry is useful only if its updated count remains
    below the effective budget. This must use the intent-scaled budget passed
    by the caller, not the larger configured ceiling.
    """
    return retries < MAX_COMPLETION_RETRIES and step_count + 1 < step_budget


_STEP_REF_RE = re.compile(r"\bstep_\d+\b")
_FILE_REF_RE = re.compile(r"[\w./\\-]+\.(?:py|md|txt|json|ya?ml|toml|ini|cfg|sh)\b")
_EXCERPT_CHARS = 240


def _sanitize_conclusion(text: str, plan: Optional[dict], corpus: str) -> tuple[str, list[str]]:
    """Keep the model's prose from citing things that were never
    recorded. Step ids that are not real recorded steps are removed; file
    names that appear nowhere in the recorded evidence/reports are returned
    so the response can flag them as unverified (never silently trusted)."""
    real = {s["step_id"] for s in evidence_steps(plan)}
    text = _STEP_REF_RE.sub(lambda m: m.group(0) if m.group(0) in real else "(unrecorded reference removed)", text)
    haystack = corpus.lower().replace("\\", "/")
    unverified = sorted({f for f in _FILE_REF_RE.findall(text)
                         if f.lower().replace("\\", "/").lstrip("./") not in haystack})
    return text, unverified


def _supporting_lines(criteria: list[dict], plan: Optional[dict]) -> list[str]:
    """Concise source evidence for the final answer, using only grader-accepted steps."""
    by_id = {s["step_id"]: s for s in evidence_steps(plan)}
    lines = []
    seen = set()
    for c in criteria:
        if not c["satisfied"]:
            continue
        for step_id in c["evidence_ids"]:
            step = by_id.get(step_id)
            if step is None or step_id in seen:
                continue
            seen.add(step_id)
            lines.append(f"- {step['action']}: {_clip(step['evidence'], _EXCERPT_CHARS)}")
    return lines


def _unresolved_lines(support: str, criteria: list[dict], plan: Optional[dict], failure_info: Optional[dict],
                      verification_info: Optional[dict]) -> list[str]:
    lines = [f"- [{c['id']}] Not established: {c['description']}"
             + (f" (grading call failed: {c['note'][len(GRADER_FAILED_NOTE):].strip(' ()')})"
                if c["note"].startswith(GRADER_FAILED_NOTE) else "")
             for c in criteria if not c["satisfied"]]
    left = pending_steps(plan)
    if support != SUPPORTED and left:
        lines.append("- Planned steps not attempted: " + "; ".join(_clip(s["description"], 80) for s in left))
    if verification_info and (verification_info.get("status") == "unavailable"
                              or verification_info.get("state") == "UNAVAILABLE"):
        lines.append("- Verification was unavailable, so the change is not verified.")
    if support != SUPPORTED:
        unusable = [s for s in evidence_steps(plan) if s["result"] != RESULT_OK]
        for s in unusable[-3:]:
            lines.append(f"- Attempt without usable evidence: {s['step_id']} ({s['action']}) -> {s['result']}")
    if failure_info is not None:
        lines.append(f"- The latest {failure_info.get('tool', 'delegated')} run failed.")
    if support != SUPPORTED:
        lines.extend(f"- Incomplete: {r}" for r in incomplete_reasons(criteria, plan))
    return lines


def build_final_response(support: str, criteria: list[dict], plan: Optional[dict], model_text: str,
                         failure_info: Optional[dict], verification_info: Optional[dict],
                         corpus: str, extra_unresolved: tuple = ()) -> tuple[str, str]:
    """Build the final answer from recorded state (deterministic).

    Returns (response text, unresolved text). Sections: Conclusion, Supporting
    evidence, Unresolved. The completion state is authoritative and is never
    contradicted: the model's own prose is used as the conclusion ONLY when
    the state is SUPPORTED (and then step ids it cites are checked against
    real steps, unseen file names flagged). For every other state the
    conclusion is fixed text; the model's prose is not shown, because it
    could claim more than the evidence supports. Evidence lines come only from
    recorded steps that the evidence gate accepted."""
    supporting = _supporting_lines(criteria, plan)
    unresolved = _unresolved_lines(support, criteria, plan, failure_info, verification_info)
    unresolved += [f"- {line}" for line in extra_unresolved]
    done = len(supporting)
    if support == SUPPORTED:
        text, unverified = _sanitize_conclusion(model_text.strip(), plan, corpus)
        if not text:
            text = "Established: " + " ".join(f"{c['description']}." for c in criteria)
        if unverified:
            unresolved.append("- Not found in the recorded evidence (treat as unverified): " + ", ".join(unverified))
    elif support == PARTIAL:
        if done == len(criteria):
            text = "The answer is supported, but some planned work remains unfinished."
        else:
            text = "The answer is only partly established; some requested facts remain unsupported."
    elif support == BLOCKED:
        text = ("The requested conclusion could not be established because the investigation was blocked "
                "(the latest change/verification failed, or no attempt produced usable evidence).")
    else:
        text = "The investigation could not establish the requested result from the recorded evidence."
    parts = ["Conclusion", text]
    if supporting:
        parts += ["", "Supporting evidence"] + supporting
    if unresolved:
        parts += ["", "Unresolved"] + unresolved
    return "\n".join(parts), "\n".join(unresolved)


def build_completion(support: str, criteria: list[dict], unresolved: str) -> dict:
    return {"support": support,
            "criteria_supported": sum(1 for c in criteria if c["satisfied"]),
            "criteria_total": len(criteria),
            "unmet": [c["id"] for c in criteria if not c["satisfied"]],
            "unresolved": unresolved}
