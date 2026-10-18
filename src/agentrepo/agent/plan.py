"""Structured plan state and transitions for Main Agent invocations.

Plans hold pending and attempted steps in execution order. Plan steps describe
goals rather than mandatory tool names; direct repository tool calls bind to
compatible pending steps based on intent and domain/target overlap rather than
tool-name matching. Revisions replace only pending work, and a monotonic
sequence keeps step identifiers unique within the invocation.
"""

from __future__ import annotations

import re
from typing import Optional

from typing_extensions import NotRequired, TypedDict

from .json_utils import parse_json_object

PENDING = "pending"
IN_PROGRESS = "in_progress"
COMPLETED = "completed"
FAILED = "failed"
SKIPPED = "skipped"

OUTCOME_SUCCESS = "SUCCESS"
OUTCOME_FAILED = "FAILED"
OUTCOME_INCOMPLETE = "INCOMPLETE"


class PlanStep(TypedDict):
    step_id: str
    description: str
    status: str  # PENDING | IN_PROGRESS | COMPLETED | FAILED | SKIPPED
    resolution_reason: NotRequired[str]


class Plan(TypedDict):
    objective: str
    steps: list[PlanStep]
    current_step_id: Optional[str]
    step_seq: int


def new_plan(objective: str) -> Plan:
    """A fresh plan for a newly-received objective, with no steps yet."""
    return {"objective": objective, "steps": [], "current_step_id": None, "step_seq": 0}


def _new_step_id(plan: Plan) -> tuple[str, int]:
    seq = plan.get("step_seq", 0) + 1
    return f"step_{seq}", seq


def seed_pending_steps(plan: Plan, descriptions: list[str]) -> Plan:
    """Give a plan its initial (or additional) PENDING steps, in order."""
    steps = list(plan["steps"])
    seq = plan.get("step_seq", 0)
    for description in descriptions:
        seq += 1
        steps.append({"step_id": f"step_{seq}", "description": description, "status": PENDING})
    return {
        "objective": plan["objective"],
        "steps": steps,
        "current_step_id": plan.get("current_step_id"),
        "step_seq": seq,
    }


_WORD_RE = re.compile(r"[a-z0-9_]+")
_SUBTOKEN_RE = re.compile(r"[a-z0-9]+")
_MATCH_MIN_OVERLAP = 0.6

_TOOL_PREFIXES = (
    "search_repository", "find_files", "find_symbol", "get_file", "get_symbol",
    "find_callers", "find_callees", "find_importers", "get_dependencies", "get_ast",
    "write_file", "verify_repository", "run_command", "revise_plan",
)
_GENERIC_STEP_WORDS = frozenset({
    "the", "a", "an", "in", "on", "of", "to", "for", "from", "with", "by", "and", "or",
    "use", "using", "call", "tool", "repository", "repo", "codebase", "file", "files",
    "symbol", "symbols", "function", "functions", "method", "methods", "class", "classes",
    "module", "modules", "search", "find", "get", "read", "inspect", "locate", "trace",
    "check", "investigate", "explore", "identify", "determine", "understand", "explain",
    "how", "what", "where", "which", "who", "when", "why", "py", "md", "src",
})
_MUTATION_STEP_RE = re.compile(
    r"\b(?:write|modify|edit|update|fix|implement|patch|create|add|remove|delete|replace|refactor|write_file)\b",
    re.IGNORECASE,
)
_VERIFY_STEP_RE = re.compile(
    r"\b(?:verify|test|tests|run_command|verify_repository|pytest|syntax|validate|validation)\b",
    re.IGNORECASE,
)


def _normalize(text: str) -> str:
    return " ".join(_WORD_RE.findall(text.lower()))


def _strip_tool_names(text: str) -> str:
    lowered = text.lower()
    for t_name in _TOOL_PREFIXES:
        lowered = re.sub(rf"\b{re.escape(t_name)}\b\s*:?", " ", lowered)
    return " ".join(_WORD_RE.findall(lowered))


def _meaningful_subtokens(text: str) -> set[str]:
    stripped = _strip_tool_names(text)
    tokens: set[str] = set()
    for word in _WORD_RE.findall(stripped):
        if word not in _GENERIC_STEP_WORDS and len(word) > 1:
            tokens.add(word)
        for sub in _SUBTOKEN_RE.findall(word.replace("_", " ")):
            if sub not in _GENERIC_STEP_WORDS and len(sub) > 1:
                tokens.add(sub)
    return tokens


def fallback_step_description(objective: str) -> str:
    """The wording `plan_task` uses to seed a plan with one generic PENDING step."""
    return f"Investigate and address the objective: {objective}"


def _step_kind_compatible(step_desc: str, tool_name: str) -> bool:
    """Prevent a read-only tool call from consuming a mutation/verification step,
    or a mutation call from consuming a pure verification step."""
    if not tool_name:
        return True
    is_write_step = bool(_MUTATION_STEP_RE.search(step_desc))
    is_verify_step = bool(_VERIFY_STEP_RE.search(step_desc)) and not is_write_step
    if tool_name == "write_file":
        return is_write_step or not is_verify_step
    if tool in ("verify_repository", "run_command") if (tool := tool_name) else False:
        return is_verify_step or is_write_step
    if is_write_step or is_verify_step:
        return False
    return True


def _match_pending_index(
    steps: list,
    description: str,
    objective: Optional[str] = None,
    tool_name: str = "",
) -> Optional[int]:
    """Index of the PENDING step that `description` refers to, decoupled from tool names.

    Tries in order:
      1. The single-step fallback-template special case.
      2. Exact match after stripping tool-name prefixes.
      3. Containment after stripping tool-name prefixes (unique match).
      4. Word/subtoken overlap against compatible pending steps.
      5. When there is only one compatible pending investigation step and it
         shares at least one meaningful domain token with `description`, bind to
         it so direct tool calls advance planned investigation steps instead of
         leaving them permanently pending.
    """
    needle = _strip_tool_names(description) or _normalize(description)
    if not needle:
        return None
    if not tool_name and ":" in description:
        prefix = description.split(":", 1)[0].strip().lower()
        if prefix in _TOOL_PREFIXES:
            tool_name = prefix

    pending = [
        (i, _strip_tool_names(s["description"]) or _normalize(s["description"]), s["description"])
        for i, s in enumerate(steps)
        if s["status"] == PENDING and _step_kind_compatible(s["description"], tool_name)
    ]
    if not pending:
        return None

    if objective is not None and len(pending) == 1:
        i, _, raw_desc = pending[0]
        if raw_desc == fallback_step_description(objective):
            return i

    exact = [i for i, text, _ in pending if text == needle]
    if exact:
        return exact[0] if len(exact) == 1 else None

    contained = [i for i, text, _ in pending if text and (needle in text or text in needle)]
    if contained:
        return contained[0] if len(contained) == 1 else None

    needle_words = set(needle.split())
    scored = []
    for i, text, _ in pending:
        words = set(text.split())
        if words:
            score = len(words & needle_words) / len(words | needle_words)
            if score >= _MATCH_MIN_OVERLAP:
                scored.append((score, i))
    scored.sort(reverse=True)
    if scored and (len(scored) == 1 or scored[0][0] > scored[1][0]):
        return scored[0][1]

    needle_tokens = _meaningful_subtokens(description)
    if needle_tokens:
        token_scored = []
        for i, _, raw_desc in pending:
            step_tokens = _meaningful_subtokens(raw_desc)
            if not step_tokens:
                continue
            shared = len(needle_tokens & step_tokens)
            if shared > 0:
                coverage = shared / min(len(needle_tokens), len(step_tokens))
                jaccard = shared / len(needle_tokens | step_tokens)
                token_scored.append((coverage, jaccard, shared, -i, i))
        token_scored.sort(reverse=True)
        if token_scored:
            best = token_scored[0]
            if best[0] >= 0.5 or best[2] >= 2 or len(pending) == 1:
                if len(token_scored) == 1 or (best[0], best[1], best[2]) > (
                    token_scored[1][0], token_scored[1][1], token_scored[1][2]
                ):
                    return best[4]

    return None


def start_step_for_call(plan: Plan, description: str, tool_name: str = "") -> tuple[Plan, str]:
    """Start a planned or ad-hoc step for one tool call and return `(updated_plan, step_id)`."""
    steps = list(plan["steps"])
    match_index = _match_pending_index(steps, description, plan.get("objective"), tool_name=tool_name)
    seq = plan.get("step_seq", 0)
    if match_index is not None:
        current_id = steps[match_index]["step_id"]
        steps[match_index] = dict(steps[match_index], status=IN_PROGRESS)  # type: ignore[assignment]
    else:
        current_id, seq = _new_step_id(plan)
        steps.append({"step_id": current_id, "description": description, "status": IN_PROGRESS})
    updated: Plan = {
        "objective": plan["objective"],
        "steps": steps,
        "current_step_id": current_id,
        "step_seq": seq,
    }
    return updated, current_id


def start_step(plan: Plan, description: str) -> Plan:
    """Record that the Main Agent is now attempting `description`."""
    updated, _ = start_step_for_call(plan, description)
    return updated


def finish_step(plan: Plan, step_id: Optional[str], succeeded: bool, reason: Optional[str] = None) -> Plan:
    """Record the outcome (`COMPLETED` or `FAILED`) of `step_id`."""
    target_id = step_id or plan.get("current_step_id")
    if target_id is None:
        return plan
    status = COMPLETED if succeeded else FAILED
    extra = {"resolution_reason": reason} if reason and not succeeded else {}
    steps = [dict(s, status=status, **extra) if s["step_id"] == target_id else s for s in plan["steps"]]
    next_in_progress = next(
        (s["step_id"] for s in steps if s.get("status") == IN_PROGRESS),
        None,
    )
    return {
        "objective": plan["objective"],
        "steps": steps,  # type: ignore[arg-type]
        "current_step_id": None if plan.get("current_step_id") == target_id else next_in_progress,
        "step_seq": plan.get("step_seq", 0),
    }


def finish_current_step(plan: Plan, succeeded: bool, reason: Optional[str] = None) -> Plan:
    """Record the result of `plan['current_step_id']`."""
    return finish_step(plan, plan.get("current_step_id"), succeeded, reason=reason)


def revise_pending_steps(plan: Plan, descriptions: list[str], *, reason: str = "Plan revised") -> Plan:
    """Resolve remaining PENDING work as skipped, then append the revised step list."""
    kept = [
        dict(s, status=SKIPPED, resolution_reason=reason)
        if s["status"] == PENDING else s
        for s in plan["steps"]
    ]
    seq = plan.get("step_seq", 0)
    steps = list(kept)
    for description in descriptions:
        seq += 1
        steps.append({"step_id": f"step_{seq}", "description": description, "status": PENDING})
    return {
        "objective": plan["objective"],
        "steps": steps,
        "current_step_id": plan.get("current_step_id"),
        "step_seq": seq,
    }


def pending_steps(plan: Optional[Plan]) -> list[PlanStep]:
    """The steps not yet attempted, in plan order."""
    if not plan:
        return []
    return [s for s in plan["steps"] if s["status"] == PENDING]


def is_readonly_step_description(description: str) -> bool:
    """True when a step description represents read-only investigation rather
    than a code modification or command execution."""
    text = str(description or "")
    return not (_MUTATION_STEP_RE.search(text) or _VERIFY_STEP_RE.search(text))


def describe_step(step: PlanStep) -> str:
    """One-line trace description of a single step."""
    return f"{step['step_id']} ({step['status']}): {step['description']}"


def describe_plan(plan: Optional[Plan]) -> str:
    """A short plan-status digest injected into `decide`'s LLM call."""
    if not plan or not plan["steps"]:
        return ""
    lines = ["Plan status for this task:"]
    pending = pending_steps(plan)
    if pending:
        lines.append("Not yet attempted:")
        lines.extend(f"  - {s['description']}" for s in pending)
    skipped = [s for s in plan["steps"] if s["status"] == SKIPPED]
    if skipped:
        lines.append("Dropped as unnecessary or superseded:")
        lines.extend(f"  - {s['description']} ({s.get('resolution_reason', 'no longer needed')})" for s in skipped)
    attempted = [s for s in plan["steps"] if s["status"] in (IN_PROGRESS, COMPLETED, FAILED)]
    if attempted:
        lines.append("Already attempted:")
        lines.extend(f"  - {describe_step(s)}" for s in attempted)
    lines.append(
        "Pursue whichever repository tool call(s) best establish the unmet acceptance criteria. "
        "Use revise_plan if the remaining steps need to change."
    )
    return "\n".join(lines)


def parse_plan_steps(text: str, max_steps: int) -> list[str]:
    """Parse an LLM planning reply into a list of plain step descriptions."""
    data = parse_json_object(text)
    if data is None:
        return []
    if isinstance(data, dict):
        data = data.get("steps")
    if not isinstance(data, list):
        return []
    steps: list[str] = []
    for item in data:
        if isinstance(item, str) and item.strip():
            steps.append(item.strip())
        elif isinstance(item, dict):
            text_value = item.get("description") or item.get("step") or item.get("title")
            if isinstance(text_value, str) and text_value.strip():
                steps.append(text_value.strip())
    return steps[: max(1, max_steps)]


def classify_outcome(
    plan: Optional[Plan],
    failure_info: Optional[dict],
    step_limit_reached: bool,
    has_evidence: bool = False,
    objective_satisfied: bool = False,
    verification_info: Optional[dict] = None,
) -> str:
    """The Main Agent's task-level outcome for this invocation."""
    if verification_info and (verification_info.get("status") == "unavailable"
                              or verification_info.get("state") == "UNAVAILABLE"):
        return OUTCOME_INCOMPLETE
    if failure_info is not None:
        return OUTCOME_FAILED
    if pending_steps(plan):
        return OUTCOME_INCOMPLETE
    if not (has_evidence and objective_satisfied):
        return OUTCOME_INCOMPLETE
    return OUTCOME_SUCCESS
