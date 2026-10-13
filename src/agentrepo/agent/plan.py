"""Structured plan state and transitions for Deep Agent invocations.

Plans hold pending and attempted steps in execution order. Revisions replace
only pending work, and a monotonic sequence keeps step identifiers unique
within the invocation. Helpers return updated plan dictionaries rather than
mutating existing state.
"""

from __future__ import annotations

import json
import re
from typing import Optional

from typing_extensions import TypedDict

PENDING = "pending"
IN_PROGRESS = "in_progress"
COMPLETED = "completed"
FAILED = "failed"

# The Deep Agent's overall task-level outcome (deep_agent.decide),
# distinct from safety.py's per-code_repository-run terminal states
# (SUCCESS/NO_TESTS/UNAVAILABLE/FAILED/TIMEOUT/REPAIR_LIMIT_REACHED). One of
# these describes the whole invocation; safety.py's states describe a single
# delegated verification/repair cycle.
OUTCOME_SUCCESS = "SUCCESS"
OUTCOME_FAILED = "FAILED"
OUTCOME_INCOMPLETE = "INCOMPLETE"


class PlanStep(TypedDict):
    step_id: str
    description: str
    status: str  # PENDING | IN_PROGRESS | COMPLETED | FAILED


class Plan(TypedDict):
    objective: str
    steps: list[PlanStep]
    # The step currently IN_PROGRESS, or None once every step recorded so
    # far has been closed out (completed/failed) and `decide` has not yet
    # started a new one.
    current_step_id: Optional[str]
    # Monotonic counter used to generate step_ids. Kept on the Plan itself
    # (rather than derived from len(steps)) because revise_pending_steps can
    # remove PENDING entries, which would otherwise let a freshly generated
    # id collide with one already used by an earlier, still-remembered step.
    step_seq: int


def new_plan(objective: str) -> Plan:
    """A fresh plan for a newly-received objective, with no steps yet."""
    return {"objective": objective, "steps": [], "current_step_id": None, "step_seq": 0}


def _new_step_id(plan: Plan) -> tuple[str, int]:
    seq = plan.get("step_seq", 0) + 1
    return f"step_{seq}", seq


def seed_pending_steps(plan: Plan, descriptions: list[str]) -> Plan:
    """Give a plan its initial (or additional) PENDING steps, in order.

    Used once per invocation by deep_agent.py's `plan_task` node, from an
    LLM-authored list of step descriptions - see parse_plan_steps. Does not
    touch `current_step_id`: seeding a plan with pending work never starts
    any of it.
    """
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
_MATCH_MIN_OVERLAP = 0.6


def _normalize(text: str) -> str:
    return " ".join(_WORD_RE.findall(text.lower()))


_MATCH_FILLER = {"the", "a", "an", "of", "in", "on", "to", "for", "and", "or", "each", "any", "all",
                 "file", "files", "repository", "return", "list", "find", "open", "with", "from", "that",
                 "is", "are", "be", "by", "as", "it", "its", "their", "e", "g"}


def fallback_step_description(objective: str) -> str:
    """The exact wording `plan_task` (deep_agent.py) uses to seed a plan
    with one generic PENDING step when no real multi-step plan exists yet -
    either the planning call failed/returned nothing parseable, or the task
    was routed as a lookup and skipped planning entirely (the common case
    for a short, single-fact question). Exposed here, rather than inlined at
    the call site, so `_match_pending_index` can recognise this specific
    template by exact text and not merely by "it happens to be the only
    PENDING step" (see the comment there for why that distinction matters).
    """
    return f"Investigate and address the objective: {objective}"


def _match_pending_index(steps: list, description: str, objective: Optional[str] = None) -> Optional[int]:
    """Index of the PENDING step that `description` most plausibly refers
    to, or None. Tried in order: the single-step fallback-template special
    case below; exact match after normalisation (case, whitespace,
    punctuation); one normalised text containing the other (the model
    usually shortens or extends a planned step's wording); then the best
    word-overlap (Jaccard) at or above _MATCH_MIN_OVERLAP. Ties go to the
    earliest step. None when nothing is close enough, in which case the
    caller records an ad-hoc step instead."""
    needle = _normalize(description)
    if not needle:
        return None
    pending = [(i, _normalize(s["description"])) for i, s in enumerate(steps) if s["status"] == PENDING]
    if objective is not None and len(pending) == 1:
        i, _ = pending[0]
        if steps[i]["description"] == fallback_step_description(objective):
            # This step does not describe independent work a delegation
            # could legitimately fail to match - it is a placeholder for
            # "whatever the agent investigates first", seeded because there
            # was no real plan. Wording-based matching below routinely fails
            # to link it to the model's own, differently (often much more
            # narrowly) phrased delegation query - especially for short
            # single-fact lookups, where the task may contribute only one or
            # two content words - which orphaned it as permanently PENDING
            # even on a fully successful run: a false "planned step not
            # attempted" note on an otherwise correct answer. Since it is
            # both the only pending step and provably this exact
            # placeholder, the next delegation is understood to be
            # attempting it, whatever words it uses.
            return i
    for i, text in pending:
        if text == needle:
            return i
    for i, text in pending:
        if text and (needle in text or text in needle):
            return i
    needle_words = set(needle.split())
    best_index, best_score = None, 0.0
    for i, text in pending:
        words = set(text.split())
        if not words:
            continue
        score = len(words & needle_words) / len(words | needle_words)
        if score > best_score:
            best_index, best_score = i, score
    if best_index is not None and best_score >= _MATCH_MIN_OVERLAP:
        return best_index
    # Jaccard punishes a long delegated query that merely restates a short planned
    # step, leaving the planned step PENDING forever (=> PARTIAL). Fall back to the
    # overlap coefficient over content words.
    needle_c = needle_words - _MATCH_FILLER
    best_index, best_score = None, 0.0
    for i, text in pending:
        words = set(text.split()) - _MATCH_FILLER
        shared = words & needle_c
        if len(shared) < 3 or not words:
            continue
        score = len(shared) / min(len(words), len(needle_c))
        if score > best_score:
            best_index, best_score = i, score
    return best_index if best_score >= 0.6 else None


def start_step(plan: Plan, description: str) -> Plan:
    """Record that the Deep Agent is now attempting `description` (the
    delegation call `decide` is about to make).

    If `description` matches an existing PENDING step (tolerantly - see
    `_match_pending_index`: the Deep Agent will not reproduce a planned
    step's wording byte-for-byte), that step is promoted to IN_PROGRESS and
    becomes the current step: this is genuine step *selection* against the
    plan, not a blind append. Otherwise a brand-new IN_PROGRESS step is
    appended - the Deep Agent
    going off-plan (new evidence made a different action necessary) is
    still recorded as real, honest history, not silently dropped.
    """
    steps = list(plan["steps"])
    match_index = _match_pending_index(steps, description, plan.get("objective"))
    seq = plan.get("step_seq", 0)
    if match_index is not None:
        current_id = steps[match_index]["step_id"]
        steps[match_index] = dict(steps[match_index], status=IN_PROGRESS)  # type: ignore[assignment]
    else:
        current_id, seq = _new_step_id(plan)
        steps.append({"step_id": current_id, "description": description, "status": IN_PROGRESS})
    return {"objective": plan["objective"], "steps": steps, "current_step_id": current_id, "step_seq": seq}


def finish_current_step(plan: Plan, succeeded: bool) -> Plan:
    """Record the result of the current step: COMPLETED if the delegation it
    represents succeeded, FAILED otherwise. A no-op (returns `plan`
    unchanged) if there is no current step - defensive only; `update_state`
    never calls this without one having just been started by `decide`."""
    current_id = plan.get("current_step_id")
    if current_id is None:
        return plan
    status = COMPLETED if succeeded else FAILED
    steps = [dict(s, status=status) if s["step_id"] == current_id else s for s in plan["steps"]]
    return {
        "objective": plan["objective"],
        "steps": steps,  # type: ignore[arg-type]
        "current_step_id": None,
        "step_seq": plan.get("step_seq", 0),
    }


def revise_pending_steps(plan: Plan, descriptions: list[str]) -> Plan:
    """Replace the plan's remaining (PENDING) work with a new list of step
    descriptions, in order - the effect of the Deep Agent's `revise_plan`
    tool (deep_agent.py). Steps that are already IN_PROGRESS, COMPLETED or
    FAILED are recorded facts and are never touched or removed.
    """
    kept = [s for s in plan["steps"] if s["status"] != PENDING]
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


def is_complete(plan: Optional[Plan]) -> bool:
    """Whether every step recorded so far ended COMPLETED - false for an
    empty plan (nothing attempted yet) as well as for one with any
    PENDING/IN_PROGRESS/FAILED step."""
    return bool(plan) and bool(plan["steps"]) and all(s["status"] == COMPLETED for s in plan["steps"])


def describe_step(step: PlanStep) -> str:
    """One-line trace description of a single step, e.g. for the console
    trace (see agent/utils.describe_update)."""
    return f"{step['step_id']} ({step['status']}): {step['description']}"


def describe_plan(plan: Optional[Plan]) -> str:
    """A short plan-status digest, injected as an extra message into
    `decide`'s LLM call (deep_agent.py) so step selection is genuinely
    informed by what is left to do and what has already happened, instead
    of the model only ever seeing raw tool-result text. Returns "" when
    there is nothing to say (no plan yet, or a plan with no steps at all -
    e.g. `plan_task` could not produce anything and the Deep Agent has not
    delegated yet either)."""
    if not plan or not plan["steps"]:
        return ""
    lines = ["Plan status for this task:"]
    pending = pending_steps(plan)
    if pending:
        lines.append("Not yet attempted:")
        lines.extend(f"  - {s['description']}" for s in pending)
    attempted = [s for s in plan["steps"] if s["status"] != PENDING]
    if attempted:
        lines.append("Already attempted:")
        lines.extend(f"  - {describe_step(s)}" for s in attempted)
    lines.append(
        "You do not have to follow the listed order - pursue whichever step best fits the evidence "
        "gathered so far. Use revise_plan if the plan needs to change. Steps left untouched keep the "
        "completion check at PARTIAL, so drop any you will not do."
    )
    return "\n".join(lines)


_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", flags=re.MULTILINE)


def parse_plan_steps(text: str, max_steps: int) -> list[str]:
    """Parse an LLM planning reply (deep_agent.py's `plan_task` node) into a
    list of plain step descriptions.

    Accepts a JSON array of strings, or of objects with a "description"/
    "step"/"title" field (some models wrap steps in objects despite being
    asked for plain strings). Also accepts the planning call's
    object form {"steps": [...], "criteria": [...]} and reads only "steps"
    here (evidence.parse_criteria reads the criteria from the same reply).
    Tolerates surrounding markdown code fences.
    Returns [] (never raises) when nothing usable could be parsed - the
    caller decides what generic fallback to use, so this module never has
    to hard-code a plan for any particular kind of task.
    """
    cleaned = _CODE_FENCE_RE.sub("", text.strip()).strip()
    try:
        data = json.loads(cleaned)
    except (TypeError, ValueError):
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


def parse_objective_verdict(text: str) -> bool:
    """Parse the objective-check reply (deep_agent.py's `_objective_satisfied`)
    into a strict yes/no. Expects a JSON object with a boolean "satisfied"
    field (tolerating code fences). Anything else - unparseable text, a
    missing field, a non-boolean value - is False: the objective is only
    treated as satisfied on an explicit, well-formed affirmative."""
    cleaned = _CODE_FENCE_RE.sub("", (text or "").strip()).strip()
    try:
        data = json.loads(cleaned)
    except (TypeError, ValueError):
        return False
    return isinstance(data, dict) and data.get("satisfied") is True


def classify_outcome(
    plan: Optional[Plan],
    failure_info: Optional[dict],
    step_limit_reached: bool,
    has_evidence: bool = False,
    objective_satisfied: bool = False,
    verification_info: Optional[dict] = None,
) -> str:
    """The Deep Agent's task-level outcome for this invocation, computed
    from objective signals rather than trusted from the model's own final
    answer text (see the module docstring and deep_agent.py's `decide`):

    - FAILED: the most recent code_repository/verify_repository delegation
      has an explicit failing result (`failure_info` is not None - the same
      field update_state clears to None on later success). An unavailable
      verification result is checked first and remains INCOMPLETE even if a
      repair wrapper also reports that its attempt limit was reached.
    - INCOMPLETE: the model stopped delegating while planned work was still
      PENDING and unaddressed (it never called revise_plan to say that work
      was no longer needed), OR the objective itself has not been shown to
      be satisfied (see below). Consuming the final allowed step alone does
      not make a task incomplete when the final evidence check confirms it.
    - SUCCESS: only when, in addition to no outstanding failure and no
      untouched planned work, there is objective-level evidence: at least
      one delegated report was actually observed (`has_evidence`), and the
      objective check against those reports/verification results
      (`objective_satisfied`, from deep_agent.py) confirmed the user's
      objective is met. Completed plan steps, a Coder reporting success, or
      the model producing a final answer are not sufficient on their own.
    """
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
