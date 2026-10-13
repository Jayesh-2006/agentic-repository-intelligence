"""Structured state shared by the Deep Agent's orchestration nodes.

The state extends the conversation fields with invocation-scoped task,
plan, evidence, intent, capability, and completion data. Optional fields
are populated as the graph runs; readers provide defaults for older or
hand-built states.
"""

from __future__ import annotations

from typing import Optional

from .plan import Plan
from .state import AgentState


class DeepAgentState(AgentState, total=False):
    # The user's original task/request. Set from the first human message if
    # not provided explicitly when the graph is invoked.
    task: str

    # Optional caller-supplied pointers to where an answer might
    # live - e.g. app.py's simple_retrieval.gather_hint after the no-LLM
    # retrieval shortcut could not itself answer. Re-injected into `decide`'s
    # system_messages every turn (same pattern as plan/criteria status
    # below), NEVER into `state["messages"]` - a second leading human message
    # there would both get mistaken for `task` by `_task_from_messages` and
    # get silently dropped by `bounded_model_history`. It is a hint only: it
    # never enters `plan["steps"]`, so it cannot be cited as evidence until
    # the agent actually reads it via get_file/investigate_repository and
    # clears the normal evidence gate.
    retrieval_hint: Optional[str]

    # The Deep Agent's explicit, structured planning state (see
    # plan.py) - PENDING steps seeded up front from the objective, plus the
    # ordered record of every step this invocation has actually attempted,
    # each with its own status, plus which one (if any) is currently in
    # progress. None only before `plan_task` (deep_agent.py) has run; a fresh
    # `deep_agent.invoke(...)` call always starts with no plan.
    plan: Optional[Plan]

    # Repository investigation/evidence gathered so far (one entry per
    # investigate_repository report received; also one entry per
    # direct get_file report received).
    evidence: list[str]

    # What the Deep Agent is currently attempting, in plain language: the
    # query/task argument of its most recent delegation call, set by
    # `decide` (deep_agent.py) each time it asks to delegate. None before
    # the first delegation, or once `decide` has given a final answer
    # instead of delegating again.
    current_objective: Optional[str]

    # The raw report text of the most recently completed delegation
    # (investigate_repository or code_repository), regardless of which.
    delegated_result: Optional[str]

    # Every delegated report received so far, in order - the running
    # observation log the Deep Agent's planner reads
    # to decide what still needs to happen.
    observations: list[str]

    # The most recent code_repository run's verification outcome, as
    # reported by the verification/repair loop
    # (delegation.py). None until a code_repository call has completed.
    verification_info: Optional[dict]

    # Explicit failure/repair information from the most recent
    # code_repository/verify_repository run. Verification availability and
    # its exact status remain separately represented in verification_info.
    failure_info: Optional[dict]

    # The Deep Agent's final answer/result for the task, set once it stops
    # delegating and reports back.
    final_result: Optional[str]

    # The Deep Agent's task-level terminal outcome for this
    # invocation - one of plan.OUTCOME_SUCCESS / OUTCOME_FAILED /
    # OUTCOME_INCOMPLETE (see plan.classify_outcome). Set by `decide`
    # (deep_agent.py) at the same time as `final_result`, whether that is a
    # genuine final answer or the forced step-limit wrap-up. None until the
    # invocation actually ends.
    outcome: Optional[str]


    # The acceptance criteria for this invocation's objective (see
    # evidence.py) - each {"id", "description", "evidence_needed",
    # "satisfied", "evidence_ids", "note"}. Seeded by `plan_task`; `satisfied`
    # is only ever set true by evidence.apply_assessment, from recorded step
    # evidence. The evidence itself lives on the plan's steps (`tool`,
    # `action`, `result`, `evidence`, `order`, `criteria` keys), next to each
    # step's status. `evidence` above (raw investigate reports) is unchanged.
    criteria: list[dict]

    # What the recorded evidence supports, set when the invocation
    # ends: {"support": SUPPORTED/PARTIAL/UNSUPPORTED/BLOCKED,
    # "criteria_supported", "criteria_total", "unmet", "unresolved"}. `outcome`
    # is derived consistently from it (SUPPORTED <-> SUCCESS, BLOCKED ->
    # FAILED, otherwise INCOMPLETE). None until the invocation ends normally.
    completion: Optional[dict]

    # How many times `decide` sent a final answer back for more
    # evidence (bounded by evidence.MAX_COMPLETION_RETRIES, on top of the
    # existing step budget).
    completion_retries: int

    # Invocation-start snapshot used to report this query's evidence grader
    # calls/cache activity even though the compiled graph is reused.
    grader_metrics_start: dict

    # Invocation-local counters for deterministic orchestration and repeated
    # no-progress actions. Values are reported with the final runtime metrics.
    control_metrics: dict

    # Semantic observations already seen and the current progress generation.
    result_fingerprints: list[str]
    progress_generation: int
    no_progress_count: int
    blocked_read_actions: list[str]

    # A vetted, read-only follow-up queued by update_state for the next graph
    # edge. It is consumed by prepare_followup before ToolNode runs.
    pending_followup: Optional[dict]

    # Last bounded planning/orchestration model failure, if the invocation
    # had to use a safe fallback or stop without a model decision.
    model_error: Optional[str]

    # This invocation's resolved request intent (see intent.py) -
    # {"kind", "depth", "rigor", "answer_style", "sources"}. Set once by
    # `plan_task`, alongside `plan`/`criteria`. None only before `plan_task`
    # has run; read it via intent.get_intent(state) (or
    # `state.get("intent") or intent.default_intent()`) so old/hand-built
    # states without it still work.
    intent: Optional[dict]

    # Runtime safety boundary: this invocation's granted capabilities -
    # {"read": True, "write": bool, "execute": bool} (see intent.capabilities_for_floor).
    # Set ONCE by `plan_task`, from the deterministic keyword floor over the
    # user's own original task text alone - never from the planning LLM's
    # proposed intent, and never updated afterwards by anything a delegated
    # sub-agent or a later LLM-generated tool call produces. `decide`
    # (deep_agent.py) consults this - not the LLM's chosen tool name or
    # arguments - before letting a WRITE-capable delegation (currently
    # `code_repository`) actually execute. None only before `plan_task` has
    # run; read it via `state.get("capabilities") or
    # intent.default_capabilities()`, which defaults to read-only, so an
    # old/hand-built state without it can never be read as granting WRITE.
    capabilities: Optional[dict]
