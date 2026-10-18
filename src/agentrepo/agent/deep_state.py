"""State shared by the Main Agent's orchestration nodes."""

from __future__ import annotations

from typing import Optional

from .plan import Plan
from .state import AgentState


class DeepAgentState(AgentState, total=False):
    # Original request text used for routing and capability checks.
    task: str

    # The plan records attempted steps and their evidence.
    plan: Optional[Plan]
    # Correlates in-flight tool_call_ids to their started plan step_ids.
    active_step_ids: dict[str, str]
    # Recorded tool evidence and observations.
    evidence: list[str]
    observations: list[str]
    current_objective: Optional[str]
    delegated_result: Optional[str]

    # Per-invocation serialized MutationTransaction state and public snapshot.
    tx_state: Optional[dict]
    transaction: Optional[dict]
    verification_info: Optional[dict]
    failure_info: Optional[dict]

    # Terminal answer and status for this invocation.
    final_result: Optional[str]
    # `outcome` describes task execution; `completion` describes evidence support.
    outcome: Optional[str]
    criteria: list[dict]
    completion: Optional[dict]
    completion_retries: int
    completion_retry_progress_generation: Optional[int]

    # Invocation-local grader caches, counters, and orchestration metrics.
    runtime_metrics: dict
    grader_metrics: dict
    grader_metrics_start: dict
    assessment_cache: dict[str, dict]
    conclusion_cache: dict[str, dict]
    control_metrics: dict
    # Explicit ledger of started/finished actions (see execution.py).
    execution: Optional[dict]
    result_fingerprints: list[str]
    progress_generation: int
    no_progress_count: int
    blocked_read_actions: list[str]
    pending_followup: Optional[dict]
    model_error: Optional[str]

    intent: Optional[dict]
    # Derived from the user's request and never upgraded by model output.
    capabilities: Optional[dict]
