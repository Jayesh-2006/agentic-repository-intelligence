"""
Safety and reliability policy for the Coder -> Verification -> Debugger
repair loop in delegation.py.

This module is deliberately plain Python control flow (no LLM calls), the
same way verification.py is - it decides *when* to checkpoint, roll back and
what terminal state to report, based on objective signals (verification
status, repair attempt count, changed files), not a scoring system.

The actual checkpoint/diff/rollback mechanics live on RepoService
(service.py); this module only holds the policy around them plus small
formatting helpers so delegation.py's control flow stays readable.
"""

from __future__ import annotations

from ..service import RepoService

# Explicit terminal states for a code_repository run. Exactly one of these
# describes how the repair loop ended; the Main Agent's report always makes
# this unambiguous instead of just trailing off after the last attempt.
SUCCESS = "SUCCESS"
NO_TESTS = "NO_TESTS"
UNAVAILABLE = "UNAVAILABLE"
FAILED = "FAILED"
TIMEOUT = "TIMEOUT"
REPAIR_LIMIT_REACHED = "REPAIR_LIMIT_REACHED"

_STATUS_TO_STATE = {
    "passed": SUCCESS,
    "no_tests": NO_TESTS,
    # run_scoped_verification's "nothing to check" outcome (verification.py) -
    # same terminal meaning as "no_tests": nothing needed verifying.
    "no_changes": NO_TESTS,
    "not_configured": UNAVAILABLE,
    "unavailable": UNAVAILABLE,
    "timeout": TIMEOUT,
    "failed": FAILED,
}


def terminal_state(verification: dict, repair_limit_reached: bool) -> str:
    """Classify how a code_repository run ended. `repair_limit_reached` is
    True only when the Debugger was invoked config.max_repair_attempts times
    and verification is still not passing - a distinct state from a plain
    "failed", because it means the *repair budget*, not just this one
    attempt, ran out."""
    if repair_limit_reached and not verification.get("passed"):
        return REPAIR_LIMIT_REACHED
    return _STATUS_TO_STATE.get(verification.get("status"), FAILED)


def verification_signature(verification: dict) -> tuple:
    """A cheap fingerprint of a verification result, used to notice that two
    repair attempts in a row produced the *same* failure (i.e. the repair is
    not making progress). Deliberately simple: exit code + status + the
    start of stderr - exact-text sensitive, but that is fine since it is only
    ever compared to a fingerprint taken the same way."""
    return (
        verification.get("status"),
        verification.get("exit_code"),
        (verification.get("stderr") or "")[:300],
    )


def format_change_summary(diff: dict) -> str:
    """Human-readable one-liner for a RepoService.diff_since_checkpoint()
    result, for inclusion in the report returned to the Main Agent."""
    if not isinstance(diff, dict) or "error" in diff or diff.get("changed") is None:
        return ""
    if not diff["changed"]:
        return "No files were changed."
    parts = []
    if diff["added"]:
        parts.append(f"{len(diff['added'])} file(s) added ({', '.join(diff['added'][:5])}{'...' if len(diff['added']) > 5 else ''})")
    if diff["modified"]:
        parts.append(f"{len(diff['modified'])} file(s) modified ({', '.join(diff['modified'][:5])}{'...' if len(diff['modified']) > 5 else ''})")
    if diff["deleted"]:
        parts.append(f"{len(diff['deleted'])} file(s) deleted ({', '.join(diff['deleted'][:5])}{'...' if len(diff['deleted']) > 5 else ''})")
    return "Changes: " + "; ".join(parts) + "."


class Checkpoint:
    """A task checkpoint and its creation status.

    A disabled checkpoint is an explicit configuration choice. A failed
    checkpoint is different: callers must not start an autonomous write run
    without the protection that was requested.
    """

    def __init__(self, service: RepoService, checkpoint_id: str | None,
                 error: dict | None = None):
        self.service = service
        self.checkpoint_id = checkpoint_id
        self.error = error

    @classmethod
    def create(cls, service: RepoService, enabled: bool, label: str) -> "Checkpoint":
        if not enabled:
            return cls(service, None)
        result = service.create_checkpoint(label=label[:200])
        checkpoint_id = result.get("checkpoint_id")
        if not isinstance(checkpoint_id, str) or not checkpoint_id:
            return cls(service, None, result.get("error") or {
                "code": "checkpoint_creation_failed",
                "message": "checkpoint creation did not return a checkpoint id",
            })
        return cls(service, checkpoint_id)

    @property
    def available(self) -> bool:
        """Whether an enabled checkpoint was successfully created."""
        return self.error is None

    def diff(self) -> dict:
        if self.checkpoint_id is None:
            return {"changed": None}
        return self.service.diff_since_checkpoint(self.checkpoint_id)

    def rollback(self) -> dict | None:
        if self.checkpoint_id is None:
            return None
        return self.service.rollback_to_checkpoint(self.checkpoint_id)

    def discard(self) -> None:
        if self.checkpoint_id is not None:
            self.service.discard_checkpoint(self.checkpoint_id)
            self.checkpoint_id = None
