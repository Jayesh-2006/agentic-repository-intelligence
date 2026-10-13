"""Delegation tools for repository investigation, changes, and verification."""

from __future__ import annotations

import json
from typing import Callable

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from ..service import RepoService
from .coder import build_coder, coder_report
from .config import AgentConfig
from .debugger import build_debugger, debugger_report
from .explorer import build_explorer, explorer_report
from .prompts import DEBUGGER_TASK_TEMPLATE
from .safety import Checkpoint, format_change_summary, terminal_state, verification_signature
from .utils import written_files
from .verification import run_scoped_verification, run_verification, verification_summary


class InvestigateRepositoryInput(BaseModel):
    query: str = Field(
        min_length=1,
        description="A focused repository question for the Explorer to investigate with read-only tools.",
    )


class CodeRepositoryInput(BaseModel):
    task: str = Field(
        min_length=1,
        description="A focused coding/implementation task for the Coder to carry out with read and write repository tools.",
    )


class VerifyRepositoryInput(BaseModel):
    """No arguments: verify_repository always runs the repository's
    configured verification command (config.verification_command) as-is."""


def build_delegation_tools(service: RepoService, config: AgentConfig, explorer_llm=None, coder_llm=None,
                            debugger_llm=None, investigated_hint: Callable[[], str] | None = None
                            ) -> list[StructuredTool]:
    """Build the repository investigation and change delegation tools.

    Each sub-agent graph is created once, while every tool invocation starts a
    fresh run and returns only its final report to the Main Agent - the
    Explorer's / Coder's / Debugger's internal tool history never joins the
    Main Agent's own message history.

    `code_repository` runs a bounded verification and repair loop after the
    Coder finishes. Quick verification is scoped to changed Python files;
    `verify_repository` runs the configured repository-wide command when
    explicitly requested. Checkpointing and rollback bound repository
    changes. The shared service refreshes after changes and verification
    because commands can create, delete, or rename files.

    `investigated_hint`, if given, is called with no arguments each time
    `investigate_repository` runs and should return a short summary (or "")
    of what the current caller's run has already investigated, built from
    its own existing evidence/plan state - this module creates no tracking
    of its own. The summary is given to the freshly-started Explorer as
    context, not as an instruction: it is informational only, so the
    Explorer remains free to re-read anything it judges necessary.
    """
    explorer = build_explorer(service, config=config, llm=explorer_llm)
    coder = build_coder(service, config=config, llm=coder_llm)
    debugger = build_debugger(service, config=config, llm=debugger_llm)

    def investigate_repository(query: str) -> str:
        messages = []
        hint = investigated_hint() if investigated_hint is not None else ""
        if hint:
            # A separate, clearly-labelled human message ahead of the actual
            # query - same pattern as the Deep Agent's own retrieval_hint
            # (see deep_agent.py's decide()). Context for the Explorer's own
            # judgment, not a tool restriction: it must not stop the
            # Explorer from reading a file again when verifying something
            # specific actually requires it.
            messages.append({"role": "user", "content": hint})
        messages.append({"role": "user", "content": query})
        result = explorer.invoke({"messages": messages}) #type: ignore
        return explorer_report(result["messages"], max_report_chars=min(config.max_observation_chars, 12000))

    def verify_repository() -> str:
        # The broader check code_repository's automatic verification no
        # longer runs by default (see build_delegation_tools docstring):
        # the repository's actual configured command, exactly as
        # run_verification has always run it for the pre-5C repair loop.
        verification = run_verification(service, config)
        state = terminal_state(verification, repair_limit_reached=False)
        # Verification commands are arbitrary and may mutate the checkout.
        # Refresh even when they fail or time out so subsequent reads reflect disk.
        try:
            refresh = service.refresh()
        except (OSError, ValueError) as exc:
            refresh = {"error": str(exc)}
            state = "UNAVAILABLE"
        report = _bound_report(f"{verification_summary(verification)}\n\nResult: {state}.",
                               min(config.max_observation_chars, 12000))
        if "error" in refresh:
            report += f"\n\nRepository index refresh failed: {refresh['error']}"
        return json.dumps({"state": state, "verification": verification,
                           "report": report,
                           "refresh": refresh})

    def code_repository(task: str) -> str:
        # Snapshot the repository before the Coder touches anything.
        # Anything already different from a clean checkout (pre-existing user
        # edits) is baked into this baseline and is therefore never rolled
        # back - only what changes *after* this point can be undone. The
        # checkpoint's temp-directory snapshot is always freed in `finally`,
        # even if the Coder/Debugger loop below raises - it is a real bug in
        # that case (per tools.py's error-handling convention) and should
        # still propagate, it just should not also leak a temp directory.
        # Even when automatic rollback is disabled, the baseline is needed
        # to discover mutations made through Debugger.run_command (which are
        # invisible to write_file tool history) and scope verification.
        checkpoint = Checkpoint.create(service, enabled=True, label=task)
        preserve_checkpoint = False
        try:
            if not checkpoint.available:
                error = checkpoint.error or {}
                return (
                    "Autonomous repository modification was not started because the required "
                    f"checkpoint could not be created ({error.get('code', 'checkpoint_creation_failed')}): "
                    f"{error.get('message', 'unknown checkpoint error')}\n\nResult: FAILED."
                )
            repair_result = _run_repair_loop(task, checkpoint)
            report = repair_result["report"]
            state = repair_result["state"]
            # code_repository is the one place that can change the
            # repository (via the Coder/Debugger, possibly through several
            # write_file calls and/or a rollback above) - refresh the shared
            # RepoService's index/graph/retriever now, once, so every tool
            # call after this delegation returns (further Explorer/Coder/
            # Debugger runs, or the Main/Deep Agent's own read tools) sees
            # the repository as it actually is on disk, not a stale snapshot
            # from before this task ran. See RepoService.refresh().
            try:
                refresh_summary = service.refresh()
            except (OSError, ValueError) as exc:
                refresh_summary = {"error": str(exc)}
            if "error" in refresh_summary:
                state = "UNAVAILABLE"
                report += f"\n\nRepository index refresh failed: {refresh_summary['error']}"
            return json.dumps({"state": state, "verification": repair_result["verification"],
                               "report": _bound_report(report, min(config.max_observation_chars, 12000)),
                               "refresh": refresh_summary})
        except Exception as exc:
            rollback = None
            refresh_summary = None
            rollback_error = None
            if checkpoint.available:
                try:
                    rollback = checkpoint.rollback()
                    if rollback is None or not rollback.get("rolled_back"):
                        rollback_error = (rollback or {}).get("errors") or rollback or "rollback returned no status"
                except Exception as rollback_exc:
                    rollback_error = f"{type(rollback_exc).__name__}: {rollback_exc}"
            try:
                refresh_summary = service.refresh()
                if "error" in refresh_summary and rollback_error is None:
                    rollback_error = f"index refresh failed: {refresh_summary['error']}"
            except Exception as refresh_exc:
                if rollback_error is None:
                    rollback_error = f"index refresh failed: {type(refresh_exc).__name__}: {refresh_exc}"
            preserve_checkpoint = rollback_error is not None
            report = (f"Repository repair stopped after an unexpected {type(exc).__name__}: {exc}. "
                      "Rollback was attempted."
                      if rollback_error is None else
                      f"Repository repair stopped after an unexpected {type(exc).__name__}: {exc}. "
                      f"Rollback or index refresh also failed: {rollback_error}")
            return json.dumps({"state": "FAILED", "verification": {"status": "execution_error"},
                               "report": f"{report}\n\nResult: FAILED.",
                               "rollback": rollback, "refresh": refresh_summary,
                               "checkpoint_id": checkpoint.checkpoint_id if preserve_checkpoint else None,
                               "error": f"{type(exc).__name__}: {exc}"})
        finally:
            if not preserve_checkpoint:
                checkpoint.discard()

    def _run_repair_loop(task: str, checkpoint: Checkpoint) -> dict:
        print(f"Starting repair loop for task '{task}' (max {config.max_repair_attempts} attempts).")
        result = coder.invoke({"messages": [{"role": "user", "content": task}]}) # type: ignore
        report = coder_report(result["messages"])
        # Scoped, not the full suite: verify only the .py files this run
        # actually wrote (see utils.written_files), so a small change is verified
        # quickly instead of re-running everything - see the
        # build_delegation_tools docstring above. Every path write_file
        # touched this run is kept (not just the latest call), since a
        # later Debugger fix can still depend on an earlier write.
        changed_files: list[str] = written_files(result["messages"])
        changed_files += _checkpoint_changed_files(checkpoint)
        verification = run_scoped_verification(service, config, changed_files)

        if verification["passed"]:
            state = terminal_state(verification, repair_limit_reached=False)
            summary = format_change_summary(checkpoint.diff())
            return {"state": state, "verification": verification,
                    "report": f"{report}\n\n{verification_summary(verification)}\n\n{summary}\n\nResult: {state}."}

        attempts = 0
        last_signature = verification_signature(verification)
        progress_note = ""
        while attempts < config.max_repair_attempts:
            attempts += 1
            debug_task = DEBUGGER_TASK_TEMPLATE.format(
                task=task, coder_report=report, verification=verification_summary(verification),
            )
            debug_result = debugger.invoke({"messages": [{"role": "user", "content": debug_task}]}) # pyright: ignore[reportArgumentType]
            report = debugger_report(debug_result["messages"])
            # Accumulate: the Debugger may have written additional files
            # this attempt, on top of whatever the Coder (or an earlier
            # Debugger attempt) already wrote.
            changed_files += written_files(debug_result["messages"])
            changed_files += _checkpoint_changed_files(checkpoint)
            verification = run_scoped_verification(service, config, changed_files)
            signature = verification_signature(verification)
            if not verification["passed"] and signature == last_signature:
                # Same status/exit-code/stderr as last time - the
                # repair loop is not making objective progress. This is
                # reported, not used to stop early: the repair budget
                # (max_repair_attempts) is the one thing that bounds the
                # loop, so the configured repair-limit behavior is
                # preserved exactly.
                progress_note = (
                    "\n\nNote: this repair attempt produced the same verification failure as the "
                    "previous attempt - the repair does not appear to be making progress."
                )
            last_signature = signature

            if verification["passed"]:
                state = terminal_state(verification, repair_limit_reached=False)
                summary = format_change_summary(checkpoint.diff())
                return {"state": state, "verification": verification,
                        "report": (f"{report}\n\n{verification_summary(verification)}\n\n"
                        f"(Verification passed after {attempts} Debugger repair attempt(s).)\n\n"
                        f"{summary}\n\nResult: {state}.")}

        # Repair budget exhausted and verification is still not passing.
        # A single failing attempt never triggers a
        # rollback (that is the normal, expected intermediate state the
        # repair loop exists to fix) - only exhausting the whole repair
        # budget while still failing does, and only when enabled.
        state = terminal_state(verification, repair_limit_reached=True)
        rollback_note = ""
        if not config.enable_checkpoints:
            rollback_note = (
                "\n\nAutomatic rollback is disabled, so changes made are left in place."
            )
        elif config.rollback_on_repair_limit:
            rollback = checkpoint.rollback()
            if rollback is not None and rollback.get("rolled_back"):
                rollback_note = (
                    "\n\nThe repository has been rolled back to its state before this task started "
                    f"(restored {len(rollback['restored_files'])} file(s), "
                    f"removed {len(rollback['removed_files'])} file(s))."
                )
            elif rollback is not None:
                rollback_note = (
                    "\n\nRollback was attempted but did not complete cleanly for every file; "
                    "the repository may be left partially modified. See the errors for details: "
                    f"{rollback.get('errors')}"
                )
        else:
            summary = format_change_summary(checkpoint.diff())
            rollback_note = f"\n\n{summary} (left in place: rollback-on-repair-limit is disabled.)"

        return {"state": state, "verification": verification,
                "report": (f"{report}\n\n{verification_summary(verification)}\n\n"
                f"Verification is still failing after {attempts} Debugger repair attempt(s) "
                "(repair limit reached). Returning to the Main Agent."
                f"{progress_note}{rollback_note}\n\nResult: {state}.")}

    def _refresh_note(summary: dict) -> str:
        """One-line, human-readable summary of RepoService.refresh()'s
        result, appended to code_repository's report. Deliberately after the
        final 'Result: <state>.' line (see terminal_state/_succeeded in
        deep_agent.py, which look for that substring anywhere in the report,
        not only at its end) - it is repository-index bookkeeping, not part
        of the task outcome itself."""
        added, removed = summary["files_added"], summary["files_removed"]
        if not added and not removed:
            return f"Repository index refreshed (no file list changes; {summary['symbol_count']} symbols indexed)."
        parts = []
        if added:
            parts.append(f"{len(added)} file(s) newly indexed ({', '.join(added[:5])}{'...' if len(added) > 5 else ''})")
        if removed:
            parts.append(f"{len(removed)} file(s) no longer present ({', '.join(removed[:5])}{'...' if len(removed) > 5 else ''})")
        return f"Repository index refreshed: {'; '.join(parts)}. ({summary['symbol_count']} symbols indexed.)"

    def _bound_report(report: str, limit: int = 12000) -> str:
        text = str(report or "")
        return text if len(text) <= limit else text[:limit - 80] + " ...[report truncated]"


    def _checkpoint_changed_files(checkpoint: Checkpoint) -> list[str]:
        diff = checkpoint.diff()
        if not isinstance(diff, dict) or diff.get("changed") is None:
            return []
        return list(diff.get("added", [])) + list(diff.get("modified", []))

    return [
        StructuredTool.from_function(
            func=investigate_repository,
            name="investigate_repository",
            description=(
                "Delegate a focused repository investigation to the read-only Explorer sub-agent, which can "
                "search code, find files by path, read files and inspect symbols/relationships across several "
                "tool calls, returning a concise evidence-based report."
            ),
            args_schema=InvestigateRepositoryInput,
        ),
        StructuredTool.from_function(
            func=code_repository,
            name="code_repository",
            description=(
                "Delegate an implementation task to the Coder sub-agent, which inspects the repository and then "
                "creates or modifies files (write_file) across several tool calls. This is the only way to change "
                "the repository - use it when the user asks you to add, fix or edit code. The change is "
                "automatically verified with a quick syntax check of just the files that changed (NOT the "
                "repository's full test suite); on failure a bounded Debugger repair loop tries to fix it and "
                "re-verifies, up to a repair-attempt limit. Use investigate_repository instead for read-only "
                "questions, and verify_repository afterwards to confirm the full test suite (or other configured "
                "verification command) still passes. Returns a concise report of what changed and whether the "
                "scoped verification passed."
            ),
            args_schema=CodeRepositoryInput,
        ),
        StructuredTool.from_function(
            func=verify_repository,
            name="verify_repository",
            description=(
                "Run the repository's full configured verification command (e.g. its test suite), with no "
                "arguments. code_repository only runs a quick syntax check of the files it just changed - use "
                "verify_repository for broader confidence, e.g. after several code_repository calls, or when the "
                "task explicitly asks you to confirm the tests still pass. Returns a concise report of the result."
            ),
            args_schema=VerifyRepositoryInput,
        ),
    ]
