"""The verification step of the Coder -> Verification -> Debugger
repair loop (see delegation.py).

Verification is deliberately NOT an LLM call: it deterministically runs a
configured command (RepoService.run_command, always inside the repository
root) and reports pass/fail from its exit code. This mirrors the existing
step-count logic in explorer.py/coder.py/debugger.py, which is also plain
Python control flow rather than something the LLM decides.
"""

from __future__ import annotations

import sys

from ..service import RepoService
from .config import AgentConfig


def run_verification(service: RepoService, config: AgentConfig) -> dict:
    """Run config.verification_command and classify the outcome.

    Returns a dict with the raw run_command() result (command, exit_code,
    stdout, stderr, timed_out - or `error` if the command could not even be
    started) plus:
      - `status`: "passed" | "no_tests" | "failed" | "timeout" | "unavailable"
      - `passed`: True when nothing needs to be sent to the Debugger repair
        loop - not only a literal exit-0 pass, but also "no_tests" (there is
        nothing to verify) and "unavailable" (the verification setup itself
        is broken, e.g. a missing executable). Neither of those is evidence
        that the Coder's/Debugger's *code change* is wrong, so the repair
        loop must not burn an attempt chasing a bug that doesn't exist.
    """
    if not config.verification_command.strip():
        return {"passed": True, "status": "not_configured", "command": "(no command configured)",
                "exit_code": None, "timed_out": False, "stdout": "", "stderr": ""}

    result = service.run_command(config.verification_command, timeout=config.command_timeout_seconds)

    if "error" in result:
        return {"passed": True, "status": "unavailable", **result}
    if result["timed_out"]:
        return {"passed": False, "status": "timeout", **result}
    if result["exit_code"] == 0:
        return {"passed": True, "status": "passed", **result}
    return {"passed": False, "status": "failed", **result}


def run_scoped_verification(service: RepoService, config: AgentConfig, changed_files: list[str]) -> dict:
    """Lightweight, change-scoped safety check for `code_repository` (see
    delegation.py): confirm the Python files the Coder/Debugger actually
    touched this run still parse, without running the repository's full
    `config.verification_command` for every small edit. `verify_repository` remains the way to
    run that full command explicitly when broader confidence is wanted.

    `py_compile` only checks syntax, not whether the change is *correct* -
    that is a deliberate, minimal safety net, not a replacement for the full
    verification command. Returns the same result shape as run_verification
    (status/passed/command/exit_code/stdout/stderr) so terminal_state(),
    verification_summary() and verification_signature() all keep working
    unchanged for either.
    """
    py_files = sorted({f for f in changed_files if f.casefold().endswith(".py")})
    if not py_files:
        # Nothing Python-shaped changed (e.g. only a README was touched) -
        # there is nothing for a syntax check to catch, so this is a pass,
        # not a skip: there are no changed Python files to validate.
        return {
            "passed": True, "status": "no_changes",
            "command": "(no changed Python files to check)",
            "exit_code": None, "timed_out": False, "stdout": "", "stderr": "",
        }

    command = [sys.executable, "-m", "py_compile", *py_files]
    result = service.run_command(command, timeout=config.command_timeout_seconds)

    if "error" in result:
        return {"passed": True, "status": "unavailable", **result}
    if result["timed_out"]:
        return {"passed": False, "status": "timeout", **result}
    if result["exit_code"] == 0:
        return {"passed": True, "status": "passed", **result}
    return {"passed": False, "status": "failed", **result}


_STATUS_LABELS = {
    "passed": "PASSED",
    "not_configured": "NOT CONFIGURED (no repository command was supplied)",
    "no_changes": "PASSED (no changed Python files needed a syntax check)",
    "unavailable": "SKIPPED (the verification command itself could not run)",
    "timeout": "FAILED (the command timed out)",
    "failed": "FAILED",
}


def verification_summary(result: dict) -> str:
    """A short, human-readable summary of a run_verification() result, used in
    reports returned to the Main Agent and in the Debugger's task prompt."""
    if "error" in result:
        error = result["error"]
        return f"Verification could not run ({error.get('code')}): {error.get('message')}"

    label = _STATUS_LABELS.get(result.get("status"), "PASSED" if result["passed"] else "FAILED")
    lines = [f"Verification command {result['command']!r} -> {label} (exit code {result.get('exit_code')})"]
    if result.get("timed_out"):
        lines.append("The command timed out.")
    if result.get("stdout"):
        lines.append(f"stdout:\n{result['stdout']}")
    if result.get("stderr"):
        lines.append(f"stderr:\n{result['stderr']}")
    return "\n".join(lines)
