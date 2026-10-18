"""Deterministic verification helpers for repository changes and test execution.

Verification is plain Python control flow (not an LLM call): it deterministically
validates and runs a configured command or scoped syntax check (`RepoService.run_command`,
always inside the repository root) and classifies pass/fail from its exit code.
"""

from __future__ import annotations

from pathlib import Path
import sys

from ..service import RepoService
from .config import AgentConfig


def run_verification(
    service: RepoService,
    config: AgentConfig,
    *,
    write_allowed: bool = False,
) -> dict:
    """Validate and run `config.verification_command` and classify the outcome.

    Returns a dict with the raw `run_command()` result (`command`, `exit_code`,
    `stdout`, `stderr`, `timed_out` - or `error` if the command was blocked by
    safety policy or could not be started) plus:
      - `status`: "passed" | "not_configured" | "failed" | "timeout" | "unavailable" | "blocked"
      - `passed`: True when nothing needs repair - not only a literal exit-0
        pass, but also "not_configured" (there is no configured command to run)
        and "unavailable" (the verification setup itself is broken, e.g. a
        missing executable). A command blocked by `validate_command_safety` is
        reported with `passed=False, status="blocked"` and its `error` dict.
    """
    cmd = config.verification_command.strip()
    if not cmd:
        return {
            "passed": True,
            "status": "not_configured",
            "command": "(no command configured)",
            "exit_code": None,
            "timed_out": False,
            "stdout": "",
            "stderr": "",
        }

    from .safety import validate_command_safety

    repo_root = Path(service.index.root).resolve() if getattr(service, "index", None) else None
    safety_err = validate_command_safety(
        cmd, write_allowed=write_allowed, repo_root=repo_root
    )
    if safety_err is not None:
        return {
            "passed": False,
            "status": "blocked",
            "command": cmd,
            "exit_code": None,
            "timed_out": False,
            "stdout": "",
            "stderr": safety_err["error"].get("message", ""),
            **safety_err,
        }

    result = service.run_command(cmd, timeout=config.command_timeout_seconds)

    if "error" in result:
        return {"passed": True, "status": "unavailable", **result}
    if result["timed_out"]:
        return {"passed": False, "status": "timeout", **result}
    if result["exit_code"] == 0:
        return {"passed": True, "status": "passed", **result}
    return {"passed": False, "status": "failed", **result}


def run_scoped_verification(
    service: RepoService, config: AgentConfig, changed_files: list[str]
) -> dict:
    """Lightweight, change-scoped safety check for `write_file`: confirm the
    Python files touched in the current transaction still parse, without running
    the repository's full `config.verification_command` for every edit.
    `verify_repository` remains the way to run that full command explicitly.

    `py_compile` checks syntax only and runs with `-I` (isolated mode) so
    repository files such as `py_compile.py` or `sitecustomize.py` in `cwd` are
    never imported during syntax verification. Returns the same result shape as
    `run_verification` (`status`/`passed`/`command`/`exit_code`/`stdout`/`stderr`)
    so `terminal_state()`, `verification_summary()`, and `verification_signature()`
    work unchanged for either.
    """
    root = Path(service.index.root).resolve() if getattr(service, "index", None) else None
    py_files = sorted(
        {
            f
            for f in changed_files
            if isinstance(f, str)
            and f.casefold().endswith(".py")
            and (root is None or (root / f).is_file())
        }
    )
    if not py_files:
        return {
            "passed": True,
            "status": "no_changes",
            "command": "(no changed Python files to check)",
            "exit_code": None,
            "timed_out": False,
            "stdout": "",
            "stderr": "",
        }

    command = [sys.executable, "-I", "-m", "py_compile", *py_files]
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
    "blocked": "BLOCKED BY COMMAND SAFETY POLICY",
    "timeout": "FAILED (the command timed out)",
    "failed": "FAILED",
}


def verification_summary(result: dict) -> str:
    """A short, human-readable summary of a `run_verification()` or
    `run_scoped_verification()` result."""
    if "error" in result:
        error = result["error"]
        return f"Verification could not run ({error.get('code')}): {error.get('message')}"

    label = _STATUS_LABELS.get(
        result.get("status"), "PASSED" if result.get("passed") else "FAILED"
    )
    lines = [
        f"Verification command {result.get('command')!r} -> {label} (exit code {result.get('exit_code')})"
    ]
    if result.get("timed_out"):
        lines.append("The command timed out.")
    if result.get("stdout"):
        lines.append(f"stdout:\n{result['stdout']}")
    if result.get("stderr"):
        lines.append(f"stderr:\n{result['stderr']}")
    return "\n".join(lines)
