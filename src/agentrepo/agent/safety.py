"""Filesystem checkpointing, command safety validation, repair bounds, and rollback transactions."""

from __future__ import annotations

import contextvars
import difflib
import os
from pathlib import Path
import posixpath
import re
import shlex
import shutil
import sys
from typing import Any, Sequence
import uuid

from ..filesystem import is_within
from ..service import RepoService, _split_command, set_current_thread_tx_id
from .config import AgentConfig
from .verification import (
    run_scoped_verification,
    run_verification,
    verification_summary,
)

# Explicit terminal states for a modification or verification transaction.
SUCCESS = "SUCCESS"
NO_TESTS = "NO_TESTS"
UNAVAILABLE = "UNAVAILABLE"
FAILED = "FAILED"
TIMEOUT = "TIMEOUT"
REPAIR_LIMIT_REACHED = "REPAIR_LIMIT_REACHED"

_STATUS_TO_STATE = {
    "passed": SUCCESS,
    "no_tests": NO_TESTS,
    "no_changes": NO_TESTS,
    "not_configured": UNAVAILABLE,
    "unavailable": UNAVAILABLE,
    "blocked": FAILED,
    "checkpoint_disabled": FAILED,
    "checkpoint_failed": FAILED,
    "unauthorized_mutation": FAILED,
    "timeout": TIMEOUT,
    "failed": FAILED,
}

MAX_COMMAND_CHARS = 4096
MAX_COMMAND_ARGS = 128

# Shell wrappers, meta-runners, non-Python interpreters, and destructive/external
# binaries that run_command and verify_repository must never execute.
_SHELL_BINARIES = frozenset({
    "sh", "bash", "zsh", "dash", "ksh", "csh", "tcsh", "fish", "busybox",
    "ash", "elvish", "nu", "nushell", "xonsh", "command",
    "cmd", "cmd.exe", "powershell", "powershell.exe", "pwsh", "pwsh.exe",
    "wsl", "wsl.exe",
})
_WRAPPER_BINARIES = frozenset({
    "env", "xargs", "nohup", "timeout", "nice", "ionice", "stdbuf", "setsid",
    "chroot", "sudo", "su", "doas", "runuser", "pkexec", "strace", "ltrace",
    "gdb", "lldb", "watch", "script", "eval", "exec", "time", "parallel",
    "entr", "expect", "unshare", "nsenter", "firejail", "bwrap", "docker",
    "podman", "kubectl", "caffeinate", "start", "open", "xdg-open", "cygstart",
    "rundll32", "regsvr32", "mshta", "cscript", "wscript", "osascript",
    "npx", "pnpx", "bunx", "uv", "uvx", "pipx", "conda", "poetry", "pipenv",
    "hatch", "pdm", "tox", "nox",
})
_ALWAYS_FORBIDDEN_BINARIES = frozenset({
    "rm", "rmdir", "del", "erase", "dd", "shred", "truncate", "mkfs", "fdisk",
    "mount", "umount", "chmod", "chown", "chgrp", "chattr", "setfacl", "icacls",
    "attrib", "ln", "mktemp", "mknod", "mkfifo",
    "curl", "wget", "nc", "netcat", "ncat", "socat", "ssh", "scp", "sftp",
    "rsync", "ftp", "tftp", "telnet", "ping", "dig", "nslookup",
    "kill", "killall", "pkill", "taskkill", "shutdown", "reboot",
    "crontab", "at", "systemctl", "service", "launchctl", "reg",
})
_FORBIDDEN_INTERPRETERS_AND_MUTATORS = frozenset({
    "node", "nodejs", "deno", "bun", "perl", "ruby", "php", "lua",
    "tclsh", "wish", "rscript", "r", "julia", "groovy", "scala", "java",
    "javaw", "jshell",
    "awk", "gawk", "mawk", "nawk", "sed", "ed", "ex", "vi", "vim", "nvim",
    "nano", "emacs",
    "mv", "move", "cp", "copy", "touch", "mkdir", "patch", "install", "tee",
    "pip", "pip3", "npm", "yarn", "pnpm", "cargo", "go", "make", "cmake",
    "ninja",
    "tar", "unzip", "zip", "gunzip", "gzip", "bzip2", "xz", "7z", "cpio", "ar",
})
_READONLY_INSPECTION_UTILITIES = frozenset({
    "find", "grep", "rg", "ls", "cat", "head", "tail", "wc", "diff", "stat", "file",
})
_DIRECT_TEST_AND_LINT_BINARIES = frozenset({
    "pytest", "py.test", "ruff", "flake8", "mypy",
})
_ALLOWED_DIRECT_EXECUTABLES = (
    frozenset({"git"})
    | _READONLY_INSPECTION_UTILITIES
    | _DIRECT_TEST_AND_LINT_BINARIES
)
_ALLOWED_PYTHON_MODULES = frozenset({
    "pytest", "unittest", "py_compile", "mypy", "flake8", "ruff",
})
_ALLOWED_READONLY_GIT_SUBCOMMANDS = frozenset({
    "status", "diff", "log", "show", "rev-parse", "ls-files", "blame",
    "shortlog", "describe",
})
_FORBIDDEN_GIT_SUBCOMMANDS = frozenset({
    "reset", "clean", "checkout", "restore", "revert", "commit", "push",
    "pull", "fetch", "merge", "rebase", "cherry-pick", "am", "apply",
    "branch", "tag", "stash", "rm", "mv", "init", "clone", "config",
    "submodule", "worktree", "gc", "prune", "reflog", "update-ref",
    "symbolic-ref", "filter-branch", "replace", "notes", "bisect",
    "archive", "bundle", "daemon", "fast-import", "send-pack", "receive-pack",
    "http-fetch", "http-push", "switch", "format-patch",
})
_SAFE_GIT_GLOBAL_BOOL_FLAGS = frozenset({
    "--no-pager", "--no-replace-objects",
    "--literal-pathspecs", "--glob-pathspecs", "--noglob-pathspecs",
    "--icase-pathspecs", "--no-optional-locks", "--version", "--help",
})
_FORBIDDEN_GIT_GLOBAL_PREFIXES = (
    "-c", "-p", "--paginate", "--config-env", "--exec-path", "--git-dir",
    "--work-tree", "--namespace", "--super-prefix", "--bare",
    "--ita-invisible-in-index",
)
_FORBIDDEN_GIT_SUBCOMMAND_FLAGS = (
    "--output", "--ext-diff", "--textconv", "--upload-pack", "--receive-pack",
    "--exec", "--git-dir", "--work-tree", "--no-index",
)
_FORBIDDEN_FIND_FLAGS = frozenset({
    "-delete", "-exec", "-execdir", "-ok", "-okdir", "-fprint", "-fprint0",
    "-fls", "-fprintf", "-l", "-follow", "-h",
})
_SHELL_CONTROL_TOKENS = frozenset({
    "|", "||", "&&", ";", ">", ">>", "<", "<<", "2>", "2>>", "&>", "&>>", ">&", "1>", "1>>",
})
_PATH_TAKING_OPTIONS = frozenset({
    "-c", "-C", "-f", "-o", "-s", "-t",
    "--rootdir", "--confcutdir", "--basetemp", "--ignore", "--ignore-glob",
    "--deselect", "--config", "--config-file", "--settings", "--output",
    "--output-file", "--cache-dir", "--start-directory", "--top-level-directory",
    "--file", "--log-file", "--shadow-file", "--write-baseline", "--junit-xml",
    "--junitxml",
})
_DISALLOWED_SCRIPT_SUFFIXES = (
    ".sh", ".bat", ".cmd", ".ps1", ".com", ".py", ".pl", ".rb", ".js", ".php", ".vbs", ".wsf",
)
_PYTHON_BIN_RE = re.compile(r"^(?:python(?:\d+(?:\.\d+)*)?|py|pypy\d*)$", re.IGNORECASE)

# Diagnostic patterns for inline Python code (note: inline `python -c` is
# unconditionally blocked before process launch; these patterns provide
# specific diagnostic detail when inline code contains system or file-mutation
# operations).
_INLINE_PYTHON_ALWAYS_FORBIDDEN_RE = re.compile(
    r"(?:\b(?:os\.system|subprocess|Popen|shutil\.rmtree|shutil\.move"
    r"|os\.remove|os\.unlink|os\.rmdir|os\.removedirs|os\.replace|os\.rename|os\.renames"
    r"|os\.chmod|os\.chown|os\.symlink|os\.link|os\.truncate|os\.spawn\w*|os\.exec\w*"
    r"|__import__|importlib|builtins|__builtins__|getattr|setattr"
    r"|ctypes|socket|urllib|http\.client|requests|multiprocessing)\b"
    r"|\b(?:eval|exec|compile)\s*\()",
    re.IGNORECASE,
)
_INLINE_PYTHON_READONLY_MUTATION_RE = re.compile(
    r"(?:\b(?:open|os\.open|io\.open|codecs\.open|writelines)\s*\("
    r"|\.(?:write|writelines|write_text|write_bytes|open|unlink|rmdir|mkdir|replace|rename|touch)\s*\("
    r"|\b(?:write_text|write_bytes|unlink|remove|removedirs|rmdir|rmtree|replace|rename|renames"
    r"|makedirs|mkdir|copy|copy2|copyfile|copytree|move|chown|chmod|touch"
    r"|shutil|tempfile|pathlib)\b)",
    re.IGNORECASE,
)
_FORBIDDEN_PYTHON_MODULES_ALWAYS = frozenset({
    "http.server", "smtpd", "webbrowser", "pip", "ensurepip", "venv",
    "zipfile", "tarfile", "pdb", "idlelib", "turtledemo", "code", "timeit",
    "runpy", "cprofile", "profile", "trace", "doctest", "pydoc", "asyncio",
    "compileall", "site", "distutils", "setuptools", "wheel", "ctypes",
})


def _strip_exe_suffix(exe_base: str) -> str:
    lowered = exe_base.casefold()
    for suffix in (".exe", ".cmd", ".bat", ".ps1", ".sh", ".com", ".py", ".pl", ".rb", ".js", ".php"):
        if lowered.endswith(suffix) and len(lowered) > len(suffix):
            return lowered[: -len(suffix)]
    return lowered


def _is_exact_sys_executable(exe_raw: str, repo_root: Path | None = None) -> bool:
    """Return True only when `exe_raw` is the exact absolute path of the current
    Python interpreter (`sys.executable`) and does not reside inside `repo_root`."""
    cleaned = exe_raw.strip().strip("\"'")
    if not cleaned:
        return False
    try:
        cand_path = Path(cleaned)
        if not cand_path.is_absolute():
            return False
        sys_exe_abs = os.path.abspath(sys.executable)
        exe_abs = os.path.abspath(cleaned)
        if os.name == "nt":
            if exe_abs.casefold() != sys_exe_abs.casefold():
                return False
        else:
            if exe_abs != sys_exe_abs:
                return False
        sys_resolved = Path(sys.executable).resolve()
        exe_resolved = cand_path.resolve()
        if os.name == "nt":
            if str(exe_resolved).casefold() != str(sys_resolved).casefold():
                return False
        else:
            if exe_resolved != sys_resolved:
                return False
        if repo_root is not None:
            root_resolved = repo_root.resolve()
            if is_within(root_resolved, Path(exe_abs)) or is_within(root_resolved, exe_resolved):
                return False
        return True
    except (OSError, RuntimeError, ValueError):
        return False


def _is_python_executable(exe_stem: str, exe_raw: str, repo_root: Path | None = None) -> bool:
    if _PYTHON_BIN_RE.match(exe_stem):
        return True
    return _is_exact_sys_executable(exe_raw, repo_root=repo_root)


def _has_disallowed_executable_suffix(exe_base: str, *, is_current_sys_exe: bool = False) -> bool:
    """Reject script/batch suffixes (`.sh`, `.bat`, `.cmd`, `.ps1`, `.py`, etc.)
    and `.exe` on non-Windows platforms so spoofed wrappers cannot impersonate
    allowlisted binaries."""
    lowered = exe_base.casefold()
    if lowered.endswith(_DISALLOWED_SCRIPT_SUFFIXES):
        return True
    if os.name != "nt" and lowered.endswith(".exe") and not is_current_sys_exe:
        return True
    return False


def _verify_executable_identity(
    exe_raw: str,
    exe_base: str,
    resolved_root: Path | None,
) -> dict | None:
    """Verify that `exe_raw` is either a bare allowlisted executable name that
    does not resolve inside `resolved_root` or the exact current `sys.executable`
    interpreter outside `resolved_root`.

    Blocks:
      - relative executable paths (`./python`, `./pytest`, `bin/pytest`, `../pytest`)
      - absolute executable paths other than `sys.executable` (`/tmp/pytest`,
        `/usr/bin/pytest`, `<repo_root>/python`)
      - script/batch extensions (`pytest.sh`, `git.bat`, `python.cmd`)
      - bare binary names shadowed by a file inside `resolved_root` via `PATH`
    """
    cleaned = exe_raw.strip().strip("\"'")
    exe_norm = cleaned.replace("\\", "/")
    has_path_qualifier = (
        "/" in exe_norm
        or "\\" in cleaned
        or ":" in exe_norm
        or exe_norm.startswith((".", "~"))
    )
    is_current_sys_exe = _is_exact_sys_executable(cleaned, repo_root=resolved_root)

    if has_path_qualifier:
        if not is_current_sys_exe:
            return {
                "error": {
                    "code": "command_not_permitted",
                    "message": (
                        f"path-qualified executable {exe_raw!r} is not permitted; "
                        "use a bare allowlisted command name (resolved outside the repository) "
                        "or the current Python interpreter (sys.executable)"
                    ),
                }
            }

    if _has_disallowed_executable_suffix(exe_base, is_current_sys_exe=is_current_sys_exe):
        return {
            "error": {
                "code": "command_not_permitted",
                "message": (
                    f"executable suffix on {exe_raw!r} is not permitted; script/batch wrappers "
                    "cannot impersonate allowlisted executables"
                ),
            }
        }

    # Check whether a bare executable name would resolve inside `resolved_root`
    # via the current environment's PATH without having a safe external binary.
    if not has_path_qualifier and resolved_root is not None:
        raw_path_env = os.environ.get("PATH", "")
        raw_resolved = shutil.which(cleaned, path=raw_path_env)
        shadowed_candidate = _repository_path_candidate(
            cleaned, raw_path_env, resolved_root
        )
        if raw_resolved is not None or shadowed_candidate is not None:
            try:
                cand_abs = (
                    Path(os.path.abspath(raw_resolved))
                    if raw_resolved is not None
                    else shadowed_candidate
                )
                cand_real = cand_abs.resolve()
                if is_within(resolved_root, cand_abs) or is_within(resolved_root, cand_real):
                    sanitized_path = RepoService._sanitized_search_path(resolved_root)
                    # `shutil.which(..., path="")` falls back to the current
                    # process PATH. An empty sanitized PATH must mean that no
                    # safe external executable was found.
                    safe_resolved = (
                        shutil.which(cleaned, path=sanitized_path)
                        if sanitized_path
                        else None
                    )
                    if safe_resolved is None:
                        return {
                            "error": {
                                "code": "command_not_permitted",
                                "message": (
                                    f"executable {exe_raw!r} resolves inside the target repository "
                                    f"({cand_abs}) and cannot be trusted"
                                ),
                            }
                        }
            except (OSError, RuntimeError, ValueError):
                return {
                    "error": {
                        "code": "command_not_permitted",
                        "message": f"could not verify executable identity for {exe_raw!r}",
                    }
                }
    return None


def _repository_path_candidate(
    executable: str,
    raw_path: str,
    repo_root: Path,
) -> Path | None:
    """Find an executable-shaped file in a repository PATH entry.

    On Windows, ``shutil.which`` only recognizes extensions from PATHEXT. A
    bare extensionless file can still represent a shadowing candidate in a
    mixed or mocked environment, so check the repository PATH entries
    directly as well.
    """
    names = [executable]
    if os.name == "nt" and not Path(executable).suffix:
        names.extend(
            executable + suffix.lower()
            for suffix in os.environ.get("PATHEXT", ".COM;.EXE;.BAT;.CMD").split(";")
            if suffix
        )
    try:
        canonical_root = repo_root.resolve()
    except OSError:
        canonical_root = Path(os.path.abspath(repo_root))
    for entry in raw_path.split(os.pathsep):
        cleaned_entry = entry.strip().strip("\"'")
        entry_path = Path(cleaned_entry) if cleaned_entry else Path.cwd()
        if not entry_path.is_absolute():
            entry_path = Path.cwd() / entry_path
        try:
            directory = Path(os.path.abspath(entry_path))
            resolved_directory = directory.resolve()
            if not (
                is_within(canonical_root, directory)
                or is_within(canonical_root, resolved_directory)
            ):
                continue
            for name in names:
                candidate = directory / name
                if candidate.is_file():
                    return candidate
        except (OSError, RuntimeError, ValueError):
            continue
    return None


def _is_unrooted_escape_path(raw_path: str) -> bool:
    cleaned = raw_path.strip().strip("\"'")
    if not cleaned:
        return False
    if cleaned.startswith(("/", "\\", "~")) or cleaned.casefold().startswith("file:"):
        return True
    if len(cleaned) >= 2 and cleaned[1] == ":":
        return True
    norm = posixpath.normpath(cleaned.replace("\\", "/"))
    parts = cleaned.replace("\\", "/").split("/")
    return ".." in parts or norm == ".." or norm.startswith("../")


def _check_path_within_repo(
    raw_path: str,
    repo_root: Path | None,
    *,
    write_allowed: bool = False,
) -> dict | None:
    """Validate that `raw_path` stays within `repo_root` and does not traverse
    `..`, expand `~`, use external drive/URI paths, or follow symlinks escaping
    `repo_root`."""
    cleaned = raw_path.strip().strip("\"'")
    if not cleaned:
        return None
    if cleaned.startswith("~") or cleaned.casefold().startswith("file:"):
        return {
            "error": {
                "code": "command_not_permitted",
                "message": (
                    f"command path {raw_path!r} uses disallowed home-expansion or URI syntax"
                ),
            }
        }
    norm = posixpath.normpath(cleaned.replace("\\", "/"))
    parts = cleaned.replace("\\", "/").split("/")
    if ".." in parts or norm == ".." or norm.startswith("../"):
        return {
            "error": {
                "code": "command_not_permitted",
                "message": (
                    f"command argument {raw_path!r} contains parent-directory traversal ('..') "
                    "outside the repository boundary"
                ),
            }
        }
    if repo_root is None:
        if _is_unrooted_escape_path(cleaned):
            return {
                "error": {
                    "code": "command_not_permitted",
                    "message": (
                        f"external or absolute path argument {raw_path!r} is not permitted "
                        "without a verified repository root"
                    ),
                }
            }
        return None

    if os.name != "nt" and len(cleaned) >= 2 and cleaned[1] == ":":
        return {
            "error": {
                "code": "command_not_permitted",
                "message": f"command path {raw_path!r} resolves outside the repository root",
            }
        }

    try:
        root_resolved = repo_root.resolve()
        candidate_unresolved = (
            Path(cleaned) if Path(cleaned).is_absolute() else (root_resolved / cleaned)
        )
        candidate_abs = Path(os.path.abspath(candidate_unresolved))
        if not is_within(root_resolved, candidate_abs):
            return {
                "error": {
                    "code": "command_not_permitted",
                    "message": (
                        f"command path {raw_path!r} resolves outside the repository root ({root_resolved})"
                    ),
                }
            }
        current = root_resolved
        for part in candidate_abs.relative_to(root_resolved).parts:
            current = current / part
            if current.is_symlink():
                return {
                    "error": {
                        "code": "command_not_permitted",
                        "message": (
                            f"command path {raw_path!r} traverses symlink "
                            f"{current.relative_to(root_resolved).as_posix()!r}"
                        ),
                    }
                }
        resolved = candidate_abs.resolve()
        if not is_within(root_resolved, resolved):
            return {
                "error": {
                    "code": "command_not_permitted",
                    "message": (
                        f"command path {raw_path!r} resolves outside the repository root ({root_resolved})"
                    ),
                }
            }
    except (OSError, RuntimeError, ValueError) as exc:
        return {
            "error": {
                "code": "command_not_permitted",
                "message": f"invalid command path {raw_path!r}: {exc}",
            }
        }
    return None


def _looks_like_path_token(token: str) -> bool:
    cleaned = token.strip().strip("\"'")
    if not cleaned or cleaned.startswith("-"):
        return False
    if "/" in cleaned or "\\" in cleaned or cleaned.startswith("~") or cleaned.casefold().startswith("file:"):
        return True
    if len(cleaned) >= 2 and cleaned[1] == ":":
        return True
    if cleaned in ("..", "."):
        return True
    return False


def _has_symlink_component_in_repo(repo_root: Path | None, token: str) -> bool:
    if repo_root is None:
        return False
    cleaned = token.strip().strip("\"'")
    if not cleaned or cleaned.startswith("-"):
        return False
    try:
        root_resolved = repo_root.resolve()
        candidate_abs = Path(os.path.abspath(root_resolved / cleaned))
        if not is_within(root_resolved, candidate_abs):
            return False
        current = root_resolved
        for part in candidate_abs.relative_to(root_resolved).parts:
            current = current / part
            if current.is_symlink():
                return True
    except (OSError, RuntimeError, ValueError):
        return False
    return False


def _validate_generic_path_args(
    args: Sequence[str],
    *,
    write_allowed: bool = False,
    repo_root: Path | None = None,
    all_positional_are_paths: bool = False,
) -> dict | None:
    """Validate path-bearing tokens and option values across `args`."""
    expect_path_for_opt: str | None = None
    for arg in args:
        if "\x00" in arg or "\n" in arg or "\r" in arg:
            return {
                "error": {
                    "code": "command_not_permitted",
                    "message": "command arguments may not contain null bytes or control newlines",
                }
            }
        if arg.strip() in _SHELL_CONTROL_TOKENS:
            return {
                "error": {
                    "code": "command_not_permitted",
                    "message": f"shell control or redirection operator {arg!r} is not permitted",
                }
            }

        if expect_path_for_opt is not None:
            expect_path_for_opt = None
            path_err = _check_path_within_repo(
                arg, repo_root, write_allowed=write_allowed
            )
            if path_err is not None:
                return path_err
            continue

        if arg in _PATH_TAKING_OPTIONS or arg.casefold() in _PATH_TAKING_OPTIONS:
            expect_path_for_opt = arg
            continue

        candidates: list[str] = []
        if arg.startswith("-") and "=" in arg:
            opt_name, val = arg.split("=", 1)
            if (
                opt_name in _PATH_TAKING_OPTIONS
                or opt_name.casefold() in _PATH_TAKING_OPTIONS
                or _looks_like_path_token(val)
                or _has_symlink_component_in_repo(repo_root, val)
            ):
                candidates.append(val)
        elif (
            arg.startswith("-")
            and not arg.startswith("--")
            and len(arg) > 2
            and (
                arg[:2] in ("-C", "-f", "-o", "-s", "-t", "-I", "-L")
                or _looks_like_path_token(arg[2:])
            )
        ):
            if _looks_like_path_token(arg[2:]) or _has_symlink_component_in_repo(repo_root, arg[2:]):
                candidates.append(arg[2:])
        elif not arg.startswith("-"):
            if (
                all_positional_are_paths
                or _looks_like_path_token(arg)
                or _has_symlink_component_in_repo(repo_root, arg)
            ):
                candidates.append(arg)

        for cand in candidates:
            path_err = _check_path_within_repo(
                cand, repo_root, write_allowed=write_allowed
            )
            if path_err is not None:
                return path_err
    return None


def _validate_git_command(
    argv: list[str],
    *,
    write_allowed: bool = False,
    repo_root: Path | None = None,
) -> dict | None:
    """Conservatively parse and validate a `git` command invocation.

    Handles global Git options (including `-C`, `-c`, `--git-dir`, `--work-tree`,
    `--exec-path`, `--config-env`, `-p`, `--paginate`) before extracting the
    subcommand so an option argument can never be mistaken for the subcommand.
    Git is only permitted for read-only repository inspection within `repo_root`
    and is never used as a mutation or rollback mechanism.
    """
    idx = 1
    n = len(argv)
    while idx < n:
        tok = argv[idx]
        tok_lower = tok.casefold()
        if tok == "--":
            idx += 1
            break
        if not tok.startswith("-"):
            break

        # Handle case-sensitive uppercase -C <path> or -C<path> before checking -c.
        if tok == "-C" or (tok.startswith("-C") and len(tok) > 2):
            if tok == "-C":
                if idx + 1 >= n:
                    return {
                        "error": {
                            "code": "invalid_input",
                            "message": "git -C requires a directory argument",
                        }
                    }
                target_dir = argv[idx + 1]
                idx += 2
            else:
                target_dir = tok[2:]
                idx += 1
            if repo_root is None:
                return {
                    "error": {
                        "code": "command_not_permitted",
                        "message": "git -C directory redirection is not permitted without a verified repository root",
                    }
                }
            path_err = _check_path_within_repo(
                target_dir, repo_root, write_allowed=write_allowed
            )
            if path_err is not None:
                return path_err
            continue

        # Block dangerous global options (-c, -p, --paginate, --git-dir, --work-tree, --exec-path, etc.)
        if tok == "-c" or (tok.startswith("-c") and len(tok) > 2) or any(
            tok_lower == prefix or tok_lower.startswith(prefix + "=")
            for prefix in _FORBIDDEN_GIT_GLOBAL_PREFIXES
            if prefix != "-c"
        ):
            return {
                "error": {
                    "code": "command_not_permitted",
                    "message": (
                        f"git global option {tok!r} is not permitted in run_command "
                        "(config overrides, external pagers, and repository redirection are blocked)"
                    ),
                }
            }

        if tok_lower in _SAFE_GIT_GLOBAL_BOOL_FLAGS:
            idx += 1
            continue

        # Fail closed on any unrecognized global Git flag before the subcommand so
        # an option argument can never be mistaken for the subcommand.
        return {
            "error": {
                "code": "command_not_permitted",
                "message": f"unrecognized or disallowed git global option {tok!r} before subcommand",
            }
        }

    if idx >= n:
        if any(a.casefold() in ("--version", "--help") for a in argv[1:]):
            return None
        return {
            "error": {
                "code": "command_not_permitted",
                "message": "git command must specify an allowed read-only subcommand",
            }
        }

    sub = argv[idx].casefold()
    if sub in _FORBIDDEN_GIT_SUBCOMMANDS or sub not in _ALLOWED_READONLY_GIT_SUBCOMMANDS:
        return {
            "error": {
                "code": "command_not_permitted",
                "message": (
                    f"git subcommand {argv[idx]!r} is not permitted in run_command; "
                    f"only read-only inspection subcommands ({', '.join(sorted(_ALLOWED_READONLY_GIT_SUBCOMMANDS))}) are allowed"
                ),
            }
        }

    # Inspect subcommand arguments for output redirection or external command execution.
    for sub_arg in argv[idx + 1 :]:
        sub_lower = sub_arg.casefold()
        if sub_lower == "-o" or (sub_lower.startswith("-o") and len(sub_lower) > 2):
            return {
                "error": {
                    "code": "command_not_permitted",
                    "message": f"git {sub} output-file option {sub_arg!r} is not permitted",
                }
            }
        if any(
            sub_lower == flag or sub_lower.startswith(flag + "=")
            for flag in _FORBIDDEN_GIT_SUBCOMMAND_FLAGS
        ):
            return {
                "error": {
                    "code": "command_not_permitted",
                    "message": f"git {sub} option {sub_arg!r} is not permitted in run_command",
                }
            }

    return _validate_generic_path_args(
        argv[idx + 1 :], write_allowed=write_allowed, repo_root=repo_root
    )


def _validate_test_or_lint_args(
    tool_name: str,
    args: Sequence[str],
    *,
    write_allowed: bool = False,
    repo_root: Path | None = None,
) -> dict | None:
    """Validate arguments to allowlisted test/lint modules and runners (`pytest`,
    `unittest`, `py_compile`, `mypy`, `flake8`, `ruff`)."""
    lower_tool = tool_name.casefold()
    if lower_tool in ("pytest", "py.test"):
        forbidden_pytest_prefixes = (
            "--junitxml", "--junit-xml", "--resultlog", "--result-log",
            "--pastebin", "--genscript", "--debug", "--override-ini",
            "--basetemp", "--import-mode=prepend", "--log-file",
        )
        for arg in args:
            arg_low = arg.casefold()
            if arg in ("-o", "-p") or (arg.startswith(("-o", "-p")) and len(arg) > 2):
                return {
                    "error": {
                        "code": "command_not_permitted",
                        "message": f"pytest plugin or configuration override flag {arg!r} is not permitted",
                    }
                }
            if any(
                arg_low == p or arg_low.startswith(p + "=")
                for p in forbidden_pytest_prefixes
            ):
                return {
                    "error": {
                        "code": "command_not_permitted",
                        "message": f"pytest option {arg!r} is not permitted in run_command",
                    }
                }
        return _validate_generic_path_args(
            args, write_allowed=write_allowed, repo_root=repo_root
        )

    if lower_tool == "unittest":
        return _validate_generic_path_args(
            args, write_allowed=write_allowed, repo_root=repo_root
        )

    if lower_tool == "py_compile":
        file_args: list[str] = []
        for arg in args:
            if arg == "-":
                return {
                    "error": {
                        "code": "command_not_permitted",
                        "message": "py_compile stdin '-' argument is not permitted",
                    }
                }
            if arg == "-q":
                continue
            if arg.startswith("-"):
                return {
                    "error": {
                        "code": "command_not_permitted",
                        "message": f"py_compile option {arg!r} is not permitted",
                    }
                }
            if not arg.strip("\"'").casefold().endswith(".py"):
                return {
                    "error": {
                        "code": "command_not_permitted",
                        "message": f"py_compile target {arg!r} must be a repository .py file",
                    }
                }
            file_args.append(arg)
        if not file_args:
            return {
                "error": {
                    "code": "command_not_permitted",
                    "message": "py_compile requires at least one repository .py file argument",
                }
            }
        return _validate_generic_path_args(
            file_args,
            write_allowed=write_allowed,
            repo_root=repo_root,
            all_positional_are_paths=True,
        )

    if lower_tool in ("ruff", "flake8", "mypy"):
        forbidden_mypy_prefixes = (
            "--install-types", "--write-baseline", "--shadow-file", "--junit-xml",
            "--html-report", "--cobertura-xml-report", "--linecount-report",
            "--linecoverage-report", "--lineprecision-report", "--txt-report",
            "--xml-report", "--xslt-html-report", "--xslt-txt-report",
            "--any-exprs-report", "--export-types",
        )
        for arg in args:
            arg_low = arg.casefold()
            if lower_tool == "mypy" and any(
                arg_low == p or arg_low.startswith(p + "=")
                for p in forbidden_mypy_prefixes
            ):
                return {
                    "error": {
                        "code": "command_not_permitted",
                        "message": f"mypy option {arg!r} is not permitted in run_command",
                    }
                }
            if not write_allowed and (
                arg_low in ("--fix", "--fix-only", "--unsafe-fixes", "--add-noqa", "-o", "--output-file")
                or arg_low.startswith(("--output-file=", "-o"))
            ):
                return {
                    "error": {
                        "code": "command_not_permitted",
                        "message": f"{tool_name} mutating or output-file option {arg!r} is not permitted without WRITE capability",
                    }
                }
        if (
            lower_tool == "ruff"
            and not write_allowed
            and any(a.casefold() == "format" for a in args)
            and not any(a.casefold() in ("--check", "--diff") for a in args)
        ):
            return {
                "error": {
                    "code": "command_not_permitted",
                    "message": "ruff format without --check or --diff is not permitted without WRITE capability",
                }
            }
        return _validate_generic_path_args(
            args, write_allowed=write_allowed, repo_root=repo_root
        )

    return _validate_generic_path_args(
        args, write_allowed=write_allowed, repo_root=repo_root
    )


def _check_repo_module_shadowing(
    mod_name: str,
    repo_root: Path | None,
    *,
    isolated_path: bool = False,
) -> dict | None:
    """When `python -m <mod>` runs in `cwd=repo_root` without `-I` or `-P`,
    Python prepends `cwd` to `sys.path` and also imports `sitecustomize` /
    `usercustomize` if present. Reject if `repo_root` contains a file or
    directory that would shadow `<mod>` or startup site hooks."""
    if repo_root is None or isolated_path:
        return None
    top_pkg = mod_name.split(".", 1)[0].strip()
    if not top_pkg:
        return None
    try:
        root_resolved = repo_root.resolve()
        shadow_candidates = (
            root_resolved / f"{top_pkg}.py",
            root_resolved / top_pkg,
            root_resolved / "sitecustomize.py",
            root_resolved / "usercustomize.py",
        )
        for cand in shadow_candidates:
            if cand.exists() or cand.is_symlink():
                rel = cand.relative_to(root_resolved).as_posix()
                return {
                    "error": {
                        "code": "command_not_permitted",
                        "message": (
                            f"python -m {mod_name} without -I is blocked because repository entry "
                            f"{rel!r} would shadow the trusted module or startup hook in working directory"
                        ),
                    }
                }
    except (OSError, RuntimeError, ValueError):
        return {
            "error": {
                "code": "command_not_permitted",
                "message": f"could not verify module shadowing safety for python -m {mod_name}",
            }
        }
    return None


def _validate_python_command(
    argv: list[str],
    *,
    write_allowed: bool = False,
    repo_root: Path | None = None,
) -> dict | None:
    """Validate a Python interpreter command invocation (`python`, `python3`, `py`,
    `sys.executable`).

    Policy:
      - Inline code execution (`-c`, including attached `-c...` or clustered
        short flags like `-Bc`, `-Ic`, `-uc`, `-OOc`), interactive mode (`-i`),
        and stdin execution (`-` or bare `python`) are ALWAYS blocked regardless
        of `write_allowed`, because arbitrary inline code can write outside
        `repo_root` or trigger external side effects outside checkpoint scope.
      - `-m <module>` is permitted ONLY for allowlisted verification/lint
        modules (`pytest`, `unittest`, `py_compile`, `mypy`, `flake8`, `ruff`),
        and is blocked if `repo_root` shadows `<module>` without `-I`.
      - Positional script execution (`python <script.py> ...`) requires
        `<script.py>` to be a `.py` file contained within `repo_root` (no `..`
        traversal, no external path, and no symlink).
    """
    if len(argv) < 2:
        return {
            "error": {
                "code": "command_not_permitted",
                "message": "interactive python REPL without a target module or repository .py script is not permitted",
            }
        }

    idx = 1
    n = len(argv)
    isolated_path = False
    while idx < n:
        tok = argv[idx]
        if tok == "-":
            return {
                "error": {
                    "code": "command_not_permitted",
                    "message": "python stdin script execution ('-') is not permitted",
                }
            }

        # Check -W and -X before generic short-flag clustering since -W/-X take values.
        if tok in ("-W", "-X") or (tok.startswith(("-W", "-X")) and len(tok) > 2):
            opt_flag = tok[:2]
            if len(tok) == 2:
                if idx + 1 >= n:
                    return {
                        "error": {
                            "code": "invalid_input",
                            "message": f"python {opt_flag} requires an argument",
                        }
                    }
                opt_val = argv[idx + 1]
                idx += 2
            else:
                opt_val = tok[2:]
                idx += 1
            if opt_flag == "-X":
                if (
                    "pycache_prefix" in opt_val.casefold()
                    or "/" in opt_val
                    or "\\" in opt_val
                ):
                    return {
                        "error": {
                            "code": "command_not_permitted",
                            "message": f"python -X {opt_val!r} option is not permitted",
                        }
                    }
                if opt_val.casefold() == "safe_path":
                    isolated_path = True
            continue

        # Detect -c (inline code) or -i (interactive inspect), whether standalone,
        # attached (-c"code"), or clustered with boolean flags (-Bc, -Ic, -uc, -Bi).
        if tok.startswith("-") and not tok.startswith("--"):
            short_body = tok[1:]
            # Strip leading safe boolean flags (b, B, E, I, O, P, q, s, S, u) to see
            # if the cluster invokes -c, -i, or -m.
            stripped_cluster = short_body.lstrip("bBEIOPqsSu")
            if (
                tok == "-c"
                or tok.startswith("-c")
                or stripped_cluster.startswith("c")
                or "c" in short_body
            ):
                code_str = ""
                if tok == "-c" or stripped_cluster == "c":
                    if idx + 1 < n:
                        code_str = argv[idx + 1]
                elif stripped_cluster.startswith("c"):
                    code_str = stripped_cluster[1:]
                detail_parts = [
                    "inline python -c code execution is not permitted in run_command or verify_repository"
                ]
                if code_str and _INLINE_PYTHON_ALWAYS_FORBIDDEN_RE.search(code_str):
                    detail_parts.append(
                        "detected forbidden system, subprocess, dynamic-import, or code-evaluation operation"
                    )
                if code_str and _INLINE_PYTHON_READONLY_MUTATION_RE.search(code_str):
                    detail_parts.append(
                        "detected file-mutation or filesystem write operation"
                    )
                return {
                    "error": {
                        "code": "command_not_permitted",
                        "message": "; ".join(detail_parts),
                    }
                }
            if (
                tok == "-i"
                or tok.startswith("-i")
                or "i" in short_body
            ):
                return {
                    "error": {
                        "code": "command_not_permitted",
                        "message": f"interactive python flag {tok!r} is not permitted",
                    }
                }

            if "I" in short_body or "P" in short_body:
                isolated_path = True

            # Detect -m (standalone -m, clustered -Bm / -Im / -um, or attached -m<mod>).
            if tok == "-m" or stripped_cluster.startswith("m"):
                if tok == "-m" or stripped_cluster == "m":
                    if idx + 1 >= n:
                        return {
                            "error": {
                                "code": "invalid_input",
                                "message": "python -m requires a module name",
                            }
                        }
                    mod_name = argv[idx + 1].strip()
                    mod_args = argv[idx + 2 :]
                else:
                    mod_name = stripped_cluster[1:].strip()
                    mod_args = argv[idx + 1 :]
                mod_lower = mod_name.casefold()
                if (
                    not mod_name
                    or mod_lower in _FORBIDDEN_PYTHON_MODULES_ALWAYS
                    or mod_lower not in _ALLOWED_PYTHON_MODULES
                ):
                    return {
                        "error": {
                            "code": "command_not_permitted",
                            "message": (
                                f"python -m {mod_name} is not permitted; only allowlisted verification "
                                f"modules ({', '.join(sorted(_ALLOWED_PYTHON_MODULES))}) are allowed"
                            ),
                        }
                    }
                shadow_err = _check_repo_module_shadowing(
                    mod_lower, repo_root, isolated_path=isolated_path
                )
                if shadow_err is not None:
                    return shadow_err
                return _validate_test_or_lint_args(
                    mod_lower,
                    mod_args,
                    write_allowed=write_allowed,
                    repo_root=repo_root,
                )

            # Safe boolean short-flag cluster (e.g. -B, -I, -O, -OO, -P, -q, -s, -S, -u, -E, -b)
            if short_body and set(short_body) <= set("bBEIOPqsSu"):
                idx += 1
                continue

            # Version / help short flags
            if tok in ("-V", "-h"):
                if idx == n - 1:
                    return None
                idx += 1
                continue

            # Windows `py` launcher version selector (-3, -3.11)
            if re.match(r"^-[23](?:\.\d+)?$", tok):
                idx += 1
                continue

            return {
                "error": {
                    "code": "command_not_permitted",
                    "message": f"unrecognized or disallowed python option {tok!r}",
                }
            }

        if tok.startswith("--"):
            if tok in ("--version", "--help"):
                if idx == n - 1:
                    return None
                idx += 1
                continue
            return {
                "error": {
                    "code": "command_not_permitted",
                    "message": f"disallowed python long option {tok!r}",
                }
            }

        # First positional argument: must be a repository-contained .py script file.
        script_arg = tok
        cleaned_script = script_arg.strip().strip("\"'")
        if not cleaned_script.casefold().endswith(".py"):
            return {
                "error": {
                    "code": "command_not_permitted",
                    "message": (
                        f"python script target {script_arg!r} is not permitted; "
                        "only repository .py scripts or allowlisted -m modules may be executed"
                    ),
                }
            }
        path_err = _check_path_within_repo(
            script_arg, repo_root, write_allowed=write_allowed
        )
        if path_err is not None:
            return path_err
        return _validate_generic_path_args(
            argv[idx + 1 :], write_allowed=write_allowed, repo_root=repo_root
        )

    return {
        "error": {
            "code": "command_not_permitted",
            "message": "python command must specify an allowlisted -m module or repository .py script",
        }
    }


def _validate_inspection_utility_command(
    exe_stem: str,
    argv: list[str],
    *,
    write_allowed: bool = False,
    repo_root: Path | None = None,
) -> dict | None:
    """Validate read-only repository inspection utilities (`find`, `grep`, `rg`,
    `ls`, `cat`, `head`, `tail`, `wc`, `diff`, `stat`, `file`)."""
    args = argv[1:]
    if exe_stem == "find":
        for arg in args:
            if arg.casefold() in _FORBIDDEN_FIND_FLAGS:
                return {
                    "error": {
                        "code": "command_not_permitted",
                        "message": f"find flag {arg!r} is not permitted in run_command",
                    }
                }
        # Any leading non-flag arguments to `find` are starting directory paths.
        leading_paths: list[str] = []
        for arg in args:
            if arg.startswith("-") or arg in ("!", "(", ")"):
                break
            leading_paths.append(arg)
        path_err = _validate_generic_path_args(
            leading_paths,
            write_allowed=write_allowed,
            repo_root=repo_root,
            all_positional_are_paths=True,
        )
        if path_err is not None:
            return path_err
        return _validate_generic_path_args(
            args, write_allowed=write_allowed, repo_root=repo_root
        )

    if exe_stem in ("grep", "rg"):
        for arg in args:
            arg_low = arg.casefold()
            if arg_low in ("--pre", "--pre-glob") or arg_low.startswith(("--pre=", "--pre-glob=")):
                return {
                    "error": {
                        "code": "command_not_permitted",
                        "message": f"{exe_stem} preprocessor option {arg!r} is not permitted",
                    }
                }
            if exe_stem == "grep" and (
                arg_low == "--dereference-recursive"
                or (arg.startswith("-") and not arg.startswith("--") and "R" in arg[1:])
            ):
                return {
                    "error": {
                        "code": "command_not_permitted",
                        "message": f"grep symlink-dereferencing flag {arg!r} is not permitted",
                    }
                }
            if exe_stem == "rg" and (
                arg_low == "--follow"
                or (arg.startswith("-") and not arg.startswith("--") and "L" in arg[1:])
            ):
                return {
                    "error": {
                        "code": "command_not_permitted",
                        "message": f"rg symlink-following flag {arg!r} is not permitted",
                    }
                }
        return _validate_generic_path_args(
            args, write_allowed=write_allowed, repo_root=repo_root
        )

    # For ls, cat, head, tail, wc, diff, stat, file: block output-file and
    # symlink-dereferencing flags, and validate every positional argument as a
    # repository path.
    for arg in args:
        arg_low = arg.casefold()
        if arg_low == "-o" or arg_low.startswith(("-o", "--output=")) or arg_low == "--output":
            return {
                "error": {
                    "code": "command_not_permitted",
                    "message": f"{exe_stem} output option {arg!r} is not permitted",
                }
            }
        if exe_stem in ("ls", "stat", "file") and (
            arg_low in ("--dereference", "--dereference-command-line")
            or (arg.startswith("-") and not arg.startswith("--") and "L" in arg[1:])
        ):
            return {
                "error": {
                    "code": "command_not_permitted",
                    "message": f"{exe_stem} symlink-dereferencing option {arg!r} is not permitted",
                }
            }
    return _validate_generic_path_args(
        args,
        write_allowed=write_allowed,
        repo_root=repo_root,
        all_positional_are_paths=True,
    )


def _parse_command_argv(command: str | Sequence[str]) -> tuple[list[str] | None, dict | None]:
    """Parse and validate `command` (either a command string or an `argv` sequence)
    into a non-empty `list[str]` without invoking a shell, using the exact same
    tokenizer (`_split_command`) as `RepoService._run_command`."""
    if isinstance(command, (list, tuple)):
        if not command or any(not isinstance(item, str) for item in command):
            return None, {
                "error": {
                    "code": "invalid_input",
                    "message": "command must be a non-empty string or list of strings",
                }
            }
        argv = [str(item) for item in command]
        if not argv[0].strip():
            return None, {
                "error": {
                    "code": "invalid_input",
                    "message": "command executable must be a non-empty string",
                }
            }
        total_chars = sum(len(item) for item in argv)
        if total_chars > MAX_COMMAND_CHARS:
            return None, {
                "error": {
                    "code": "invalid_input",
                    "message": f"command exceeds maximum length ({MAX_COMMAND_CHARS} characters)",
                }
            }
        if any("\x00" in item for item in argv):
            return None, {
                "error": {
                    "code": "invalid_input",
                    "message": "command contains null bytes",
                }
            }
        if len(argv) > MAX_COMMAND_ARGS:
            return None, {
                "error": {
                    "code": "invalid_input",
                    "message": f"command exceeds maximum argument count ({MAX_COMMAND_ARGS})",
                }
            }
        return argv, None

    if not isinstance(command, str) or not command.strip():
        return None, {
            "error": {
                "code": "invalid_input",
                "message": "command must be a non-empty string",
            }
        }
    if len(command) > MAX_COMMAND_CHARS:
        return None, {
            "error": {
                "code": "invalid_input",
                "message": f"command exceeds maximum length ({MAX_COMMAND_CHARS} characters)",
            }
        }
    if "\x00" in command:
        return None, {
            "error": {
                "code": "invalid_input",
                "message": "command contains null bytes",
            }
        }
    try:
        argv = _split_command(command)
    except ValueError as exc:
        return None, {
            "error": {
                "code": "invalid_input",
                "message": f"could not parse command: {exc}",
            }
        }
    if not argv:
        return None, {
            "error": {
                "code": "invalid_input",
                "message": "command must be a non-empty string",
            }
        }
    if len(argv) > MAX_COMMAND_ARGS:
        return None, {
            "error": {
                "code": "invalid_input",
                "message": f"command exceeds maximum argument count ({MAX_COMMAND_ARGS})",
            }
        }
    return argv, None


def validate_command_safety(
    command: str | Sequence[str],
    *,
    write_allowed: bool = False,
    repo_root: Path | str | None = None,
) -> dict | None:
    """Validate that `command` (a string or `argv` list/tuple) conforms to ARI's
    command execution policy before any subprocess is created.

    Returns an `{"error": {"code": ..., "message": ...}}` dict when blocked, or
    `None` when permitted. Note: this pre-execution check is paired with
    mandatory pre-command checkpointing and post-command diff inspection in
    `MutationTransaction`.
    """
    argv, parse_err = _parse_command_argv(command)
    if parse_err is not None or argv is None:
        return parse_err

    resolved_root: Path | None = None
    if repo_root is not None:
        try:
            resolved_root = Path(repo_root).expanduser().resolve()
        except OSError:
            resolved_root = None

    exe_raw = argv[0].strip().strip("\"'")
    exe_norm = exe_raw.replace("\\", "/")
    exe_base = posixpath.basename(exe_norm).casefold()
    exe_stem = _strip_exe_suffix(exe_base)

    if exe_base in _SHELL_BINARIES or exe_stem in _SHELL_BINARIES:
        return {
            "error": {
                "code": "command_not_permitted",
                "message": (
                    f"shell wrapper {argv[0]!r} is not permitted in run_command; "
                    "invoke the target verification executable directly (e.g. pytest or python)"
                ),
            }
        }

    if exe_base in _WRAPPER_BINARIES or exe_stem in _WRAPPER_BINARIES:
        return {
            "error": {
                "code": "command_not_permitted",
                "message": f"command wrapper {argv[0]!r} is not permitted in run_command",
            }
        }

    if exe_base in _ALWAYS_FORBIDDEN_BINARIES or exe_stem in _ALWAYS_FORBIDDEN_BINARIES:
        return {
            "error": {
                "code": "command_not_permitted",
                "message": f"destructive or external command {argv[0]!r} is not permitted in run_command",
            }
        }

    if (
        exe_base in _FORBIDDEN_INTERPRETERS_AND_MUTATORS
        or exe_stem in _FORBIDDEN_INTERPRETERS_AND_MUTATORS
    ):
        for arg in argv[1:]:
            if arg in ("-e", "-E", "--eval", "-p", "--print", "-r", "-i", "--in-place") or arg.startswith(("-i", "-e")):
                return {
                    "error": {
                        "code": "command_not_permitted",
                        "message": f"inline script or in-place flag {arg!r} on {argv[0]!r} is not permitted",
                    }
                }
        return {
            "error": {
                "code": "command_not_permitted",
                "message": (
                    f"interpreter or file-mutating command {argv[0]!r} is not permitted in "
                    "run_command; use repository tools (get_file, write_file) or allowlisted "
                    "Python/Git verification commands"
                ),
            }
        }

    # Verify executable identity before checking allowlisted Python or tool names:
    # blocks relative paths (`./python`, `./pytest`, `sub/git`), external paths
    # other than `sys.executable`, script/batch suffixes (`pytest.sh`, `git.bat`),
    # and PATH shadowing inside `resolved_root`.
    identity_err = _verify_executable_identity(exe_raw, exe_base, resolved_root)
    if identity_err is not None:
        return identity_err

    is_python = _is_python_executable(exe_stem, exe_raw, repo_root=resolved_root)

    if is_python:
        return _validate_python_command(
            argv, write_allowed=write_allowed, repo_root=resolved_root
        )

    if exe_stem not in _ALLOWED_DIRECT_EXECUTABLES:
        return {
            "error": {
                "code": "command_not_permitted",
                "message": (
                    f"executable {argv[0]!r} is not in the command allowlist; permitted commands "
                    "are python (-m pytest/unittest/py_compile/mypy/flake8/ruff or repo .py script), "
                    "pytest, ruff, flake8, mypy, read-only git, and read-only inspection utilities"
                ),
            }
        }

    if exe_stem == "git":
        return _validate_git_command(
            argv, write_allowed=write_allowed, repo_root=resolved_root
        )

    if exe_stem in _DIRECT_TEST_AND_LINT_BINARIES:
        return _validate_test_or_lint_args(
            exe_stem,
            argv[1:],
            write_allowed=write_allowed,
            repo_root=resolved_root,
        )

    if exe_stem in _READONLY_INSPECTION_UTILITIES:
        return _validate_inspection_utility_command(
            exe_stem,
            argv,
            write_allowed=write_allowed,
            repo_root=resolved_root,
        )

    return _validate_generic_path_args(
        argv[1:], write_allowed=write_allowed, repo_root=resolved_root
    )


def _is_provably_readonly_command(
    command: str | Sequence[str],
    *,
    repo_root: Path | str | None = None,
) -> bool:
    """Return True only when `command` is statically proven to be a read-only
    inspection command that does not execute repository code or write files
    (used when `config.enable_checkpoints=False` to fail closed on any command
    that requires rollback protection)."""
    if validate_command_safety(command, write_allowed=False, repo_root=repo_root) is not None:
        return False
    argv, err = _parse_command_argv(command)
    if err is not None or not argv:
        return False
    exe_raw = argv[0].strip().strip("\"'")
    exe_base = posixpath.basename(exe_raw.replace("\\", "/")).casefold()
    exe_stem = _strip_exe_suffix(exe_base)
    if exe_stem == "git" or exe_stem in _READONLY_INSPECTION_UTILITIES:
        return True
    if _is_python_executable(exe_stem, exe_raw) or exe_stem in _DIRECT_TEST_AND_LINT_BINARIES:
        # Only `--version` / `-V` / `--help` / `-h` is provably free of file mutations
        # without a checkpoint; running tests, scripts, or compilers can write files.
        return len(argv) == 2 and argv[1] in ("--version", "-V", "--help", "-h")
    return False


def terminal_state(verification: dict, repair_limit_reached: bool) -> str:
    """Classify how a modification or verification check ended."""
    if repair_limit_reached and not verification.get("passed"):
        return REPAIR_LIMIT_REACHED
    return _STATUS_TO_STATE.get(verification.get("status"), FAILED)


def verification_signature(verification: dict) -> tuple:
    """A cheap fingerprint of a verification result, used to notice that two
    repair attempts in a row produced the same failure."""
    return (
        verification.get("status"),
        verification.get("exit_code"),
        (verification.get("stderr") or "")[:300],
    )


def format_change_summary(diff: dict) -> str:
    """Human-readable one-liner for a RepoService.diff_since_checkpoint() result."""
    if not isinstance(diff, dict) or "error" in diff or diff.get("changed") is None:
        return ""
    if not diff["changed"]:
        return "No files were changed."
    parts = []
    if diff.get("added"):
        parts.append(
            f"{len(diff['added'])} file(s) added ({', '.join(diff['added'][:5])}{'...' if len(diff['added']) > 5 else ''})"
        )
    if diff.get("modified"):
        parts.append(
            f"{len(diff['modified'])} file(s) modified ({', '.join(diff['modified'][:5])}{'...' if len(diff['modified']) > 5 else ''})"
        )
    if diff.get("deleted"):
        parts.append(
            f"{len(diff['deleted'])} file(s) deleted ({', '.join(diff['deleted'][:5])}{'...' if len(diff['deleted']) > 5 else ''})"
        )
    if diff.get("added_directories"):
        parts.append(
            f"{len(diff['added_directories'])} directory(s) added ({', '.join(diff['added_directories'][:5])})"
        )
    if diff.get("removed_directories"):
        parts.append(
            f"{len(diff['removed_directories'])} directory(s) removed ({', '.join(diff['removed_directories'][:5])})"
        )
    return "Changes: " + "; ".join(parts) + "."


def _bound_report(report: str, limit: int = 12000) -> str:
    text = str(report or "")
    return text if len(text) <= limit else text[: limit - 80] + " ...[report truncated]"


def _display_cmd_str(command: str | Sequence[str]) -> str:
    if isinstance(command, str):
        return command
    if isinstance(command, (list, tuple)):
        return shlex.join(str(x) for x in command)
    return str(command)


class Checkpoint:
    """A task checkpoint and its creation status.

    A disabled checkpoint is an explicit configuration choice. A failed
    checkpoint is different: callers must not start a protected mutation or
    command without the protection that was requested.
    """

    def __init__(
        self,
        service: RepoService,
        checkpoint_id: str | None,
        error: dict | None = None,
    ):
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
            return cls(
                service,
                None,
                result.get("error")
                or {
                    "code": "checkpoint_creation_failed",
                    "message": "checkpoint creation did not return a checkpoint id",
                },
            )
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


class MutationTransaction:
    """Invocation-scoped deterministic checkpoint -> modification/command ->
    verification -> bounded repair -> rollback lifecycle.

    Tracks transaction success independently from final-response generation so
    a verified code change is never rolled back merely because a later LLM call
    fails, while failed unrepaired changes or unauthorized command mutations
    are rolled back according to policy.

    Serializes modifying/checkpointed operations on the same canonical
    repository root via `service.repo_lock` owned by `self.tx_id`.
    """

    def __init__(
        self,
        service: RepoService,
        config: AgentConfig,
        label: str = "",
        tx_id: str | None = None,
    ):
        self.service = service
        self.config = config
        self.label = label
        self.tx_id = tx_id or f"tx_{uuid.uuid4().hex[:12]}"
        self.checkpoint: Checkpoint | None = None
        self.preserve_checkpoint: bool = False
        self.changed_files: list[str] = []
        self.file_diffs: dict[str, dict[str, Any]] = {}
        self.repair_attempts: int = 0
        self.in_repair_loop: bool = False
        self.repair_limit_exhausted: bool = False
        self.termination_reason: str | None = None
        self.last_signature: tuple | None = None
        self.status: str = "clean"
        self.last_verification: dict | None = None
        self.last_state: str | None = None
        self.last_report: str = ""
        self.last_rollback: dict | None = None
        self.last_refresh: dict | None = None

    def __enter__(self) -> "MutationTransaction":
        self._acquire_repo_lock()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_type is not None and self.in_repair_loop:
            self.finalize_if_unrepaired(
                termination_reason=f"exception:{getattr(exc_type, '__name__', 'Exception')}"
            )
        else:
            self.cleanup()
        return False

    def _repo_root(self) -> Path | None:
        if getattr(self.service, "index", None) is not None and getattr(self.service.index, "root", None):
            try:
                return Path(self.service.index.root).resolve()
            except OSError:
                return None
        return None

    def _acquire_repo_lock(self) -> None:
        lock = getattr(self.service, "repo_lock", None)
        if lock is not None and not lock.is_owned_by(self.tx_id):
            lock.acquire(self.tx_id)
        set_current_thread_tx_id(self.tx_id)

    def _release_repo_lock(self) -> None:
        lock = getattr(self.service, "repo_lock", None)
        if lock is not None:
            lock.release_all(self.tx_id)
        set_current_thread_tx_id(None)

    def to_state_dict(self) -> dict[str, Any]:
        """Serialize complete transaction state for per-invocation LangGraph state."""
        return {
            "tx_id": self.tx_id,
            "label": self.label,
            "checkpoint_id": self.checkpoint.checkpoint_id if self.checkpoint is not None else None,
            "checkpoint_error": self.checkpoint.error if self.checkpoint is not None else None,
            "checkpoint_initialized": self.checkpoint is not None,
            "preserve_checkpoint": self.preserve_checkpoint,
            "changed_files": list(self.changed_files),
            "file_diffs": dict(self.file_diffs),
            "repair_attempts": self.repair_attempts,
            "in_repair_loop": self.in_repair_loop,
            "repair_limit_exhausted": self.repair_limit_exhausted,
            "termination_reason": self.termination_reason,
            "last_signature": list(self.last_signature) if self.last_signature is not None else None,
            "status": self.status,
            "last_verification": self.last_verification,
            "last_state": self.last_state,
            "last_report": self.last_report,
            "last_rollback": self.last_rollback,
            "last_refresh": self.last_refresh,
            "checkpoint_active": bool(self.checkpoint and self.checkpoint.checkpoint_id),
        }

    def snapshot(self) -> dict[str, Any]:
        """Alias for `to_state_dict()` stored in `DeepAgentState['transaction']`."""
        return self.to_state_dict()

    @classmethod
    def from_state_dict(
        cls,
        service: RepoService,
        config: AgentConfig,
        data: dict[str, Any] | None,
        label: str = "",
    ) -> "MutationTransaction":
        """Reconstruct the invocation's `MutationTransaction` from `DeepAgentState`."""
        if not isinstance(data, dict):
            return cls(service, config, label=label)
        tx = cls(
            service,
            config,
            label=str(data.get("label") or label),
            tx_id=str(data.get("tx_id") or "") or None,
        )
        cp_id = data.get("checkpoint_id")
        cp_err = data.get("checkpoint_error")
        if data.get("checkpoint_initialized") or cp_id is not None or cp_err is not None:
            tx.checkpoint = Checkpoint(
                service,
                cp_id if isinstance(cp_id, str) and cp_id else None,
                cp_err if isinstance(cp_err, dict) else None,
            )
        tx.preserve_checkpoint = bool(data.get("preserve_checkpoint", False))
        tx.changed_files = [
            str(f) for f in (data.get("changed_files") or []) if isinstance(f, str) and f
        ]
        raw_fdiffs = data.get("file_diffs")
        if isinstance(raw_fdiffs, dict):
            tx.file_diffs = {str(k): dict(v) for k, v in raw_fdiffs.items() if isinstance(v, dict)}
        tx.repair_attempts = int(data.get("repair_attempts", 0) or 0)
        tx.in_repair_loop = bool(data.get("in_repair_loop", False))
        tx.repair_limit_exhausted = bool(data.get("repair_limit_exhausted", False))
        tx.termination_reason = data.get("termination_reason")
        raw_sig = data.get("last_signature")
        tx.last_signature = tuple(raw_sig) if isinstance(raw_sig, (list, tuple)) else None
        tx.status = str(data.get("status") or "clean")
        tx.last_verification = (
            data.get("last_verification")
            if isinstance(data.get("last_verification"), dict)
            else None
        )
        tx.last_state = (
            data.get("last_state") if isinstance(data.get("last_state"), str) else None
        )
        tx.last_report = str(data.get("last_report") or "")
        tx.last_rollback = (
            data.get("last_rollback") if isinstance(data.get("last_rollback"), dict) else None
        )
        tx.last_refresh = (
            data.get("last_refresh") if isinstance(data.get("last_refresh"), dict) else None
        )
        return tx

    def _checkpoint_changed_files(self) -> list[str]:
        if self.checkpoint is None:
            return []
        diff = self.checkpoint.diff()
        if not isinstance(diff, dict) or diff.get("changed") is None:
            return []
        return (
            list(diff.get("added", []))
            + list(diff.get("modified", []))
            + list(diff.get("deleted", []))
            + list(diff.get("added_directories", []))
            + list(diff.get("removed_directories", []))
        )

    def _record_file_diff(
        self,
        rel_path: str,
        *,
        before_content: str | None,
        after_content: str | None,
        existed_before: bool,
    ) -> None:
        """Record or update the unified diff and pre-task snapshot for `rel_path`."""
        norm_path = rel_path.replace("\\", "/")
        if norm_path in self.file_diffs:
            orig_before = self.file_diffs[norm_path].get("before_content")
            orig_existed = bool(self.file_diffs[norm_path].get("existed_before", orig_before is not None))
        else:
            orig_before = before_content
            orig_existed = existed_before

        before_lines = (orig_before or "").splitlines(keepends=True)
        after_lines = (after_content or "").splitlines(keepends=True)
        diff_lines = list(
            difflib.unified_diff(
                before_lines,
                after_lines,
                fromfile=f"a/{norm_path}" if orig_existed else "/dev/null",
                tofile=f"b/{norm_path}" if after_content is not None else "/dev/null",
                n=3,
            )
        )
        unified_text = "".join(diff_lines)
        added_cnt = sum(1 for ln in diff_lines if ln.startswith("+") and not ln.startswith("+++"))
        removed_cnt = sum(1 for ln in diff_lines if ln.startswith("-") and not ln.startswith("---"))
        change_type = "added" if not orig_existed else ("deleted" if after_content is None else "modified")
        self.file_diffs[norm_path] = {
            "file": norm_path,
            "change_type": change_type,
            "existed_before": orig_existed,
            "before_content": orig_before,
            "after_content": after_content,
            "unified_diff": unified_text,
            "added_lines": added_cnt,
            "removed_lines": removed_cnt,
        }

    def build_mutation_summary(self) -> dict[str, Any] | None:
        """Build a UI-ready summary of guarded file modifications, unified diffs, and rollback state."""
        if not self.changed_files and not self.file_diffs and not self.last_rollback:
            return None
        rolled_back = bool(
            self.status == "rolled_back"
            or (isinstance(self.last_rollback, dict) and self.last_rollback.get("rolled_back"))
        )
        diffs_list = [self.file_diffs[k] for k in sorted(self.file_diffs)]
        total_added = sum(int(d.get("added_lines", 0)) for d in diffs_list)
        total_removed = sum(int(d.get("removed_lines", 0)) for d in diffs_list)
        return {
            "tx_id": self.tx_id,
            "checkpoint_id": self.checkpoint.checkpoint_id if self.checkpoint is not None else None,
            "status": "rolled_back" if rolled_back else self.status,
            "state": self.last_state,
            "changed_files": list(self.changed_files),
            "file_diffs": diffs_list,
            "total_added_lines": total_added,
            "total_removed_lines": total_removed,
            "repair_attempts": self.repair_attempts,
            "rolled_back": rolled_back,
            "rollback": self.last_rollback,
            "verification": self.last_verification,
            "can_rollback": bool(diffs_list and not rolled_back),
        }

    def _record_changed_file(
        self, path: str | None = None, extra_files: list[str] | None = None
    ) -> None:
        if path and path not in self.changed_files:
            self.changed_files.append(path)
        for item in extra_files or []:
            if item and item not in self.changed_files:
                self.changed_files.append(item)
        for cp_file in self._checkpoint_changed_files():
            if cp_file and cp_file not in self.changed_files:
                self.changed_files.append(cp_file)

    def _fail_closed_checkpoint_disabled(self, operation_label: str) -> dict:
        """Fail closed when `config.enable_checkpoints=False` and an operation
        requires rollback protection."""
        err = {
            "code": "checkpoint_disabled",
            "message": (
                f"{operation_label} requires ENABLE_CHECKPOINTS=true because it can modify "
                "repository state and cannot be proven safely read-only without rollback protection"
            ),
        }
        report = (
            f"Repository operation ({operation_label}) was not started because checkpoints are "
            f"disabled and the operation requires rollback protection.\n\nResult: {FAILED}."
        )
        if self.changed_files:
            if not self.in_repair_loop:
                self.in_repair_loop = True
                self.repair_attempts = 0
            self.status = "in_repair"
        else:
            self.status = "checkpoint_disabled"
        self.last_state = FAILED
        self.last_verification = {"passed": False, "status": "checkpoint_disabled", "error": err}
        self.last_report = report
        return {
            "state": FAILED,
            "verification": self.last_verification,
            "report": report,
            "error": err,
        }

    def ensure_checkpoint(self, label: str = "") -> dict | None:
        """Acquire the per-repository transaction lock and create the required
        checkpoint before a protected mutation or command.
        Returns an error result dict if checkpoints are disabled or creation
        fails, or None on success."""
        if not self.config.enable_checkpoints:
            return self._fail_closed_checkpoint_disabled(label or self.label or "repository_modification")

        self._acquire_repo_lock()
        if self.checkpoint is not None:
            if not self.checkpoint.available or self.checkpoint.checkpoint_id is None:
                error = self.checkpoint.error or {
                    "code": "checkpoint_creation_failed",
                    "message": "unknown checkpoint error",
                }
                report = (
                    "Repository operation was not started because the required "
                    f"checkpoint could not be created ({error.get('code', 'checkpoint_creation_failed')}): "
                    f"{error.get('message', 'unknown checkpoint error')}\n\nResult: FAILED."
                )
                if not self.changed_files and not self.in_repair_loop:
                    self._release_repo_lock()
                return {
                    "state": FAILED,
                    "verification": {"passed": False, "status": "checkpoint_failed"},
                    "report": report,
                    "error": error,
                }
            return None

        self.checkpoint = Checkpoint.create(
            self.service,
            enabled=True,
            label=label or self.label or "repository_modification",
        )
        if not self.checkpoint.available or self.checkpoint.checkpoint_id is None:
            error = self.checkpoint.error or {
                "code": "checkpoint_creation_failed",
                "message": "unknown checkpoint error",
            }
            report = (
                "Repository operation was not started because the required "
                f"checkpoint could not be created ({error.get('code', 'checkpoint_creation_failed')}): "
                f"{error.get('message', 'unknown checkpoint error')}\n\nResult: FAILED."
            )
            self.status = "in_repair" if self.changed_files else "checkpoint_failed"
            if self.changed_files and not self.in_repair_loop:
                self.in_repair_loop = True
                self.repair_attempts = 0
            self.last_state = FAILED
            self.last_verification = {"passed": False, "status": "checkpoint_failed"}
            self.last_report = report
            if not self.changed_files and not self.in_repair_loop:
                self._release_repo_lock()
            return {
                "state": FAILED,
                "verification": self.last_verification,
                "report": report,
                "error": error,
            }
        return None

    def _apply_rollback_policy(self) -> tuple[str, dict | None]:
        """Execute the configured rollback policy and return `(rollback_note, rollback_dict)`."""
        rollback: dict | None = None
        if not self.config.enable_checkpoints:
            rollback_note = "\n\nAutomatic rollback is disabled, so changes made are left in place."
            self.status = "failed_unrepaired"
        elif self.config.rollback_on_repair_limit and self.checkpoint is not None:
            rollback = self.checkpoint.rollback()
            if rollback is not None and rollback.get("rolled_back"):
                rollback_note = (
                    "\n\nThe repository has been rolled back to its state before this task started "
                    f"(restored {len(rollback.get('restored_files', []))} file(s), "
                    f"removed {len(rollback.get('removed_files', []))} file(s))."
                )
                self.status = "rolled_back"
                self.changed_files.clear()
            elif rollback is not None:
                rollback_note = (
                    "\n\nRollback was attempted but did not complete cleanly for every file; "
                    "the repository may be left partially modified. See the errors for details: "
                    f"{rollback.get('errors')}"
                )
                self.status = "rollback_failed"
                self.preserve_checkpoint = True
            else:
                rollback_note = "\n\nNo active checkpoint was available to roll back."
                self.status = "failed_unrepaired"
        else:
            diff = (
                self.checkpoint.diff()
                if self.checkpoint is not None
                else {"changed": None}
            )
            summary = format_change_summary(diff)
            rollback_note = (
                f"\n\n{summary} (left in place: rollback-on-repair-limit is disabled.)"
                if summary
                else "\n\nChanges left in place (rollback-on-repair-limit is disabled)."
            )
            self.status = "failed_unrepaired"
        self.last_rollback = rollback
        return rollback_note, rollback

    def _handle_repair_limit_exhausted(
        self,
        verification: dict,
        progress_note: str = "",
        raw_write: dict | None = None,
    ) -> dict:
        """Finalize when actual repair attempts reached `config.max_repair_attempts`."""
        self.repair_limit_exhausted = True
        self.in_repair_loop = False
        self.termination_reason = "repair_limit_reached"
        state_label = terminal_state(verification, repair_limit_reached=True)
        rollback_note, rollback = self._apply_rollback_policy()
        try:
            refresh_summary = self.service.refresh()
        except (OSError, ValueError) as exc:
            refresh_summary = {"error": {"code": "index_refresh_failed", "message": str(exc)}}
        if isinstance(refresh_summary, dict) and "error" in refresh_summary:
            self.status = "rollback_failed"
            self.preserve_checkpoint = True
        self.last_refresh = refresh_summary
        report = (
            f"{verification_summary(verification)}\n\n"
            f"Verification is still failing after {self.repair_attempts} repair attempt(s) "
            f"(repair limit reached)."
            f"{progress_note}{rollback_note}\n\nResult: {state_label}."
        )
        if isinstance(refresh_summary, dict) and "error" in refresh_summary:
            report += f"\n\nRepository index refresh failed: {refresh_summary['error']}"
        bounded = _bound_report(report, min(self.config.max_observation_chars, 12000))
        self.last_state = state_label
        self.last_verification = verification
        self.last_report = bounded
        if not self.preserve_checkpoint and self.checkpoint is not None:
            self.checkpoint.discard()
        self._release_repo_lock()
        out: dict[str, Any] = {
            **(raw_write or {}),
            "state": state_label,
            "verification": verification,
            "report": bounded,
            "rollback": rollback,
            "refresh": refresh_summary,
            "repair_attempts": self.repair_attempts,
            "repair_limit_reached": True,
            "termination_reason": "repair_limit_reached",
        }
        if self.status == "rollback_failed":
            out["error"] = {
                "code": "rollback_failed",
                "message": "rollback or post-rollback index refresh did not complete cleanly",
                "rollback_errors": (rollback or {}).get("errors"),
                "refresh_error": (refresh_summary or {}).get("error") if isinstance(refresh_summary, dict) else None,
            }
        return out

    def _handle_unexpected_exception(self, exc: Exception) -> dict:
        rollback = None
        refresh_summary = None
        rollback_error = None
        if (
            self.checkpoint is not None
            and self.checkpoint.available
            and self.checkpoint.checkpoint_id is not None
        ):
            try:
                rollback = self.checkpoint.rollback()
                if rollback is None or not rollback.get("rolled_back"):
                    rollback_error = (
                        (rollback or {}).get("errors")
                        or rollback
                        or "rollback returned no status"
                    )
            except Exception as rollback_exc:
                rollback_error = f"{type(rollback_exc).__name__}: {rollback_exc}"
        try:
            refresh_summary = self.service.refresh()
            if isinstance(refresh_summary, dict) and "error" in refresh_summary and rollback_error is None:
                rollback_error = f"index refresh failed: {refresh_summary['error']}"
        except Exception as refresh_exc:
            if rollback_error is None:
                rollback_error = (
                    f"index refresh failed: {type(refresh_exc).__name__}: {refresh_exc}"
                )
        self.preserve_checkpoint = rollback_error is not None
        self.status = "rollback_failed" if self.preserve_checkpoint else "rolled_back"
        self.in_repair_loop = False
        self.termination_reason = f"exception:{type(exc).__name__}"
        report_body = (
            f"Repository operation stopped after an unexpected {type(exc).__name__}: {exc}. "
            "Rollback was attempted."
            if rollback_error is None
            else f"Repository operation stopped after an unexpected {type(exc).__name__}: {exc}. "
            f"Rollback or index refresh also failed: {rollback_error}"
        )
        report = f"{report_body}\n\nResult: FAILED."
        self.last_state = FAILED
        self.last_verification = {"passed": False, "status": "execution_error"}
        self.last_report = report
        self.last_rollback = rollback
        self.last_refresh = refresh_summary
        checkpoint_id = (
            self.checkpoint.checkpoint_id
            if (self.preserve_checkpoint and self.checkpoint is not None)
            else None
        )
        if not self.preserve_checkpoint and self.checkpoint is not None:
            self.checkpoint.discard()
        self._release_repo_lock()
        return {
            "state": FAILED,
            "verification": self.last_verification,
            "report": report,
            "rollback": rollback,
            "refresh": refresh_summary,
            "checkpoint_id": checkpoint_id,
            "error": {
                "code": "execution_error",
                "message": f"{type(exc).__name__}: {exc}",
                **({"rollback_error": str(rollback_error)} if rollback_error else {}),
            },
        }

    def execute_write(self, path: str, content: str, label: str = "") -> dict:
        """Run a guarded `write_file` with checkpoint creation, index refresh,
        scoped verification, bounded repair tracking, and rollback on limit."""
        if self.repair_limit_exhausted:
            return {
                "state": REPAIR_LIMIT_REACHED,
                "verification": self.last_verification or {"passed": False, "status": "failed"},
                "report": self.last_report
                or (
                    f"Repair limit ({self.config.max_repair_attempts}) was already reached.\n\n"
                    f"Result: {REPAIR_LIMIT_REACHED}."
                ),
                "error": {
                    "code": "repair_limit_reached",
                    "message": f"repair limit ({self.config.max_repair_attempts}) exhausted",
                },
            }

        cp_error = self.ensure_checkpoint(label=label or f"write_file:{path}")
        if cp_error is not None:
            return cp_error

        pre_existed = False
        pre_content: str | None = None
        try:
            root_dir = Path(self.service.index.root).resolve()
            cand_path = (Path(path) if Path(path).is_absolute() else root_dir / path).resolve()
            if is_within(root_dir, cand_path) and cand_path.is_file() and not cand_path.is_symlink():
                pre_existed = True
                pre_content = cand_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            pass

        try:
            raw = self.service.write_file(path, content)
            if "error" in raw:
                self.last_state = FAILED
                self.last_verification = {
                    "passed": False,
                    "status": "failed",
                    "error": raw["error"],
                }
                if self.changed_files:
                    if not self.in_repair_loop:
                        self.in_repair_loop = True
                        self.repair_attempts = 0
                    self.status = "in_repair"
                else:
                    self.status = "failed"
                    if self.checkpoint is not None:
                        self.checkpoint.discard()
                        self.checkpoint = None
                    self._release_repo_lock()
                return raw

            written_path = str(raw.get("file") or path)
            self._record_changed_file(written_path)
            self._record_file_diff(
                written_path,
                before_content=pre_content,
                after_content=content,
                existed_before=pre_existed,
            )
            if self.in_repair_loop:
                self.repair_attempts += 1

            verification = run_scoped_verification(
                self.service, self.config, self.changed_files
            )
            if verification["passed"]:
                state_label = terminal_state(verification, repair_limit_reached=False)
                diff = (
                    self.checkpoint.diff()
                    if self.checkpoint is not None
                    and self.checkpoint.checkpoint_id is not None
                    else {
                        "changed": True,
                        "added": [written_path] if raw.get("created") else [],
                        "modified": [] if raw.get("created") else [written_path],
                        "deleted": [],
                    }
                )
                summary = format_change_summary(diff)
                repair_note = (
                    f"\n\n(Verification passed after {self.repair_attempts} repair attempt(s).)"
                    if self.in_repair_loop and self.repair_attempts > 0
                    else ""
                )
                report = (
                    f"Wrote {written_path} ({raw.get('line_count', 0)} lines).\n\n"
                    f"{verification_summary(verification)}{repair_note}\n\n"
                    f"{summary}\n\nResult: {state_label}."
                )
                bounded = _bound_report(
                    report, min(self.config.max_observation_chars, 12000)
                )
                if state_label in (SUCCESS, NO_TESTS):
                    self.in_repair_loop = False
                    self.last_signature = None
                    self.status = "passed"
                else:
                    self.status = "in_repair" if self.in_repair_loop else "unavailable"
                self.last_state = state_label
                self.last_verification = verification
                self.last_report = bounded
                return {
                    **raw,
                    "state": state_label,
                    "verification": verification,
                    "report": bounded,
                    "changed_files": list(self.changed_files),
                    "repair_attempts": self.repair_attempts,
                }

            signature = verification_signature(verification)
            progress_note = ""
            if (
                self.in_repair_loop
                and self.last_signature is not None
                and signature == self.last_signature
            ):
                progress_note = (
                    "\n\nNote: this repair attempt produced the same verification failure as the "
                    "previous attempt - the repair does not appear to be making progress."
                )
            self.last_signature = signature
            if not self.in_repair_loop:
                self.in_repair_loop = True
                self.repair_attempts = 0

            if self.repair_attempts >= self.config.max_repair_attempts:
                return self._handle_repair_limit_exhausted(
                    verification, progress_note, raw_write=raw
                )

            remaining = self.config.max_repair_attempts - self.repair_attempts
            state_label = terminal_state(verification, repair_limit_reached=False)
            report = (
                f"Wrote {written_path} ({raw.get('line_count', 0)} lines), but scoped verification failed.\n\n"
                f"{verification_summary(verification)}{progress_note}\n\n"
                f"Repair attempts used: {self.repair_attempts}/{self.config.max_repair_attempts} "
                f"({remaining} remaining). Inspect the error and use write_file to repair {written_path}.\n\n"
                f"Result: {state_label}."
            )
            bounded = _bound_report(
                report, min(self.config.max_observation_chars, 12000)
            )
            self.status = "in_repair"
            self.last_state = state_label
            self.last_verification = verification
            self.last_report = bounded
            return {
                **raw,
                "state": state_label,
                "verification": verification,
                "report": bounded,
                "changed_files": list(self.changed_files),
                "repair_attempts": self.repair_attempts,
                "repair_attempts_remaining": remaining,
            }
        except Exception as exc:
            return self._handle_unexpected_exception(exc)

    def _rollback_unauthorized_command_mutation(
        self,
        command_label: str,
        step_checkpoint: Checkpoint,
        diff: dict,
    ) -> dict:
        """Roll back repository file changes caused by a command when the
        invocation lacks `WRITE` capability (`execute=True, write=False`)."""
        mutated_files = (
            list(diff.get("added", []))
            + list(diff.get("modified", []))
            + list(diff.get("deleted", []))
            + list(diff.get("added_directories", []))
            + list(diff.get("removed_directories", []))
        )
        rollback = step_checkpoint.rollback()
        try:
            refresh = self.service.refresh()
        except (OSError, ValueError) as exc:
            refresh = {"error": str(exc)}
        rolled_back_ok = bool(
            rollback
            and rollback.get("rolled_back")
            and not (isinstance(refresh, dict) and "error" in refresh)
        )
        if rolled_back_ok:
            step_checkpoint.discard()
            self.status = "rolled_back"
        else:
            self.preserve_checkpoint = True
            self.status = "rollback_failed"
        self.last_state = FAILED
        self.last_rollback = rollback
        self.last_refresh = refresh
        rb_detail = (
            f"Repository file changes ({', '.join(mutated_files[:5])}) were rolled back to the "
            "pre-command checkpoint."
            if rolled_back_ok
            else f"Rollback was attempted for ({', '.join(mutated_files[:5])}), but errors occurred: "
            f"{(rollback or {}).get('errors') or (refresh or {}).get('error')}"
        )
        external_caveat = (
            " Note: a filesystem checkpoint only restores tracked repository files and cannot "
            "undo external side effects outside the repository."
        )
        report = (
            f"Command {command_label!r} modified repository files while running in read-only "
            f"validation mode (WRITE capability was not granted). {rb_detail}{external_caveat}\n\n"
            f"Result: {FAILED}."
        )
        self.last_verification = {"passed": False, "status": "unauthorized_mutation"}
        self.last_report = report
        if (self.checkpoint is None or self.checkpoint.checkpoint_id is None) and not self.in_repair_loop and not self.changed_files:
            self._release_repo_lock()
        return {
            "command": command_label,
            "state": FAILED,
            "verification": self.last_verification,
            "report": report,
            "rollback": rollback,
            "refresh": refresh,
            "error": {
                "code": "unauthorized_command_mutation",
                "message": (
                    f"command modified repository files ({', '.join(mutated_files[:5])}) without WRITE "
                    f"capability; {rb_detail}{external_caveat}"
                ),
                "changed_files": mutated_files,
                "rolled_back": rolled_back_ok,
            },
        }

    def execute_verify(self, *, write_allowed: bool = False) -> dict:
        """Run the configured repository verification command with command-safety
        validation and checkpoint protection against unexpected file mutations."""
        if self.repair_limit_exhausted:
            return {
                "state": REPAIR_LIMIT_REACHED,
                "verification": self.last_verification or {"passed": False, "status": "failed"},
                "report": self.last_report
                or (
                    f"Repair limit ({self.config.max_repair_attempts}) was already reached.\n\n"
                    f"Result: {REPAIR_LIMIT_REACHED}."
                ),
            }

        cmd = self.config.verification_command.strip()
        if cmd:
            safety_err = validate_command_safety(
                cmd, write_allowed=write_allowed, repo_root=self._repo_root()
            )
            if safety_err is not None:
                report = (
                    f"Configured verification command {cmd!r} was blocked by command safety policy "
                    f"({safety_err['error'].get('code')}): {safety_err['error'].get('message')}\n\n"
                    f"Result: {FAILED}."
                )
                if self.changed_files:
                    if not self.in_repair_loop:
                        self.in_repair_loop = True
                        self.repair_attempts = 0
                    self.status = "in_repair"
                else:
                    self.status = "blocked"
                self.last_state = FAILED
                self.last_verification = {"passed": False, "status": "blocked", **safety_err}
                self.last_report = report
                return {
                    "state": FAILED,
                    "verification": self.last_verification,
                    "report": report,
                    **safety_err,
                }

            if not self.config.enable_checkpoints and not _is_provably_readonly_command(
                cmd, repo_root=self._repo_root()
            ):
                return self._fail_closed_checkpoint_disabled(f"verify_repository:{cmd[:120]}")

        self._acquire_repo_lock()
        if write_allowed and self.config.enable_checkpoints:
            cp_error = self.ensure_checkpoint(label="verify_repository")
            if cp_error is not None:
                return cp_error

        has_configured_cmd = bool(cmd)
        step_cp: Checkpoint | None = None
        if has_configured_cmd and self.config.enable_checkpoints:
            # Always create a step checkpoint before running a configured verification
            # command so command-induced file mutations are detected deterministically.
            step_cp = Checkpoint.create(
                self.service,
                enabled=True,
                label="verify_repository_step",
            )
            if not step_cp.available or step_cp.checkpoint_id is None:
                error = step_cp.error or {
                    "code": "checkpoint_creation_failed",
                    "message": "could not create pre-verification safety checkpoint",
                }
                report = (
                    "Repository verification was not started because the required safety "
                    f"checkpoint could not be created ({error.get('code')}): {error.get('message')}\n\n"
                    f"Result: {FAILED}."
                )
                if self.changed_files:
                    if not self.in_repair_loop:
                        self.in_repair_loop = True
                        self.repair_attempts = 0
                    self.status = "in_repair"
                else:
                    self.status = "checkpoint_failed"
                self.last_state = FAILED
                self.last_verification = {
                    "passed": False,
                    "status": "checkpoint_failed",
                    "error": error,
                }
                self.last_report = report
                if (self.checkpoint is None or self.checkpoint.checkpoint_id is None) and not self.in_repair_loop and not self.changed_files:
                    self._release_repo_lock()
                return {
                    "state": FAILED,
                    "verification": self.last_verification,
                    "report": report,
                    "error": error,
                }

        try:
            verification = run_verification(
                self.service, self.config, write_allowed=write_allowed
            )
            step_diff = (
                step_cp.diff()
                if step_cp is not None and step_cp.checkpoint_id is not None
                else {"changed": False}
            )
            cmd_mutated = bool(
                isinstance(step_diff, dict)
                and ("error" in step_diff or step_diff.get("changed"))
            )
            if cmd_mutated and not write_allowed and step_cp is not None:
                return self._rollback_unauthorized_command_mutation(
                    self.config.verification_command, step_cp, step_diff
                )
            if step_cp is not None:
                step_cp.discard()

            if cmd_mutated and isinstance(step_diff, dict):
                self._record_changed_file(
                    extra_files=(
                        list(step_diff.get("added", []))
                        + list(step_diff.get("modified", []))
                        + list(step_diff.get("deleted", []))
                        + list(step_diff.get("added_directories", []))
                        + list(step_diff.get("removed_directories", []))
                    )
                )

            state_label = terminal_state(verification, repair_limit_reached=False)
            try:
                refresh = self.service.refresh()
            except (OSError, ValueError) as exc:
                refresh = {"error": {"code": "index_refresh_failed", "message": str(exc)}}
            self.last_refresh = refresh

            if isinstance(refresh, dict) and "error" in refresh:
                state_label = FAILED
                verification = {
                    **verification,
                    "passed": False,
                    "status": "failed",
                    "refresh_error": refresh["error"],
                }

            if (cmd_mutated or self.changed_files) and verification.get("passed"):
                scoped_ver = run_scoped_verification(
                    self.service, self.config, self.changed_files
                )
                if not scoped_ver.get("passed"):
                    verification = scoped_ver
                    state_label = terminal_state(
                        verification, repair_limit_reached=False
                    )

            if self.changed_files:
                if verification.get("passed") and state_label in (SUCCESS, NO_TESTS):
                    self.in_repair_loop = False
                    self.last_signature = None
                    self.status = "passed"
                elif state_label == UNAVAILABLE:
                    # Do not clear an active repair loop or leave a stale "passed"
                    # status when the configured verification command was unavailable.
                    self.status = "in_repair" if self.in_repair_loop else "unavailable"
                else:
                    signature = verification_signature(verification)
                    progress_note = ""
                    if (
                        self.in_repair_loop
                        and self.last_signature is not None
                        and signature == self.last_signature
                    ):
                        progress_note = (
                            "\n\nNote: this repair attempt produced the same verification failure as the "
                            "previous attempt - the repair does not appear to be making progress."
                        )
                    self.last_signature = signature
                    if not self.in_repair_loop:
                        self.in_repair_loop = True
                        self.repair_attempts = 0
                    elif cmd_mutated:
                        self.repair_attempts += 1
                    if self.repair_attempts >= self.config.max_repair_attempts:
                        return self._handle_repair_limit_exhausted(
                            verification, progress_note
                        )
                    self.status = "in_repair"
            else:
                if state_label in (SUCCESS, NO_TESTS):
                    self.status = "passed"
                elif state_label == UNAVAILABLE:
                    self.status = "unavailable"
                elif state_label == TIMEOUT:
                    self.status = "timeout"
                else:
                    self.status = "failed"

            report = _bound_report(
                f"{verification_summary(verification)}\n\nResult: {state_label}.",
                min(self.config.max_observation_chars, 12000),
            )
            if isinstance(refresh, dict) and "error" in refresh:
                report += f"\n\nRepository index refresh failed: {refresh['error']}"
            self.last_state = state_label
            self.last_verification = verification
            self.last_report = report
            if not self.changed_files and not self.in_repair_loop:
                if self.checkpoint is not None:
                    self.checkpoint.discard()
                    self.checkpoint = None
                self._release_repo_lock()
            out: dict[str, Any] = {
                "state": state_label,
                "verification": verification,
                "report": report,
                "refresh": refresh,
            }
            if isinstance(refresh, dict) and "error" in refresh:
                out["error"] = refresh["error"]
            return out
        except Exception as exc:
            if step_cp is not None:
                step_cp.discard()
            return self._handle_unexpected_exception(exc)

    def execute_command(
        self,
        command: str | Sequence[str],
        timeout: int = 60,
        *,
        write_allowed: bool = False,
    ) -> dict:
        """Run a guarded repository command with pre-execution command policy,
        mandatory checkpoint establishment, unauthorized-mutation rollback when
        `write_allowed` is False, and post-command diff/verification tracking
        when `write_allowed` is True."""
        cmd_display = _display_cmd_str(command)
        safety_err = validate_command_safety(
            command, write_allowed=write_allowed, repo_root=self._repo_root()
        )
        if safety_err is not None:
            report = (
                f"Command {cmd_display!r} was blocked by command safety policy "
                f"({safety_err['error'].get('code')}): {safety_err['error'].get('message')}\n\n"
                f"Result: {FAILED}."
            )
            if self.changed_files:
                if not self.in_repair_loop:
                    self.in_repair_loop = True
                    self.repair_attempts = 0
                self.status = "in_repair"
            else:
                self.status = "blocked"
            self.last_state = FAILED
            self.last_verification = {"passed": False, "status": "blocked", **safety_err}
            self.last_report = report
            return {
                "command": cmd_display,
                "state": FAILED,
                "verification": self.last_verification,
                "report": report,
                **safety_err,
            }

        # If checkpoints are disabled in configuration, block any command that
        # is not statically proven to be read-only because command-induced
        # mutations cannot be detected or rolled back without a checkpoint.
        if not self.config.enable_checkpoints and not _is_provably_readonly_command(
            command, repo_root=self._repo_root()
        ):
            return self._fail_closed_checkpoint_disabled(f"run_command:{cmd_display[:120]}")

        if self.repair_limit_exhausted:
            return {
                "state": REPAIR_LIMIT_REACHED,
                "verification": self.last_verification or {"passed": False, "status": "failed"},
                "report": self.last_report
                or (
                    f"Repair limit ({self.config.max_repair_attempts}) was already reached.\n\n"
                    f"Result: {REPAIR_LIMIT_REACHED}."
                ),
                "error": {
                    "code": "repair_limit_reached",
                    "message": f"repair limit ({self.config.max_repair_attempts}) exhausted",
                },
            }

        self._acquire_repo_lock()
        if write_allowed and self.config.enable_checkpoints:
            cp_error = self.ensure_checkpoint(label=f"run_command:{cmd_display[:120]}")
            if cp_error is not None:
                return cp_error

        # Establish a step checkpoint before running the command (whenever
        # checkpoints are enabled) so any files modified by this command
        # invocation are detected deterministically and rolled back if
        # write_allowed is False.
        step_cp: Checkpoint | None = None
        if self.config.enable_checkpoints:
            step_cp = Checkpoint.create(
                self.service,
                enabled=True,
                label=f"cmd_step:{cmd_display[:120]}",
            )
            if not step_cp.available or step_cp.checkpoint_id is None:
                error = step_cp.error or {
                    "code": "checkpoint_creation_failed",
                    "message": "could not create pre-command safety checkpoint",
                }
                report = (
                    "Command execution was not started because the required safety "
                    f"checkpoint could not be created ({error.get('code')}): {error.get('message')}\n\n"
                    f"Result: {FAILED}."
                )
                if self.changed_files:
                    if not self.in_repair_loop:
                        self.in_repair_loop = True
                        self.repair_attempts = 0
                    self.status = "in_repair"
                else:
                    self.status = "checkpoint_failed"
                self.last_state = FAILED
                self.last_verification = {"passed": False, "status": "checkpoint_failed", "error": error}
                self.last_report = report
                if not self.changed_files and not self.in_repair_loop:
                    if self.checkpoint is not None:
                        self.checkpoint.discard()
                        self.checkpoint = None
                    self._release_repo_lock()
                return {
                    "state": FAILED,
                    "verification": self.last_verification,
                    "report": report,
                    "error": error,
                }

        try:
            raw = self.service.run_command(command, timeout=timeout)
            step_diff = (
                step_cp.diff()
                if step_cp is not None and step_cp.available and step_cp.checkpoint_id is not None
                else {"changed": False}
            )
            cmd_mutated = bool(
                isinstance(step_diff, dict)
                and ("error" in step_diff or step_diff.get("changed"))
            )

            # 1. Read-only validation mode (`write_allowed=False`): any file
            # modification on disk is unauthorized and must be rolled back immediately.
            if cmd_mutated and not write_allowed and step_cp is not None:
                return self._rollback_unauthorized_command_mutation(
                    cmd_display, step_cp, step_diff
                )

            if step_cp is not None:
                step_cp.discard()

            try:
                refresh = self.service.refresh()
            except (OSError, ValueError) as exc:
                refresh = {"error": {"code": "index_refresh_failed", "message": str(exc)}}
            self.last_refresh = refresh

            if "error" in raw:
                err_ver = {
                    "passed": False,
                    "status": "failed",
                    "command": cmd_display,
                    "exit_code": None,
                    "timed_out": False,
                    "stdout": "",
                    "stderr": raw["error"].get("message", ""),
                    "error": raw["error"],
                }
                if self.changed_files:
                    if not self.in_repair_loop:
                        self.in_repair_loop = True
                        self.repair_attempts = 0
                    self.status = "in_repair"
                else:
                    self.status = "failed"
                self.last_state = FAILED
                self.last_verification = err_ver
                self.last_report = _bound_report(
                    f"{verification_summary(err_ver)}\n\nResult: {FAILED}.",
                    min(self.config.max_observation_chars, 12000),
                )
                if not self.changed_files and not self.in_repair_loop:
                    if self.checkpoint is not None:
                        self.checkpoint.discard()
                        self.checkpoint = None
                    self._release_repo_lock()
                return {
                    **raw,
                    "state": FAILED,
                    "verification": err_ver,
                    "report": self.last_report,
                }

            cmd_passed = (
                not raw.get("timed_out")
                and raw.get("exit_code") == 0
                and not (isinstance(refresh, dict) and "error" in refresh)
            )

            # 2. Write-enabled mode where the command modified files on disk:
            # record the modified files, run scoped verification, and update
            # transaction/repair status so command-induced changes are never left
            # unverified under a stale `status == "passed"`.
            if cmd_mutated and isinstance(step_diff, dict):
                mutated_files = (
                    list(step_diff.get("added", []))
                    + list(step_diff.get("modified", []))
                    + list(step_diff.get("deleted", []))
                    + list(step_diff.get("added_directories", []))
                    + list(step_diff.get("removed_directories", []))
                )
                self._record_changed_file(extra_files=mutated_files)
                if self.in_repair_loop:
                    self.repair_attempts += 1

                scoped_ver = run_scoped_verification(
                    self.service, self.config, self.changed_files
                )
                if cmd_passed and scoped_ver.get("passed"):
                    state_label = terminal_state(scoped_ver, repair_limit_reached=False)
                    diff_for_summary = (
                        self.checkpoint.diff()
                        if self.checkpoint is not None
                        and self.checkpoint.checkpoint_id is not None
                        else step_diff
                    )
                    summary = format_change_summary(diff_for_summary)
                    report = (
                        f"Command {cmd_display!r} exited 0 and modified {len(mutated_files)} file(s).\n\n"
                        f"{verification_summary(scoped_ver)}\n\n{summary}\n\nResult: {state_label}."
                    )
                    bounded = _bound_report(
                        report, min(self.config.max_observation_chars, 12000)
                    )
                    self.in_repair_loop = False
                    self.last_signature = None
                    self.status = "passed"
                    self.last_state = state_label
                    self.last_verification = scoped_ver
                    self.last_report = bounded
                    return {
                        **raw,
                        "state": state_label,
                        "verification": scoped_ver,
                        "report": bounded,
                        "changed_files": list(self.changed_files),
                        "repair_attempts": self.repair_attempts,
                    }

                eff_ver = (
                    scoped_ver
                    if not scoped_ver.get("passed")
                    else {
                        "passed": False,
                        "status": "timeout" if raw.get("timed_out") else "failed",
                        "command": raw.get("command", cmd_display),
                        "exit_code": raw.get("exit_code"),
                        "timed_out": bool(raw.get("timed_out")),
                        "stdout": raw.get("stdout", ""),
                        "stderr": raw.get("stderr", ""),
                    }
                )
                signature = verification_signature(eff_ver)
                progress_note = ""
                if (
                    self.in_repair_loop
                    and self.last_signature is not None
                    and signature == self.last_signature
                ):
                    progress_note = (
                        "\n\nNote: this repair attempt produced the same verification failure as the "
                        "previous attempt - the repair does not appear to be making progress."
                    )
                self.last_signature = signature
                if not self.in_repair_loop:
                    self.in_repair_loop = True
                    self.repair_attempts = 0

                if self.repair_attempts >= self.config.max_repair_attempts:
                    return self._handle_repair_limit_exhausted(
                        eff_ver, progress_note, raw_write=raw
                    )

                remaining = self.config.max_repair_attempts - self.repair_attempts
                state_label = terminal_state(eff_ver, repair_limit_reached=False)
                report = (
                    f"Command {cmd_display!r} modified files ({', '.join(mutated_files[:5])}), "
                    f"but verification failed.\n\n{verification_summary(eff_ver)}{progress_note}\n\n"
                    f"Repair attempts used: {self.repair_attempts}/{self.config.max_repair_attempts} "
                    f"({remaining} remaining).\n\nResult: {state_label}."
                )
                if isinstance(refresh, dict) and "error" in refresh:
                    report += f"\n\nRepository index refresh failed: {refresh['error']}"
                bounded = _bound_report(
                    report, min(self.config.max_observation_chars, 12000)
                )
                self.status = "in_repair"
                self.last_state = state_label
                self.last_verification = eff_ver
                self.last_report = bounded
                return {
                    **raw,
                    "state": state_label,
                    "verification": eff_ver,
                    "report": bounded,
                    "changed_files": list(self.changed_files),
                    "repair_attempts": self.repair_attempts,
                    "repair_attempts_remaining": remaining,
                }

            # 3. Command did not modify files on disk: record terminal state and,
            # if the command failed/timed out after earlier file changes,
            # invalidate any prior `status == "passed"` and enter repair.
            if not cmd_passed:
                cmd_ver = {
                    "passed": False,
                    "status": "timeout" if raw.get("timed_out") else "failed",
                    "command": raw.get("command", cmd_display),
                    "exit_code": raw.get("exit_code"),
                    "timed_out": bool(raw.get("timed_out")),
                    "stdout": raw.get("stdout", ""),
                    "stderr": raw.get("stderr", ""),
                }
                state_label = (
                    FAILED
                    if isinstance(refresh, dict) and "error" in refresh
                    else terminal_state(cmd_ver, repair_limit_reached=False)
                )
                if self.changed_files:
                    if not self.in_repair_loop:
                        self.in_repair_loop = True
                        self.repair_attempts = 0
                    self.status = "in_repair"
                else:
                    self.status = "timeout" if state_label == TIMEOUT else "failed"
                self.last_state = state_label
                self.last_verification = cmd_ver
                self.last_report = _bound_report(
                    f"{verification_summary(cmd_ver)}\n\nResult: {state_label}.",
                    min(self.config.max_observation_chars, 12000),
                )
                raw = {
                    **raw,
                    "state": state_label,
                    "verification": cmd_ver,
                    "report": self.last_report,
                }
            elif not self.in_repair_loop:
                cmd_ver = {
                    "passed": True,
                    "status": "passed",
                    "command": raw.get("command", cmd_display),
                    "exit_code": raw.get("exit_code"),
                    "timed_out": False,
                    "stdout": raw.get("stdout", ""),
                    "stderr": raw.get("stderr", ""),
                }
                self.last_state = SUCCESS
                self.last_verification = cmd_ver
                if not self.changed_files:
                    self.status = "passed"
                raw = {
                    **raw,
                    "state": SUCCESS,
                }

            if isinstance(refresh, dict) and "error" in refresh:
                raw = {**raw, "state": FAILED, "refresh_error": refresh["error"], "error": refresh["error"]}

            if not self.changed_files and not self.in_repair_loop:
                if self.checkpoint is not None:
                    self.checkpoint.discard()
                    self.checkpoint = None
                self._release_repo_lock()
            return raw
        except Exception as exc:
            if step_cp is not None:
                step_cp.discard()
            return self._handle_unexpected_exception(exc)

    def finalize_if_unrepaired(
        self,
        termination_reason: str = "run ended with repairs still outstanding",
    ) -> tuple[dict | None, dict | None]:
        """Ensure failed, unrepaired changes are rolled back on termination
        without ever rolling back changes whose verification already passed.

        Distinguishes actual repair-limit exhaustion (`REPAIR_LIMIT_REACHED`)
        from early termination during an active repair loop (`FAILED` /
        `TIMEOUT` with `termination_reason` reported accurately).
        """
        if (
            not self.in_repair_loop
            or self.status == "passed"
            or self.repair_limit_exhausted
        ):
            self.cleanup()
            return None, None

        self._acquire_repo_lock()
        verification = self.last_verification or {"passed": False, "status": "failed"}
        if self.repair_attempts >= self.config.max_repair_attempts:
            result = self._handle_repair_limit_exhausted(verification, "")
            v_info = {
                "tool": "write_file",
                "state": result["state"],
                "status": verification.get("status"),
                "report": result["report"],
                "rollback": result.get("rollback"),
                "termination_reason": "repair_limit_reached",
                "repair_limit_reached": True,
                **({"error": result["error"]} if "error" in result else {}),
            }
            f_info = {
                "tool": "write_file",
                "report": result["report"],
                "termination_reason": "repair_limit_reached",
                "repair_limit_reached": True,
                **({"error": result["error"]} if "error" in result else {}),
            }
            return v_info, f_info

        # Terminated before the repair limit was exhausted (e.g. step limit,
        # model failure, no-progress stop, or early conclusion while repairs
        # remained outstanding). Preserve the rollback policy, but report the
        # real termination reason and non-exhausted state (`FAILED` / `TIMEOUT`).
        self.in_repair_loop = False
        self.termination_reason = termination_reason
        state_label = terminal_state(verification, repair_limit_reached=False)
        rollback_note, rollback = self._apply_rollback_policy()
        try:
            refresh_summary = self.service.refresh()
        except (OSError, ValueError) as exc:
            refresh_summary = {"error": {"code": "index_refresh_failed", "message": str(exc)}}
        if isinstance(refresh_summary, dict) and "error" in refresh_summary:
            self.status = "rollback_failed"
            self.preserve_checkpoint = True
            state_label = FAILED
        self.last_refresh = refresh_summary
        report = (
            f"{verification_summary(verification)}\n\n"
            f"Verification remained failing ({self.repair_attempts}/{self.config.max_repair_attempts} "
            f"repair attempt(s) used) when the run ended ({termination_reason}); "
            f"repairs were not completed (repair limit was not reached)."
            f"{rollback_note}\n\nResult: {state_label}."
        )
        if isinstance(refresh_summary, dict) and "error" in refresh_summary:
            report += f"\n\nRepository index refresh failed: {refresh_summary['error']}"
        bounded = _bound_report(report, min(self.config.max_observation_chars, 12000))
        self.last_state = state_label
        self.last_verification = verification
        self.last_report = bounded
        self.cleanup()
        rb_err = (
            {
                "code": "rollback_failed",
                "message": "rollback or post-rollback index refresh did not complete cleanly",
                "rollback_errors": (rollback or {}).get("errors"),
            }
            if self.status == "rollback_failed"
            else None
        )
        v_info = {
            "tool": "write_file",
            "state": state_label,
            "status": verification.get("status"),
            "report": bounded,
            "rollback": rollback,
            "termination_reason": termination_reason,
            "repair_limit_reached": False,
            **({"error": rb_err} if rb_err else {}),
        }
        f_info = {
            "tool": "write_file",
            "report": bounded,
            "termination_reason": termination_reason,
            "repair_limit_reached": False,
            **({"error": rb_err} if rb_err else {}),
        }
        return v_info, f_info

    def cleanup(self) -> None:
        """Discard the active checkpoint when no longer needed and release the
        per-repository transaction lock."""
        try:
            if not self.preserve_checkpoint and self.checkpoint is not None:
                self.checkpoint.discard()
        finally:
            self._release_repo_lock()


# ------------------------------------------------------------------
# Per-invocation ContextVar binding for tool execution isolation
# ------------------------------------------------------------------


class _ActiveExecutionScope:
    def __init__(
        self,
        transaction: MutationTransaction,
        write_allowed: bool = False,
        execute_allowed: bool = False,
        intent: dict | None = None,
    ):
        self.transaction = transaction
        self.write_allowed = write_allowed
        self.execute_allowed = execute_allowed
        self.intent = intent


_ACTIVE_SCOPE: contextvars.ContextVar[_ActiveExecutionScope | None] = (
    contextvars.ContextVar("ari_active_mutation_scope", default=None)
)


def activate_transaction_scope(
    transaction: MutationTransaction,
    *,
    write_allowed: bool = False,
    execute_allowed: bool = False,
    intent: dict | None = None,
) -> contextvars.Token:
    """Bind `transaction`, capability grants (`write_allowed`, `execute_allowed`),
    and `intent` (`sources`) to the current execution context."""
    set_current_thread_tx_id(transaction.tx_id)
    return _ACTIVE_SCOPE.set(
        _ActiveExecutionScope(
            transaction,
            write_allowed=write_allowed,
            execute_allowed=execute_allowed,
            intent=intent,
        )
    )


def reset_transaction_scope(token: contextvars.Token) -> None:
    """Restore the previous execution context scope."""
    try:
        _ACTIVE_SCOPE.reset(token)
    except Exception:
        pass
    finally:
        scope = _ACTIVE_SCOPE.get()
        set_current_thread_tx_id(scope.transaction.tx_id if scope is not None else None)


def get_active_execution_scope() -> _ActiveExecutionScope | None:
    """Return the active `_ActiveExecutionScope` if bound, else `None`."""
    return _ACTIVE_SCOPE.get()


def get_active_transaction_scope(
    service: RepoService,
    config: AgentConfig,
    label: str = "",
) -> tuple[MutationTransaction, bool]:
    """Return `(transaction, write_allowed)` for the active invocation scope,
    creating an isolated local transaction if called outside a graph node."""
    scope = _ACTIVE_SCOPE.get()
    if scope is not None:
        return scope.transaction, scope.write_allowed
    return MutationTransaction(service, config, label=label), False
