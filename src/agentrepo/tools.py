"""
The agent-facing repository tools.

Each tool is a plain function that delegates to one RepoService method and
returns a JSON-serialisable dict (or {"error": {...}}). The docstrings are
written for the LLM agent: they describe what each tool is for and what it
returns.

    tools = make_tools(service)
    tools["find_symbol"]("Entity.save")

make_tools() returns a dict {tool name: function} for the ten read-only
repository tools plus write_file. make_verification_tools() returns
run_command. Capability gating and transactional safeguards are enforced by
the agent orchestration layer around these tools.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from .service import RepoService


def make_tools(service: RepoService) -> dict[str, Callable[..., dict]]:
    def search_repository(query: str, k: int = 5) -> dict:
        """CONTENT search over the indexed Python code chunks (words, identifiers, code text).
        Returns up to k ranked chunks with symbol, file, line range, score and a short snippet.
        It does NOT list files and does not prove a file exists - use find_files for that."""
        return service.search_repository(query, k)

    def find_files(pattern: str, limit: int = 50) -> dict:
        """FILE/PATH search: which files exist in the repository (any file type). Matches the
        pattern against repository-relative paths only, never file contents. Plain text is a
        case-insensitive substring ('README'); globs work too ('*.md', 'docs/*.rst').
        Returns repo-relative paths; read one with get_file."""
        return service.find_files(pattern, limit)

    def find_symbol(query: str, kind: str | None = None, limit: int = 20) -> dict:
        """Look up a symbol by fully-qualified id ('shop.models.Entity.save'), by 'Class.method',
        or by short name ('save'). Returns every match; `ambiguous` is true when there are several.
        kind optionally filters: module, class, function or method."""
        return service.find_symbol(query, kind, limit)

    def find_callers(symbol: str, limit: int = 50) -> dict:
        """Who calls this function/method/class. `callers` are definite (resolved) call sites;
        `possible_callers` are unresolved attribute calls with the same name, which may or may not
        target this symbol."""
        return service.find_callers(symbol, limit)

    def find_callees(symbol: str) -> dict:
        """What this symbol calls. `callees` are resolved repository symbols, `possible_callees`
        are ambiguous obj.method() calls, `external_calls` are stdlib/third-party/unresolved names."""
        return service.find_callees(symbol)

    def find_importers(target: str) -> dict:
        """Which repository modules import the given module (dotted name like 'shop.utils' or a file
        path like 'shop/utils.py'). Returns importing module, file and line."""
        return service.find_importers(target)

    def get_file(path: str, start_line: int | None = None, end_line: int | None = None) -> dict:
        """Read a file in the repository by a KNOWN repository-relative path (any file type), optionally a 1-based inclusive
        line range. Paths outside the repository are rejected. Long files are truncated."""
        return service.get_file(path, start_line, end_line)

    def get_symbol(symbol: str) -> dict:
        """Get the source code and location of one function, method or class (not a whole module)."""
        return service.get_symbol(symbol)

    def get_dependencies(target: str) -> dict:
        """What a symbol, module or file depends on and what depends on it (imports, inheritance,
        calls). The `legend` field explains each key in the result."""
        return service.get_dependencies(target)

    def get_ast(target: str) -> dict:
        """Structural (AST-derived) view of a Python symbol, module or file: node type, child
        definitions, calls with how well each resolved, imports, decorators, bases, and (for
        functions) argument names. Python only; use get_symbol/get_file to read the source text."""
        return service.get_ast(target)

    def write_file(path: str, content: str) -> dict:
        """Create a new file or overwrite an existing file's full contents at a repository-relative
        path (creates parent directories as needed). Paths outside the repository are rejected.
        This replaces the entire file, so read it first with get_file when modifying it."""
        return service.write_file(path, content)

    return {f.__name__: f for f in (
        search_repository, find_files, find_symbol, find_callers, find_callees, find_importers,
        get_file, get_symbol, get_dependencies, get_ast, write_file,
    )}


def make_verification_tools(service: RepoService) -> dict[str, Callable[..., dict]]:
    """The command-execution tool (run_command), kept separate from make_tools()
    so the eleven read/write functions in make_tools() remain unchanged."""

    def run_command(command: str, timeout: int = 60) -> dict:
        """Run a command with the repository root as its working directory (no
        way to point it elsewhere), to check whether an implementation works
        (e.g. running a test suite). Returns stdout, stderr and the exit code;
        a command that runs too long is killed and reported as timed out."""
        from .agent.safety import get_active_execution_scope, validate_command_safety

        scope = get_active_execution_scope()
        write_allowed = bool(scope.write_allowed) if scope is not None else False
        repo_root = (
            Path(service.index.root).resolve()
            if getattr(service, "index", None) is not None and getattr(service.index, "root", None)
            else None
        )
        safety_err = validate_command_safety(
            command, write_allowed=write_allowed, repo_root=repo_root
        )
        if safety_err is not None:
            return safety_err
        return service.run_command(command, timeout)

    return {"run_command": run_command}
