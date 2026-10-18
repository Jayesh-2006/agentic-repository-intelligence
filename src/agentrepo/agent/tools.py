"""
Adapt repository tools into LangChain tools the Main Agent can call directly.

No repository logic lives here. Each adapter
    1. receives arguments validated by a typed pydantic schema,
    2. enforces active capability and source-scope restrictions at the tool boundary,
    3. calls the repository function (or guarded lifecycle handler),
    4. returns the result as a JSON string (see utils.to_observation).

Expected problems (unknown symbol, bad path, ...) come back from the
repository layer as {"error": {...}} and are passed to the LLM as observations
so it can recover. Wrong argument types from the LLM are turned into a message
for the LLM as well. Any other exception is a real bug and is NOT caught.
"""

from __future__ import annotations

from typing import Callable, Literal

from .state import StructuredTool
from pydantic import BaseModel, Field, ValidationError

from ..service import RepoService
from ..tools import make_tools, make_verification_tools
from . import intent as intent_mod
from .observability import (
    emit_trace,
    get_run_context,
    log_tool_execution_end,
    log_tool_execution_exception,
    log_tool_execution_start,
)
from .safety import FAILED, get_active_execution_scope
from .utils import to_observation

Kind = Literal["module", "class", "function", "method"]


class SearchCodeInput(BaseModel):
    query: str = Field(
        min_length=1,
        description="Specific keywords, metric names, identifiers, or phrases to search inside repository files.",
    )
    k: int = Field(default=5, ge=1, le=20, description="Max results to return (default 5; prefer 5 to save context).")


class FindFilesInput(BaseModel):
    pattern: str = Field(
        min_length=1,
        description="File path/name substring or glob ONLY (e.g. 'README', '*.md', 'model.py'); never content keywords.",
    )
    limit: int = Field(default=50, ge=1, le=200, description="Max paths to return.")


class FindSymbolInput(BaseModel):
    query: str = Field(min_length=1, description="Fully-qualified id ('pkg.mod.Class.method'), 'Class.method', or short name.")
    kind: Kind | None = Field(default=None, description="Optional kind filter.")
    limit: int = Field(default=20, ge=1, le=100, description="Max matches.")


class FindCallersInput(BaseModel):
    symbol: str = Field(min_length=1, description="Function, method, or class id/name to find callers of.")
    limit: int = Field(default=50, ge=1, le=500, description="Max callers.")


class SymbolInput(BaseModel):
    symbol: str = Field(min_length=1, description="Symbol id or name (prefer fully-qualified id).")


class ModuleInput(BaseModel):
    target: str = Field(min_length=1, description="Module name ('pkg.utils') or repo-relative path ('pkg/utils.py').")


class TargetInput(BaseModel):
    target: str = Field(min_length=1, description="Symbol id/name, module name, or repo-relative file path.")


class GetFileInput(BaseModel):
    path: str = Field(min_length=1, description="Repo-relative file path, e.g. 'README.md' or 'shop/utils.py'.")
    start_line: int | None = Field(default=None, ge=1, description="1-based inclusive start line.")
    end_line: int | None = Field(default=None, ge=1, description="1-based inclusive end line.")


class WriteFileInput(BaseModel):
    path: str = Field(min_length=1, description="Repo-relative file path to create or overwrite.")
    content: str = Field(description="Full text content (replaces entire file).")


class RunCommandInput(BaseModel):
    command: str = Field(min_length=1, description="Validation or test command to run in the repository root.")
    timeout: int = Field(default=60, ge=1, le=300, description="Timeout in seconds.")


class VerifyRepositoryInput(BaseModel):
    """No arguments: runs config.verification_command."""


# (name, description for the LLM, argument schema). Order is the order the LLM sees.
_TOOL_SPECS: list[tuple[str, str, type[BaseModel]]] = [
    ("search_repository",
     "CONTENT search over indexed code and documentation chunks (e.g. README.md). Returns ranked hits with symbol, file, "
     "line range, and snippet (default k=5). Not for filenames (use find_files). Follow up with get_file or get_symbol.",
     SearchCodeInput),
    ("find_files",
     "FILE/PATH search by filename/path substring or glob only (never file contents or symbol names).",
     FindFilesInput),
    ("find_symbol",
     "Locate a symbol by id, 'Class.method', or short name. Returns kind, file, line range, parent, and module. "
     "If `ambiguous` is true, disambiguate with get_symbol or find_callers/find_callees.",
     FindSymbolInput),
    ("find_callers",
     "Find callers of a function, method, or class (`callers` are definite; `possible_callers` are unresolved attribute calls).",
     FindCallersInput),
    ("find_callees",
     "Find what a symbol/module calls (`callees`, `possible_callees`, and `external_calls`).",
     SymbolInput),
    ("find_importers",
     "Find repository modules that import a given module name or file path.",
     ModuleInput),
    ("get_file",
     "Read a file at a repo-relative path (e.g. README.md or source file), optionally for `start_line..end_line`.",
     GetFileInput),
    ("get_symbol",
     "Read the exact source code and line range of one function, method, or class by symbol id/name.",
     SymbolInput),
    ("get_dependencies",
     "Get upstream/downstream dependencies (imports, base classes, calls, importers, subclasses, callers) for a target.",
     TargetInput),
    ("get_ast",
     "Get AST structure of a Python symbol/module/file: args, decorators, bases, children, calls, and imports.",
     TargetInput),
]

_WRITE_TOOL_SPEC: tuple[str, str, type[BaseModel]] = (
    "write_file",
    "Create or overwrite a repo-relative file with full contents (requires WRITE capability; read existing files with get_file first).",
    WriteFileInput,
)

_RUN_COMMAND_SPEC: tuple[str, str, type[BaseModel]] = (
    "run_command",
    "Run a validation/test command in the repository root (requires EXECUTE capability).",
    RunCommandInput,
)

_VERIFY_REPOSITORY_SPEC: tuple[str, str, type[BaseModel]] = (
    "verify_repository",
    "Run the configured repository verification command (requires EXECUTE capability).",
    VerifyRepositoryInput,
)


def _explain_validation_error(error: ValidationError) -> str:
    details = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in error.errors())
    msg = f"Invalid tool arguments: {details}. Fix the arguments and call the tool again."
    ctx = get_run_context()
    agent = ctx.current_agent if ctx else "Main"
    emit_trace(
        f"[{agent}] tool_argument_validation -> FAILED: {details[:180]}"
    )
    return msg


def _check_boundary_policy(name: str, kwargs: dict, service: RepoService | None = None) -> dict | None:
    """Enforce active source-scope and WRITE/EXECUTE capability grants at the
    final tool adapter boundary."""
    scope = get_active_execution_scope()
    eff_service = service or (scope.transaction.service if scope is not None else None)
    if scope is not None and isinstance(scope.intent, dict):
        allowed, note = intent_mod.source_allows_tool(
            scope.intent, name, args=kwargs, service=eff_service
        )
        if not allowed:
            return {
                "state": FAILED,
                "report": note,
                "error": {
                    "code": "source_restriction",
                    "message": note,
                },
            }
    if name in MUTATION_TOOL_NAMES:
        if scope is None or not scope.write_allowed:
            msg = (
                "MUTATION_BLOCKED (system data): this execution context does not have WRITE "
                f"capability, so {name!r} was not run and the repository was NOT modified."
            )
            return {
                "state": FAILED,
                "report": msg,
                "error": {
                    "code": "capability_denied",
                    "message": msg,
                },
            }
    if name in EXECUTION_TOOL_NAMES:
        if scope is None or not scope.execute_allowed:
            msg = (
                "EXECUTION_BLOCKED (system data): this execution context does not have EXECUTE "
                f"capability, so {name!r} was not run and no repository command was executed."
            )
            return {
                "state": FAILED,
                "report": msg,
                "error": {
                    "code": "capability_denied",
                    "message": msg,
                },
            }
    return None


def _adapt(
    fn: Callable[..., dict],
    name: str,
    description: str,
    schema: type[BaseModel],
    max_chars: int,
    service: RepoService | None = None,
) -> StructuredTool:
    def run(**kwargs) -> str:
        started, agent, tc_id = log_tool_execution_start(name, kwargs)
        try:
            policy_err = _check_boundary_policy(name, kwargs, service=service)
            raw = policy_err if policy_err is not None else fn(**kwargs)
            obs = to_observation(raw, max_chars) if isinstance(raw, dict) else str(raw)
        except Exception as exc:
            log_tool_execution_exception(name, started, agent, tc_id, exc)
            raise
        log_tool_execution_end(name, started, agent, tc_id, raw, obs)
        return obs

    return StructuredTool.from_function(
        func=run,
        name=name,
        description=description,
        args_schema=schema,
        handle_validation_error=_explain_validation_error,
    )


def build_repo_tools(service: RepoService, max_observation_chars: int = 6000) -> list[StructuredTool]:
    """Build the ten read-only repository tools exposed directly to the Main Agent."""
    repository_tools = make_tools(service)
    return [
        _adapt(repository_tools[name], name, description, schema, max_observation_chars, service=service)
        for name, description, schema in _TOOL_SPECS
    ]


def build_write_file_tool(
    service: RepoService,
    max_observation_chars: int = 6000,
    write_fn: Callable[..., dict] | None = None,
) -> StructuredTool:
    """Build the guarded write_file tool."""
    fn = write_fn if write_fn is not None else make_tools(service)["write_file"]
    name, description, schema = _WRITE_TOOL_SPEC
    return _adapt(fn, name, description, schema, max_observation_chars, service=service)


def build_run_command_tool(
    service: RepoService,
    max_observation_chars: int = 6000,
    run_command_fn: Callable[..., dict] | None = None,
) -> StructuredTool:
    """Build the guarded run_command tool."""
    fn = run_command_fn if run_command_fn is not None else make_verification_tools(service)["run_command"]
    name, description, schema = _RUN_COMMAND_SPEC
    return _adapt(fn, name, description, schema, max_observation_chars, service=service)


def build_verify_repository_tool(
    verify_fn: Callable[[], dict],
    max_observation_chars: int = 6000,
    service: RepoService | None = None,
) -> StructuredTool:
    """Build the guarded verify_repository tool."""
    name, description, schema = _VERIFY_REPOSITORY_SPEC
    return _adapt(verify_fn, name, description, schema, max_observation_chars, service=service)


READ_ONLY_TOOL_NAMES = tuple(name for name, _, _ in _TOOL_SPECS)
STRUCTURAL_TOOL_NAMES = (
    "find_symbol",
    "get_symbol",
    "find_callers",
    "find_callees",
    "find_importers",
    "get_dependencies",
    "get_ast",
)
MUTATION_TOOL_NAMES = (_WRITE_TOOL_SPEC[0],)
EXECUTION_TOOL_NAMES = (_VERIFY_REPOSITORY_SPEC[0], _RUN_COMMAND_SPEC[0])
TOOL_NAMES = list(READ_ONLY_TOOL_NAMES)
