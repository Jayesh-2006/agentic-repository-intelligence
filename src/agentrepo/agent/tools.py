"""
Adapt repository tools into LangChain tools the LLM can call.

No repository logic lives here. Each adapter
    1. receives arguments validated by a typed pydantic schema,
    2. calls the repository function (which delegates to RepoService),
    3. returns the result as a JSON string (see utils.to_observation).

Expected problems (unknown symbol, bad path, ...) already come back from
the repository layer as {"error": {...}} and are passed to the LLM as observations,
so it can recover. Wrong argument types from the LLM are turned into a message
for the LLM as well. Any other exception is a real bug and is NOT caught.
"""

from __future__ import annotations

from typing import Callable, Literal

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, ValidationError

from ..service import RepoService
from ..tools import make_tools, make_verification_tools
from .utils import to_observation

Kind = Literal["module", "class", "function", "method"]


class SearchCodeInput(BaseModel):
    query: str = Field(min_length=1, description="Words, identifiers or code text to look for inside the code, e.g. 'reciprocal rank fusion' or 'slugify'.")
    k: int = Field(default=5, ge=1, le=20, description="Maximum number of results.")


class FindFilesInput(BaseModel):
    pattern: str = Field(min_length=1, description="Part of a file path/name, e.g. 'README' or 'pyproject'; or a glob such as '*.md', 'README*' or 'docs/*.rst'. Case-insensitive.")
    limit: int = Field(default=50, ge=1, le=200, description="Maximum number of paths to return.")


class FindSymbolInput(BaseModel):
    query: str = Field(min_length=1, description="Fully-qualified id ('pkg.mod.Class.method'), 'Class.method' or a short name.")
    kind: Kind | None = Field(default=None, description="Optional filter on symbol kind.")
    limit: int = Field(default=20, ge=1, le=100, description="Maximum number of matches.")


class FindCallersInput(BaseModel):
    symbol: str = Field(min_length=1, description="Function, method or class to find callers of. Prefer the fully-qualified id.")
    limit: int = Field(default=50, ge=1, le=500, description="Maximum number of callers.")


class SymbolInput(BaseModel):
    symbol: str = Field(min_length=1, description="Symbol id or name. Prefer the fully-qualified id.")


class ModuleInput(BaseModel):
    target: str = Field(min_length=1, description="Module name like 'pkg.utils' or a repository-relative file path like 'pkg/utils.py'.")


class TargetInput(BaseModel):
    target: str = Field(min_length=1, description="A symbol id/name, a module name, or a repository-relative file path.")


class GetFileInput(BaseModel):
    path: str = Field(min_length=1, description="Repository-relative file path, e.g. 'shop/utils.py'.")
    start_line: int | None = Field(default=None, ge=1, description="First line to return (1-based, inclusive).")
    end_line: int | None = Field(default=None, ge=1, description="Last line to return (inclusive).")


class WriteFileInput(BaseModel):
    path: str = Field(min_length=1, description="Repository-relative file path to create or overwrite, e.g. 'shop/utils.py'.")
    content: str = Field(description="The full text content to write. This REPLACES the file's entire contents.")


class RunCommandInput(BaseModel):
    command: str = Field(min_length=1, description="A repository command to run for targeted validation, such as a syntax check or application command.")
    timeout: int = Field(default=60, ge=1, le=300, description="Maximum seconds to let the command run before it is killed.")


# (name, description for the LLM, argument schema). Order is the order the LLM sees.
_TOOL_SPECS: list[tuple[str, str, type[BaseModel]]] = [
    ("search_repository",
     "CONTENT search: look for keywords, identifiers, code or text inside the indexed repository content, "
    "including Python code chunks and supported text-file chunks. "
    "Returns ranked hits with symbol id, file, line range and a snippet. "
    "k accepts AT MOST 20 (higher values are rejected) - if 20 results is not enough, call this again with a "
    "narrower/different query rather than asking for a larger k. "
    "It is NOT a file or path lookup: it cannot list which files exist. "
    "To find out what files exist, use find_files. "
    "Follow up with get_symbol or find_callers on a promising code hit.",
     SearchCodeInput),
    ("find_files",
     "FILE/PATH search: find which files exist in the repository (any file type, e.g. README.md, pyproject.toml, *.md, tests/). "
     "Matches the pattern against repository-relative file paths only and never looks inside files. "
     "Use it for questions about what files exist, or to get the exact path before reading a file. "
     "Returns repository-relative paths; use get_file to read one. Not for finding code, symbols or text inside files.",
     FindFilesInput),
    ("find_symbol",
     "Look up a symbol by fully-qualified id, 'Class.method' or short name. Returns kind, file, line range, parent and module. "
     "If several symbols match, `ambiguous` is true: choose the right fully-qualified id. Use it to confirm a symbol exists "
     "or to get its exact id.",
     FindSymbolInput),
    ("find_callers",
     "Find the symbols that call a function, method or class (use it when asked where something is called or used). "
     "`callers` are definite call sites; `possible_callers` are unresolved same-name attribute calls that may or may not "
     "target it. The target should be a symbol id or name; an ambiguous short name returns candidates.",
     FindCallersInput),
    ("find_callees",
     "Find what a function, method or module calls. `callees` are resolved repository symbols; `possible_callees` are "
     "ambiguous obj.method() calls; `external_calls` are stdlib/third-party/unresolved names.",
     SymbolInput),
    ("find_importers",
     "Find which repository modules import a given module (dotted name or file path). Returns importing module, file and line.",
     ModuleInput),
    ("get_file",
     "Read the contents of a file whose repository-relative path you already know (any file type, e.g. a README or module), "
     "optionally a 1-based inclusive line range. If you are not sure of the exact path, use find_files first. Also useful for "
     "module-level code and context that get_symbol does not cover. Paths outside the repository are rejected; long files are truncated.",
     GetFileInput),
    ("get_symbol",
     "Get the source code and location of ONE function, method or class by symbol id. Use it to read an implementation "
     "after locating the symbol. Not for whole modules (use get_file).",
     SymbolInput),
    ("get_dependencies",
     "Get what a symbol, module or file depends on (imports, base classes, calls) and what depends on it (importers, "
     "subclasses, callers). The `legend` field explains each key. Use it for 'what does X depend on / what uses X'.",
     TargetInput),
    ("get_ast",
     "Get the AST-derived structure of a known Python symbol, module or .py file: node type, argument names, decorators, bases, "
     "child definitions (classes/functions/methods in a module), calls (each marked resolved / ambiguous / external) and imports. "
     "Use it when the question is about how code is organised (what a module or class defines, what it imports, what a function "
     "takes and calls) rather than about its source text. Python code only; for source text use get_symbol or get_file.",
     TargetInput),
]

# write_file is deliberately kept out of _TOOL_SPECS above, so the
# Explorer and the Main Agent's own tool set stay read-only. Only the Coder
# (build_coder_tools) gets it.
_WRITE_TOOL_SPEC: tuple[str, str, type[BaseModel]] = (
    "write_file",
    "Create a new file or overwrite an EXISTING file's full contents at a repository-relative path "
    "(e.g. 'shop/utils.py'). Creates parent directories automatically. Paths that resolve outside the "
    "repository root are rejected. This replaces the whole file - when modifying an existing file, read it "
    "first with get_file and write back the full updated content, not just the changed lines. "
    "There is no separate patch/diff tool.",
    WriteFileInput,
)

# run_command is kept out of _TOOL_SPECS and _WRITE_TOOL_SPEC's list
# too - it is only wired into the Debugger's tool list (build_debugger_tools),
# alongside the read tools and write_file.
_VERIFY_TOOL_SPEC: tuple[str, str, type[BaseModel]] = (
    "run_command",
    "Run a command with the repository root as its working directory (e.g. a test suite, "
    "linter or the target script), to check whether an implementation actually works. There is no "
    "way to point it at a directory outside the repository. Returns stdout, stderr and the exit code "
    "(0 usually means success). A command that runs too long is killed and reported as timed out.",
    RunCommandInput,
)


def _explain_validation_error(error: ValidationError) -> str:
    details = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in error.errors())
    return f"Invalid tool arguments: {details}. Fix the arguments and call the tool again."


def _adapt(fn: Callable[..., dict], name: str, description: str, schema: type[BaseModel], max_chars: int) -> StructuredTool:
    def run(**kwargs) -> str:
        return to_observation(fn(**kwargs), max_chars)

    return StructuredTool.from_function(
        func=run,
        name=name,
        description=description,
        args_schema=schema,
        handle_validation_error=_explain_validation_error,
    )


def build_repo_tools(service: RepoService, max_observation_chars: int = 6000) -> list[StructuredTool]:
    """The explicit registry: exactly these ten read-only tools are exposed to the LLM.
    Used by the Explorer and directly by the Main Agent."""
    repository_tools = make_tools(service)
    return [_adapt(repository_tools[name], name, description, schema, max_observation_chars)
            for name, description, schema in _TOOL_SPECS]


def build_coder_tools(service: RepoService, max_observation_chars: int = 6000) -> list[StructuredTool]:
    """The Coder's tools: the same ten read-only tools plus write_file. Only the
    Coder gets this list - Explorer and the Main Agent's own tools stay read-only."""
    repository_tools = make_tools(service)
    name, description, schema = _WRITE_TOOL_SPEC
    write_tool = _adapt(repository_tools[name], name, description, schema, max_observation_chars)
    return build_repo_tools(service, max_observation_chars) + [write_tool]


def build_debugger_tools(service: RepoService, max_observation_chars: int = 6000) -> list[StructuredTool]:
    """The Debugger's tools: the Coder's repository tools plus run_command."""
    verification_tools = make_verification_tools(service)
    name, description, schema = _VERIFY_TOOL_SPEC
    verify_tool = _adapt(verification_tools[name], name, description, schema, max_observation_chars)
    return build_coder_tools(service, max_observation_chars) + [verify_tool]


def build_get_file_tool(service: RepoService, max_observation_chars: int = 6000) -> StructuredTool:
    """The single `get_file` tool, exposed directly to the Deep
    Agent (deep_agent.py) for simple, targeted reads of a known repository
    path - without giving it the other nine read-only tools build_repo_tools
    exposes to the Explorer. Built from the same `_TOOL_SPECS` entry and
    repository-safe implementation (`service.get_file`, via `make_tools`) as
    build_repo_tools' own get_file.
    """
    repository_tools = make_tools(service)
    name, description, schema = next(spec for spec in _TOOL_SPECS if spec[0] == "get_file")
    return _adapt(repository_tools[name], name, description, schema, max_observation_chars)


def build_find_files_tool(service: RepoService, max_observation_chars: int = 6000) -> StructuredTool:
    """The single `find_files` tool (path/name discovery only - it never reads file
    contents), exposed directly to the Deep Agent so README/docs discovery does not
    need an extra Explorer LLM hop. Same spec and implementation as the Explorer's."""
    repository_tools = make_tools(service)
    name, description, schema = next(spec for spec in _TOOL_SPECS if spec[0] == "find_files")
    return _adapt(repository_tools[name], name, description, schema, max_observation_chars)


def build_find_symbol_tool(service: RepoService, max_observation_chars: int = 6000) -> StructuredTool:
    """Expose exact symbol lookup to the Main Agent as a read-only tool."""
    repository_tools = make_tools(service)
    name, description, schema = next(spec for spec in _TOOL_SPECS if spec[0] == "find_symbol")
    return _adapt(repository_tools[name], name, description, schema, max_observation_chars)


TOOL_NAMES = [name for name, _, _ in _TOOL_SPECS]
CODER_TOOL_NAMES = TOOL_NAMES + [_WRITE_TOOL_SPEC[0]]
DEBUGGER_TOOL_NAMES = CODER_TOOL_NAMES + [_VERIFY_TOOL_SPEC[0]]
