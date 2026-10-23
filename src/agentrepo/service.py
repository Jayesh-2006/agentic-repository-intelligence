"""Repository service layer over the symbol index, code graph, and lexical retriever."""

from __future__ import annotations

import ast
from contextlib import contextmanager
import difflib
import fnmatch
import os
import posixpath
import re
import shlex
import shutil
import signal
import stat
import subprocess
import tempfile
import textwrap
import threading
import time
import uuid
from pathlib import Path

from .filesystem import (
    PROTECTED_PATH_PARTS,
    is_ignored_name,
    is_ignored_path,
    is_virtualenv_root,
    is_within,
)
from .graph import CodeGraph
from .indexer import index_repository
from .models import RepositoryIndex, Symbol
from .persistence import load_index
from .retrieval import LexicalRetriever, Retriever, query_tokens, tokenize

MAX_K = 50
MAX_FILE_RESULTS = 200
CALLABLE_KINDS = ("function", "method", "class")
_POSSIBLE_REASON = "attribute call on a receiver whose type is unknown"

DEFAULT_COMMAND_TIMEOUT = 60
MAX_COMMAND_TIMEOUT = 300
MAX_COMMAND_OUTPUT_CHARS = 4000

_SECRET_ENV_KEY_RE = re.compile(
    r"(?:API[_-]?KEY|SECRET|TOKEN|PASSWORD|PASSWD|CREDENTIAL|PRIVATE[_-]?KEY|AUTH[_-]?HEADER|AUTHORIZATION)",
    re.IGNORECASE,
)
_INLINE_OUTPUT_SECRET_PATTERNS = (
    (re.compile(r"\bgsk_[A-Za-z0-9]{12,}\b"), "[REDACTED]"),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"), "[REDACTED]"),
    (re.compile(r"(?i)\b(Bearer\s+)[A-Za-z0-9._\-+/=]{8,}"), r"\1[REDACTED]"),
    (
        re.compile(
            r"(?i)\b(api[_-]?key|secret|token|password|passwd|authorization)\b(\s*[:=]\s*['\"]?)([^\s'\",;}\]]+)"
        ),
        r"\1\2[REDACTED]",
    ),
)


class _RepoTransactionLock:
    """Owner-token re-entrant lock per canonical repository root.

    Locking hierarchy (outer -> inner):
      1. `_REPO_LOCKS_GUARD` (only while looking up/creating a `_RepoTransactionLock`)
      2. `_RepoTransactionLock` (per canonical repository root, owned by `tx_id`)
      3. `RepoService._checkpoint_lock` (protects `_checkpoints` map)
      4. `RepoService._runtime_metrics_lock` (protects `_runtime_metrics` counters)

    Using an explicit `tx_id` owner token (rather than OS thread identity alone)
    allows a `MutationTransaction` to hold exclusive mutation ownership of a
    repository root across multiple graph turns—even when LangGraph dispatches
    tool nodes on worker threads—while blocking any concurrent transaction on
    the same repository root until the owning transaction finalizes or cleans up.
    """

    def __init__(self, root_key: str) -> None:
        self.root_key = root_key
        self._cond = threading.Condition(threading.Lock())
        self._owner_tx_id: str | None = None
        self._depth: int = 0

    def acquire(self, tx_id: str, timeout: float | None = None) -> bool:
        if not tx_id:
            tx_id = f"thread_{threading.get_ident()}"
        deadline = (time.monotonic() + timeout) if timeout is not None else None
        with self._cond:
            while self._owner_tx_id is not None and self._owner_tx_id != tx_id:
                remaining = (deadline - time.monotonic()) if deadline is not None else None
                if remaining is not None and remaining <= 0:
                    return False
                self._cond.wait(timeout=remaining)
            self._owner_tx_id = tx_id
            self._depth += 1
            return True

    def release(self, tx_id: str) -> None:
        if not tx_id:
            tx_id = f"thread_{threading.get_ident()}"
        with self._cond:
            if self._owner_tx_id != tx_id:
                return
            self._depth = max(0, self._depth - 1)
            if self._depth == 0:
                self._owner_tx_id = None
                self._cond.notify_all()

    def release_all(self, tx_id: str) -> None:
        """Release all re-entrant acquisitions held by `tx_id`."""
        if not tx_id:
            return
        with self._cond:
            if self._owner_tx_id == tx_id:
                self._owner_tx_id = None
                self._depth = 0
                self._cond.notify_all()

    def is_owned_by(self, tx_id: str) -> bool:
        with self._cond:
            return bool(tx_id and self._owner_tx_id == tx_id)

    @property
    def active_owner(self) -> str | None:
        with self._cond:
            return self._owner_tx_id


_REPO_LOCKS_GUARD = threading.Lock()
_REPO_LOCKS: dict[str, _RepoTransactionLock] = {}
_ACTIVE_TX_ID: threading.local = threading.local()


def get_repo_transaction_lock(root: str | Path) -> _RepoTransactionLock:
    """Return the shared `_RepoTransactionLock` for the canonical repository root."""
    try:
        canonical = str(Path(root).expanduser().resolve())
    except OSError:
        canonical = os.path.abspath(str(root))
    if os.name == "nt":
        canonical = canonical.casefold()
    with _REPO_LOCKS_GUARD:
        lock = _REPO_LOCKS.get(canonical)
        if lock is None:
            lock = _RepoTransactionLock(canonical)
            _REPO_LOCKS[canonical] = lock
        return lock


def set_current_thread_tx_id(tx_id: str | None) -> str | None:
    prev = getattr(_ACTIVE_TX_ID, "tx_id", None)
    _ACTIVE_TX_ID.tx_id = tx_id
    return prev


def get_current_thread_tx_id() -> str | None:
    return getattr(_ACTIVE_TX_ID, "tx_id", None)


_STRIPPED_COMMAND_ENV_KEYS = frozenset({
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_CONFIG",
    "GIT_CONFIG_GLOBAL",
    "GIT_CONFIG_SYSTEM",
    "GIT_CONFIG_PARAMETERS",
    "GIT_EXEC_PATH",
    "GIT_EXTERNAL_DIFF",
    "GIT_SSH",
    "GIT_SSH_COMMAND",
    "GIT_ASKPASS",
    "SSH_ASKPASS",
    "GIT_EDITOR",
    "EDITOR",
    "VISUAL",
    "PAGER",
    "GIT_PAGER",
    "BROWSER",
    "PYTHONSTARTUP",
    "PYTHONINSPECT",
    "LD_PRELOAD",
    "DYLD_INSERT_LIBRARIES",
})


def _sanitized_search_path(repo_root: Path | None = None) -> str:
    """Return a sanitized PATH string with empty, ., relative, and
    repository-contained directories removed so untrusted repository files
    cannot shadow system executables."""
    raw_path = os.environ.get("PATH", "")
    if not raw_path:
        return ""
    root_resolved: Path | None = None
    if repo_root is not None:
        try:
            root_resolved = repo_root.resolve()
        except OSError:
            root_resolved = None
    kept: list[str] = []
    for entry in raw_path.split(os.pathsep):
        cleaned = entry.strip().strip("\"'")
        if not cleaned or cleaned in (".", ".."):
            continue
        entry_path = Path(cleaned)
        if not entry_path.is_absolute():
            continue
        if root_resolved is not None:
            try:
                abs_entry = Path(os.path.abspath(entry_path))
                if is_within(root_resolved, abs_entry):
                    continue
                res_entry = abs_entry.resolve()
                if is_within(root_resolved, res_entry):
                    continue
            except OSError:
                continue
        kept.append(cleaned)
    return os.pathsep.join(kept)


def _sanitized_command_env(repo_root: Path | None = None) -> tuple[dict[str, str], list[str]]:
    """Build a child-process environment with API keys/secrets and dangerous
    VCS/interpreter redirection variables removed, PATH sanitized against
    repository/relative shadowing, and return `(sanitized_env, secret_values_to_redact)`."""
    clean_env: dict[str, str] = {}
    secret_values: list[str] = []
    for k, v in os.environ.items():
        if _SECRET_ENV_KEY_RE.search(k):
            if isinstance(v, str) and len(v.strip()) >= 4:
                secret_values.append(v.strip())
            continue
        if k.upper() in _STRIPPED_COMMAND_ENV_KEYS:
            continue
        clean_env[k] = v
    clean_env["PATH"] = _sanitized_search_path(repo_root)
    clean_env["GIT_TERMINAL_PROMPT"] = "0"
    clean_env["GIT_PAGER"] = "cat"
    clean_env["PAGER"] = "cat"
    clean_env["GIT_OPTIONAL_LOCKS"] = "0"
    clean_env["PYTHONNOUSERSITE"] = "1"
    return clean_env, secret_values


def _redact_command_output(text: str, secret_values: list[str]) -> str:
    if not text:
        return ""
    out = text
    for secret in secret_values:
        if secret and secret in out:
            out = out.replace(secret, "[REDACTED]")
    for pattern, replacement in _INLINE_OUTPUT_SECRET_PATTERNS:
        out = pattern.sub(replacement, out)
    return out


def _path_key(path: str) -> str:
    normalized = posixpath.normpath(path.replace("\\", "/"))
    return normalized.casefold() if os.name == "nt" else normalized


def _error(code: str, message: str, **extra) -> dict:
    return {"error": {"code": code, "message": message, **extra}}


def _clip(text: str, max_chars: int = MAX_COMMAND_OUTPUT_CHARS) -> str:
    """Keep a command's stdout and stderr concise."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"... [truncated: showing {max_chars} of {len(text)} characters]"


def _decode(output) -> str:
    """Normalise subprocess output to UTF-8 text."""
    if output is None:
        return ""
    if isinstance(output, bytes):
        return output.decode("utf-8", errors="replace")
    return output


def _read_limited(stream, limit: int, state: dict) -> None:
    """Drain a child pipe while retaining only a bounded output prefix."""
    kept = bytearray()
    total = 0
    while True:
        block = stream.read(8192)
        if not block:
            break
        total += len(block)
        if len(kept) < limit:
            kept.extend(block[: limit - len(kept)])
        state["data"] = bytes(kept)
        state["total"] = total


def _command_text(state: dict) -> str:
    data = state.get("data", b"")
    result = _decode(data)
    total = state.get("total", len(data))
    if total > len(data):
        result += f"... [truncated: showing {len(data)} of {total} bytes]"
    return result


def _split_command(command: str) -> list[str]:
    """Parse a command into argv without invoking a shell."""
    if os.name != "nt":
        return shlex.split(command)

    import ctypes

    shell32 = ctypes.WinDLL("shell32", use_last_error=True)
    command_line_to_argv = shell32.CommandLineToArgvW
    command_line_to_argv.argtypes = [ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_int)]
    command_line_to_argv.restype = ctypes.POINTER(ctypes.c_wchar_p)
    local_free = ctypes.WinDLL("kernel32", use_last_error=True).LocalFree
    local_free.argtypes = [ctypes.c_void_p]
    local_free.restype = ctypes.c_void_p

    argc = ctypes.c_int()
    argv = command_line_to_argv(command, ctypes.byref(argc))
    if not argv:
        raise ValueError("could not parse Windows command line")
    try:
        return [argv[index] for index in range(argc.value)]
    finally:
        local_free(argv)


def _check_int(name: str, value, low: int, high: int | None = None) -> dict | None:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < low
        or (high is not None and value > high)
    ):
        bound = f"between {low} and {high}" if high is not None else f">= {low}"
        return _error("invalid_input", f"{name} must be an integer {bound}, got {value!r}")
    return None


def _read_line_window(
    source_file, start: int, end: int | None, max_lines: int
) -> tuple[list[str], int, bool]:
    """Read a bounded line window while counting the complete file."""
    selected: list[str] = []
    total = 0
    truncated = False
    for physical_line in source_file:
        for line in physical_line.splitlines():
            total += 1
            if total < start or (end is not None and total > end):
                continue
            if len(selected) < max_lines:
                selected.append(line)
            else:
                truncated = True
    return selected, total, truncated


class RepoService:
    _sanitized_search_path = staticmethod(_sanitized_search_path)
    def __init__(
        self,
        index: RepositoryIndex,
        graph: CodeGraph,
        retriever: Retriever | None = None,
    ) -> None:
        self._state_lock = threading.RLock()
        self.graph = graph
        self.retriever = retriever if retriever is not None else LexicalRetriever(index)
        if (
            not callable(getattr(self.retriever, "rebuild", None))
            or getattr(type(self.retriever), "rebuild", None) is Retriever.rebuild
        ):
            raise TypeError("custom retrievers must implement rebuild(index)")
        self._set_index(index)
        # In-memory checkpoints are keyed by checkpoint id. Each holds a full
        # snapshot copy of the repository (outside the repo, in a temp dir)
        # taken at a point in time, so the transaction layer can ask "what
        # changed?" and, if needed, restore it.
        #
        # Snapshot semantics:
        #   - Includes all non-ignored tracked, untracked, and hidden files and
        #     directories inside `index.root`.
        #   - Does NOT follow symlinks (symlink entries themselves are recorded
        #     by link target bytes without traversing into symlinked directories).
        #   - Excludes `IGNORED_DIRS` (`.git`, `.hg`, `.svn`, `__pycache__`,
        #     `.venv`, `venv`, `node_modules`, `.mypy_cache`, `.pytest_cache`).
        self._checkpoints: dict[str, dict] = {}
        self._checkpoint_lock = threading.Lock()
        self._runtime_metrics = {}
        self._runtime_metrics_lock = threading.Lock()

    @property
    def repo_lock(self) -> _RepoTransactionLock:
        return get_repo_transaction_lock(self.index.root)

    @contextmanager
    def transaction_lock(self, tx_id: str | None = None):
        """Acquire the per-repository transaction lock for `tx_id` (or the
        current thread's active `tx_id` / thread fallback) and release on exit."""
        eff_id = tx_id or get_current_thread_tx_id() or f"thread_{threading.get_ident()}_{uuid.uuid4().hex[:6]}"
        lock = self.repo_lock
        lock.acquire(eff_id)
        prev = set_current_thread_tx_id(eff_id)
        try:
            yield eff_id
        finally:
            set_current_thread_tx_id(prev)
            lock.release(eff_id)

    def runtime_metrics_snapshot(self) -> dict:
        """Return a small, thread-safe snapshot of repository operation totals."""
        with self._runtime_metrics_lock:
            return {key: dict(value) for key, value in self._runtime_metrics.items()}

    def _measure_runtime(self, name: str, function, *args, **kwargs):
        started = time.perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            elapsed = (time.perf_counter() - started) * 1000
            with self._runtime_metrics_lock:
                totals = self._runtime_metrics.setdefault(
                    name, {"calls": 0, "latency_ms": 0.0}
                )
                totals["calls"] += 1
                totals["latency_ms"] += elapsed

    def _set_index(self, index: RepositoryIndex) -> None:
        """(Re)build the lookup dicts derived from `index.files`."""
        module_by_path = {f.path: f.module for f in index.files}
        path_by_module = {f.module: f.path for f in index.files}
        file_records_by_key = {_path_key(f.path): f for f in index.files}
        with self._state_lock:
            self.index = index
            self._module_by_path = module_by_path
            self._path_by_module = path_by_module
            self._file_records_by_key = file_records_by_key

    @classmethod
    def from_index_dir(
        cls, index_dir: str | Path, repo_root: str | Path | None = None
    ) -> "RepoService":
        """Load a saved index."""
        index, graph = load_index(index_dir)
        if repo_root is not None:
            try:
                resolved_root = Path(repo_root).expanduser().resolve(strict=True)
            except (OSError, RuntimeError) as exc:
                raise ValueError(
                    f"repository root is not accessible: {repo_root}: {exc}"
                ) from exc
            if not resolved_root.is_dir():
                raise ValueError(f"repository root is not a directory: {resolved_root}")
            index.root = str(resolved_root)
        return cls(index, graph)

    @classmethod
    def from_repository(cls, root: str | Path) -> "RepoService":
        index, graph = index_repository(root)
        return cls(index, graph)

    # ------------------------------------------------------------------
    # Repository state consistency
    #
    # Every successful write_file call refreshes at this mutation boundary,
    # so the next query sees updated symbols, chunks, graph and paths.
    # The transaction layer in safety.py also refreshes after command
    # execution and rollback so the index always matches the working tree.
    # ------------------------------------------------------------------

    def refresh(self) -> dict:
        return self._measure_runtime("refresh", self._refresh)

    def _refresh(self) -> dict:
        """Re-index the repository from its current on-disk state and
        replace this RepoService's index/graph/retriever with the result."""
        with self.transaction_lock():
            before = set(self.index.repo_files or [])
            try:
                new_index, new_graph = index_repository(self.index.root)
            except (OSError, ValueError) as exc:
                return _error("index_refresh_failed", str(exc))
            after = set(new_index.repo_files or [])

            warning = None
            try:
                new_retriever = self.retriever.rebuild(new_index)
            except Exception as exc:
                new_retriever = LexicalRetriever(new_index)
                warning = f"custom retriever rebuild failed; switched to lexical retrieval: {exc}"
            if not callable(getattr(new_retriever, "search", None)):
                new_retriever = LexicalRetriever(new_index)
                warning = "custom retriever rebuild returned an invalid object; switched to lexical retrieval"
            with self._state_lock:
                self._set_index(new_index)
                self.graph = new_graph
                self.retriever = new_retriever

            result = {
                "files_added": sorted(after - before),
                "files_removed": sorted(before - after),
                "file_count": len(after),
                "symbol_count": len(new_index.symbols),
            }
            if warning:
                result["warning"] = warning
            return result

    # ------------------------------------------------------------------
    # shared helpers
    # ------------------------------------------------------------------

    def _module_of(self, sym: Symbol) -> str:
        return self._module_by_path.get(sym.file, sym.id)

    def _summary(self, sym: Symbol) -> dict:
        return {
            "symbol": sym.id,
            "kind": sym.kind,
            "file": sym.file,
            "line_start": sym.lineno,
            "line_end": sym.end_lineno,
            "parent": sym.parent,
            "module": self._module_of(sym),
        }

    def _site(self, symbol_id: str, line: int | None) -> dict:
        """A place in the code: which symbol, in which file, at which line."""
        sym = self.index.symbols.get(symbol_id)
        return {"symbol": symbol_id, "file": sym.file if sym else None, "line": line}

    def _find_matches(self, name: str) -> tuple[list[Symbol], str]:
        """Exact id first; then dotted-suffix (`Class.method`, `method`);
        then the same suffix match ignoring case."""
        symbols = self.index.symbols
        if name in symbols:
            exact = symbols[name]
            base_id = f"{exact.parent}.{exact.name}" if exact.parent else exact.name
            declarations = (
                [
                    s
                    for s in symbols.values()
                    if s.id != exact.id
                    and s.name == exact.name
                    and s.parent == exact.parent
                ]
                if exact.id == base_id
                else []
            )
            return [
                exact,
                *sorted(declarations, key=lambda symbol: (symbol.lineno, symbol.id)),
            ], "exact"
        suffix = [s for sid, s in symbols.items() if sid.endswith("." + name)]
        if suffix:
            expanded = list(suffix)
            for symbol in suffix:
                base_id = (
                    f"{symbol.parent}.{symbol.name}" if symbol.parent else symbol.name
                )
                if symbol.id == base_id:
                    expanded.extend(
                        s
                        for s in symbols.values()
                        if s.name == symbol.name
                        and s.parent == symbol.parent
                        and s.id != symbol.id
                    )
            unique = {symbol.id: symbol for symbol in expanded}
            return sorted(unique.values(), key=lambda symbol: symbol.id), "suffix"
        lowered = "." + name.lower()
        folded = [
            s
            for sid, s in symbols.items()
            if sid.lower().endswith(lowered) or sid.lower() == name.lower()
        ]
        expanded = list(folded)
        for symbol in folded:
            base_id = f"{symbol.parent}.{symbol.name}" if symbol.parent else symbol.name
            if symbol.id == base_id:
                expanded.extend(
                    s
                    for s in symbols.values()
                    if s.name == symbol.name
                    and s.parent == symbol.parent
                    and s.id != symbol.id
                )
        unique = {symbol.id: symbol for symbol in expanded}
        return sorted(unique.values(), key=lambda symbol: symbol.id), "case_insensitive"

    def _suggestions(self, name: str) -> list[str]:
        short = name.split(".")[-1]
        by_name: dict[str, list[str]] = {}
        for sid, sym in self.index.symbols.items():
            by_name.setdefault(sym.name, []).append(sid)
        close = difflib.get_close_matches(short, list(by_name), n=5, cutoff=0.6)
        return [sid for n in close for sid in by_name[n]][:5]

    def _resolve_one(
        self, target, kinds: tuple[str, ...] | None = None
    ) -> tuple[Symbol | None, dict | None]:
        """Turn a user-supplied name (or file path) into exactly one Symbol,
        or an error dict. Never guesses when several symbols match."""
        if not isinstance(target, str) or not target.strip():
            return None, _error("invalid_input", "target must be a non-empty string")
        target = target.strip()

        wants_module = kinds is None or "module" in kinds
        if wants_module and (
            target.casefold().endswith(".py") or "/" in target or "\\" in target
        ):
            root = Path(self.index.root).resolve()
            try:
                supplied = Path(target)
                if supplied.is_absolute():
                    resolved = supplied.resolve()
                    if not is_within(root, resolved):
                        return None, _error(
                            "file_not_indexed", f"no indexed file at {target!r}"
                        )
                    path = resolved.relative_to(root).as_posix()
                else:
                    path = posixpath.normpath(target.replace("\\", "/"))
            except (OSError, RuntimeError, ValueError):
                return None, _error("invalid_path", "path contains invalid characters")
            if path == ".." or path.startswith("../"):
                return None, _error("file_not_indexed", f"no indexed file at {target!r}")
            record = self._file_records_by_key.get(_path_key(path))
            if record is None:
                return None, _error("file_not_indexed", f"no indexed file at {path!r}")
            if not record.parse_ok:
                return None, _error(
                    "file_not_parsed",
                    f"{path} could not be parsed: {record.error}",
                )
            return self.index.symbols[record.module], None

        matches, _ = self._find_matches(target)
        if kinds is not None:
            allowed = [m for m in matches if m.kind in kinds]
            if matches and not allowed:
                return None, _error(
                    "wrong_kind",
                    f"{target!r} matches {[m.id + ' (' + m.kind + ')' for m in matches[:5]]}, "
                    f"but this tool needs one of {list(kinds)}",
                )
            matches = allowed
        if not matches:
            return None, _error(
                "symbol_not_found",
                f"no symbol matches {target!r}",
                suggestions=self._suggestions(target),
            )
        if len(matches) > 1:
            return None, _error(
                "ambiguous_symbol",
                f"{target!r} matches {len(matches)} symbols; use a fully-qualified id",
                candidates=[self._summary(m) for m in matches[:10]],
            )
        return matches[0], None

    def _definite_callers(self, symbol_id: str) -> list[dict]:
        seen, out = set(), []
        for e in self.graph.find_callers(symbol_id):
            if e.resolved and (e.source, e.lineno) not in seen:
                seen.add((e.source, e.lineno))
                out.append(self._site(e.source, e.lineno))
        return sorted(out, key=lambda c: (c["file"] or "", c["line"] or 0, c["symbol"]))

    def _possible_callers(self, sym: Symbol) -> list[dict]:
        seen, out = set(), []
        for e in self.graph.find_callers(sym.name):
            if (
                not e.resolved
                and e.confidence == "ambiguous"
                and (e.source, e.lineno) not in seen
            ):
                seen.add((e.source, e.lineno))
                out.append({**self._site(e.source, e.lineno), "reason": _POSSIBLE_REASON})
        return sorted(out, key=lambda c: (c["file"] or "", c["line"] or 0, c["symbol"]))

    def _calls_from(self, symbol_id: str):
        """Split a symbol's outgoing CALLS edges by how well they resolved."""
        definite, ambiguous, external = [], [], []
        seen = set()
        for e in self.graph.find_callees(symbol_id):
            if (e.target, e.lineno) in seen:
                continue
            seen.add((e.target, e.lineno))
            if e.resolved:
                definite.append(e)
            elif e.confidence == "ambiguous":
                ambiguous.append(e)
            else:
                external.append(e)
        return definite, ambiguous, external

    # ------------------------------------------------------------------
    # 1. search_repository
    # ------------------------------------------------------------------

    def search_repository(
        self, query: str, k: int = 5, file_path: str | None = None
    ) -> dict:
        return self._measure_runtime(
            "search_repository", self._search_repository, query, k, file_path
        )

    def _search_repository(
        self, query: str, k: int = 5, file_path: str | None = None
    ) -> dict:
        with self._state_lock:
            return self._search_repository_unlocked(query, k, file_path)

    def _search_repository_unlocked(
        self, query: str, k: int = 5, file_path: str | None = None
    ) -> dict:
        if not isinstance(query, str) or not query.strip():
            return _error("invalid_query", "query must be a non-empty string")
        if not query_tokens(query):
            return _error("invalid_query", "query has no searchable words")
        bad = _check_int("k", k, 1, MAX_K)
        if bad:
            return bad

        if file_path is not None:
            if not isinstance(file_path, str) or not file_path.strip():
                return _error(
                    "invalid_input",
                    "file_path must be a non-empty repository-relative path",
                )
            normalized_path = posixpath.normpath(file_path.replace("\\", "/").strip("/"))
            if normalized_path.startswith("../") or normalized_path in ("..", ".", ""):
                return _error(
                    "invalid_input", "file_path must stay inside the repository"
                )
        else:
            normalized_path = None

        terms = set(query_tokens(query))
        results = []
        try:
            hits = self.retriever.search(query, top_k=k, file_path=normalized_path)
        except TypeError:
            hits = self.retriever.search(query, top_k=MAX_K)
        for hit in hits:
            chunk = self.index.chunks.get(hit.chunk_id)
            if chunk is None:
                continue
            if (
                normalized_path is not None
                and chunk.file.replace("\\", "/").casefold()
                != normalized_path.casefold()
            ):
                continue
            symbol_id = (
                chunk.file
                if chunk.kind == "text"
                else chunk.symbol_id or hit.chunk_id.removesuffix("#module")
            )
            snippet, snippet_start, snippet_end = self._snippet(chunk.text, chunk.start_line, terms)
            results.append(
                {
                    "symbol": symbol_id,
                    "kind": chunk.kind,
                    "file": chunk.file,
                    "line_start": chunk.start_line,
                    "line_end": chunk.end_line,
                    "score": round(hit.score, 3),
                    "snippet": snippet,
                    "snippet_line_start": snippet_start,
                    "snippet_line_end": snippet_end,
                }
            )
        return {"query": query, "count": len(results), "results": results}

    @staticmethod
    def _snippet(
        text: str,
        first_line: int,
        terms: set[str],
        height: int = 8,
        max_chars: int = 600,
    ) -> tuple[str, int, int]:
        """A window of lines around the cluster that matches the most query terms."""
        lines = text.splitlines() or [""]
        scores = [len(terms & set(tokenize(line))) for line in lines]
        # Score 3-line sliding windows so multi-line code/table matches stay centered
        window_scores = [
            scores[i]
            + (scores[i + 1] if i + 1 < len(scores) else 0)
            + (scores[i + 2] if i + 2 < len(scores) else 0)
            for i in range(len(scores))
        ]
        best = window_scores.index(max(window_scores))
        start = max(0, min(best - 1, max(0, len(lines) - height)))
        selected = lines[start : start + height]
        snippet = "\n".join(selected)
        end_line = first_line + start + max(0, len(selected) - 1)
        return snippet[:max_chars], first_line + start, end_line

    # ------------------------------------------------------------------
    # 10. find_files  (file/path discovery - never looks at file contents)
    # ------------------------------------------------------------------

    def find_files(self, pattern: str, limit: int = 50) -> dict:
        with self._state_lock:
            return self._find_files_unlocked(pattern, limit)

    def _find_files_unlocked(self, pattern: str, limit: int = 50) -> dict:
        """Find repository files by path/name. Uses the file list stored in the index."""
        if not isinstance(pattern, str) or not pattern.strip():
            return _error("invalid_input", "pattern must be a non-empty string")
        bad = _check_int("limit", limit, 1, MAX_FILE_RESULTS)
        if bad:
            return bad

        pattern = pattern.strip().replace("\\", "/").removeprefix("./").lower()
        paths, note = self._searchable_paths()
        is_glob = any(ch in pattern for ch in "*?[")

        matches = []
        for path in paths:
            lowered = path.lower()
            name = lowered.rsplit("/", 1)[-1]
            if is_glob:
                name_match = fnmatch.fnmatchcase(name, pattern)
                found = name_match or fnmatch.fnmatchcase(lowered, pattern)
            else:
                name_match = pattern in name
                found = name_match or pattern in lowered
            if found:
                matches.append((path, name_match))

        matches.sort(key=lambda item: (not item[1], item[0].count("/"), item[0]))
        matches = [path for path, _ in matches]
        result = {
            "pattern": pattern,
            "count": min(len(matches), limit),
            "total_matches": len(matches),
            "truncated": len(matches) > limit,
        }
        if note:
            result["note"] = note
        result["files"] = matches[:limit]
        return result

    def _searchable_paths(self) -> tuple[list[str], str | None]:
        if self.index.repo_files is not None:
            return self.index.repo_files, None
        if not self.index.files:
            return [], None
        return [f.path for f in self.index.files], (
            "this index has no full file list, so only indexed Python files were searched; "
            "non-Python files (README, .md, .toml ...) cannot be found until the index is rebuilt "
            "with scripts/build_index.py"
        )

    # ------------------------------------------------------------------
    # 2. find_symbol
    # ------------------------------------------------------------------

    def find_symbol(
        self, query: str, kind: str | None = None, limit: int = 20
    ) -> dict:
        with self._state_lock:
            return self._find_symbol_unlocked(query, kind, limit)

    def _find_symbol_unlocked(
        self, query: str, kind: str | None = None, limit: int = 20
    ) -> dict:
        if not isinstance(query, str) or not query.strip():
            return _error("invalid_input", "query must be a non-empty string")
        bad = _check_int("limit", limit, 1, 100)
        if bad:
            return bad
        matches, match_type = self._find_matches(query.strip())
        if kind is not None:
            matches = [m for m in matches if m.kind == kind]
        if not matches:
            return _error(
                "symbol_not_found",
                f"no symbol matches {query!r}",
                suggestions=self._suggestions(query.strip()),
            )
        return {
            "query": query,
            "match_type": match_type,
            "count": len(matches),
            "ambiguous": len(matches) > 1,
            "matches": [self._summary(m) for m in matches[:limit]],
        }

    # ------------------------------------------------------------------
    # 3./4. find_callers / find_callees
    # ------------------------------------------------------------------

    def find_callers(self, symbol: str, limit: int = 50) -> dict:
        with self._state_lock:
            return self._find_callers_unlocked(symbol, limit)

    def _find_callers_unlocked(self, symbol: str, limit: int = 50) -> dict:
        bad = _check_int("limit", limit, 1, 500)
        if bad:
            return bad
        sym, err = self._resolve_one(symbol, CALLABLE_KINDS)
        if err:
            return err
        callers = self._definite_callers(sym.id)
        possible = self._possible_callers(sym)
        result = {
            "symbol": sym.id,
            "kind": sym.kind,
            "callers": callers[:limit],
            "possible_callers": possible[:limit],
            "truncated": len(callers) > limit or len(possible) > limit,
        }
        if possible:
            result["possible_callers_note"] = (
                f"possible_callers call something named {sym.name!r} on a variable whose type is not "
                "known statically; they may or may not call this symbol. `callers` are definite."
            )
        return result

    def find_callees(self, symbol: str) -> dict:
        with self._state_lock:
            return self._find_callees_unlocked(symbol)

    def _find_callees_unlocked(self, symbol: str) -> dict:
        sym, err = self._resolve_one(symbol)
        if err:
            return err
        definite, ambiguous, external = self._calls_from(sym.id)
        return {
            "symbol": sym.id,
            "kind": sym.kind,
            "callees": [
                {
                    **self._site(e.target, e.lineno),
                    "kind": self.index.symbols[e.target].kind,
                }
                for e in definite
            ],
            "possible_callees": [
                {"name": e.target, "line": e.lineno, "reason": _POSSIBLE_REASON}
                for e in ambiguous
            ],
            "external_calls": [
                {"name": e.target, "line": e.lineno} for e in external
            ],
        }

    # ------------------------------------------------------------------
    # 5. find_importers
    # ------------------------------------------------------------------

    def find_importers(self, target: str) -> dict:
        with self._state_lock:
            return self._find_importers_unlocked(target)

    def _find_importers_unlocked(self, target: str) -> dict:
        module, err = self._resolve_one(target, ("module",))
        if err:
            return err
        return {
            "module": module.id,
            "file": module.file,
            "importers": self._importers(module.id),
        }

    def _importers(self, module_id: str) -> list[dict]:
        seen, out = set(), []
        for e in self.graph.find_importers(module_id):
            if (e.source, e.lineno) not in seen:
                seen.add((e.source, e.lineno))
                out.append(
                    {
                        "module": e.source,
                        "file": self._path_by_module.get(e.source),
                        "line": e.lineno,
                    }
                )
        return sorted(out, key=lambda i: (i["file"] or "", i["line"] or 0))

    # ------------------------------------------------------------------
    # 6. get_file
    # ------------------------------------------------------------------

    def get_file(
        self,
        path: str,
        start_line: int | None = None,
        end_line: int | None = None,
        max_lines: int = 300,
    ) -> dict:
        return self._measure_runtime(
            "get_file", self._get_file, path, start_line, end_line, max_lines
        )

    def _get_file(
        self,
        path: str,
        start_line: int | None = None,
        end_line: int | None = None,
        max_lines: int = 300,
    ) -> dict:
        if not isinstance(path, str) or not path.strip():
            return _error("invalid_input", "path must be a non-empty string")
        bad = _check_int("max_lines", max_lines, 1, 5000)
        if bad:
            return bad
        for name, value in (("start_line", start_line), ("end_line", end_line)):
            if value is not None and (bad := _check_int(name, value, 1)):
                return bad
        if start_line is not None and end_line is not None and end_line < start_line:
            return _error(
                "invalid_range",
                "end_line must be greater than or equal to start_line",
            )

        root = Path(self.index.root).resolve()
        if is_ignored_path(Path(path.replace("\\", "/"))):
            return _error(
                "protected_path",
                "agent file tools cannot access ignored repository metadata or cache paths",
            )
        try:
            unresolved = Path(path) if Path(path).is_absolute() else root / path
            unresolved = Path(os.path.abspath(unresolved))
            if not is_within(root, unresolved):
                return _error(
                    "path_outside_repository", f"{path!r} is outside the repository"
                )
            current = root
            for part in unresolved.relative_to(root).parts:
                current = current / part
                if current.is_symlink():
                    return _error(
                        "invalid_path", "refusing to read through a symlink"
                    )
                if is_virtualenv_root(current):
                    return _error(
                        "protected_path",
                        "agent file tools cannot access virtual environment directories",
                    )
            candidate = unresolved.resolve()
        except (OSError, RuntimeError, ValueError):
            return _error("invalid_path", "path contains invalid characters")
        if not is_within(root, candidate):
            return _error(
                "path_outside_repository", f"{path!r} is outside the repository"
            )
        if is_ignored_path(candidate.relative_to(root)):
            return _error(
                "protected_path",
                "agent file tools cannot access ignored repository metadata or cache paths",
            )
        if not candidate.is_file():
            return _error(
                "file_not_found", f"no such file in the repository: {path!r}"
            )

        try:
            if candidate.suffix.casefold() == ".py":
                import tokenize as py_tokenize

                with py_tokenize.open(candidate) as source_file:
                    selected, total, truncated = _read_line_window(
                        source_file, start_line or 1, end_line, max_lines
                    )
            else:
                with candidate.open("r", encoding="utf-8", errors="replace") as source_file:
                    selected, total, truncated = _read_line_window(
                        source_file, start_line or 1, end_line, max_lines
                    )
        except (OSError, UnicodeError, SyntaxError) as exc:
            return _error("file_read_failed", f"could not read {path!r}: {exc}")
        start = 1 if start_line is None else start_line
        if total == 0:
            return {
                "file": candidate.relative_to(root).as_posix(),
                "line_start": None,
                "line_end": None,
                "total_lines": 0,
                "truncated": False,
                "content": "",
            }
        if start > total:
            return _error(
                "invalid_range",
                f"line range {start_line}-{end_line} is invalid for a file with {total} lines",
            )
        end = start + len(selected) - 1
        return {
            "file": candidate.relative_to(root).as_posix(),
            "line_start": start,
            "line_end": end,
            "total_lines": total,
            "truncated": truncated,
            **({"next_start_line": end + 1} if truncated else {}),
            "content": "\n".join(selected),
        }

    # ------------------------------------------------------------------
    # write_file
    # ------------------------------------------------------------------

    def write_file(self, path: str, content: str) -> dict:
        """Create or overwrite a file at a repository-relative path.

        Uses the same root-containment check as get_file. Parent directories
        are created automatically so a new module can be added without a
        separate directory-creation step.
        """
        if not isinstance(path, str) or not path.strip():
            return _error("invalid_input", "path must be a non-empty string")
        if not isinstance(content, str):
            return _error("invalid_input", "content must be a string")

        with self.transaction_lock():
            root = Path(self.index.root).resolve()
            if is_ignored_path(Path(path.replace("\\", "/"))):
                return _error(
                    "protected_path",
                    "agent file tools cannot access ignored repository metadata or cache paths",
                )
            try:
                unresolved = Path(path) if Path(path).is_absolute() else root / path
                unresolved = Path(os.path.abspath(unresolved))
                if not is_within(root, unresolved):
                    return _error(
                        "path_outside_repository", f"{path!r} is outside the repository"
                    )
                current = root
                for part in unresolved.relative_to(root).parts:
                    current = current / part
                    if current.is_symlink():
                        return _error(
                            "invalid_path",
                            "refusing to write through or replace a symlink",
                        )
                    if is_virtualenv_root(current):
                        return _error(
                            "protected_path",
                            "agent file tools cannot modify virtual environment directories",
                        )
                candidate = unresolved.resolve()
            except (OSError, RuntimeError, ValueError):
                return _error("invalid_path", "path contains invalid characters")
            if not is_within(root, candidate):
                return _error(
                    "path_outside_repository", f"{path!r} is outside the repository"
                )
            rel_parts = candidate.relative_to(root).parts
            if any(is_ignored_name(part) for part in rel_parts):
                return _error(
                    "protected_path",
                    f"{path!r} is inside an ignored metadata/cache directory",
                )
            if any(
                (part.casefold() if os.name == "nt" else part) in PROTECTED_PATH_PARTS
                for part in rel_parts
            ):
                return _error(
                    "protected_path",
                    f"{path!r} is inside version-control metadata and cannot be modified",
                )
            if candidate.is_dir():
                return _error("invalid_path", f"{path!r} is a directory, not a file")

            existed = candidate.is_file() and not candidate.is_symlink()
            backup_dir = None
            backup_path = None
            try:
                if existed:
                    backup_dir = Path(tempfile.mkdtemp(prefix="agentrepo_write_backup_"))
                    backup_path = backup_dir / candidate.name
                    shutil.copy2(candidate, backup_path)
                candidate.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(
                    "w",
                    encoding="utf-8",
                    dir=candidate.parent,
                    prefix=f".{candidate.name}.",
                    suffix=".tmp",
                    delete=False,
                ) as temp:
                    temp_path = Path(temp.name)
                    temp.write(content)
                    temp.flush()
                    os.fsync(temp.fileno())
                if existed:
                    os.chmod(temp_path, stat.S_IMODE(candidate.stat().st_mode))
                os.replace(temp_path, candidate)
            except OSError as exc:
                try:
                    if "temp_path" in locals():
                        temp_path.unlink(missing_ok=True)
                except OSError:
                    pass
                if backup_dir is not None:
                    shutil.rmtree(backup_dir, ignore_errors=True)
                return _error("file_write_failed", f"could not write {path!r}: {exc}")
            rel_path = candidate.relative_to(root).as_posix()
            try:
                refresh = self.refresh()
            except Exception as exc:
                refresh = _error("index_refresh_failed", str(exc))
            if "error" in refresh:
                try:
                    if existed and backup_path is not None:
                        shutil.copy2(backup_path, candidate)
                    else:
                        candidate.unlink(missing_ok=True)
                    self.refresh()
                except Exception as restore_exc:
                    return _error(
                        "index_refresh_failed",
                        f"write refresh failed ({refresh['error']}); file rollback failed ({restore_exc})",
                        recovery_backup=str(backup_path) if backup_path else None,
                    )
                if backup_dir is not None:
                    shutil.rmtree(backup_dir, ignore_errors=True)
                return _error(
                    "index_refresh_failed",
                    f"write was reverted because repository index refresh failed: {refresh['error']}",
                )
            if backup_dir is not None:
                shutil.rmtree(backup_dir, ignore_errors=True)
            return {
                "file": rel_path,
                "created": not existed,
                "bytes_written": len(content.encode("utf-8")),
                "line_count": len(content.splitlines()),
            }

    # ------------------------------------------------------------------
    # Checkpoint, diff, and rollback
    #
    # These are internal safety primitives called by the deterministic
    # MutationTransaction layer in agent/safety.py around write_file,
    # verify_repository, and run_command.
    # ------------------------------------------------------------------

    def _iter_repo_files(self, root: Path) -> list[Path]:
        """Return regular files and symlinks without following symlinks or entering ignored/venv dirs."""
        entries = []

        def raise_walk_error(error):
            raise error

        for directory, directories, names in os.walk(
            root, followlinks=False, onerror=raise_walk_error
        ):
            current = Path(directory)
            symlink_directories = [
                name for name in directories if (current / name).is_symlink()
            ]
            directories[:] = [
                name
                for name in directories
                if not is_ignored_name(name)
                and name not in symlink_directories
                and not is_virtualenv_root(current / name)
            ]
            entries.extend(
                current / name
                for name in symlink_directories
                if not is_ignored_name(name)
            )
            entries.extend(
                current / name
                for name in names
                if not is_ignored_path((current / name).relative_to(root))
                and ((current / name).is_symlink() or (current / name).is_file())
            )
        return entries

    def _iter_repo_directories(self, root: Path) -> set[str]:
        result: set[str] = set()

        def raise_walk_error(error):
            raise error

        for directory, dirs, _ in os.walk(
            root, followlinks=False, onerror=raise_walk_error
        ):
            current = Path(directory)
            symlinks = [name for name in dirs if (current / name).is_symlink()]
            dirs[:] = [
                name
                for name in dirs
                if not is_ignored_name(name)
                and name not in symlinks
                and not is_virtualenv_root(current / name)
            ]
            result.update(
                (current / name).relative_to(root).as_posix() for name in dirs
            )
        return result

    @staticmethod
    def _entry_signature(path: Path) -> tuple[str, bytes, int]:
        """Entry identity includes its type, data, and permission mode; symlinks are never followed."""
        if path.is_symlink():
            return "symlink", os.readlink(path).encode("utf-8", errors="surrogateescape"), 0
        if path.is_dir():
            return "directory", b"", 0
        mode = stat.S_IMODE(path.lstat().st_mode) if os.name != "nt" else 0
        return "file", path.read_bytes(), mode

    def _tree_signatures(self, root: Path) -> dict[str, tuple[str, bytes]]:
        paths = self._iter_repo_files(root) + [
            root / rel for rel in self._iter_repo_directories(root)
        ]
        return {
            p.relative_to(root).as_posix(): self._entry_signature(p) for p in paths
        }

    @staticmethod
    def _has_symlink_parent(root: Path, path: Path) -> bool:
        current = root
        for part in path.relative_to(root).parts[:-1]:
            current = current / part
            if current.is_symlink():
                return True
        return False

    def create_checkpoint(self, label: str | None = None) -> dict:
        return self._measure_runtime("create_checkpoint", self._create_checkpoint, label)

    def _create_checkpoint(self, label: str | None = None) -> dict:
        """Snapshot the current repository state. Returns {"checkpoint_id": ...}."""
        with self.transaction_lock():
            root = Path(self.index.root).resolve()
            if not root.is_dir():
                return _error(
                    "repository_not_found", f"repository root does not exist: {root}"
                )

            checkpoint_id = uuid.uuid4().hex[:12]
            snapshot_dir = None
            try:
                snapshot_dir = Path(tempfile.mkdtemp(prefix="agentrepo_checkpoint_"))
                files = self._iter_repo_files(root)
                directories = self._iter_repo_directories(root)
                for rel in sorted(directories, key=lambda value: (value.count("/"), value)):
                    (snapshot_dir / rel).mkdir(parents=True, exist_ok=True)
                for f in files:
                    rel = f.relative_to(root)
                    dest = snapshot_dir / rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(f, dest, follow_symlinks=False)
            except OSError as exc:
                if snapshot_dir is not None:
                    shutil.rmtree(snapshot_dir, ignore_errors=True)
                return _error(
                    "checkpoint_creation_failed",
                    f"could not create repository checkpoint: {exc}",
                )

            with self._checkpoint_lock:
                self._checkpoints[checkpoint_id] = {
                    "id": checkpoint_id,
                    "label": label,
                    "created_at": time.time(),
                    "root": str(root),
                    "snapshot_dir": str(snapshot_dir),
                    "file_count": len(files),
                    "directories": sorted(directories),
                }
            return {
                "checkpoint_id": checkpoint_id,
                "file_count": len(files),
                "pre_existing_git_state": self._git_status(root),
            }

    @staticmethod
    def _terminate_process_tree(proc: subprocess.Popen) -> None:
        if os.name == "nt":
            try:
                killed = subprocess.run(
                    ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=5,
                    check=False,
                )
                if killed.returncode != 0:
                    proc.kill()
            except (OSError, subprocess.TimeoutExpired):
                proc.kill()
        else:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except OSError:
                try:
                    proc.kill()
                except OSError:
                    pass
        try:
            proc.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            pass

    def _git_status(self, root: Path) -> dict | None:
        """Best-effort informational Git status using trusted system git."""
        try:
            clean_env, _ = _sanitized_command_env(root)
            git_exe = shutil.which("git", path=clean_env.get("PATH", ""))
            if not git_exe:
                return None
            proc = subprocess.Popen(
                [git_exe, "status", "--porcelain", "--untracked-files=all"],
                cwd=root,
                env=clean_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                start_new_session=(os.name != "nt"),
                creationflags=(
                    subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
                ),
            )
            output_state = {}
            reader = threading.Thread(
                target=_read_limited,
                args=(proc.stdout, 100_000, output_state),
                daemon=True,
            )
            reader.start()
            try:
                return_code = proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._terminate_process_tree(proc)
                return None
            finally:
                if proc.poll() is None:
                    self._terminate_process_tree(proc)
                reader.join(timeout=1)
                if proc.stdout is not None:
                    proc.stdout.close()
        except (FileNotFoundError, OSError):
            return None
        if return_code != 0:
            return None
        output = _command_text(output_state)
        dirty = [line[3:] for line in output.splitlines() if line.strip()]
        return {
            "clean": not dirty,
            "dirty_files": dirty,
            "truncated": output_state.get("total", 0) > 100_000,
        }

    def diff_since_checkpoint(self, checkpoint_id: str) -> dict:
        """Compare the current working tree to a checkpoint's snapshot."""
        with self.transaction_lock():
            with self._checkpoint_lock:
                checkpoint = self._checkpoints.get(checkpoint_id)
            if checkpoint is None:
                return _error(
                    "checkpoint_not_found", f"no checkpoint with id {checkpoint_id!r}"
                )

            root = Path(checkpoint["root"])
            snapshot_dir = Path(checkpoint["snapshot_dir"])
            try:
                before = self._tree_signatures(snapshot_dir)
                after = self._tree_signatures(root)
            except OSError as exc:
                return _error(
                    "checkpoint_diff_failed",
                    f"could not compare repository state: {exc}",
                )
            # Only report file/symlink entries in added/modified/deleted; directory
            # changes are reported in added_directories / removed_directories.
            before_files = {k: v for k, v in before.items() if v[0] != "directory"}
            after_files = {k: v for k, v in after.items() if v[0] != "directory"}
            added = sorted(set(after_files) - set(before_files))
            deleted = sorted(set(before_files) - set(after_files))
            modified = sorted(
                path
                for path in set(before_files) & set(after_files)
                if before_files[path] != after_files[path]
            )
            before_dirs = set(checkpoint.get("directories", []))
            try:
                after_dirs = self._iter_repo_directories(root)
            except OSError as exc:
                return _error(
                    "checkpoint_diff_failed",
                    f"could not inspect repository directories: {exc}",
                )
            added_dirs = sorted(after_dirs - before_dirs)
            removed_dirs = sorted(before_dirs - after_dirs)
            return {
                "checkpoint_id": checkpoint_id,
                "added": added,
                "modified": modified,
                "deleted": deleted,
                "added_directories": added_dirs,
                "removed_directories": removed_dirs,
                "changed": bool(added or modified or deleted or added_dirs or removed_dirs),
            }

    @staticmethod
    def _restore_entry_safely(root: Path, src: Path, dest: Path, kind: str) -> None:
        """Restore a single regular file or symlink from `src` to `dest` without
        following symlinks or writing outside `root`."""
        unresolved_dest = Path(os.path.abspath(dest))
        if not is_within(root, unresolved_dest):
            raise OSError(f"refusing to restore {dest} outside repository root {root}")
        if RepoService._has_symlink_parent(root, unresolved_dest):
            raise OSError("refusing to restore through a symlinked parent directory")
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.is_symlink() or dest.is_file():
            dest.unlink()
        if kind == "symlink":
            link_target = os.readlink(src)
            os.symlink(link_target, dest)
        else:
            shutil.copy2(src, dest, follow_symlinks=False)

    def rollback_to_checkpoint(self, checkpoint_id: str) -> dict:
        """Undo every change made since the checkpoint."""
        with self.transaction_lock():
            diff = self.diff_since_checkpoint(checkpoint_id)
            if "error" in diff:
                return diff
            with self._checkpoint_lock:
                checkpoint = self._checkpoints.get(checkpoint_id)
            if checkpoint is None:
                return _error(
                    "checkpoint_not_found", f"no checkpoint with id {checkpoint_id!r}"
                )
            root = Path(checkpoint["root"]).resolve()
            snapshot_dir = Path(checkpoint["snapshot_dir"]).resolve()
            try:
                before = self._tree_signatures(snapshot_dir)
                after = self._tree_signatures(root)
            except OSError as exc:
                return {
                    "checkpoint_id": checkpoint_id,
                    "restored_files": [],
                    "removed_files": [],
                    "removed_directories": [],
                    "created_directories": [],
                    "errors": [{"file": "<repository-tree>", "error": str(exc)}],
                    "rolled_back": False,
                }
            changed_paths = {
                p for p in set(before) | set(after) if before.get(p) != after.get(p)
            }
            roots: list[str] = []
            for rel in sorted(changed_paths, key=lambda value: (value.count("/"), value)):
                if not any(rel.startswith(parent + "/") for parent in roots):
                    roots.append(rel)
            restored: list[str] = []
            removed: list[str] = []
            errors: list[dict] = []
            removed_directories: list[str] = []
            created_directories: list[str] = []
            for rel in roots:
                dest = root / rel
                try:
                    if not is_within(root, Path(os.path.abspath(dest))):
                        raise OSError(f"refusing to restore path outside repository: {rel}")
                    if self._has_symlink_parent(root, dest):
                        raise OSError(
                            "refusing to restore through a symlinked parent directory"
                        )
                    if dest.is_symlink() or dest.is_file():
                        dest.unlink()
                        if rel not in before or before[rel][0] == "directory":
                            removed.append(rel)
                    elif dest.is_dir():
                        self._remove_tree_without_ignored(dest, root, removed)
                        if dest.exists() and any(dest.iterdir()):
                            raise OSError("ignored contents remain under changed directory")
                        if dest.exists():
                            dest.rmdir()
                        removed_directories.append(rel)
                    baseline = before.get(rel)
                    if baseline is None:
                        continue
                    src = snapshot_dir / rel
                    if baseline[0] == "directory":
                        dest.mkdir(parents=True, exist_ok=True)
                        created_directories.append(rel)
                    else:
                        self._restore_entry_safely(root, src, dest, baseline[0])
                        restored.append(rel)
                    prefix = rel + "/"
                    for child, signature in sorted(
                        before.items(), key=lambda pair: pair[0].count("/")
                    ):
                        if not child.startswith(prefix):
                            continue
                        child_src, child_dest = snapshot_dir / child, root / child
                        if signature[0] == "directory":
                            if self._has_symlink_parent(root, child_dest):
                                raise OSError(
                                    "refusing to restore directory through symlinked parent"
                                )
                            child_dest.mkdir(parents=True, exist_ok=True)
                        else:
                            self._restore_entry_safely(
                                root, child_src, child_dest, signature[0]
                            )
                            restored.append(child)
                except OSError as exc:
                    errors.append({"file": rel, "error": str(exc)})

            refresh = self.refresh()
            if "error" in refresh:
                errors.append({"file": "<repository-index>", "error": refresh["error"]})

            return {
                "checkpoint_id": checkpoint_id,
                "restored_files": restored,
                "removed_files": removed,
                "removed_directories": removed_directories,
                "created_directories": created_directories,
                "errors": errors,
                "rolled_back": not errors,
            }

    def _remove_tree_without_ignored(
        self,
        directory: Path,
        root: Path,
        removed_files: list[str] | None = None,
    ) -> None:
        """Remove ordinary descendants without following links or touching ignored trees."""
        for child in list(directory.iterdir()):
            if is_ignored_name(child.name) or (child.is_dir() and not child.is_symlink() and is_virtualenv_root(child)):
                continue
            if child.is_symlink() or not child.is_dir():
                rel = child.relative_to(root).as_posix()
                child.unlink()
                if removed_files is not None and rel not in removed_files:
                    removed_files.append(rel)
            else:
                self._remove_tree_without_ignored(child, root, removed_files)
                if not any(child.iterdir()):
                    child.rmdir()

    def discard_checkpoint(self, checkpoint_id: str) -> dict:
        """Free the snapshot for a checkpoint that is no longer needed."""
        with self._checkpoint_lock:
            checkpoint = self._checkpoints.pop(checkpoint_id, None)
        if checkpoint is None:
            return _error(
                "checkpoint_not_found", f"no checkpoint with id {checkpoint_id!r}"
            )
        shutil.rmtree(checkpoint["snapshot_dir"], ignore_errors=True)
        return {"checkpoint_id": checkpoint_id, "discarded": True}

    def approve_mutation_checkpoint(self, mutation_info: dict | None) -> dict:
        """Approve a guarded mutation and discard its temporary checkpoint if still retained."""
        if not isinstance(mutation_info, dict):
            return {"approved": False, "error": "invalid mutation info"}
        cp_id = mutation_info.get("checkpoint_id")
        if isinstance(cp_id, str) and cp_id:
            with self._checkpoint_lock:
                checkpoint = self._checkpoints.pop(cp_id, None)
            if checkpoint is not None:
                shutil.rmtree(checkpoint["snapshot_dir"], ignore_errors=True)
        return {
            "approved": True,
            "checkpoint_id": cp_id,
            "changed_files": list(mutation_info.get("changed_files") or []),
        }

    def apply_mutation_rollback(self, mutation_info: dict | None) -> dict:
        """Roll back a guarded mutation either via an active checkpoint or recorded pre-mutation file snapshots."""
        if not isinstance(mutation_info, dict):
            return _error("invalid_input", "mutation_info must be a dictionary")
        cp_id = mutation_info.get("checkpoint_id")
        if isinstance(cp_id, str) and cp_id:
            with self._checkpoint_lock:
                has_cp = cp_id in self._checkpoints
            if has_cp:
                rb = self.rollback_to_checkpoint(cp_id)
                self.discard_checkpoint(cp_id)
                ref = self.refresh()
                return {**rb, "refresh": ref, "rolled_back": bool(rb.get("rolled_back", True))}

        file_diffs = mutation_info.get("file_diffs") or []
        if not file_diffs:
            return _error("rollback_unavailable", "no checkpoint or file snapshots available to roll back")

        restored_files: list[str] = []
        removed_files: list[str] = []
        errors: list[dict] = []
        with self.transaction_lock():
            root = Path(self.index.root).resolve()
            for item in file_diffs:
                if not isinstance(item, dict):
                    continue
                rel_path = str(item.get("file") or "").strip().replace("\\", "/")
                if not rel_path:
                    continue
                target = Path(os.path.abspath(root / rel_path))
                if not is_within(root, target):
                    errors.append({"file": rel_path, "error": "path outside repository"})
                    continue
                change_type = str(item.get("change_type") or "modified")
                before_content = item.get("before_content")
                try:
                    if change_type == "added" and before_content is None:
                        if target.is_file() or target.is_symlink():
                            target.unlink()
                            removed_files.append(rel_path)
                    elif isinstance(before_content, str):
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_text(before_content, encoding="utf-8")
                        restored_files.append(rel_path)
                except OSError as exc:
                    errors.append({"file": rel_path, "error": str(exc)})
            ref = self.refresh()
        return {
            "checkpoint_id": cp_id,
            "restored_files": restored_files,
            "removed_files": removed_files,
            "errors": errors,
            "refresh": ref,
            "rolled_back": not errors,
        }

    # ------------------------------------------------------------------
    # run_command (verification and command execution primitive, wrapped by
    # MutationTransaction in agent/safety.py for checkpointing and rollback)
    # ------------------------------------------------------------------

    def run_command(
        self, command: str | list[str], timeout: int = DEFAULT_COMMAND_TIMEOUT
    ) -> dict:
        return self._measure_runtime("run_command", self._run_command, command, timeout)

    def _run_command(
        self, command: str | list[str], timeout: int = DEFAULT_COMMAND_TIMEOUT
    ) -> dict:
        """Run a command with the repository root as its working directory.

        Uses platform-appropriate argv parsing with `shell=False`, strips
        API keys and secret environment variables from the child environment,
        redacts secret patterns from stdout/stderr, and sets `cwd` to the
        repository root. Because a working-directory constraint is not an
        OS-level sandbox, agent-facing command execution is gated by EXECUTE
        capability and wrapped by `MutationTransaction` (agent/safety.py),
        which enforces command restrictions, establishes a pre-command
        checkpoint, rolls back unauthorized file modifications when WRITE
        capability is absent, and verifies/rolls back changes when WRITE
        capability is present.
        """
        if isinstance(command, list):
            if not command or any(not isinstance(arg, str) for arg in command):
                return _error(
                    "invalid_input", "command argument list must contain strings"
                )
            display_command = (
                subprocess.list2cmdline(command)
                if os.name == "nt"
                else shlex.join(command)
            )
            args = command
        elif isinstance(command, str) and command.strip():
            display_command = command
            try:
                args = _split_command(command)
            except ValueError as exc:
                return _error("invalid_input", f"could not parse command: {exc}")
        else:
            return _error("invalid_input", "command must be a non-empty string")
        bad = _check_int("timeout", timeout, 1, MAX_COMMAND_TIMEOUT)
        if bad:
            return bad

        if not args:
            return _error("invalid_input", "command must be a non-empty string")

        with self.transaction_lock():
            root = Path(self.index.root).resolve()
            clean_env, secret_values = _sanitized_command_env(root)
            display_command = _redact_command_output(display_command, secret_values)

            exe_token = args[0].strip()
            if not exe_token:
                return _error("invalid_input", "command executable must be a non-empty string")
            if Path(exe_token).is_absolute():
                resolved_exe = os.path.abspath(exe_token)
            elif "/" in exe_token or "\\" in exe_token or exe_token.startswith("."):
                return _error(
                    "command_not_permitted",
                    f"relative or repository-local executable path {args[0]!r} is not permitted",
                )
            else:
                found_exe = shutil.which(exe_token, path=clean_env.get("PATH", ""))
                resolved_exe = os.path.abspath(found_exe) if found_exe else exe_token

            try:
                resolved_exe_path = Path(resolved_exe)
                if resolved_exe_path.is_absolute():
                    if is_within(root, Path(os.path.abspath(resolved_exe_path))) or is_within(
                        root, resolved_exe_path.resolve()
                    ):
                        return _error(
                            "command_not_permitted",
                            f"refusing to execute binary inside the target repository: {args[0]!r}",
                        )
            except OSError:
                pass

            exec_args = [resolved_exe, *args[1:]]
            try:
                proc = subprocess.Popen(
                    exec_args,
                    cwd=root,
                    env=clean_env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    start_new_session=(os.name != "nt"),
                    creationflags=(
                        subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
                    ),
                )
                stdout_state, stderr_state = {}, {}
                readers = [
                    threading.Thread(
                        target=_read_limited,
                        args=(proc.stdout, MAX_COMMAND_OUTPUT_CHARS, stdout_state),
                        daemon=True,
                    ),
                    threading.Thread(
                        target=_read_limited,
                        args=(proc.stderr, MAX_COMMAND_OUTPUT_CHARS, stderr_state),
                        daemon=True,
                    ),
                ]
                for reader in readers:
                    reader.start()
                timed_out = False
                return_code = None
                try:
                    try:
                        return_code = proc.wait(timeout=timeout)
                    except subprocess.TimeoutExpired:
                        timed_out = True
                        self._terminate_process_tree(proc)
                    else:
                        if os.name != "nt":
                            try:
                                os.killpg(proc.pid, signal.SIGKILL)
                            except OSError:
                                pass
                finally:
                    if proc.poll() is None:
                        self._terminate_process_tree(proc)
                    for reader in readers:
                        reader.join(timeout=1)
                    if proc.stdout is not None:
                        proc.stdout.close()
                    if proc.stderr is not None:
                        proc.stderr.close()

                if timed_out:
                    return {
                        "command": display_command,
                        "exit_code": None,
                        "timed_out": True,
                        "stdout": _redact_command_output(
                            _command_text(stdout_state), secret_values
                        ),
                        "stderr": _redact_command_output(
                            _command_text(stderr_state), secret_values
                        ),
                    }
            except FileNotFoundError:
                return _error(
                    "command_not_found", f"executable not found: {args[0]!r}"
                )
            except OSError as exc:
                return _error(
                    "command_execution_failed", f"could not start {args[0]!r}: {exc}"
                )

            return {
                "command": display_command,
                "exit_code": return_code,
                "timed_out": False,
                "stdout": _redact_command_output(
                    _command_text(stdout_state), secret_values
                ),
                "stderr": _redact_command_output(
                    _command_text(stderr_state), secret_values
                ),
            }

    # ------------------------------------------------------------------
    # 7. get_symbol
    # ------------------------------------------------------------------

    def get_symbol(self, symbol: str, max_lines: int = 200) -> dict:
        with self._state_lock:
            return self._get_symbol_unlocked(symbol, max_lines)

    def _get_symbol_unlocked(self, symbol: str, max_lines: int = 200) -> dict:
        bad = _check_int("max_lines", max_lines, 1, 5000)
        if bad:
            return bad
        sym, err = self._resolve_one(symbol, CALLABLE_KINDS)
        if err:
            if err["error"]["code"] == "wrong_kind":
                err["error"]["message"] += "; use get_file for whole modules"
            return err
        chunk = self.index.chunks.get(f"{sym.id}#chunk")
        if chunk is None:
            return _error("no_source", f"no source chunk stored for {sym.id}")
        lines = chunk.text.splitlines()
        return {
            "symbol": sym.id,
            "kind": sym.kind,
            "file": sym.file,
            "line_start": chunk.start_line,
            "line_end": chunk.end_line,
            "decorators": sym.decorators,
            "truncated": len(lines) > max_lines,
            "source": "\n".join(lines[:max_lines]),
        }

    # ------------------------------------------------------------------
    # 8. get_dependencies
    # ------------------------------------------------------------------

    def get_dependencies(self, target: str) -> dict:
        with self._state_lock:
            return self._get_dependencies_unlocked(target)

    def _get_dependencies_unlocked(self, target: str) -> dict:
        sym, err = self._resolve_one(target)
        if err:
            return err
        depends_on: dict[str, list[str]] = {}
        depended_on_by: dict = {}

        out_edges = self.graph.neighbors(sym.id)
        if sym.kind == "module":
            depends_on["imports"] = sorted(
                {e.target for e in out_edges if e.kind == "IMPORTS" and e.resolved}
            )
            depends_on["external_imports"] = sorted(
                {e.target for e in out_edges if e.kind == "IMPORTS" and not e.resolved}
            )
            depended_on_by["imported_by"] = sorted(
                {i["module"] for i in self._importers(sym.id)}
            )
        if sym.kind == "class":
            depends_on["bases"] = sorted(
                {e.target for e in out_edges if e.kind == "INHERITS" and e.resolved}
            )
            depends_on["external_bases"] = sorted(
                {e.target for e in out_edges if e.kind == "INHERITS" and not e.resolved}
            )
            depended_on_by["subclasses"] = sorted(
                {e.source for e in self.graph.find_subclasses(sym.id)}
            )
        if sym.kind != "class":
            definite, ambiguous, external = self._calls_from(sym.id)
            depends_on["calls"] = sorted({e.target for e in definite})
            depends_on["ambiguous_calls"] = sorted({e.target for e in ambiguous})
            depends_on["external_calls"] = sorted({e.target for e in external})
        if sym.kind != "module":
            depended_on_by["callers"] = sorted(
                {c["symbol"] for c in self._definite_callers(sym.id)}
            )
            depended_on_by["possible_callers_count"] = len(self._possible_callers(sym))

        legend = {
            "imports": "repository modules this module imports",
            "external_imports": "imports of stdlib/third-party/unresolved modules",
            "imported_by": "repository modules that import this module",
            "bases": "repository classes this class inherits from",
            "external_bases": "base classes that could not be resolved inside the repository",
            "subclasses": "repository classes that inherit from this class",
            "calls": "repository symbols called with a definite target",
            "ambiguous_calls": "attribute-call names whose receiver type is unknown (target not determinable)",
            "external_calls": "calls to stdlib/third-party/unresolved names",
            "callers": "repository symbols that definitely call (or instantiate) this symbol",
            "possible_callers_count": "number of unresolved same-name attribute calls; see find_callers",
        }
        used = {
            k: v for k, v in legend.items() if k in depends_on or k in depended_on_by
        }
        return {
            "symbol": sym.id,
            "kind": sym.kind,
            "file": sym.file,
            "depends_on": depends_on,
            "depended_on_by": depended_on_by,
            "legend": used,
        }

    # ------------------------------------------------------------------
    # 9. get_ast
    # ------------------------------------------------------------------

    def get_ast(self, target: str) -> dict:
        with self._state_lock:
            return self._get_ast_unlocked(target)

    def _get_ast_unlocked(self, target: str) -> dict:
        sym, err = self._resolve_one(target)
        if err:
            return err
        result = {
            "symbol": sym.id,
            "kind": sym.kind,
            "node_type": {"module": "Module", "class": "ClassDef"}.get(
                sym.kind, "FunctionDef"
            ),
            "file": sym.file,
            "line_start": sym.lineno,
            "line_end": sym.end_lineno,
            "decorators": sym.decorators,
            "docstring": sym.docstring,
            "children": [
                {
                    "symbol": e.target,
                    "kind": self.index.symbols[e.target].kind,
                    "line_start": self.index.symbols[e.target].lineno,
                    "line_end": self.index.symbols[e.target].end_lineno,
                }
                for e in self.graph.find_children(sym.id)
            ],
            "calls": [
                {
                    "name": e.target,
                    "line": e.lineno,
                    "resolution": "resolved" if e.resolved else e.confidence,
                }
                for e in self.graph.find_callees(sym.id)
            ],
        }
        out_edges = self.graph.neighbors(sym.id)
        if sym.kind == "module":
            imports, seen = [], set()
            for e in out_edges:
                if e.kind == "IMPORTS" and (e.target, e.lineno) not in seen:
                    seen.add((e.target, e.lineno))
                    imports.append(
                        {"module": e.target, "line": e.lineno, "resolved": e.resolved}
                    )
            result["imports"] = imports
        if sym.kind == "class":
            result["bases"] = sym.bases
            result["inherits"] = [
                {"target": e.target, "resolved": e.resolved}
                for e in out_edges
                if e.kind == "INHERITS"
            ]
        if sym.kind in ("function", "method", "class"):
            result.update(self._parse_definition(sym))
        return result

    def _parse_definition(self, sym: Symbol) -> dict:
        chunk = self.index.chunks.get(f"{sym.id}#chunk")
        if chunk is None:
            return {}
        try:
            node = ast.parse(textwrap.dedent(chunk.text)).body[0]
        except (SyntaxError, IndexError):
            return {}
        info = {"node_type": type(node).__name__}
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            a = node.args
            info["args"] = (
                [x.arg for x in a.posonlyargs + a.args]
                + (["*" + a.vararg.arg] if a.vararg else [])
                + [x.arg for x in a.kwonlyargs]
                + (["**" + a.kwarg.arg] if a.kwarg else [])
            )
            info["returns"] = ast.unparse(node.returns) if node.returns else None
        return info
