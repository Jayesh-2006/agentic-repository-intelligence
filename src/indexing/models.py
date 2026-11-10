"""Core data models for repository indexing, symbols, edges, and text chunks."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional, Literal


SymbolKind = Literal["module", "class", "function", "method"]
ChunkKind = Literal["module", "class", "function", "method", "text"]  # "text" = a slice of a non-Python text file
EdgeKind = Literal["CONTAINS", "IMPORTS", "INHERITS", "CALLS"]
# Only set (and only meaningful) when Edge.resolved is False:
#   "ambiguous" -> the receiver's type is unknown (e.g. `obj.save()`), so we
#                  genuinely cannot tell which implementation would run.
#   "external"  -> the receiver/target is known but not defined inside this
#                  repository (stdlib, third-party, or an unresolved import).
CallConfidence = Literal["ambiguous", "external"]


@dataclass
class Symbol:
    """A single named entity in the codebase: a module, class, function or method."""

    id: str  # fully-qualified name, e.g. "shop.models.User.save"
    kind: SymbolKind
    name: str  # short name, e.g. "save"
    file: str  # path relative to repo root
    lineno: int
    end_lineno: Optional[int]
    parent: Optional[str]  # id of enclosing symbol (module/class), None for top-level module
    decorators: list[str] = field(default_factory=list)
    bases: list[str] = field(default_factory=list)  # for classes: raw or resolved base names
    docstring: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Edge:
    """A directed relationship between two symbols."""

    source: str  # symbol id
    target: str  # symbol id, OR raw name string if unresolved
    kind: EdgeKind
    resolved: bool = True  # False when target is a best-effort guess / raw name
    confidence: Optional[CallConfidence] = None  # only meaningful for CALLS edges
    lineno: Optional[int] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Chunk:
    """A retrievable slice of source code, used later by retrieval tools."""

    id: str  # e.g. "shop.models.User.save#chunk"
    symbol_id: Optional[str]  # None for a module-level chunk (imports/top-level code)
    file: str
    kind: ChunkKind
    start_line: int
    end_line: int
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FileRecord:
    """Metadata about a single indexed file."""

    path: str  # relative to repo root, e.g. "shop/models.py"
    module: str  # dotted module name, e.g. "shop.models"
    loc: int  # lines of code
    parse_ok: bool
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RepositoryIndex:
    """The full result of indexing a repository."""

    root: str
    files: list[FileRecord] = field(default_factory=list)
    symbols: dict[str, Symbol] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)
    chunks: dict[str, Chunk] = field(default_factory=dict)
    # Every non-ignored file in the repository (any type, repo-relative, sorted).
    # `files` above only holds parsed Python modules; this is what find_files searches.
    #
    # None is a deliberate sentinel distinct from an empty list: it means no
    # full file listing was recorded for this index (see persistence.load_index).
    # An empty list means a
    # full listing WAS taken and the repository genuinely had no files at
    # that time - a real, current index of an empty repository. Collapsing
    # these two into the same falsy `[]` is exactly the bug this sentinel
    # fixes (see RepoService._searchable_paths, write_file and
    # rollback_to_checkpoint, which all check `is not None`, never
    # truthiness). index_repository() always produces a real list (possibly
    # empty) here, never None; None only ever comes from loading a legacy index.
    repo_files: Optional[list[str]] = None

    def to_dict(self) -> dict:
        return {
            "root": self.root,
            "files": [f.to_dict() for f in self.files],
            "repo_files": list(self.repo_files) if self.repo_files is not None else None,
            "symbols": {k: v.to_dict() for k, v in self.symbols.items()},
            "edges": [e.to_dict() for e in self.edges],
            "chunks": {k: v.to_dict() for k, v in self.chunks.items()},
        }
