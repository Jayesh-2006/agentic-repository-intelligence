"""In-memory bidirectional adjacency-list graph over repository symbols."""

from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict

from .models import Edge, EdgeKind, RepositoryIndex


@dataclass
class CodeGraph:
    nodes: set[str]
    _out: dict[str, list[Edge]]
    _in: dict[str, list[Edge]]

    def __init__(self) -> None:
        self.nodes = set()
        self._out = defaultdict(list)
        self._in = defaultdict(list)

    # -- construction -----------------------------------------------------

    @classmethod
    def from_index(cls, index: RepositoryIndex) -> "CodeGraph":
        """Build the graph from an index's symbols and edges. Used both after
        indexing and after loading a persisted index, so both behave the same."""
        graph = cls()
        for symbol_id in index.symbols:
            graph.add_node(symbol_id)
        for edge in index.edges:
            graph.add_edge(edge)
        return graph

    def add_node(self, node_id: str) -> None:
        self.nodes.add(node_id)

    def add_edge(self, edge: Edge) -> None:
        self.add_node(edge.source)
        # Only register the target as a graph node if it's a resolved
        # symbol id. Unresolved targets (raw names like "save") are kept
        # on the edge itself but are not first-class nodes.
        if edge.resolved:
            self.add_node(edge.target)
        self._out[edge.source].append(edge)
        self._in[edge.target].append(edge)

    # -- queries ------------------------------------------------------------

    def neighbors(self, node_id: str) -> list[Edge]:
        """All outgoing edges from a node."""
        return list(self._out.get(node_id, []))

    def find_callers(self, symbol_id: str) -> list[Edge]:
        """Who calls this symbol. Includes ambiguous (unresolved) matches
        where the raw call name equals the symbol's short name, since those
        may or may not actually be a call to this symbol."""
        return [e for e in self._in.get(symbol_id, []) if e.kind == "CALLS"]

    def find_callees(self, symbol_id: str) -> list[Edge]:
        """What this symbol calls."""
        return [e for e in self._out.get(symbol_id, []) if e.kind == "CALLS"]

    def find_importers(self, module_id: str) -> list[Edge]:
        """Which modules import this module."""
        return [e for e in self._in.get(module_id, []) if e.kind == "IMPORTS"]

    def find_children(self, symbol_id: str) -> list[Edge]:
        """Direct children in the containment tree (module->class->method)."""
        return [e for e in self._out.get(symbol_id, []) if e.kind == "CONTAINS"]

    def find_subclasses(self, class_id: str) -> list[Edge]:
        """Classes that inherit from this class."""
        return [e for e in self._in.get(class_id, []) if e.kind == "INHERITS"]

    def edges_of_kind(self, kind: EdgeKind) -> list[Edge]:
        out = []
        for edges in self._out.values():
            out.extend(e for e in edges if e.kind == kind)
        return out

    def all_edges(self) -> list[Edge]:
        out = []
        for edges in self._out.values():
            out.extend(edges)
        return out
