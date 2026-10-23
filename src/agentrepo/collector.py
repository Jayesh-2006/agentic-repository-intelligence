"""Single-pass Python AST symbol, import, inheritance, and call collector."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Optional

from .models import Symbol


@dataclass
class RawImport:
    lineno: int
    alias: str  # the local name this import is bound to
    target_module: str  # dotted module the alias refers to
    imported_name: Optional[str]  # for `from X import Y`, this is Y; else None


@dataclass
class RawBase:
    class_id: str
    raw_expr: str  # source text of the base expression, e.g. "Base" or "pkg.Base"
    lineno: int


@dataclass
class RawCall:
    source_id: str  # symbol id of the function/method the call happens inside
    shape: str  # "name" | "self_attr" | "obj_attr" | "complex"
    lineno: int
    name: Optional[str] = None  # for "name": the called name
    obj_name: Optional[str] = None  # for "obj_attr": the receiver variable name
    attr: Optional[str] = None  # for "self_attr" / "obj_attr" / "complex": the attribute


@dataclass
class ModuleCollection:
    module: str
    file_path: str
    symbols: dict[str, Symbol] = field(default_factory=dict)
    imports: list[RawImport] = field(default_factory=list)
    bases: list[RawBase] = field(default_factory=list)
    calls: list[RawCall] = field(default_factory=list)
    # name -> most recently declared qualified id in this file. This remains
    # the base-class lookup map; call resolution uses the scoped bindings map.
    local_defs: dict[str, str] = field(default_factory=dict)
    # All declarations remain in symbols; this map records the currently
    # bound declaration in each lexical scope.
    bindings: dict[tuple[str, str], str] = field(default_factory=dict)


class _Collector(ast.NodeVisitor):
    def __init__(self, module: str, file_path: str, source_lines: list[str]):
        self.result = ModuleCollection(module=module, file_path=file_path)
        self.source_lines = source_lines
        self._is_package = file_path.rsplit("/", 1)[-1].casefold() == "__init__.py"
        # stack of enclosing symbol ids (module, then classes/functions)
        self._scope_stack: list[str] = [module]
        # stack of enclosing *class* ids, so we know what "self" means
        # inside the current function (None entries for non-method funcs)
        self._class_stack: list[Optional[str]] = [None]

        module_symbol = Symbol(
            id=module,
            kind="module",
            name=module.rsplit(".", 1)[-1] if module else file_path,
            file=file_path,
            lineno=1,
            end_lineno=len(source_lines),
            parent=None,
        )
        self.result.symbols[module] = module_symbol

    # -- helpers --------------------------------------------------------

    @property
    def current_scope(self) -> str:
        return self._scope_stack[-1]

    @property
    def current_class(self) -> Optional[str]:
        return self._class_stack[-1]

    def _qualify(self, name: str) -> str:
        return f"{self.current_scope}.{name}"

    def _unique_declaration_id(self, name: str, lineno: int) -> str:
        base = self._qualify(name)
        if base not in self.result.symbols:
            return base
        candidate = f"{base}@{lineno}"
        suffix = 2
        while candidate in self.result.symbols:
            candidate = f"{base}@{lineno}_{suffix}"
            suffix += 1
        return candidate

    @staticmethod
    def _docstring(node) -> Optional[str]:
        doc = ast.get_docstring(node, clean=True)

        if not doc:
            return None

        lines = doc.strip().splitlines()
        if not lines:
            return None

        return lines[0][:200]

    @staticmethod
    def _decorator_names(node) -> list[str]:
        names = []
        for dec in node.decorator_list:
            try:
                names.append(ast.unparse(dec))
            except Exception:
                names.append("<decorator>")
        return names

    # -- definitions ------------------------------------------------------

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        qid = self._unique_declaration_id(node.name, node.lineno)
        sym = Symbol(
            id=qid,
            kind="class",
            name=node.name,
            file=self.result.file_path,
            lineno=node.lineno,
            end_lineno=getattr(node, "end_lineno", None),
            parent=self.current_scope,
            decorators=self._decorator_names(node),
            docstring=self._docstring(node),
        )
        self.result.symbols[qid] = sym
        self.result.local_defs[node.name] = qid
        self.result.bindings[(self.current_scope, node.name)] = qid

        base_exprs = []
        for base in node.bases:
            try:
                raw_expr = ast.unparse(base)
            except Exception:
                raw_expr = "<base>"
            base_exprs.append(raw_expr)
            self.result.bases.append(RawBase(class_id=qid, raw_expr=raw_expr, lineno=node.lineno))
        sym.bases = base_exprs

        self._scope_stack.append(qid)
        self._class_stack.append(qid)
        for child in node.body:
            self.visit(child)
        self._class_stack.pop()
        self._scope_stack.pop()

    def _visit_function_like(self, node) -> None:
        qid = self._unique_declaration_id(node.name, node.lineno)
        kind = "method" if self.current_class is not None and self._scope_stack[-1] == self.current_class else "function"
        sym = Symbol(
            id=qid,
            kind=kind,
            name=node.name,
            file=self.result.file_path,
            lineno=node.lineno,
            end_lineno=getattr(node, "end_lineno", None),
            parent=self.current_scope,
            decorators=self._decorator_names(node),
            docstring=self._docstring(node),
        )
        self.result.symbols[qid] = sym
        self.result.local_defs[node.name] = qid
        self.result.bindings[(self.current_scope, node.name)] = qid

        self._scope_stack.append(qid)
        self._class_stack.append(None)  # a function's own body is not "in a class" for self-resolution
        for child in node.body:
            self.visit(child)
        self._class_stack.pop()
        self._scope_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function_like(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function_like(node)

    # -- imports ------------------------------------------------------------

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            local_name = alias.asname or alias.name.split(".")[0]
            self.result.imports.append(
                RawImport(lineno=node.lineno, alias=local_name, target_module=alias.name, imported_name=None)
            )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        target_module = self._resolve_relative_module(node)
        for alias in node.names:
            local_name = alias.asname or alias.name
            self.result.imports.append(
                RawImport(
                    lineno=node.lineno,
                    alias=local_name,
                    target_module=target_module,
                    imported_name=alias.name,
                )
            )

    def _resolve_relative_module(self, node: ast.ImportFrom) -> str:
        if not node.level:
            return node.module or ""
        # Relative import: walk up `level` packages from *this module's own
        # package* (its own dotted name if it's an __init__.py, else its
        # dotted name minus the last component).
        parts = self.result.module.split(".") if self.result.module else []
        package_parts = parts if self._is_package else parts[:-1]
        # level==1 means "current package" -> drop nothing further.
        cut = node.level - 1
        base = package_parts[: len(package_parts) - cut] if cut else package_parts
        prefix = ".".join(base)
        if node.module:
            return f"{prefix}.{node.module}" if prefix else node.module
        return prefix

    # -- calls ------------------------------------------------------------

    def visit_Call(self, node: ast.Call) -> None:
        source_id = self.current_scope
        func = node.func
        if isinstance(func, ast.Name):
            self.result.calls.append(RawCall(source_id=source_id, shape="name", lineno=node.lineno, name=func.id))
        elif isinstance(func, ast.Attribute):
            if isinstance(func.value, ast.Name) and func.value.id == "self":
                self.result.calls.append(
                    RawCall(source_id=source_id, shape="self_attr", lineno=node.lineno, attr=func.attr)
                )
            elif isinstance(func.value, ast.Name):
                self.result.calls.append(
                    RawCall(
                        source_id=source_id,
                        shape="obj_attr",
                        lineno=node.lineno,
                        obj_name=func.value.id,
                        attr=func.attr,
                    )
                )
            else:
                self.result.calls.append(
                    RawCall(source_id=source_id, shape="complex", lineno=node.lineno, attr=func.attr)
                )
        # else: calling the result of a subscript/lambda/etc. -- too dynamic
        # to say anything useful about, so no edge is recorded at all.

        self.generic_visit(node)


def collect_module(module: str, file_path: str, source: str) -> ModuleCollection:
    tree = ast.parse(source, filename=file_path)
    lines = source.splitlines()
    collector = _Collector(module=module, file_path=file_path, source_lines=lines)
    collector.result.symbols[module].docstring = collector._docstring(tree)
    for stmt in tree.body:
        collector.visit(stmt)
    return collector.result
