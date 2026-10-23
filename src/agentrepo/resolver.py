"""Cross-file symbol, inheritance, import, and call resolution."""

from __future__ import annotations

from dataclasses import dataclass

from .models import Symbol, Edge
from .collector import ModuleCollection, RawImport


@dataclass
class ImportTableEntry:
    target_module: str
    imported_name: str | None


def _build_import_table(imports: list[RawImport]) -> dict[str, ImportTableEntry]:
    table: dict[str, ImportTableEntry] = {}
    for imp in imports:
        table[imp.alias] = ImportTableEntry(target_module=imp.target_module, imported_name=imp.imported_name)
    return table


def _submodule_id(imp: RawImport) -> str | None:
    """For `from X import Y`, the dotted id `X.Y` (Y might be a submodule).
    None for a plain `import X`."""
    if imp.imported_name is None:
        return None
    return f"{imp.target_module}.{imp.imported_name}" if imp.target_module else imp.imported_name


def resolve_repository(collections: list[ModuleCollection]) -> tuple[dict[str, Symbol], list[Edge]]:
    all_symbols: dict[str, Symbol] = {}
    bindings: dict[tuple[str, str], str] = {}
    for c in collections:
        all_symbols.update(c.symbols)
        bindings.update(c.bindings)

    edges: list[Edge] = []

    # 1. CONTAINS -- derived straight from the parent field, no ambiguity.
    for sym in all_symbols.values():
        if sym.parent is not None:
            edges.append(Edge(source=sym.parent, target=sym.id, kind="CONTAINS", resolved=True))

    # 2. IMPORTS + per-file import alias tables (needed for bases & calls below)
    import_tables: dict[str, dict[str, ImportTableEntry]] = {}
    for c in collections:
        table = _build_import_table(c.imports)
        import_tables[c.module] = table
        for imp in c.imports:
            # `from pkg import utils`: if pkg.utils is itself a module in this
            # repo, the import targets that module, not just the package.
            submodule = _submodule_id(imp)
            if submodule is not None and submodule in all_symbols and all_symbols[submodule].kind == "module":
                edges.append(Edge(source=c.module, target=submodule, kind="IMPORTS", resolved=True, lineno=imp.lineno))
            elif imp.target_module in all_symbols and all_symbols[imp.target_module].kind == "module":
                edges.append(Edge(source=c.module, target=imp.target_module, kind="IMPORTS", resolved=True, lineno=imp.lineno))
            else:
                # not an in-repo module: stdlib, third-party, or unresolved.
                target = imp.target_module or (imp.imported_name or "?")
                edges.append(
                    Edge(source=c.module, target=target, kind="IMPORTS", resolved=False, confidence="external", lineno=imp.lineno)
                )

    # 3. INHERITS -- also builds a resolved-bases map used by self.method() resolution below.
    resolved_bases: dict[str, list[str]] = {}
    for c in collections:
        table = import_tables[c.module]
        for raw_base in c.bases:
            target, resolved = _resolve_base(raw_base.raw_expr, c.local_defs, table, all_symbols)
            confidence = None if resolved else "external"
            edges.append(
                Edge(source=raw_base.class_id, target=target, kind="INHERITS", resolved=resolved, confidence=confidence, lineno=raw_base.lineno)
            )
            if resolved:
                resolved_bases.setdefault(raw_base.class_id, []).append(target)

    # 4. CALLS
    for c in collections:
        table = import_tables[c.module]
        for call in c.calls:
            target, resolved, confidence = _resolve_call(call, table, all_symbols, resolved_bases, bindings)
            if target is None:
                continue
            edges.append(
                Edge(source=call.source_id, target=target, kind="CALLS", resolved=resolved, confidence=confidence, lineno=call.lineno)
            )

    return all_symbols, edges


def _resolve_base(raw_expr: str, local_defs: dict[str, str], table: dict[str, ImportTableEntry], all_symbols: dict[str, Symbol]) -> tuple[str, bool]:
    # Simple same-file class name, e.g. `class B(A):`
    if raw_expr in local_defs:
        candidate = local_defs[raw_expr]
        if all_symbols.get(candidate) and all_symbols[candidate].kind == "class":
            return candidate, True

    # `alias.Name` or bare `alias` where alias came from an import
    head = raw_expr.split(".")[0]
    if head in table:
        entry = table[head]
        rest = raw_expr[len(head):].lstrip(".")
        if entry.imported_name is not None:
            candidate = f"{entry.target_module}.{entry.imported_name}"
            if rest:
                candidate = f"{candidate}.{rest}"
        else:
            candidate = f"{entry.target_module}.{rest}" if rest else entry.target_module
        if candidate in all_symbols and all_symbols[candidate].kind == "class":
            return candidate, True
        return candidate, False

    return raw_expr, False


def _resolve_bare_name(source_id: str, name: str, all_symbols: dict[str, Symbol], bindings=None) -> str | None:
    """
    Resolve a bare `name()` call by walking outward through the scopes that
    enclose the call: the calling function's own nested defs, then its
    parent's, ... up to the module. A class body is skipped unless the call
    is directly inside it, because Python does not make class-level names
    visible to the methods inside that class (that is why `helper()` in a
    method means the module-level helper, not `Repo.helper`).
    Returns None when nothing in scope matches -- the caller then falls back
    to imports, and otherwise leaves the call unresolved.
    """
    scope_id: str | None = source_id
    while scope_id is not None:
        scope = all_symbols.get(scope_id)
        if scope is None:
            return None
        if scope_id == source_id or scope.kind != "class":
            candidate = (bindings or {}).get((scope_id, name), f"{scope_id}.{name}")
            found = all_symbols.get(candidate)
            if found is not None and found.kind in ("function", "method", "class"):
                return candidate
        scope_id = scope.parent
    return None


def _resolve_call(call, table, all_symbols, resolved_bases, bindings=None):
    if call.shape == "name":
        name = call.name
        local = _resolve_bare_name(call.source_id, name, all_symbols, bindings)
        if local is not None:
            return local, True, None
        if name in table:
            entry = table[name]
            candidate = f"{entry.target_module}.{entry.imported_name}" if entry.imported_name else entry.target_module
            if candidate in all_symbols and all_symbols[candidate].kind in ("function", "class"):
                return candidate, True, None
            return candidate, False, "external"
        return name, False, "external"

    if call.shape == "self_attr":
        symbol = all_symbols.get(call.source_id)
        class_id = None
        if symbol and symbol.parent and all_symbols.get(symbol.parent) and all_symbols[symbol.parent].kind == "class":
            class_id = symbol.parent
        if class_id is None:
            return call.attr, False, "ambiguous"
        found = _find_method_in_hierarchy(class_id, call.attr, all_symbols, resolved_bases, seen=set(), bindings=bindings)
        if found:
            return found, True, None
        return call.attr, False, "external"

    if call.shape == "obj_attr":
        entry = table.get(call.obj_name)
        if entry is not None:
            module_target = f"{entry.target_module}.{entry.imported_name}" if entry.imported_name else entry.target_module
            candidate = f"{module_target}.{call.attr}"
            if candidate in all_symbols and all_symbols[candidate].kind in ("function", "method"):
                return candidate, True, None
            return candidate, False, "external"
        # obj is a local variable / parameter of unknown type -- this is the
        # textbook `obj.save()` case: we cannot know which save() runs.
        return call.attr, False, "ambiguous"

    # "complex": chained/dynamic call expression
    if call.attr:
        return call.attr, False, "ambiguous"
    return None, False, None  # caller filters out None targets


def _find_method_in_hierarchy(class_id: str, attr: str, all_symbols, resolved_bases, seen: set[str], bindings=None) -> str | None:
    if class_id in seen:
        return None
    seen.add(class_id)
    candidate = (bindings or {}).get((class_id, attr), f"{class_id}.{attr}")
    if candidate in all_symbols and all_symbols[candidate].kind in ("method", "function"):
        return candidate
    for base_id in resolved_bases.get(class_id, []):
        found = _find_method_in_hierarchy(base_id, attr, all_symbols, resolved_bases, seen, bindings)
        if found:
            return found
    return None
