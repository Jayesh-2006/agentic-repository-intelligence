"""Streamlit repository explorer, directory tree, and interactive call-graph builder."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
from typing import Any

from .filesystem import IGNORED_DIRS, is_ignored_name, is_virtualenv_root, is_within

_FILE_VIEWER_MAX_BYTES = 8 * 1024 * 1024
_FILE_READ_CHUNK_LINES = 5000
_MAX_FOLDER_BUTTONS_PER_DIR = 80

DEFAULT_EXCLUDED_SUFFIXES: frozenset[str] = frozenset({
    ".pyc", ".pyo", ".pyd", ".so", ".dll", ".dylib", ".class", ".o", ".obj",
})

_MAX_DIR_TREE_FILES = 5000
_MAX_DIR_TREE_DEPTH = 16

_DIR_TREE_CACHE: dict[tuple[str, int, str], dict[str, Any]] = {}

_FILE_ICONS: dict[str, str] = {
    ".py": "🐍",
    ".md": "📝",
    ".rst": "📝",
    ".txt": "📄",
    ".json": "⚙️",
    ".toml": "⚙️",
    ".yaml": "⚙️",
    ".yml": "⚙️",
    ".ini": "⚙️",
    ".cfg": "⚙️",
    ".sh": "🐚",
    ".sql": "🗄️",
    ".html": "🌐",
    ".css": "🎨",
    ".js": "📜",
    ".ts": "📜",
}


def _file_icon(filename: str) -> str:
    return _FILE_ICONS.get(Path(filename).suffix.lower(), "📄")


def _format_size(size_bytes: int | None) -> str:
    if not isinstance(size_bytes, int) or size_bytes < 0:
        return ""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / (1024 * 1024):.1f} MB"


def _is_within_root(candidate: Path, root: Path) -> bool:
    try:
        return is_within(root.resolve(), candidate.resolve())
    except Exception:
        return False


def _is_excluded_name(name: str, excluded_dirs: set[str] | frozenset[str], excluded_suffixes: set[str] | frozenset[str]) -> bool:
    if name in excluded_dirs or is_ignored_name(name):
        return True
    low = name.lower()
    return any(low.endswith(suf) for suf in excluded_suffixes)


def _should_exclude_rel_path(
    rel_posix: str,
    root: Path,
    excluded_dirs: set[str] | frozenset[str],
    excluded_suffixes: set[str] | frozenset[str],
) -> bool:
    parts = [p for p in PurePosixPath(rel_posix).parts if p not in ("", ".")]
    if not parts:
        return True
    for part in parts[:-1]:
        if part in excluded_dirs or is_ignored_name(part):
            return True
    if _is_excluded_name(parts[-1], excluded_dirs, excluded_suffixes):
        return True
    cur = root
    for part in parts[:-1]:
        cur = cur / part
        if is_virtualenv_root(cur):
            return True
    return False


def _scan_safe_repo_files(
    root: Path,
    excluded_dirs: set[str] | frozenset[str],
    excluded_suffixes: set[str] | frozenset[str],
    max_files: int,
    max_depth: int,
) -> tuple[list[str], list[str], bool]:
    files: list[str] = []
    warnings: list[str] = []
    truncated = False

    def _walk(current: Path, depth: int) -> None:
        nonlocal truncated
        if truncated or depth > max_depth:
            if depth > max_depth:
                truncated = True
            return
        try:
            entries = sorted(os.scandir(current), key=lambda e: (not e.is_dir(follow_symlinks=False), e.name.lower()))
        except OSError as exc:
            rel = current.relative_to(root).as_posix() if current != root else "."
            warnings.append(f"Directory `{rel}` could not be read: {exc.strerror or exc}")
            return

        for entry in entries:
            if len(files) >= max_files:
                truncated = True
                return
            name = entry.name
            try:
                if entry.is_symlink():
                    continue
            except OSError:
                continue

            entry_path = current / name
            if not _is_within_root(entry_path, root):
                continue

            try:
                is_dir = entry.is_dir(follow_symlinks=False)
                is_file = entry.is_file(follow_symlinks=False)
            except OSError as exc:
                rel = entry_path.relative_to(root).as_posix()
                warnings.append(f"Path `{rel}` could not be inspected: {exc.strerror or exc}")
                continue

            if is_dir:
                if name in excluded_dirs or is_ignored_name(name) or is_virtualenv_root(entry_path):
                    continue
                _walk(entry_path, depth + 1)
            elif is_file:
                if _is_excluded_name(name, excluded_dirs, excluded_suffixes):
                    continue
                files.append(entry_path.relative_to(root).as_posix())

    _walk(root, 0)
    return sorted(set(files)), warnings, truncated


def build_directory_tree(
    root: str | Path,
    repo_files: list[str] | None = None,
    *,
    excluded_dirs: set[str] | frozenset[str] | None = None,
    excluded_suffixes: set[str] | frozenset[str] | None = None,
    max_files: int = _MAX_DIR_TREE_FILES,
    max_depth: int = _MAX_DIR_TREE_DEPTH,
) -> dict[str, Any]:
    try:
        root_path = Path(root).resolve()
    except Exception as exc:
        return {
            "ok": False,
            "error": f"Invalid repository root path: {exc}",
            "root": str(root),
            "total_files": 0,
            "total_dirs": 0,
            "total_python_files": 0,
            "truncated": False,
            "warnings": [],
            "files": [],
            "tree": None,
        }

    if not root_path.exists() or not root_path.is_dir():
        return {
            "ok": False,
            "error": f"Repository root does not exist or is not a directory: {root_path}",
            "root": str(root_path),
            "total_files": 0,
            "total_dirs": 0,
            "total_python_files": 0,
            "truncated": False,
            "warnings": [],
            "files": [],
            "tree": None,
        }

    ex_dirs = excluded_dirs if excluded_dirs is not None else IGNORED_DIRS
    ex_suffixes = excluded_suffixes if excluded_suffixes is not None else DEFAULT_EXCLUDED_SUFFIXES

    warnings: list[str] = []
    truncated = False

    if repo_files is not None:
        cleaned_files: list[str] = []
        for raw_rel in repo_files:
            rel_posix = str(raw_rel).strip().replace("\\", "/").lstrip("/")
            if not rel_posix or ".." in PurePosixPath(rel_posix).parts:
                continue
            if _should_exclude_rel_path(rel_posix, root_path, ex_dirs, ex_suffixes):
                continue
            candidate = root_path / rel_posix
            if candidate.is_symlink() or not _is_within_root(candidate, root_path):
                continue
            if len(PurePosixPath(rel_posix).parts) > max_depth:
                truncated = True
                continue
            cleaned_files.append(rel_posix)
            if len(cleaned_files) >= max_files:
                truncated = True
                break
        safe_files = sorted(set(cleaned_files))
    else:
        safe_files, warnings, truncated = _scan_safe_repo_files(
            root_path,
            ex_dirs,
            ex_suffixes,
            max_files=max_files,
            max_depth=max_depth,
        )

    files_digest = hashlib.sha1("\n".join(safe_files).encode("utf-8")).hexdigest()[:16]
    cache_key = (str(root_path), len(safe_files), files_digest)
    cached = _DIR_TREE_CACHE.get(cache_key)
    if cached is not None and not warnings:
        return cached

    root_node: dict[str, Any] = {
        "name": root_path.name or str(root_path),
        "path": "",
        "type": "directory",
        "readable": True,
        "file_count": 0,
        "python_file_count": 0,
        "dir_count": 0,
        "children": [],
        "_dir_map": {},
    }

    total_dirs = 0
    total_py = 0

    for rel_posix in safe_files:
        parts = [p for p in PurePosixPath(rel_posix).parts if p not in ("", ".")]
        if not parts:
            continue
        is_py = rel_posix.lower().endswith(".py")
        if is_py:
            total_py += 1

        cur_node = root_node
        cur_node["file_count"] += 1
        if is_py:
            cur_node["python_file_count"] += 1

        accum_parts: list[str] = []
        for folder_name in parts[:-1]:
            accum_parts.append(folder_name)
            folder_rel = "/".join(accum_parts)
            dir_map: dict[str, dict[str, Any]] = cur_node["_dir_map"]
            if folder_name not in dir_map:
                folder_abs = root_path / folder_rel
                readable = os.access(folder_abs, os.R_OK) if folder_abs.exists() else True
                new_dir: dict[str, Any] = {
                    "name": folder_name,
                    "path": folder_rel,
                    "type": "directory",
                    "readable": readable,
                    "file_count": 0,
                    "python_file_count": 0,
                    "dir_count": 0,
                    "children": [],
                    "_dir_map": {},
                }
                dir_map[folder_name] = new_dir
                cur_node["children"].append(new_dir)
                total_dirs += 1
            cur_node = dir_map[folder_name]
            cur_node["file_count"] += 1
            if is_py:
                cur_node["python_file_count"] += 1

        file_name = parts[-1]
        file_abs = root_path / rel_posix
        size_bytes: int | None = None
        readable_file = True
        try:
            if file_abs.exists():
                stat = file_abs.stat()
                size_bytes = int(stat.st_size)
                readable_file = os.access(file_abs, os.R_OK)
        except OSError:
            readable_file = False

        cur_node["children"].append({
            "name": file_name,
            "path": rel_posix,
            "type": "file",
            "extension": Path(file_name).suffix.lower(),
            "is_python": is_py,
            "icon": _file_icon(file_name),
            "size_bytes": size_bytes,
            "readable": readable_file,
        })

    def _finalize(node: dict[str, Any]) -> int:
        node.pop("_dir_map", None)
        children = node.get("children") or []
        dirs = sorted([c for c in children if c.get("type") == "directory"], key=lambda c: c.get("name", "").lower())
        files = sorted([c for c in children if c.get("type") == "file"], key=lambda c: c.get("name", "").lower())
        node["children"] = dirs + files
        descendant_dirs = len(dirs)
        for d in dirs:
            descendant_dirs += _finalize(d)
        node["dir_count"] = descendant_dirs
        return descendant_dirs

    _finalize(root_node)

    result = {
        "ok": True,
        "root": root_path.name or str(root_path),
        "root_path": str(root_path),
        "total_files": len(safe_files),
        "total_dirs": total_dirs,
        "total_python_files": total_py,
        "truncated": truncated,
        "warnings": warnings,
        "files": safe_files,
        "tree": root_node,
    }
    if not warnings:
        if len(_DIR_TREE_CACHE) >= 32:
            _DIR_TREE_CACHE.clear()
        _DIR_TREE_CACHE[cache_key] = result
    return result


def filter_directory_tree(tree_node: dict[str, Any] | None, query: str) -> dict[str, Any] | None:
    if tree_node is None:
        return None
    q = (query or "").strip().lower()
    if not q:
        return tree_node

    def _filter_node(node: dict[str, Any], is_root: bool = False) -> dict[str, Any] | None:
        node_type = node.get("type")
        node_path = str(node.get("path") or "").lower()
        node_name = str(node.get("name") or "").lower()

        if node_type == "file":
            return dict(node) if (q in node_path or q in node_name) else None

        dir_matches = (not is_root) and (q in node_path or q in node_name)
        kept_children: list[dict[str, Any]] = []
        file_count = 0
        py_count = 0
        dir_count = 0

        for child in node.get("children") or []:
            filtered_child = child if dir_matches else _filter_node(child, is_root=False)
            if filtered_child is not None:
                kept_children.append(filtered_child)
                if filtered_child.get("type") == "file":
                    file_count += 1
                    if filtered_child.get("is_python"):
                        py_count += 1
                else:
                    file_count += int(filtered_child.get("file_count", 0))
                    py_count += int(filtered_child.get("python_file_count", 0))
                    dir_count += 1 + int(filtered_child.get("dir_count", 0))

        if not kept_children and not is_root and not dir_matches:
            return None

        cloned = dict(node)
        cloned["children"] = kept_children
        cloned["file_count"] = file_count
        cloned["python_file_count"] = py_count
        cloned["dir_count"] = dir_count
        return cloned

    return _filter_node(tree_node, is_root=True)


def read_complete_source(service, path: str) -> tuple[dict | None, str | None]:
    chunks: list[str] = []
    next_line = 1
    expected_total: int | None = None
    total_bytes = 0

    while True:
        try:
            result = service.get_file(path, start_line=next_line, max_lines=_FILE_READ_CHUNK_LINES)
        except TypeError:
            result = service.get_file(path, max_lines=350)
        if not isinstance(result, dict) or result.get("error"):
            return result if isinstance(result, dict) else {"error": "read_failed"}, None

        total = int(result.get("total_lines", 0))
        if expected_total is None:
            expected_total = total
        elif total != expected_total:
            return {
                "error": "file_changed",
                "message": "The file changed while it was being read. Select it again to refresh.",
            }, None

        content = result.get("content", "")
        total_bytes += len(content.encode("utf-8"))
        if total_bytes > _FILE_VIEWER_MAX_BYTES:
            return {
                "error": "file_too_large",
                "message": "This file is too large to display in the browser (8 MiB limit). Open it locally to inspect the full source.",
            }, None
        chunks.append(content)

        if not result.get("truncated"):
            return result, "\n".join(chunks)

        following_line = result.get("next_start_line")
        if not isinstance(following_line, int) or following_line < next_line + 1:
            return result, "\n".join(chunks)
        next_line = following_line


def _session_dict(st) -> dict[str, Any]:
    state = getattr(st, "session_state", None)
    if isinstance(state, dict):
        return state
    if state is not None and hasattr(state, "__getitem__") and hasattr(state, "__setitem__"):
        return state  # type: ignore[return-value]
    return {}


def _sync_repo_explorer_state(st, repo_root: str, available_files: list[str]) -> dict[str, Any]:
    state = _session_dict(st)
    if state.get("ari_explorer_repo_root") != repo_root:
        state["ari_explorer_repo_root"] = repo_root
        state["ari_selected_file"] = available_files[0] if available_files else None
        state["ari_selected_line_range"] = None

    if not state.get("ari_selected_file") and available_files:
        state["ari_selected_file"] = available_files[0]

    return state


def _display_source_viewer(
    st,
    service,
    selected_file: str,
    *,
    start_line: int | None = None,
    end_line: int | None = None,
    header_prefix: str = "",
) -> tuple[dict | None, str | None]:
    source, content = read_complete_source(service, selected_file)
    if source is None or source.get("error"):
        try:
            repo_root = Path(service.index.root).resolve()
            candidate = (repo_root / selected_file).resolve()
            if candidate.is_relative_to(repo_root) and candidate.is_file() and not candidate.is_symlink():
                raw = candidate.read_bytes()
                if len(raw) <= _FILE_VIEWER_MAX_BYTES:
                    content = raw.decode("utf-8", errors="replace")
                    lines = content.splitlines()
                    source = {"total_lines": len(lines), "error": None}
        except (OSError, ValueError, TypeError, AttributeError):
            pass
    if source is None or source.get("error"):
        err = source.get("error") if isinstance(source, dict) else "Unable to read the selected file."
        message = (
            source.get("message", err["message"] if isinstance(err, dict) and "message" in err else err)
            if isinstance(source, dict)
            else "Unable to read the selected file."
        )
        st.warning(str(message))
        return source, None

    full_text = content or ""
    lines = full_text.splitlines()
    total_lines = source.get("total_lines", len(lines))
    eff_start = 1
    eff_end = len(lines)
    displayed_text = full_text

    if start_line is not None and start_line >= 1 and lines:
        eff_start = min(max(1, start_line), len(lines))
        eff_end = min(len(lines), end_line if (end_line is not None and end_line >= eff_start) else len(lines))
        displayed_text = "\n".join(lines[eff_start - 1 : eff_end])
        caption_text = f"{header_prefix}{selected_file} · lines {eff_start}–{eff_end} of {total_lines} lines"
    else:
        caption_text = f"{header_prefix}{selected_file} · {total_lines} lines"

    if hasattr(st, "markdown"):
        st.markdown("""<style>
        [data-testid="stCodeBlock"] pre, .stCodeBlock pre,
        [data-testid="stCode"] pre {
          max-height: 65vh !important;
          overflow-y: auto !important;
          overflow-x: auto !important;
          white-space: pre-wrap !important;
          overflow-wrap: anywhere !important;
        }
        </style>""", unsafe_allow_html=True)

    suffix = Path(selected_file).suffix.lower()
    if suffix in {".md", ".markdown"} and start_line is None:
        try:
            with st.container(height=650, border=True):
                st.markdown(displayed_text, unsafe_allow_html=False)
        except TypeError:
            st.markdown(displayed_text, unsafe_allow_html=False)
    else:
        st.caption(caption_text)
        language = {
            ".rst": "rst",
            ".txt": "text",
            ".env": "text",
        }.get(suffix, suffix.lstrip(".") or "text")
        try:
            with st.container(height=650, border=True):
                st.code(displayed_text, language=language, line_numbers=True, wrap_lines=True)
        except TypeError:
            st.code(displayed_text, language=language, line_numbers=True, wrap_lines=True)
    return source, full_text


def _render_directory_node(
    st,
    node: dict[str, Any],
    state: dict[str, Any],
    *,
    depth: int = 0,
    filter_active: bool = False,
) -> None:
    children = node.get("children") or []
    dirs = [c for c in children if c.get("type") == "directory"]
    files = [c for c in children if c.get("type") == "file"]

    for d_node in dirs:
        d_path = d_node.get("path", "")
        d_name = d_node.get("name", "folder")
        f_cnt = d_node.get("file_count", 0)
        py_cnt = d_node.get("python_file_count", 0)
        badge = f"{f_cnt} file{'s' if f_cnt != 1 else ''}" + (f", {py_cnt} py" if py_cnt else "")
        expanded_default = bool(filter_active or depth == 0)
        with st.expander(f"📁 {d_name}/  ({badge})", expanded=expanded_default):
            if not d_node.get("readable", True):
                st.warning(f"Directory `{d_path}` could not be read.")
            elif not d_node.get("children"):
                st.caption("*(empty directory)*")
            else:
                _render_directory_node(
                    st,
                    d_node,
                    state,
                    depth=depth + 1,
                    filter_active=filter_active,
                )

    shown_files = files[:_MAX_FOLDER_BUTTONS_PER_DIR]
    for f_node in shown_files:
        f_path = str(f_node.get("path") or "")
        f_name = str(f_node.get("name") or f_path)
        icon = f_node.get("icon", "📄")
        size_str = _format_size(f_node.get("size_bytes"))
        size_suffix = f"  ·  {size_str}" if size_str else ""
        is_selected = state.get("ari_selected_file") == f_path
        btn_label = f"{icon} {f_name}{size_suffix}"
        if hasattr(st, "button"):
            if st.button(
                btn_label,
                key=f"dirtree_file_{f_path}",
                width="stretch",
                type="primary" if is_selected else "secondary",
                help=f_path,
            ):
                state["ari_selected_file"] = f_path
                state["ari_selected_line_range"] = None
        else:
            st.caption(btn_label)

    if len(files) > _MAX_FOLDER_BUTTONS_PER_DIR:
        st.caption(
            f"Showing {_MAX_FOLDER_BUTTONS_PER_DIR} of {len(files)} files in this folder. "
            "Use the path filter above to narrow further."
        )


def _render_file_symbol_outline(st, service, selected_file: str, state: dict[str, Any]) -> None:
    index = getattr(service, "index", None)
    symbols_map = getattr(index, "symbols", None) or {}
    file_symbols = [
        sym for sym in symbols_map.values()
        if str(getattr(sym, "file", "")).replace("\\", "/") == selected_file.replace("\\", "/")
        and getattr(sym, "kind", "") != "module"
    ]
    if not file_symbols or not hasattr(st, "expander"):
        return

    file_symbols.sort(key=lambda s: (getattr(s, "lineno", 0) or 0, getattr(s, "name", "")))
    with st.expander(f"🧩 File Symbols & Outline ({len(file_symbols)} symbols in `{Path(selected_file).name}`)", expanded=False):
        if state.get("ari_selected_line_range") and hasattr(st, "button"):
            if st.button("🔄 Show Full File", key=f"reset_lines_{selected_file}", width="stretch"):
                state["ari_selected_line_range"] = None

        rows = []
        for sym in file_symbols:
            s_line = getattr(sym, "lineno", 1)
            s_end = getattr(sym, "end_lineno", s_line) or s_line
            rows.append({
                "Kind": getattr(sym, "kind", "symbol"),
                "Symbol": getattr(sym, "name", ""),
                "Qualified ID": getattr(sym, "id", ""),
                "Lines": f"{s_line}–{s_end}",
                "Docstring": (getattr(sym, "docstring", "") or "")[:80],
            })
        if hasattr(st, "dataframe"):
            st.dataframe(rows, hide_index=True, width="stretch")


def render_directory_tree_tab(st, service, state: dict[str, Any], paths: list[str]) -> None:
    index = service.index
    tree_result = build_directory_tree(index.root, repo_files=None)
    if not tree_result.get("ok") or not tree_result.get("tree"):
        st.warning(tree_result.get("error") or "Unable to build repository directory tree.")
        return

    st.caption(
        f"**Root:** `{tree_result['root']}` · "
        f"**{tree_result['total_dirs']}** directories · "
        f"**{tree_result['total_files']}** files "
        f"({tree_result.get('total_python_files', 0)} Python)"
    )
    if tree_result.get("truncated"):
        st.info("Large repository detected: directory tree is bounded to keep navigation responsive.")
    for warn in tree_result.get("warnings") or []:
        st.warning(warn)

    if tree_result["total_files"] == 0 and tree_result["total_dirs"] == 0:
        st.info("This repository does not contain any non-ignored files or folders.")
        return

    filter_query = ""
    if hasattr(st, "text_input"):
        filter_query = st.text_input(
            "Filter directory tree by path or filename",
            key="explorer_dirtree_filter",
            placeholder="e.g. agent/, service.py, .md",
        )

    active_tree = filter_directory_tree(tree_result["tree"], filter_query)
    if active_tree is None or (not active_tree.get("children") and filter_query.strip()):
        st.info(f"No repository files or folders matched `{filter_query.strip()}`.")
        return

    if hasattr(st, "markdown"):
        st.markdown(
            """<style>
            div[data-testid="stExpanderDetails"] div[data-testid="stButton"] > button {
              justify-content: flex-start !important;
              text-align: left !important;
              padding: 0.3rem 0.7rem !important;
              font-size: 0.84rem !important;
            }
            div[data-testid="stExpanderDetails"] div[data-testid="stButton"] > button p {
              text-align: left !important;
            }
            </style>""",
            unsafe_allow_html=True,
        )

    if hasattr(st, "columns"):
        tree_col, viewer_col = st.columns([5, 7])
    else:
        tree_col = viewer_col = st

    all_tree_files = tree_result.get("files") or paths
    selected_file = state.get("ari_selected_file")
    if all_tree_files and (not selected_file or selected_file not in all_tree_files):
        selected_file = all_tree_files[0]
        state["ari_selected_file"] = selected_file

    with tree_col:
        if hasattr(st, "markdown"):
            st.markdown(f"##### 🗂️ `{tree_result['root']}/`")
        if hasattr(st, "container"):
            try:
                with st.container(height=650, border=True):
                    _render_directory_node(
                        st,
                        active_tree,
                        state,
                        depth=0,
                        filter_active=bool(filter_query.strip()),
                    )
            except TypeError:
                _render_directory_node(
                    st,
                    active_tree,
                    state,
                    depth=0,
                    filter_active=bool(filter_query.strip()),
                )
        else:
            _render_directory_node(
                st,
                active_tree,
                state,
                depth=0,
                filter_active=bool(filter_query.strip()),
            )

    with viewer_col:
        selected_file = state.get("ari_selected_file")
        if hasattr(st, "markdown"):
            current_name = Path(str(selected_file or "Select a file")).name
            st.markdown(f"##### 📄 `{current_name}`")

        if not selected_file:
            st.info("Select a file from the directory tree to view its source.")
        else:
            _render_file_symbol_outline(st, service, selected_file, state)
            line_range = state.get("ari_selected_line_range")
            s_line = line_range[0] if isinstance(line_range, (list, tuple)) and len(line_range) == 2 else None
            e_line = line_range[1] if isinstance(line_range, (list, tuple)) and len(line_range) == 2 else None
            _display_source_viewer(st, service, selected_file, start_line=s_line, end_line=e_line)


def build_symbol_subgraph(
    index,
    graph,
    focal_symbol: str,
    *,
    max_hops: int = 1,
    allowed_kinds: set[str] | None = None,
    include_unresolved: bool = False,
    max_nodes: int = 42,
) -> dict[str, Any]:
    kinds = allowed_kinds or {"CALLS", "IMPORTS", "INHERITS", "CONTAINS"}
    symbols = getattr(index, "symbols", {}) or {}
    all_edges = list(graph.all_edges()) if hasattr(graph, "all_edges") else []

    visited_nodes: set[str] = {focal_symbol}
    frontier: set[str] = {focal_symbol}
    collected_edges: list[dict[str, Any]] = []
    seen_edge_keys: set[tuple[str, str, str, int | None]] = set()

    for _hop in range(max(1, min(2, int(max_hops)))):
        next_frontier: set[str] = set()
        for edge in all_edges:
            if edge.kind not in kinds:
                continue
            if not edge.resolved and not include_unresolved:
                continue
            src = str(edge.source)
            tgt = str(edge.target)
            if src not in frontier and tgt not in frontier:
                continue
            other = tgt if src in frontier else src
            if len(visited_nodes) >= max_nodes and other not in visited_nodes:
                continue
            ekey = (src, tgt, str(edge.kind), getattr(edge, "lineno", None))
            if ekey not in seen_edge_keys:
                seen_edge_keys.add(ekey)
                collected_edges.append({
                    "source": src,
                    "target": tgt,
                    "kind": str(edge.kind),
                    "resolved": bool(edge.resolved),
                    "lineno": getattr(edge, "lineno", None),
                })
            if other not in visited_nodes:
                visited_nodes.add(other)
                next_frontier.add(other)
        frontier = next_frontier
        if not frontier or len(visited_nodes) >= max_nodes:
            break

    nodes_list: list[dict[str, Any]] = []
    for idx_n, nid in enumerate(sorted(visited_nodes, key=lambda x: (x != focal_symbol, x))):
        sym = symbols.get(nid)
        short_name = getattr(sym, "name", None) or nid.rsplit(".", 1)[-1]
        kind = getattr(sym, "kind", "external" if sym is None else "symbol")
        nodes_list.append({
            "id": nid,
            "label": short_name,
            "kind": kind,
            "file": getattr(sym, "file", None) or "",
            "lineno": getattr(sym, "lineno", None),
            "docstring": getattr(sym, "docstring", None) or "",
            "is_focal": nid == focal_symbol,
            "order": idx_n,
        })

    return {
        "focal_symbol": focal_symbol,
        "nodes": nodes_list,
        "edges": collected_edges,
    }


def _render_interactive_call_graph_html(subgraph: dict[str, Any]) -> str:
    nodes = list(subgraph.get("nodes") or [])
    edges = list(subgraph.get("edges") or [])
    focal_id = str(subgraph.get("focal_symbol") or "")

    width, height = 880, 500
    cx, cy = width / 2.0, height / 2.0

    non_focal = [n for n in nodes if not n.get("is_focal")]
    incoming_ids = {e["source"] for e in edges if e["target"] == focal_id}
    outgoing_ids = {e["target"] for e in edges if e["source"] == focal_id}

    ring1 = [n for n in non_focal if n["id"] in incoming_ids or n["id"] in outgoing_ids]
    ring2 = [n for n in non_focal if n not in ring1]

    for n in nodes:
        if n.get("is_focal"):
            n["x"] = cx
            n["y"] = cy

    for idx, n in enumerate(ring1):
        angle = (2.0 * math.pi * idx) / max(1, len(ring1)) - (math.pi / 2.0)
        n["x"] = round(cx + 155.0 * math.cos(angle), 1)
        n["y"] = round(cy + 140.0 * math.sin(angle), 1)

    for idx, n in enumerate(ring2):
        angle = (2.0 * math.pi * idx) / max(1, len(ring2)) - (math.pi / 4.0)
        n["x"] = round(cx + 265.0 * math.cos(angle), 1)
        n["y"] = round(cy + 215.0 * math.sin(angle), 1)

    payload_json = json.dumps({"nodes": nodes, "edges": edges, "focal": focal_id}, ensure_ascii=True)

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<style>
  body {{
    margin: 0;
    padding: 0;
    background: #0c1320;
    color: #e2e8f0;
    font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    overflow: hidden;
  }}
  .cg-wrapper {{
    display: flex;
    flex-direction: column;
    height: 540px;
    border: 1px solid rgba(148, 163, 184, 0.22);
    border-radius: 12px;
    background: radial-gradient(circle at 50% 45%, rgba(30, 58, 138, 0.18), #0b111d 72%);
    overflow: hidden;
  }}
  .cg-toolbar {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 8px 14px;
    background: rgba(15, 23, 42, 0.92);
    border-bottom: 1px solid rgba(148, 163, 184, 0.16);
    font-size: 12px;
  }}
  .cg-legend {{
    display: flex;
    gap: 12px;
    align-items: center;
    flex-wrap: wrap;
  }}
  .cg-dot {{
    display: inline-block;
    width: 9px;
    height: 9px;
    border-radius: 50%;
    margin-right: 4px;
  }}
  .cg-btns button {{
    background: rgba(30, 41, 59, 0.9);
    color: #cbd5e1;
    border: 1px solid rgba(148, 163, 184, 0.25);
    border-radius: 6px;
    padding: 4px 9px;
    font-size: 11.5px;
    cursor: pointer;
    margin-left: 5px;
  }}
  .cg-btns button:hover {{
    background: rgba(59, 130, 246, 0.25);
    color: #fff;
  }}
  .cg-main {{
    position: relative;
    flex: 1;
    overflow: hidden;
  }}
  svg#cg-svg {{
    width: 100%;
    height: 100%;
    cursor: grab;
  }}
  svg#cg-svg:active {{
    cursor: grabbing;
  }}
  .cg-inspector {{
    position: absolute;
    bottom: 10px;
    left: 12px;
    right: 12px;
    background: rgba(15, 23, 42, 0.92);
    border: 1px solid rgba(148, 163, 184, 0.22);
    border-radius: 8px;
    padding: 8px 12px;
    font-size: 12px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    pointer-events: none;
  }}
  .cg-node {{ cursor: pointer; }}
  .cg-node text {{
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 11px;
    fill: #f1f5f9;
    pointer-events: none;
    text-shadow: 0 1px 3px rgba(0,0,0,0.9);
  }}
</style>
</head>
<body>
<div class="cg-wrapper">
  <div class="cg-toolbar">
    <div class="cg-legend">
      <span><i class="cg-dot" style="background:#38bdf8;box-shadow:0 0 6px #38bdf8;"></i>Focal</span>
      <span><i class="cg-dot" style="background:#10b981;"></i>Function/Method</span>
      <span><i class="cg-dot" style="background:#f59e0b;"></i>Class</span>
      <span><i class="cg-dot" style="background:#818cf8;"></i>Module</span>
      <span><i class="cg-dot" style="background:#64748b;"></i>External</span>
      <span style="margin-left:6px;color:#94a3b8;">Edges: <b style="color:#60a5fa;">CALLS</b> · <b style="color:#c084fc;">IMPORTS</b> · <b style="color:#fbbf24;">INHERITS</b> · <b style="color:#94a3b8;">CONTAINS</b></span>
    </div>
    <div class="cg-btns">
      <button onclick="zoomBy(1.2)">＋ Zoom In</button>
      <button onclick="zoomBy(0.84)">－ Zoom Out</button>
      <button onclick="resetView()">⟲ Reset Layout</button>
    </div>
  </div>
  <div class="cg-main">
    <svg id="cg-svg" viewBox="0 0 {width} {height}">
      <defs>
        <marker id="arrow-CALLS" viewBox="0 0 10 10" refX="22" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 1 L 10 5 L 0 9 z" fill="#60a5fa"/>
        </marker>
        <marker id="arrow-IMPORTS" viewBox="0 0 10 10" refX="22" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 1 L 10 5 L 0 9 z" fill="#c084fc"/>
        </marker>
        <marker id="arrow-INHERITS" viewBox="0 0 10 10" refX="22" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 1 L 10 5 L 0 9 z" fill="#fbbf24"/>
        </marker>
        <marker id="arrow-CONTAINS" viewBox="0 0 10 10" refX="22" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 1 L 10 5 L 0 9 z" fill="#64748b"/>
        </marker>
      </defs>
      <g id="cg-viewport">
        <g id="cg-edges"></g>
        <g id="cg-nodes"></g>
      </g>
    </svg>
    <div class="cg-inspector" id="cg-info">
      <span>💡 <b>Interactive Call Graph:</b> Drag nodes to reposition, scroll to zoom, or click any symbol node to inspect its definition and call edges.</span>
    </div>
  </div>
</div>
<script>
const DATA = {payload_json};
const KIND_COLORS = {{
  function: "#10b981",
  method: "#14b8a6",
  class: "#f59e0b",
  module: "#818cf8",
  external: "#64748b",
  symbol: "#38bdf8"
}};
const EDGE_COLORS = {{
  CALLS: "#60a5fa",
  IMPORTS: "#c084fc",
  INHERITS: "#fbbf24",
  CONTAINS: "#64748b"
}};

const svg = document.getElementById("cg-svg");
const viewport = document.getElementById("cg-viewport");
const edgesGroup = document.getElementById("cg-edges");
const nodesGroup = document.getElementById("cg-nodes");
const infoBox = document.getElementById("cg-info");

let panX = 0, panY = 0, scale = 1.0;
let isPanning = false, startPanX = 0, startPanY = 0;
let draggedNode = null;
const initialPos = DATA.nodes.map(n => ({{id: n.id, x: n.x, y: n.y}}));
const nodeMap = {{}};
DATA.nodes.forEach(n => nodeMap[n.id] = n);

function updateTransform() {{
  viewport.setAttribute("transform", `translate(${{panX}},${{panY}}) scale(${{scale}})`);
}}

function zoomBy(factor) {{
  scale = Math.max(0.35, Math.min(3.0, scale * factor));
  updateTransform();
}}

function resetView() {{
  panX = 0; panY = 0; scale = 1.0;
  initialPos.forEach(p => {{
    if (nodeMap[p.id]) {{ nodeMap[p.id].x = p.x; nodeMap[p.id].y = p.y; }}
  }});
  updateTransform();
  renderGraph();
}}

function renderGraph(selectedId = null) {{
  edgesGroup.innerHTML = "";
  nodesGroup.innerHTML = "";

  DATA.edges.forEach(e => {{
    const s = nodeMap[e.source];
    const t = nodeMap[e.target];
    if (!s || !t) return;
    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("x1", s.x);
    line.setAttribute("y1", s.y);
    line.setAttribute("x2", t.x);
    line.setAttribute("y2", t.y);
    const col = EDGE_COLORS[e.kind] || "#60a5fa";
    const active = !selectedId || e.source === selectedId || e.target === selectedId;
    line.setAttribute("stroke", col);
    line.setAttribute("stroke-width", active ? "2.0" : "1.1");
    line.setAttribute("stroke-opacity", active ? "0.85" : "0.22");
    if (!e.resolved) line.setAttribute("stroke-dasharray", "4,4");
    line.setAttribute("marker-end", `url(#arrow-${{e.kind in EDGE_COLORS ? e.kind : 'CALLS'}})`);
    edgesGroup.appendChild(line);
  }});

  DATA.nodes.forEach(n => {{
    const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
    g.setAttribute("class", "cg-node");
    g.setAttribute("transform", `translate(${{n.x}},${{n.y}})`);

    const c = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    const r = n.is_focal ? 16 : 11;
    c.setAttribute("r", r);
    c.setAttribute("fill", n.is_focal ? "#0284c7" : (KIND_COLORS[n.kind] || "#38bdf8"));
    c.setAttribute("stroke", n.id === selectedId ? "#ffffff" : (n.is_focal ? "#38bdf8" : "#0f172a"));
    c.setAttribute("stroke-width", n.is_focal || n.id === selectedId ? "3" : "2");
    g.appendChild(c);

    const txt = document.createElementNS("http://www.w3.org/2000/svg", "text");
    txt.setAttribute("dy", r + 13);
    txt.setAttribute("text-anchor", "middle");
    txt.textContent = n.label.length > 26 ? n.label.slice(0, 24) + "…" : n.label;
    g.appendChild(txt);

    g.addEventListener("mousedown", (ev) => {{
      ev.stopPropagation();
      draggedNode = n;
      selectNode(n);
    }});

    nodesGroup.appendChild(g);
  }});
}}

function selectNode(n) {{
  const inCnt = DATA.edges.filter(e => e.target === n.id).length;
  const outCnt = DATA.edges.filter(e => e.source === n.id).length;
  const loc = n.file ? `${{n.file}}${{n.lineno ? ':' + n.lineno : ''}}` : 'external / unresolved';
  const doc = n.docstring ? ` · "${{n.docstring}}"` : '';
  infoBox.innerHTML = `<span>🎯 <b>${{n.id}}</b> (<code>${{n.kind}}</code> · <code>${{loc}}</code>)${{doc}}</span><span>Incoming: <b>${{inCnt}}</b> · Outgoing: <b>${{outCnt}}</b></span>`;
  renderGraph(n.id);
}}

svg.addEventListener("mousedown", (ev) => {{
  isPanning = true;
  startPanX = ev.clientX - panX;
  startPanY = ev.clientY - panY;
}});

window.addEventListener("mousemove", (ev) => {{
  if (draggedNode) {{
    const pt = svg.createSVGPoint();
    pt.x = ev.clientX;
    pt.y = ev.clientY;
    const ctm = viewport.getScreenCTM();
    if (ctm) {{
      const loc = pt.matrixTransform(ctm.inverse());
      draggedNode.x = loc.x;
      draggedNode.y = loc.y;
      renderGraph(draggedNode.id);
    }}
  }} else if (isPanning) {{
    panX = ev.clientX - startPanX;
    panY = ev.clientY - startPanY;
    updateTransform();
  }}
}});

window.addEventListener("mouseup", () => {{
  isPanning = false;
  draggedNode = null;
}});

svg.addEventListener("wheel", (ev) => {{
  ev.preventDefault();
  zoomBy(ev.deltaY < 0 ? 1.1 : 0.9);
}}, {{passive: false}});

renderGraph(DATA.focal);
</script>
</body>
</html>"""


def render_relationships_tab(st, service, index) -> None:
    symbols = index.symbols
    if not symbols:
        st.info("No symbols are present in this index. Rebuild the index to extract Python relationships.")
        return

    graph = service.graph
    all_symbols_sorted = sorted(symbols)

    if hasattr(st, "columns"):
        c1, c2, c3, c4 = st.columns([4, 2, 3, 2])
        with c1:
            selected_symbol = st.selectbox("Focal Symbol", all_symbols_sorted, key="explorer_symbol")
        with c2:
            hop_depth = st.selectbox("Graph Depth", [1, 2], index=0, format_func=lambda h: f"{h}-hop neighborhood", key="explorer_graph_hops")
        with c3:
            edge_kinds = st.multiselect(
                "Relationship Types",
                options=["CALLS", "IMPORTS", "INHERITS", "CONTAINS"],
                default=["CALLS", "IMPORTS", "INHERITS"],
                key="explorer_edge_kinds",
            )
        with c4:
            include_unresolved = st.checkbox("Include unresolved", value=False, key="explorer_include_unresolved")
    else:
        selected_symbol = st.selectbox("Symbol", all_symbols_sorted, key="explorer_symbol")
        hop_depth = 1
        edge_kinds = ["CALLS", "IMPORTS", "INHERITS", "CONTAINS"]
        include_unresolved = False

    symbol = symbols[selected_symbol]
    st.caption(f"**Selected:** `{symbol.kind}` · `{symbol.file}:{symbol.lineno}`" + (f" — *{symbol.docstring}*" if getattr(symbol, "docstring", None) else ""))

    subgraph = build_symbol_subgraph(
        index,
        graph,
        selected_symbol,
        max_hops=hop_depth,
        allowed_kinds=set(edge_kinds or ["CALLS", "IMPORTS", "INHERITS", "CONTAINS"]),
        include_unresolved=include_unresolved,
    )

    graph_html = _render_interactive_call_graph_html(subgraph)
    if hasattr(st, "iframe"):
        st.iframe(graph_html, height=555)
    elif hasattr(st, "html"):
        st.html(graph_html, unsafe_allow_javascript=True)
    else:
        import streamlit.components.v1 as components
        components.html(graph_html, height=555, scrolling=False)

    outgoing = graph.neighbors(selected_symbol)
    incoming = [edge for edge in graph.all_edges() if edge.resolved and edge.target == selected_symbol]
    edge_rows = []
    for direction, edges in (("Outgoing", outgoing), ("Incoming", incoming)):
        for edge in edges:
            other = edge.target if direction == "Outgoing" else edge.source
            edge_rows.append({
                "Direction": direction,
                "Relationship": edge.kind,
                "Symbol": other,
                "Resolved": edge.resolved,
                "Line": edge.lineno,
            })

    with st.expander(f"📋 Tabular Relationship Inspector ({len(edge_rows)} direct edge(s))", expanded=False):
        if edge_rows:
            st.dataframe(edge_rows, hide_index=True, width="stretch")
        else:
            st.info("No direct relationships for this symbol are stored in the current graph.")

    unresolved = [edge for edge in outgoing if not edge.resolved]
    if unresolved:
        with st.expander(f"Unresolved outgoing references ({len(unresolved)})"):
            st.dataframe([
                {"Relationship": edge.kind, "Target": edge.target, "Line": edge.lineno}
                for edge in unresolved
            ], hide_index=True, width="stretch")


def render_repository_explorer(st, app) -> None:
    service = app.service
    index = service.index
    st.subheader("Repository explorer")
    st.caption("Search indexed content, browse the repository directory tree, and inspect interactive symbol call graphs.")

    paths = sorted(set(index.repo_files or [record.path for record in index.files]))
    state = _sync_repo_explorer_state(st, str(index.root), paths)

    search_tab, dirtree_tab, graph_tab = st.tabs(["Search", "Directory Tree", "Relationships"])

    with search_tab:
        query = st.text_input("Search indexed source and documentation", key="explorer_search")
        limit = st.slider("Results", 1, 12, 5, key="explorer_search_limit")
        if query.strip():
            result = service.search_repository(query.strip(), k=limit)
            if result.get("error"):
                st.warning(result.get("message", result["error"]))
            else:
                hits = result.get("results", [])
                if not hits:
                    st.info("No indexed passages matched that query.")
                for hit in hits:
                    title = (
                        f"{hit.get('file', 'unknown file')} · lines "
                        f"{hit.get('line_start', '?')}–{hit.get('line_end', '?')} · "
                        f"{hit.get('score', '')}"
                    )
                    with st.expander(title):
                        st.caption(f"{hit.get('kind', 'source')} · {hit.get('symbol', '')}")
                        st.code(hit.get("snippet", ""), language="text")

    with dirtree_tab:
        render_directory_tree_tab(st, service, state, paths)

    with graph_tab:
        render_relationships_tab(st, service, index)