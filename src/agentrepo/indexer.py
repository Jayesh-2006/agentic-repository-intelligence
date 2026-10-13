"""
Top-level entry point: index_repository(root) walks a directory of Python
files and returns a RepositoryIndex plus a CodeGraph built from its edges.
"""

from __future__ import annotations

import os
import ast
import hashlib
import tokenize
from pathlib import Path

from .models import RepositoryIndex, FileRecord, Chunk
from .module_naming import path_to_module
from .collector import collect_module, ModuleCollection
from .resolver import resolve_repository
from .graph import CodeGraph
from .filesystem import is_ignored_name, is_ignored_path, is_virtualenv_root

# Non-Python files whose text is chunked so search_repository can find it. Every other file
# (images, archives, unknown types...) is only listed in repo_files.
TEXT_EXTENSIONS = {".md", ".rst", ".txt", ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".xml",
                   ".html", ".htm", ".css", ".sh", ".bash", ".bat", ".csv", ".conf", ".gitignore"}
TEXT_FILENAMES = {"readme", "license", "licence", "copying", "notice", "authors", "changelog",
                  "dockerfile", "makefile", "requirements"}  # compared to the lower-cased name without extension
MAX_TEXT_BYTES = 200_000  # bigger files are listed but not content-indexed
TEXT_CHUNK_LINES = 40

def _iter_python_files(root: Path):
    paths = []
    def raise_walk_error(error):
        raise error
    for folder, dirs, names in os.walk(root, followlinks=False, onerror=raise_walk_error):
        dirs[:] = [d for d in dirs if not is_ignored_name(d)
                   and not (Path(folder) / d).is_symlink()
                   and not is_virtualenv_root(Path(folder) / d)]
        for name in names:
            path = Path(folder) / name
            if name.casefold().endswith(".py") and not path.is_symlink():
                paths.append(path)
    yield from sorted(paths)


def _list_repository_files(root: Path) -> list[str]:
    """Repo-relative paths of every file (any type) outside the ignored directories."""
    paths = []
    def raise_walk_error(error):
        raise error
    for folder, dirs, names in os.walk(root, followlinks=False, onerror=raise_walk_error):
        dirs[:] = [d for d in dirs if not is_ignored_name(d)
                   and not (Path(folder) / d).is_symlink()
                   and not is_virtualenv_root(Path(folder) / d)]
        for name in names:
            path = Path(folder) / name
            if not path.is_symlink() and not is_ignored_path(path.relative_to(root)):
                paths.append(path.relative_to(root).as_posix())
    return sorted(paths)


def _is_text_candidate(rel_path: str) -> bool:
    name = rel_path.rsplit("/", 1)[-1].lower()
    return not name.endswith(".py") and (Path(name).suffix.lower() in TEXT_EXTENSIONS or name in TEXT_EXTENSIONS
                                        or name.split(".")[0] in TEXT_FILENAMES)


def _read_text_file(path: Path) -> str | None:
    """The file's text, or None if it is too big, empty, binary or not UTF-8."""
    try:
        if path.stat().st_size > MAX_TEXT_BYTES:
            return None
        data = path.read_bytes()
        return data.decode("utf-8") if data and b"\0" not in data else None
    except (OSError, UnicodeDecodeError):
        return None


def _build_text_chunks(root: Path, repo_files: list[str]) -> dict[str, Chunk]:
    """Simple line-window chunks for supported text files (no symbols)."""
    chunks: dict[str, Chunk] = {}
    for rel_path in repo_files:
        if not _is_text_candidate(rel_path):
            continue
        text = _read_text_file(root / rel_path)
        if not text or not text.strip():
            continue
        lines = text.splitlines()
        for number, start in enumerate(range(0, len(lines), TEXT_CHUNK_LINES), start=1):
            part = lines[start:start + TEXT_CHUNK_LINES]
            if not "".join(part).strip():
                continue
            chunk_id = f"{rel_path}#text{number}"
            chunks[chunk_id] = Chunk(id=chunk_id, symbol_id=None, file=rel_path, kind="text",
                                     start_line=start + 1, end_line=start + len(part), text="\n".join(part))
    return chunks


def index_repository(root: str | Path) -> tuple[RepositoryIndex, CodeGraph]:
    try:
        root = Path(root).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"repository root is not accessible: {root}: {exc}") from exc
    if not root.is_dir():
        raise ValueError(f"repository root is not a directory: {root}")
    collections: list[ModuleCollection] = []
    file_records: list[FileRecord] = []
    file_sources: dict[str, list[str]] = {}  # module -> source lines, for chunking

    try:
        python_paths = list(_iter_python_files(root))
    except OSError as exc:
        raise ValueError(f"could not scan repository root {root}: {exc}") from exc
    logical_modules = [path_to_module(path.relative_to(root).as_posix()) for path in python_paths]
    module_paths: dict[str, list[Path]] = {}
    for path, module_name in zip(python_paths, logical_modules):
        module_paths.setdefault(module_name, []).append(path)
    canonical_paths = {}
    for module_name, paths in module_paths.items():
        package_paths = [path for path in paths if path.name.casefold() == "__init__.py"]
        canonical_paths[module_name] = min(package_paths or paths)
    for path in python_paths:
        rel_path = str(path.relative_to(root).as_posix())
        module = path_to_module(rel_path)
        if len(module_paths[module]) > 1 and path != canonical_paths[module]:
            # The dotted module is a logical import name; this suffix gives
            # each colliding physical source file a separate index identity.
            digest = hashlib.sha1(rel_path.encode("utf-8")).hexdigest()[:8]
            module = f"{module}.__physical_{digest}" if module else f"__physical_{digest}"
        try:
            with tokenize.open(path) as source_file:
                source = source_file.read()
        except (OSError, UnicodeError, SyntaxError) as exc:
            file_records.append(FileRecord(path=rel_path, module=module, loc=0, parse_ok=False,
                                           error=f"could not read file: {exc}"))
            continue
        loc = len(source.splitlines())

        try:
            collection = collect_module(module=module, file_path=rel_path, source=source)
            collections.append(collection)
            file_sources[module] = source.splitlines()
            file_records.append(FileRecord(path=rel_path, module=module, loc=loc, parse_ok=True))
        except SyntaxError as e:
            file_records.append(FileRecord(path=rel_path, module=module, loc=loc, parse_ok=False, error=str(e)))

    symbols, edges = resolve_repository(collections)
    chunks = _build_chunks(collections, file_sources)
    try:
        repo_files = _list_repository_files(root)
    except OSError as exc:
        raise ValueError(f"could not list repository files under {root}: {exc}") from exc
    chunks.update(_build_text_chunks(root, repo_files))

    index = RepositoryIndex(root=str(root), files=file_records, symbols=symbols, edges=edges, chunks=chunks,
                            repo_files=repo_files)

    return index, CodeGraph.from_index(index)


def _build_chunks(collections: list[ModuleCollection], file_sources: dict[str, list[str]]) -> dict[str, Chunk]:
    chunks: dict[str, Chunk] = {}
    for c in collections:
        lines = file_sources[c.module]
        try:
            tree = ast.parse("\n".join(lines), filename=c.file_path)
            module_lines = list(lines)
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    end = getattr(node, "end_lineno", node.lineno)
                    for line_number in range(node.lineno - 1, end):
                        module_lines[line_number] = ""
            text = "\n".join(module_lines)
            if text.strip():
                chunk_id = f"{c.module}#module"
                chunks[chunk_id] = Chunk(id=chunk_id, symbol_id=None, file=c.file_path, kind="module",
                                         start_line=1, end_line=len(lines), text=text)
        except SyntaxError:
            pass

        for sym in c.symbols.values():
            if sym.kind == "module":
                continue
            start = sym.lineno
            end = sym.end_lineno or sym.lineno
            text = "\n".join(lines[start - 1 : end])
            chunk_id = f"{sym.id}#chunk"
            chunks[chunk_id] = Chunk(
                id=chunk_id, symbol_id=sym.id, file=sym.file, kind=sym.kind, start_line=start, end_line=end, text=text
            )
    return chunks
