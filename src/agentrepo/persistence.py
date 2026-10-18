"""Persist a RepositoryIndex as three plain, inspectable JSON files."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from .filesystem import is_within
from .graph import CodeGraph
from .models import RepositoryIndex, Symbol, Edge, Chunk, FileRecord


def _write_json(path: Path, value) -> None:
    """Replace one persisted index file atomically."""
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as temp:
            temp_path = Path(temp.name)
            json.dump(value, temp, indent=2)
            temp.flush()
            os.fsync(temp.fileno())
        os.replace(temp_path, path)
    except Exception:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise


def save_index(index: RepositoryIndex, out_dir: str | Path) -> None:
    out_dir_resolved = Path(out_dir).expanduser().resolve()
    root_resolved = Path(index.root).expanduser().resolve()
    if is_within(root_resolved, out_dir_resolved):
        raise ValueError("index output directory must be outside the repository being indexed")
    out_dir_resolved.mkdir(parents=True, exist_ok=True)

    files_and_symbols = {
        "root": index.root,
        "files": [f.to_dict() for f in index.files],
        "repo_files": index.repo_files,
        "symbols": {k: v.to_dict() for k, v in index.symbols.items()},
    }
    _write_json(out_dir_resolved / "symbols.json", files_and_symbols)
    _write_json(out_dir_resolved / "graph.json", [e.to_dict() for e in index.edges])
    _write_json(out_dir_resolved / "chunks.json", {k: v.to_dict() for k, v in index.chunks.items()})


def load_index(index_dir: str | Path) -> tuple[RepositoryIndex, CodeGraph]:
    """Inverse of save_index: rebuild (RepositoryIndex, CodeGraph) from the
    three JSON files, without re-parsing the repository."""
    index_dir = Path(index_dir)
    for name in ("symbols.json", "graph.json", "chunks.json"):
        if not (index_dir / name).is_file():
            raise FileNotFoundError(f"{index_dir / name} not found - run scripts/build_index.py first")

    try:
        symbols_data = json.loads((index_dir / "symbols.json").read_text(encoding="utf-8"))
        edges_data = json.loads((index_dir / "graph.json").read_text(encoding="utf-8"))
        chunks_data = json.loads((index_dir / "chunks.json").read_text(encoding="utf-8"))
        if (
            not isinstance(symbols_data, dict)
            or not isinstance(symbols_data.get("root"), str)
            or not isinstance(symbols_data.get("files"), list)
            or not isinstance(symbols_data.get("symbols"), dict)
            or not isinstance(edges_data, list)
            or not isinstance(chunks_data, dict)
        ):
            raise ValueError("persisted index JSON has an invalid top-level structure")
        index = RepositoryIndex(
            root=symbols_data["root"],
            files=[FileRecord(**f) for f in symbols_data["files"]],
            repo_files=symbols_data.get("repo_files"),
            symbols={k: Symbol(**v) for k, v in symbols_data["symbols"].items()},
            edges=[Edge(**e) for e in edges_data],
            chunks={k: Chunk(**v) for k, v in chunks_data.items()},
        )
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"corrupt or incompatible index in {index_dir}: {exc}") from exc
    return index, CodeGraph.from_index(index)
