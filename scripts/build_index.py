#!/usr/bin/env python3
"""
Usage: python scripts/build_index.py <repo_path> [out_dir]

Indexes a Python repository and writes symbols.json, graph.json and
chunks.json into out_dir (default: ./data).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agentrepo import index_repository, save_index


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    repo_path = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "data"

    try:
        index, graph = index_repository(repo_path)
        save_index(index, out_dir)
    except (OSError, ValueError) as error:
        print(f"Could not build index: {error}", file=sys.stderr)
        sys.exit(2)

    n_resolved_calls = sum(1 for e in index.edges if e.kind == "CALLS" and e.resolved)
    n_ambiguous_calls = sum(1 for e in index.edges if e.kind == "CALLS" and e.confidence == "ambiguous")
    n_external_calls = sum(1 for e in index.edges if e.kind == "CALLS" and e.confidence == "external")

    print(f"Indexed {len(index.repo_files)} repository files ({len(index.files)} Python),  {len(index.symbols)} symbols, {len(index.edges)} edges, {len(index.chunks)} chunks")
    print(f"CALLS edges -> resolved: {n_resolved_calls}, ambiguous: {n_ambiguous_calls}, external: {n_external_calls}")
    print(f"Wrote symbols.json / graph.json / chunks.json to {out_dir}/")


if __name__ == "__main__":
    main()
