"""Turn a file path (relative to the repo root) into a dotted module name."""

from __future__ import annotations

from pathlib import PurePosixPath


def path_to_module(rel_path: str) -> str:
    """
    'shop/models.py'          -> 'shop.models'
    'shop/__init__.py'        -> 'shop'
    '__init__.py'             -> ''  (repo-root package, rare/edge case)
    'shop/services/order.py'  -> 'shop.services.order'
    """
    p = PurePosixPath(rel_path)
    parts = list(p.parts)
    if not parts:
        return ""
    if parts[-1].casefold() == "__init__.py":
        parts = parts[:-1]
    else:
        parts[-1] = p.stem
    return ".".join(parts)
