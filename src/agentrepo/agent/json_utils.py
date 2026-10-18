"""Small helpers for parsing structured model replies."""

from __future__ import annotations

import json
import re


_MAX_REPLY_CHARS = 200_000
_MAX_OBJECT_CANDIDATES = 64


def parse_json_object(text: str) -> dict | None:
    """Find the first JSON object in a bounded model reply.

    Providers sometimes wrap otherwise valid JSON in a sentence or a code
    fence. Callers still validate the object's schema before using any field.
    """
    if not isinstance(text, str):
        return None
    text = text[:_MAX_REPLY_CHARS]
    decoder = json.JSONDecoder()
    candidates = 0
    for match in re.finditer(r"\{", text):
        candidates += 1
        if candidates > _MAX_OBJECT_CANDIDATES:
            break
        try:
            value, _ = decoder.raw_decode(text, match.start())
        except (TypeError, ValueError):
            continue
        if isinstance(value, dict):
            return value
    return None
