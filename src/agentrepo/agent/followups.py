"""Small, conservative policies for obvious read-only follow-up actions."""

from __future__ import annotations

import hashlib
import json
import re


_CONTENT_REQUEST = re.compile(
    r"\b(what|who|how|why|explain|describe|show|read|content|implementation|"
    r"defined|definition|function|class|method|flow|used|uses|works|value|accuracy)\b",
    re.IGNORECASE,
)
_PATH_ONLY_REQUEST = re.compile(
    r"\b(which|what|list|find|show)\b.{0,35}\b(files?|paths?|filenames?)\b",
    re.IGNORECASE,
)
_LOCATION_ONLY_REQUEST = re.compile(
    r"^\s*(where|which\s+file|what\s+file)\b|\bwhere\b.{0,35}\b(defined|located|location|line|file)\b",
    re.IGNORECASE,
)
_REQUEST_ECHO_KEYS = {"query", "pattern", "args", "objective", "question", "request", "task"}
_ACTION_WORDS = {"find", "search", "investigate", "repository", "repo", "file", "files",
                 "read", "show", "tell", "explain", "what", "where", "which", "the", "and", "for"}


def deterministic_followup(tool_name: str, report: str, task: str) -> dict | None:
    """Return only an exact, single-result read, directly implied by a lookup."""
    if (not task or not _CONTENT_REQUEST.search(task) or _PATH_ONLY_REQUEST.search(task)
            or _LOCATION_ONLY_REQUEST.search(task)):
        return None
    try:
        payload = json.loads(report)
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict) or "error" in payload:
        return None

    if tool_name == "find_files":
        files = payload.get("files")
        pattern = str(payload.get("pattern") or "").replace("\\", "/").strip(" ./").casefold()
        if (not isinstance(files, list) or len(files) != 1 or payload.get("total_matches") != 1
                or payload.get("truncated") or not pattern or any(char in pattern for char in "*?[")):
            return None
        path = files[0]
        if not isinstance(path, str) or not _same_file_name(pattern, path):
            return None
        return {"tool": "get_file", "arguments": {"path": path},
                "reason": "one exact file matched the requested filename"}

    if tool_name == "find_symbol":
        matches = payload.get("matches")
        if (payload.get("match_type") != "exact" or payload.get("ambiguous") is not False
                or payload.get("count") != 1 or not isinstance(matches, list) or len(matches) != 1):
            return None
        match = matches[0]
        if not isinstance(match, dict):
            return None
        path, start, end = match.get("file"), match.get("line_start"), match.get("line_end")
        if (not isinstance(path, str) or not isinstance(start, int) or isinstance(start, bool)
                or not isinstance(end, int) or isinstance(end, bool) or start < 1 or end < start):
            return None
        return {"tool": "get_file", "arguments": {"path": path, "start_line": start, "end_line": end},
                "reason": "one exact, unambiguous symbol definition matched"}
    return None


def result_fingerprint(tool_name: str, report: str) -> str:
    """Hash the returned information without request wording or wrapper noise."""
    try:
        data = json.loads(report)
    except (TypeError, ValueError):
        data = " ".join(str(report).split())
    if isinstance(data, dict):
        if "error" in data:
            semantic = {"error": data["error"]}
        elif tool_name == "find_files":
            semantic = {key: data.get(key) for key in ("files", "total_matches", "truncated")}
        elif tool_name == "find_symbol":
            semantic = {key: data.get(key) for key in ("match_type", "ambiguous", "matches")}
        elif tool_name == "get_file":
            semantic = {key: data.get(key) for key in
                        ("file", "line_start", "line_end", "total_lines", "truncated", "content")}
        elif tool_name == "investigate_repository":
            semantic = {"observations": _strip_request_echo(data.get("raw_observations", []))}
        else:
            semantic = _strip_request_echo(data)
    else:
        semantic = data
    encoded = json.dumps([tool_name, semantic], ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def similar_action(left: str, right: str) -> bool:
    """Conservative text similarity used only to block repeated read-only work."""
    a = set(re.findall(r"[a-z0-9_]+", (left or "").casefold())) - _ACTION_WORDS
    b = set(re.findall(r"[a-z0-9_]+", (right or "").casefold())) - _ACTION_WORDS
    if not a or not b:
        return " ".join((left or "").casefold().split()) == " ".join((right or "").casefold().split())
    return len(a & b) / min(len(a), len(b)) >= 0.85


def _same_file_name(pattern: str, path: str) -> bool:
    """True when `pattern` plausibly names `path`: the full normalized path,
    its basename, or the basename without its extension (so a pattern like
    "README" matches a found path of "README.md" - a one-exact-match
    find_files result naming a file by its common/stem name, without the
    extension, is exactly the case this auto-follow-up exists for)."""
    normalized = path.replace("\\", "/").strip(" ./").casefold()
    basename = normalized.rsplit("/", 1)[-1]
    stem = basename.rsplit(".", 1)[0] if "." in basename else basename
    return pattern in (normalized, basename, stem)


def _strip_request_echo(value):
    if isinstance(value, dict):
        return {key: _strip_request_echo(item) for key, item in value.items()
                if str(key).casefold() not in _REQUEST_ECHO_KEYS}
    if isinstance(value, list):
        return [_strip_request_echo(item) for item in value]
    if isinstance(value, str) and value[:1] in ("{", "["):
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            return value
        if isinstance(parsed, (dict, list)):
            return _strip_request_echo(parsed)
    return value
