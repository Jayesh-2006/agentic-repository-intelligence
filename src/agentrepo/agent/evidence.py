"""
Evidence-grounded completion for the Main Agent.

    objective -> acceptance criteria -> steps with evidence
              -> which criteria the evidence supports -> completion state

All ten read-only repository tools (`search_repository`, `find_files`,
`find_symbol`, `get_file`, `get_symbol`, `find_callers`, `find_callees`,
`find_importers`, `get_dependencies`, `get_ast`) as well as guarded mutation
and verification tools (`write_file`, `verify_repository`, `run_command`)
produce first-class evidence records with structured provenance, source
locations, and file/line-range coverage.
"""

from __future__ import annotations

import json
import re
from typing import Optional

from .json_utils import parse_json_object
from .plan import (
    COMPLETED,
    FAILED,
    IN_PROGRESS,
    _normalize,
    is_readonly_step_description,
    pending_steps,
)

# Completion states (what the gathered evidence supports).
SUPPORTED = "SUPPORTED"      # every acceptance criterion has usable evidence
PARTIAL = "PARTIAL"          # some criteria supported, or work still outstanding
UNSUPPORTED = "UNSUPPORTED"  # no criterion is supported by usable evidence
BLOCKED = "BLOCKED"          # the latest change/verification failed, or nothing usable could be gathered
GRADING_FAILED = "GRADING_FAILED"  # evidence grading failed after its bounded format retry

# Result of a single step's tool call.
RESULT_OK = "ok"
RESULT_FAILED = "failed"
RESULT_INCONCLUSIVE = "inconclusive"  # ran, but partial/truncated or proves nothing (NO_TESTS / UNAVAILABLE)
RESULT_EMPTY = "empty"

MAX_CRITERIA = 4
MAX_COMPLETION_RETRIES = 2
MAX_EVIDENCE_CHARS = 6000
GRADER_FAILED_NOTE = "grading call failed"
_ACTION_LABEL_CHARS = 160
_NOTE_CHARS = 200


def _clip(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


# ---------------------------------------------------------------- criteria


def new_criterion(index: int, description: str, evidence_needed: str) -> dict:
    return {"id": f"c{index}", "description": description, "evidence_needed": evidence_needed,
            "satisfied": False, "evidence_ids": [], "note": ""}


def default_criteria(task: str) -> list[dict]:
    """Generic fallback when the planning call gave no usable criteria: one
    criterion parameterised by the task text."""
    return [new_criterion(1, f"The objective is fully addressed: {task}",
                          "Concrete repository evidence (file/document contents, code, structure, or "
                          "verification results, as applicable) that directly shows it")]


def parse_criteria(text: str, max_criteria: int = MAX_CRITERIA) -> list[dict]:
    """Extract acceptance criteria from the planning reply."""
    data = parse_json_object(text or "")
    items = data.get("criteria") if data is not None else None
    if not isinstance(items, list):
        return []
    criteria: list[dict] = []
    for item in items:
        if isinstance(item, str):
            description, needed = item, ""
        elif isinstance(item, dict):
            description = item.get("criterion") or item.get("description") or ""
            needed = item.get("evidence_needed") or item.get("evidence") or ""
        else:
            continue
        if isinstance(description, str) and description.strip():
            criteria.append(new_criterion(len(criteria) + 1, _clip(description, 240),
                                          _clip(needed, 240) if isinstance(needed, str) else ""))
        if len(criteria) >= max(1, max_criteria):
            break
    return criteria


# ------------------------------------------------------------ step evidence


def _action_key(tool: str, arg: str) -> str:
    return f"{tool}|{_normalize(arg)}"[:300]


def _is_valid_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _extract_collection_items(value) -> list | None:
    if isinstance(value, list):
        return value
    if isinstance(value, dict) and isinstance(value.get("items"), list):
        return value["items"]
    return None


def _collection_omitted_count(value, total_matches: int | None = None) -> int:
    if isinstance(value, dict):
        omitted = value.get("omitted")
        if _is_valid_int(omitted) and omitted > 0:
            return int(omitted)
        items = value.get("items")
        if isinstance(items, list) and _is_valid_int(total_matches):
            return max(0, int(total_matches) - len(items))
    if isinstance(value, list) and _is_valid_int(total_matches):
        return max(0, int(total_matches) - len(value))
    return 0


def _has_valid_optional_location(entry: dict) -> bool:
    path = entry.get("file")
    if path is not None and not (isinstance(path, str) and path.strip()):
        return False
    line = entry.get("line")
    if line is not None and not (_is_valid_int(line) and line >= 1):
        return False
    start, end = entry.get("line_start"), entry.get("line_end")
    if start is not None or end is not None:
        if not (_is_valid_int(start) and _is_valid_int(end) and 1 <= start <= end):
            return False
    return True


def _is_valid_site_entry(entry) -> bool:
    if not isinstance(entry, dict) or not _has_valid_optional_location(entry):
        return False
    sym = entry.get("symbol")
    if sym is not None and not (isinstance(sym, str) and sym.strip()):
        return False
    path = entry.get("file")
    return bool((isinstance(sym, str) and sym.strip()) or (isinstance(path, str) and path.strip()))


def _is_valid_callee_entry(entry) -> bool:
    if not isinstance(entry, dict) or not _has_valid_optional_location(entry):
        return False
    name = entry.get("name")
    if name is not None and not (isinstance(name, str) and name.strip()):
        return False
    if isinstance(name, str) and name.strip():
        return True
    return _is_valid_site_entry(entry)


def _is_valid_importer_entry(entry) -> bool:
    if not isinstance(entry, dict) or not _has_valid_optional_location(entry):
        return False
    mod = entry.get("module")
    if mod is not None and not (isinstance(mod, str) and mod.strip()):
        return False
    path = entry.get("file")
    return bool((isinstance(mod, str) and mod.strip()) or (isinstance(path, str) and path.strip()))


_GET_FILE_ERROR_PREFIX = '{"error":'
_GET_FILE_SUCCESS_PREFIX = '{"file":'


def classify_get_file_result(report: str) -> str:
    """Dedicated, conservative classifier for a direct get_file result."""
    text = (report or "").strip()
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        data = None
    if isinstance(data, dict):
        if "error" in data:
            return RESULT_FAILED
        content = data.get("content")
        if not (isinstance(content, str) and content):
            return RESULT_INCONCLUSIVE if data.get("truncated") else RESULT_EMPTY
        return RESULT_INCONCLUSIVE if data.get("truncated") else RESULT_OK
    if text.startswith(_GET_FILE_ERROR_PREFIX):
        return RESULT_FAILED
    return RESULT_INCONCLUSIVE


def classify_get_symbol_result(report: str) -> str:
    """Classify a direct get_symbol observation."""
    try:
        data = json.loads((report or "").strip())
    except (TypeError, ValueError):
        return RESULT_INCONCLUSIVE
    if not isinstance(data, dict):
        return RESULT_INCONCLUSIVE
    if "error" in data:
        return RESULT_FAILED
    sym = data.get("symbol")
    src = data.get("source")
    if not (isinstance(sym, str) and sym.strip() and isinstance(src, str)):
        return RESULT_INCONCLUSIVE
    if not src.strip():
        return RESULT_INCONCLUSIVE if data.get("truncated") else RESULT_EMPTY
    return RESULT_OK


def classify_find_files_result(report: str) -> str:
    """Classify both ordinary and list-bounded find_files observations."""
    try:
        data = json.loads((report or "").strip())
    except (TypeError, ValueError):
        return RESULT_INCONCLUSIVE
    if not isinstance(data, dict):
        return RESULT_INCONCLUSIVE
    if "error" in data:
        return RESULT_FAILED
    items = _extract_collection_items(data.get("files"))
    if items is not None:
        return RESULT_OK if items else RESULT_EMPTY
    return RESULT_INCONCLUSIVE


def classify_find_symbol_result(report: str) -> str:
    """Classify a direct find_symbol observation."""
    try:
        data = json.loads((report or "").strip())
    except (TypeError, ValueError):
        return RESULT_INCONCLUSIVE
    if not isinstance(data, dict):
        return RESULT_INCONCLUSIVE
    if "error" in data:
        err = data.get("error")
        if isinstance(err, dict) and err.get("code") == "symbol_not_found":
            return RESULT_EMPTY
        return RESULT_FAILED
    matches = _extract_collection_items(data.get("matches"))
    return RESULT_OK if matches else RESULT_EMPTY


def classify_search_repository_result(report: str) -> str:
    """Classify a direct search_repository observation."""
    try:
        data = json.loads((report or "").strip())
    except (TypeError, ValueError):
        return RESULT_INCONCLUSIVE
    if not isinstance(data, dict):
        return RESULT_INCONCLUSIVE
    if "error" in data:
        return RESULT_FAILED
    items = _extract_collection_items(data.get("results"))
    if items is None:
        return RESULT_INCONCLUSIVE
    if not items:
        return RESULT_EMPTY
    valid_hits = [
        hit for hit in items
        if isinstance(hit, dict) and isinstance(hit.get("file"), str) and hit["file"].strip()
        and isinstance(hit.get("snippet"), str) and hit["snippet"].strip()
    ]
    return RESULT_OK if valid_hits else RESULT_INCONCLUSIVE


def classify_find_callers_result(report: str) -> str:
    """Classify a direct find_callers observation."""
    try:
        parsed = json.loads((report or "").strip())
    except (TypeError, ValueError):
        return RESULT_INCONCLUSIVE
    if not isinstance(parsed, dict):
        return RESULT_INCONCLUSIVE
    if "error" in parsed:
        return RESULT_FAILED
    sym = parsed.get("symbol")
    if not (isinstance(sym, str) and sym.strip()):
        return RESULT_INCONCLUSIVE
    present_keys = [k for k in ("callers", "possible_callers") if k in parsed]
    if not present_keys:
        return RESULT_INCONCLUSIVE
    caller_lists: list[list] = []
    for key in present_keys:
        items = _extract_collection_items(parsed.get(key))
        if items is None:
            return RESULT_INCONCLUSIVE
        caller_lists.append(items)
    entries = [entry for items in caller_lists for entry in items]
    if not entries:
        return RESULT_EMPTY
    if not all(_is_valid_site_entry(entry) for entry in entries):
        return RESULT_INCONCLUSIVE
    return RESULT_OK


def classify_find_callees_result(report: str) -> str:
    """Classify a direct find_callees observation."""
    try:
        parsed = json.loads((report or "").strip())
    except (TypeError, ValueError):
        return RESULT_INCONCLUSIVE
    if not isinstance(parsed, dict):
        return RESULT_INCONCLUSIVE
    if "error" in parsed:
        return RESULT_FAILED
    sym = parsed.get("symbol")
    if not (isinstance(sym, str) and sym.strip()):
        return RESULT_INCONCLUSIVE
    present_keys = [k for k in ("callees", "possible_callees", "external_calls") if k in parsed]
    if not present_keys:
        return RESULT_INCONCLUSIVE
    by_key: dict[str, list] = {}
    for key in present_keys:
        items = _extract_collection_items(parsed.get(key))
        if items is None:
            return RESULT_INCONCLUSIVE
        by_key[key] = items
    if not any(by_key.values()):
        return RESULT_EMPTY
    if not all(_is_valid_site_entry(entry) for entry in by_key.get("callees", [])):
        return RESULT_INCONCLUSIVE
    if not all(_is_valid_callee_entry(entry) for entry in by_key.get("possible_callees", [])):
        return RESULT_INCONCLUSIVE
    if not all(_is_valid_callee_entry(entry) for entry in by_key.get("external_calls", [])):
        return RESULT_INCONCLUSIVE
    return RESULT_OK


def classify_find_importers_result(report: str) -> str:
    """Classify a direct find_importers observation."""
    try:
        parsed = json.loads((report or "").strip())
    except (TypeError, ValueError):
        return RESULT_INCONCLUSIVE
    if not isinstance(parsed, dict):
        return RESULT_INCONCLUSIVE
    if "error" in parsed:
        return RESULT_FAILED
    mod = parsed.get("module")
    importers = _extract_collection_items(parsed.get("importers"))
    if not (isinstance(mod, str) and mod.strip()) or importers is None:
        return RESULT_INCONCLUSIVE
    if not importers:
        return RESULT_EMPTY
    if not all(_is_valid_importer_entry(entry) for entry in importers):
        return RESULT_INCONCLUSIVE
    return RESULT_OK


def classify_get_dependencies_result(report: str) -> str:
    """Classify a direct get_dependencies observation."""
    try:
        parsed = json.loads((report or "").strip())
    except (TypeError, ValueError):
        return RESULT_INCONCLUSIVE
    if not isinstance(parsed, dict):
        return RESULT_INCONCLUSIVE
    if "error" in parsed:
        return RESULT_FAILED
    sym = parsed.get("symbol")
    if not (isinstance(sym, str) and sym.strip()
            and isinstance(parsed.get("depends_on"), dict)
            and isinstance(parsed.get("depended_on_by"), dict)):
        return RESULT_INCONCLUSIVE
    return RESULT_OK


def classify_get_ast_result(report: str) -> str:
    """Classify a direct get_ast observation."""
    try:
        parsed = json.loads((report or "").strip())
    except (TypeError, ValueError):
        return RESULT_INCONCLUSIVE
    if not isinstance(parsed, dict):
        return RESULT_INCONCLUSIVE
    if "error" in parsed:
        return RESULT_FAILED
    sym = parsed.get("symbol")
    node_type = parsed.get("node_type")
    if not (isinstance(sym, str) and sym.strip() and isinstance(node_type, str) and node_type.strip()):
        return RESULT_INCONCLUSIVE
    return RESULT_OK


def classify_write_file_result(report: str, payload: dict) -> str:
    """Classify a guarded write_file observation."""
    if not isinstance(payload, dict) or "error" in payload:
        return RESULT_FAILED
    state = payload.get("state")
    if state in ("FAILED", "TIMEOUT", "REPAIR_LIMIT_REACHED"):
        return RESULT_FAILED
    if state == "UNAVAILABLE":
        return RESULT_INCONCLUSIVE
    if state in ("SUCCESS", "NO_TESTS") and isinstance(payload.get("file"), str) and payload["file"].strip():
        return RESULT_OK
    if state == "SUCCESS":
        return RESULT_OK
    if isinstance(payload.get("file"), str) and payload["file"].strip() and state is None:
        return RESULT_OK
    return RESULT_INCONCLUSIVE


def classify_run_command_result(report: str, payload: dict) -> str:
    """Classify a direct run_command observation, including command-mutation verification state."""
    if not isinstance(payload, dict) or "error" in payload:
        return RESULT_FAILED
    state = payload.get("state")
    if state in ("FAILED", "TIMEOUT", "REPAIR_LIMIT_REACHED"):
        return RESULT_FAILED
    if state == "UNAVAILABLE":
        return RESULT_INCONCLUSIVE
    if payload.get("timed_out"):
        return RESULT_FAILED
    if payload.get("exit_code") == 0:
        return RESULT_OK
    return RESULT_FAILED


def classify_result(tool: str, report: str, payload: dict, code_tools: tuple = ("write_file", "verify_repository")) -> str:
    """Deterministic classification of any direct repository or verification tool result."""
    raw = (report or "").strip()
    if raw.startswith("Invalid tool arguments:"):
        return RESULT_FAILED
    if tool == "get_file":
        return classify_get_file_result(raw)
    if tool == "get_symbol":
        return classify_get_symbol_result(raw)
    if tool == "find_files":
        return classify_find_files_result(raw)
    if tool == "find_symbol":
        return classify_find_symbol_result(raw)
    if tool == "search_repository":
        return classify_search_repository_result(raw)
    if tool == "find_callers":
        return classify_find_callers_result(raw)
    if tool == "find_callees":
        return classify_find_callees_result(raw)
    if tool == "find_importers":
        return classify_find_importers_result(raw)
    if tool == "get_dependencies":
        return classify_get_dependencies_result(raw)
    if tool == "get_ast":
        return classify_get_ast_result(raw)
    if tool == "write_file":
        return classify_write_file_result(raw, payload)
    if tool == "run_command":
        return classify_run_command_result(raw, payload)
    if tool in code_tools or tool == "verify_repository":
        state = payload.get("state")
        if state == "SUCCESS":
            return RESULT_OK
        if state in ("NO_TESTS", "UNAVAILABLE"):
            return RESULT_INCONCLUSIVE
        return RESULT_FAILED
    if not raw:
        return RESULT_EMPTY
    if isinstance(payload, dict) and "error" in payload:
        return RESULT_FAILED
    return RESULT_OK


def extract_tool_file_ranges(tool_name: str, parsed: dict) -> dict[str, list[list[int]]]:
    """Extract verified (file -> merged [[line_start, line_end], ...]) intervals from a
    direct repository tool result dict."""
    if not isinstance(parsed, dict) or "error" in parsed:
        return {}
    ranges: dict[str, list[tuple[int, int]]] = {}
    path = parsed.get("file")
    start, end = parsed.get("line_start"), parsed.get("line_end")
    if isinstance(path, str) and path.strip() and _is_valid_int(start) and _is_valid_int(end) and 1 <= start <= end:
        ranges.setdefault(path.strip(), []).append((int(start), int(end)))

    for list_key in ("results", "matches", "children"):
        items = _extract_collection_items(parsed.get(list_key))
        for entry in items or []:
            if not isinstance(entry, dict):
                continue
            entry_path = entry.get("file") or (path if list_key == "children" and isinstance(path, str) else None)
            e_start, e_end = entry.get("line_start"), entry.get("line_end")
            if (isinstance(entry_path, str) and entry_path.strip()
                    and _is_valid_int(e_start) and _is_valid_int(e_end) and 1 <= e_start <= e_end):
                ranges.setdefault(entry_path.strip(), []).append((int(e_start), int(e_end)))

    for site_key in ("callers", "possible_callers", "callees", "possible_callees", "external_calls", "importers"):
        items = _extract_collection_items(parsed.get(site_key))
        for entry in items or []:
            if not isinstance(entry, dict):
                continue
            entry_path = entry.get("file") or (
                path if site_key in ("possible_callees", "external_calls") and isinstance(path, str) else None
            )
            if not (isinstance(entry_path, str) and entry_path.strip()):
                continue
            line = entry.get("line")
            e_start, e_end = entry.get("line_start"), entry.get("line_end")
            if _is_valid_int(line) and line >= 1:
                ranges.setdefault(entry_path.strip(), []).append((int(line), int(line)))
            elif _is_valid_int(e_start) and _is_valid_int(e_end) and 1 <= e_start <= e_end:
                ranges.setdefault(entry_path.strip(), []).append((int(e_start), int(e_end)))

    if tool_name == "get_ast" and isinstance(path, str) and path.strip():
        for list_key in ("calls", "imports"):
            items = _extract_collection_items(parsed.get(list_key))
            for entry in items or []:
                if isinstance(entry, dict) and _is_valid_int(entry.get("line")) and entry["line"] >= 1:
                    ln = int(entry["line"])
                    ranges.setdefault(path.strip(), []).append((ln, ln))

    return {
        f_path: [[a, b] for a, b in _merge_line_ranges(intervals)]
        for f_path, intervals in ranges.items()
        if intervals
    }


def _search_repository_provenance(payload: dict) -> dict:
    results = _extract_collection_items(payload.get("results")) if isinstance(payload, dict) else None
    hits = [item for item in (results or []) if isinstance(item, dict)]
    files: list[str] = []
    for hit in hits:
        path = hit.get("file")
        if isinstance(path, str) and path.strip():
            p_str = path.strip()
            if p_str not in files:
                files.append(p_str)
    merged_file_ranges = extract_tool_file_ranges("search_repository", payload)
    omitted = _collection_omitted_count(payload.get("results"), payload.get("count"))
    return {
        "count": payload.get("count", len(hits)),
        "files": files[:20],
        "file_ranges": merged_file_ranges,
        "truncated": bool(omitted or payload.get("truncated") or payload.get("observation_truncated")),
    }


def _find_files_provenance(payload: dict) -> dict:
    files = payload.get("files")
    bounded = files if isinstance(files, dict) else {}
    shown = bounded.get("shown", len(files) if isinstance(files, list) else None)
    total = payload.get("total_matches")
    inferred_omitted = (
        max(0, int(total) - int(shown))
        if isinstance(total, int) and isinstance(shown, int)
        else 0
    )
    omitted = bounded.get("omitted", inferred_omitted)
    return {
        "pattern": payload.get("pattern"),
        "total_matches": total,
        "shown": shown,
        "omitted": omitted,
        "truncated": bool(omitted or payload.get("truncated") or payload.get("observation_truncated")),
    }


def _find_symbol_provenance(payload: dict) -> dict:
    matches = _extract_collection_items(payload.get("matches")) or []
    symbol_matches = [m for m in matches if isinstance(m, dict)]
    files: list[str] = []
    symbols: list[str] = []
    for m in symbol_matches:
        sym = m.get("symbol")
        if isinstance(sym, str) and sym and sym not in symbols:
            symbols.append(sym)
        f_path = m.get("file")
        if isinstance(f_path, str) and f_path and f_path not in files:
            files.append(f_path)
    omitted = _collection_omitted_count(payload.get("matches"), payload.get("count"))
    return {
        "match_type": payload.get("match_type"),
        "ambiguous": bool(payload.get("ambiguous")),
        "count": payload.get("count", len(symbol_matches)),
        "symbols": symbols[:25],
        "files": files[:20],
        "file_ranges": extract_tool_file_ranges("find_symbol", payload),
        "omitted": omitted,
        "truncated": bool(omitted or payload.get("truncated") or payload.get("observation_truncated")),
    }


def _get_symbol_provenance(payload: dict) -> dict:
    path = payload.get("file")
    start, end = payload.get("line_start"), payload.get("line_end")
    return {
        "symbol": payload.get("symbol"),
        "kind": payload.get("kind"),
        "file": path,
        "line_start": start,
        "line_end": end,
        "decorators": payload.get("decorators") or [],
        "truncated": bool(payload.get("truncated") or payload.get("observation_truncated")),
        "file_ranges": extract_tool_file_ranges("get_symbol", payload),
    }


def _find_callers_provenance(payload: dict) -> dict:
    callers = _extract_collection_items(payload.get("callers")) or []
    possible = _extract_collection_items(payload.get("possible_callers")) or []
    files: list[str] = []
    for entry in [*callers, *possible]:
        if isinstance(entry, dict):
            f_path = entry.get("file")
            if isinstance(f_path, str) and f_path and f_path not in files:
                files.append(f_path)
    omitted = (
        _collection_omitted_count(payload.get("callers"))
        + _collection_omitted_count(payload.get("possible_callers"))
    )
    return {
        "symbol": payload.get("symbol"),
        "kind": payload.get("kind"),
        "caller_count": len(callers),
        "possible_caller_count": len(possible),
        "files": files[:20],
        "file_ranges": extract_tool_file_ranges("find_callers", payload),
        "omitted": omitted,
        "truncated": bool(omitted or payload.get("truncated") or payload.get("observation_truncated")),
    }


def _find_callees_provenance(payload: dict) -> dict:
    callees = _extract_collection_items(payload.get("callees")) or []
    possible = _extract_collection_items(payload.get("possible_callees")) or []
    external = _extract_collection_items(payload.get("external_calls")) or []
    files: list[str] = []
    for entry in callees:
        if isinstance(entry, dict):
            f_path = entry.get("file")
            if isinstance(f_path, str) and f_path and f_path not in files:
                files.append(f_path)
    omitted = (
        _collection_omitted_count(payload.get("callees"))
        + _collection_omitted_count(payload.get("possible_callees"))
        + _collection_omitted_count(payload.get("external_calls"))
    )
    return {
        "symbol": payload.get("symbol"),
        "kind": payload.get("kind"),
        "callee_count": len(callees),
        "possible_callee_count": len(possible),
        "external_call_count": len(external),
        "files": files[:20],
        "file_ranges": extract_tool_file_ranges("find_callees", payload),
        "omitted": omitted,
        "truncated": bool(omitted or payload.get("truncated") or payload.get("observation_truncated")),
    }


def _find_importers_provenance(payload: dict) -> dict:
    importers = _extract_collection_items(payload.get("importers")) or []
    files: list[str] = []
    target_file = payload.get("file")
    if isinstance(target_file, str) and target_file:
        files.append(target_file)
    for entry in importers:
        if isinstance(entry, dict):
            f_path = entry.get("file")
            if isinstance(f_path, str) and f_path and f_path not in files:
                files.append(f_path)
    omitted = _collection_omitted_count(payload.get("importers"))
    return {
        "module": payload.get("module"),
        "file": target_file,
        "importer_count": len(importers),
        "files": files[:20],
        "file_ranges": extract_tool_file_ranges("find_importers", payload),
        "omitted": omitted,
        "truncated": bool(omitted or payload.get("truncated") or payload.get("observation_truncated")),
    }


def _get_dependencies_provenance(payload: dict) -> dict:
    dep_on = payload.get("depends_on") if isinstance(payload.get("depends_on"), dict) else {}
    dep_by = payload.get("depended_on_by") if isinstance(payload.get("depended_on_by"), dict) else {}
    f_path = payload.get("file")
    return {
        "symbol": payload.get("symbol"),
        "kind": payload.get("kind"),
        "file": f_path,
        "files": [f_path] if isinstance(f_path, str) and f_path else [],
        "depends_on_keys": sorted(dep_on.keys()),
        "depended_on_by_keys": sorted(dep_by.keys()),
        "file_ranges": extract_tool_file_ranges("get_dependencies", payload),
        "truncated": bool(payload.get("truncated") or payload.get("observation_truncated")),
    }


def _get_ast_provenance(payload: dict) -> dict:
    children = _extract_collection_items(payload.get("children")) or []
    calls = _extract_collection_items(payload.get("calls")) or []
    f_path = payload.get("file")
    return {
        "symbol": payload.get("symbol"),
        "kind": payload.get("kind"),
        "node_type": payload.get("node_type"),
        "file": f_path,
        "files": [f_path] if isinstance(f_path, str) and f_path else [],
        "line_start": payload.get("line_start"),
        "line_end": payload.get("line_end"),
        "children_count": len(children),
        "calls_count": len(calls),
        "file_ranges": extract_tool_file_ranges("get_ast", payload),
        "truncated": bool(payload.get("truncated") or payload.get("observation_truncated")),
    }


def get_file_provenance(report: str) -> dict:
    """What a get_file result actually covered: file, line range, total lines
    and whether the read was cut short. Never raises."""
    text = (report or "").strip()
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        data = None
    if isinstance(data, dict) and "error" not in data:
        content = data.get("content")
        content = content if isinstance(content, str) else ""
        path = data.get("file", "?")
        start, end, total = data.get("line_start"), data.get("line_end"), data.get("total_lines")
        if start is not None and end is not None:
            range_note = f"lines {start}-{end} of {total}" if total is not None else f"lines {start}-{end}"
        else:
            range_note = f"{total} line(s)" if total is not None else "unknown range"
        truncated = bool(data.get("truncated"))
        partial_range = bool(
            start is not None and end is not None and total is not None
            and (int(start) > 1 or int(end) < int(total))
        )
        next_start = data.get("next_start_line")
        if next_start is None and end is not None and total is not None and int(end) < int(total):
            next_start = int(end) + 1
        header = f"{path} [{range_note}{' (truncated - not the full file)' if truncated else ''}]:\n"
        out: dict = {
            "file": data.get("file"),
            "line_start": start,
            "line_end": end,
            "total_lines": total,
            "truncated": truncated,
            "partial_range": partial_range,
            "range_verified": bool(start is not None and end is not None and content),
            "empty_content": not bool(content),
            "evidence_excerpt_complete": len(header + content) <= MAX_EVIDENCE_CHARS,
            "file_ranges": extract_tool_file_ranges("get_file", data),
        }
        if next_start is not None:
            out["next_start_line"] = int(next_start)
        return out
    if text.startswith(_GET_FILE_SUCCESS_PREFIX):
        def num(key):
            m = re.search(rf'"{key}":(\d+)', text)
            return int(m.group(1)) if m else None
        f = re.search(r'"file":"([^"]*)"', text)
        start, total = num("line_start"), num("total_lines")
        shown = text.count("\\n")
        end = (start + shown) if start is not None else None
        content_match = re.search(r'"content":"(.*)', text, re.DOTALL)
        has_content = bool(content_match and content_match.group(1).strip(' "}'))
        return {
            "file": f.group(1) if f else None,
            "line_start": start,
            "line_end": end,
            "total_lines": total,
            "truncated": True,
            "partial_range": True,
            "range_verified": False,
            "observation_cut": True,
            "empty_content": not has_content,
            "evidence_excerpt_complete": False,
        }
    return {}


_FULL_FILE_RE = re.compile(
    r"\b(?:entire|whole|complete|full)\s+(?:(?:source|target)\s+|[\w./\\-]+\.(?:py|md|txt|json|ya?ml|toml|ini|cfg|sh)\s+)?file\b"
    r"|\b(?:entire|whole|full)\s+[\w./\\-]+\.(?:py|md|txt|json|ya?ml|toml|ini|cfg|sh)\b"
    r"|\bfile(?:'s)?\s+(?:entire|whole|complete|full)\s+contents\b",
    re.IGNORECASE,
)
_ENUMERATION_RE = re.compile(
    r"\b(how\s+many|all|every|list|count(?:\s+of)?|number\s+of|"
    r"complete\s+(?:list|listing|inventory|enumeration)|"
    r"full\s+(?:list|listing|inventory|enumeration)|inventory|enumeration)\b",
    re.IGNORECASE,
)


def _requires_full_file(criteria: Optional[list[dict]] = None, plan: Optional[dict] = None) -> bool:
    parts = []
    if isinstance(plan, dict) and plan.get("objective"):
        parts.append(str(plan["objective"]))
    for c in criteria or []:
        if isinstance(c, dict):
            parts.append(f"{c.get('description', '')} {c.get('evidence_needed', '')}")
    return bool(_FULL_FILE_RE.search(" ".join(parts)))


def _merge_line_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping or adjacent 1-based inclusive line intervals."""
    merged: list[tuple[int, int]] = []
    valid = [
        (int(a), int(b)) for a, b in ranges
        if _is_valid_int(a) and _is_valid_int(b) and 1 <= int(a) <= int(b)
    ]
    for start, end in sorted(valid):
        if merged and start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _file_coverage(plan: Optional[dict]) -> tuple[dict, dict, dict]:
    """Per-file line coverage from the latest repository version read."""
    from .intent import extract_hard_sources
    objective_text = str((plan or {}).get("objective") or "") if isinstance(plan, dict) else ""
    hard_sources = extract_hard_sources(objective_text) if objective_text else []
    reads: list[tuple[str, object, int, int, int]] = []
    latest_epoch: dict[str, tuple[object, int]] = {}
    for step in evidence_steps(plan):
        if hard_sources and not _step_within_hard_sources(step, hard_sources):
            continue
        provenance = step.get("provenance") or {}
        path = provenance.get("file")
        start, end, total = (provenance.get("line_start"), provenance.get("line_end"),
                             provenance.get("total_lines"))
        tool_name = step.get("tool")
        if tool_name == "get_file":
            if step.get("result") not in (RESULT_OK, RESULT_INCONCLUSIVE) \
                    or not path or start is None or end is None or total is None \
                    or provenance.get("observation_cut"):
                continue
        elif tool_name == "get_symbol":
            if step.get("result") != RESULT_OK or not path or start is None or end is None \
                    or provenance.get("truncated"):
                continue
        else:
            continue
        epoch = provenance.get("repository_epoch")
        order = int(step.get("order", 0))
        reads.append((path, epoch, start, end, total))
        if path not in latest_epoch or order >= latest_epoch[path][1]:
            latest_epoch[path] = (epoch, order)

    covered: dict[str, list[tuple[int, int]]] = {}
    totals: dict[str, int] = {}
    epochs: dict[str, object] = {}
    for path, epoch, start, end, total in reads:
        if epoch != latest_epoch[path][0]:
            continue
        covered.setdefault(path, []).append((int(start), int(end)))
        if total is not None:
            totals[path] = int(total)
        epochs[path] = epoch
    return covered, totals, epochs


def _reach(ranges: list[tuple[int, int]]) -> int:
    """Highest line reached by contiguous coverage starting at line 1."""
    reach = 0
    for a, b in sorted(ranges):
        if a <= reach + 1:
            reach = max(reach, b)
    return reach


def _read_gaps(plan: Optional[dict]) -> list[str]:
    """Files whose recorded get_file ranges do not cover the complete file."""
    covered, totals, _ = _file_coverage(plan)
    gaps = []
    for path in sorted(covered):
        if path not in totals:
            continue
        reach = _reach(covered[path])
        total = totals[path]
        if reach < total:
            max_end = max((b for _, b in covered[path]), default=0)
            if reach == 0 and 0 < max_end < total:
                gaps.append(
                    f"{path}: read only lines {min(a for a, _ in covered[path])}-{max_end} of {total} - "
                    f"continue with get_file start_line={max_end + 1} (or start_line=1 for the opening lines) "
                    "before answering anything that needs the whole file"
                )
            else:
                gaps.append(
                    f"{path}: read only up to line {reach} of {total} - read the rest with "
                    f"get_file start_line={reach + 1} before answering anything that needs the whole file"
                )
    return gaps


def _fully_read_file_epochs(plan: Optional[dict]) -> dict[str, object]:
    """Repository version for each file whose ranges cover the whole file."""
    covered, totals, epochs = _file_coverage(plan)
    return {path: epochs[path] for path in covered if path in totals and _reach(covered[path]) >= totals[path]}


def _is_range_verified_get_file(step: dict) -> bool:
    if step.get("tool") != "get_file" or step.get("result") not in (RESULT_OK, RESULT_INCONCLUSIVE):
        return False
    provenance = step.get("provenance") or {}
    if provenance.get("observation_cut") or provenance.get("empty_content"):
        return False
    if provenance.get("range_verified") is False:
        return False
    path = provenance.get("file")
    start, end = provenance.get("line_start"), provenance.get("line_end")
    if not (isinstance(path, str) and path and _is_valid_int(start) and _is_valid_int(end) and 1 <= start <= end):
        return False
    return bool(str(step.get("evidence") or "").strip())


def _step_within_hard_sources(step: dict, hard_sources: list[str]) -> bool:
    if not hard_sources:
        return True
    from .intent import source_allows_path
    scope_intent = {"sources": hard_sources}
    prov = step.get("provenance") if isinstance(step.get("provenance"), dict) else {}
    direct_file = prov.get("file")
    if isinstance(direct_file, str) and direct_file.strip():
        if not source_allows_path(scope_intent, direct_file):
            return False
    files_list = prov.get("files")
    if isinstance(files_list, list) and files_list:
        if any(isinstance(f, str) and f.strip() and not source_allows_path(scope_intent, f) for f in files_list):
            return False
    file_ranges = prov.get("file_ranges")
    if isinstance(file_ranges, dict) and file_ranges:
        if any(isinstance(f, str) and f.strip() and not source_allows_path(scope_intent, f) for f in file_ranges):
            return False
    return True


def usable_evidence_steps(plan: Optional[dict], criteria: Optional[list[dict]] = None) -> list[dict]:
    """Return individual usable results, including range-verified bounded reads
    for excerpt-scoped questions and full-file validated reads for whole-file requests."""
    from .intent import extract_hard_sources
    objective_text = str((plan or {}).get("objective") or "") if isinstance(plan, dict) else ""
    hard_sources = extract_hard_sources(objective_text) if objective_text else []
    fully_read = _fully_read_file_epochs(plan)
    needs_full_file = _requires_full_file(criteria, plan)
    all_steps = evidence_steps(plan)
    latest_mutation_order = max(
        (
            int(s.get("order", 0))
            for s in all_steps
            if s.get("tool") == "write_file"
            or (
                s.get("tool") in ("run_command", "verify_repository")
                and bool((s.get("provenance") or {}).get("changed_files"))
            )
        ),
        default=0,
    )
    usable = []
    for step in all_steps:
        if hard_sources and not _step_within_hard_sources(step, hard_sources):
            continue
        if step.get("tool") == "get_file":
            provenance = step.get("provenance") or {}
            path = provenance.get("file")
            in_fully_read = bool(path in fully_read and provenance.get("repository_epoch") == fully_read[path])
            if needs_full_file:
                is_complete_single_read = (
                    step.get("result") == RESULT_OK
                    and not provenance.get("truncated")
                    and not provenance.get("partial_range")
                )
                if in_fully_read or is_complete_single_read:
                    usable.append(step)
            else:
                if step.get("result") == RESULT_OK or in_fully_read or _is_range_verified_get_file(step):
                    usable.append(step)
        elif step.get("result") == RESULT_OK:
            if (
                step.get("tool") in ("verify_repository", "run_command")
                and int(step.get("order", 0)) < latest_mutation_order
            ):
                continue
            usable.append(step)
    return usable


def _clipped_full_files(plan: Optional[dict]) -> list[str]:
    covered, totals, coverage_epochs = _file_coverage(plan)
    fully_read = {path for path, ranges in covered.items()
                  if path in totals and _reach(ranges) >= totals[path]}
    visible_ranges: dict[str, list[tuple[int, int]]] = {}
    for step in evidence_steps(plan):
        provenance = step.get("provenance") or {}
        path = provenance.get("file")
        start, end = provenance.get("line_start"), provenance.get("line_end")
        excerpt_complete = provenance.get("evidence_excerpt_complete")
        if excerpt_complete is None:
            excerpt_complete = " [...] " not in str(step.get("evidence") or "")
        if (step.get("tool") == "get_file" and path in fully_read
                and provenance.get("repository_epoch") == coverage_epochs.get(path)
                and excerpt_complete is True
                and start is not None and end is not None):
            visible_ranges.setdefault(path, []).append((int(start), int(end)))
    return sorted({
        path for path in fully_read
        if _reach(visible_ranges.get(path, [])) < totals[path]
    })


def incomplete_reasons(criteria: list[dict], plan: Optional[dict]) -> list[str]:
    """Deterministic reasons a fully-graded result must not be called SUPPORTED:
    incomplete file coverage or grader-visible excerpts for complete claims,
    incomplete file/symbol/caller listings, or a failed attempt after the last usable read."""
    reasons = []
    text = " ".join(
        [str((plan or {}).get("objective", "")),
         *(f"{c.get('description', '')} {c.get('evidence_needed', '')}" for c in criteria)]
    )
    if _FULL_FILE_RE.search(text):
        covered, _, _ = _file_coverage(plan)
        if not covered:
            reasons.append("no complete file read was recorded for the whole-file request")
        reasons.extend(_read_gaps(plan))
        reasons.extend(
            f"{path}: the file was read across all lines, but clipped excerpts did not expose all content to grading"
            for path in _clipped_full_files(plan)
        )
    if _ENUMERATION_RE.search(text):
        for step in evidence_steps(plan):
            provenance = step.get("provenance") or {}
            omitted = int(provenance.get("omitted") or 0)
            if step.get("tool") == "find_files" and omitted > 0:
                reasons.append(
                    f"find_files omitted {omitted} of {provenance.get('total_matches')} matches; "
                    "the returned paths are not a complete listing"
                )
            elif step.get("tool") in ("find_symbol", "find_callers") and omitted > 0:
                reasons.append(
                    f"{step.get('tool')} omitted {omitted} entries; the returned results are not a complete listing"
                )
    steps = evidence_steps(plan)
    usable_orders = [s["order"] for s in usable_evidence_steps(plan, criteria)]
    failed_after = [s for s in steps if s["result"] == RESULT_FAILED
                    and (not usable_orders or s["order"] > max(usable_orders))]
    if usable_orders and failed_after:
        reasons.append(f"the latest attempt failed ({failed_after[-1]['step_id']}) after the last usable evidence")
    return reasons


def summarize_get_file_evidence(report: str) -> str:
    """get_file's own evidence excerpt - the actual file content with its
    path/line-range metadata."""
    text = (report or "").strip()
    data = parse_json_object(text)
    if not isinstance(data, dict) or "error" in data:
        return _clip(text, MAX_EVIDENCE_CHARS)
    path = data.get("file", "?")
    start, end, total = data.get("line_start"), data.get("line_end"), data.get("total_lines")
    if start is not None and end is not None:
        range_note = f"lines {start}-{end} of {total}" if total is not None else f"lines {start}-{end}"
    else:
        range_note = f"{total} line(s)" if total is not None else "unknown range"
    bounded_note = " (truncated - not the full file)" if data.get("truncated") else ""
    header = f"{path} [{range_note}{bounded_note}]:\n"
    excerpt = header + str(data.get("content", ""))
    if len(excerpt) > MAX_EVIDENCE_CHARS:
        excerpt = excerpt[: MAX_EVIDENCE_CHARS - 190] + " [...] " + excerpt[-180:]
    return excerpt


def summarize_get_symbol_evidence(report: str) -> str:
    """First-class evidence excerpt for get_symbol: symbol identity, file,
    line range, decorators, and source code."""
    text = (report or "").strip()
    data = _parse_report_object(text)
    if not data or "error" in data:
        return _clip(text, MAX_EVIDENCE_CHARS)
    sym = data.get("symbol", "?")
    kind = data.get("kind", "symbol")
    path = data.get("file", "?")
    start, end = data.get("line_start"), data.get("line_end")
    range_note = f"lines {start}-{end}" if start is not None and end is not None else "unknown range"
    trunc_note = " (truncated)" if data.get("truncated") else ""
    decs = data.get("decorators") or []
    dec_note = f" decorators={decs}" if decs else ""
    header = f"RAW GET_SYMBOL RESULT: {sym} ({kind}) in {path} [{range_note}{trunc_note}]{dec_note}:\n"
    excerpt = header + str(data.get("source", ""))
    if len(excerpt) > MAX_EVIDENCE_CHARS:
        excerpt = excerpt[: MAX_EVIDENCE_CHARS - 190] + " [...] " + excerpt[-180:]
    return excerpt


def summarize_evidence(
    tool: str,
    report: str,
    payload: dict,
    code_tools: tuple = ("write_file", "verify_repository"),
) -> str:
    """Build a concise, explicitly labelled first-class evidence excerpt for any direct tool."""
    if tool == "get_file":
        return summarize_get_file_evidence(report)
    if tool == "get_symbol":
        return summarize_get_symbol_evidence(report)

    text = " ".join(str(report).split())
    if len(text) > MAX_EVIDENCE_CHARS:
        text = text[: MAX_EVIDENCE_CHARS - 190] + " [...] " + text[-180:]

    if tool in code_tools:
        status = (payload.get("verification") or {}).get("status")
        file_note = f"file={payload.get('file')}; " if payload.get("file") else ""
        return (
            f"RAW {tool.upper()} RESULT: {file_note}state={payload.get('state')}; status={status}; "
            f"exit_code={(payload.get('verification') or {}).get('exit_code')}\n"
            f"DETAIL: {text}"
        )

    if tool == "run_command":
        data = _parse_report_object(report) or payload
        cmd = data.get("command", "")
        exit_code = data.get("exit_code")
        timed_out = data.get("timed_out")
        stdout = _clip(data.get("stdout", ""), 2000)
        stderr = _clip(data.get("stderr", ""), 2000)
        return (
            f"RAW RUN_COMMAND RESULT: command={cmd!r}; exit_code={exit_code}; timed_out={timed_out}; "
            f"stdout={stdout}; stderr={stderr}"
        )

    if tool == "find_files":
        data = _parse_report_object(report)
        files = _extract_collection_items(data.get("files")) if isinstance(data, dict) else None
        paths = [str(item) for item in files or [] if isinstance(item, str)]
        provenance = _find_files_provenance(payload)
        return (
            "RAW FIND_FILES RESULTS: " + ", ".join(paths[:30])
            + f"; total_matches={provenance.get('total_matches')}; omitted={provenance.get('omitted', 0)}"
        )

    if tool == "find_symbol":
        data = _parse_report_object(report)
        matches = _extract_collection_items(data.get("matches")) if isinstance(data, dict) else None
        return "RAW FIND_SYMBOL RESULTS: " + json.dumps(matches or [], ensure_ascii=False, separators=(",", ":"))

    if tool == "search_repository":
        data = _parse_report_object(report)
        results = _extract_collection_items(data.get("results")) if isinstance(data, dict) else None
        hits = [item for item in (results or []) if isinstance(item, dict)]
        parts = []
        for hit in hits[:8]:
            path = hit.get("file", "?")
            s_start = hit.get("snippet_line_start") or hit.get("line_start")
            s_end = hit.get("snippet_line_end") or hit.get("line_end")
            start, end = hit.get("line_start"), hit.get("line_end")
            if s_start is not None and s_end is not None and (s_start != start or s_end != end):
                loc = f"{path}:{s_start}-{s_end} (chunk {start}-{end})"
            elif start is not None and end is not None:
                loc = f"{path}:{start}-{end}"
            else:
                loc = str(path)
            snippet = _clip(hit.get("snippet", ""), 380)
            parts.append(f"{loc}: {snippet}")
        return "RAW SEARCH_REPOSITORY RESULTS: " + (" | ".join(parts) if parts else "no matches")

    if tool == "find_callers":
        data = _parse_report_object(report)
        sym = data.get("symbol", "?")
        kind = data.get("kind", "")
        callers = _extract_collection_items(data.get("callers")) or []
        possible = _extract_collection_items(data.get("possible_callers")) or []
        summary_obj = {
            "symbol": sym,
            "kind": kind,
            "callers": callers[:30],
            "possible_callers": possible[:20],
            "truncated": bool(data.get("truncated")),
        }
        return _clip(
            "RAW FIND_CALLERS RESULTS: " + json.dumps(summary_obj, ensure_ascii=False, separators=(",", ":")),
            MAX_EVIDENCE_CHARS,
        )

    if tool == "find_callees":
        data = _parse_report_object(report)
        sym = data.get("symbol", "?")
        kind = data.get("kind", "")
        callees = _extract_collection_items(data.get("callees")) or []
        possible = _extract_collection_items(data.get("possible_callees")) or []
        external = _extract_collection_items(data.get("external_calls")) or []
        summary_obj = {
            "symbol": sym,
            "kind": kind,
            "callees": callees[:30],
            "possible_callees": possible[:20],
            "external_calls": external[:20],
        }
        return _clip(
            "RAW FIND_CALLEES RESULTS: " + json.dumps(summary_obj, ensure_ascii=False, separators=(",", ":")),
            MAX_EVIDENCE_CHARS,
        )

    if tool == "find_importers":
        data = _parse_report_object(report)
        importers = _extract_collection_items(data.get("importers")) or []
        summary_obj = {
            "module": data.get("module"),
            "file": data.get("file"),
            "importers": importers[:30],
        }
        return _clip(
            "RAW FIND_IMPORTERS RESULTS: " + json.dumps(summary_obj, ensure_ascii=False, separators=(",", ":")),
            MAX_EVIDENCE_CHARS,
        )

    if tool == "get_dependencies":
        data = _parse_report_object(report)
        summary_obj = {
            "symbol": data.get("symbol"),
            "kind": data.get("kind"),
            "file": data.get("file"),
            "depends_on": data.get("depends_on"),
            "depended_on_by": data.get("depended_on_by"),
        }
        return _clip(
            "RAW GET_DEPENDENCIES RESULTS: " + json.dumps(summary_obj, ensure_ascii=False, separators=(",", ":")),
            MAX_EVIDENCE_CHARS,
        )

    if tool == "get_ast":
        data = _parse_report_object(report)
        summary_obj = {k: v for k, v in data.items() if k != "error"}
        return _clip(
            "RAW GET_AST RESULTS: " + json.dumps(summary_obj, ensure_ascii=False, separators=(",", ":")),
            MAX_EVIDENCE_CHARS,
        )

    return f"RAW TOOL OBSERVATION ({tool}): {text}"


def _parse_report_object(report: str) -> dict:
    try:
        parsed = json.loads(report or "")
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def evidence_provenance(
    tool: str,
    payload: dict,
    code_tools: tuple = ("write_file", "verify_repository"),
    report: str = "",
) -> dict:
    """Structured provenance kept beside the evidence excerpt for every direct tool."""
    data = _parse_report_object(report) if report else {}
    merged_payload = {**payload, **data} if isinstance(payload, dict) else data

    if tool in code_tools:
        verification = merged_payload.get("verification") or {}
        prov: dict = {
            "report_kind": "raw_tool_observation",
            "source_tool": tool,
            "state": merged_payload.get("state"),
            "raw_verification": {
                "status": verification.get("status"),
                "exit_code": verification.get("exit_code"),
                "timed_out": verification.get("timed_out"),
            },
        }
        if isinstance(merged_payload.get("file"), str) and merged_payload["file"]:
            prov["file"] = merged_payload["file"]
            lc = merged_payload.get("line_count")
            if _is_valid_int(lc) and lc >= 1:
                prov["line_start"] = 1
                prov["line_end"] = int(lc)
                prov["total_lines"] = int(lc)
                prov["file_ranges"] = {merged_payload["file"]: [[1, int(lc)]]}
        if merged_payload.get("rollback") is not None:
            prov["rollback"] = merged_payload.get("rollback")
        if isinstance(merged_payload.get("changed_files"), list):
            prov["changed_files"] = [str(f) for f in merged_payload["changed_files"] if isinstance(f, str)]
        return prov

    if tool == "run_command":
        prov = {
            "report_kind": "raw_tool_observation",
            "source_tool": tool,
            "command": merged_payload.get("command"),
            "exit_code": merged_payload.get("exit_code"),
            "timed_out": bool(merged_payload.get("timed_out")),
        }
        if merged_payload.get("state") is not None:
            prov["state"] = merged_payload.get("state")
        if isinstance(merged_payload.get("changed_files"), list):
            prov["changed_files"] = [str(f) for f in merged_payload["changed_files"] if isinstance(f, str)]
        return prov

    if tool == "get_file":
        return {
            "report_kind": "raw_tool_observation",
            "source_tool": tool,
            **get_file_provenance(report or json.dumps(merged_payload)),
        }

    if tool == "get_symbol":
        return {
            "report_kind": "raw_tool_observation",
            "source_tool": tool,
            **_get_symbol_provenance(merged_payload),
        }

    if tool == "find_files":
        return {
            "report_kind": "raw_tool_observation",
            "source_tool": tool,
            **_find_files_provenance(merged_payload),
        }

    if tool == "find_symbol":
        return {
            "report_kind": "raw_tool_observation",
            "source_tool": tool,
            **_find_symbol_provenance(merged_payload),
        }

    if tool == "search_repository":
        return {
            "report_kind": "raw_tool_observation",
            "source_tool": tool,
            **_search_repository_provenance(merged_payload),
        }

    if tool == "find_callers":
        return {
            "report_kind": "raw_tool_observation",
            "source_tool": tool,
            **_find_callers_provenance(merged_payload),
        }

    if tool == "find_callees":
        return {
            "report_kind": "raw_tool_observation",
            "source_tool": tool,
            **_find_callees_provenance(merged_payload),
        }

    if tool == "find_importers":
        return {
            "report_kind": "raw_tool_observation",
            "source_tool": tool,
            **_find_importers_provenance(merged_payload),
        }

    if tool == "get_dependencies":
        return {
            "report_kind": "raw_tool_observation",
            "source_tool": tool,
            **_get_dependencies_provenance(merged_payload),
        }

    if tool == "get_ast":
        return {
            "report_kind": "raw_tool_observation",
            "source_tool": tool,
            **_get_ast_provenance(merged_payload),
        }

    return {"report_kind": "raw_tool_observation", "source_tool": tool}


def _format_step_action(tool: str, arg: str, provenance: dict | None = None) -> str:
    if tool in ("get_file", "get_symbol") and isinstance(provenance, dict):
        path = provenance.get("file")
        start, end = provenance.get("line_start"), provenance.get("line_end")
        if (isinstance(path, str) and path and _is_valid_int(start) and _is_valid_int(end)
                and (tool == "get_symbol" or provenance.get("truncated") or provenance.get("partial_range") or ":" in str(arg))):
            label = f"{arg} ({path}:{start}-{end})" if tool == "get_symbol" else f"{path}:{start}-{end}"
            return f"{tool}: {_clip(label, _ACTION_LABEL_CHARS)}"
    return f"{tool}: {_clip(arg, _ACTION_LABEL_CHARS)}"


def record_step_evidence(
    plan: dict,
    tool: str,
    arg: str,
    result: str,
    evidence: str,
    provenance: dict | None = None,
    step_id: str | None = None,
) -> dict:
    """Attach one tool result to its corresponding plan step (`step_id` or
    `plan['current_step_id']`).

    A missing or already-populated step becomes an explicit ad-hoc step so
    multiple tool results in the same turn never overwrite each other.
    """
    steps = list(plan["steps"])
    target_id = step_id or plan.get("current_step_id")
    current = next((step for step in steps if step["step_id"] == target_id), None)
    action_label = _format_step_action(tool, arg, provenance)
    if current is None or current.get("result"):
        if current is not None and current.get("status") == IN_PROGRESS:
            current_status = COMPLETED if (
                current.get("result") == RESULT_OK
                or (current.get("tool") == "get_file" and current.get("result") == RESULT_INCONCLUSIVE)
            ) else FAILED
            steps = [dict(step, status=current_status) if step["step_id"] == target_id else step
                     for step in steps]
        recorded_sequences = []
        for step in steps:
            match = re.search(r"(\d+)$", step["step_id"])
            if match:
                recorded_sequences.append(int(match.group(1)))
        seq = max([int(plan.get("step_seq", 0)), *recorded_sequences]) + 1
        target_id = f"step_{seq}"
        description = f"Additional result: {action_label}"
        steps.append({"step_id": target_id, "description": description, "status": IN_PROGRESS})
        plan = dict(plan, steps=steps, current_step_id=target_id, step_seq=seq)
    order = 1 + max([s.get("order", 0) for s in plan["steps"]] + [0])
    steps = []
    for step in plan["steps"]:
        if step["step_id"] == target_id:
            step = dict(step, tool=tool, action=action_label,
                        action_key=_action_key(tool, arg), result=result, evidence=evidence,
                        provenance=provenance or {}, order=order)
        steps.append(step)
    return dict(plan, steps=steps)


def evidence_steps(plan: Optional[dict]) -> list[dict]:
    """The plan steps that carry evidence, in execution order."""
    if not plan:
        return []
    return sorted((s for s in plan["steps"] if s.get("result")), key=lambda s: s.get("order", 0))


def is_blind_repeat(plan: Optional[dict], tool: str, arg: str, mutating_tool: str = "write_file") -> bool:
    """True if this exact action already ran WITHOUT usable evidence and
    nothing has changed since."""
    steps = evidence_steps(plan)
    key = _action_key(tool, arg)
    usable_ids = {s["step_id"] for s in usable_evidence_steps(plan)}
    bad = [s["order"] for s in steps if s["action_key"] == key and s["step_id"] not in usable_ids]
    if not bad:
        return False
    return not any(s["order"] > max(bad) and s["tool"] == mutating_tool for s in steps)


# ------------------------------------------------------------------ grading


def evidence_for_grader(plan: Optional[dict], criteria: list[dict] | None = None) -> list[dict]:
    """The evidence steps shown to the grading LLM for this call."""
    steps = evidence_steps(plan)
    usable_ids = {s["step_id"] for s in usable_evidence_steps(plan, criteria)}
    if criteria and all(c["satisfied"] for c in criteria):
        cited_ids = {step_id for c in criteria for step_id in (c.get("evidence_ids") or [])}
        steps = [s for s in steps if s["step_id"] in cited_ids]
    raw_items = [
        {"id": s["step_id"], "action": s["action"],
         "result": RESULT_OK if s["step_id"] in usable_ids else s["result"],
         "evidence": s["evidence"], "provenance": s.get("provenance", {})}
        for s in steps
    ]
    from .context_budget import prepare_model_evidence_items
    prepared, _ = prepare_model_evidence_items(raw_items)
    return prepared


def evidence_for_conclusion(plan: Optional[dict], criteria: Optional[list[dict]] = None) -> list[dict]:
    """Evidence available to check every factual claim in a final answer."""
    steps = usable_evidence_steps(plan, criteria)
    if criteria and all(c.get("satisfied") for c in criteria):
        cited_ids = {step_id for c in criteria for step_id in (c.get("evidence_ids") or [])}
        steps = [step for step in steps if step["step_id"] in cited_ids]
    raw_items = [
        {"id": step["step_id"], "action": step["action"], "result": RESULT_OK,
         "evidence": step["evidence"], "provenance": step.get("provenance", {})}
        for step in steps
    ]
    from .context_budget import prepare_model_evidence_items
    prepared, _ = prepare_model_evidence_items(raw_items)
    return prepared


def parse_assessment(text: str) -> dict[str, dict]:
    """Parse the grader's reply into {criterion_id: verdict}."""
    data = parse_json_object(text or "")
    items = data.get("criteria") if data is not None else None
    if not isinstance(items, list):
        return {}
    result = {}
    for item in items:
        if (not isinstance(item, dict) or not isinstance(item.get("id"), str)
                or not isinstance(item.get("satisfied"), bool)
                or not isinstance(item.get("evidence_ids"), list)
                or any(not isinstance(step_id, str) for step_id in item["evidence_ids"])
                or ("reason" in item and not isinstance(item["reason"], str))
                or item["id"] in result):
            return {}
        result[item["id"]] = item
    return result


def parse_conclusion_assessment(text: str) -> dict | None:
    """Parse per-claim grounding decisions from the conclusion checker."""
    data = parse_json_object(text or "")
    claims = data.get("claims") if data is not None else None
    if not isinstance(claims, list) or not claims:
        return None
    parsed = []
    for item in claims:
        if (not isinstance(item, dict) or not isinstance(item.get("claim"), str)
                or not item["claim"].strip() or not isinstance(item.get("supported"), bool)
                or not isinstance(item.get("evidence_ids"), list)
                or any(not isinstance(step_id, str) for step_id in item["evidence_ids"])
                or ("certainty" in item and item["certainty"] not in ("confirmed", "possible", "unknown"))
                or ("reason" in item and not isinstance(item["reason"], str))):
            return None
        parsed.append(item)
    return {"claims": parsed}


_STEP_CITATION_RE = re.compile(
    r"^\s*\[?(step_\d+)\]?"
    r"(?:\s*(?:[:(\[]\s*(?:([^:\]()]+?):\s*)?(?:lines?\s*)?(\d+)(?:\s*[-–]\s*(\d+))?\s*[\])]?)?)?\s*$",
    re.IGNORECASE,
)


def _matches_cited_file(candidate_path: str, cite_file: str) -> bool:
    cand_norm = candidate_path.strip().replace("\\", "/").lstrip("./").casefold()
    cite_norm = cite_file.strip().replace("\\", "/").lstrip("./").casefold()
    if not cand_norm or not cite_norm:
        return False
    return cand_norm == cite_norm or cand_norm.rsplit("/", 1)[-1] == cite_norm


def _step_covers_line_range(
    prov: dict,
    cite_start: int,
    cite_end: int,
    cite_file: Optional[str] = None,
) -> bool:
    """True only when `cite_start..cite_end` (and optional `cite_file`) lies
    within a step's recorded `line_start..line_end` or `prov.get("file_ranges")`.
    Steps without recorded line ranges (e.g. `find_files`) cannot satisfy
    explicit line-range citations."""
    step_file = prov.get("file")
    step_start, step_end = prov.get("line_start"), prov.get("line_end")
    if _is_valid_int(step_start) and _is_valid_int(step_end):
        file_ok = (
            cite_file is None
            or (isinstance(step_file, str) and _matches_cited_file(step_file, cite_file))
        )
        if file_ok and int(step_start) <= cite_start and cite_end <= int(step_end):
            return True
    file_ranges = prov.get("file_ranges")
    if isinstance(file_ranges, dict) and file_ranges:
        for f_path, intervals in file_ranges.items():
            if cite_file is not None and not (isinstance(f_path, str) and _matches_cited_file(f_path, cite_file)):
                continue
            if not isinstance(intervals, list):
                continue
            for pair in intervals:
                if (isinstance(pair, (list, tuple)) and len(pair) == 2
                        and _is_valid_int(pair[0]) and _is_valid_int(pair[1])
                        and int(pair[0]) <= cite_start and cite_end <= int(pair[1])):
                    return True
    return False


def resolve_evidence_ids(submitted, plan: Optional[dict], usable_ids: set[str]) -> list[str]:
    """Resolve step citations, preserving range-annotated step references when
    their cited file and line range lie within the step's delivered file range."""
    if not isinstance(submitted, list):
        return []
    by_id = {s["step_id"]: s for s in evidence_steps(plan)}
    resolved: list[str] = []
    for raw in submitted:
        if not isinstance(raw, str):
            continue
        text = raw.strip()
        if text in usable_ids:
            if text not in resolved:
                resolved.append(text)
            continue
        match = _STEP_CITATION_RE.match(text)
        if not match:
            continue
        step_id = match.group(1).lower()
        if step_id not in usable_ids:
            continue
        cite_file, start_s, end_s = match.group(2), match.group(3), match.group(4)
        if start_s is not None:
            cite_start = int(start_s)
            cite_end = int(end_s) if end_s is not None else cite_start
            if cite_end < cite_start:
                continue
            step = by_id.get(step_id) or {}
            prov = step.get("provenance") or {}
            if not _step_covers_line_range(prov, cite_start, cite_end, cite_file=cite_file):
                continue
        if step_id not in resolved:
            resolved.append(step_id)
    return resolved


def apply_assessment(criteria: list[dict], plan: dict, verdicts: dict[str, dict]) -> tuple[list[dict], dict]:
    """Validate grader decisions against real, usable evidence records."""
    all_usable_ids = {s["step_id"] for s in usable_evidence_steps(dict(plan, objective=""), None)}
    full_file_usable_ids = {s["step_id"] for s in usable_evidence_steps(plan, criteria)}
    clipped_files = set(_clipped_full_files(plan))
    by_id = {s["step_id"]: s for s in evidence_steps(plan)}
    plan_needs_full = _requires_full_file(None, plan)
    supports: dict[str, list[str]] = {}
    new_criteria = []
    for criterion in criteria:
        verdict = verdicts.get(criterion["id"], {})
        cited = verdict.get("evidence_ids")
        submitted_ids = [i for i in cited if isinstance(i, str)] if isinstance(cited, list) else []
        criterion_needs_full = plan_needs_full or _requires_full_file([criterion], None)
        allowed_ids = full_file_usable_ids if criterion_needs_full else all_usable_ids
        valid = resolve_evidence_ids(submitted_ids, plan, allowed_ids)
        if criterion_needs_full and clipped_files:
            valid = [
                step_id for step_id in valid
                if ((by_id.get(step_id) or {}).get("provenance") or {}).get("file") not in clipped_files
            ]
        grader_approved = verdict.get("satisfied") is True
        satisfied = grader_approved and bool(valid)
        reason = verdict.get("reason") if isinstance(verdict.get("reason"), str) else ""
        if grader_approved and not valid:
            if criterion_needs_full and resolve_evidence_ids(submitted_ids, plan, all_usable_ids):
                reason = ("citation check rejected the grader's verdict: cited get_file evidence does not "
                          "cover the complete file or its unclipped contents")
            else:
                reason = ("citation check rejected the grader's verdict: it cited no real usable evidence item"
                          if not submitted_ids else
                          "citation check rejected the grader's verdict: cited IDs do not identify successful, usable evidence")
        elif not grader_approved and not reason:
            reason = "the grader did not approve this criterion"
        new_criteria.append(dict(
            criterion,
            satisfied=satisfied,
            evidence_ids=valid if satisfied else [],
            grader_satisfied=grader_approved,
            grader_evidence_ids=submitted_ids,
            validation="accepted" if satisfied else "citation_rejected" if grader_approved else "grader_rejected",
            note=_clip(reason, _NOTE_CHARS),
        ))
        if satisfied:
            for step_id in valid:
                supports.setdefault(step_id, []).append(criterion["id"])
    steps = [dict(s, criteria=supports.get(s["step_id"], [])) if s.get("result") else s for s in plan["steps"]]
    return new_criteria, dict(plan, steps=steps)


def classify_support(criteria: list[dict], plan: Optional[dict], failure_info: Optional[dict],
                     verification_info: Optional[dict]) -> str:
    """What the gathered evidence supports - never stronger than that."""
    steps = evidence_steps(plan)
    unavailable = bool(verification_info and (verification_info.get("status") == "unavailable"
                                              or verification_info.get("state") == "UNAVAILABLE"))
    if not unavailable:
        if failure_info is not None:
            return BLOCKED
        if steps and not usable_evidence_steps(plan) \
                and any(s["result"] == RESULT_FAILED for s in steps):
            return BLOCKED
    if criteria and all(
            not criterion.get("satisfied")
            and str(criterion.get("note", "")).startswith(GRADER_FAILED_NOTE)
            for criterion in criteria):
        return GRADING_FAILED
    satisfied = [c for c in criteria if c["satisfied"]]
    if not satisfied:
        return UNSUPPORTED
    if len(satisfied) < len(criteria) or pending_steps(plan) or unavailable:
        return PARTIAL
    if incomplete_reasons(criteria, plan) or not criteria_established(criteria, plan):
        return PARTIAL
    return SUPPORTED


def criteria_established(criteria: list[dict], plan: Optional[dict]) -> bool:
    """Whether every criterion is satisfied AND each is backed by at least one
    currently usable evidence step."""
    if not criteria or not all(c.get("satisfied") for c in criteria):
        return False
    if not plan:
        return False
    usable_ids = {s["step_id"] for s in usable_evidence_steps(dict(plan, objective=""), None)}
    return all(c.get("evidence_ids") and set(c["evidence_ids"]) <= usable_ids for c in criteria)


def can_drop_pending_readonly_steps(
    support: str,
    criteria: list[dict],
    plan: Optional[dict],
    *,
    light_readonly: bool = True,
    failure_info: Optional[dict] = None,
    verification_info: Optional[dict] = None,
    intent: Optional[dict] = None,
    capabilities: Optional[dict] = None,
) -> bool:
    """Whether satisfied evidence makes leftover read-only plan steps superseded.

    When every requested criterion is established by real, gate-accepted
    evidence, there are no incomplete reasons, no unresolved failure invalidates
    the result, and every remaining pending step is a read-only investigation
    step (never an unexecuted required mutation or verification), the leftover
    read-only steps are superseded and may be resolved as skipped rather than
    falsely downgrading a complete answer to PARTIAL.
    """
    if support != PARTIAL or not criteria or not all(c.get("satisfied") for c in criteria):
        return False
    if failure_info is not None:
        return False
    if verification_info and (
        verification_info.get("status") == "unavailable"
        or verification_info.get("state") == "UNAVAILABLE"
    ):
        return False
    pending = pending_steps(plan)
    if not pending:
        return False
    if not criteria_established(criteria, plan) or incomplete_reasons(criteria, plan):
        return False
    if not all(is_readonly_step_description(s.get("description", "")) for s in pending):
        return False
    if intent and intent.get("kind") == "change":
        has_verified_write = any(
            s.get("tool") == "write_file" and s.get("result") == RESULT_OK
            for s in evidence_steps(plan)
        )
        if not has_verified_write:
            return False
    if capabilities and capabilities.get("execute") and not capabilities.get("write"):
        has_executed_check = any(
            s.get("tool") in ("verify_repository", "run_command") and s.get("result") == RESULT_OK
            for s in evidence_steps(plan)
        )
        if not has_executed_check:
            return False
    return True


# ------------------------------------------------------------- text for LLM


def describe_criteria(criteria: list[dict], plan: Optional[dict]) -> str:
    """Digest shown to `decide`: what must be shown, whether the last grading
    supported it, and which attempts gave no usable evidence so they are not
    repeated unchanged."""
    if not criteria:
        return ""
    lines = ["Acceptance criteria (your final answer is checked against recorded evidence):"]
    for c in criteria:
        mark = "supported" if c["satisfied"] else "not yet supported"
        lines.append(f"  - [{c['id']}] {c['description']} -- {mark}"
                     + (f" (needs: {c['evidence_needed']})" if c["evidence_needed"] and not c["satisfied"] else ""))
    steps = evidence_steps(plan)
    usable_ids = {step["step_id"] for step in usable_evidence_steps(plan, criteria)}
    if steps:
        lines.append("Evidence recorded so far:")
        for s in steps:
            prov = s.get("provenance") or {}
            next_start = prov.get("next_start_line")
            if s["step_id"] in usable_ids:
                if s.get("tool") == "get_file" and prov.get("truncated") and next_start:
                    flag = f" -- bounded read; continue with get_file start_line={next_start} if needed"
                elif s.get("tool") == "get_symbol":
                    flag = " -- symbol source already recorded; do NOT re-read these lines with get_file"
                elif s.get("tool") == "search_repository":
                    fr = prov.get("file_ranges") or {}
                    locs = [
                        f"{fp}:{rng[0][0]}-{rng[-1][1]}"
                        for fp, rng in list(fr.items())[:4]
                        if isinstance(rng, list) and rng and isinstance(rng[0], list) and len(rng[0]) == 2
                    ]
                    flag = (f" -- matched in {', '.join(locs)}; read unread relevant lines with get_file" if locs else "")
                else:
                    flag = ""
            elif s.get("tool") == "get_file" and prov.get("truncated") and next_start:
                flag = f" -- partial read only; continue with get_file start_line={next_start}"
            elif s.get("tool") == "find_files" and s.get("result") == RESULT_EMPTY:
                flag = (
                    " -- 0 matching file paths (find_files matches file paths/names only, never file contents, "
                    "metric values, or symbols; do NOT retry find_files with shortened or variant keyword patterns "
                    "-- switch strategy to search_repository for content/metric text, find_symbol for symbol "
                    "definitions, or get_file on documentation such as README.md)"
                )
            elif s.get("tool") == "search_repository" and s.get("result") == RESULT_EMPTY:
                flag = (
                    " -- 0 matching content chunks; do not repeat unchanged -- try different search keywords, "
                    "find_symbol/get_symbol for symbol definitions, or get_file on a relevant file"
                )
            elif s.get("tool") in ("find_symbol", "get_symbol") and s.get("result") in (RESULT_EMPTY, RESULT_FAILED):
                flag = (
                    " -- no matching symbol definition; do not repeat unchanged -- switch strategy to "
                    "search_repository for content/text, find_files for module paths, or get_file"
                )
            elif s.get("tool") in ("find_callers", "find_callees", "find_importers") and s.get("result") == RESULT_EMPTY:
                flag = (
                    " -- 0 graph edges returned; inspect the symbol/module source with get_symbol or get_file, "
                    "or check get_dependencies / get_ast"
                )
            else:
                flag = " -- no usable evidence; change strategy rather than repeating a similar action"
            lines.append(f"  - {s['step_id']} {s['action']} -> {s['result']}{flag}")
    gaps = _read_gaps(plan)
    if gaps and any(not c.get("satisfied") for c in criteria):
        lines.append("Unread file ranges:")
        for gap in gaps:
            lines.append(f"  - {gap}")
    return "\n".join(lines)


def completion_feedback(support: str, criteria: list[dict], plan: Optional[dict]) -> str:
    """Message that sends `decide` back to work instead of accepting a final
    answer the evidence does not support."""
    lines = [f"Completion check (system data): your answer was not accepted yet - evidence support is {support}."]
    unmet = [c for c in criteria if not c["satisfied"]]
    grader_down = [c for c in unmet if c["note"].startswith(GRADER_FAILED_NOTE)]
    really_unmet = [c for c in unmet if c not in grader_down]
    for c in really_unmet:
        line = f"  - [{c['id']}] not supported: {c['description']}"
        if c["note"]:
            label = "citation check" if c.get("validation") == "citation_rejected" else "grader"
            line += f" ({label}: {c['note']})"
        if c["evidence_needed"]:
            line += f" -- evidence that would show it: {c['evidence_needed']}"
        lines.append(line)
    if grader_down:
        lines.append("  - The grading call itself failed for: " + ", ".join(f"[{c['id']}]" for c in grader_down)
                     + ". That says nothing about your evidence.")
    left = pending_steps(plan)
    if left:
        lines.append("  - Planned steps never started: " + "; ".join(_clip(s["description"], 80) for s in left)
                     + ". Untouched planned steps keep the result PARTIAL until they are done or dropped.")
    done = usable_evidence_steps(plan, criteria)
    if done:
        lines.append("Evidence already recorded with usable results (do not re-investigate what these cover): "
                     + "; ".join(f"{s['step_id']} {s['action']}" for s in done[-6:]))
    if really_unmet:
        lines.append(
            "Gather ONLY the evidence for the unmet criteria above using targeted repository tool calls "
            "(such as get_symbol, get_file, find_callers, find_callees, get_dependencies, get_ast, or "
            "search_repository). Do not repeat an attempt that gave no usable evidence or reread covered "
            "file ranges unless the repository changed. If it truly cannot be obtained, give a final answer "
            "that states plainly what could not be established."
            + (" If a planned step is no longer needed, call revise_plan with only the steps you still "
               "intend to do." if left else "")
        )
    elif grader_down:
        lines.append("The grading call failed; it did not reject the evidence. Do not repeat existing research "
                     "solely because of this error. Give a grounded answer so grading can retry, or gather only "
                     "new evidence that is independently useful.")
    elif left:
        lines.append("Every criterion is supported by recorded evidence; the only thing blocking acceptance is the "
                     "planned steps above. Do NOT repeat investigation. Drop the steps you no longer need with "
                     "revise_plan (only the steps you genuinely still need), then give your final answer.")
    reasons = incomplete_reasons(criteria, plan)
    for reason in reasons:
        lines.append(f"  - Not complete: {reason}.")
    if really_unmet:
        for gap in _read_gaps(plan):
            if gap not in reasons:
                lines.append(f"  - Unread file range: {gap}.")
    return "\n".join(lines)


def completion_retry_allowed(step_count: int, step_budget: int, retries: int, *,
                             progress_generation: int = 0,
                             last_retry_generation: Optional[int] = None) -> bool:
    """Whether another tools-enabled decide turn remains for evidence recovery."""
    if retries >= MAX_COMPLETION_RETRIES or step_count + 1 >= step_budget:
        return False
    if retries == 0:
        return True
    return last_retry_generation is None or progress_generation > last_retry_generation


_STEP_REF_RE = re.compile(r"\bstep_\d+\b")
_FILE_REF_RE = re.compile(r"[\w./\\-]+\.(?:py|md|txt|json|ya?ml|toml|ini|cfg|sh)\b")
_FILE_RANGE_REF_RE = re.compile(
    r"([\w./\\-]+\.(?:py|md|txt|json|ya?ml|toml|ini|cfg|sh))\b"
    r"(?:"
    r":(\d+)(?:-(\d+))?"
    r"|\s*(?:,\s*|\(\s*|\[\s*)lines?\s+(\d+)(?:\s*[-–]\s*(\d+))?"
    r"|[†#](?:L|lines?\s*)?(\d+)(?:\s*[-–]\s*L?(\d+))?"
    r")",
    re.IGNORECASE,
)
_DAGGER_BRACKET_RE = re.compile(r"【([^】†]+?)†([^】]+)】")
_EXCERPT_CHARS = 520


def _focused_excerpt(evidence_text: str, objective: str = "", limit: int = _EXCERPT_CHARS) -> str:
    """Extract a window of `evidence_text` centered on lines that match `objective`
    terms so long multi-line file reads show the actual matching lines rather
    than only the opening header lines."""
    raw = str(evidence_text or "").strip()
    if not raw:
        return ""
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if len(lines) <= 2 or not objective:
        return _clip(raw, limit)
    from ..retrieval import query_tokens, tokenize
    q_terms = set(query_tokens(objective))
    if not q_terms:
        return _clip(raw, limit)
    header = lines[0] if lines[0].endswith(":") or lines[0].startswith("RAW ") else ""
    body_lines = lines[1:] if header else lines
    if not body_lines:
        return _clip(raw, limit)
    scores = [len(q_terms & set(tokenize(ln))) for ln in body_lines]
    if max(scores, default=0) == 0:
        return _clip(raw, limit)
    best_idx = scores.index(max(scores))
    start_idx = max(0, best_idx - 1)
    chosen = body_lines[start_idx : start_idx + 12]
    joined = (f"{header} " if header else "") + " ".join(chosen)
    return _clip(joined, limit)


def _sanitize_conclusion(text: str, plan: Optional[dict], corpus: str) -> tuple[str, list[str]]:
    """Keep the model's prose from citing things that were never recorded in usable evidence."""
    real = {s["step_id"] for s in usable_evidence_steps(plan)}
    text = _STEP_REF_RE.sub(lambda m: m.group(0) if m.group(0) in real else "(unrecorded reference removed)", text)
    haystack = corpus.lower().replace("\\", "/")
    unverified = sorted({f for f in _FILE_REF_RE.findall(text)
                         if f.lower().replace("\\", "/").lstrip("./") not in haystack})
    return text, unverified


def _recorded_file_ranges(plan: Optional[dict]) -> dict[str, list[tuple[int, int]]]:
    """Return merged recorded line ranges per file across usable direct read-only,
    structural, and mutation tools at the latest repository epoch for each file."""
    covered, _, epochs = _file_coverage(plan)
    combined: dict[str, list[tuple[int, int]]] = {
        path: list(ranges) for path, ranges in covered.items()
    }
    usable_steps = usable_evidence_steps(plan)
    latest_epoch: dict[str, tuple[object, int]] = {}
    for step in usable_steps:
        if step.get("result") not in (RESULT_OK, RESULT_INCONCLUSIVE):
            continue
        prov = step.get("provenance") or {}
        epoch = prov.get("repository_epoch")
        order = int(step.get("order", 0))
        file_ranges = prov.get("file_ranges")
        if isinstance(file_ranges, dict):
            for path in file_ranges:
                if path not in latest_epoch or order >= latest_epoch[path][1]:
                    latest_epoch[path] = (epoch, order)
        direct_path = prov.get("file")
        if isinstance(direct_path, str) and direct_path.strip():
            p_str = direct_path.strip()
            if p_str not in latest_epoch or order >= latest_epoch[p_str][1]:
                latest_epoch[p_str] = (epoch, order)

    for step in usable_steps:
        if step.get("result") not in (RESULT_OK, RESULT_INCONCLUSIVE):
            continue
        prov = step.get("provenance") or {}
        epoch = prov.get("repository_epoch")
        file_ranges = prov.get("file_ranges")
        if isinstance(file_ranges, dict):
            for path, intervals in file_ranges.items():
                target_epoch = epochs.get(path, (latest_epoch.get(path) or (epoch, 0))[0])
                if epoch != target_epoch or not isinstance(intervals, list):
                    continue
                for pair in intervals:
                    if (isinstance(pair, (list, tuple)) and len(pair) == 2
                            and _is_valid_int(pair[0]) and _is_valid_int(pair[1]) and 1 <= pair[0] <= pair[1]):
                        combined.setdefault(path, []).append((int(pair[0]), int(pair[1])))
        direct_path = prov.get("file")
        start, end = prov.get("line_start"), prov.get("line_end")
        if (isinstance(direct_path, str) and direct_path.strip()
                and _is_valid_int(start) and _is_valid_int(end) and 1 <= start <= end):
            p_str = direct_path.strip()
            target_epoch = epochs.get(p_str, (latest_epoch.get(p_str) or (epoch, 0))[0])
            if epoch == target_epoch:
                combined.setdefault(p_str, []).append((int(start), int(end)))

    return {path: _merge_line_ranges(ranges) for path, ranges in combined.items() if ranges}


def _unread_file_range_issues(text: str, plan: Optional[dict]) -> list[str]:
    """Detect explicit file line-range citations that fall outside merged recorded evidence ranges."""
    if not plan or not text or not evidence_steps(plan):
        return []
    merged_by_path = _recorded_file_ranges(plan)
    by_norm = {path.lower().replace("\\", "/").lstrip("./"): (path, ranges)
               for path, ranges in merged_by_path.items()}
    issues: list[str] = []
    for match in _FILE_RANGE_REF_RE.finditer(text):
        raw_path = match.group(1)
        norm_path = raw_path.lower().replace("\\", "/").lstrip("./")
        entry = by_norm.get(norm_path)
        if entry is None:
            basename_matches = [val for key, val in by_norm.items() if key.rsplit("/", 1)[-1] == norm_path]
            if len(basename_matches) == 1:
                entry = basename_matches[0]
        start_s = match.group(2) or match.group(4) or match.group(6)
        end_s = match.group(3) or match.group(5) or match.group(7) or start_s
        if not start_s:
            continue
        cite_start, cite_end = int(start_s), int(end_s)
        if entry is None:
            issues.append(
                f"the draft cited unread lines {cite_start}-{cite_end} of {raw_path} "
                "(no recorded read ranges for this file)"
            )
            continue
        canonical_path, merged_ranges = entry
        if cite_end < cite_start:
            issues.append(f"the draft cited an invalid line range {cite_start}-{cite_end} for {canonical_path}")
            continue
        if not any(r_start <= cite_start and cite_end <= r_end for r_start, r_end in merged_ranges):
            read_desc = ", ".join(f"{a}-{b}" for a, b in merged_ranges)
            issues.append(
                f"the draft cited unread lines {cite_start}-{cite_end} of {canonical_path} "
                f"(recorded read ranges: {read_desc})"
            )
    for dag_match in _DAGGER_BRACKET_RE.finditer(text):
        raw_file = dag_match.group(1).strip()
        raw_range = dag_match.group(2).strip()
        valid_file = bool(re.fullmatch(r"[\w./\\-]+\.(?:py|md|txt|json|ya?ml|toml|ini|cfg|sh)", raw_file, re.IGNORECASE))
        valid_range = bool(re.fullmatch(r"(?:L|lines?\s*)?\d+(?:\s*[-–]\s*L?\d+)?", raw_range, re.IGNORECASE))
        if not (valid_file and valid_range):
            issues.append(f"the draft contained a malformed citation {dag_match.group(0)}")
    return issues


def conclusion_reference_issues(text: str, plan: Optional[dict], corpus: str) -> list[str]:
    """Return deterministic reference problems in a proposed conclusion."""
    sanitized, unverified_files = _sanitize_conclusion(text, plan, corpus)
    issues = []
    if "(unrecorded reference removed)" in sanitized:
        issues.append("the draft cited a step that was not recorded")
    if unverified_files:
        issues.append("the draft named files absent from recorded evidence: " + ", ".join(unverified_files))
    issues.extend(_unread_file_range_issues(text, plan))
    return issues


def _answer_claim_groups(answer_check: Optional[dict], plan: Optional[dict], corpus: str) -> dict[str, list[dict]]:
    """Return only claims whose labels and citations pass local safeguards."""
    groups: dict[str, list[dict]] = {"confirmed": [], "possible": [], "unresolved": []}
    if not answer_check:
        return groups
    items = answer_check.get("claims")
    if not isinstance(items, list):
        return groups
    usable_ids = {step["step_id"] for step in usable_evidence_steps(plan)}
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("claim"), str):
            continue
        claim = item["claim"].strip()
        if not claim:
            continue
        submitted = item.get("evidence_ids")
        cited = submitted if isinstance(submitted, list) and all(isinstance(i, str) for i in submitted) else []
        resolved_ids = resolve_evidence_ids(cited, plan, usable_ids)
        citations_valid = bool(cited) and len(resolved_ids) == len(set(cited)) and bool(resolved_ids)
        item = dict(item, evidence_ids=resolved_ids if citations_valid else cited)
        issues = conclusion_reference_issues(claim, plan, corpus)
        certainty = item.get("certainty")
        if certainty not in ("confirmed", "possible", "unknown"):
            certainty = "confirmed" if item.get("supported") is True else "unknown"
        if issues:
            item = dict(item, certainty="unknown", supported=False,
                        validation_source="deterministic_reference_check",
                        reason="; ".join(issues))
            certainty = "unknown"
        if certainty == "confirmed" and item.get("supported") is True and citations_valid:
            groups["confirmed"].append(item)
        elif certainty == "possible" and item.get("supported") is False and citations_valid:
            groups["possible"].append(item)
        else:
            reason = item.get("reason") or "no usable evidence citation established this claim"
            groups["unresolved"].append(dict(item, certainty="unknown", reason=reason))
    return groups


def _supporting_lines(criteria: list[dict], plan: Optional[dict],
                      answer_check: Optional[dict] = None, corpus: str = "") -> list[str]:
    """Show accepted criterion evidence and direct source reads used to ground the answer."""
    by_id = {s["step_id"]: s for s in evidence_steps(plan)}
    objective = str((plan or {}).get("objective") or "") if isinstance(plan, dict) else ""
    lines = []
    seen = set()
    ordered_ids = [step_id for c in criteria if c["satisfied"] for step_id in c["evidence_ids"]]
    claims = _answer_claim_groups(answer_check, plan, corpus)
    for claim in claims["confirmed"]:
        ordered_ids.extend(claim.get("evidence_ids", []))
    if not claims["confirmed"] and answer_check and answer_check.get("supported") is True:
        ordered_ids.extend(answer_check.get("evidence_ids", []))
    usable = usable_evidence_steps(plan, criteria)
    usable_ids = {step["step_id"] for step in usable}
    # Ensure direct get_file / get_symbol reads are included alongside search_repository hits
    if ordered_ids:
        for step in usable:
            if step.get("tool") in ("get_file", "get_symbol") and step["step_id"] not in ordered_ids:
                ordered_ids.append(step["step_id"])
    for step_id in ordered_ids:
        step = by_id.get(step_id)
        if step is None or step_id in seen or step_id not in usable_ids:
            continue
        seen.add(step_id)
        excerpt = _focused_excerpt(step.get("evidence", ""), objective, _EXCERPT_CHARS)
        lines.append(f"- [{step_id}] {step['action']}: {excerpt}")
    return lines


def _reviewed_lines(criteria: list[dict], plan: Optional[dict], answer_check: Optional[dict] = None,
                    corpus: str = "") -> list[str]:
    """Show usable reads not accepted as criterion or conclusion support."""
    accepted = {step_id for criterion in criteria if criterion.get("satisfied")
                for step_id in criterion.get("evidence_ids", [])}
    claims = _answer_claim_groups(answer_check, plan, corpus)
    accepted.update(step_id for certainty in ("confirmed", "possible")
                    for claim in claims[certainty] for step_id in claim.get("evidence_ids", []))
    return [
        f"- [{step['step_id']}] {step['action']}: {_clip(step['evidence'], _EXCERPT_CHARS)}"
        for step in usable_evidence_steps(plan, criteria)
        if step["step_id"] not in accepted
    ]


def _unresolved_lines(support: str, criteria: list[dict], plan: Optional[dict], failure_info: Optional[dict],
                      verification_info: Optional[dict]) -> list[str]:
    lines = [f"- [{c['id']}] Not established: {c['description']}"
             + (f" (grading call failed: {c['note'][len(GRADER_FAILED_NOTE):].strip(' ()')})"
                if c["note"].startswith(GRADER_FAILED_NOTE) else
                f" (citation check: {c['note']})" if c.get("validation") == "citation_rejected" else
                f" (grader: {c['note']})" if c["note"] else "")
             for c in criteria if not c["satisfied"]]
    left = pending_steps(plan)
    if support != SUPPORTED and left:
        if criteria and all(c.get("satisfied") for c in criteria):
            lines.append(
                "- Planned steps not attempted (pending investigation work; all requested criteria are supported): "
                + "; ".join(_clip(s["description"], 80) for s in left)
            )
        else:
            lines.append("- Planned steps not attempted: " + "; ".join(_clip(s["description"], 80) for s in left))
    if verification_info and (verification_info.get("status") == "unavailable"
                              or verification_info.get("state") == "UNAVAILABLE"):
        lines.append("- Verification was unavailable, so the change is not verified.")
    if support != SUPPORTED:
        usable_ids = {s["step_id"] for s in usable_evidence_steps(plan, criteria)}
        unusable = [s for s in evidence_steps(plan) if s["step_id"] not in usable_ids]
        for s in unusable[-3:]:
            lines.append(f"- Attempt without usable evidence: {s['step_id']} ({s['action']}) -> {s['result']}")
    if failure_info is not None:
        lines.append(f"- The latest {failure_info.get('tool', 'modification/verification')} run failed.")
    if support != SUPPORTED:
        lines.extend(f"- Incomplete: {r}" for r in incomplete_reasons(criteria, plan))
    return lines


_SOURCE_EXCERPT_HEADER_RE = re.compile(
    r"^(?P<path>.+?) \[lines (?P<start>\d+)-(?P<end>\d+)[^\]]*\]:$"
)
_SYMBOL_EXCERPT_HEADER_RE = re.compile(
    r"^RAW GET_SYMBOL RESULT: .*? in (?P<path>.+?) \[lines (?P<start>\d+)-(?P<end>\d+)[^\]]*\]:$"
)
_RESULT_NUMBER_RE = re.compile(
    r"(?<![A-Za-z_])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
    r"\s?(?:%|ms|seconds?|minutes?|hours?|GB|MB|KB)?\b",
    re.IGNORECASE,
)
_NUMERIC_RESULT_REQUEST_RE = re.compile(
    r"\b(?:numeric(?:al)?\s+(?:results?|values?|scores?|metrics?)|numbers?|values?|scores?|"
    r"results?|how\s+many|counts?)\b",
    re.IGNORECASE,
)
_RESULT_SECTION_RE = re.compile(
    r"\b(?:results?|metrics?|scores?|performance|benchmarks?|outcomes?|evaluation)\b",
    re.IGNORECASE,
)
_RESULT_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s*(?P<title>[^#].*?)\s*#*\s*$")
_PLAIN_RESULT_HEADING_RE = re.compile(
    r"^\s*(?:results?|metrics?|scores?|performance|benchmarks?|outcomes?|evaluation(?:\s+results?)?)\s*:?\s*$",
    re.IGNORECASE,
)
_CALCULATION_REQUEST_RE = re.compile(
    r"\b(?:calculat\w*|comput\w*|formula|equation|derive\w*)\b", re.IGNORECASE
)
_NUMERIC_QUERY_FILLER = {
    "what", "which", "were", "was", "is", "are", "the", "does", "do", "did", "on", "in",
    "of", "to", "for", "from", "and", "or", "give", "show", "tell", "provide", "list",
    "me", "please", "number", "numbers", "numeric", "numerical", "result", "results", "score",
    "scores", "value", "values", "metric", "metrics", "reported", "report", "obtained", "obtain",
    "readme", "not", "how", "calculate", "calculated", "calculating", "calculation",
}
_DOCUMENT_SUFFIXES = (".md", ".markdown", ".rst", ".adoc", ".txt")
_EVALUATION_CONFIG_LINE_RE = re.compile(
    r"\b(?:dataset|split|subset|sample|validation|evaluation|test|dataloader|"
    r"data[_ ]?dir|file|path|annotation|question|image)\b",
    re.IGNORECASE,
)


def _source_excerpt(evidence: str) -> tuple[str, int, list[str]] | None:
    """Parse the stable path/range header emitted for direct source reads."""
    lines = evidence.splitlines()
    if not lines:
        return None
    match = _SOURCE_EXCERPT_HEADER_RE.match(lines[0]) or _SYMBOL_EXCERPT_HEADER_RE.match(lines[0])
    if not match:
        return None
    return match.group("path"), int(match.group("start")), lines[1:]


def _numeric_result_requested(task: str) -> bool:
    if _NUMERIC_RESULT_REQUEST_RE.search(task):
        return True
    asks_for_value = re.search(r"\b(?:what|which|give|show|tell|provide|list)\b", task, re.I)
    asks_for_metric = re.search(
        r"\b(?:accuracy|loss|version|rate|precision|recall|f1|metric|count|size)\b", task, re.I
    )
    return bool(asks_for_value and asks_for_metric and not _CALCULATION_REQUEST_RE.search(task))


def _numeric_topic_terms(task: str) -> set[str]:
    words = re.findall(r"[a-z][a-z0-9_]*", task.casefold())
    return {
        part
        for word in words
        for part in word.split("_")
        if part not in _NUMERIC_QUERY_FILLER and len(part) > 1
    }


def _is_reported_number_line(line: str) -> bool:
    match = _RESULT_NUMBER_RE.search(line)
    if not match or not re.search(r"[A-Za-z]", line):
        return False
    if not any(mark in line for mark in ("|", ":", "=", "%")):
        return False
    if "|" in line:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) > 1 and not any(_RESULT_NUMBER_RE.search(cell) for cell in cells[1:]):
            return False
    else:
        separator = re.search(r"[:=]", line)
        if separator and match.start() < separator.start() and not _RESULT_NUMBER_RE.search(line[separator.end():]):
            return False
    if "|" not in line and re.search(r"(?:\+=|-=|\*=|/=|==|!=|<=|>=|\breturn\b)", line):
        return False
    return not re.match(r"^\s*(?:def|class|if|elif|else|for|while|return|import|from)\b", line)


def _has_results_section_context(lines: list[str], index: int) -> bool:
    for line in reversed(lines[:index]):
        stripped = line.strip()
        if not stripped:
            continue
        heading = _RESULT_HEADING_RE.match(line)
        if heading:
            return bool(_RESULT_SECTION_RE.search(heading.group("title")))
        if _PLAIN_RESULT_HEADING_RE.match(line):
            return True
        if re.fullmatch(r"(?:[-*_]\s*){3,}", stripped):
            return False
    return False


def _reported_numeric_values(task: str, excerpts: list[tuple[str, int, list[str]]]) -> list[tuple[str, int, str]]:
    topic = _numeric_topic_terms(task)
    values: list[tuple[str, int, str]] = []
    seen = set()
    for path, first, lines in excerpts:
        for index, line in enumerate(lines):
            if not _is_reported_number_line(line):
                continue
            line_words = set(re.findall(r"[a-z][a-z0-9_]*", line.casefold()))
            for word in tuple(line_words):
                line_words.update(word.split("_"))
            if topic:
                if not (topic & line_words):
                    continue
            else:
                if not _has_results_section_context(lines, index):
                    continue
            item = (path, first + index, line.strip())
            if item not in seen:
                seen.add(item)
                values.append(item)
    return values


def _purpose_document_paragraphs(
    excerpts: list[tuple[str, int, list[str]]]
) -> list[tuple[str, int, int, str]]:
    paragraphs = []
    for path, first, lines in excerpts:
        if not path.casefold().endswith(_DOCUMENT_SUFFIXES):
            continue
        current: list[tuple[int, str]] = []

        def flush() -> None:
            if not current:
                return
            text = " ".join(line.strip().strip("*_") for _, line in current).strip()
            if len(text) >= 45 and not text.startswith(("---", "===", "|", "```")):
                paragraphs.append((path, current[0][0], current[-1][0], text))
            current.clear()

        for index, line in enumerate(lines):
            number = first + index
            stripped = line.strip()
            if (not stripped or stripped.startswith(("#", "|", "```", "- ", "* ", ">"))
                    or re.match(r"^[=\-_*`]{3,}$", stripped)):
                flush()
                continue
            current.append((number, stripped))
        flush()
    return paragraphs


def _evaluation_configuration_lines(
    excerpts: list[tuple[str, int, list[str]]]
) -> list[tuple[str, int, str]]:
    found = []
    seen = set()
    for path, first, lines in excerpts:
        for index, line in enumerate(lines):
            stripped = line.strip()
            if not stripped or not _EVALUATION_CONFIG_LINE_RE.search(stripped):
                continue
            item = (path, first + index, stripped)
            if item not in seen:
                seen.add(item)
                found.append(item)
    return found


def synthesize_deterministic_answer(task: str, plan: Optional[dict] = None, observations: Any = None) -> str:
    """Deterministically synthesize a structured, readable Markdown answer from recorded evidence.
    Ensures clean conclusions without raw tool dumps or validation failures when the model emits empty text.
    """
    usable = []
    if plan and isinstance(plan, dict) and "steps" in plan:
        try:
            usable = usable_evidence_steps(plan)
        except Exception:
            usable = [s for s in plan.get("steps", []) if isinstance(s, dict) and s.get("result")]
    if not usable and observations:
        if isinstance(observations, list):
            usable = [{"action": "Recorded observation", "evidence": str(o), "tool": "observation"} for o in observations[-6:]]
        elif isinstance(observations, str) and observations.strip():
            usable = [{"action": "Recorded observation", "evidence": observations, "tool": "observation"}]

    is_caller_query = bool(re.search(r"\b(who\s+calls|callers?|calls|usage|where\s+is\s+.*called)\b", task, re.I))
    is_purpose_query = bool(re.search(r"\b(purpose|overview|about|what\s+is\s+this\s+repo|architecture)\b", task, re.I))

    stops = {"who", "calls", "function", "method", "class", "symbol", "where", "what", "how", "is", "the", "in", "of", "to", "for", "called", "repo", "repository", "this", "does", "do", "any"}
    quoted_symbols = re.findall(r"`([^`]+)`", task)
    candidates = [w for w in re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", task) if w.lower() not in stops and len(w) > 2]
    target_sym = re.sub(r"\(.*$", "", quoted_symbols[-1]).strip() if quoted_symbols else None
    if not target_sym:
        target_sym = candidates[0] if candidates else None

    if not target_sym and usable:
        for s in usable:
            act = s.get("action", "")
            if any(t in act for t in ("find_callers", "get_symbol", "search_repository", "find_symbol")):
                parts = act.split(":", 1)
                if len(parts) > 1:
                    clean_target = re.sub(r"['\"()]", "", parts[1]).strip()
                    if clean_target and clean_target.isidentifier():
                        target_sym = clean_target
                        break

    all_ev_texts = []
    callers = []
    possible_callers = []
    candidate_callers = []
    caller_search_complete = False
    caller_search_seen = False
    defs = []
    imports = []
    symbol_defs = []
    overview_paragraphs = []
    feature_bullets = []

    for step in usable:
        tool = step.get("tool")
        ev_text = str(step.get("evidence") or "").strip()
        if not ev_text:
            continue
        all_ev_texts.append(ev_text)

        if tool == "find_callers":
            caller_search_seen = True
            sym_act = str(step.get("action", "")).replace("find_callers", "").strip(" ('\":)")
            if sym_act:
                target_sym = target_sym or sym_act
            for line in ev_text.splitlines():
                if "{" in line and "}" in line:
                    try:
                        data = json.loads(line[line.index("{"):line.rindex("}")+1])
                        caller_search_complete = not bool(data.get("truncated"))
                        for c in data.get("callers", []):
                            callers.append((c.get("file", "?"), str(c.get("line", "?")), str(c.get("symbol") or c.get("caller") or "direct caller")))
                        for c in data.get("possible_callers", []):
                            possible_callers.append((c.get("file", "?"), str(c.get("line", "?")), str(c.get("symbol") or c.get("caller") or "possible caller")))
                    except Exception:
                        pass

        elif tool == "get_symbol":
            sym_lines = [ln for ln in ev_text.splitlines() if not ln.startswith("RAW ") and not ln.startswith("get_symbol:")]
            code_snippet = "\n".join(sym_lines[:25]).strip()
            if code_snippet:
                symbol_defs.append(code_snippet)

        clean = re.sub(r"^RAW [A-Z_]+ RESULTS?:?\s*", "", ev_text)
        parts = clean.split(" | ") if " | " in clean else clean.splitlines()
        for part in parts:
            p_strip = part.strip()
            if not p_strip:
                continue
            m = re.match(r"([a-zA-Z0-9_\-\.\\/]+):(\d+(?:-\d+)?).*?:\s*(.*)", p_strip, re.DOTALL)
            if m:
                fpath, lrange, code = m.group(1), m.group(2), m.group(3).strip()
                if target_sym and target_sym in code:
                    if f"def {target_sym}" in code:
                        if not any(d[0] == fpath for d in defs):
                            defs.append((fpath, lrange, code))
                    elif "import" in code and target_sym in code:
                        if not any(imp[0] == fpath for imp in imports):
                            imports.append((fpath, lrange, code))
                    elif f"{target_sym}(" in code or target_sym in code:
                        if not any(c[0] == fpath and c[1] == lrange for c in candidate_callers) and not f"def {target_sym}" in code:
                            candidate_callers.append((fpath, lrange, code))
            else:
                for line in p_strip.splitlines():
                    sline = line.strip()
                    if sline.startswith(("* ", "- ", "• ")) and len(sline) > 5:
                        b = sline.lstrip("*-• ").strip()
                        if b and b not in feature_bullets and len(b) > 10:
                            feature_bullets.append(b)
                    else:
                        h = re.sub(r"^#+\s*", "", sline).strip()
                        if len(h) > 30 and not h.startswith(("===", "---", "RAW ", "utils/", "src/")):
                            if not any(h in p for p in overview_paragraphs):
                                overview_paragraphs.append(h)

    source_excerpts = [
        excerpt for s in usable
        if (excerpt := _source_excerpt(str(s.get("evidence") or ""))) is not None
    ]

    # 1. Caller discovery: only a structured call-graph response is treated as
    # a complete set. Lexical hits are labeled as candidates, never as proof of
    # exhaustive coverage.
    if is_caller_query or caller_search_seen:
        sym_display = f"`{target_sym}`" if target_sym else "The function"
        lines = [f"### Caller search for {sym_display}"]
        if caller_search_seen:
            lines.append("Definite direct callers found by the indexed call graph:")
            if callers:
                for fpath, line, caller in callers:
                    lines.append(f"- `{caller}` — `{fpath}:{line}`")
            else:
                lines.append("- No definite direct callers were returned.")
            if possible_callers:
                lines.append("Possible same-name call sites (not confirmed as calls to this symbol):")
                lines.extend(f"- `{caller}` — `{fpath}:{line}`" for fpath, line, caller in possible_callers)
            if not caller_search_complete:
                lines.append("The call-graph result was truncated; additional callers may exist.")
            else:
                lines.append("This covers direct calls resolved by the indexed static analysis; dynamic dispatch may not be discoverable.")
        else:
            lines.append("Search evidence contains these candidate call sites; this search is not an exhaustive caller list:")
            if candidate_callers:
                lines.extend(f"- `{fpath}:{line}`: `{code.strip()}`" for fpath, line, code in candidate_callers)
            else:
                lines.append("- No caller call sites were established from the recorded search evidence.")
        if imports:
            lines.append("Import references (not call sites):")
            for fpath, lrange, code in imports:
                lines.append(f"- `{fpath}:{lrange}`: `{code}`")
        if defs:
            lines.append("Definitions (not call sites):")
            for fpath, lrange, _ in defs:
                lines.append(f"- `{fpath}:{lrange}`")
        return "\n".join(lines)

    # 2. Numeric-result questions must use reported values from source evidence,
    # not a calculation snippet that happens to mention the same metric.
    if _numeric_result_requested(task):
        values = _reported_numeric_values(task, source_excerpts)
        if values:
            lines = ["### Reported numerical values"]
            lines.extend(f"- `{path}:{line}` — `{value}`" for path, line, value in values[:12])
            return "\n".join(lines)
        return "The recorded source evidence does not contain a labeled numerical result for this request."

    # 3. Repository purpose comes from documentation actually present in the
    # evidence. Do not substitute a repository-specific description.
    if is_purpose_query:
        paragraphs = _purpose_document_paragraphs(source_excerpts)
        if paragraphs:
            lines = ["### Documented repository purpose", "The repository documentation describes it as:"]
            for path, start, end, paragraph in paragraphs[:2]:
                cite = f"{path}:{start}" if start == end else f"{path}:{start}-{end}"
                lines.append(f"- `{cite}` — {paragraph}")
            return "\n".join(lines)
        return "The recorded evidence does not establish the repository's documented purpose."

    # 4. Dataset/evaluation questions should expose configured inputs and
    # sampling while clearly separating source configuration from an executed
    # run or full-split result.
    if re.search(r"\bdataset\b", task, re.I) and re.search(r"\bevaluat\w*\b", task, re.I):
        config_lines = _evaluation_configuration_lines(source_excerpts)
        if config_lines:
            lines = ["### Evaluation data configuration"]
            lines.extend(f"- `{path}:{line}` — `{text}`" for path, line, text in config_lines[:18])
            lines.append("These source lines show configuration; they do not establish that an evaluation run completed.")
            if any(re.search(r"\b(?:subset|sample|random\.sample)\b", text, re.I) for _, _, text in config_lines):
                lines.append("The evidence includes subset or sampling configuration, so it does not establish full-split evaluation.")
            return "\n".join(lines)

    # 5. Function behavior is grounded in the retrieved implementation.
    if symbol_defs:
        citation = ""
        for step in usable:
            excerpt = _source_excerpt(str(step.get("evidence") or ""))
            if excerpt and step.get("tool") == "get_symbol":
                path, first, lines = excerpt
                citation = f"\nSource: `{path}:{first}`."
                break
        return f"### Source implementation\n\n```python\n{symbol_defs[0]}\n```{citation}"

    # Generic fallback: return only source lines that overlap the requested
    # topic, with their original locations.
    topic = _numeric_topic_terms(task)
    if topic:
        matches = []
        for path, first, source_lines in source_excerpts:
            for index, line in enumerate(source_lines):
                words = set(re.findall(r"[a-z][a-z0-9_]*", line.casefold()))
                for word in tuple(words):
                    words.update(word.split("_"))
                if topic & words and line.strip():
                    matches.append((path, first + index, line.strip()))
        if matches:
            lines = ["### Relevant source evidence"]
            lines.extend(f"- `{path}:{line}` — `{text}`" for path, line, text in matches[:12])
            return "\n".join(lines)

    return "The recorded repository evidence does not establish an answer to this request."


def build_final_response(support: str, criteria: list[dict], plan: Optional[dict], model_text: str,
                         failure_info: Optional[dict], verification_info: Optional[dict],
                         corpus: str, extra_unresolved: tuple = (),
                         answer_check: Optional[dict] = None) -> tuple[str, str]:
    """Build the final answer from recorded state (deterministic)."""
    claim_groups = _answer_claim_groups(answer_check, plan, corpus)
    supporting = _supporting_lines(criteria, plan, answer_check, corpus)
    reviewed = _reviewed_lines(criteria, plan, answer_check, corpus) if support != SUPPORTED else []
    unresolved = _unresolved_lines(support, criteria, plan, failure_info, verification_info)
    unresolved += [f"- {line}" for line in extra_unresolved]
    is_rate_limit_validation = bool(
        answer_check
        and answer_check.get("error")
        and (
            answer_check.get("validation_status") == "validation_unavailable: rate_limit"
            or "429" in str(answer_check.get("error", ""))
            or "rate limit" in str(answer_check.get("error", "")).lower()
        )
    )
    if support != SUPPORTED and answer_check and not answer_check.get("supported"):
        if is_rate_limit_validation:
            unresolved.append("- Validation status: validation_unavailable: rate_limit")
        if answer_check.get("error"):
            unresolved.append("- Final answer check failed: " + _clip(answer_check["error"], 240))
    if support == SUPPORTED:
        clean_draft = model_text.strip()
        objective = str((plan or {}).get("objective") or "") if isinstance(plan, dict) else ""
        if not clean_draft:
            clean_draft = synthesize_deterministic_answer(objective, plan, corpus)

        if clean_draft:
            text, unverified = _sanitize_conclusion(clean_draft, plan, corpus)
            if not text.strip():
                text = clean_draft
            if unverified:
                unresolved.append("- Not found in the recorded evidence (treat as unverified): " + ", ".join(unverified))
        elif claim_groups.get("confirmed"):
            text = " ".join(item["claim"] for item in claim_groups["confirmed"])
        else:
            text = synthesize_deterministic_answer(objective, plan, corpus)
    elif support == PARTIAL:
        all_criteria_satisfied = bool(criteria) and all(c.get("satisfied") for c in criteria)
        has_pending = bool(pending_steps(plan))
        decision_failed = any(
            "model" in str(item).lower() and ("failed" in str(item).lower() or "error" in str(item).lower())
            for item in extra_unresolved
        )
        if all_criteria_satisfied:
            if answer_check and answer_check.get("error"):
                clean_draft = model_text.strip()
                draft_ref_issues = conclusion_reference_issues(clean_draft, plan, corpus) if clean_draft else []
                if is_rate_limit_validation and clean_draft and not draft_ref_issues and not has_pending:
                    sanitized_draft, _ = _sanitize_conclusion(clean_draft, plan, corpus)
                    text = (
                        f"{sanitized_draft}\n\n"
                        "*(Validation status: `validation_unavailable: rate_limit` — the evidence supports the "
                        "requested criteria and deterministic citation checks passed, but final claim validation "
                        "failed due to provider rate limiting; the draft remains unverified.)*"
                    )
                elif is_rate_limit_validation and draft_ref_issues:
                    for ref_issue in draft_ref_issues:
                        unresolved.append(f"- Citation check rejected draft: {ref_issue}")
                    text = (
                        "The evidence supports the requested criteria, but final claim validation failed "
                        "and deterministic citation checks rejected the draft; the draft remains unverified."
                    )
                elif has_pending:
                    text = (
                        "The evidence supports the requested criteria, but final claim validation failed "
                        "(the draft remains unverified) and planned investigation steps remained pending."
                    )
                else:
                    text = (
                        "The evidence supports the requested criteria, but final claim validation failed; "
                        "the draft remains unverified."
                    )
            elif decision_failed:
                if has_pending:
                    text = (
                        "Every requested criterion is supported by the evidence below, but the run stopped "
                        "because a model call failed before a validated final answer was written "
                        "(planned investigation steps remained pending; no requested criterion is unmet)."
                    )
                else:
                    text = (
                        "Every requested criterion is supported by the evidence below, but the run stopped "
                        "before a validated final answer was written."
                    )
            elif not model_text.strip():
                if has_pending:
                    text = (
                        "Every requested criterion is supported by the evidence below, but the run stopped "
                        "before a validated final answer was written while planned investigation steps "
                        "remained pending."
                    )
                else:
                    text = (
                        "Every requested criterion is supported by the evidence below, but the run stopped "
                        "before a validated final answer was written."
                    )
            elif answer_check is None:
                text = (
                    "Every requested criterion is supported by the evidence below, though planned "
                    "investigation steps remained pending and the draft answer was not validated."
                )
            elif not answer_check.get("supported"):
                text = (
                    "Every requested criterion is supported by the evidence below, though planned "
                    "investigation steps remained pending and unverified draft claims were not accepted."
                )
            else:
                text = (
                    "Every requested criterion is supported by the evidence below, though planned "
                    "investigation steps remained pending before completion could be finalized."
                )
        else:
            text = "Some findings are established by the recorded evidence; other requested details remain unresolved."
    elif support == BLOCKED:
        text = ("The requested conclusion could not be established because the investigation was blocked "
                "(the latest change/verification failed, or no attempt produced usable evidence).")
    elif support == GRADING_FAILED:
        text = "The evidence grader failed after a bounded retry, so the gathered evidence could not be assessed."
    else:
        if claim_groups["confirmed"] or claim_groups["possible"]:
            text = "The overall request remains unestablished, though the evidence contains the findings below."
        else:
            text = "The investigation could not establish the requested result from the recorded evidence."
    parts = ["Conclusion", text]
    if support != SUPPORTED and claim_groups["confirmed"]:
        parts += ["", "Confirmed findings"]
        parts.extend(f"- {item['claim']} [{', '.join(item.get('evidence_ids', []))}]"
                     for item in claim_groups["confirmed"])
    if support != SUPPORTED and claim_groups["possible"]:
        by_id = {step["step_id"]: step for step in usable_evidence_steps(plan, criteria)}
        parts += ["", "Reasonable but unconfirmed possibilities"]
        for item in claim_groups["possible"]:
            ids = ", ".join(item.get("evidence_ids", []))
            parts.append(f"- {item['claim']} (possible, not established) [{ids}]")
            for step_id in item.get("evidence_ids", []):
                step = by_id.get(step_id)
                if step:
                    parts.append(f"  - Evidence considered ({step_id}, {step['action']}): "
                                 f"{_clip(step['evidence'], _EXCERPT_CHARS)}")
    if supporting:
        parts += ["", "Supporting evidence"] + supporting
    if reviewed:
        parts += ["", "Evidence reviewed but not accepted as support"] + reviewed
    if unresolved:
        heading = "Unknown / unresolved" if support in (PARTIAL, UNSUPPORTED) else "Unresolved"
        parts += ["", heading] + unresolved
    if (support != SUPPORTED and model_text.strip()
            and (answer_check is None or answer_check.get("error"))):
        draft, unverified = _sanitize_conclusion(_clip(model_text.strip(), 1200), plan, corpus)
        parts += ["", "Agent draft (unverified; not accepted as the answer)", draft]
        if unverified:
            parts += ["Unverified file references in that draft: " + ", ".join(unverified)]
    return "\n".join(parts), "\n".join(unresolved)


def answer_validation_status(answer_check: Optional[dict], plan: Optional[dict] = None,
                             corpus: str = "") -> str:
    """not_run | failed | no_accepted_claims | claims_rejected | passed."""
    if answer_check is None:
        return "not_run"
    if answer_check.get("error"):
        return "failed"
    groups = _answer_claim_groups(answer_check, plan, corpus)
    if not groups["confirmed"]:
        return "no_accepted_claims"
    if groups["unresolved"] or groups["possible"] or not answer_check.get("supported"):
        return "claims_rejected"
    return "passed"


def build_completion(support: str, criteria: list[dict], unresolved: str,
                     answer_check: Optional[dict] = None, plan: Optional[dict] = None,
                     corpus: str = "") -> dict:
    """The completion contract."""
    validation = answer_validation_status(answer_check, plan, corpus)
    established = criteria_established(criteria, plan) and not incomplete_reasons(criteria, plan)
    if support == SUPPORTED and not established:
        support = PARTIAL if any(c.get("satisfied") for c in criteria) else UNSUPPORTED
    answer_supported = support == SUPPORTED and established
    if not answer_supported:
        source = "none"
    elif validation == "passed":
        source = "validated_claims"
    elif validation == "claims_rejected":
        source = "accepted_claims"
    else:
        source = "criteria_evidence"
    comp = {"support": support,
            "criteria_supported": sum(1 for c in criteria if c["satisfied"]),
            "criteria_total": len(criteria),
            "unmet": [c["id"] for c in criteria if not c["satisfied"]],
            "answer_supported": answer_supported,
            "answer_validation": validation,
            "answer_source": source,
            "unresolved": unresolved}
    if answer_check and (
        answer_check.get("validation_status") == "validation_unavailable: rate_limit"
        or (
            answer_check.get("error")
            and ("429" in str(answer_check.get("error", "")) or "rate limit" in str(answer_check.get("error", "")).lower())
        )
    ):
        comp["validation_status"] = "validation_unavailable: rate_limit"
    return comp
