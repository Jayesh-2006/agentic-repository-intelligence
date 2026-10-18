"""Deterministic request intent, routing, source-scope enforcement, and capability limits.

An intent is a JSON-safe dictionary:

    {"kind": "explain" | "check" | "audit" | "change",
     "depth": "minimal" | "standard" | "deep",
     "rigor": "light" | "standard" | "strict",
     "answer_style": "short" | "normal" | "detailed",
     "sources": ["optional/path/to/file", ...],
     "scope": {"mentioned": [...], "starting_points": [...], "targets": [...], "cross_file": bool}}

`sources` is the ONLY hard read/scope restriction: it is set only when the user
explicitly limits investigation to particular files, and it is what tool
authorization enforces across the full repository tool suite. `scope` is
advisory structure about the files a request merely mentions: a mentioned file
does not restrict the repository; starting points are where tracing should
begin; targets are what the user wants explained/inspected/changed; `cross_file`
says investigation may follow imports, callers and data flow elsewhere. Read
scope is separate from mutation authorization (`capabilities_for_floor`), which
never looks at `scope` or `sources`.

The deterministic floor captures explicit user constraints. A planner may
raise intent values but cannot lower that floor. State readers tolerate
older or hand-built states without an intent value:

    intent = state.get("intent") or default_intent()

`get_intent(state)` provides this default.
"""

from __future__ import annotations

import posixpath
import re
from typing import Any, Optional

from typing_extensions import NotRequired, TypedDict

from .json_utils import parse_json_object

# --------------------------------------------------------------- enums

KIND_EXPLAIN = "explain"
KIND_CHECK = "check"
KIND_AUDIT = "audit"
KIND_CHANGE = "change"
_KINDS = (KIND_EXPLAIN, KIND_CHECK, KIND_AUDIT, KIND_CHANGE)

DEPTH_MINIMAL = "minimal"
DEPTH_STANDARD = "standard"
DEPTH_DEEP = "deep"
_DEPTHS = (DEPTH_MINIMAL, DEPTH_STANDARD, DEPTH_DEEP)
_DEPTH_ORDER = {v: i for i, v in enumerate(_DEPTHS)}

RIGOR_LIGHT = "light"
RIGOR_STANDARD = "standard"
RIGOR_STRICT = "strict"
_RIGORS = (RIGOR_LIGHT, RIGOR_STANDARD, RIGOR_STRICT)
_RIGOR_ORDER = {v: i for i, v in enumerate(_RIGORS)}

STYLE_SHORT = "short"
STYLE_NORMAL = "normal"
STYLE_DETAILED = "detailed"
_STYLES = (STYLE_SHORT, STYLE_NORMAL, STYLE_DETAILED)
_STYLE_ORDER = {v: i for i, v in enumerate(_STYLES)}


class Intent(TypedDict):
    kind: str
    depth: str
    rigor: str
    answer_style: str
    sources: list[str]
    scope: NotRequired[dict]


def empty_scope() -> dict:
    return {"mentioned": [], "starting_points": [], "targets": [], "cross_file": False}


def default_intent() -> Intent:
    """The Intent used when nothing else - no deterministic floor, no
    planner-proposed intent - applies. Deliberately the most permissive,
    middle-of-the-road choice on every axis."""
    return {"kind": KIND_EXPLAIN, "depth": DEPTH_STANDARD, "rigor": RIGOR_STANDARD,
            "answer_style": STYLE_NORMAL, "sources": [], "scope": empty_scope()}


def get_intent(state: dict) -> Intent:
    """Read `DeepAgentState.intent` tolerant of old/hand-built states that
    do not have it at all."""
    return state.get("intent") or default_intent()


def _max_enum(a: Optional[str], b: Optional[str], order: dict) -> Optional[str]:
    if a is None:
        return b
    if b is None:
        return a
    return a if order[a] >= order[b] else b


def _min_enum(a: Optional[str], b: Optional[str], order: dict) -> Optional[str]:
    if a is None:
        return b
    if b is None:
        return a
    return a if order[a] <= order[b] else b


# ---------------------------------------------------------- normalisation

_KIND_SYNONYMS = {
    "explain": KIND_EXPLAIN, "explanation": KIND_EXPLAIN, "describe": KIND_EXPLAIN, "summary": KIND_EXPLAIN,
    "check": KIND_CHECK, "verify": KIND_CHECK, "confirm": KIND_CHECK, "verification": KIND_CHECK,
    "audit": KIND_AUDIT, "review": KIND_AUDIT,
    "change": KIND_CHANGE, "implement": KIND_CHANGE, "modify": KIND_CHANGE, "fix": KIND_CHANGE,
    "update": KIND_CHANGE, "refactor": KIND_CHANGE,
}
_DEPTH_SYNONYMS = {
    "minimal": DEPTH_MINIMAL, "shallow": DEPTH_MINIMAL, "quick": DEPTH_MINIMAL, "minimum": DEPTH_MINIMAL,
    "standard": DEPTH_STANDARD, "normal": DEPTH_STANDARD, "medium": DEPTH_STANDARD,
    "deep": DEPTH_DEEP, "thorough": DEPTH_DEEP, "exhaustive": DEPTH_DEEP, "full": DEPTH_DEEP,
}
_RIGOR_SYNONYMS = {
    "light": RIGOR_LIGHT, "loose": RIGOR_LIGHT, "low": RIGOR_LIGHT,
    "standard": RIGOR_STANDARD, "normal": RIGOR_STANDARD, "medium": RIGOR_STANDARD,
    "strict": RIGOR_STRICT, "rigorous": RIGOR_STRICT, "high": RIGOR_STRICT,
}
_STYLE_SYNONYMS = {
    "short": STYLE_SHORT, "brief": STYLE_SHORT, "concise": STYLE_SHORT, "tldr": STYLE_SHORT, "quick": STYLE_SHORT,
    "normal": STYLE_NORMAL, "standard": STYLE_NORMAL, "medium": STYLE_NORMAL,
    "detailed": STYLE_DETAILED, "long": STYLE_DETAILED, "thorough": STYLE_DETAILED, "verbose": STYLE_DETAILED,
}


def _normalize_enum(value, synonyms: dict, valid: tuple, default: str) -> str:
    """Lowercase + synonym-map an enum value from an untrusted (LLM) source."""
    if not isinstance(value, str):
        return default
    key = value.strip().lower()
    if key in valid:
        return key
    return synonyms.get(key, default)


def normalize_source_path(raw: str) -> str:
    """Normalize obvious path-formatting differences (./README.md vs
    README.md, backslashes vs forward slashes) WITHOUT case-folding and WITHOUT
    allowing absolute paths or parent-directory (..) traversal.
    Returns "" for input that normalizes to nothing usable or attempts traversal."""
    if not isinstance(raw, str):
        return ""
    text = raw.strip().replace("\\", "/")
    if not text or "\x00" in text:
        return ""
    if text.startswith("/") or (len(text) >= 2 and text[1] == ":"):
        return ""
    raw_parts = [p for p in text.split("/") if p and p != "."]
    if ".." in raw_parts:
        return ""
    normalized = posixpath.normpath(text)
    if normalized in (".", "", "..") or normalized.startswith("../") or normalized.startswith("/"):
        return ""
    parts = [p for p in normalized.split("/") if p and p != "."]
    if not parts or ".." in parts:
        return ""
    return "/".join(parts)


def source_allows_path(intent: dict, path: str, service: Any = None) -> bool:
    """Whether a file operation on `path` is authorized by this invocation's
    `sources` restriction. True when there is no restriction at all, or when
    `path` - normalized via `normalize_source_path` and verified against
    symlinks/canonical containment when `service` is available - is one of the
    restricted paths."""
    sources = intent.get("sources") or []
    if not sources:
        return True
    norm = normalize_source_path(path)
    if not norm or norm not in set(sources):
        return False
    if service is not None and getattr(service, "index", None) is not None:
        root_str = getattr(service.index, "root", None)
        if isinstance(root_str, str) and root_str:
            from pathlib import Path
            import os
            from ..filesystem import is_within
            try:
                root = Path(root_str).resolve()
                unresolved = Path(os.path.abspath(root / norm))
                if not is_within(root, unresolved):
                    return False
                current = root
                for part in unresolved.relative_to(root).parts:
                    current = current / part
                    if current.is_symlink():
                        return False
                resolved = unresolved.resolve()
                if not is_within(root, resolved):
                    return False
                rel_posix = resolved.relative_to(root).as_posix()
                if rel_posix not in set(sources):
                    return False
            except (OSError, RuntimeError, ValueError):
                return False
    return True


def _structural_target_in_scope(
    intent: dict,
    tool_name: str,
    target: str,
    service: Any,
) -> bool:
    """Return True only when a `get_symbol` or `get_ast` target and all of its
    resulting data (including recursively nested child definitions and their
    resolved outgoing edges) are demonstrably within the permitted `sources`
    scope."""
    if service is None or not isinstance(target, str) or not target.strip():
        return False
    index = getattr(service, "index", None)
    graph = getattr(service, "graph", None)
    find_matches = getattr(service, "_find_matches", None)
    resolve_one = getattr(service, "_resolve_one", None)
    if index is None or not callable(find_matches) or not callable(resolve_one):
        return False

    cleaned = target.strip()
    if tool_name == "get_symbol":
        matches, _ = find_matches(cleaned)
        callable_matches = [m for m in matches if getattr(m, "kind", None) in ("function", "method", "class")]
        if len(callable_matches) != 1 or len(matches) != 1:
            return False
        sym = callable_matches[0]
        return bool(getattr(sym, "file", None) and source_allows_path(intent, sym.file, service=service))

    if tool_name == "get_ast":
        if cleaned.casefold().endswith(".py") or "/" in cleaned or "\\" in cleaned:
            if not source_allows_path(intent, cleaned, service=service):
                return False
            sym, err = resolve_one(cleaned)
            if err is not None or sym is None:
                return False
        else:
            matches, _ = find_matches(cleaned)
            if len(matches) != 1:
                return False
            sym = matches[0]
        if not getattr(sym, "file", None) or not source_allows_path(intent, sym.file, service=service):
            return False
        if graph is None:
            return False
        symbols_map = getattr(index, "symbols", {}) or {}
        edges_to_check: list[Any] = []
        find_children = getattr(graph, "find_children", None)
        visited_ids: set[str] = set()
        queue: list[str] = [sym.id]
        while queue:
            current_id = queue.pop(0)
            if not current_id or current_id in visited_ids:
                continue
            visited_ids.add(current_id)
            edges_to_check.extend(graph.neighbors(current_id))
            if callable(find_children):
                for child_edge in find_children(current_id):
                    edges_to_check.append(child_edge)
                    child_id = getattr(child_edge, "target", "")
                    if isinstance(child_id, str) and child_id and child_id not in visited_ids:
                        queue.append(child_id)
        for edge in edges_to_check:
            if getattr(edge, "resolved", False):
                target_sym = symbols_map.get(getattr(edge, "target", ""))
                if target_sym is None or not getattr(target_sym, "file", None):
                    return False
                if not source_allows_path(intent, target_sym.file, service=service):
                    return False
        return True

    return False


def source_allows_tool(
    intent: dict,
    tool_name: str,
    args: Optional[dict] = None,
    service: Any = None,
) -> tuple[bool, str]:
    """Enforce explicit `sources` restrictions across the entire repository tool suite."""
    sources = intent.get("sources") or []
    if not sources or tool_name == "revise_plan":
        return True, ""

    safe_args = args or {}
    if tool_name in ("get_file", "write_file"):
        requested_path = str(safe_args.get("path") or "")
        if source_allows_path(intent, requested_path, service=service):
            return True, ""
        action_desc = "direct file access" if tool_name == "get_file" else "file modification"
        return (
            False,
            f"Source restriction (system data): this task restricts {action_desc} to "
            f"{sources}, so {tool_name} was not run for {requested_path!r}. "
            "Choose an explicitly allowed source or state that the restricted evidence could not "
            "be established.",
        )

    if tool_name in ("get_symbol", "get_ast"):
        target = str(safe_args.get("symbol") if tool_name == "get_symbol" else safe_args.get("target") or "")
        if _structural_target_in_scope(intent, tool_name, target, service):
            return True, ""
        return (
            False,
            f"Source restriction (system data): this task is limited to {sources}. "
            f"{tool_name}({target!r}) was blocked because its target or resolved relationships "
            "are not demonstrably confined to the permitted file(s). Use get_file on an "
            "explicitly allowed path instead.",
        )

    tool_labels = {
        "find_symbol": "exact symbol lookup",
        "search_repository": "repository content search",
        "find_files": "repository file search",
        "find_callers": "repository-wide caller search",
        "find_callees": "cross-file callee resolution",
        "find_importers": "repository-wide importer search",
        "get_dependencies": "repository-wide dependency analysis",
    }
    if tool_name in tool_labels:
        return (
            False,
            f"Source restriction (system data): {tool_labels[tool_name]} ({tool_name}) inspects the "
            f"full repository, so it is unavailable for a task restricted to {sources}. "
            "Use get_file only for an explicitly allowed path.",
        )

    return (
        False,
        f"Source restriction (system data): this task is limited to {sources}. "
        f"The requested capability {tool_name!r} is not permitted because it could access other "
        "repository paths. Use direct get_file only for an explicitly allowed path.",
    )


def delegation_allows(intent: dict, tool: str, args: Optional[dict] = None, service: Any = None) -> bool:
    """Return whether `tool` is permitted by this invocation's explicit source restriction."""
    allowed, _ = source_allows_tool(intent, tool, args=args, service=service)
    return allowed


def normalize_model_intent(raw) -> dict:
    """Best-effort normalisation of a planner-supplied `intent` object."""
    if not isinstance(raw, dict):
        return {}
    out: dict = {}
    if "kind" in raw:
        out["kind"] = _normalize_enum(raw.get("kind"), _KIND_SYNONYMS, _KINDS, default_intent()["kind"])
    if "depth" in raw:
        out["depth"] = _normalize_enum(raw.get("depth"), _DEPTH_SYNONYMS, _DEPTHS, default_intent()["depth"])
    if "rigor" in raw:
        out["rigor"] = _normalize_enum(raw.get("rigor"), _RIGOR_SYNONYMS, _RIGORS, default_intent()["rigor"])
    if "answer_style" in raw:
        out["answer_style"] = _normalize_enum(raw.get("answer_style"), _STYLE_SYNONYMS, _STYLES,
                                              default_intent()["answer_style"])
    sources = raw.get("sources")
    if isinstance(sources, list):
        cleaned = [normalize_source_path(s) for s in sources if isinstance(s, str)]
        out["sources"] = [s for s in cleaned if s]
    return out


def parse_intent_reply(text: str) -> Optional[dict]:
    """Extract the raw (not yet normalized) "intent" object from the
    planning call's JSON reply."""
    data = parse_json_object(text or "")
    if data is not None and isinstance(data.get("intent"), dict):
        return data["intent"]
    return None


# ------------------------------------------------------- deterministic floor

_SHORT_RE = re.compile(r"\b(short|brief|concise|tl;?dr|one[\s-]sentence|one[\s-]line)\b", re.IGNORECASE)

# Source scope. Only an EXPLICIT limit ("only read X", "restrict yourself to X",
# "based solely on X", "nothing other than X") is a hard read restriction. A
# filename in an ordinary sentence ("in app.py", "from infer.py", "according to X")
# is merely mentioned.
_SOURCE_EXTENSIONS = (
    "py|md|rst|txt|json|ya?ml|toml|ini|cfg|sh|csv|xml|html?|css|scss|js|jsx|ts|tsx|"
    "go|rs|java|kt|c|h|cpp|hpp|sql|lock"
)
_SOURCE_PATH = rf"[\w./\\-]+\.(?:{_SOURCE_EXTENSIONS})\b"
_PATH_RE = re.compile(_SOURCE_PATH, re.IGNORECASE)
_PATH_LIST = rf"[\"']?{_SOURCE_PATH}[\"']?(?:\s*(?:,|&|\band\b)\s*[\"']?{_SOURCE_PATH}[\"']?)*"
_ONLY = r"(?:only|solely|exclusively|just)"
_SOURCE_RE = re.compile(
    rf"\b{_ONLY}\s+(?:read|use|using|check|look\s+at|consult|rely\s+on|refer\s+to)\s+"
    rf"(?:the\s+)?(?:files?\s+)?({_PATH_LIST})"
    rf"|\b(?:read|use|check|look\s+at|consult|rely\s+on|refer\s+to|search|stay)\s+{_ONLY}\s+"
    rf"(?:(?:in|inside|within|from|on)\s+)?(?:the\s+)?(?:files?\s+)?({_PATH_LIST})"
    rf"|\b(?:(?:based|relying|drawing)\s+{_ONLY}|{_ONLY}\s+(?:based|relying|drawing))\s+(?:on|from)\s+(?:the\s+)?(?:files?\s+)?({_PATH_LIST})"
    rf"|\b(?:restrict|limit|confine|constrain)\w*\s+(?:(?:your|the|this)\s+)?"
    rf"(?:yourself|reading|reads|search(?:es)?|investigation|analysis|exploration|lookups?|sources?|files?|scope)\s+"
    rf"to\s+(?:the\s+)?(?:files?\s+)?({_PATH_LIST})"
    rf"|\b(?:(?:read|use|check|look\s+at|consult)\s+)?(?:nothing\s+(?:but|other\s+than)|no\s+(?:other\s+)?files?\s+(?:other\s+)?than)\s+"
    rf"(?:the\s+)?(?:files?\s+)?({_PATH_LIST})"
    rf"|\bcheck\s+({_PATH_LIST})\s+only\b",
    re.IGNORECASE,
)
_START_RE = re.compile(
    rf"\b(?:start(?:ing)?|begin(?:ning)?)\s+(?:from|with|at|in)\s+(?:the\s+)?(?:files?\s+)?[\"']?({_SOURCE_PATH})"
    rf"|\b(?:trace|follow|walk|explore|track)\w*\b[^.?!\n]{{0,60}}?\b(?:from|starting\s+(?:at|from|in))\s+"
    rf"(?:the\s+)?(?:files?\s+)?[\"']?({_SOURCE_PATH})",
    re.IGNORECASE,
)
_TARGET_RE = re.compile(
    rf"\b(?:explain|describe|summari[sz]e|inspect|review|analy[sz]e|audit|open|read|show|"
    rf"fix|edit|modify|update|change|refactor|rewrite|implement\s+in|add\s+(?:\w+\s+){{0,3}}?to)\s+"
    rf"(?:me\s+)?(?:the\s+)?(?:file\s+)?[\"']?({_SOURCE_PATH})",
    re.IGNORECASE,
)
_CROSS_FILE_RE = re.compile(
    r"\b(?:across|trace|tracing|callers?|callees?|imports?|imported|depend\w*|data\s+flow|"
    r"call\s+graph|who\s+calls|used\s+by|end[- ]to[- ]end|flows?\s+(?:from|through)|interacts?)\b",
    re.IGNORECASE,
)


def _clean_paths(raw: list) -> list:
    out: list = []
    for item in raw:
        path = normalize_source_path(item.strip("\"'"))
        if path and path not in out:
            out.append(path)
    return out


def extract_hard_sources(text: str) -> list:
    """Paths the user explicitly limits investigation to (the hard read scope)."""
    found: list = []
    for match in _SOURCE_RE.finditer(text or ""):
        group = next((g for g in match.groups() if g), "")
        found.extend(_PATH_RE.findall(group))
    return _clean_paths(found)


def extract_scope(text: str, hard_sources: Optional[list] = None) -> dict:
    """Structured, advisory scope for the request (see module docstring).
    Never restricts anything by itself."""
    text = text or ""
    hard = extract_hard_sources(text) if hard_sources is None else list(hard_sources)
    starting = _clean_paths([g for m in _START_RE.finditer(text) for g in m.groups() if g])
    targets = _clean_paths([m.group(1) for m in _TARGET_RE.finditer(text)])
    mentioned = _clean_paths(_PATH_RE.findall(text))
    cross = not hard and bool(starting or _CROSS_FILE_RE.search(text))
    return {"mentioned": mentioned, "starting_points": starting, "targets": targets, "cross_file": cross}


def scope_of(intent: dict) -> dict:
    return {**empty_scope(), **(intent.get("scope") or {})}


_SERIOUS_RE = re.compile(
    r"\b(verify|audit|confirm|actually)\b|\bwhether\b.{0,40}\bimplemented\b",
    re.IGNORECASE,
)

# The user explicitly asking to RUN the repository's tests/checks.
_RUN_CHECKS_RE = re.compile(
    r"\b(?:run|execute|re-?run)\b(?:\s+\w+){0,4}?\s+(?:tests?|test\s+suite|pytest|checks?|verification)\b"
    r"|\b(?:do|does|did)\b(?:\s+\w+){0,4}?\s+(?:tests?|test\s+suite)\s+(?:pass|fail)\b"
    r"|\btests?\s+(?:pass|passing|are\s+passing)\b",
    re.IGNORECASE,
)

_NEGATED_RUN_PREFIX_RE = re.compile(
    r"\b(?:do|does|did|must|should|shall|will|can|could|may|need)\s+not\b"
    r"(?:\s+\w+){0,8}\s*$|"
    r"\b(?:don't|doesn't|didn't|mustn't|shouldn't|shan't|won't|can't|couldn't|wouldn't|cannot|never|avoid)\b"
    r"(?:\s+\w+){0,8}\s*$|"
    r"\bno\s+need\s+to\b(?:\s+\w+){0,8}\s*$|"
    r"\bnot\s+to\b(?:\s+\w+){0,6}\s*$|"
    r"\bno\s+(?:requirement|need|necessity)\s+to\b(?:\s+\w+){0,6}\s*$|"
    r"\b(?:not|never)\b(?:\s+\w+){0,8}\s*$|"
    r"\bnot\s+(?:necessary|allowed|supposed|required|intended)\s+to\b(?:\s+\w+){0,6}\s*$",
    re.IGNORECASE,
)
_NEGATED_RUN_SUFFIX_RE = re.compile(
    r"^\s*(?:is|are|was|were|would\s+be|should\s+be|will\s+be|has\s+been|have\s+been)?\s*"
    r"not\s+(?:necessary|required|needed|allowed|requested|intended|desired|permitted)\b",
    re.IGNORECASE,
)


def _requests_run_checks(task: str) -> bool:
    """Whether the user affirmatively asks to run checks."""
    for match in _RUN_CHECKS_RE.finditer(task or ""):
        prefix = task[:match.start()].strip()
        prefix = re.split(r"[.!?;\n]|\b(?:but|however|instead)\b", prefix, flags=re.IGNORECASE)[-1]
        prefix = re.sub(r"[^\w']+", " ", prefix).strip()
        if _NEGATED_RUN_PREFIX_RE.search(prefix):
            continue
        suffix = task[match.end():]
        suffix = re.split(r"[.!?;\n]|\b(?:but|however|instead)\b", suffix, flags=re.IGNORECASE)[0]
        if _NEGATED_RUN_SUFFIX_RE.search(suffix):
            continue
        if re.search(r"\b(?:no|without)\s+(?:the\s+)?(?:tests?|test\s+suite|checks?|verification)\b",
                     match.group(), re.IGNORECASE):
            continue
        return True
    return False


_MUTATION_RE = re.compile(
    r"\b(implement|fix|modify|update|refactor|rename|add|remove|delete|create|write|change)\b",
    re.IGNORECASE,
)

_ACTION_VERBS = (r"implement|fix|modify|update|refactor|rename|add|remove|delete|create|write|change"
                 r"|edit|rewrite|replace|patch")
_DELEGATED_CHANGE = (r"(?:make|apply)\s+(?:all\s+)?(?:the|those|these|that|such|any)\s+"
                     r"(?:\w+\s+)?(?:changes?|edits?|modifications?|fix(?:es)?|updates?|corrections?|patch(?:es)?)\b")
_ACTION_START_RE = re.compile(r"^(?:(?:" + _ACTION_VERBS + r")\b|" + _DELEGATED_CHANGE + r")", re.IGNORECASE)

_FILLER_RE = re.compile(
    r"^(?:(?:ok(?:ay)?|now|then|also|next|so|and|first|finally|please|kindly|just|hey|hi|thanks)\b[\s,]*"
    r"|go\s+ahead\s+and\s+|(?:i|we)\s+(?:want|need|would\s+like|'d\s+like|expect)\s+you\s+to\s+"
    r"|i'd\s+like\s+you\s+to\s+|(?:can|could|would|will)\s+you\s+(?:please\s+|kindly\s+)?"
    r"|you\s+(?:should|must|need\s+to|have\s+to)\s+|let'?s\s+)",
    re.IGNORECASE,
)
_LEAD_SCOPE_RE = re.compile(
    r"^(?:in|on|for|from|within|inside|at|according\s+to|based\s+on)\s+[^,;:?!\s]+(?:\s+[^,;:?!\s]+){0,8},\s*",
    re.IGNORECASE)
_DISCUSSION_START_RE = re.compile(
    r"^(?:how|what|which|where|why|when|who|whom|whose|is|are|was|were|do|does|did|don'?t|not|never|"
    r"has|have|had|can|could|would|should|shall|will|may|might|"
    r"explain|describe|show|tell|list|summari[sz]e|walk|outline|suggest|recommend|advise|"
    r"review|analy[sz]e|compare|find|give|i\s+(?:wonder|want\s+to|would\s+like\s+to|am|'m))\b",
    re.IGNORECASE,
)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?;])\s+|\n+")
_CLAUSE_SPLIT_RE = re.compile(r"\s*(?:,\s*then\b|\bthen\b|,?\s*and\s+then\b|,?\s+and\b)\s*", re.IGNORECASE)
_THEN_SPLIT_RE = re.compile(r"\s*(?:,\s*then\b|,?\s*and\s+then\b|\bthen\b)\s*", re.IGNORECASE)


def _strip_lead(clause: str) -> str:
    """Remove leading filler / scope phrases (repeatedly) from a clause."""
    prev = None
    text = clause.strip().strip("\"'`")
    while text != prev:
        prev = text
        text = _FILLER_RE.sub("", text, count=1).strip()
        text = _LEAD_SCOPE_RE.sub("", text, count=1).strip()
    return text


def is_mutation_request(task: str) -> bool:
    """Deterministic: does the user's text ASK THE AGENT to change the
    repository (imperative/requested action), as opposed to merely mentioning
    or discussing a change?"""
    for sentence in _SENTENCE_SPLIT_RE.split(task or ""):
        sentence = sentence.strip()
        if not sentence:
            continue
        head = _strip_lead(sentence)
        if _DISCUSSION_START_RE.search(head):
            later = _THEN_SPLIT_RE.split(sentence)[1:]
            if any(_ACTION_START_RE.search(_strip_lead(c)) and not _DISCUSSION_START_RE.search(_strip_lead(c))
                   for c in later):
                return True
            continue
        for clause in _CLAUSE_SPLIT_RE.split(sentence):
            if _ACTION_START_RE.search(_strip_lead(clause)):
                return True
    return False


_SIMPLE_COMPLEX_RE = re.compile(
    r"\b(across|throughout|all|every|multiple|architecture|dependencies|dependency|"
    r"callers|callees|trace|tracing|compare|comparison|contrast|difference|differences|differ|"
    r"versus|vs\.?|relationship|relationships|interaction|interactions|workflow|workflows|"
    r"lifecycle|life\s+cycle|pipeline|end[- ]to[- ]end|repository[- ]wide|"
    r"detailed|detail|flows?|stages?|components?)\b|"
    r"\bcomplete\s+flow\b|"
    r"\bflow\s+from\s+.+\s+to\s+.+\b",
    re.IGNORECASE,
)
_EXPLANATORY_QUERY_RE = re.compile(
    r"\b(?:explain|describe|outline|summari[sz]e|walk\s+me\s+through|"
    r"overview|walkthrough)\b",
    re.IGNORECASE,
)
_HOW_BEHAVIOR_RE = re.compile(
    r"\bhow\b.{0,100}\b(?:work|works|function|functions|operate|operates|handle|handles|"
    r"flow|flows|interact|interacts|connect|connects|communicate|communicates|process|processes|"
    r"behave|behaves|implemented|organized|structured)\b",
    re.IGNORECASE,
)
_MULTIPART_QUERY_RE = re.compile(
    r"\b(?:and|or)\s+(?:also\s+)?(?:what|where|which|who|when|why|how|does|do|is|are|"
    r"list|explain|describe|compare|trace)\b|"
    r"\b(?:what|which|where|who|how)\b.{0,120}\b\w+\s+(?:and|or)\s+\w+\b",
    re.IGNORECASE,
)


def _requires_investigation(text: str) -> bool:
    """Conservatively route synthesis and multi-part questions to the planner."""
    return bool(
        _SIMPLE_COMPLEX_RE.search(text)
        or _EXPLANATORY_QUERY_RE.search(text)
        or _HOW_BEHAVIOR_RE.search(text)
        or _MULTIPART_QUERY_RE.search(text)
        or len(_SENTENCE_SPLIT_RE.split(text)) > 1
    )


_PURPOSE_RE = re.compile(
    r"\b(purpose|overview|summar(?:y|ize|ise)|about\s+(?:this|the)\s+(?:repo|repository|project)|"
    r"(?:use|uses|usage)\s+of\s+(?:this|the)\s+(?:repo|repository|project)|"
    r"what\s+(?:is|does)\s+(?:this|the)\s+(?:repo|repository|project)(?:\s+do)?)\b",
    re.IGNORECASE,
)


def is_purpose_query(task: str) -> bool:
    """Whether the request asks for a repository-level purpose or overview."""
    return bool(_PURPOSE_RE.search(" ".join((task or "").split())))


_SOURCE_QUESTION_WORDS = r"what|where|which|who|is|are|does|do|can|show|explain|how"
_SOURCE_LEAD_IN_RE = re.compile(
    rf"^(?:(?:(?:only|just)\s+)?(?:read|use|check|look\s+at)\s+(?:only\s+|nothing\s+(?:but|other\s+than)\s+|no\s+(?:other\s+)?files?\s+(?:other\s+)?than\s+)?"
    rf"|(?:nothing\s+(?:but|other\s+than)|no\s+(?:other\s+)?files?\s+(?:other\s+)?than)\s+"
    rf"|(?:restrict|limit|confine|constrain)\w*\s+(?:yourself|reading|reads|search(?:es)?|investigation|analysis|scope|files?|sources?)\s+to\s+)"
    rf"(?:the\s+)?(?:files?\s+)?(?:{_PATH_LIST})"
    rf"(?:\s*[,;:]\s*|\s+(?=(?:for\b|to\s+answer\b|and\s+answer\b|{_SOURCE_QUESTION_WORDS})\b))",
    re.IGNORECASE,
)


def normalize_simple_question(task: str) -> str:
    """Remove a recognized explicit source restriction from the front of a query."""
    text = " ".join((task or "").split())
    if not text:
        return ""
    return _SOURCE_LEAD_IN_RE.sub("", text, count=1).strip()



_LOOKUP_MAX_WORDS = 20


def is_lookup_task(task: str, floor: Optional[dict] = None) -> bool:
    """Route: may this task skip the planning LLM call and go straight to the
    read tools?"""
    text = " ".join((task or "").split())
    if not text:
        return False
    floor = floor if floor is not None else deterministic_floor(text)
    if floor.get("kind") == KIND_CHANGE or floor.get("rigor") in (RIGOR_STANDARD, RIGOR_STRICT):
        return False
    if floor.get("run_checks"):
        return False
    if (_MUTATION_RE.search(text) or _SERIOUS_RE.search(text)
            or _requires_investigation(text)):
        return False
    question = normalize_simple_question(text)
    if len(question.split()) > _LOOKUP_MAX_WORDS:
        return False
    if text.count("?") > 1 or ";" in text:
        return False
    return True


def default_capabilities() -> dict:
    """The safe default when no capability grant is present on state at all:
    read-only."""
    return {"read": True, "write": False, "execute": False}


def capabilities_for_floor(floor: dict) -> dict:
    """Runtime-safety capability grant for this invocation's execution
    context, derived ONLY from `deterministic_floor(task)`."""
    write = floor.get("kind") == KIND_CHANGE
    return {"read": True, "write": write, "execute": write or bool(floor.get("run_checks"))}


def deterministic_floor(task: str) -> dict:
    """Infer minimum intent requirements from the task text alone."""
    text = task or ""
    floor: dict = {}

    if _SHORT_RE.search(text):
        floor["answer_style_ceiling"] = STYLE_SHORT

    hard_sources = extract_hard_sources(text)
    if hard_sources:
        floor["sources"] = hard_sources
    scope = extract_scope(text, hard_sources)
    if any(scope.values()):
        floor["scope"] = scope

    rigor_floor = RIGOR_STRICT if _SERIOUS_RE.search(text) else None

    if is_mutation_request(text):
        floor["kind"] = KIND_CHANGE
        rigor_floor = _max_enum(rigor_floor, RIGOR_STANDARD, _RIGOR_ORDER)

    if rigor_floor is not None:
        floor["rigor"] = rigor_floor

    if _requests_run_checks(text):
        floor["run_checks"] = True

    return floor


# -------------------------------------------------------------- resolution

_PLANNER_SOURCE_RE = re.compile(
    rf"\b(?:nothing\s+(?:but|other\s+than)|no\s+(?:other\s+)?files?\s+(?:other\s+)?than)\s+({_PATH_LIST})"
    rf"|({_PATH_LIST})\s+only\s*(?:[.,;:!?]|$)",
    re.IGNORECASE,
)


def _associated_sources(text: str) -> set:
    found = list(extract_hard_sources(text))
    for match in _PLANNER_SOURCE_RE.finditer(text or ""):
        group = next((g for g in match.groups() if g), "")
        found.extend(_clean_paths(_PATH_RE.findall(group)))
    return set(found)


def _model_sources_credible(task: str, sources: list) -> bool:
    """A planner-proposed restriction is credible only if EVERY proposed path
    is explicitly tied to a restriction phrase in the user's own text."""
    if not sources:
        return True
    associated = _associated_sources(task or "")
    return all(src in associated for src in sources)


def resolve_intent(task: str, model_intent: Optional[dict] = None, *, floor: Optional[dict] = None) -> Intent:
    """Combine the deterministic floor with the (optional) planner-proposed
    intent into the final Intent for this invocation."""
    if floor is None:
        floor = deterministic_floor(task)
    base = default_intent()
    model = normalize_model_intent(model_intent) if model_intent else {}

    kind = floor.get("kind") or model.get("kind") or base["kind"]
    depth = model.get("depth", base["depth"])
    rigor = _max_enum(model.get("rigor", base["rigor"]), floor.get("rigor"), _RIGOR_ORDER)

    answer_style = model.get("answer_style", base["answer_style"])
    style_ceiling = floor.get("answer_style_ceiling")
    if style_ceiling is not None:
        answer_style = _min_enum(answer_style, style_ceiling, _STYLE_ORDER)

    sources = floor.get("sources")
    if sources is None:
        sources = model.get("sources", base["sources"])
        if not _model_sources_credible(task, sources):
            sources = base["sources"]

    return {"kind": kind, "depth": depth, "rigor": rigor, "answer_style": answer_style,
            "sources": list(sources), "scope": {**empty_scope(), **(floor.get("scope") or {})}}


# --------------------------------------------------------------- planning bounds

_DEPTH_STEP_LIMIT = {DEPTH_MINIMAL: 1, DEPTH_STANDARD: 3}
_RIGOR_CRITERIA_LIMIT = {RIGOR_LIGHT: 1, RIGOR_STANDARD: 3}


def plan_step_limit(intent: Intent, configured_max: int) -> int:
    """Max PENDING steps the planning call's output is clamped to, from
    `intent["depth"]`."""
    depth = intent.get("depth", DEPTH_STANDARD)
    if depth == DEPTH_DEEP:
        return max(1, configured_max)
    limit = _DEPTH_STEP_LIMIT.get(depth, _DEPTH_STEP_LIMIT[DEPTH_STANDARD])
    return max(1, min(configured_max, limit))


_DEPTH_AGENT_STEP_LIMIT = {DEPTH_MINIMAL: 4}


def agent_step_budget(intent: Intent, configured_max: int) -> int:
    """Max actual tool-using `decide` turns (`step_count`) this invocation
    may spend, from `intent["depth"]`."""
    if intent.get("depth") == DEPTH_MINIMAL:
        return max(1, min(configured_max, _DEPTH_AGENT_STEP_LIMIT[DEPTH_MINIMAL]))
    return max(1, configured_max)


def criteria_limit(intent: Intent, configured_max: int) -> int:
    """Max acceptance criteria, from `intent["rigor"]`."""
    rigor = intent.get("rigor", RIGOR_STANDARD)
    if rigor == RIGOR_STRICT:
        limit = configured_max
    else:
        limit = _RIGOR_CRITERIA_LIMIT.get(rigor, _RIGOR_CRITERIA_LIMIT[RIGOR_STANDARD])
    if intent.get("kind") == KIND_CHANGE:
        limit = max(limit, _RIGOR_CRITERIA_LIMIT[RIGOR_STANDARD])
    return max(1, min(configured_max, limit))
