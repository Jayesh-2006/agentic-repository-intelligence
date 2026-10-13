"""Deterministic request intent, routing, and capability limits.

An intent is a JSON-safe dictionary:

    {"kind": "explain" | "check" | "audit" | "change",
     "depth": "minimal" | "standard" | "deep",
     "rigor": "light" | "standard" | "strict",
     "answer_style": "short" | "normal" | "detailed",
     "sources": ["optional/path/to/file", ...]}

"check" (not "verify") is used for the intent kind specifically to avoid
confusion with the existing `verify_repository` delegation tool and
`DeepAgentState.verification_info` - those are about running the
repository's real test/verification command, which is unrelated to what
KIND of request the user made.

The deterministic floor captures explicit user constraints. A planner may
raise intent values but cannot lower that floor. State readers tolerate
older or hand-built states without an intent value:

    intent = state.get("intent") or default_intent()

`get_intent(state)` provides this default.
"""

from __future__ import annotations

import json
import posixpath
import re
from typing import Optional

from typing_extensions import TypedDict

from .plan import _CODE_FENCE_RE

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


def default_intent() -> Intent:
    """The Intent used when nothing else - no deterministic floor, no
    planner-proposed intent - applies. Deliberately the most permissive,
    middle-of-the-road choice on every axis."""
    return {"kind": KIND_EXPLAIN, "depth": DEPTH_STANDARD, "rigor": RIGOR_STANDARD,
            "answer_style": STYLE_NORMAL, "sources": []}


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
    """Lowercase + synonym-map an enum value from an untrusted (LLM) source.
    Anything unrecognised falls back to `default` rather than raising -
    the planner's own wording is never trusted verbatim."""
    if not isinstance(value, str):
        return default
    key = value.strip().lower()
    if key in valid:
        return key
    return synonyms.get(key, default)


def normalize_source_path(raw: str) -> str:
    """Normalize obvious path-formatting differences (./README.md vs
    README.md, backslashes vs forward slashes, a safe ".." normalization)
    WITHOUT case-folding - matching is left case-sensitive on purpose, so
    this never silently selects a different file on a case-sensitive
    filesystem. Returns "" for input that normalizes to nothing usable."""
    if not isinstance(raw, str):
        return ""
    text = raw.strip().replace("\\", "/")
    if not text:
        return ""
    normalized = posixpath.normpath(text)
    if normalized in (".", ""):
        return ""
    # Drop any ".." segments rather than let a restriction climb outside the
    # repo - this is intent bookkeeping only, not the actual file guard
    # so the safe thing here is simply to ignore escape attempts.
    parts = [p for p in normalized.split("/") if p and p != ".."]
    return "/".join(parts)


def source_allows_path(intent: dict, path: str) -> bool:
    """Whether a DIRECT get_file call for `path` is authorized by
    this invocation's `sources` restriction (see deep_agent.py's `decide`,
    get_file branch). True when there is no restriction at all, or when
    `path` - normalized the same way `sources` itself already was (see
    `normalize_source_path`) - is exactly one of the restricted paths.

    An explicit restriction such as "only read README.md" narrows what
    get_file may retrieve; it must never be read as blanket permission for
    get_file to read anything else. Deliberately exact-match, case-sensitive
    (matching `normalize_source_path`'s own contract) - this is a safety
    boundary, not a fuzzy lookup, so it never widens on a near-match."""
    sources = intent.get("sources") or []
    if not sources:
        return True
    return normalize_source_path(path) in set(sources)


def normalize_model_intent(raw) -> dict:
    """Best-effort normalisation of a planner-supplied `intent` object
    (parsed by `parse_intent_reply`). Only recognised keys are kept; every
    value is normalized safely (see `_normalize_enum`) so a malformed or
    unexpected planner reply can never raise - it just falls back per
    field."""
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
    planning call's JSON reply (see prompts.PLANNING_SYSTEM_PROMPT /
    plan.parse_plan_steps / evidence.parse_criteria, which read the same
    reply for "steps"/"criteria"). Returns None - never raises - when there
    is nothing usable; the caller then relies on the deterministic floor
    and defaults alone."""
    cleaned = _CODE_FENCE_RE.sub("", (text or "").strip()).strip()
    try:
        data = json.loads(cleaned)
    except (TypeError, ValueError):
        return None
    if isinstance(data, dict):
        intent = data.get("intent")
        if isinstance(intent, dict):
            return intent
    return None


# ------------------------------------------------------- deterministic floor

_SHORT_RE = re.compile(r"\b(short|brief|concise|tl;?dr|one[\s-]sentence|one[\s-]line)\b", re.IGNORECASE)

# Natural source limits such as "according to README.md", plus explicit
# forms such as "only read README.md" / "use only README.md".
_SOURCE_RE = re.compile(
    r"\b(?:according\s+to|based\s+on|from|in)\s+(?:file\s+)?[\"']?([\w./\\-]+\.\w+)"
    r"|\b(?:only|just)\s+(?:read|use|check|look\s+at)\s+[\"']?([\w./\\-]+\.\w+)"
    r"|\b(?:read|use|check|look\s+at)\s+only\s+[\"']?([\w./\\-]+\.\w+)"
    r"|\bcheck\s+[\"']?([\w./\\-]+\.\w+)\s+only\b",
    re.IGNORECASE,
)

_SERIOUS_RE = re.compile(
    r"\b(verify|audit|confirm|actually)\b|\bwhether\b.{0,40}\bimplemented\b",
    re.IGNORECASE,
)

# The user explicitly asking to RUN the repository's tests/checks. This is the
# only read-only-kind request that may execute the configured verification
# command (see capabilities_for_floor): "verify"/"audit"/"confirm" alone are
# reading words and do not grant execution.
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
    """Whether the user affirmatively asks to run checks.

    The execute capability is sensitive to negation: matching the words
    "run tests" inside "do not run tests" must never grant it. Evaluate each
    matching phrase independently so a later affirmative clause can still
    express permission after a negated one.
    """
    for match in _RUN_CHECKS_RE.finditer(task or ""):
        prefix = task[:match.start()].strip()
        # Only the current sentence/clause can negate this occurrence.
        prefix = re.split(r"[.!?;\n]|\b(?:but|however|instead)\b", prefix, flags=re.IGNORECASE)[-1]
        # Keep interrupted negatives such as "do not, under any
        # circumstances, run tests" intact for the bounded look-back above.
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

# Verbs that, used as an imperative, ask for a repository change.
_ACTION_VERBS = (r"implement|fix|modify|update|refactor|rename|add|remove|delete|create|write|change"
                 r"|edit|rewrite|replace|patch")
# "Make the changes you described", "apply those edits": delegated execution
# of already-discussed changes. Bare "make" (make sure / make a list) is NOT
# an action verb on its own.
_DELEGATED_CHANGE = (r"(?:make|apply)\s+(?:all\s+)?(?:the|those|these|that|such|any)\s+"
                     r"(?:\w+\s+)?(?:changes?|edits?|modifications?|fix(?:es)?|updates?|corrections?|patch(?:es)?)\b")
_ACTION_START_RE = re.compile(r"^(?:(?:" + _ACTION_VERBS + r")\b|" + _DELEGATED_CHANGE + r")", re.IGNORECASE)

# Leading filler that does not change whether the sentence is a command.
_FILLER_RE = re.compile(
    r"^(?:(?:ok(?:ay)?|now|then|also|next|so|and|first|finally|please|kindly|just|hey|hi|thanks)\b[\s,]*"
    r"|go\s+ahead\s+and\s+|(?:i|we)\s+(?:want|need|would\s+like|'d\s+like|expect)\s+you\s+to\s+"
    r"|i'd\s+like\s+you\s+to\s+|(?:can|could|would|will)\s+you\s+(?:please\s+|kindly\s+)?"
    r"|you\s+(?:should|must|need\s+to|have\s+to)\s+|let'?s\s+)",
    re.IGNORECASE,
)
# A leading scope phrase such as "In src/foo.py," / "According to X,".
_LEAD_SCOPE_RE = re.compile(
    r"^(?:in|on|for|from|within|inside|at|according\s+to|based\s+on)\s+[^,;:?!\s]+(?:\s+[^,;:?!\s]+){0,8},\s*",
    re.IGNORECASE)
# Sentences that open as a question, advice request or explanation request
# are discussion of a change, never a command to perform one.
_DISCUSSION_START_RE = re.compile(
    r"^(?:how|what|which|where|why|when|who|whom|whose|is|are|was|were|do|does|did|don'?t|not|never|"
    r"has|have|had|can|could|would|should|shall|will|may|might|"
    r"explain|describe|show|tell|list|summari[sz]e|walk|outline|suggest|recommend|advise|"
    r"review|analy[sz]e|compare|find|give|i\s+(?:wonder|want\s+to|would\s+like\s+to|am|'m))\b",
    re.IGNORECASE,
)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?;])\s+|\n+")
_CLAUSE_SPLIT_RE = re.compile(r"\s*(?:,\s*then\b|\bthen\b|,?\s*and\s+then\b|,?\s+and\b)\s*", re.IGNORECASE)
# Discussion sentences are only split on an explicit sequencing "then", never
# on a bare "and" ("explain how to fix and update foo" stays discussion).
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
    repository (imperative/delegated action), as opposed to merely mentioning
    or discussing a change (a how-to question, a request for advice or an
    explanation, a question about past changes)?

    Sentence by sentence, a sentence counts as a command only if - after
    optional filler such as "please" / "can you" / "I want you to" and a
    leading scope phrase - it starts with an action verb (or a delegation such
    as "make the changes ..."). A sentence that opens as a question or
    explanation/advice request is discussion regardless of verbs it contains
    ("How do I fix ...", "What changes should I make ..."). Within a command
    sentence, later coordinated clauses ("read foo.py and fix the bug") are
    checked the same way; discussion sentences are never split.
    """
    for sentence in _SENTENCE_SPLIT_RE.split(task or ""):
        sentence = sentence.strip()
        if not sentence:
            continue
        head = _strip_lead(sentence)
        if _DISCUSSION_START_RE.search(head):
            # Only a later, explicitly sequenced clause ("explain X, then
            # fix Y") can still be a command.
            later = _THEN_SPLIT_RE.split(sentence)[1:]
            if any(_ACTION_START_RE.search(_strip_lead(c)) and not _DISCUSSION_START_RE.search(_strip_lead(c))
                   for c in later):
                return True
            continue
        for clause in _CLAUSE_SPLIT_RE.split(sentence):
            if _ACTION_START_RE.search(_strip_lead(clause)):
                return True
    return False


# Conservative markers for questions that are likely to need repository-wide
# investigation rather than a single targeted read/investigation turn.
#
# "explain"-opened questions never reach this check in the first place -
# _SIMPLE_OPEN_RE (below) does not list "explain" as a lookup opener, on the
# principle that an explanation (a flow, a design) is not something a single
# matched source line can BE. But "what"/"is"/"does" etc. ARE listed openers,
# and a flow/pipeline/detailed-walkthrough question can just as easily start
# with one of those ("What is the complete flow from input to output?"), so
# this list also has to catch open-ended multi-step requests by their own
# wording, independent of how they open.
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
# Open-ended purpose/overview questions have no single matching line, so a
# keyword shortcut can only ever return an accidental match.
_PURPOSE_RE = re.compile(
    r"\b(purpose|overview|summar(?:y|ize|ise)|about\s+(?:this|the)\s+(?:repo|repository|project)|"
    r"(?:use|uses|usage)\s+of\s+(?:this|the)\s+(?:repo|repository|project)|"
    r"what\s+(?:is|does)\s+(?:this|the)\s+(?:repo|repository|project)(?:\s+do)?)\b",
    re.IGNORECASE,
)
# "explain" is deliberately absent: it asks for an explanation (a flow, a
# design), which a single matched source line cannot be.
_SIMPLE_OPEN_RE = re.compile(
    r"^(?:where|what|which|who|is|are|does|do|can|show|read)\b",
    re.IGNORECASE,
)
_SOURCE_QUESTION_WORDS = r"what|where|which|who|is|are|does|do|can|show|explain|how"
_SOURCE_LEAD_IN_RE = re.compile(
    r"^(?:(?:according\s+to|based\s+on|from|in)\s+(?:file\s+)?[\"']?[\w./\\-]+\.\w+[\"']?"
    r"(?:\s*[,;:]\s*|\s+(?=(?:" + _SOURCE_QUESTION_WORDS + r")\b))"
    r"|(?:(?:only|just)\s+)?(?:read|use|check|look\s+at)\s+(?:only\s+)?[\"']?[\w./\\-]+\.\w+[\"']?"
    r"(?:\s*[,;:]\s*|\s+(?=(?:for\b|to\s+answer\b|and\s+answer\b|" + _SOURCE_QUESTION_WORDS + r")\b)))",
    re.IGNORECASE,
)


def normalize_simple_question(task: str) -> str:
    """Remove a recognized source restriction from the front of a query.

    Source extraction/authorization remains owned by ``deterministic_floor``;
    this helper only removes the already-recognized lead-in so downstream
    simple retrieval ranks terms from the actual question rather than words
    such as ``only read README.md``. Unrecognized prefixes are left unchanged.
    """
    text = " ".join((task or "").split())
    if not text:
        return ""
    return _SOURCE_LEAD_IN_RE.sub("", text, count=1).strip()
_SIMPLE_SOURCE_TASK_RE = re.compile(
    r"^(?:(?:only|just)\s+(?:read|use|check|look\s+at)|"
    r"(?:read|use|check|look\s+at)\s+only)\b",
    re.IGNORECASE,
)


def looks_simple(task: str, floor: Optional[dict] = None) -> bool:
    """Conservative deterministic check for a single, targeted, read-only task.

    This helper exists only to decide whether the upfront planning LLM call can
    be skipped.  It must never override a deterministic floor: explicit
    mutation/verification wording always disables the shortcut.  The test is
    intentionally narrow so normal or complex questions still take the
    ordinary planning path.
    """
    text = " ".join((task or "").split())
    floor = floor if floor is not None else deterministic_floor(text)
    if not text:
        return False
    if floor.get("kind") == KIND_CHANGE or floor.get("rigor") == RIGOR_STRICT:
        return False
    if _MUTATION_RE.search(text) or _SERIOUS_RE.search(text):
        return False
    if _requires_investigation(text) or _PURPOSE_RE.search(text):
        return False
    question = normalize_simple_question(text)
    if len(question.split()) > 16:
        return False
    source_task = bool(floor.get("sources")) and bool(_SIMPLE_SOURCE_TASK_RE.search(text))
    if not _SIMPLE_OPEN_RE.search(question) and not source_task:
        return False
    # A direct file restriction is compatible with the shortcut because it
    # narrows the permitted read; it does not justify widening investigation.
    return True


# A lookup is one short, single-sentence, read-only request. 20 words leaves
# room for a natural question plus a file mention; anything longer is likely
# multi-part and goes through the planner.
_LOOKUP_MAX_WORDS = 20


def is_lookup_task(task: str, floor: Optional[dict] = None) -> bool:
    """Route: may this task skip the planning LLM call and go straight to the
    read tools (find_files -> get_file, or investigate_repository)?

    Unlike `looks_simple` (which decides whether the no-LLM keyword shortcut
    may answer, and so must match the question's exact shape), this does not
    look at how the question is phrased - "In what languages is the README
    available?" and "How many ..." are lookups just like "Where is X?". It
    only asks what the task can NOT be: not a change, not a strict/verify/
    audit request, not repository-wide in scope, not long or multi-part.
    The model still chooses which read tool to call; this only decides
    whether a plan and several criteria are drafted first.
    """
    text = " ".join((task or "").split())
    if not text:
        return False
    floor = floor if floor is not None else deterministic_floor(text)
    if floor.get("kind") == KIND_CHANGE or floor.get("rigor") in (RIGOR_STANDARD, RIGOR_STRICT):
        return False
    if floor.get("run_checks"):
        return False
    if (_MUTATION_RE.search(text) or _SERIOUS_RE.search(text)
            or _requires_investigation(text) or _PURPOSE_RE.search(text)):
        return False
    question = normalize_simple_question(text)
    if len(question.split()) > _LOOKUP_MAX_WORDS:
        return False
    if text.count("?") > 1 or ";" in text:
        return False
    return True


def default_capabilities() -> dict:
    """The safe default when no capability grant is present on state at all
    (e.g. an old/hand-built state that predates this field) - read-only,
    exactly like `default_intent()` defaults to the least assuming shape on
    every other axis. Never used to grant WRITE; only ever to withhold it."""
    return {"read": True, "write": False, "execute": False}


def capabilities_for_floor(floor: dict) -> dict:
    """Runtime-safety capability grant for this invocation's execution
    context, derived ONLY from `deterministic_floor(task)` - i.e. only from
    the user's own original task text, matched by the same deterministic
    `is_mutation_request` intent rule that forces `kind=KIND_CHANGE` (an
    actual imperative change request - not a question/advice/explanation
    that merely mentions a mutation word).

    This is deliberately NOT derived from the resolved `Intent` returned by
    `resolve_intent()`, because that also folds in whatever `kind` the
    planning LLM call proposed (`model_intent`) - an untrusted, LLM-produced
    value. If WRITE capability were read off the resolved intent, a planning
    reply (or, further downstream, an LLM-generated delegation task) could
    grant itself WRITE simply by claiming `"kind": "change"`. Computing it
    here instead, once, before any LLM call, and only from `floor`, keeps
    the grant a property of the execution context (what the user actually
    asked for) rather than of anything the model later generates.

    A `code_repository` (or other mutation-capable) delegation call is only
    ever executed by the runtime when this invocation's capabilities allow
    it - see deep_agent.py's `decide`. The Explorer's/Coder's/Debugger's own
    tool selection, and the Main Agent's/Deep Agent's choice of WHICH tool
    to call, are completely unaffected: this only answers whether a
    requested WRITE-capable delegation may actually run.
    """
    write = floor.get("kind") == KIND_CHANGE
    # EXECUTE gates verify_repository: it runs the repository's configured
    # command (e.g. the test suite), which can create or modify files in the
    # checkout even though the LLM supplies no arguments. A change request
    # implies it; a read-only request needs to ask for a run explicitly.
    return {"read": True, "write": write, "execute": write or bool(floor.get("run_checks"))}


def delegation_allows(intent: dict, tool: str) -> bool:
    """Return whether a Deep-Agent delegation is permitted by this invocation's
    explicit source restriction.

    An explicit sources list is a hard per-invocation read scope.  The only
    repository capability that can safely honor it without granting additional
    repository access is the direct get_file read.  The read-only Explorer,
    Coder and verification delegations can otherwise inspect arbitrary paths,
    so allowing them would bypass the restriction.
    """
    sources = intent.get("sources") or []
    return not sources or tool == "get_file"


def deterministic_floor(task: str) -> dict:
    """Infer minimum intent requirements from the task text alone. Only
    the keys that actually have a floor are present in the result:

    - "answer_style_ceiling": the most detailed `answer_style` allowed
      (STYLE_SHORT when the task asks for something short/brief/concise) -
      a ceiling, not a minimum, because "short" is the tightest constraint
      on this axis, not the loosest.
    - "sources": the exact source list the task explicitly restricts work
      to (authoritative - never widened by the model).
    - "rigor": the minimum rigor (STANDARD or STRICT) required by
      verification language and/or a clear mutation request.
    - "kind": KIND_CHANGE when the task clearly asks to modify the
      repository - authoritative (the model must not report a mutation
      request as a mere "explain"/"check").

    Deliberately does NOT infer a `depth` floor: depth is left to the planner
    (and the default) rather than guessed from keywords.
    """
    text = task or ""
    floor: dict = {}

    if _SHORT_RE.search(text):
        floor["answer_style_ceiling"] = STYLE_SHORT

    source_match = _SOURCE_RE.search(text)
    if source_match:
        raw_source = next((group for group in source_match.groups() if group), "")
        normalized = normalize_source_path(raw_source)
        if normalized:
            floor["sources"] = [normalized]

    rigor_floor = RIGOR_STRICT if _SERIOUS_RE.search(text) else None

    if is_mutation_request(text):
        floor["kind"] = KIND_CHANGE
        # A mutation request must not stay "light" rigor, even if no
        # verification language was also present.
        rigor_floor = _max_enum(rigor_floor, RIGOR_STANDARD, _RIGOR_ORDER)

    if rigor_floor is not None:
        floor["rigor"] = rigor_floor

    if _requests_run_checks(text):
        floor["run_checks"] = True

    return floor


# -------------------------------------------------------------- resolution


def resolve_intent(task: str, model_intent: Optional[dict] = None, *, floor: Optional[dict] = None) -> Intent:
    """Combine the deterministic floor with the (optional) planner-proposed
    intent into the final Intent for this invocation.

    `floor` should be `deterministic_floor(task)`, computed by the caller
    BEFORE the planning LLM call (see deep_agent.py's `plan_task`) so that a
    planner failure can never lose it - passing it in here (rather than
    recomputing it) makes that ordering explicit, though recomputing it
    would give the same result since the floor depends only on `task`.

    The model may set any field freely and may raise depth/rigor above the
    floor, but can never lower it: an explicit source restriction and a
    mutation-implied "change" kind are authoritative and are never
    overridden; rigor is raised to the floor if the model proposed less;
    answer_style is capped at an explicit brevity ceiling if the model
    proposed more.
    """
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

    return {"kind": kind, "depth": depth, "rigor": rigor, "answer_style": answer_style, "sources": list(sources)}


# --------------------------------------------------------------- planning bounds

# PLANNED-step limits only (see the module docstring and deep_agent.py's
# `plan_task`) - this says nothing about, and does not by itself bound,
# actual Explorer/Coder tool calls or other downstream execution.
_DEPTH_STEP_LIMIT = {DEPTH_MINIMAL: 1, DEPTH_STANDARD: 3}

# Acceptance-criteria count limits from rigor (see evidence.py's MAX_CRITERIA
# for the absolute ceiling, used here for RIGOR_STRICT).
_RIGOR_CRITERIA_LIMIT = {RIGOR_LIGHT: 1, RIGOR_STANDARD: 3}


def plan_step_limit(intent: Intent, configured_max: int) -> int:
    """Max PENDING steps the planning call's output is clamped to, from
    `intent["depth"]`. DEPTH_DEEP uses the existing configured
    `max_plan_steps` unchanged; DEPTH_MINIMAL/STANDARD use the smaller
    fixed limits above (still never above `configured_max`)."""
    depth = intent.get("depth", DEPTH_STANDARD)
    if depth == DEPTH_DEEP:
        return max(1, configured_max)
    limit = _DEPTH_STEP_LIMIT.get(depth, _DEPTH_STEP_LIMIT[DEPTH_STANDARD])
    return max(1, min(configured_max, limit))


# Bound actual delegate/get_file turns separately from planned steps.
# Minimal-depth requests are capped; standard and deep requests keep the
# configured budget. Leave room for an initial failed or insufficient read
# and a follow-up investigation.
_DEPTH_AGENT_STEP_LIMIT = {DEPTH_MINIMAL: 4}


def agent_step_budget(intent: Intent, configured_max: int) -> int:
    """Max actual tool-using `decide` turns (`step_count`) this invocation
    may spend, from `intent["depth"]`. Only DEPTH_MINIMAL is clamped; every
    other depth (including an intent with no explicit depth at all) keeps
    `configured_max`, exactly as before this function existed."""
    if intent.get("depth") == DEPTH_MINIMAL:
        return max(1, min(configured_max, _DEPTH_AGENT_STEP_LIMIT[DEPTH_MINIMAL]))
    return max(1, configured_max)


def criteria_limit(intent: Intent, configured_max: int) -> int:
    """Max acceptance criteria, from `intent["rigor"]`. RIGOR_STRICT uses
    the existing configured max (`evidence.MAX_CRITERIA`) unchanged.

    A request whose `kind` is
    KIND_CHANGE is never clamped below the RIGOR_STANDARD limit, even if
    rigor somehow ended up RIGOR_LIGHT - a mutation/change task must not be
    accidentally reduced to a minimal, read-only-shaped criteria set.
    """
    rigor = intent.get("rigor", RIGOR_STANDARD)
    if rigor == RIGOR_STRICT:
        limit = configured_max
    else:
        limit = _RIGOR_CRITERIA_LIMIT.get(rigor, _RIGOR_CRITERIA_LIMIT[RIGOR_STANDARD])
    if intent.get("kind") == KIND_CHANGE:
        limit = max(limit, _RIGOR_CRITERIA_LIMIT[RIGOR_STANDARD])
    return max(1, min(configured_max, limit))
