"""
Token-budgeted, model-facing context management.

`bounded_model_history()` (utils.py) limits the NUMBER of retained turns, but a
single tool observation can hold a large source excerpt, so a request can still
exceed a provider's request-size / tokens-per-minute limit. This module bounds
the SIZE of the request that is actually sent, without touching run state:

  * `estimate_request_tokens` - a documented, deliberately conservative
    ESTIMATE, not a token count. Every figure derived from it is named `est_*`
    and must be read as approximate.
  * `fit_messages_to_budget` - selectively compacts the model-facing copy of a
    message list, oldest/least-relevant material first, stopping as soon as the
    request fits. Normal passes reduce only tool observations and older
    assistant prose. If normal compaction is insufficient, a bounded fallback
    compaction strategy further compacts the newest tool observation, auxiliary
    system-data status messages, and oldest complete AI-turn groups while
    preserving all system instructions, the user's current request, and
    tool-call / tool-result pairing integrity. If the essential content still
    exceeds the budget, `fits` is False and `essential_unfittable` is True.
  * `fit_evidence_items_to_budget` - bounds the serialized JSON representation
    of grader and conclusion-validation evidence items without mutating run
    state or enforcing a minimum excerpt size that exceeds the available budget.
  * `ContextBudgetExceededError` - raised before sending when a request remains
    unfittable after bounded compaction, so unfittable requests are never
    knowingly sent to the provider.
  * `is_context_size_error` / `retry_budget_for` - generic classification of a
    provider failure as "request too large" and a tighter budget for a bounded
    retry (used by `observability.invoke_model_logged`).

Pure helpers: only message attributes (`type`, `content`, `tool_calls`) are
read, so there are no LangChain imports here.
"""

from __future__ import annotations

import copy
import json
import math
import os
from typing import Any

from .config import (
    DEFAULT_CONTEXT_TOKEN_BUDGET,
    ConfigError,
    load_context_token_budget,
)
from .utils import (
    _bound_collections,
    _bound_get_file_result,
    _bound_symbol_result,
    _is_get_file_result,
    _is_get_symbol_result,
)

# ---- Estimation -------------------------------------------------------------
CHARS_PER_TOKEN = 2.5
MESSAGE_OVERHEAD_TOKENS = 6
REPLY_PRIMING_TOKENS = 3
PAYLOAD_REPLY_RESERVE_TOKENS = 900

# ---- Recovery ---------------------------------------------------------------
MAX_CONTEXT_RECOVERY_RETRIES = 2
MIN_RETRY_BUDGET_TOKENS = 500
PROVIDER_LIMIT_HEADROOM = 0.85
BLIND_RETRY_FRACTION = 0.6

# ---- Compaction limits (characters of one message's content) ---------------
_MID_CHARS = 1500
_SMALL_CHARS = 600
_MIN_CHARS = 160
_LAST_TOOL_CHARS = 3000
_LAST_TOOL_MIN_CHARS = 1200
_AI_TEXT_CHARS = 400
_AUX_STATUS_CHARS = 900
_AUX_STATUS_MIN_CHARS = 320
_JSON_LADDER = ((12, 1500), (8, 600), (4, 300), (2, 160), (1, 80))

EXCERPT_CLIP_MARKER = " ...[excerpt clipped to fit the size limit]"
_SHORT_CLIP_MARKER = " ...[clipped]"

_AUXILIARY_HUMAN_PREFIXES = (
    "Plan status data:\n",
    "Acceptance criteria and evidence data:\n",
    "Completion check (system data):",
    "Progress guard (system data):",
    "Strategy reassessment (system data):",
    "Repeat check (system data):",
    "Sufficiency check (system data):",
)
_ESSENTIAL_PROVENANCE_KEYS = (
    "source_tool",
    "report_kind",
    "file",
    "symbol",
    "line_start",
    "line_end",
    "total_lines",
    "truncated",
)


class ContextBudgetExceededError(RuntimeError):
    """Raised when a model request cannot fit within the context token budget
    even after bounded fallback compaction, preventing an unfittable request
    from being sent."""

    def __init__(self, message: str, fit_report: dict[str, Any] | None = None):
        super().__init__(message)
        self.status_code = 413
        self.error_category = "context_budget_exceeded"
        self.fit_report = dict(fit_report or {})


class FittedEvidenceList(list):
    """A standard list of model-facing evidence dicts carrying fit diagnostics."""

    def __init__(self, items: list[dict], report: dict[str, Any] | None = None):
        super().__init__(items)
        self.report: dict[str, Any] = dict(report or {})
        self.fits: bool = bool(self.report.get("fits", True))
        self.essential_unfittable: bool = bool(self.report.get("essential_unfittable", not self.fits))


def get_context_token_budget() -> int:
    """Budget for one model request, in estimated tokens (MODEL_CONTEXT_TOKEN_BUDGET)."""
    try:
        return load_context_token_budget(os.environ)
    except ConfigError:
        return DEFAULT_CONTEXT_TOKEN_BUDGET


# ---- Message accessors (LangChain objects or role/content dicts) -----------
def _kind(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("role") or message.get("type") or "")
    return str(getattr(message, "type", "") or "")


def _content(message: Any) -> Any:
    return message.get("content", "") if isinstance(message, dict) else getattr(message, "content", "")


def _text(message: Any) -> str:
    content = _content(message)
    if isinstance(content, list):  # content blocks
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return str(content or "")


def _tool_calls(message: Any) -> Any:
    return message.get("tool_calls") if isinstance(message, dict) else getattr(message, "tool_calls", None)


def _with_content(message: Any, content: str) -> Any:
    """A copy of `message` with new text content; every other field (ids,
    tool_call_id, tool_calls, name) is preserved. The original is untouched."""
    if isinstance(message, dict):
        return {**message, "content": content}
    try:
        return message.model_copy(update={"content": content})
    except AttributeError:
        clone = copy.copy(message)
        clone.content = content
        return clone


# ---- Estimator --------------------------------------------------------------
def estimate_text_tokens(text: str) -> int:
    """Approximate (over-)estimate of tokens in `text`; never an exact count."""
    if not text:
        return 0
    ascii_chars = len(text.encode("ascii", "ignore"))
    return math.ceil(ascii_chars / CHARS_PER_TOKEN) + (len(text) - ascii_chars)


def estimate_message_tokens(message: Any) -> int:
    tokens = MESSAGE_OVERHEAD_TOKENS + estimate_text_tokens(_text(message))
    calls = _tool_calls(message)
    if calls:
        try:
            tokens += estimate_text_tokens(json.dumps(calls, ensure_ascii=False, default=str))
        except (TypeError, ValueError):
            pass
    return tokens


def estimate_request_tokens(messages: list, fixed_tokens: int = 0) -> int:
    """Conservative estimate of the request: all message text + tool calls +
    per-message overhead + `fixed_tokens` (e.g. bound tool schemas)."""
    return fixed_tokens + REPLY_PRIMING_TOKENS + sum(estimate_message_tokens(m) for m in messages)


def _serialize_bound_tool_for_estimate(tool: Any) -> Any:
    if isinstance(tool, dict):
        return tool
    name = getattr(tool, "name", None)
    desc = getattr(tool, "description", None)
    schema_cls = getattr(tool, "args_schema", None)
    if name or desc or schema_cls is not None:
        schema_dict = {}
        if schema_cls is not None and hasattr(schema_cls, "model_json_schema"):
            try:
                schema_dict = schema_cls.model_json_schema()
            except Exception:
                schema_dict = {}
        return {"name": name or str(tool), "description": desc or "", "parameters": schema_dict}
    return str(tool)


def estimate_tool_schema_tokens(model: Any) -> int:
    """Tokens taken by tool schemas bound to the model (they are sent with every
    request but are not in the message list). 0 when none can be found."""
    try:
        kwargs = getattr(model, "kwargs", None)
        tools = kwargs.get("tools") if isinstance(kwargs, dict) else None
        if not tools:
            return 0
        serialized = [_serialize_bound_tool_for_estimate(t) for t in tools]
        return estimate_text_tokens(json.dumps(serialized, ensure_ascii=False, default=str))
    except Exception:
        return 0


# ---- Compaction of one piece of text ---------------------------------------
def _clip_text(text: str, limit: int) -> str:
    """Head+tail clip with an explicit marker (never silent)."""
    if len(text) <= limit:
        return text
    marker = f"\n...[{len(text) - limit} characters omitted to fit the model context budget]...\n"
    room = max(limit - len(marker), 0)
    head = int(room * 0.65)
    tail = room - head
    return text[:head] + marker + (text[len(text) - tail:] if tail else "")


def _shrink_strings(value: Any, max_str: int) -> Any:
    if isinstance(value, str):
        if len(value) <= max_str:
            return value
        return f"{value[:max_str]}...[+{len(value) - max_str} chars omitted]"
    if isinstance(value, list):
        return [_shrink_strings(v, max_str) for v in value]
    if isinstance(value, dict):
        return {k: _shrink_strings(v, max_str) for k, v in value.items()}
    return value


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _skeleton(data: Any, limit: int, note: str, original: int) -> dict:
    """Smallest-structure fallback: a VALID dict that keeps whichever top-level
    fields are short and names every field it left out with its type and size."""
    base: dict = {"context_compacted": note, "original_chars": original}
    if isinstance(data, dict):
        fields = list(data.items())
        kind = "fields"
    else:
        fields = [(f"[{i}]", v) for i, v in enumerate(data)]
        kind = "items"
    base[f"{kind}_total"] = len(fields)
    kept: dict = {}
    omitted: list = []
    used = len(_dump(base)) + 40
    for name, value in sorted(fields, key=lambda kv: len(_dump(kv[1]))):
        size = len(_dump(value))
        if used + size + len(_dump(name)) + 4 <= limit:
            kept[name] = value
            used += size + len(_dump(name)) + 4
        else:
            omitted.append(name)
    out = dict(base)
    if kept:
        out["kept"] = {n: kept[n] for n, _ in fields if n in kept}
    sizes = {n: (type(v).__name__, len(_dump(v))) for n, v in fields if n in set(omitted)}
    out["omitted"] = [f"{n} ({t}, {c} chars)" for n, (t, c) in list(sizes.items())[:20]]
    if len(sizes) > 20:
        out["omitted_more"] = len(sizes) - 20
    if len(_dump(out)) > limit:
        out.pop("omitted", None)
        out.pop("omitted_more", None)
        out["omitted_count"] = len(omitted)
    return out


def _compact_get_file(data: dict, limit: int, note: str) -> str:
    """get_file result: shrink non-content fields progressively, cut `content`
    at whole-line boundaries and rewrite line_end/next_start_line to match
    exactly the lines kept."""
    reserved = ("content", "line_start", "line_end", "next_start_line")
    out = ""
    for max_items, max_str in _JSON_LADDER:
        base = {k: (v if k in reserved else _shrink_strings(_bound_collections(v, max_items), max_str))
                for k, v in data.items()}
        bounded = _bound_get_file_result(base, max(limit - len(note) - 40, 0))
        bounded["context_compacted"] = note
        out = _dump(bounded)
        if len(out) <= limit:
            break
    return out


def _compact_get_symbol(data: dict, limit: int, note: str) -> str:
    """get_symbol result: cut `source` at whole-line boundaries and update
    `line_end` to match delivered lines."""
    reserved = ("source", "line_start", "line_end")
    out = ""
    for max_items, max_str in _JSON_LADDER:
        base = {k: (v if k in reserved else _shrink_strings(_bound_collections(v, max_items), max_str))
                for k, v in data.items()}
        bounded = _bound_symbol_result(base, max(limit - len(note) - 40, 0))
        bounded["context_compacted"] = note
        out = _dump(bounded)
        if len(out) <= limit:
            break
    return out


def compact_observation(text: str, limit: int) -> str:
    """Reduce one tool observation to about `limit` characters while keeping
    valid JSON structure."""
    if len(text) <= limit:
        return text
    original = len(text)
    note = (f"reduced from {original} chars to fit the context budget; "
            "full result kept in run state, not shown to the model")
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        return _clip_text(text, limit)
    if isinstance(data, str):
        return _dump(_shrink_strings(data, max(limit - len(note) - 20, 40)) + f" [{note}]")
    if not isinstance(data, (dict, list)):
        return _dump({"context_compacted": note, "value": data})

    if _is_get_file_result(data):
        return _compact_get_file(data, limit, note)
    if _is_get_symbol_result(data):
        return _compact_get_symbol(data, limit, note)
    for max_items, max_str in _JSON_LADDER:
        shrunk = _shrink_strings(_bound_collections(data, max_items), max_str)
        shrunk = ({**shrunk, "context_compacted": note} if isinstance(shrunk, dict)
                  else {"items": shrunk, "context_compacted": note})
        out = _dump(shrunk)
        if len(out) <= limit:
            return out
    skeleton = _dump(_skeleton(data, limit, note, original))
    if len(skeleton) <= limit:
        return skeleton
    return _dump({"context_compacted": note, "original_chars": original, "truncated": True})


# ---- Fitting serialized evidence payloads (grader & conclusion check) ------
def _clip_excerpt_to_cap(text: str, cap: int) -> str:
    """Clip `text` so the returned string (including any marker) has length <= `cap`.
    Never enforces a minimum excerpt size that exceeds `cap`."""
    if cap <= 0:
        return ""
    if len(text) <= cap:
        return text
    if cap > len(EXCERPT_CLIP_MARKER):
        keep = cap - len(EXCERPT_CLIP_MARKER)
        return text[:keep] + EXCERPT_CLIP_MARKER
    if cap > len(_SHORT_CLIP_MARKER):
        keep = cap - len(_SHORT_CLIP_MARKER)
        return text[:keep] + _SHORT_CLIP_MARKER
    if cap >= 3:
        return text[: cap - 3] + "..."
    return ""


def prepare_model_evidence_items(items: list[dict]) -> tuple[list[dict], int]:
    """Build model-facing copies of evidence items without mutating run state.

    Preserves item IDs, actions, results, paths, line ranges, provenance, and
    status metadata while stripping any redundant raw observation dumps from the
    model-facing provenance copy.
    """
    prepared: list[dict] = []
    compacted_count = 0

    for item in items:
        if not isinstance(item, dict):
            continue
        clone = dict(item)
        raw_prov = item.get("provenance")
        if isinstance(raw_prov, dict):
            prov = dict(raw_prov)
            if "raw_observations" in prov:
                prov.pop("raw_observations", None)
                compacted_count += 1
            clone["provenance"] = prov
        prepared.append(clone)
    return prepared, compacted_count


def _slim_provenance_items(items: list[dict]) -> list[dict]:
    """Bounded fallback for model-facing evidence items when full provenance
    dicts alone exceed the available budget: retain only core scalar provenance
    keys while keeping item IDs, actions, results, and excerpts intact."""
    slimmed: list[dict] = []
    for item in items:
        clone = dict(item)
        prov = clone.get("provenance")
        if isinstance(prov, dict):
            clone["provenance"] = {
                k: prov[k] for k in _ESSENTIAL_PROVENANCE_KEYS if k in prov
            }
        slimmed.append(clone)
    return slimmed


def fit_evidence_items_to_budget(
    items: list[dict],
    fixed_chars: int,
    scale: float = 1.0,
    *,
    budget_tokens: int | None = None,
) -> tuple[FittedEvidenceList, dict[str, Any]]:
    """Bound the actual serialized JSON size of `items` within the token budget.

    Accounts for JSON escaping, metadata, list separators, clipping markers,
    and message framing overhead. Never mutates `items` and never enforces a
    minimum excerpt length that exceeds the available budget. If normal excerpt
    compaction is insufficient because provenance dicts are large, applies a
    bounded fallback that slims model-facing provenance dicts before declaring
    the payload unfittable.
    """
    raw_budget_tokens = budget_tokens if budget_tokens is not None else get_context_token_budget()
    framing_tokens = 2 * MESSAGE_OVERHEAD_TOKENS + REPLY_PRIMING_TOKENS + 2
    if raw_budget_tokens > framing_tokens:
        reserve_tokens = min(
            PAYLOAD_REPLY_RESERVE_TOKENS,
            max(int(raw_budget_tokens * 0.15), framing_tokens),
        )
    else:
        reserve_tokens = 0
    eff_scale = max(0.05, float(scale))
    target_request_tokens = max(0, int((raw_budget_tokens - reserve_tokens) * eff_scale))
    text_budget_tokens = max(0, target_request_tokens - framing_tokens)
    allowed_total_chars = max(0, int(text_budget_tokens * CHARS_PER_TOKEN))

    clean_items, deduped_count = prepare_model_evidence_items(items)
    raw_serialized_before = len(json.dumps(items, ensure_ascii=True))
    clean_serialized = json.dumps(clean_items, ensure_ascii=True)
    total_before_chars = max(0, int(fixed_chars)) + raw_serialized_before
    est_before_tokens = framing_tokens + math.ceil(total_before_chars / CHARS_PER_TOKEN)
    est_fixed_tokens = framing_tokens + math.ceil(max(0, int(fixed_chars)) / CHARS_PER_TOKEN)

    total_clean_chars = max(0, int(fixed_chars)) + len(clean_serialized)
    if total_clean_chars <= allowed_total_chars:
        est_after_tokens = framing_tokens + math.ceil(total_clean_chars / CHARS_PER_TOKEN)
        report = {
            "est_budget_tokens": target_request_tokens,
            "est_context_budget_tokens": raw_budget_tokens,
            "est_before_tokens": est_before_tokens,
            "est_after_tokens": est_after_tokens,
            "est_fixed_tokens": est_fixed_tokens,
            "budget_chars": allowed_total_chars,
            "serialized_chars": total_clean_chars,
            "fixed_chars": int(fixed_chars),
            "compacted_items": deduped_count,
            "fits": True,
            "essential_unfittable": False,
        }
        return FittedEvidenceList(clean_items, report=report), report

    working_items = clean_items
    empty_items = [dict(i, evidence="") for i in working_items]
    min_serialized = json.dumps(empty_items, ensure_ascii=True)
    min_total_chars = max(0, int(fixed_chars)) + len(min_serialized)
    if min_total_chars > allowed_total_chars:
        # Bounded fallback: slim model-facing provenance metadata before declaring unfittable.
        working_items = _slim_provenance_items(clean_items)
        empty_items = [dict(i, evidence="") for i in working_items]
        min_serialized = json.dumps(empty_items, ensure_ascii=True)
        min_total_chars = max(0, int(fixed_chars)) + len(min_serialized)

    if min_total_chars > allowed_total_chars:
        est_after_tokens = framing_tokens + math.ceil(min_total_chars / CHARS_PER_TOKEN)
        report = {
            "est_budget_tokens": target_request_tokens,
            "est_context_budget_tokens": raw_budget_tokens,
            "est_before_tokens": est_before_tokens,
            "est_after_tokens": est_after_tokens,
            "est_fixed_tokens": est_fixed_tokens,
            "budget_chars": allowed_total_chars,
            "serialized_chars": min_total_chars,
            "fixed_chars": int(fixed_chars),
            "compacted_items": len(working_items),
            "fits": False,
            "essential_unfittable": True,
            "reason": (
                "essential metadata (evidence IDs, actions, provenance, criteria) and fixed prompt "
                "instructions exceed the available payload budget even with empty excerpts"
            ),
        }
        return FittedEvidenceList(empty_items, report=report), report

    texts = [str(i.get("evidence") or "") for i in working_items]

    def build_candidate(cap: int) -> list[dict]:
        return [
            dict(item, evidence=_clip_excerpt_to_cap(text, cap))
            for item, text in zip(working_items, texts)
        ]

    low, high = 0, max((len(t) for t in texts), default=0)
    while low < high:
        mid = (low + high + 1) // 2
        cand_chars = max(0, int(fixed_chars)) + len(json.dumps(build_candidate(mid), ensure_ascii=True))
        if cand_chars <= allowed_total_chars:
            low = mid
        else:
            high = mid - 1

    cap = low
    fitted = build_candidate(cap)
    serialized_chars = max(0, int(fixed_chars)) + len(json.dumps(fitted, ensure_ascii=True))
    while cap > 0 and serialized_chars > allowed_total_chars:
        cap = max(0, cap - max(1, (serialized_chars - allowed_total_chars)))
        fitted = build_candidate(cap)
        serialized_chars = max(0, int(fixed_chars)) + len(json.dumps(fitted, ensure_ascii=True))

    clipped_count = sum(1 for orig, cur in zip(items, fitted) if orig.get("evidence") != cur.get("evidence"))
    est_after_tokens = framing_tokens + math.ceil(serialized_chars / CHARS_PER_TOKEN)
    fits = serialized_chars <= allowed_total_chars and est_after_tokens <= raw_budget_tokens
    report = {
        "est_budget_tokens": target_request_tokens,
        "est_context_budget_tokens": raw_budget_tokens,
        "est_before_tokens": est_before_tokens,
        "est_after_tokens": est_after_tokens,
        "est_fixed_tokens": est_fixed_tokens,
        "budget_chars": allowed_total_chars,
        "serialized_chars": serialized_chars,
        "fixed_chars": int(fixed_chars),
        "compacted_items": max(clipped_count, deduped_count),
        "excerpt_cap_chars": cap,
        "fits": fits,
        "essential_unfittable": not fits,
    }
    if not fits:
        report["reason"] = "serialized payload still exceeds target budget after maximum excerpt compaction"
    return FittedEvidenceList(fitted, report=report), report


# ---- Fitting a message list to a budget ------------------------------------
def _is_auxiliary_status_human(message: Any) -> bool:
    """True only for runtime-injected system-data human messages (plan status,
    criteria digest, guard feedback, or wrap-up evidence block),
    never the user's original query."""
    if _kind(message) not in ("human", "user"):
        return False
    text = _text(message)
    if text.startswith(_AUXILIARY_HUMAN_PREFIXES):
        return True
    return "\n\nEvidence gathered so far:\n" in text and text.startswith("The investigation reached its step limit")


def _compact_auxiliary_human_text(text: str, limit: int) -> str:
    """Compact an auxiliary status/wrap-up human message while keeping its
    essential header and question intact."""
    if len(text) <= limit:
        return text
    if "\n\nEvidence gathered so far:\n" in text:
        head, sep, evidence_part = text.partition("\n\nEvidence gathered so far:\n")
        avail = max(80, limit - len(head) - len(sep))
        return head + sep + _clip_text(evidence_part, avail)
    return _clip_text(text, limit)


def _drop_oldest_turn_groups(messages: list, budget_tokens: int, fixed_tokens: int) -> tuple[list, int]:
    """Bounded fallback: drop the oldest complete AI-turn groups (AI message +
    paired tool results + intermediate guard/feedback human messages before the
    next AI turn) while preserving all system messages, leading human messages
    (user task and current-turn status data), and at least the latest AI turn
    group so tool-call / tool-result integrity is never broken."""
    ai_indices = [i for i, m in enumerate(messages) if _kind(m) in ("ai", "assistant")]
    if len(ai_indices) <= 1:
        return messages, 0

    first_ai = ai_indices[0]
    prefix = messages[:first_ai]
    groups: list[list[Any]] = []
    for idx_pos, start_idx in enumerate(ai_indices):
        end_idx = ai_indices[idx_pos + 1] if idx_pos + 1 < len(ai_indices) else len(messages)
        groups.append(messages[start_idx:end_idx])

    dropped = 0
    kept_groups = list(groups)
    while len(kept_groups) > 1:
        candidate = [*prefix, *(m for grp in kept_groups for m in grp)]
        if estimate_request_tokens(candidate, fixed_tokens) <= budget_tokens:
            return candidate, dropped
        kept_groups.pop(0)
        dropped += 1

    final_candidate = [*prefix, *(m for grp in kept_groups for m in grp)]
    return final_candidate, dropped


def fit_messages_to_budget(messages: list, budget_tokens: int, fixed_tokens: int = 0) -> tuple[list, dict]:
    """Return (model-facing messages, report). Never mutates `messages`.

    Normal passes reduce tool observations (oldest first, newest last) and the
    prose of older assistant messages. If the request still exceeds
    `budget_tokens`, a bounded fallback strategy compacts the newest tool
    observation further, compacts runtime-injected auxiliary status messages
    (never the user's task or system instructions), and drops the oldest
    complete AI+tool turn groups while keeping at least the latest turn group.

    If the request still exceeds the budget after bounded fallback compaction,
    the report has `fits` False and `essential_unfittable` True.
    """
    out = list(messages)
    before = estimate_request_tokens(out, fixed_tokens)
    report: dict[str, Any] = {
        "est_budget_tokens": budget_tokens,
        "est_before_tokens": before,
        "est_after_tokens": before,
        "est_fixed_tokens": fixed_tokens,
        "compacted_messages": 0,
        "fits": before <= budget_tokens,
        "essential_unfittable": False,
    }
    if before <= budget_tokens:
        return out, report

    tool_ix = [i for i, m in enumerate(out) if _kind(m) == "tool"]
    last = tool_ix[-1] if tool_ix else None
    recent = set(tool_ix[-2:])
    old = [i for i in tool_ix if i not in recent]
    near = [i for i in tool_ix if i in recent and i != last]
    newest = [last] if last is not None else []
    ai_ix = [i for i, m in enumerate(out) if _kind(m) in ("ai", "assistant")]
    old_ai = ai_ix[:-1]  # the latest assistant message is never touched
    passes = [
        (old, _MID_CHARS, True), (old, _SMALL_CHARS, True), (old_ai, _AI_TEXT_CHARS, False),
        (near, _MID_CHARS, True), (newest, _LAST_TOOL_CHARS, True),
        (old, _MIN_CHARS, True), (near, _SMALL_CHARS, True), (newest, _LAST_TOOL_MIN_CHARS, True),
        (near, _MIN_CHARS, True),
    ]
    total = before
    touched: set[int] = set()
    for indices, limit, structured in passes:
        for i in indices:
            if total <= budget_tokens:
                break
            text = _text(out[i])
            if len(text) <= limit:
                continue
            new_text = compact_observation(text, limit) if structured else _clip_text(text, limit)
            if len(new_text) >= len(text):
                continue
            replacement = _with_content(out[i], new_text)
            total += estimate_message_tokens(replacement) - estimate_message_tokens(out[i])
            out[i] = replacement
            touched.add(i)
        if total <= budget_tokens:
            break

    # Bounded fallback compaction before declaring the request unfittable.
    if total > budget_tokens:
        fallback_passes = [
            (newest, _SMALL_CHARS, True),
            (newest, _MIN_CHARS, True),
        ]
        for indices, limit, structured in fallback_passes:
            for i in indices:
                if total <= budget_tokens:
                    break
                text = _text(out[i])
                if len(text) <= limit:
                    continue
                new_text = compact_observation(text, limit) if structured else _clip_text(text, limit)
                if len(new_text) >= len(text):
                    continue
                replacement = _with_content(out[i], new_text)
                total += estimate_message_tokens(replacement) - estimate_message_tokens(out[i])
                out[i] = replacement
                touched.add(i)
            if total <= budget_tokens:
                break

    if total > budget_tokens:
        aux_ix = [i for i, m in enumerate(out) if _is_auxiliary_status_human(m)]
        for aux_limit in (_AUX_STATUS_CHARS, _AUX_STATUS_MIN_CHARS):
            for i in aux_ix:
                if total <= budget_tokens:
                    break
                text = _text(out[i])
                if len(text) <= aux_limit:
                    continue
                new_text = _compact_auxiliary_human_text(text, aux_limit)
                if len(new_text) >= len(text):
                    continue
                replacement = _with_content(out[i], new_text)
                total += estimate_message_tokens(replacement) - estimate_message_tokens(out[i])
                out[i] = replacement
                touched.add(i)
            if total <= budget_tokens:
                break

    if total > budget_tokens:
        trimmed_out, dropped_groups = _drop_oldest_turn_groups(out, budget_tokens, fixed_tokens)
        if dropped_groups > 0:
            out = trimmed_out
            total = estimate_request_tokens(out, fixed_tokens)
            report["dropped_turn_groups"] = dropped_groups

    fits = total <= budget_tokens
    report.update(
        est_after_tokens=total,
        compacted_messages=len(touched) + int(report.get("dropped_turn_groups", 0)),
        fits=fits,
        essential_unfittable=not fits,
    )
    if not fits:
        report["reason"] = (
            "essential content (system and human messages, tool-call arguments, latest "
            "assistant turn) plus tool schemas is still over the estimated budget after "
            "compacting every compressible observation; it was not truncated"
        )
    return out, report


# ---- Provider error handling ------------------------------------------------
def is_context_size_error(err_info: dict) -> bool:
    """True when `analyze_provider_error` classified a failure as an oversized
    request / context-window overflow."""
    category = err_info.get("error_category")
    if category in ("provider_request_too_large", "provider_context_window_limit"):
        return True
    return category == "provider_tpm_limit" and err_info.get("http_status") == 413


def retry_budget_for(current_tokens: int, err_info: dict) -> int:
    """A strictly smaller budget for the retry."""
    limit, seen = err_info.get("tpm_limit"), err_info.get("requested_tokens")
    if isinstance(limit, int) and isinstance(seen, int) and 0 < limit < seen:
        target = int(current_tokens * (limit * PROVIDER_LIMIT_HEADROOM) / seen)
    else:
        target = int(current_tokens * BLIND_RETRY_FRACTION)
    return max(min(target, int(current_tokens * 0.9)), MIN_RETRY_BUDGET_TOKENS)
