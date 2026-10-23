"""Runtime trace context and diagnostic event logging."""

from __future__ import annotations

import contextvars
from dataclasses import dataclass, field
import json
import os
import re
import time
import uuid
from typing import Any, Callable


_MAX_ARG_CHARS = 160
_MAX_SUMMARY_CHARS = 200
_MAX_ERROR_CHARS = 280

_SECRET_KEY_RE = re.compile(
    r"(?:api[_-]?key|secret|token|password|passwd|credential|authorization|auth[_-]?header|private[_-]?key)",
    re.IGNORECASE,
)
_INLINE_SECRET_PATTERNS = (
    (re.compile(r"\bAIza[0-9A-Za-z_-]{30,}\b"), "[REDACTED]"),
    (re.compile(r"\bgsk_[A-Za-z0-9]{12,}\b"), "[REDACTED]"),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"), "[REDACTED]"),
    (re.compile(r"(?i)\b(Bearer\s+)[A-Za-z0-9._\-+/=]{8,}"), r"\1[REDACTED]"),
    (
        re.compile(
            r"(?i)\b(api[_-]?key|secret|token|password|passwd|authorization)\b(\s*[:=]\s*['\"]?)([^\s'\",;}\]]+)"
        ),
        r"\1\2[REDACTED]",
    ),
)
_TPM_LIMIT_RE_1 = re.compile(r"Limit\s+(\d+)\s*,\s*Requested\s+(\d+)", re.IGNORECASE)
_TPM_LIMIT_RE_2 = re.compile(r"Requested\s+(\d+)\s*,\s*Limit\s+(\d+)", re.IGNORECASE)
_HTTP_STATUS_RE = re.compile(r"\b(?:HTTP|Error\s+code:?|status[_ ]code[=:]?)\s*(\d{3})\b", re.IGNORECASE)
_GUARD_PREFIX_RE = re.compile(r"^[A-Za-z_ ]+\(system data\):\s*", re.IGNORECASE)


@dataclass
class RunTraceContext:
    run_id: str
    on_line: Callable[[str], None] | None = None
    api_key: str | None = None
    started_at: float = field(default_factory=time.perf_counter)

    # Current execution context stack
    current_agent: str = "Main"
    parent_agent: str | None = None
    delegation_tool: str | None = None
    current_stage: str = "init"
    current_step_id: str | None = None
    current_tool_call_id: str | None = None
    current_tool_target: str | None = None
    pending_tool_call_ids: dict[str, list[str]] = field(default_factory=dict)
    active_tool_kwargs: list[tuple[str, str, dict[str, Any]]] = field(default_factory=list)

    # Attribution counters
    agent_llm_calls: dict[str, int] = field(default_factory=dict)
    agent_tool_calls: dict[str, dict[str, int]] = field(default_factory=dict)
    agent_tool_outcomes: dict[str, dict[str, int]] = field(default_factory=dict)
    stage_details: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Failure & lifecycle records
    model_failures: list[dict[str, Any]] = field(default_factory=list)
    context_events: list[dict[str, Any]] = field(default_factory=list)
    tool_failures: list[dict[str, Any]] = field(default_factory=list)
    subagent_runs: list[dict[str, Any]] = field(default_factory=list)
    grading_events: list[dict[str, Any]] = field(default_factory=list)
    conclusion_events: list[dict[str, Any]] = field(default_factory=list)
    guard_events: list[dict[str, Any]] = field(default_factory=list)
    completion_diagnostics: dict[str, Any] | None = None


_CURRENT_CONTEXT: contextvars.ContextVar[RunTraceContext | None] = contextvars.ContextVar(
    "ari_run_trace_context", default=None
)


def start_run_context(
    on_line: Callable[[str], None] | None = None,
    api_key: str | None = None,
    run_id: str | None = None,
) -> tuple[RunTraceContext, contextvars.Token]:
    """Create and activate a new correlated RunTraceContext."""
    ctx = RunTraceContext(
        run_id=run_id or f"run_{uuid.uuid4().hex[:10]}",
        on_line=on_line,
        api_key=api_key,
    )
    token = _CURRENT_CONTEXT.set(ctx)
    return ctx, token


def ensure_run_context(api_key: str | None = None) -> RunTraceContext:
    """Return the active RunTraceContext, creating a local one if invoked outside run_query."""
    ctx = _CURRENT_CONTEXT.get()
    if ctx is None:
        ctx = RunTraceContext(run_id=f"run_{uuid.uuid4().hex[:10]}", on_line=None, api_key=api_key)
        _CURRENT_CONTEXT.set(ctx)
    elif api_key and not ctx.api_key:
        ctx.api_key = api_key
    return ctx


def get_run_context() -> RunTraceContext | None:
    return _CURRENT_CONTEXT.get()


def reset_run_context(token: contextvars.Token) -> None:
    try:
        _CURRENT_CONTEXT.reset(token)
    except Exception:
        pass


class ScopeGuard:
    """Context manager to push/pop agent, stage, step_id, or tool_call_id safely."""

    def __init__(
        self,
        *,
        agent: str | None = None,
        parent_agent: str | None = None,
        delegation_tool: str | None = None,
        stage: str | None = None,
        step_id: str | None = None,
        tool_call_id: str | None = None,
    ):
        self.updates = {
            "current_agent": agent,
            "parent_agent": parent_agent,
            "delegation_tool": delegation_tool,
            "current_stage": stage,
            "current_step_id": step_id,
            "current_tool_call_id": tool_call_id,
        }
        self._saved: dict[str, Any] = {}
        self._ctx: RunTraceContext | None = None

    def __enter__(self) -> RunTraceContext | None:
        try:
            self._ctx = _CURRENT_CONTEXT.get()
            if self._ctx is not None:
                for key, val in self.updates.items():
                    if val is not None:
                        self._saved[key] = getattr(self._ctx, key)
                        setattr(self._ctx, key, val)
        except Exception:
            pass
        return self._ctx

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        try:
            if self._ctx is not None:
                for key, val in self._saved.items():
                    setattr(self._ctx, key, val)
        except Exception:
            pass
        return False


def emit_trace(line: str, *, fallback_stdout: bool = False) -> None:
    """Emit one diagnostic line to the active trace callback without raising."""
    try:
        ctx = _CURRENT_CONTEXT.get()
        clean = redact_secrets(line, ctx.api_key if ctx else None)
        if ctx is not None and ctx.on_line is not None:
            ctx.on_line(clean)
        elif fallback_stdout:
            print(clean)
    except Exception:
        pass


def redact_secrets(text: str, api_key: str | None = None) -> str:
    """Redact API keys, tokens, and secret-bearing values from a log string."""
    try:
        out = str(text or "")
        if api_key and len(api_key.strip()) >= 4:
            out = out.replace(api_key.strip(), "[REDACTED]")
        for env_var in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GROQ_API_KEY"):
            env_key = os.environ.get(env_var, "").strip()
            if env_key and len(env_key) >= 4:
                out = out.replace(env_key, "[REDACTED]")
        for pattern, replacement in _INLINE_SECRET_PATTERNS:
            out = pattern.sub(replacement, out)
        return out
    except Exception:
        return "[redaction_error]"


def _sanitize_arg_value(key: str, value: Any, api_key: str | None = None) -> Any:
    if _SECRET_KEY_RE.search(str(key)):
        return "[REDACTED]"
    if str(key) == "content" and isinstance(value, str):
        return f"<content: {len(value)} chars, {len(value.splitlines())} lines>"
    if isinstance(value, str):
        redacted = redact_secrets(value, api_key)
        return redacted if len(redacted) <= 120 else redacted[:117] + "..."
    if isinstance(value, dict):
        return {k: _sanitize_arg_value(str(k), v, api_key) for k, v in list(value.items())[:10]}
    if isinstance(value, list):
        items = [_sanitize_arg_value(key, item, api_key) for item in value[:6]]
        if len(value) > 6:
            items.append(f"...(+{len(value) - 6} more)")
        return items
    return value


def format_tool_args(args: Any, api_key: str | None = None, max_chars: int = _MAX_ARG_CHARS) -> str:
    """Serialize tool arguments safely with redaction and length bounding."""
    try:
        if isinstance(args, dict):
            safe = {str(k): _sanitize_arg_value(str(k), v, api_key) for k, v in args.items()}
            text = json.dumps(safe, ensure_ascii=False, separators=(",", ":"))
        else:
            text = redact_secrets(str(args), api_key)
        if len(text) > max_chars:
            return text[: max_chars - 3] + "..."
        return text
    except Exception:
        return "{}"


def _format_error_detail(err: Any, api_key: str | None = None) -> str:
    if isinstance(err, dict):
        code = err.get("code")
        msg = redact_secrets(str(err.get("message") or ""), api_key)
        return f"{code}: {msg}" if code and msg else (str(code) if code else msg or "unknown error")
    return redact_secrets(str(err or "unknown error"), api_key)[:200]


def summarize_tool_observation(
    tool_name: str, raw_result: Any, observation_text: str, api_key: str | None = None
) -> dict[str, Any]:
    """Classify and summarize a tool output without mutating it."""
    obs_str = str(observation_text or "")
    obs_truncated = "... [truncated:" in obs_str or "...[report truncated]" in obs_str
    status = "ok"
    summary_parts: list[str] = []

    data = None
    if obs_str:
        try:
            parsed = json.loads(obs_str)
            if isinstance(parsed, dict):
                data = parsed
        except (TypeError, ValueError):
            data = None
    if data is None and isinstance(raw_result, dict):
        data = raw_result

    if obs_str.startswith("Invalid tool arguments:"):
        status = "failed"
        summary_parts.append(obs_str[:160])
    elif isinstance(data, dict):
        if "error" in data:
            err = data.get("error")
            err_code = err.get("code") if isinstance(err, dict) else None
            if tool_name == "find_symbol" and err_code == "symbol_not_found":
                status = "empty"
                msg = redact_secrets(str(err.get("message") or "no matching symbol"), api_key)
                sugs = err.get("suggestions") if isinstance(err, dict) else None
                if isinstance(sugs, list) and sugs:
                    msg += f" (suggestions: {', '.join(map(str, sugs[:3]))})"
                summary_parts.append(msg)
            else:
                status = "failed"
                summary_parts.append(_format_error_detail(err, api_key))
        else:
            if data.get("truncated") or data.get("observation_truncated"):
                obs_truncated = True
            if tool_name == "get_file" or ("file" in data and "content" in data and "line_start" in data):
                content = data.get("content")
                if not isinstance(content, str) or not content.strip():
                    status = "inconclusive" if data.get("truncated") else "empty"
                    summary_parts.append("empty file or range")
                else:
                    status = "ok"
                    summary_parts.append("OK (truncated)" if obs_truncated else "OK")
            elif tool_name == "find_files" or ("files" in data and "pattern" in data):
                files = data.get("files")
                shown = (
                    files.get("shown", len(files.get("items", [])))
                    if isinstance(files, dict)
                    else (len(files) if isinstance(files, list) else 0)
                )
                total = data.get("total_matches", shown)
                status = "ok" if shown > 0 else "empty"
                summary_parts.append(
                    f"{total} file" if total == 1 else (f"{total} files" if shown > 0 else "no matching files")
                )
            elif tool_name == "find_symbol" or ("matches" in data and "query" in data):
                matches = data.get("matches")
                items = (
                    matches.get("items", [])
                    if isinstance(matches, dict)
                    else (matches if isinstance(matches, list) else [])
                )
                count = data.get("count", len(items)) if isinstance(data.get("count"), int) else len(items)
                status = "ok" if count > 0 else "empty"
                if count == 0:
                    summary_parts.append("no matching symbol")
                elif count == 1 and items and isinstance(items[0], dict) and items[0].get("file"):
                    m = items[0]
                    loc = (
                        f"{m['file']}:{m.get('line_start')}-{m.get('line_end')}"
                        if m.get("line_start") is not None
                        else str(m["file"])
                    )
                    summary_parts.append(f"1 match ({loc})")
                else:
                    summary_parts.append(f"{count} matches")
            elif tool_name == "search_repository" or ("results" in data and "query" in data):
                results = data.get("results")
                items = (
                    results.get("items", [])
                    if isinstance(results, dict)
                    else (results if isinstance(results, list) else [])
                )
                count = data.get("count", len(items)) if isinstance(data.get("count"), int) else len(items)
                status = "ok" if count > 0 else "empty"
                summary_parts.append(
                    "no matching content"
                    if count == 0
                    else (f"{count} result" if count == 1 else f"{count} results")
                )
            elif tool_name == "get_symbol" or ("symbol" in data and "source" in data):
                src = data.get("source")
                status = "ok" if isinstance(src, str) and src.strip() else "empty"
                if status == "empty":
                    summary_parts.append("empty symbol source")
                elif data.get("file") and data.get("line_start") is not None:
                    summary_parts.append(f"OK ({data.get('file')}:{data.get('line_start')}-{data.get('line_end')})")
                else:
                    summary_parts.append("OK")
            elif tool_name == "find_callers":
                callers = data.get("callers")
                p_callers = data.get("possible_callers")
                c_len = (
                    len(callers.get("items", []))
                    if isinstance(callers, dict)
                    else (len(callers) if isinstance(callers, list) else 0)
                )
                p_len = (
                    len(p_callers.get("items", []))
                    if isinstance(p_callers, dict)
                    else (len(p_callers) if isinstance(p_callers, list) else 0)
                )
                status = "ok" if (c_len + p_len) > 0 else "empty"
                if status == "empty":
                    summary_parts.append("no callers found")
                else:
                    summary_parts.append(
                        f"{c_len} caller" + ("" if c_len == 1 else "s")
                        + (f", {p_len} possible" if p_len else "")
                    )
            elif tool_name == "find_callees":
                callees = data.get("callees")
                p_callees = data.get("possible_callees")
                ext = data.get("external_calls")
                c_len = (
                    len(callees.get("items", []))
                    if isinstance(callees, dict)
                    else (len(callees) if isinstance(callees, list) else 0)
                )
                p_len = (
                    len(p_callees.get("items", []))
                    if isinstance(p_callees, dict)
                    else (len(p_callees) if isinstance(p_callees, list) else 0)
                )
                e_len = (
                    len(ext.get("items", []))
                    if isinstance(ext, dict)
                    else (len(ext) if isinstance(ext, list) else 0)
                )
                status = "ok" if (c_len + p_len + e_len) > 0 else "empty"
                if status == "empty":
                    summary_parts.append("no callees found")
                else:
                    summary_parts.append(
                        f"{c_len} callee" + ("" if c_len == 1 else "s")
                        + (f", {p_len} possible" if p_len else "")
                        + (f", {e_len} external" if e_len else "")
                    )
            elif tool_name == "find_importers":
                imp = data.get("importers")
                i_len = (
                    len(imp.get("items", []))
                    if isinstance(imp, dict)
                    else (len(imp) if isinstance(imp, list) else 0)
                )
                status = "ok" if i_len > 0 else "empty"
                summary_parts.append(
                    "no importers found" if i_len == 0 else (f"{i_len} importer" if i_len == 1 else f"{i_len} importers")
                )
            elif tool_name == "get_dependencies":
                status = "ok" if isinstance(data.get("depends_on"), dict) else "inconclusive"
                summary_parts.append("OK" if status == "ok" else "inconclusive")
            elif tool_name == "get_ast":
                status = "ok" if data.get("symbol") else "inconclusive"
                summary_parts.append("OK" if status == "ok" else "inconclusive")
            elif tool_name == "write_file":
                state_val = data.get("state")
                lines_n = data.get("line_count")
                if state_val in ("FAILED", "TIMEOUT", "REPAIR_LIMIT_REACHED"):
                    status = "failed"
                    summary_parts.append(f"{state_val}")
                else:
                    summary_parts.append(f"OK ({lines_n} lines)" if lines_n is not None else "OK")
            elif tool_name == "verify_repository":
                state_val = data.get("state")
                if state_val == "SUCCESS":
                    status = "ok"
                    summary_parts.append("PASSED")
                elif state_val in ("NO_TESTS", "UNAVAILABLE"):
                    status = "inconclusive"
                    summary_parts.append(str(state_val))
                else:
                    status = "failed"
                    summary_parts.append(str(state_val or "FAILED"))
            elif tool_name == "run_command":
                exit_code = data.get("exit_code")
                timed_out = data.get("timed_out")
                status = "ok" if (exit_code == 0 and not timed_out) else "failed"
                if status == "ok":
                    summary_parts.append("OK")
                else:
                    stderr_snip = " ".join(str(data.get("stderr") or "").split())[:120]
                    detail = f"exit_code={exit_code}" + (" (timed out)" if timed_out else "")
                    if stderr_snip:
                        detail += f"; {stderr_snip}"
                    summary_parts.append(detail)
            else:
                summary_parts.append("OK")
    else:
        if not obs_str.strip():
            status = "empty"
            summary_parts.append("empty result")
        else:
            summary_parts.append("OK")

    raw_summary = redact_secrets(" ".join(summary_parts) or status, api_key)
    summary_truncated = len(raw_summary) > _MAX_SUMMARY_CHARS
    if summary_truncated:
        raw_summary = raw_summary[: _MAX_SUMMARY_CHARS - 3] + "..."
    return {
        "status": status,
        "summary": raw_summary,
        "obs_chars": len(obs_str),
        "obs_truncated": obs_truncated,
        "summary_truncated": summary_truncated,
    }


def _format_tool_call_target(
    tool_name: str, kwargs: dict[str, Any], raw_result: Any, api_key: str | None = None, observation_text: str = ""
) -> str:
    """Format a concise human-readable tool + target label."""
    data: dict[str, Any] = {}
    if observation_text:
        try:
            parsed = json.loads(observation_text)
            if isinstance(parsed, dict):
                data = parsed
        except (TypeError, ValueError):
            pass
    if not data and isinstance(raw_result, dict):
        data = raw_result
    if tool_name == "get_file":
        path = redact_secrets(str(data.get("file") or kwargs.get("path") or ""), api_key)
        req_s, req_e = kwargs.get("start_line"), kwargs.get("end_line")
        res_s, res_e = data.get("line_start"), data.get("line_end")
        truncated = bool(data.get("truncated") or data.get("observation_truncated"))
        if req_s is not None or req_e is not None:
            s = res_s if res_s is not None else (req_s if req_s is not None else 1)
            e = res_e if res_e is not None else (req_e if req_e is not None else "")
            return f"get_file {path}:{s}-{e}"
        if truncated and res_s is not None and res_e is not None:
            return f"get_file {path}:{res_s}-{res_e}"
        return f"get_file {path}".strip()
    if tool_name == "write_file":
        path = redact_secrets(str(data.get("file") or kwargs.get("path") or ""), api_key)
        return f"write_file {path}".strip()
    if tool_name == "search_repository":
        q = redact_secrets(str(kwargs.get("query") or data.get("query") or ""), api_key)
        fp = kwargs.get("file_path") or data.get("file_path")
        base = f"search_repository({json.dumps(q[:90], ensure_ascii=False)})"
        return f"{base} in {fp}" if fp else base
    if tool_name == "find_files":
        pat = redact_secrets(str(kwargs.get("pattern") or data.get("pattern") or ""), api_key)
        return f"find_files({json.dumps(pat[:90], ensure_ascii=False)})"
    if tool_name == "find_symbol":
        q = redact_secrets(str(kwargs.get("query") or data.get("query") or ""), api_key)
        return f"find_symbol({json.dumps(q[:90], ensure_ascii=False)})"
    if tool_name in ("get_symbol", "find_callers", "find_callees", "get_dependencies", "get_ast"):
        sym = redact_secrets(
            str(kwargs.get("symbol") or kwargs.get("target") or data.get("symbol") or ""),
            api_key,
        )
        return f"{tool_name}({json.dumps(sym[:90], ensure_ascii=False)})"
    if tool_name == "find_importers":
        mod = redact_secrets(str(kwargs.get("target") or kwargs.get("module") or data.get("module") or ""), api_key)
        return f"find_importers({json.dumps(mod[:90], ensure_ascii=False)})"
    if tool_name == "run_command":
        cmd = redact_secrets(str(kwargs.get("command") or ""), api_key)
        return f"run_command({json.dumps(cmd[:90], ensure_ascii=False)})"
    if kwargs:
        return f"{tool_name}({format_tool_args(kwargs, api_key, max_chars=90)})"
    return tool_name


# ------------------------------------------------------------------
# Model-call inspection & error classification (silent on success)
# ------------------------------------------------------------------


def _msg_text(msg: Any) -> str:
    content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
    if isinstance(content, list):
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return str(content or "")


def _msg_type(msg: Any) -> str:
    if isinstance(msg, dict):
        return str(msg.get("role") or msg.get("type") or "")
    return str(getattr(msg, "type", "") or "")


def compute_request_breakdown(messages: list[Any], stage: str = "") -> dict[str, int]:
    """Compute character and message-size breakdown without modifying the request."""
    sys_chars = 0
    history_chars = 0
    tool_obs_chars = 0
    evidence_chars = 0
    plan_criteria_chars = 0
    hint_chars = 0
    tool_call_chars = 0

    for msg in messages or []:
        kind = _msg_type(msg)
        text = _msg_text(msg)
        length = len(text)
        if kind in ("system",):
            sys_chars += length
        elif kind == "tool":
            tool_obs_chars += length
        elif kind in ("human", "user"):
            if text.startswith(("Plan status data:\n", "Acceptance criteria and evidence data:\n")):
                plan_criteria_chars += length
            elif text.startswith("Already investigated earlier this run"):
                hint_chars += length
            else:
                history_chars += length
        else:
            history_chars += length
            tcs = msg.get("tool_calls") if isinstance(msg, dict) else getattr(msg, "tool_calls", None)
            if tcs:
                try:
                    tool_call_chars += len(json.dumps(tcs, ensure_ascii=False, default=str))
                except Exception:
                    pass

    total_chars = (
        sys_chars
        + history_chars
        + tool_obs_chars
        + evidence_chars
        + plan_criteria_chars
        + hint_chars
        + tool_call_chars
    )
    est_input_tokens = max(1, (total_chars + 3) // 4) if total_chars > 0 else 0
    return {
        "messages": len(messages or []),
        "total_chars": total_chars,
        "est_input_tokens": est_input_tokens,
        "system_chars": sys_chars,
        "history_chars": history_chars,
        "tool_obs_chars": tool_obs_chars,
        "evidence_chars": evidence_chars,
        "plan_criteria_chars": plan_criteria_chars,
        "hint_chars": hint_chars,
    }


def extract_model_identity(model: Any) -> dict[str, Any]:
    """Inspect provider name, model identifier, and configured max_retries."""
    target = getattr(model, "bound", None) or model
    model_name = (
        getattr(target, "model_name", None)
        or getattr(target, "model", None)
        or getattr(model, "model_name", None)
        or getattr(model, "model", None)
        or type(target).__name__
    )
    cls_name = type(target).__name__
    mod_name = type(target).__module__.lower()
    if (
        "google" in mod_name
        or "gemini" in mod_name
        or "google" in cls_name.lower()
        or "gemini" in cls_name.lower()
    ):
        provider = "google"
    elif "groq" in mod_name or "groq" in cls_name.lower():
        provider = "groq"
    elif "openai" in mod_name or "openai" in cls_name.lower():
        provider = "openai"
    elif "anthropic" in mod_name or "anthropic" in cls_name.lower():
        provider = "anthropic"
    else:
        provider = cls_name
    max_retries = getattr(target, "max_retries", getattr(model, "max_retries", None))
    return {
        "provider": str(provider),
        "model": str(model_name),
        "max_retries": max_retries if isinstance(max_retries, int) else None,
    }


def extract_provider_token_usage(reply: Any) -> dict[str, Any]:
    """Extract actual provider token usage from an AIMessage when exposed."""
    usage = getattr(reply, "usage_metadata", None) or {}
    resp_meta = getattr(reply, "response_metadata", None) or {}
    if not usage and isinstance(resp_meta, dict):
        usage = resp_meta.get("token_usage") or resp_meta.get("usage") or {}
    incoming = usage.get("input_tokens", usage.get("prompt_tokens")) if isinstance(usage, dict) else None
    outgoing = usage.get("output_tokens", usage.get("completion_tokens")) if isinstance(usage, dict) else None
    total = usage.get("total_tokens") if isinstance(usage, dict) else None
    if isinstance(incoming, (int, float)) and isinstance(outgoing, (int, float)):
        in_tok = int(incoming)
        out_tok = int(outgoing)
        tot_tok = int(total) if isinstance(total, (int, float)) else (in_tok + out_tok)
        return {
            "available": True,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "total_tokens": tot_tok,
        }
    return {"available": False, "input_tokens": None, "output_tokens": None, "total_tokens": None}


def analyze_provider_error(
    error: Exception, api_key: str | None = None, configured_max_retries: int | None = None
) -> dict[str, Any]:
    """Classify a model call exception without conflating TPM limits with context window limits."""
    raw_text = " ".join(str(error).split())
    safe_text = redact_secrets(raw_text, api_key)
    if len(safe_text) > _MAX_ERROR_CHARS:
        safe_text = safe_text[: _MAX_ERROR_CHARS - 3] + "..."

    if getattr(error, "error_category", None) == "context_budget_exceeded":
        return {
            "error_origin": "context_budget",
            "error_type": type(error).__name__,
            "error_category": "context_budget_exceeded",
            "limit_kind": "context_budget",
            "http_status": 413,
            "provider_code": "context_budget_exceeded",
            "provider_type": "context_budget",
            "requested_tokens": None,
            "tpm_limit": None,
            "sdk_max_retries": configured_max_retries,
            "retries_exhausted": False,
            "retry_outcome": "aborted_unfittable_before_send",
            "message": safe_text,
        }

    status = getattr(error, "status_code", None)
    response = getattr(error, "response", None)
    if status is None and response is not None:
        status = getattr(response, "status_code", None)
    if status is None:
        m_status = _HTTP_STATUS_RE.search(safe_text)
        if m_status:
            status = int(m_status.group(1))

    body = getattr(error, "body", None)
    provider_code = None
    provider_type = None
    if isinstance(body, dict):
        err_obj = body.get("error") if isinstance(body.get("error"), dict) else body
        if isinstance(err_obj, dict):
            provider_code = err_obj.get("code")
            provider_type = err_obj.get("type")

    tpm_limit = None
    requested_tokens = None
    m1 = _TPM_LIMIT_RE_1.search(safe_text)
    if m1:
        tpm_limit, requested_tokens = int(m1.group(1)), int(m1.group(2))
    else:
        m2 = _TPM_LIMIT_RE_2.search(safe_text)
        if m2:
            requested_tokens, tpm_limit = int(m2.group(1)), int(m2.group(2))

    lower = safe_text.casefold()
    err_cls = type(error).__name__
    is_tpm = (
        "tokens per minute" in lower
        or bool(re.search(r"\btpm\b", lower))
        or (
            status in (413, 429)
            and tpm_limit is not None
            and requested_tokens is not None
            and "context" not in lower
        )
    )
    is_context_window = (
        "context_length" in lower
        or "context window" in lower
        or "maximum context length" in lower
        or "exceeds the maximum number of tokens" in lower
        or "token count exceeds" in lower
        or "input token limit" in lower
        or provider_code == "context_length_exceeded"
    )
    is_rate_or_quota = (
        status == 429
        or "ratelimit" in err_cls.casefold()
        or "resourceexhausted" in err_cls.casefold()
        or "resource_exhausted" in lower
        or "resource exhausted" in lower
        or "quota exceeded" in lower
        or "quota" in lower
        or provider_code in ("rate_limit_exceeded", "resource_exhausted")
    )
    is_auth_error = (
        status in (401, 403)
        or "auth" in err_cls.casefold()
        or "permission" in err_cls.casefold()
        or "api_key_invalid" in lower
        or "api key not valid" in lower
    )

    if is_tpm:
        error_category = "provider_tpm_limit"
        limit_kind = "tokens_per_minute"
    elif is_context_window:
        error_category = "provider_context_window_limit"
        limit_kind = "context_window"
    elif status == 413:
        error_category = "provider_request_too_large"
        limit_kind = "request_payload_size"
    elif is_rate_or_quota:
        error_category = "provider_rate_limit"
        limit_kind = "rate_limit"
    elif is_auth_error:
        error_category = "provider_auth_error"
        limit_kind = "none"
    elif isinstance(status, int) and 500 <= status < 600:
        error_category = "provider_server_error"
        limit_kind = "none"
    elif "timeout" in err_cls.casefold() or "timed out" in lower:
        error_category = "provider_timeout"
        limit_kind = "none"
    else:
        error_category = "provider_request_error"
        limit_kind = "none"

    if status == 413:
        retry_outcome = "failed_non_retryable_http_413"
        retries_exhausted = False
    elif status in (429, 500, 502, 503, 504) or "timeout" in error_category or "rate_limit" in error_category:
        retries_exhausted = bool(configured_max_retries and configured_max_retries > 0)
        retry_outcome = (
            f"exhausted_sdk_retries(max_retries={configured_max_retries})"
            if retries_exhausted
            else "failed_without_sdk_retries"
        )
    else:
        retries_exhausted = False
        retry_outcome = "failed"

    return {
        "error_origin": "provider_request",
        "error_type": err_cls,
        "error_category": error_category,
        "limit_kind": limit_kind,
        "http_status": status,
        "provider_code": provider_code,
        "provider_type": provider_type,
        "requested_tokens": requested_tokens,
        "tpm_limit": tpm_limit,
        "sdk_max_retries": configured_max_retries,
        "retries_exhausted": retries_exhausted,
        "retry_outcome": retry_outcome,
        "message": safe_text,
    }


def invoke_model_logged(
    model: Any,
    messages: list[Any],
    *,
    agent: str,
    stage: str,
    invoke_config: dict | None = None,
    api_key: str | None = None,
    step_id: str | None = None,
    parent_agent: str | None = None,
    turn: int | None = None,
    format_attempt: int = 1,
    max_format_attempts: int = 1,
) -> Any:
    """Invoke a LangChain model; stay silent on routine success and log concise detail on failure.

    Never sends a request if `fit_messages_to_budget` reports that essential
    content cannot fit within the context budget (`not fit["fits"]` or
    `fit["essential_unfittable"]`).
    """
    ctx = _CURRENT_CONTEXT.get()
    eff_key = api_key or (ctx.api_key if ctx else None)
    ident = extract_model_identity(model)

    if ctx is not None:
        ctx.agent_llm_calls[agent] = ctx.agent_llm_calls.get(agent, 0) + 1
        st = ctx.stage_details.setdefault(
            f"{agent}:{stage}",
            {
                "agent": agent,
                "stage": stage,
                "calls": 0,
                "failures": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "est_input_tokens": 0,
                "latency_ms": 0,
            },
        )
        st["calls"] += 1

    from .context_budget import (
         MAX_CONTEXT_RECOVERY_RETRIES,
        ContextBudgetExceededError,
        estimate_tool_schema_tokens,
        fit_messages_to_budget,
        get_context_token_budget,
        is_context_size_error,
        retry_budget_for,
    )

    fixed_tokens = estimate_tool_schema_tokens(model)
    outgoing, fit = fit_messages_to_budget(messages, get_context_token_budget(), fixed_tokens)
    base_event = {"agent": agent, "stage": stage, "turn": turn}
    if ctx is not None:
        st = ctx.stage_details.get(f"{agent}:{stage}")
        if st is not None:
            st["est_input_tokens"] += fit["est_after_tokens"]
        if fit["compacted_messages"] or not fit["fits"]:
            ctx.context_events.append({**base_event, "event": "compacted" if fit["fits"] else "budget_not_met", **fit})

    if not fit.get("fits", True) or fit.get("essential_unfittable"):
        reason_msg = (
            f"request (~{fit['est_after_tokens']} est. tokens) exceeds context budget "
            f"(~{fit['est_budget_tokens']} est. tokens): {fit.get('reason', 'essential content cannot fit')}"
        )
        exc = ContextBudgetExceededError(reason_msg, fit_report=fit)
        err_info = analyze_provider_error(exc, eff_key, ident["max_retries"])
        if ctx is not None:
            st = ctx.stage_details.get(f"{agent}:{stage}")
            if st is not None:
                st["failures"] += 1
            ctx.model_failures.append({
                "agent": agent,
                "parent_agent": parent_agent or (ctx.parent_agent if ctx else None),
                "stage": stage,
                "step_id": step_id or (ctx.current_step_id if ctx else None),
                "turn": turn,
                "duration_ms": 0,
                **err_info,
            })
        emit_trace(
            f"[{agent}] model_call({stage}) -> FAILED (ContextBudgetExceededError, context_budget_exceeded): "
            f"{err_info['message']}"
        )
        raise exc

    recovery: list[dict[str, Any]] = []
    started = time.perf_counter()
    while True:
        try:
            with ScopeGuard(agent=agent, parent_agent=parent_agent, stage=stage, step_id=step_id):
                if invoke_config is not None:
                    reply = model.invoke(outgoing, config=invoke_config)
                else:
                    reply = model.invoke(outgoing)
            break
        except Exception as exc:
            err_info = analyze_provider_error(exc, eff_key, ident["max_retries"])
            if len(recovery) < MAX_CONTEXT_RECOVERY_RETRIES and is_context_size_error(err_info):
                target = retry_budget_for(fit["est_after_tokens"], err_info)
                retry_out, retry_fit = fit_messages_to_budget(messages, target, fixed_tokens)
                step = {
                    **base_event,
                    "event": "context_retry",
                    "attempt": len(recovery) + 1,
                    "error_category": err_info["error_category"],
                    "http_status": err_info.get("http_status"),
                    "requested_tokens": err_info.get("requested_tokens"),
                    "tpm_limit": err_info.get("tpm_limit"),
                    "original_error": err_info["message"],
                    "est_tokens_before": fit["est_after_tokens"],
                    "est_tokens_after": retry_fit["est_after_tokens"],
                    "est_target_tokens": target,
                    "outcome": "retrying",
                }
                if (
                    retry_fit.get("fits", True)
                    and not retry_fit.get("essential_unfittable")
                    and retry_fit["est_after_tokens"] < fit["est_after_tokens"]
                ):
                    recovery.append(step)
                    if ctx is not None:
                        ctx.context_events.append(step)
                    emit_trace(
                        f"[{agent}] model_call({stage}) -> request too large "
                        f"({err_info['error_category']}); retrying with compacted context "
                        f"(~{fit['est_after_tokens']} -> ~{retry_fit['est_after_tokens']} est. tokens, "
                        f"attempt {len(recovery)}/{MAX_CONTEXT_RECOVERY_RETRIES})"
                    )
                    outgoing, fit = retry_out, retry_fit
                    continue
                step["outcome"] = (
                    "unfittable_within_retry_budget"
                    if not retry_fit.get("fits", True) or retry_fit.get("essential_unfittable")
                    else "no_further_compaction_possible"
                )
                recovery.append(step)
                if ctx is not None:
                    ctx.context_events.append(step)
            elif recovery:
                recovery.append({**base_event, "event": "context_retry_failed", "outcome": "non_context_error",
                                 "error_category": err_info["error_category"]})
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            if ctx is not None:
                st = ctx.stage_details.get(f"{agent}:{stage}")
                if st is not None:
                    st["failures"] += 1
                    st["latency_ms"] += elapsed_ms
                failure = {
                    "agent": agent,
                    "parent_agent": parent_agent or (ctx.parent_agent if ctx else None),
                    "stage": stage,
                    "step_id": step_id or (ctx.current_step_id if ctx else None),
                    "turn": turn,
                    "duration_ms": elapsed_ms,
                    **err_info,
                }
                if recovery:
                    failure["context_recovery"] = {
                        "outcome": "failed",
                        "attempts": [r for r in recovery if r.get("event") == "context_retry"
                                     and r.get("outcome") == "retrying"],
                        "original_error": recovery[0].get("original_error"),
                    }
                ctx.model_failures.append(failure)
            meta_parts = [err_info["error_type"]]
            if err_info.get("http_status"):
                meta_parts.append(f"HTTP {err_info['http_status']}")
            if err_info.get("error_category") and err_info["error_category"] != "provider_request_error":
                meta_parts.append(err_info["error_category"])
            if err_info.get("requested_tokens") is not None and err_info.get("tpm_limit") is not None:
                meta_parts.append(f"requested={err_info['requested_tokens']} limit={err_info['tpm_limit']}")
            if recovery:
                meta_parts.append(f"context recovery failed after {len([r for r in recovery if r.get('outcome') == 'retrying'])} retries")
            meta_str = ", ".join(meta_parts)
            emit_trace(
                f"[{agent}] model_call({stage}) -> FAILED ({meta_str}): {err_info['message']}"
            )
            if recovery:
                try:
                    exc.context_recovery = {"outcome": "failed", "original_error": recovery[0].get("original_error")}
                except Exception:
                    pass
            raise

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    if recovery and ctx is not None:
        ctx.context_events.append({**base_event, "event": "context_recovered", "outcome": "recovered",
                                   "attempts": len(recovery), "est_final_tokens": fit["est_after_tokens"]})
    if ctx is not None:
        st = ctx.stage_details.get(f"{agent}:{stage}")
        if st is not None:
            st["latency_ms"] += elapsed_ms
            usage = extract_provider_token_usage(reply)
            if usage["available"]:
                st["input_tokens"] += usage["input_tokens"]
                st["output_tokens"] += usage["output_tokens"]
    if stage == "decision":
        t_calls = getattr(reply, "tool_calls", None) or []
        action_note = f"calling {len(t_calls)} tool(s)" if t_calls else "writing final answer"
        turn_note = f" (turn {turn})" if turn is not None else ""
        emit_trace(f"[{agent}] decision{turn_note} -> {action_note} ({elapsed_ms} ms)")
    return reply


# ------------------------------------------------------------------
# Plan, Tool, and Guard event logging
# ------------------------------------------------------------------


def register_requested_tool_calls(agent: str, calls: list[dict], *, turn: int | None = None) -> None:
    """Queue tool call IDs for correlation without emitting redundant pre-execution lines."""
    try:
        ctx = _CURRENT_CONTEXT.get()
        if ctx is None:
            return
        for call in calls or []:
            name = str(call.get("name") or "tool")
            call_id = str(call.get("id") or "")
            if call_id:
                queue_key = f"{agent}:{name}"
                ctx.pending_tool_call_ids.setdefault(queue_key, []).append(call_id)
    except Exception:
        pass


def pop_pending_tool_call_id(agent: str, tool_name: str) -> str | None:
    try:
        ctx = _CURRENT_CONTEXT.get()
        if ctx is None:
            return None
        queue = ctx.pending_tool_call_ids.get(f"{agent}:{tool_name}")
        if queue:
            return queue.pop(0)
        return ctx.current_tool_call_id
    except Exception:
        return None


def log_main_received_tool_result(
    tool_name: str,
    result_status: str,
    step_succeeded: bool,
    *,
    step_id: str | None = None,
    payload: dict | None = None,
) -> None:
    """Suppressed on normal execution to avoid duplicating tool completion lines."""
    return


def _concise_guard_reason(guard_name: str, reason: str, api_key: str | None = None) -> str:
    cleaned = _GUARD_PREFIX_RE.sub("", redact_secrets(str(reason or ""), api_key).strip())
    for splitter in (". Use ", ". Choose ", ". Switch strategy:", ". Give ", ". If "):
        if splitter in cleaned:
            cleaned = cleaned.split(splitter, 1)[0].strip()
    return f"{guard_name}: {cleaned[:200]}" if guard_name else cleaned[:220]


def log_guard_event(
    guard_name: str,
    tool_name: str,
    reason: str,
    *,
    step_id: str | None = None,
    target: str | None = None,
) -> None:
    """Emit a single concise BLOCKED line when a decision or safety guard prevents an action."""
    try:
        ctx = _CURRENT_CONTEXT.get()
        api_key = ctx.api_key if ctx else None
        if ctx is not None:
            ctx.guard_events.append({"guard": guard_name, "tool": tool_name, "reason": reason})
        eff_target = target if target is not None else (ctx.current_tool_target if ctx else None)
        if eff_target and eff_target != tool_name:
            clean_target = redact_secrets(str(eff_target), api_key)[:90]
            if tool_name == "get_file":
                action_str = f"get_file {clean_target}"
            else:
                action_str = f"{tool_name}({json.dumps(clean_target, ensure_ascii=False)})"
        else:
            action_str = tool_name
        short_reason = _concise_guard_reason(guard_name, reason, api_key)
        emit_trace(f"[Main] {action_str} -> BLOCKED: {short_reason}")
    except Exception:
        pass


def log_tool_execution_start(tool_name: str, kwargs: dict) -> tuple[float, str, str | None]:
    """Record start time and arguments for a tool execution without emitting a duplicate start line."""
    started = time.perf_counter()
    ctx = _CURRENT_CONTEXT.get()
    agent = ctx.current_agent if ctx else "Main"
    tool_call_id = pop_pending_tool_call_id(agent, tool_name)
    try:
        if ctx is not None:
            ctx.active_tool_kwargs.append((agent, tool_name, dict(kwargs or {})))
    except Exception:
        pass
    return started, agent, tool_call_id


def _pop_tool_kwargs(ctx: RunTraceContext | None, agent: str, tool_name: str) -> dict[str, Any]:
    if ctx is None or not ctx.active_tool_kwargs:
        return {}
    for idx in range(len(ctx.active_tool_kwargs) - 1, -1, -1):
        a, t, kw = ctx.active_tool_kwargs[idx]
        if a == agent and t == tool_name:
            ctx.active_tool_kwargs.pop(idx)
            return kw
    return ctx.active_tool_kwargs.pop()[2]


def log_tool_execution_end(
    tool_name: str,
    started: float,
    agent: str,
    tool_call_id: str | None,
    raw_result: Any,
    observation_text: str,
) -> None:
    """Emit a single concise tool execution line and record per-agent tool attribution."""
    try:
        ctx = _CURRENT_CONTEXT.get()
        api_key = ctx.api_key if ctx else None
        kwargs = _pop_tool_kwargs(ctx, agent, tool_name)
        info = summarize_tool_observation(tool_name, raw_result, observation_text, api_key)
        status = info["status"]

        if ctx is not None:
            by_tool = ctx.agent_tool_calls.setdefault(agent, {})
            by_tool[tool_name] = by_tool.get(tool_name, 0) + 1
            by_outcome = ctx.agent_tool_outcomes.setdefault(agent, {})
            by_outcome[status] = by_outcome.get(status, 0) + 1
            if status == "failed":
                ctx.tool_failures.append(
                    {
                        "agent": agent,
                        "parent_agent": ctx.parent_agent,
                        "tool": tool_name,
                        "step_id": ctx.current_step_id,
                        "tool_call_id": tool_call_id,
                        "status": status,
                        "reason": info["summary"],
                    }
                )

        target_str = _format_tool_call_target(tool_name, kwargs, raw_result, api_key, observation_text)
        summary = info["summary"]
        if status == "failed":
            outcome_str = f"FAILED: {summary}"
        elif status == "empty":
            outcome_str = f"EMPTY; {summary}"
        elif status == "inconclusive":
            outcome_str = f"PARTIAL; {summary}"
        else:
            outcome_str = summary
        emit_trace(f"[{agent}] {target_str} -> {outcome_str}")
    except Exception:
        pass


def log_tool_execution_exception(
    tool_name: str, started: float, agent: str, tool_call_id: str | None, error: Exception
) -> None:
    """Emit a concise tool-exception trace line before re-raising."""
    try:
        ctx = _CURRENT_CONTEXT.get()
        api_key = ctx.api_key if ctx else None
        kwargs = _pop_tool_kwargs(ctx, agent, tool_name)
        err_text = redact_secrets(f"{type(error).__name__}: {error}", api_key)[:240]
        if ctx is not None:
            by_tool = ctx.agent_tool_calls.setdefault(agent, {})
            by_tool[tool_name] = by_tool.get(tool_name, 0) + 1
            by_outcome = ctx.agent_tool_outcomes.setdefault(agent, {})
            by_outcome["exception"] = by_outcome.get("exception", 0) + 1
            ctx.tool_failures.append(
                {
                    "agent": agent,
                    "parent_agent": ctx.parent_agent,
                    "tool": tool_name,
                    "step_id": ctx.current_step_id,
                    "tool_call_id": tool_call_id,
                    "status": "exception",
                    "reason": err_text,
                }
            )
        target_str = _format_tool_call_target(tool_name, kwargs, None, api_key)
        emit_trace(f"[{agent}] {target_str} -> FAILED: {err_text}")
    except Exception:
        pass


def log_plan_initialized(plan: dict, criteria: list[dict], intent: dict) -> None:
    """Emit a concise planning summary line."""
    try:
        steps = (plan or {}).get("steps") or []
        kind = (intent or {}).get("kind", "explain")
        depth = (intent or {}).get("depth", "standard")
        emit_trace(
            f"[Main] planning -> {len(steps)} step(s), {len(criteria or [])} criterion/criteria "
            f"(intent: {kind}/{depth})"
        )
    except Exception:
        pass


def log_plan_step_transition(
    step_id: str | None, status: str, description: str = "", *, stage: str = "decision", reason: str = ""
) -> None:
    """Suppressed on normal execution to avoid repeated plan transition noise."""
    return


def log_evidence_recorded(step: dict, usable: bool, source_agent: str) -> None:
    """Suppressed on normal execution to avoid repeating tool completion events."""
    return


def record_payload_fit_event(
    *,
    agent: str,
    stage: str,
    fit_report: dict[str, Any] | None,
) -> None:
    """Record explicit payload-fitting compaction or budget-not-met diagnostics."""
    try:
        if not isinstance(fit_report, dict):
            return
        ctx = _CURRENT_CONTEXT.get()
        fits = bool(fit_report.get("fits", True))
        compacted = int(fit_report.get("compacted_items", 0) or 0)
        if ctx is not None and (compacted > 0 or not fits):
            ctx.context_events.append(
                {
                    "agent": agent,
                    "stage": stage,
                    "turn": None,
                    "event": "payload_compacted" if fits else "budget_not_met",
                    **fit_report,
                }
            )
        if not fits:
            emit_trace(
                f"[{agent}] payload_fit({stage}) -> budget NOT met: ~{fit_report.get('est_after_tokens')} "
                f"est. tokens vs target ~{fit_report.get('est_budget_tokens')} "
                f"({fit_report.get('reason', 'essential metadata exceeds budget')})"
            )
    except Exception:
        pass


def record_stage_size_recovery(
    *,
    agent: str,
    stage: str,
    recovered: bool,
    original_error: str,
    fit_report: dict[str, Any] | None = None,
) -> None:
    """Record a reduced-payload retry after an oversized-request failure."""
    try:
        ctx = _CURRENT_CONTEXT.get()
        if ctx is None:
            return
        if recovered and ctx.model_failures:
            for idx in range(len(ctx.model_failures) - 1, -1, -1):
                mf = ctx.model_failures[idx]
                if mf.get("agent") == agent and str(mf.get("stage") or "").startswith(stage):
                    ctx.model_failures.pop(idx)
                    st = ctx.stage_details.get(f"{agent}:{stage}")
                    if st is not None and st.get("failures", 0) > 0:
                        st["failures"] -= 1
                    break
        ctx.context_events.append(
            {
                "agent": agent,
                "stage": stage,
                "turn": None,
                "event": "context_recovered" if recovered else "context_retry",
                "outcome": "recovered" if recovered else "retrying_reduced_payload",
                "original_error": original_error,
                **({"est_final_tokens": fit_report.get("est_after_tokens")} if isinstance(fit_report, dict) else {}),
            }
        )
    except Exception:
        pass


def log_grading_diagnostics(
    *,
    plan: dict,
    criteria: list[dict],
    usable_steps: list[dict],
    verdicts: dict,
    grader_ran: bool,
    cache_hit: bool,
    grader_error: str,
    format_retried: bool = False,
    size_retried: bool = False,
    payload_fit: dict[str, Any] | None = None,
) -> None:
    """Record grading state internally and emit a trace line only if grading failed."""
    try:
        ctx = _CURRENT_CONTEXT.get()
        decision_produced = bool(verdicts) and not grader_error
        if grader_error:
            status = "failed_before_decision"
            diag_case = "model_request_failed_before_grading"
        elif not usable_steps and not cache_hit:
            status = "skipped_no_usable_evidence"
            all_ev = [s for s in (plan or {}).get("steps", []) if s.get("result")]
            diag_case = "tool_call_failed" if any(s.get("result") == "failed" for s in all_ev) else "required_evidence_never_gathered"
        else:
            status = "decision_returned"
            diag_case = (
                "all_criteria_established"
                if criteria and all(c.get("satisfied") for c in criteria)
                else "grader_evaluated_criteria"
            )

        if ctx is not None:
            ctx.grading_events.append(
                {
                    "grader_ran": grader_ran,
                    "cache_hit": cache_hit,
                    "format_retried": format_retried,
                    "size_retried": size_retried,
                    "payload_fits": payload_fit.get("fits", True) if isinstance(payload_fit, dict) else True,
                    "decision_produced": decision_produced,
                    "status": status,
                    "diagnostic_case": diag_case,
                    "error": grader_error or None,
                    "usable_evidence_count": len(usable_steps),
                }
            )
        if grader_error:
            emit_trace(f"[Main] evidence_grading -> FAILED: {grader_error[:200]}")
    except Exception:
        pass


def log_conclusion_validation_diagnostics(
    result: dict,
    *,
    cache_hit: bool = False,
    format_retried: bool = False,
    size_retried: bool = False,
    payload_fit: dict[str, Any] | None = None,
) -> None:
    """Record conclusion check state and emit a trace line only when an error occurred."""
    try:
        ctx = _CURRENT_CONTEXT.get()
        err = result.get("error")
        claims = result.get("claims") or []
        confirmed = sum(1 for c in claims if c.get("certainty") == "confirmed" and c.get("supported"))
        possible = sum(1 for c in claims if c.get("certainty") == "possible")
        rejected = len(claims) - confirmed - possible
        status = "failed_before_decision" if err else ("passed" if result.get("supported") else "claims_rejected")

        if ctx is not None:
            ctx.conclusion_events.append(
                {
                    "status": status,
                    "cache_hit": cache_hit,
                    "format_retried": format_retried,
                    "size_retried": size_retried,
                    "payload_fits": payload_fit.get("fits", True) if isinstance(payload_fit, dict) else True,
                    "error": err,
                    "supported": bool(result.get("supported")),
                    "confirmed": confirmed,
                    "possible": possible,
                    "rejected": rejected,
                }
            )
        if err and err != "no usable evidence was recorded":
            emit_trace(f"[Main] conclusion_validation -> FAILED: {str(err)[:200]}")
    except Exception:
        pass


def record_completion_diagnostics(
    *,
    support: str,
    outcome: str,
    criteria: list[dict],
    plan: dict | None,
    failure_info: dict | None,
    verification_info: dict | None,
    model_error: str | None,
    answer_check: dict | None,
    step_limit_reached: bool,
    extra_unresolved: tuple = (),
    incomplete_reasons_list: list[str] | None = None,
) -> dict[str, Any]:
    """Record completion state and failure context for the final completion line."""
    ctx = _CURRENT_CONTEXT.get()
    steps = [s for s in (plan or {}).get("steps", []) if s.get("result")]
    pending = [s for s in (plan or {}).get("steps", []) if s.get("status") == "pending"]
    usable = [
        s
        for s in steps
        if s.get("result") == "ok" or (s.get("tool") == "get_file" and s.get("result") == "inconclusive")
    ]
    satisfied = [c for c in (criteria or []) if c.get("satisfied")]
    unmet = [c for c in (criteria or []) if not c.get("satisfied")]
    grader_failed_notes = [
        c for c in unmet if str(c.get("note", "")).startswith("grading call failed")
    ]

    failing_stage = "none"
    failing_agent = "none"
    error_category = "none"
    if ctx is not None and ctx.model_failures:
        last_mf = ctx.model_failures[-1]
        failing_stage = str(last_mf.get("stage") or "model_call")
        failing_agent = str(last_mf.get("agent") or "Main")
        error_category = str(last_mf.get("error_category") or "provider_request_error")
    if not model_error and ctx is not None:
        for mf in reversed(ctx.model_failures):
            if mf.get("agent") == "Main" and mf.get("stage") in ("decision", "wrap_up", "planning"):
                model_error = f"{mf.get('stage')} model call failed: {mf.get('message', '')}"
                break
    if ctx is not None and ctx.model_failures:
        pass
    elif model_error:
        failing_stage = "decision" if "decision" in model_error else ("wrap-up" if "wrap-up" in model_error else "planning")
        failing_agent = "Main"
        error_category = "provider_tpm_limit" if "413" in model_error else ("provider_rate_limit" if "429" in model_error else "provider_request_error")
    elif ctx is not None and ctx.tool_failures:
        last_tf = ctx.tool_failures[-1]
        failing_stage = f"tool:{last_tf.get('tool')}"
        failing_agent = str(last_tf.get("agent") or "Main")
        error_category = "tool_execution_failure"

    if support == "SUPPORTED":
        diag_case = "supported"
        reason = ""
    elif support == "GRADING_FAILED":
        diag_case = "case_4_model_request_failed_before_grading"
        note_detail = grader_failed_notes[0].get("note", "") if grader_failed_notes else "grader call failed"
        reason = f"evidence grading failed ({note_detail})"
    elif support == "BLOCKED":
        if failure_info is not None:
            diag_case = "case_3_tool_call_failed"
            reason = f"latest {failure_info.get('tool', 'modification/verification')} run failed"
        elif steps and not usable and any(s.get("result") == "failed" for s in steps):
            diag_case = "case_3_tool_call_failed"
            failed_steps = [s["step_id"] for s in steps if s.get("result") == "failed"]
            if ctx is not None and ctx.model_failures:
                reason = f"attempted steps ({','.join(failed_steps)}) failed after {failing_agent} {failing_stage} error ({error_category})"
            else:
                reason = f"attempted steps ({','.join(failed_steps)}) failed with no usable evidence"
        else:
            diag_case = "case_6_stopped_by_failure_or_stopping_condition"
            reason = "investigation was blocked before usable evidence could be established"
    elif support == "PARTIAL":
        if model_error and len(satisfied) == len(criteria or []) and criteria:
            diag_case = "case_6_stopped_by_failure_or_stopping_condition"
            reason = f"stopped early due to {model_error}"
        elif answer_check and answer_check.get("error") and len(satisfied) == len(criteria or []) and criteria:
            diag_case = "case_6_stopped_by_failure_or_stopping_condition"
            reason = f"conclusion validation failed ({answer_check['error']})"
        elif len(satisfied) < len(criteria or []):
            citation_rej = [c["id"] for c in unmet if c.get("validation") == "citation_rejected"]
            if citation_rej:
                diag_case = "case_1_evidence_rejected_as_insufficient"
                reason = f"citation check rejected {','.join(citation_rej)}"
            else:
                diag_case = "case_5_grader_could_not_establish_criterion"
                reason = ""
        elif pending:
            diag_case = "case_6_stopped_by_failure_or_stopping_condition"
            reason = f"pending planned steps not attempted ({','.join(s['step_id'] for s in pending)})"
        elif incomplete_reasons_list:
            diag_case = "case_1_evidence_rejected_as_insufficient"
            reason = "; ".join(incomplete_reasons_list)
        elif verification_info and (verification_info.get("status") == "unavailable" or verification_info.get("state") == "UNAVAILABLE"):
            diag_case = "case_6_stopped_by_failure_or_stopping_condition"
            reason = "verification command unavailable"
        else:
            diag_case = "case_6_stopped_by_failure_or_stopping_condition"
            reason = ""
    else:  # UNSUPPORTED
        if model_error and not usable:
            diag_case = "case_6_stopped_by_failure_or_stopping_condition"
            reason = f"stopped due to {model_error}"
        elif not steps or not usable:
            if any(s.get("result") == "failed" for s in steps):
                diag_case = "case_3_tool_call_failed"
                reason = "tool calls failed and produced no usable evidence"
            else:
                diag_case = "case_2_required_evidence_never_gathered"
                reason = "no usable evidence was gathered"
        elif any(c.get("validation") == "citation_rejected" for c in unmet):
            diag_case = "case_1_evidence_rejected_as_insufficient"
            reason = "citation check rejected proposed evidence"
        else:
            diag_case = "case_5_grader_could_not_establish_criterion"
            reason = f"usable evidence ({len(usable)} step(s)) did not establish criteria"
        if step_limit_reached:
            reason = f"{reason}; step limit reached" if reason else "step limit reached"

    record = {
        "support": support,
        "outcome": outcome,
        "criteria_supported": len(satisfied),
        "criteria_total": len(criteria or []),
        "unmet": [c["id"] for c in unmet],
        "diagnostic_case": diag_case,
        "failing_stage": failing_stage,
        "failing_agent": failing_agent,
        "error_category": error_category,
        "step_limit_reached": step_limit_reached,
        "reason": reason,
    }
    if ctx is not None:
        ctx.completion_diagnostics = record
    return record


def format_attribution_metrics(ctx: RunTraceContext | None) -> list[str]:
    """Kept for API compatibility; routine metrics lines are not emitted in normal runs."""
    return []


def format_run_diagnostic_summary(
    ctx: RunTraceContext | None,
    completion: dict | None,
    run_exception: Exception | None = None,
) -> str:
    """Build the single concise `[Main] run completed -> ...` line at the end of a run."""
    diag = ctx.completion_diagnostics if ctx else None

    if run_exception is not None and diag is None:
        err_info = analyze_provider_error(run_exception, ctx.api_key if ctx else None)
        return f"[Main] run completed -> FAILED: {err_info['error_type']}: {err_info['message']}"

    support = (completion or {}).get("support") or (diag or {}).get("support") or "UNKNOWN"
    if support == "SUPPORTED":
        return "[Main] run completed -> SUPPORTED"

    sat = (completion or {}).get("criteria_supported", (diag or {}).get("criteria_supported", 0))
    tot = (completion or {}).get("criteria_total", (diag or {}).get("criteria_total", 0))
    unmet = (completion or {}).get("unmet") or (diag or {}).get("unmet") or []
    reason = str((diag or {}).get("reason") or "").strip()

    parts = [str(support)]
    if tot:
        parts.append(f"{sat}/{tot} criteria supported")
    if unmet:
        parts.append("unresolved: " + ", ".join(str(u) for u in unmet))
    if reason:
        parts.append(reason[:200])
    return "[Main] run completed -> " + "; ".join(parts)
