"""Single-agent plan-driven orchestration for repository investigation, modification, and debugging.

Each invocation creates an initial plan and acceptance criteria, then loops
through direct repository tool selection, execution, and state updates within
configured bounds. The Main Agent has direct access to all ten read-only
repository tools (including symbol, caller/callee, importer, dependency, and
AST analysis), plan revision (`revise_plan`), and capability-gated modification
and verification tools (`write_file`, `verify_repository`, `run_command`).
Deterministic code enforces permissions, source restrictions, checkpoints,
scoped verification, bounded repair, rollback, and evidence grading.
"""

from __future__ import annotations

import contextvars
import hashlib
import json
from pathlib import Path
import re
import time
import uuid

from .state import (
    AIMessage,
    END,
    HumanMessage,
    START,
    StateGraph,
    StructuredTool,
    SystemMessage,
    ToolNode,
)
from pydantic import BaseModel, Field

from ..service import RepoService
from . import evidence as ev
from . import execution as execution_mod
from . import intent as intent_mod
from . import plan as planning
from .config import AgentConfig, load_config
from .deep_state import DeepAgentState
from .followups import deterministic_followup, result_fingerprint
from .model import build_llm
from .observability import (
    analyze_provider_error,
    ensure_run_context,
    get_run_context,
    invoke_model_logged,
    log_conclusion_validation_diagnostics,
    log_evidence_recorded,
    log_grading_diagnostics,
    log_guard_event,
    log_main_received_tool_result,
    log_plan_initialized,
    log_plan_step_transition,
    record_completion_diagnostics,
    record_payload_fit_event,
    record_stage_size_recovery,
    register_requested_tool_calls,
)
from .prompts import (
    CONCLUSION_CHECK_PROMPT,
    CRITERIA_CHECK_PROMPT,
    DEEP_AGENT_SYSTEM_PROMPT,
    PLANNING_SYSTEM_PROMPT,
    SYNTHESIS_SYSTEM_PROMPT,
    SYNTHESIS_USER_PROMPT,
)
from .safety import (
    FAILED,
    NO_TESTS,
    REPAIR_LIMIT_REACHED,
    SUCCESS,
    UNAVAILABLE,
    MutationTransaction,
    activate_transaction_scope,
    get_active_execution_scope,
    get_active_transaction_scope,
    reset_transaction_scope,
)
from .tools import (
    EXECUTION_TOOL_NAMES,
    MUTATION_TOOL_NAMES,
    READ_ONLY_TOOL_NAMES,
    STRUCTURAL_TOOL_NAMES,
    build_repo_tools,
    build_run_command_tool,
    build_verify_repository_tool,
    build_write_file_tool,
)
from .utils import bounded_model_history, route_after_agent, tool_evidence, wrap_up_prompt

_REVISE_PLAN = "revise_plan"
_GET_FILE = "get_file"
_GET_SYMBOL = "get_symbol"
_FIND_FILES = "find_files"
_FIND_SYMBOL = "find_symbol"
_SEARCH_REPOSITORY = "search_repository"
_FIND_CALLERS = "find_callers"
_FIND_CALLEES = "find_callees"
_FIND_IMPORTERS = "find_importers"
_GET_DEPENDENCIES = "get_dependencies"
_GET_AST = "get_ast"
_WRITE_FILE = "write_file"
_VERIFY = "verify_repository"
_RUN_COMMAND = "run_command"

MAX_READ_CALLS_PER_TURN = 3

_READ_ONLY_PLAN_TOOLS = frozenset(READ_ONLY_TOOL_NAMES)
_STRUCTURAL_PLAN_TOOLS = frozenset(STRUCTURAL_TOOL_NAMES)
_WRITE_TOOLS = frozenset(MUTATION_TOOL_NAMES)
_EXECUTE_TOOLS = frozenset(EXECUTION_TOOL_NAMES)
_MUTATION_OR_EXECUTION_TOOLS = _WRITE_TOOLS | _EXECUTE_TOOLS
_PROGRESS_TOOLS = _READ_ONLY_PLAN_TOOLS | _MUTATION_OR_EXECUTION_TOOLS
_PLAN_AWARE_TOOL_NAMES = (*READ_ONLY_TOOL_NAMES, _WRITE_FILE, _VERIFY, _RUN_COMMAND, _REVISE_PLAN)
_CODE_VERIFICATION_TOOLS = (_WRITE_FILE, _VERIFY)

_CHECK_MAX_CHARS = 2000

_INVOCATION_TRACKER: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "ari_active_invocation_tracker", default=None
)


def _record_live_tx_state(tx_state: dict | None, *, task: str = "", concluded: bool = False) -> None:
    tracker = _INVOCATION_TRACKER.get()
    if isinstance(tracker, dict):
        if isinstance(tx_state, dict):
            tracker["tx_state"] = tx_state
        if task:
            tracker["task"] = task
        if concluded:
            tracker["concluded"] = True


def _empty_grader_metrics() -> dict[str, int]:
    return {
        "grader_calls": 0,
        "grader_cache_hits": 0,
        "grader_cache_misses": 0,
        "grader_latency_ms": 0,
        "grader_failures": 0,
        "criterion_acceptances": 0,
        "criterion_model_rejections": 0,
        "criterion_citation_rejections": 0,
        "grader_format_retries": 0,
        "grader_size_retries": 0,
        "grader_conclusion_calls": 0,
        "grader_conclusion_cache_hits": 0,
        "grader_conclusion_failures": 0,
        "grader_conclusion_format_retries": 0,
        "grader_conclusion_size_retries": 0,
        "criterion_answer_rejections": 0,
        "conclusion_model_rejections": 0,
        "conclusion_citation_rejections": 0,
        "conclusion_reference_rejections": 0,
    }


_RETRY_DELAY_RE = re.compile(
    r"try\s+again\s+in\s+(\d+(?:\.\d+)?)\s*(ms|milliseconds?|s|sec|seconds?)\b",
    re.IGNORECASE,
)
_PURPOSE_DETAIL_EXCLUSION_RE = re.compile(
    r"\b(?:implement\w*|calculat\w*|comput\w*|caller\w*|callee\w*|import\w*|function|method|def\b|"
    r"bug|error|traceback|test\w*|line\s+\d+)\b|[\w./\\-]+\.py\b",
    re.IGNORECASE,
)


def _parse_conclusion_retry_delay(error_text: str) -> float | None:
    """Parse a provider-supplied retry delay (in seconds) when 0 < delay <= 3.0s."""
    m = _RETRY_DELAY_RE.search(error_text or "")
    if not m:
        return None
    val = float(m.group(1))
    unit = m.group(2).lower()
    seconds = (val / 1000.0) if unit.startswith("m") else val
    return seconds if 0.0 < seconds <= 3.0 else None


def _find_readme_path(service: RepoService) -> str | None:
    """Return the repository-relative path of a root README file if present."""
    idx = getattr(service, "index", None)
    repo_files = getattr(idx, "repo_files", None) or []
    for candidate in ("README.md", "readme.md", "README.rst", "README.txt", "README"):
        for rf in repo_files:
            if rf.replace("\\", "/").casefold() == candidate.casefold():
                return rf.replace("\\", "/")
    root_dir = getattr(idx, "root", None)
    if root_dir:
        for candidate in ("README.md", "readme.md", "README.rst", "README.txt"):
            p = Path(root_dir) / candidate
            if p.is_file():
                return candidate
    return None


def _is_general_purpose_query(task: str, intent_floor: dict | None = None) -> bool:
    """True when the query asks for high-level project/model purpose or overview,
    not specific code implementation details or restricted sources."""
    floor = intent_floor if isinstance(intent_floor, dict) else intent_mod.deterministic_floor(task)
    if floor.get("kind") == intent_mod.KIND_CHANGE or floor.get("sources") or floor.get("run_checks"):
        return False
    text = " ".join((task or "").split())
    if not text or _PURPOSE_DETAIL_EXCLUSION_RE.search(text):
        return False
    return bool(
        intent_mod.is_purpose_query(text)
        or re.search(
            r"\b(?:main\s+purpose|project\s+overview|what\s+does\s+this\s+(?:repo|repository|project|model|system)\s+do)\b",
            text,
            re.IGNORECASE,
        )
    )


class RevisePlanInput(BaseModel):
    steps: list[str] = Field(
        description="Full replacement list of remaining plan steps (empty list drops all pending steps).",
    )


def build_deep_agent(
    service: RepoService,
    config: AgentConfig | None = None,
    llm=None,
    **_legacy_kwargs,
):
    """Return a compiled single-agent LangGraph graph for ARI:
        agent.invoke({"messages": [{"role": "user", "content": "..."}]})

    All transaction state, task labels, and grading caches/counters are stored in
    per-invocation `DeepAgentState` and bound via `ContextVar` during tool
    execution, so concurrent invocations sharing a compiled graph instance are
    strictly isolated.
    """
    if config is None:
        config = load_config(require_api_key=llm is None)
    if llm is None:
        llm = build_llm(config)

    def _tx_from_state(state: DeepAgentState, task: str = "") -> MutationTransaction:
        resolved_task = task or state.get("task") or _task_from_messages(state.get("messages") or [])
        return MutationTransaction.from_state_dict(
            service, config, state.get("tx_state"), label=resolved_task
        )

    def _metrics_from_state(state: DeepAgentState) -> dict[str, int]:
        base = _empty_grader_metrics()
        current = state.get("grader_metrics")
        if isinstance(current, dict):
            for k in base:
                if isinstance(current.get(k), int):
                    base[k] = current[k]
        return base

    def _assessment_cache_from_state(state: DeepAgentState) -> dict[str, dict]:
        cache = state.get("assessment_cache")
        return dict(cache) if isinstance(cache, dict) else {}

    def _conclusion_cache_from_state(state: DeepAgentState) -> dict[str, dict]:
        cache = state.get("conclusion_cache")
        return dict(cache) if isinstance(cache, dict) else {}

    def _guarded_write_file(path: str, content: str) -> dict:
        scope = get_active_execution_scope()
        if scope is None or not scope.write_allowed:
            msg = (
                "MUTATION_BLOCKED (system data): this execution context does not have WRITE "
                "capability, so 'write_file' was not run and the repository was NOT modified."
            )
            return {"state": FAILED, "report": msg, "error": {"code": "capability_denied", "message": msg}}
        if isinstance(scope.intent, dict):
            allowed, note = intent_mod.source_allows_tool(scope.intent, _WRITE_FILE, args={"path": path}, service=service)
            if not allowed:
                return {"state": FAILED, "report": note, "error": {"code": "source_restriction", "message": note}}
        tx, _ = get_active_transaction_scope(service, config)
        return tx.execute_write(path, content, label=tx.label)

    def _guarded_verify_repository() -> dict:
        scope = get_active_execution_scope()
        if scope is None or not scope.execute_allowed:
            msg = (
                "EXECUTION_BLOCKED (system data): this execution context does not have EXECUTE "
                "capability, so 'verify_repository' was not run and no repository command was executed."
            )
            return {"state": FAILED, "report": msg, "error": {"code": "capability_denied", "message": msg}}
        if isinstance(scope.intent, dict):
            allowed, note = intent_mod.source_allows_tool(scope.intent, _VERIFY, args={}, service=service)
            if not allowed:
                return {"state": FAILED, "report": note, "error": {"code": "source_restriction", "message": note}}
        tx, write_allowed = get_active_transaction_scope(service, config)
        return tx.execute_verify(write_allowed=write_allowed)

    def _guarded_run_command(command: str, timeout: int = 60) -> dict:
        scope = get_active_execution_scope()
        if scope is None or not scope.execute_allowed:
            msg = (
                "EXECUTION_BLOCKED (system data): this execution context does not have EXECUTE "
                "capability, so 'run_command' was not run and no repository command was executed."
            )
            return {"state": FAILED, "report": msg, "error": {"code": "capability_denied", "message": msg}}
        if isinstance(scope.intent, dict):
            allowed, note = intent_mod.source_allows_tool(
                scope.intent, _RUN_COMMAND, args={"command": command, "timeout": timeout}, service=service
            )
            if not allowed:
                return {"state": FAILED, "report": note, "error": {"code": "source_restriction", "message": note}}
        tx, write_allowed = get_active_transaction_scope(service, config)
        return tx.execute_command(command, timeout=timeout, write_allowed=write_allowed)

    def revise_plan(steps: list[str]) -> str:
        cleaned = [s.strip() for s in steps if isinstance(s, str) and s.strip()]
        cleaned = cleaned[: config.max_plan_steps]
        return json.dumps({"requested_steps": cleaned})

    revise_plan_tool = StructuredTool.from_function(
        func=revise_plan,
        name=_REVISE_PLAN,
        description=(
            "Replace remaining pending plan steps with a full updated list (empty list drops superseded "
            "pending steps). Updates planning state only; criteria still require recorded evidence."
        ),
        args_schema=RevisePlanInput,
    )

    repo_tools = build_repo_tools(service, config.max_observation_chars)
    write_tool = build_write_file_tool(
        service, config.max_observation_chars, write_fn=_guarded_write_file
    )
    verify_tool = build_verify_repository_tool(
        _guarded_verify_repository, config.max_observation_chars
    )
    run_command_tool = build_run_command_tool(
        service, config.max_observation_chars, run_command_fn=_guarded_run_command
    )

    all_tools = [*repo_tools, write_tool, verify_tool, run_command_tool, revise_plan_tool]
    tools_by_caps = {
        (False, False): [*repo_tools, revise_plan_tool],
        (True, False): [*repo_tools, write_tool, revise_plan_tool],
        (False, True): [*repo_tools, verify_tool, run_command_tool, revise_plan_tool],
        (True, True): all_tools,
    }
    llm_by_caps = {
        caps_key: llm.bind_tools(tool_list)
        for caps_key, tool_list in tools_by_caps.items()
    }

    def invocation_metrics(state: DeepAgentState, grader_metrics: dict[str, int] | None = None) -> dict:
        active_grader = grader_metrics if grader_metrics is not None else _metrics_from_state(state)
        start_metrics = state.get("grader_metrics_start") or {}
        grader_delta = {
            key: value - start_metrics.get(key, 0)
            for key, value in active_grader.items()
        }
        control = dict(state.get("control_metrics") or {})
        control["deterministic_followups"] = list(control.get("deterministic_followups") or [])
        attempts = execution_mod.finished_attempts(state.get("execution"))
        control["execution_attempts"] = len(attempts)
        control["delegated_attempts"] = 0
        control["retried_attempts"] = sum(1 for a in attempts if a.get("retry_of"))
        return {**grader_delta, **control}

    def conclusion_candidate(
        text: str,
        criteria: list[dict],
        task: str = "",
        plan: dict | None = None,
        observations: Any = None,
    ) -> str:
        clean = text.strip()
        if not clean and (task or plan or observations):
            clean = ev.synthesize_deterministic_answer(task, plan, observations)
        return clean

    def record_conclusion_rejections(result: dict, grader_metrics: dict[str, int]) -> None:
        metric_by_source = {
            "model": "conclusion_model_rejections",
            "citation": "conclusion_citation_rejections",
            "deterministic_reference_check": "conclusion_reference_rejections",
        }
        for claim in result.get("claims", []):
            if not claim.get("supported"):
                metric = metric_by_source.get(claim.get("validation_source"))
                if metric:
                    grader_metrics[metric] = grader_metrics.get(metric, 0) + 1

    def validate_conclusion(
        task: str,
        answer: str,
        criteria: list[dict],
        plan,
        corpus: str,
        grader_metrics: dict[str, int],
        conclusion_cache: dict[str, dict],
    ) -> dict:
        """Check each final factual claim and validate its evidence citations."""
        usable = ev.usable_evidence_steps(plan, criteria)
        if not answer.strip():
            answer = ev.synthesize_deterministic_answer(task, plan, corpus)
        if not answer.strip():
            res = {"supported": False, "claims": [], "error": "no answer text was produced"}
            log_conclusion_validation_diagnostics(res, cache_hit=False, format_retried=False)
            return res
        if not usable:
            res = {"supported": False, "claims": [], "error": "no usable evidence was recorded"}
            log_conclusion_validation_diagnostics(res, cache_hit=False, format_retried=False)
            return res
        payload = {
            "objective": task,
            "proposed_answer": answer,
            "accepted_criteria": [
                {"id": c["id"], "description": c["description"]}
                for c in criteria if c.get("satisfied")
            ],
            "evidence": [],
        }
        full_items = ev.evidence_for_conclusion(plan, criteria)
        fixed_chars = len(CONCLUSION_CHECK_PROMPT) + len(json.dumps(dict(payload, evidence=[]), ensure_ascii=True))
        fit_report: dict = {}
        payload["evidence"] = _fit_evidence_items(
            full_items, fixed_chars, report_out=fit_report, stage="conclusion_validation"
        )
        if not fit_report.get("fits", True) or fit_report.get("essential_unfittable"):
            grader_metrics["grader_conclusion_failures"] += 1
            budget_err = (
                "conclusion payload exceeds context budget: "
                + str(fit_report.get("reason") or "essential metadata and fixed instructions cannot fit")
            )
            err_res = {"supported": False, "claims": [], "error": budget_err}
            log_conclusion_validation_diagnostics(
                err_res, cache_hit=False, format_retried=False, size_retried=False, payload_fit=fit_report
            )
            return err_res

        fingerprint = hashlib.sha256(json.dumps(
            {"repository": f"{service.index.root}:{id(service.index)}", "conclusion": payload,
             "reference_corpus": corpus},
            ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")).hexdigest()
        cached = conclusion_cache.get(fingerprint)
        if cached is not None:
            grader_metrics["grader_conclusion_cache_hits"] += 1
            record_conclusion_rejections(cached, grader_metrics)
            log_conclusion_validation_diagnostics(
                cached, cache_hit=True, format_retried=False, size_retried=False, payload_fit=fit_report
            )
            return cached

        format_retried = False
        size_retried = False
        rate_limit_retried = False
        active_retry_kind: str | None = None  # None | "format" | "size" | "rate_limit"
        size_retry_error = ""
        parsed = None
        while True:
            if not fit_report.get("fits", True) or fit_report.get("essential_unfittable"):
                grader_metrics["grader_conclusion_failures"] += 1
                budget_err = (
                    "conclusion payload exceeds context budget: "
                    + str(fit_report.get("reason") or "essential metadata and fixed instructions cannot fit")
                )
                err_res = {"supported": False, "claims": [], "error": budget_err}
                log_conclusion_validation_diagnostics(
                    err_res, cache_hit=False, format_retried=format_retried,
                    size_retried=size_retried, payload_fit=fit_report,
                )
                return err_res
            try:
                system_prompt = CONCLUSION_CHECK_PROMPT
                if active_retry_kind == "format":
                    system_prompt += (
                        "\nYour previous reply was not valid JSON in the required shape. "
                        "Return the required JSON object now."
                    )
                    stage = "conclusion_validation_retry"
                    fmt_attempt = 2
                elif active_retry_kind == "size":
                    stage = "conclusion_validation_size_retry"
                    fmt_attempt = 1
                else:
                    stage = "conclusion_validation"
                    fmt_attempt = 1
                grader_metrics["grader_conclusion_calls"] += 1
                reply = invoke_model_logged(
                    llm,
                    [SystemMessage(system_prompt),
                     HumanMessage(json.dumps(payload, ensure_ascii=True))],
                    agent="Main",
                    stage=stage,
                    invoke_config={"metadata": {"ari_stage": stage}},
                    api_key=config.api_key,
                    format_attempt=fmt_attempt,
                    max_format_attempts=2,
                )
                if size_retried and size_retry_error:
                    record_stage_size_recovery(
                        agent="Main",
                        stage="conclusion_validation",
                        recovered=True,
                        original_error=size_retry_error,
                        fit_report=fit_report,
                    )
                    size_retry_error = ""
                parsed = ev.parse_conclusion_assessment(_text(reply.content))
                if parsed is not None:
                    break
                if not format_retried:
                    retry_prompt = (
                        CONCLUSION_CHECK_PROMPT
                        + "\nYour previous reply was not valid JSON in the required shape. "
                        "Return the required JSON object now."
                    )
                    retry_fixed_chars = len(retry_prompt) + len(
                        json.dumps(dict(payload, evidence=[]), ensure_ascii=True)
                    )
                    retry_scale = 0.5 if size_retried else 1.0
                    payload["evidence"] = _fit_evidence_items(
                        full_items, retry_fixed_chars, scale=retry_scale,
                        report_out=fit_report, stage="conclusion_validation_retry",
                    )
                    format_retried = True
                    active_retry_kind = "format"
                    grader_metrics["grader_conclusion_format_retries"] += 1
                    continue
                break
            except Exception as exc:
                err_reason = _model_error_reason(exc, config.api_key)
                if not size_retried and _is_oversized_request_error(exc, config.api_key, err_reason):
                    size_retried = True
                    active_retry_kind = "size"
                    size_retry_error = err_reason
                    payload["evidence"] = _fit_evidence_items(
                        full_items, fixed_chars, scale=0.5, report_out=fit_report,
                        stage="conclusion_validation_size_retry",
                    )
                    if not fit_report.get("fits", True) or fit_report.get("essential_unfittable"):
                        grader_metrics["grader_conclusion_failures"] += 1
                        budget_err = (
                            f"{err_reason}; reduced conclusion payload still exceeds context budget "
                            f"({fit_report.get('reason') or 'essential metadata and fixed instructions cannot fit'})"
                        )
                        err_res = {"supported": False, "claims": [], "error": budget_err}
                        log_conclusion_validation_diagnostics(
                            err_res, cache_hit=False, format_retried=format_retried,
                            size_retried=size_retried, payload_fit=fit_report,
                        )
                        return err_res
                    grader_metrics["grader_conclusion_size_retries"] += 1
                    record_stage_size_recovery(
                        agent="Main",
                        stage="conclusion_validation",
                        recovered=False,
                        original_error=err_reason,
                        fit_report=fit_report,
                    )
                    continue
                if "429" in err_reason or "rate limit" in err_reason.lower():
                    retry_delay = _parse_conclusion_retry_delay(str(exc)) or _parse_conclusion_retry_delay(err_reason)
                    if not rate_limit_retried and retry_delay is not None:
                        rate_limit_retried = True
                        active_retry_kind = "rate_limit"
                        time.sleep(retry_delay)
                        continue
                    reference_issues = ev.conclusion_reference_issues(answer, plan, corpus)
                    claims_list = []
                    if reference_issues:
                        grader_metrics["conclusion_reference_rejections"] = (
                            grader_metrics.get("conclusion_reference_rejections", 0) + 1
                        )
                        claims_list.append({
                            "claim": "The draft contains unsupported references",
                            "supported": False,
                            "evidence_ids": [],
                            "reason": "; ".join(reference_issues),
                            "validation_source": "deterministic_reference_check",
                        })
                    err_res = {
                        "supported": False,
                        "claims": claims_list,
                        "error": f"{err_reason}; final claim validation did not complete",
                        "rate_limited": True,
                        "validation_status": "validation_unavailable: rate_limit",
                        "reference_check_passed": not bool(reference_issues),
                    }
                    grader_metrics["grader_conclusion_failures"] += 1
                    log_conclusion_validation_diagnostics(
                        err_res, cache_hit=False, format_retried=format_retried,
                        size_retried=size_retried, payload_fit=fit_report,
                    )
                    return err_res
                grader_metrics["grader_conclusion_failures"] += 1
                err_res = {"supported": False, "claims": [], "error": err_reason}
                log_conclusion_validation_diagnostics(
                    err_res, cache_hit=False, format_retried=format_retried,
                    size_retried=size_retried, payload_fit=fit_report,
                )
                return err_res

        if parsed is None:
            result = {"supported": False, "claims": [],
                      "error": "unparseable conclusion-check reply after one format retry"}
            grader_metrics["grader_conclusion_failures"] += 1
        else:
            usable_ids = {step["step_id"] for step in usable}
            claims = []
            all_ids: list[str] = []
            for item in parsed["claims"]:
                submitted = item["evidence_ids"]
                valid = ev.resolve_evidence_ids(submitted, plan, usable_ids)
                citations_valid = bool(valid) and len(valid) == len(set(submitted))
                supported = item["supported"] is True and citations_valid
                reason = item.get("reason", "")
                validation_source = "model" if item["supported"] is not True else "accepted"
                if item["supported"] is True and not citations_valid:
                    reason = "the conclusion check cited missing or unusable evidence items"
                    validation_source = "citation"
                certainty = item.get("certainty")
                if supported:
                    certainty = "confirmed"
                elif (certainty == "possible" and citations_valid
                      and item["supported"] is False):
                    certainty = "possible"
                elif certainty == "possible" and not citations_valid:
                    reason = "the proposed possibility cited missing or unusable evidence items"
                    validation_source = "citation"
                    certainty = "unknown"
                else:
                    certainty = "unknown"
                reference_issues = ev.conclusion_reference_issues(item["claim"], plan, corpus)
                if reference_issues:
                    supported = False
                    certainty = "unknown"
                    reason = "; ".join(reference_issues)
                    validation_source = "deterministic_reference_check"
                claims.append({"claim": item["claim"], "supported": supported,
                               "evidence_ids": valid, "reason": reason,
                               "validation_source": validation_source, "certainty": certainty})
                if supported or certainty == "possible":
                    all_ids.extend(valid)
            result = {"supported": bool(claims) and all(item["supported"] for item in claims),
                      "claims": claims, "evidence_ids": list(dict.fromkeys(all_ids))}
        reference_issues = ev.conclusion_reference_issues(answer, plan, corpus)
        if reference_issues:
            result["supported"] = False
            result.setdefault("claims", []).append({
                "claim": "The draft contains unsupported references",
                "supported": False,
                "evidence_ids": [],
                "reason": "; ".join(reference_issues),
                "validation_source": "deterministic_reference_check",
            })
        if not result.get("error"):
            conclusion_cache[fingerprint] = result
        record_conclusion_rejections(result, grader_metrics)
        log_conclusion_validation_diagnostics(
            result, cache_hit=False, format_retried=format_retried,
            size_retried=size_retried, payload_fit=fit_report,
        )
        return result

    def plan_task(state: DeepAgentState) -> dict:
        ensure_run_context(config.api_key)
        task = state.get("task") or _task_from_messages(state["messages"])
        tx = MutationTransaction(service, config, label=task)
        initial_grader_metrics = _empty_grader_metrics()

        intent_floor = intent_mod.deterministic_floor(task)
        planning_error = None
        planning_parse_errors = 0
        criteria: list[dict] = []
        model_intent_raw = None
        descriptions: list[str] = []
        readme_for_purpose = _find_readme_path(service) if _is_general_purpose_query(task, intent_floor) else None
        if readme_for_purpose is not None:
            model_intent_raw = {"depth": intent_mod.DEPTH_MINIMAL, "rigor": intent_mod.RIGOR_LIGHT}
            descriptions = [f"Read {readme_for_purpose} with get_file to determine the documented project purpose and overview"]
            criteria = ev.default_criteria(task)
        elif intent_mod.is_lookup_task(task, intent_floor):
            model_intent_raw = {"depth": intent_mod.DEPTH_MINIMAL, "rigor": intent_mod.RIGOR_LIGHT}
        else:
            try:
                reply = invoke_model_logged(
                    llm,
                    [SystemMessage(PLANNING_SYSTEM_PROMPT),
                     HumanMessage(json.dumps({"objective": task,
                                              "max_steps": config.max_plan_steps,
                                              "max_criteria": ev.MAX_CRITERIA,
                                              "intent_floor": intent_floor}, ensure_ascii=True))],
                    agent="Main",
                    stage="planning",
                    invoke_config={"metadata": {"ari_stage": "planning"}},
                    api_key=config.api_key,
                )
                reply_text = _text(reply.content)
                descriptions = planning.parse_plan_steps(reply_text, config.max_plan_steps)
                criteria = ev.parse_criteria(reply_text)
                model_intent_raw = intent_mod.parse_intent_reply(reply_text)
                planning_parse_errors += int(not descriptions)
                planning_parse_errors += int(not criteria)
                planning_parse_errors += int(model_intent_raw is None)
            except Exception as exc:
                descriptions = []
                planning_error = f"planning model call failed: {_model_error_reason(exc, config.api_key)}"
        if not descriptions:
            descriptions = [planning.fallback_step_description(task)]
        if not criteria:
            criteria = ev.default_criteria(task)
        intent = intent_mod.resolve_intent(task, model_intent_raw, floor=intent_floor)
        descriptions = descriptions[: intent_mod.plan_step_limit(intent, config.max_plan_steps)]
        criteria = criteria[: intent_mod.criteria_limit(intent, ev.MAX_CRITERIA)]
        capabilities = intent_mod.capabilities_for_floor(intent_floor)
        seeded_plan = planning.seed_pending_steps(planning.new_plan(task), descriptions)
        log_plan_initialized(seeded_plan, criteria, intent)
        tx_dict = tx.to_state_dict()
        _record_live_tx_state(tx_dict, task=task)
        return {
            "task": task,
            "intent": intent,
            "capabilities": capabilities,
            "plan": seeded_plan,
            "active_step_ids": {},
            "criteria": criteria,
            "completion": None,
            "completion_retries": 0,
            "completion_retry_progress_generation": None,
            "model_error": planning_error,
            "evidence": [],
            "observations": [],
            "current_objective": None,
            "delegated_result": None,
            "tx_state": tx_dict,
            "transaction": tx.snapshot(),
            "verification_info": None,
            "failure_info": None,
            "final_result": None,
            "outcome": None,
            "step_count": 0,
            "grader_metrics": dict(initial_grader_metrics),
            "grader_metrics_start": dict(initial_grader_metrics),
            "assessment_cache": {},
            "conclusion_cache": {},
            "control_metrics": {
                "deterministic_followups": [],
                "deterministic_followup_count": 0,
                "planning_parse_errors": planning_parse_errors,
                "no_progress_stops": 0,
                "repeated_action_blocks": 0,
            },
            "execution": execution_mod.empty_execution(),
            "result_fingerprints": [],
            "progress_generation": 0,
            "no_progress_count": 0,
            "blocked_read_actions": [],
            "pending_followup": None,
        }

    def grade(
        state: DeepAgentState,
        task: str,
        plan,
        grader_metrics: dict[str, int] | None = None,
        assessment_cache: dict[str, dict] | None = None,
    ) -> tuple[list[dict], dict, str, dict[str, int], dict[str, dict]]:
        """Grade acceptance criteria against recorded evidence and resolve
        superseded read-only plan steps when all criteria are established."""
        plan = plan or planning.new_plan(task)
        metrics_local = grader_metrics if grader_metrics is not None else _metrics_from_state(state)
        cache_local = assessment_cache if assessment_cache is not None else _assessment_cache_from_state(state)
        criteria, plan = _assess(
            llm, task, state.get("criteria") or ev.default_criteria(task), plan,
            state.get("verification_info"), cache_local, metrics_local,
            repository_identity=f"{service.index.root}:{id(service.index)}",
            api_key=config.api_key,
        )
        support = ev.classify_support(criteria, plan, state.get("failure_info"), state.get("verification_info"))
        intent_for_grade = intent_mod.get_intent(state)
        capabilities_for_grade = state.get("capabilities") or intent_mod.default_capabilities()
        if ev.can_drop_pending_readonly_steps(
            support,
            criteria,
            plan,
            light_readonly=_is_light_readonly_task(intent_for_grade, plan),
            failure_info=state.get("failure_info"),
            verification_info=state.get("verification_info"),
            intent=intent_for_grade,
            capabilities=capabilities_for_grade,
        ):
            dropped = planning.pending_steps(plan)
            plan = planning.revise_pending_steps(
                plan,
                [],
                reason="All requested criteria are supported by recorded evidence; remaining read-only steps are superseded",
            )
            for s in dropped:
                log_plan_step_transition(
                    s.get("step_id"), "skipped", s.get("description", ""),
                    stage="evidence_grading",
                    reason="All requested criteria are supported by recorded evidence; remaining read-only steps are superseded",
                )
            support = ev.classify_support(criteria, plan, state.get("failure_info"), state.get("verification_info"))
        return criteria, plan, support, metrics_local, cache_local

    def conclude(
        state: DeepAgentState,
        plan,
        criteria,
        support: str,
        final_text: str,
        step_limit_reached: bool,
        extra_unresolved: tuple = (),
        answer_check: dict | None = None,
        termination_reason: str = "run ended with repairs still outstanding",
        grader_metrics: dict[str, int] | None = None,
        assessment_cache: dict[str, dict] | None = None,
        conclusion_cache: dict[str, dict] | None = None,
    ) -> dict:
        """Finalize any unrepaired transaction, validate the conclusion draft,
        and build the grounded terminal response."""
        task = state.get("task") or _task_from_messages(state.get("messages") or [])
        tx = _tx_from_state(state, task)
        metrics_local = grader_metrics if grader_metrics is not None else _metrics_from_state(state)
        a_cache = assessment_cache if assessment_cache is not None else _assessment_cache_from_state(state)
        c_cache = conclusion_cache if conclusion_cache is not None else _conclusion_cache_from_state(state)

        final_v_info, final_f_info = tx.finalize_if_unrepaired(termination_reason=termination_reason)
        eff_v_info = final_v_info if final_v_info is not None else state.get("verification_info")
        eff_f_info = final_f_info if final_f_info is not None else state.get("failure_info")
        if final_f_info is not None:
            support = ev.classify_support(criteria, plan, eff_f_info, eff_v_info)

        corpus = "\n".join(list(state.get("observations", [])) + [s.get("evidence", "") for s in ev.usable_evidence_steps(plan, criteria)])
        should_check_partial_answer = (
            support in (ev.PARTIAL, ev.UNSUPPORTED)
            and bool(final_text.strip())
            and bool(ev.usable_evidence_steps(plan))
        )
        if answer_check is None and (support == ev.SUPPORTED or should_check_partial_answer):
            answer = conclusion_candidate(final_text, criteria, task=task, plan=plan, observations=state.get("observations", []))
            final_text = answer
            answer_check = validate_conclusion(
                task, answer, criteria, plan, corpus, metrics_local, c_cache
            )
        if support == ev.SUPPORTED and answer_check is not None and answer_check.get("error"):
            # Evidence may support the criteria, but a draft is not a validated
            # answer when its required claim check failed or was rate-limited.
            support = ev.PARTIAL
        if (support == ev.SUPPORTED and answer_check is not None
                and not answer_check.get("error") and not answer_check.get("supported")):
            metrics_local["criterion_answer_rejections"] += 1
        response, unresolved = ev.build_final_response(
            support, criteria, plan, final_text, eff_f_info, eff_v_info, corpus,
            extra_unresolved=extra_unresolved, answer_check=answer_check,
        )
        outcome = planning.classify_outcome(
            plan, eff_f_info, step_limit_reached=step_limit_reached,
            has_evidence=bool(state.get("observations")), objective_satisfied=(support == ev.SUPPORTED),
            verification_info=eff_v_info,
        )
        if support == ev.BLOCKED:
            outcome = planning.OUTCOME_FAILED
        record_completion_diagnostics(
            support=support,
            outcome=outcome,
            criteria=criteria,
            plan=plan,
            failure_info=eff_f_info,
            verification_info=eff_v_info,
            model_error=state.get("model_error"),
            answer_check=answer_check,
            step_limit_reached=step_limit_reached,
            extra_unresolved=extra_unresolved,
            incomplete_reasons_list=ev.incomplete_reasons(criteria, plan),
        )
        tx_dict = tx.to_state_dict()
        _record_live_tx_state(tx_dict, task=task, concluded=True)
        return {
            "plan": plan,
            "active_step_ids": {},
            "criteria": criteria,
            "outcome": outcome,
            "current_objective": None,
            "tx_state": tx_dict,
            "transaction": tx.snapshot(),
            "verification_info": eff_v_info,
            "failure_info": eff_f_info,
            "final_result": response,
            "completion": ev.build_completion(support, criteria, unresolved, answer_check, plan, corpus),
            "grader_metrics": metrics_local,
            "assessment_cache": a_cache,
            "conclusion_cache": c_cache,
            "runtime_metrics": invocation_metrics(state, metrics_local),
        }

    def decide(state: DeepAgentState) -> dict:
        task = state.get("task") or _task_from_messages(state["messages"])
        plan = state.get("plan")
        steps = state.get("step_count", 0)
        intent = intent_mod.get_intent(state)
        step_budget = intent_mod.agent_step_budget(intent, config.max_agent_steps)

        if steps >= step_budget:
            wrap_up_error = None
            try:
                reply = invoke_model_logged(
                    llm,
                    [SystemMessage(DEEP_AGENT_SYSTEM_PROMPT),
                     HumanMessage(wrap_up_prompt(state["messages"], step_budget, task=task))],
                    agent="Main",
                    stage="wrap_up",
                    invoke_config={"metadata": {"ari_stage": "wrap_up"}},
                    api_key=config.api_key,
                    turn=steps + 1,
                )
                model_error = state.get("model_error")
            except Exception as exc:
                reply = AIMessage(content="")
                model_error = f"wrap-up model call failed: {_model_error_reason(exc, config.api_key)}"
                wrap_up_error = model_error
            final_text = _reply_text(reply)
            if not final_text.strip():
                final_text = ev.synthesize_deterministic_answer(task, plan, state.get("observations", []))
            wrap_state = dict(state, model_error=model_error) if model_error else state
            criteria, plan, support, g_metrics, a_cache = grade(wrap_state, task, plan)
            update = conclude(
                wrap_state, plan, criteria, support, final_text,
                step_limit_reached=True,
                extra_unresolved=(wrap_up_error,) if wrap_up_error else (),
                termination_reason=f"step limit ({step_budget}) reached",
                grader_metrics=g_metrics,
                assessment_cache=a_cache,
            )
            update.update({
                "messages": [AIMessage(content=update["final_result"])],
                "task": task,
                "step_count": steps + 1,
                "model_error": model_error,
            })
            return update

        system_messages = [SystemMessage(DEEP_AGENT_SYSTEM_PROMPT)]
        plan_status = planning.describe_plan(plan)
        guidance_status = _describe_investigation_guidance(
            task, intent, plan, state.get("criteria", []), state,
            steps_left=step_budget - steps,
            service=service,
        )
        if guidance_status:
            plan_status = f"{plan_status}\n\n{guidance_status}" if plan_status else guidance_status
        if plan_status:
            system_messages.append(HumanMessage("Plan status data:\n" + plan_status))
        criteria_status = ev.describe_criteria(state.get("criteria", []), plan)
        if criteria_status:
            system_messages.append(HumanMessage("Acceptance criteria and evidence data:\n" + criteria_status))

        try:
            prior_err = str(state.get("model_error") or "")
            if steps == 0 and prior_err.startswith("planning model call failed:") and any(
                marker in prior_err.lower()
                for marker in ("401", "403", "429", "rate limit", "auth", "api_key", "connection", "api key")
            ):
                raise RuntimeError(prior_err.removeprefix("planning model call failed:").strip())
            caps_for_turn = state.get("capabilities") or intent_mod.default_capabilities()
            active_llm_with_tools = llm_by_caps[
                (bool(caps_for_turn.get("write")), bool(caps_for_turn.get("execute")))
            ]
            model_history = bounded_model_history(
                state["messages"], task=task,
                max_turns=min(config.max_agent_steps, 6),
            )
            reply = invoke_model_logged(
                active_llm_with_tools,
                system_messages + model_history,
                agent="Main",
                stage="decision",
                invoke_config={"metadata": {"ari_stage": "decision"}},
                api_key=config.api_key,
                turn=steps + 1,
            )
        except Exception as exc:
            reason = _model_error_reason(exc, config.api_key)
            message = f"The agent stopped because the model request failed: {reason}."
            error = f"decision model call failed: {reason}"
            failed_state = dict(state, model_error=error)
            if ev.usable_evidence_steps(plan):
                criteria, graded_plan, support, g_metrics, a_cache = grade(failed_state, task, plan)
                if support == ev.SUPPORTED:
                    support = ev.PARTIAL
                det_draft = ev.synthesize_deterministic_answer(task, graded_plan or plan, state.get("observations", []))
                fields = conclude(
                    failed_state, graded_plan, criteria, support, det_draft,
                    step_limit_reached=False, extra_unresolved=(message,),
                    answer_check={"supported": False, "claims": [], "error": reason},
                    termination_reason=f"model request failed ({reason})",
                    grader_metrics=g_metrics,
                    assessment_cache=a_cache,
                )
                return {**fields, "messages": [AIMessage(content=fields["final_result"])], "task": task,
                        "step_count": steps + 1, "model_error": error}
            criteria = state.get("criteria") or ev.default_criteria(task)
            plan = plan or planning.new_plan(task)
            fields = conclude(
                failed_state, plan, criteria, ev.UNSUPPORTED, "",
                step_limit_reached=False, extra_unresolved=(message,),
                termination_reason=f"model request failed ({reason})",
            )
            return {**fields, "messages": [AIMessage(content=fields["final_result"])], "task": task,
                    "step_count": steps + 1, "current_objective": None, "model_error": error}

        update: dict = {"messages": [reply], "task": task, "step_count": steps + 1}
        raw_calls = list(getattr(reply, "tool_calls", None) or [])
        calls, batch_note = _normalize_turn_tool_calls(raw_calls)
        if batch_note:
            reply = AIMessage(content=f"{_text(reply.content)}\n\n{batch_note}".strip(), tool_calls=calls)
            update["messages"] = [reply]

        action_calls = [c for c in calls if c.get("name") != _REVISE_PLAN]
        revise_calls = [c for c in calls if c.get("name") == _REVISE_PLAN]

        if action_calls:
            capabilities = state.get("capabilities") or intent_mod.default_capabilities()
            valid_action_calls: list[dict] = []
            first_guard_return: dict | None = None
            blocked_list = list(state.get("blocked_read_actions") or [])
            metrics = dict(state.get("control_metrics") or {})

            for call in action_calls:
                tool_name = str(call.get("name") or "")
                args = call.get("args") if isinstance(call.get("args"), dict) else {}
                objective_text = _action_arg_for_call(tool_name, args)
                if (ctx_g := get_run_context()) is not None:
                    ctx_g.current_tool_target = objective_text

                allowed_by_source, source_note = intent_mod.source_allows_tool(
                    intent, tool_name, args=args, service=service
                )
                if not allowed_by_source:
                    log_guard_event("source_restriction", tool_name, source_note, target=objective_text)
                    if first_guard_return is None:
                        first_guard_return = {
                            "messages": [
                                AIMessage(content=_text(reply.content) or f"(restricted {tool_name} skipped)"),
                                HumanMessage(source_note),
                            ],
                            "task": task,
                            "step_count": steps + 1,
                            "current_objective": None,
                        }
                    continue

                if tool_name in _WRITE_TOOLS and not capabilities.get("write"):
                    note = (
                        "MUTATION_BLOCKED (system data): this execution context does not have WRITE "
                        f"capability, so {tool_name!r} was not run and the repository was "
                        "NOT modified - no file was created, changed, deleted or renamed. The original "
                        "request was not recognized as a change/implementation task. If the user genuinely "
                        "wants a repository change, they need to ask for it explicitly; otherwise continue "
                        "with read-only tools or give a final answer."
                    )
                    log_guard_event("mutation_blocked", tool_name, note, target=objective_text)
                    if first_guard_return is None:
                        first_guard_return = {
                            "messages": [
                                AIMessage(content=_text(reply.content) or "(mutation blocked - no WRITE capability)"),
                                HumanMessage(note),
                            ],
                            "task": task,
                            "step_count": steps + 1,
                            "current_objective": None,
                        }
                    continue

                if tool_name in _EXECUTE_TOOLS and not capabilities.get("execute", False):
                    note = (
                        "EXECUTION_BLOCKED (system data): this execution context does not have EXECUTE "
                        f"capability, so {tool_name!r} was not run and no repository command "
                        "was executed. The original request did not ask for tests or checks to be run. If "
                        "the user wants that, they need to ask explicitly (for example 'run the tests'); "
                        "otherwise continue with read-only tools or give a final answer."
                    )
                    log_guard_event("execution_blocked", tool_name, note, target=objective_text)
                    if first_guard_return is None:
                        first_guard_return = {
                            "messages": [
                                AIMessage(content=_text(reply.content) or "(execution blocked - no EXECUTE capability)"),
                                HumanMessage(note),
                            ],
                            "task": task,
                            "step_count": steps + 1,
                            "current_objective": None,
                        }
                    continue

                repeat_info = (
                    _repeated_read_action(
                        plan, tool_name, objective_text,
                        state.get("progress_generation", 0), id(service.index),
                        args=args,
                    )
                    if tool_name in _READ_ONLY_PLAN_TOOLS
                    else None
                )
                if repeat_info is not None:
                    repeat_key, repeat_hint = repeat_info
                    if len(action_calls) > 1:
                        # In a multi-call batch, skip the redundant call while allowing
                        # any non-redundant calls in the same batch to run.
                        if repeat_key not in blocked_list:
                            blocked_list.append(repeat_key)
                        metrics["repeated_action_blocks"] = metrics.get("repeated_action_blocks", 0) + 1
                        log_guard_event("repeated_read_action", tool_name, repeat_hint, target=objective_text)
                        if first_guard_return is None:
                            first_guard_return = _build_repeat_guard_return(
                                state, task, plan, reply, tool_name, objective_text,
                                repeat_key, repeat_hint, blocked_list, metrics,
                                steps, step_budget, intent, grade, conclude, invocation_metrics,
                            )
                        continue
                    return _build_repeat_guard_return(
                        state, task, plan, reply, tool_name, objective_text,
                        repeat_key, repeat_hint, blocked_list, metrics,
                        steps, step_budget, intent, grade, conclude, invocation_metrics,
                    )

                if (tool_name not in _READ_ONLY_PLAN_TOOLS and tool_name != _WRITE_FILE and plan
                        and ev.is_blind_repeat(plan, tool_name, objective_text, _WRITE_FILE)):
                    note = (
                        "Repeat check (system data): that exact action already ran without producing usable "
                        "evidence and the repository has not changed since, so it was not run again. Choose a "
                        "different, more targeted approach, or give a final answer stating what could not be "
                        "established."
                    )
                    log_guard_event("blind_repeat", tool_name, note, target=objective_text)
                    if first_guard_return is None:
                        first_guard_return = {
                            "messages": [
                                AIMessage(content=_text(reply.content) or "(repeated action skipped)"),
                                HumanMessage(note),
                            ],
                            "task": task,
                            "step_count": steps + 1,
                            "current_objective": None,
                        }
                    continue

                if tool_name in _READ_ONLY_PLAN_TOOLS and _redundant_investigation_guard(state, plan):
                    note = (
                        "Sufficiency check (system data): every acceptance criterion is already supported by "
                        "recorded evidence, so further read-only investigation was not run. Give your final "
                        "answer now, or use write_file/verify_repository if a requested modification or "
                        "verification is still needed."
                    )
                    log_guard_event("redundant_investigation", tool_name, note, target=objective_text)
                    if first_guard_return is None:
                        first_guard_return = {
                            "messages": [
                                AIMessage(content=_text(reply.content) or "(redundant investigation skipped)"),
                                HumanMessage(note),
                            ],
                            "task": task,
                            "step_count": steps + 1,
                            "current_objective": None,
                        }
                    continue

                valid_action_calls.append(call)

            if not valid_action_calls and not revise_calls:
                return first_guard_return or {
                    "messages": [AIMessage(content=_text(reply.content) or "(action skipped)")],
                    "task": task,
                    "step_count": steps + 1,
                    "current_objective": None,
                }

            final_calls = [*revise_calls, *valid_action_calls]
            if len(final_calls) != len(calls):
                reply = AIMessage(content=_text(reply.content), tool_calls=final_calls)
                update["messages"] = [reply]

            if valid_action_calls:
                current_plan = plan or planning.new_plan(task)
                execution = state.get("execution")
                active_step_ids: dict[str, str] = {}
                objectives: list[str] = []

                for idx, call in enumerate(valid_action_calls):
                    t_name = str(call.get("name") or "")
                    c_args = call.get("args") if isinstance(call.get("args"), dict) else {}
                    c_id = str(call.get("id") or f"call_{steps + 1}_{idx + 1}")
                    obj_text = _action_arg_for_call(t_name, c_args)
                    objectives.append(obj_text)
                    pending_before = {s["step_id"] for s in planning.pending_steps(current_plan)}
                    current_plan, step_id = _start_planned_step(current_plan, t_name, obj_text)
                    active_step_ids[c_id] = step_id
                    execution = execution_mod.start_attempt(
                        execution,
                        step_id=step_id,
                        tool_call_id=c_id,
                        tool=t_name,
                        action_key=ev._action_key(t_name, obj_text),
                        binding="planned" if step_id in pending_before else "ad_hoc",
                        generation=int(state.get("progress_generation", 0)),
                    )
                    cur_step_obj = next(
                        (s for s in current_plan.get("steps", []) if s.get("step_id") == step_id),
                        None,
                    )
                    log_plan_step_transition(
                        step_id,
                        "in_progress",
                        cur_step_obj.get("description", "") if cur_step_obj else f"{t_name}: {obj_text}",
                        stage="decision",
                    )

                update["current_objective"] = "; ".join(objectives)
                update["plan"] = current_plan
                update["active_step_ids"] = active_step_ids
                update["execution"] = execution
                update["blocked_read_actions"] = blocked_list
                update["control_metrics"] = metrics
                ctx = get_run_context()
                if ctx is not None:
                    first_cid = str(valid_action_calls[0].get("id") or "")
                    ctx.current_step_id = active_step_ids.get(first_cid) or current_plan.get("current_step_id")
                    ctx.current_tool_call_id = first_cid or None
                register_requested_tool_calls("Main", final_calls, turn=steps + 1)
            else:
                update["current_objective"] = None
                update["active_step_ids"] = {}
                register_requested_tool_calls("Main", revise_calls, turn=steps + 1)
        elif calls:
            update["current_objective"] = None
            update["active_step_ids"] = {}
            register_requested_tool_calls("Main", calls, turn=steps + 1)
        else:
            final_text = _reply_text(reply)
            if not final_text.strip():
                # Direct synthesis: model stopped calling tools but emitted no text.
                usable_steps = ev.usable_evidence_steps(plan)
                evidence_digest = ""
                if usable_steps:
                    parts = []
                    for s in usable_steps[-4:]:
                        act = s.get("action", "tool")
                        ev_str = str(s.get("evidence", "")).strip()
                        if len(ev_str) > 900:
                            ev_str = ev_str[:850] + " ...[truncated]"
                        parts.append(f"### {act}:\n{ev_str}")
                    evidence_digest = "\n\n".join(parts)
                elif state.get("observations"):
                    parts = []
                    for o in state["observations"][-4:]:
                        o_str = str(o).strip()
                        if len(o_str) > 900:
                            o_str = o_str[:850] + " ...[truncated]"
                        parts.append(o_str)
                    evidence_digest = "\n\n".join(parts)
                else:
                    evidence_digest = tool_evidence(state.get("messages", []))[:2500]

                synth_prompt = SYNTHESIS_USER_PROMPT.format(
                    question=task,
                    answer_style=intent.get("answer_style", "normal"),
                    evidence=evidence_digest,
                )
                try:
                    synth_reply = invoke_model_logged(
                        llm,
                        [
                            SystemMessage(SYNTHESIS_SYSTEM_PROMPT),
                            HumanMessage(synth_prompt),
                        ],
                        agent="Main",
                        stage="wrap_up",
                        invoke_config={"metadata": {"ari_stage": "wrap_up"}},
                        api_key=config.api_key,
                        turn=steps + 1,
                    )
                    final_text = _reply_text(synth_reply)
                except Exception as exc:
                    log_guard_event("answer_synthesis_failed", "final_answer", str(exc))

            if not final_text.strip():
                final_text = ev.synthesize_deterministic_answer(task, plan, state.get("observations", []))
            criteria, graded_plan, support, g_metrics, a_cache = grade(state, task, plan)
            c_cache = _conclusion_cache_from_state(state)
            retries = state.get("completion_retries", 0)
            retry_generation = int(state.get("progress_generation", 0))
            answer_check = None
            if support == ev.SUPPORTED:
                answer = conclusion_candidate(final_text, criteria, task=task, plan=graded_plan or plan, observations=state.get("observations", []))
                final_text = answer
                corpus = "\n".join(
                    list(state.get("observations", []))
                    + [step.get("evidence", "") for step in ev.usable_evidence_steps(graded_plan, criteria)]
                )
                answer_check = validate_conclusion(
                    task, answer, criteria, graded_plan, corpus, g_metrics, c_cache
                )
                if not answer_check.get("error") and not answer_check.get("supported"):
                    g_metrics["criterion_answer_rejections"] += 1
            if (support not in (ev.SUPPORTED, ev.GRADING_FAILED)
                    and not (answer_check and answer_check.get("error"))
                    and ev.completion_retry_allowed(
                        steps, step_budget, retries,
                        progress_generation=retry_generation,
                        last_retry_generation=state.get("completion_retry_progress_generation"),
                    )):
                feedback = ev.completion_feedback(support, criteria, graded_plan)
                hint = execution_mod.strategy_hint(
                    state.get("execution"),
                    sources_restricted=bool(intent.get("sources")),
                    unmet_criteria=sum(1 for c in criteria if not c.get("satisfied")),
                    steps_left=step_budget - steps,
                )
                if hint:
                    feedback = f"{feedback}\n{hint}"
                update["messages"] = [reply, HumanMessage(feedback)]
                update.update({
                    "plan": graded_plan,
                    "criteria": criteria,
                    "completion_retries": retries + 1,
                    "completion_retry_progress_generation": retry_generation,
                    "current_objective": None,
                    "grader_metrics": g_metrics,
                    "assessment_cache": a_cache,
                    "conclusion_cache": c_cache,
                })
                return update
            fields = conclude(
                state, graded_plan, criteria, support, final_text,
                step_limit_reached=False, answer_check=answer_check,
                termination_reason="agent concluded before repairs succeeded",
                grader_metrics=g_metrics,
                assessment_cache=a_cache,
                conclusion_cache=c_cache,
            )
            update["messages"] = [AIMessage(content=fields["final_result"])]
            update.update(fields)
        return update

    def prepare_followup(state: DeepAgentState) -> dict:
        followup = state.get("pending_followup") or {}
        if followup.get("tool") != _GET_FILE:
            return {"pending_followup": None}
        call_id = "auto_" + uuid.uuid4().hex
        tool_call = {
            "name": _GET_FILE,
            "args": dict(followup.get("arguments") or {}),
            "id": call_id,
            "type": "tool_call",
        }
        register_requested_tool_calls("Main", [tool_call])
        control_metrics = dict(state.get("control_metrics") or {})
        control_metrics["automatic_followup_pending"] = True
        active_step_ids = dict(state.get("active_step_ids") or {})
        cur_sid = (state.get("plan") or {}).get("current_step_id")
        if cur_sid:
            active_step_ids[call_id] = cur_sid
        return {
            "messages": [AIMessage(content="", tool_calls=[tool_call])],
            "pending_followup": None,
            "active_step_ids": active_step_ids,
            "control_metrics": control_metrics,
        }

    tool_runner = ToolNode(all_tools, handle_tool_errors=False)

    def execute_tools(state: DeepAgentState) -> dict:
        """Execute the current turn's tool call(s) with this invocation's
        isolated `MutationTransaction` and capability context bound via ContextVar."""
        task = state.get("task") or _task_from_messages(state.get("messages") or [])
        tx = _tx_from_state(state, task)
        capabilities = state.get("capabilities") or intent_mod.default_capabilities()
        write_allowed = bool(capabilities.get("write", False))
        execute_allowed = bool(capabilities.get("execute", False))
        intent = intent_mod.get_intent(state)
        token = activate_transaction_scope(
            tx,
            write_allowed=write_allowed,
            execute_allowed=execute_allowed,
            intent=intent,
        )
        try:
            tool_out = tool_runner.invoke(state)
        finally:
            reset_transaction_scope(token)
            _record_live_tx_state(tx.to_state_dict(), task=task)
        out = dict(tool_out) if isinstance(tool_out, dict) else {"messages": tool_out}
        out["tx_state"] = tx.to_state_dict()
        out["transaction"] = tx.snapshot()
        return out

    def update_state(state: DeepAgentState) -> dict:
        results = _pending_tool_results(state["messages"])
        if not results:
            return {}

        task = state.get("task") or _task_from_messages(state["messages"])
        tx = _tx_from_state(state, task)
        evidence = list(state.get("evidence", []))
        observations = list(state.get("observations", []))
        verification_info = state.get("verification_info")
        failure_info = state.get("failure_info")
        delegated_result = state.get("delegated_result")
        plan = state.get("plan")
        active_step_ids = dict(state.get("active_step_ids") or {})
        criteria = list(state.get("criteria") or [])
        fingerprints = list(state.get("result_fingerprints") or [])
        fingerprint_set = set(fingerprints)
        generation = int(state.get("progress_generation", 0))
        no_progress_count = int(state.get("no_progress_count", 0))
        control_metrics = dict(state.get("control_metrics") or {})
        was_automatic_followup = bool(control_metrics.pop("automatic_followup_pending", False))
        execution = state.get("execution")
        completed_read_actions: list[tuple[str, str, str]] = []

        for call_id, name, report, call_args in results:
            if name == _REVISE_PLAN:
                new_steps = _parsed_revision(report)
                if new_steps is not None:
                    before_pending = {s["step_id"]: s for s in planning.pending_steps(plan)} if plan else {}
                    before_ids = {s["step_id"] for s in (plan or {}).get("steps", [])}
                    plan = planning.revise_pending_steps(
                        plan or planning.new_plan(task), new_steps,
                        reason="Superseded by the revised plan",
                    )
                    for s in plan.get("steps", []):
                        sid = s.get("step_id")
                        if sid in before_pending and s.get("status") == "skipped":
                            log_plan_step_transition(
                                sid, "skipped", s.get("description", ""),
                                stage="update_state", reason="Superseded by the revised plan",
                            )
                        elif sid not in before_ids:
                            log_plan_step_transition(
                                sid, s.get("status", "pending"), s.get("description", ""),
                                stage="update_state", reason="Added by revise_plan",
                            )
                continue

            payload = _delegation_payload(report)
            has_verification_report = (
                name in _CODE_VERIFICATION_TOOLS
                or (name == _RUN_COMMAND and isinstance(payload.get("report"), str))
            )
            readable_report = (
                str(payload.get("report", report))
                if has_verification_report
                else report
            )
            observations.append(readable_report)
            delegated_result = readable_report
            action_arg = _action_arg_for_call(name, call_args)

            result = ev.classify_result(name, report, payload, _CODE_VERIFICATION_TOOLS)
            step_evidence_excerpt = ev.summarize_evidence(name, report, payload, _CODE_VERIFICATION_TOOLS)

            generation_before_result = generation
            if name in _PROGRESS_TOOLS:
                fingerprint = result_fingerprint(name, report)
                if result in (ev.RESULT_OK, ev.RESULT_INCONCLUSIVE) and fingerprint not in fingerprint_set:
                    fingerprint_set.add(fingerprint)
                    fingerprints.append(fingerprint)
                    generation += 1
                    no_progress_count = 0
                else:
                    if fingerprint not in fingerprint_set:
                        fingerprint_set.add(fingerprint)
                        fingerprints.append(fingerprint)
                    no_progress_count += 1

            provenance = ev.evidence_provenance(name, payload, _CODE_VERIFICATION_TOOLS, report=report)
            provenance.update({
                "repository_epoch": id(service.index),
                "progress_generation": generation,
            })

            target_step_id = active_step_ids.get(call_id) or (plan.get("current_step_id") if plan else None)
            if plan is not None:
                plan = ev.record_step_evidence(
                    plan, name, action_arg, result, step_evidence_excerpt,
                    provenance=provenance, step_id=target_step_id,
                )
                recorded_for_call = [s for s in plan.get("steps", []) if s.get("result")]
                if recorded_for_call:
                    target_step_id = max(recorded_for_call, key=lambda s: s["order"])["step_id"]

            step_succeeded = True
            step_failure_reason: str | None = None
            if name in _READ_ONLY_PLAN_TOOLS:
                step_succeeded = (
                    result == ev.RESULT_OK
                    or (name == _GET_FILE and result == ev.RESULT_INCONCLUSIVE)
                    or (name in (_FIND_SYMBOL, _FIND_FILES) and result == ev.RESULT_EMPTY)
                )
                if step_succeeded and generation <= generation_before_result:
                    step_succeeded = False
                    step_failure_reason = "the call succeeded but added no new information"
                evidence.append(report)
                if result in (ev.RESULT_OK, ev.RESULT_INCONCLUSIVE):
                    completed_read_actions.append((name, report, action_arg))
            elif name in _CODE_VERIFICATION_TOOLS:
                state_val = payload.get("state")
                ver_dict = payload.get("verification") if isinstance(payload.get("verification"), dict) else {}
                verification_info = {
                    "tool": name,
                    "state": state_val,
                    "status": ver_dict.get("status"),
                    "report": readable_report,
                    "rollback": payload.get("rollback"),
                }
                if "error" in payload or state_val not in (SUCCESS, NO_TESTS, UNAVAILABLE):
                    failure_info = {"tool": name, "report": readable_report}
                    step_succeeded = False
                else:
                    failure_info = None
                    step_succeeded = True
            elif name == _RUN_COMMAND:
                state_val = payload.get("state")
                ver_dict = payload.get("verification") if isinstance(payload.get("verification"), dict) else None
                if state_val is not None or ver_dict is not None:
                    verification_info = {
                        "tool": name,
                        "state": state_val,
                        "status": (ver_dict or {}).get("status"),
                        "report": readable_report,
                        "rollback": payload.get("rollback"),
                    }
                    if "error" in payload or (state_val is not None and state_val not in (SUCCESS, NO_TESTS, UNAVAILABLE)):
                        failure_info = {"tool": name, "report": readable_report}
                    elif state_val in (SUCCESS, NO_TESTS):
                        failure_info = None
                elif "error" in payload and isinstance(payload.get("error"), dict):
                    err_code = payload["error"].get("code")
                    if err_code in ("unauthorized_command_mutation", "command_blocked_by_policy", "checkpoint_creation_failed"):
                        failure_info = {"tool": name, "report": readable_report}
                step_succeeded = result == ev.RESULT_OK
                if not step_succeeded:
                    step_failure_reason = "command exited with a non-zero status, timed out, or failed safety/verification"

            execution = execution_mod.finish_attempt(
                execution,
                step_id=target_step_id,
                tool_call_id=call_id,
                tool=name,
                result=result,
                generation=generation,
                succeeded=step_succeeded,
                interrupted=False,
                baseline_generation=generation_before_result,
            )

            if plan is not None and target_step_id is not None:
                closing_obj = next(
                    (s for s in plan.get("steps", []) if s.get("step_id") == target_step_id),
                    None,
                )
                plan = planning.finish_step(plan, target_step_id, step_succeeded, step_failure_reason)
                log_plan_step_transition(
                    target_step_id,
                    "completed" if step_succeeded else "failed",
                    closing_obj.get("description", "") if closing_obj else "",
                    stage="update_state",
                    **({"reason": step_failure_reason} if step_failure_reason and not step_succeeded else {}),
                )
                usable_set = {s["step_id"] for s in ev.usable_evidence_steps(plan, criteria)}
                log_main_received_tool_result(
                    name, result, step_succeeded,
                    step_id=target_step_id, payload=payload,
                )
                if closing_obj is not None:
                    log_evidence_recorded(
                        closing_obj,
                        target_step_id in usable_set,
                        "Main",
                    )

        ctx = get_run_context()
        if ctx is not None:
            ctx.current_step_id = None
            ctx.current_tool_call_id = None

        tx_dict = tx.to_state_dict()
        _record_live_tx_state(tx_dict, task=task)
        update: dict = {
            "evidence": evidence,
            "execution": execution,
            "observations": observations,
            "delegated_result": delegated_result,
            "tx_state": tx_dict,
            "transaction": tx.snapshot(),
            "verification_info": verification_info,
            "failure_info": failure_info,
            "active_step_ids": {},
            "plan": plan,
            "result_fingerprints": fingerprints,
            "progress_generation": generation,
            "no_progress_count": no_progress_count,
            "control_metrics": control_metrics,
        }

        # Deterministic exact-source follow-up when a single lookup identified one exact target
        # that has not already been read in the current repository state.
        if len(completed_read_actions) == 1 and plan is not None:
            name, report, action_arg = completed_read_actions[-1]
            candidate = deterministic_followup(name, report, task)
            if candidate is not None:
                f_args = candidate.get("arguments") or {}
                path = str(f_args.get("path") or "")
                request_intent = intent_mod.get_intent(state)
                if not intent_mod.source_allows_path(request_intent, path, service=service):
                    candidate = None
                elif _repeated_read_action(
                    plan, _GET_FILE, _action_arg_for_call(_GET_FILE, f_args),
                    generation, id(service.index), args=f_args,
                ) is not None:
                    candidate = None
            if candidate is not None and candidate.get("tool") == _GET_FILE:
                f_args = candidate.get("arguments") or {}
                read_objective = f"Read {f_args.get('path')} source identified by {name}"
                plan, followup_step_id = planning.start_step_for_call(
                    plan, f"{_GET_FILE}: {read_objective}", tool_name=_GET_FILE
                )
                if ctx is not None:
                    ctx.current_step_id = followup_step_id
                log_plan_step_transition(
                    followup_step_id,
                    "in_progress",
                    f"{_GET_FILE}: {read_objective}",
                    stage="followup",
                    reason=str(candidate.get("reason") or ""),
                )
                update["plan"] = plan
                update["current_objective"] = _action_arg_for_call(_GET_FILE, f_args)
                update["pending_followup"] = candidate
                control_metrics["deterministic_followups"] = list(
                    control_metrics.get("deterministic_followups") or []
                )
                control_metrics["deterministic_followups"].append(f"{name}->get_file ({candidate['reason']})")
                control_metrics["deterministic_followup_count"] = len(control_metrics["deterministic_followups"])
                update["control_metrics"] = control_metrics
                update["runtime_metrics"] = invocation_metrics(dict(state, control_metrics=control_metrics))
                return update

        if (plan is not None and was_automatic_followup and completed_read_actions
                and _likely_completing_followup(plan)):
            grading_state = dict(
                state, plan=plan, criteria=criteria, observations=observations,
                verification_info=verification_info, failure_info=failure_info,
            )
            criteria, plan, _, g_metrics, a_cache = grade(grading_state, task, plan)
            update.update({
                "criteria": criteria,
                "plan": plan,
                "grader_metrics": g_metrics,
                "assessment_cache": a_cache,
            })
            update["runtime_metrics"] = invocation_metrics(
                dict(state, control_metrics=control_metrics), g_metrics
            )
            return update

        update["runtime_metrics"] = invocation_metrics(dict(state, control_metrics=control_metrics))
        return update

    builder = StateGraph(DeepAgentState)
    builder.add_node("plan_task", plan_task)
    builder.add_node("decide", decide)
    builder.add_node("delegate", execute_tools)
    builder.add_node("update_state", update_state)
    builder.add_node("prepare_followup", prepare_followup)
    builder.add_edge(START, "plan_task")
    builder.add_edge("plan_task", "decide")
    builder.add_conditional_edges(
        "decide", _route_after_decide,
        {"tools": "delegate", "end": END, "retry": "decide"},
    )
    builder.add_edge("delegate", "update_state")
    builder.add_conditional_edges(
        "update_state", _route_after_update,
        {"followup": "prepare_followup", "decide": "decide", "end": END},
    )
    builder.add_edge("prepare_followup", "delegate")

    compiled = builder.compile().with_config(recursion_limit=6 * config.max_agent_steps + 8)
    return _SafeGraphWrapper(compiled, service, config)


def _normalize_turn_tool_calls(calls: list[dict]) -> tuple[list[dict], str]:
    """Validate and bound one turn's requested tool calls:
      - Allow up to MAX_READ_CALLS_PER_TURN (3) independent read-only calls per turn.
      - Never mix mutation/execution calls (`write_file`, `verify_repository`,
        `run_command`) with other tool calls in the same turn.
      - Deduplicate identical read-only calls within the same turn.
    """
    if len(calls) <= 1:
        return calls, ""

    revise_calls = [c for c in calls if c.get("name") == _REVISE_PLAN]
    action_calls = [c for c in calls if c.get("name") != _REVISE_PLAN]
    if not action_calls:
        return [revise_calls[-1]], ""

    first_name = str(action_calls[0].get("name") or "")
    if first_name in _MUTATION_OR_EXECUTION_TOOLS:
        if len(calls) > 1:
            note = (
                "Mutation and execution tools run one at a time and cannot be mixed with other "
                "tool calls; running only the first requested action now."
            )
            return [action_calls[0]], note
        return [action_calls[0]], ""

    # Leading calls are read-only: stop before any mutation/execution call so
    # dependent read -> write/execute operations are never parallelized.
    read_batch: list[dict] = []
    seen_keys: set[str] = set()
    dropped_mutation_or_exec = False
    dropped_overflow = False

    for c in action_calls:
        name = str(c.get("name") or "")
        if name in _MUTATION_OR_EXECUTION_TOOLS:
            dropped_mutation_or_exec = True
            break
        args = c.get("args") if isinstance(c.get("args"), dict) else {}
        key = ev._action_key(name, _action_arg_for_call(name, args))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        if len(read_batch) < MAX_READ_CALLS_PER_TURN:
            read_batch.append(c)
        else:
            dropped_overflow = True

    kept = [*(revise_calls[-1:] if revise_calls else []), *read_batch]
    notes: list[str] = []
    if dropped_mutation_or_exec:
        notes.append(
            "Mutation/execution calls cannot be mixed with read-only calls in the same turn; "
            "running the read-only call(s) first."
        )
    if dropped_overflow:
        notes.append(
            f"At most {MAX_READ_CALLS_PER_TURN} independent read-only tool calls are run per turn; "
            f"running the first {MAX_READ_CALLS_PER_TURN} now."
        )
    return kept, " ".join(notes)


def _build_repeat_guard_return(
    state: DeepAgentState,
    task: str,
    plan: dict | None,
    reply: AIMessage,
    tool_name: str,
    objective_text: str,
    repeat_key: str,
    repeat_hint: str,
    blocked: list[str],
    metrics: dict,
    steps: int,
    step_budget: int,
    intent: dict,
    grade_fn,
    conclude_fn,
    invocation_metrics_fn,
) -> dict:
    """Handle a blocked repeated read-only action with grading, bounded strategy
    reassessment, or no-progress termination."""
    criteria, plan, support, g_metrics, a_cache = grade_fn(state, task, plan)
    graded_state = dict(
        state, criteria=criteria, plan=plan, grader_metrics=g_metrics, assessment_cache=a_cache
    )
    if support == ev.SUPPORTED:
        fields = conclude_fn(
            graded_state, plan, criteria, support, "", step_limit_reached=False,
            termination_reason="all criteria supported after repeated read check",
            grader_metrics=g_metrics,
            assessment_cache=a_cache,
        )
        return {
            **fields,
            "messages": [AIMessage(content=fields["final_result"])],
            "task": task,
            "step_count": steps + 1,
        }

    already_blocked = repeat_key in (state.get("blocked_read_actions") or [])
    reassess = (
        execution_mod.reassessment_strategy(
            state.get("execution"),
            metrics,
            sources_restricted=bool(intent.get("sources")),
            repeated_tool=tool_name,
            steps_left=step_budget - (steps + 1),
        )
        if already_blocked
        else None
    )
    if reassess is not None:
        metrics["strategy_reassessments"] = metrics.get("strategy_reassessments", 0) + 1
        if reassess == "structural":
            note = (
                "Strategy reassessment (system data): the same read-only action was proposed again "
                "without new evidence. It was not repeated. Switch to structural repository tools "
                "(find_symbol, get_symbol, find_callers, find_callees, find_importers, "
                "get_dependencies, get_ast) to trace definitions and call/data flow directly, or "
                "finish with the unresolved facts stated."
            )
        else:
            note = (
                "Strategy reassessment (system data): the same read-only action was proposed again "
                f"without new evidence ({repeat_hint}). It was not repeated. Fill the specific "
                "remaining gap by reading an unread file/line range or symbol, or finish with the "
                "unresolved facts stated."
            )
        log_guard_event("strategy_reassessment", tool_name, note, target=objective_text)
        return {
            "messages": [
                AIMessage(content=_text(reply.content) or "(repeated action blocked)"),
                HumanMessage(note),
            ],
            "task": task,
            "step_count": steps + 1,
            "current_objective": None,
            "criteria": criteria,
            "plan": plan,
            "grader_metrics": g_metrics,
            "assessment_cache": a_cache,
            "control_metrics": metrics,
            "runtime_metrics": invocation_metrics_fn(dict(graded_state, control_metrics=metrics), g_metrics),
        }

    if already_blocked:
        metrics["no_progress_stops"] = metrics.get("no_progress_stops", 0) + 1
        graded_state["control_metrics"] = metrics
        log_guard_event("no_progress_stop", tool_name, f"Repeated blocked action: {repeat_key}", target=objective_text)
        fields = conclude_fn(
            graded_state, plan, criteria, support, "", step_limit_reached=False,
            extra_unresolved=(
                "Stopped after the same read-only action was proposed again without "
                "new evidence or a repository change.",
            ),
            termination_reason="stopped after repeated action without progress",
            grader_metrics=g_metrics,
            assessment_cache=a_cache,
        )
        return {
            **fields,
            "messages": [AIMessage(content=fields["final_result"])],
            "task": task,
            "step_count": steps + 1,
            "control_metrics": metrics,
        }

    if repeat_key not in blocked:
        blocked.append(repeat_key)
    metrics["repeated_action_blocks"] = metrics.get("repeated_action_blocks", 0) + 1
    if tool_name == _FIND_FILES:
        note = (
            "Progress guard (system data): filename search (find_files) already returned no matching "
            "paths for this pattern family against the unchanged repository state, so shortening or "
            "varying the filename pattern was not run. find_files searches file paths only, not file "
            "contents, metric values, or symbol names. Switch strategy: use search_repository to search "
            "inside code and documentation, find_symbol/get_symbol to locate a function/class definition, "
            "or get_file to read documentation such as README.md."
        )
    else:
        note = (
            "Progress guard (system data): a same or already-covered read-only action already ran "
            f"against this unchanged repository state ({repeat_hint}). It was not repeated. "
            "Choose a genuinely different unread source, line range, or structural tool "
            "(get_symbol, find_callers, find_callees, get_dependencies, get_ast), or finish with "
            "the unresolved facts stated."
        )
    log_guard_event("repeated_read_action", tool_name, note, target=objective_text)
    return {
        "messages": [
            AIMessage(content=_text(reply.content) or "(repeated action blocked)"),
            HumanMessage(note),
        ],
        "task": task,
        "step_count": steps + 1,
        "current_objective": None,
        "criteria": criteria,
        "plan": plan,
        "blocked_read_actions": blocked,
        "grader_metrics": g_metrics,
        "assessment_cache": a_cache,
        "control_metrics": metrics,
        "runtime_metrics": invocation_metrics_fn(dict(graded_state, control_metrics=metrics), g_metrics),
    }


def _likely_completing_followup(plan: dict) -> bool:
    """Whether the latest automatic exact-source read is complete and usable."""
    steps = ev.evidence_steps(plan)
    if not steps:
        return False
    latest = steps[-1]
    if latest.get("tool") != _GET_FILE:
        return False
    return latest["step_id"] in {step["step_id"] for step in ev.usable_evidence_steps(plan)}


def _route_after_decide(state: dict) -> str:
    if route_after_agent(state) == "tools":
        return "tools"
    return "end" if state.get("final_result") is not None else "retry"


def _route_after_update(state: dict) -> str:
    if state.get("final_result") is not None:
        return "end"
    return "followup" if state.get("pending_followup") else "decide"


def _start_planned_step(plan: dict, tool_name: str, objective_text: str) -> tuple[dict, str]:
    """Start the pending plan step that `planning.start_step_for_call` matches
    (decoupled from tool names), or record an ad-hoc step."""
    return planning.start_step_for_call(plan, f"{tool_name}: {objective_text}", tool_name=tool_name)


def _describe_investigation_guidance(
    task: str,
    intent: dict,
    plan: dict | None,
    criteria: list[dict],
    state: DeepAgentState,
    steps_left: int | None = None,
    service: RepoService | None = None,
) -> str:
    """Concise, gap-focused single-agent investigation and repair guidance."""
    tx_state = state.get("transaction") or {}
    if tx_state.get("in_repair_loop"):
        used = tx_state.get("repair_attempts", 0)
        v_report = (state.get("verification_info") or {}).get("report") or ""
        return (
            f"Repair status (active repair loop, attempt {used} used): the latest code modification "
            "failed verification. Inspect the verification error details, read any needed context, and "
            "use write_file to fix the issue.\n"
            + (f"Latest verification summary:\n{_clip(v_report, 600)}\n" if v_report else "")
            + execution_mod.digest(state.get("execution"), plan, criteria)
        ).strip()

    if intent.get("sources"):
        text = (
            f"Scope restriction: this task is explicitly limited to {intent['sources']}. "
            "Use get_file (or write_file if a modification was requested) only on the permitted path(s)."
        )
    elif criteria and all(c.get("satisfied") for c in criteria):
        if planning.pending_steps(plan):
            text = (
                "Investigation status: all acceptance criteria are supported by recorded evidence. "
                "Do not repeat read-only lookups; drop superseded pending steps with revise_plan if "
                "needed and give your final answer."
            )
        else:
            text = (
                "Investigation status: all acceptance criteria are supported by recorded evidence. "
                "Integrate the findings and give your final answer."
            )
    else:
        unmet = [c for c in (criteria or []) if not c.get("satisfied")]
        unmet_ids = ", ".join(c["id"] for c in unmet)
        readme_cand = _find_readme_path(service) if (service is not None and _is_general_purpose_query(task, intent)) else None
        covered_files = {
            str((s.get("provenance") or {}).get("file") or "").replace("\\", "/").casefold()
            for s in ev.evidence_steps(plan)
            if s.get("tool") == _GET_FILE and s.get("result") in (ev.RESULT_OK, ev.RESULT_INCONCLUSIVE)
        }
        if steps_left is not None and steps_left < execution_mod.MIN_TURNS_FOR_REASSESSMENT:
            text = (
                f"Investigation status ({len(unmet)} unmet criteria: {unmet_ids}): step budget is nearly "
                "spent; issue up to 3 targeted read-only calls for the remaining gaps or give your final "
                "answer stating what could not be established."
            )
        elif readme_cand and readme_cand.casefold() not in covered_files:
            text = (
                f"Investigation status ({len(unmet)} unmet criteria: {unmet_ids}): start by reading "
                f"{readme_cand} with get_file (start_line=1, end_line=80) before running broad "
                f"search_repository queries; inspect code only if {readme_cand} does not answer the question."
            )
        elif intent_mod.scope_of(intent).get("cross_file") or intent_mod._requires_investigation(task):
            text = (
                f"Investigation status ({len(unmet)} unmet criteria: {unmet_ids}): use find_symbol, "
                "get_symbol, find_callers, find_callees, find_importers, get_dependencies, get_ast, "
                "and get_file (up to 3 independent read-only calls per turn) to trace definitions and "
                "data flow across modules without repeating earlier reads or searches."
            )
        else:
            text = (
                f"Investigation status ({len(unmet)} unmet criteria: {unmet_ids}): use targeted direct "
                "tools (get_file, get_symbol, search_repository, find_symbol, find_files) to establish "
                "the remaining criteria."
            )

    summary = execution_mod.digest(state.get("execution"), plan, criteria)
    return f"{text}\n{summary}" if summary else text


_PATH_EXTENSION_TOKENS = {
    "py", "md", "rst", "txt", "json", "yaml", "yml", "toml", "ini", "cfg", "sh", "csv", "html", "js", "ts"
}


def _find_files_keyword_tokens(pattern: str) -> set[str]:
    """Extract alphanumeric sub-tokens (splitting on underscores/spaces) from a non-glob find_files pattern."""
    cleaned = (pattern or "").strip().replace("\\", "/").casefold()
    if not cleaned or any(ch in cleaned for ch in "*?[") or "/" in cleaned:
        return set()
    return {tok for tok in re.findall(r"[a-z0-9]+", cleaned) if tok and tok not in _PATH_EXTENSION_TOKENS}


def _is_redundant_empty_find_files(current_arg: str, previous_arg: str) -> bool:
    """True when a follow-up find_files call merely shortens or re-separates an already-empty keyword pattern."""
    cur_norm = (current_arg or "").strip().replace("\\", "/").casefold()
    prev_norm = (previous_arg or "").strip().replace("\\", "/").casefold()
    if not cur_norm or not prev_norm:
        return False
    if any(ch in cur_norm for ch in "*?[") or any(ch in prev_norm for ch in "*?["):
        return False
    if "." in cur_norm or "/" in cur_norm:
        return False
    cur_tokens = _find_files_keyword_tokens(cur_norm)
    prev_tokens = _find_files_keyword_tokens(prev_norm)
    if cur_tokens and prev_tokens and (cur_tokens & prev_tokens):
        return True
    cur_compact = re.sub(r"[^a-z0-9]+", "", cur_norm)
    prev_compact = re.sub(r"[^a-z0-9]+", "", prev_norm)
    return bool(cur_compact and prev_compact and (cur_compact in prev_compact or prev_compact in cur_compact))



def _repeated_read_action(
    plan,
    tool: str,
    argument: str,
    generation: int,
    repository_epoch: int,
    args: dict | None = None,
) -> tuple[str, str] | None:
    """Detect repeated or already-covered read-only actions against the same
    `repository_epoch`.

    Returns `(repeat_key, human_explanation)` when the proposed read-only call:
      1. Repeats the exact same `(tool, normalized_arg)` already executed at
         this `repository_epoch` (deterministic repository tools return the
         exact same result unless the repository index changed).
      2. Requests a `get_file` path/line-range that is already completely
         covered by an earlier unclipped `get_file` read at this
         `repository_epoch`, or re-requests `start_line=None` on a partially
         read file instead of advancing to `next_start_line`.
      3. Requests `get_symbol` for a symbol whose definition was already read
         via `get_symbol` or `get_file` at this `repository_epoch`.
      4. Repeats a `find_files` keyword family that already returned 0 matches.
      5. Repeats a `search_repository` query with the exact same normalized
         token set at this `repository_epoch`.
    """
    if not plan or tool not in _READ_ONLY_PLAN_TOOLS:
        return None
    safe_args = args or {}
    normalized_arg = planning._normalize(argument.replace("\\", "/"))

    # Check structured file-range coverage for get_file.
    if tool == _GET_FILE:
        req_path = intent_mod.normalize_source_path(str(safe_args.get("path") or argument.split(":", 1)[0]))
        req_start = safe_args.get("start_line")
        req_end = safe_args.get("end_line")
        covered, totals, epochs = ev._file_coverage(plan)
        for cov_path, intervals in covered.items():
            if intent_mod.normalize_source_path(cov_path) != req_path:
                continue
            if epochs.get(cov_path) != repository_epoch:
                continue
            merged = ev._merge_line_ranges(intervals)
            total = totals.get(cov_path)
            if total is not None and ev._reach(merged) >= total and cov_path not in set(ev._clipped_full_files(plan)):
                return (
                    f"get_file|full_covered:{req_path}|{repository_epoch}",
                    f"{cov_path} (lines 1-{total} of {total}) has already been read in full",
                )
            if isinstance(req_start, int) and isinstance(req_end, int):
                for m_start, m_end in merged:
                    if m_start <= req_start and req_end <= m_end:
                        return (
                            f"get_file|range_covered:{req_path}:{req_start}-{req_end}|{repository_epoch}",
                            f"{cov_path} lines {req_start}-{req_end} are already covered by recorded read {m_start}-{m_end}",
                        )
            elif req_start is None and req_end is None and merged and merged[0][0] == 1:
                reach = merged[0][1]
                if total is not None and reach < total:
                    return (
                        f"get_file|head_already_read:{req_path}:1-{reach}|{repository_epoch}",
                        f"{cov_path} lines 1-{reach} of {total} were already read; pass start_line={reach + 1} to read the remaining lines",
                    )

    for step in reversed(ev.evidence_steps(plan)):
        provenance = step.get("provenance") or {}
        if provenance.get("repository_epoch") != repository_epoch:
            continue
        prev_tool = step.get("tool")
        action_key = str(step.get("action_key") or "")
        previous_argument = action_key.partition("|")[2]

        if prev_tool == tool:
            if normalized_arg == previous_argument:
                return (
                    f"{tool}|{previous_argument}|{repository_epoch}",
                    f"{tool}({argument!r}) already ran in {step.get('step_id')} -> {step.get('result')}",
                )
            if tool == _FIND_FILES and step.get("result") == ev.RESULT_EMPTY:
                raw_prev = str(provenance.get("pattern") or previous_argument)
                if _is_redundant_empty_find_files(argument, raw_prev):
                    return (
                        f"{tool}|empty_keyword_family:{previous_argument}|{repository_epoch}",
                        f"find_files({raw_prev!r}) already returned 0 matches in {step.get('step_id')}",
                    )
            if tool == _SEARCH_REPOSITORY:
                cur_tokens = set(normalized_arg.split())
                prev_tokens = set(previous_argument.split())
                if cur_tokens and cur_tokens == prev_tokens:
                    return (
                        f"{tool}|same_query_tokens:{' '.join(sorted(cur_tokens))}|{repository_epoch}",
                        f"search_repository with the same query terms already ran in {step.get('step_id')}",
                    )
            if tool == _GET_SYMBOL and step.get("result") == ev.RESULT_OK:
                resolved_sym = str(provenance.get("symbol") or "")
                if resolved_sym and planning._normalize(resolved_sym) == normalized_arg:
                    return (
                        f"get_symbol|{planning._normalize(resolved_sym)}|{repository_epoch}",
                        f"symbol {resolved_sym!r} was already read in {step.get('step_id')}",
                    )
    return None


def _action_arg_for_call(name: str, args: dict) -> str:
    """Build the canonical action argument string for a tool call."""
    if name == _FIND_FILES:
        return str(args.get("pattern") or name)
    if name == _GET_FILE:
        path = str(args.get("path") or "")
        start, end = args.get("start_line"), args.get("end_line")
        if start is not None or end is not None:
            return f"{path}:{start if start is not None else ''}-{end if end is not None else ''}"
        return path
    if name == _WRITE_FILE:
        return str(args.get("path") or name)
    if name == _RUN_COMMAND:
        return str(args.get("command") or name)
    if name in (_GET_SYMBOL, _FIND_CALLERS, _FIND_CALLEES):
        return str(args.get("symbol") or name)
    if name in (_FIND_IMPORTERS, _GET_DEPENDENCIES, _GET_AST):
        return str(args.get("target") or name)
    if name == _FIND_SYMBOL:
        q = str(args.get("query") or name)
        kind = args.get("kind")
        return f"{q} ({kind})" if kind else q
    return str(args.get("query") or args.get("task") or args.get("target") or args.get("symbol") or name)


def _is_light_readonly_task(intent: dict, plan) -> bool:
    """Whether this invocation is a light/standard read-only task."""
    if intent.get("kind") == intent_mod.KIND_CHANGE:
        return False
    if intent.get("rigor") == intent_mod.RIGOR_STRICT:
        return False
    if intent.get("depth") == intent_mod.DEPTH_DEEP:
        return False
    if plan and any(s.get("tool") == _WRITE_FILE for s in ev.evidence_steps(plan)):
        return False
    return True


def _redundant_investigation_guard(state: DeepAgentState, plan) -> bool:
    """True when another read-only call would only repeat evidence-gathering
    after a prior grading pass already marked every criterion satisfied."""
    criteria = state.get("criteria") or []
    if not criteria or not all(c["satisfied"] for c in criteria):
        return False
    if _has_failed_evidence(state, plan):
        return False
    return _is_light_readonly_task(intent_mod.get_intent(state), plan)


def _has_failed_evidence(state: DeepAgentState, plan) -> bool:
    """Conservative check for any failed tool step in the current plan."""
    if state.get("failure_info") is not None:
        return True
    return any(
        s.get("result") == ev.RESULT_FAILED
        for s in ev.evidence_steps(plan)
    )


def _assess(
    llm,
    task: str,
    criteria: list[dict],
    plan: dict,
    verification_info,
    cache: dict[str, dict] | None = None,
    metrics: dict | None = None,
    repository_identity: str = "",
    api_key: str | None = None,
) -> tuple[list[dict], dict]:
    """Ask the LLM which acceptance criteria the recorded evidence supports,
    citing evidence item ids, then validate citations deterministically."""
    items = ev.evidence_for_grader(plan, criteria)
    verdicts: dict = {}
    grader_error = ""
    fresh_assessment = False
    cached_failure = False
    grader_ran = False
    cache_hit = False
    format_retried = False
    size_retried = False
    fit_report: dict = {}
    if ev.usable_evidence_steps(plan):
        verification = _clip(verification_info["report"]) if verification_info else "(no verification was run)"
        payload = {
            "objective": task,
            "criteria": [{k: c[k] for k in ("id", "description", "evidence_needed")} for c in criteria],
            "evidence": items,
            "verification": verification,
        }
        fixed_chars = len(CRITERIA_CHECK_PROMPT) + len(json.dumps(dict(payload, evidence=[]), ensure_ascii=True))
        full_items = items
        payload["evidence"] = _fit_evidence_items(
            full_items, fixed_chars, report_out=fit_report, stage="evidence_grading"
        )
        cache_payload = dict(payload, evidence=ev.evidence_for_grader(plan))
        fingerprint = hashlib.sha256(json.dumps(
            {"repository": repository_identity, "assessment": cache_payload},
            ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")).hexdigest()
        if not fit_report.get("fits", True) or fit_report.get("essential_unfittable"):
            grader_error = (
                "grading payload exceeds context budget: "
                + str(fit_report.get("reason") or "essential metadata and fixed instructions cannot fit")
            )
        else:
            cached = cache.get(fingerprint) if cache is not None else None
            if cached is not None:
                cache_hit = True
                if isinstance(cached.get("_grader_error"), str):
                    grader_error = cached["_grader_error"]
                    cached_failure = True
                else:
                    verdicts = cached
                if metrics is not None:
                    metrics["grader_cache_hits"] += 1
            else:
                grader_ran = True
                if metrics is not None:
                    metrics["grader_cache_misses"] += 1
                started = time.perf_counter()
                expected_ids = {criterion["id"] for criterion in criteria}
                active_retry_kind: str | None = None
                size_retry_error = ""
                while True:
                    if not fit_report.get("fits", True) or fit_report.get("essential_unfittable"):
                        verdicts = {}
                        grader_error = (
                            "grading payload exceeds context budget: "
                            + str(fit_report.get("reason") or "essential metadata and fixed instructions cannot fit")
                        )
                        break
                    try:
                        system_prompt = CRITERIA_CHECK_PROMPT
                        if active_retry_kind == "format":
                            system_prompt += (
                                "\nYour previous reply was malformed or omitted criterion ids. "
                                "Return the required JSON object with every criterion id exactly once."
                            )
                            stage = "evidence_grading_retry"
                            fmt_attempt = 2
                        elif active_retry_kind == "size":
                            stage = "evidence_grading_size_retry"
                            fmt_attempt = 1
                        else:
                            stage = "evidence_grading"
                            fmt_attempt = 1
                        if metrics is not None:
                            metrics["grader_calls"] += 1
                        reply = invoke_model_logged(
                            llm,
                            [SystemMessage(system_prompt),
                             HumanMessage(json.dumps(payload, ensure_ascii=True))],
                            agent="Main",
                            stage=stage,
                            invoke_config={"metadata": {"ari_stage": stage}},
                            api_key=api_key,
                            format_attempt=fmt_attempt,
                            max_format_attempts=2,
                        )
                        if size_retried and size_retry_error:
                            record_stage_size_recovery(
                                agent="Main",
                                stage="evidence_grading",
                                recovered=True,
                                original_error=size_retry_error,
                                fit_report=fit_report,
                            )
                            size_retry_error = ""
                        parsed = ev.parse_assessment(_text(reply.content))
                        if parsed and set(parsed) == expected_ids:
                            verdicts = parsed
                            grader_error = ""
                            fresh_assessment = True
                            if cache is not None:
                                cache[fingerprint] = verdicts
                                if len(cache) > 128:
                                    cache.pop(next(iter(cache)))
                            break
                        verdicts = {}
                        if not format_retried:
                            retry_prompt = (
                                CRITERIA_CHECK_PROMPT
                                + "\nYour previous reply was malformed or omitted criterion ids. "
                                "Return the required JSON object with every criterion id exactly once."
                            )
                            retry_fixed_chars = len(retry_prompt) + len(
                                json.dumps(dict(payload, evidence=[]), ensure_ascii=True)
                            )
                            retry_scale = 0.5 if size_retried else 1.0
                            payload["evidence"] = _fit_evidence_items(
                                full_items, retry_fixed_chars, scale=retry_scale,
                                report_out=fit_report, stage="evidence_grading_retry",
                            )
                            format_retried = True
                            active_retry_kind = "format"
                            if metrics is not None:
                                metrics["grader_format_retries"] += 1
                            continue
                        grader_error = "unparseable or incomplete reply after one format retry"
                        break
                    except Exception as exc:
                        verdicts = {}
                        grader_error = _model_error_reason(exc, api_key)
                        if not size_retried and _is_oversized_request_error(exc, api_key, grader_error):
                            size_retried = True
                            active_retry_kind = "size"
                            size_retry_error = grader_error
                            payload["evidence"] = _fit_evidence_items(
                                full_items, fixed_chars, scale=0.5, report_out=fit_report,
                                stage="evidence_grading_size_retry",
                            )
                            if not fit_report.get("fits", True) or fit_report.get("essential_unfittable"):
                                grader_error = (
                                    f"{grader_error}; reduced grading payload still exceeds context budget "
                                    f"({fit_report.get('reason') or 'essential metadata and fixed instructions cannot fit'})"
                                )
                                break
                            if metrics is not None:
                                metrics["grader_size_retries"] = metrics.get("grader_size_retries", 0) + 1
                            record_stage_size_recovery(
                                agent="Main",
                                stage="evidence_grading",
                                recovered=False,
                                original_error=grader_error,
                                fit_report=fit_report,
                            )
                            continue
                        break
                if metrics is not None:
                    metrics["grader_latency_ms"] += int((time.perf_counter() - started) * 1000)
                if grader_error and ("429" in grader_error or "rate limit" in grader_error.lower()):
                    usable_for_fallback = ev.usable_evidence_steps(plan, criteria)
                    direct_usable = [
                        s for s in usable_for_fallback
                        if s.get("tool") in ("get_file", "get_symbol", "write_file", "verify_repository", "run_command")
                    ] or usable_for_fallback
                    if direct_usable and not ev.incomplete_reasons(criteria, plan):
                        fb_ids = [s["step_id"] for s in direct_usable[:4]]
                        verdicts = {
                            c["id"]: {
                                "id": c["id"],
                                "satisfied": True,
                                "evidence_ids": fb_ids,
                                "reason": "supported by recorded repository evidence (deterministic fallback on rate limit)",
                            }
                            for c in criteria
                        }
                        grader_error = ""
                        fresh_assessment = True
                if grader_error and cache is not None:
                    cache[fingerprint] = {"_grader_error": grader_error}
                    if len(cache) > 128:
                        cache.pop(next(iter(cache)))
    new_criteria, new_plan = ev.apply_assessment(criteria, plan, verdicts)
    if metrics is not None:
        if grader_error and not cached_failure:
            metrics["grader_failures"] += 1
        elif fresh_assessment:
            metrics["criterion_acceptances"] += sum(c.get("validation") == "accepted" for c in new_criteria)
            metrics["criterion_model_rejections"] += sum(c.get("validation") == "grader_rejected" for c in new_criteria)
            metrics["criterion_citation_rejections"] += sum(c.get("validation") == "citation_rejected" for c in new_criteria)
    if grader_error:
        new_criteria = [
            c if c["satisfied"] else dict(c, note=f"{ev.GRADER_FAILED_NOTE} ({grader_error})")
            for c in new_criteria
        ]
    log_grading_diagnostics(
        plan=new_plan,
        criteria=new_criteria,
        usable_steps=ev.usable_evidence_steps(new_plan, new_criteria),
        verdicts=verdicts,
        grader_ran=grader_ran,
        cache_hit=cache_hit,
        grader_error=grader_error,
        format_retried=format_retried,
        size_retried=size_retried,
        payload_fit=fit_report,
    )
    return new_criteria, new_plan


def _fit_evidence_items(
    items: list[dict],
    fixed_chars: int,
    scale: float = 1.0,
    *,
    budget_tokens: int | None = None,
    report_out: dict | None = None,
    stage: str = "evidence_payload",
) -> list[dict]:
    """Return `items` (never mutated) with long `evidence` excerpts clipped so
    the serialized JSON payload plus fixed instructions fit the token budget."""
    from .context_budget import fit_evidence_items_to_budget

    fitted, report = fit_evidence_items_to_budget(
        items, fixed_chars, scale=scale, budget_tokens=budget_tokens
    )
    if isinstance(report_out, dict):
        report_out.clear()
        report_out.update(report)
    record_payload_fit_event(agent="Main", stage=stage, fit_report=report)
    return fitted


def _clip(text: str, limit: int = _CHECK_MAX_CHARS) -> str:
    text = str(text)
    return text if len(text) <= limit else text[:limit] + " ...[truncated]"


def _is_oversized_request_error(error: Exception, api_key: str | None = None, reason: str = "") -> bool:
    """Recognize HTTP 413 / request-too-large / context-window overflow errors."""
    if getattr(error, "error_category", None) == "context_budget_exceeded":
        return True
    if reason.startswith("provider request/context limit"):
        return True
    try:
        from .context_budget import is_context_size_error
        err_info = analyze_provider_error(error, api_key)
        return is_context_size_error(err_info)
    except Exception:
        return False


def _model_error_reason(error: Exception, api_key: str | None = None) -> str:
    """Keep provider failure details visible without exposing configured secrets."""
    detail = " ".join(str(error).split())
    if api_key:
        detail = detail.replace(api_key, "[redacted]")
    detail = detail[:320]
    if getattr(error, "error_category", None) == "context_budget_exceeded":
        return f"context budget limit: {detail}" if detail else "context budget limit"
    status = getattr(error, "status_code", None)
    response = getattr(error, "response", None)
    if status is None and response is not None:
        status = getattr(response, "status_code", None)
    err_info = analyze_provider_error(error, api_key)
    from .context_budget import is_context_size_error
    request_limited = status == 413 or "413" in detail or is_context_size_error(err_info)
    rate_limited = (
        not request_limited
        and (status == 429 or "429" in detail or "ratelimit" in type(error).__name__.casefold())
    )
    if request_limited:
        prefix = "provider request/context limit (HTTP 413)" if (status == 413 or "413" in detail) else "provider request/context limit"
    elif rate_limited:
        prefix = "provider rate limit (HTTP 429)"
    else:
        prefix = type(error).__name__
    return f"{prefix}: {detail}" if detail else prefix


def _task_from_messages(messages: list) -> str:
    return next((str(m.content) for m in reversed(messages) if getattr(m, "type", "") == "human"), "")


def _text(content) -> str:
    if isinstance(content, list):
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return str(content)


def _reply_text(reply) -> str:
    txt = _text(getattr(reply, "content", "")).strip()
    if txt:
        return txt
    if not getattr(reply, "tool_calls", None):
        ak = getattr(reply, "additional_kwargs", None)
        if isinstance(ak, dict):
            for key in ("text", "answer", "reasoning_content"):
                val = ak.get(key)
                if isinstance(val, str) and val.strip():
                    return val.strip()
    return "" 


def _pending_tool_results(messages: list) -> list[tuple[str, str, str, dict]]:
    """Return `(tool_call_id, tool_name, report_text, call_args)` for every tool
    result produced by the most recent `decide -> delegate` step, in message order."""
    last_calls: dict[str, tuple[str, dict]] = {}
    for message in reversed(messages):
        if getattr(message, "type", "") == "ai":
            for call in getattr(message, "tool_calls", None) or []:
                args = call.get("args")
                last_calls[call["id"]] = (call["name"], args if isinstance(args, dict) else {})
            if last_calls:
                break
    if not last_calls:
        return []

    results: list[tuple[str, str, str, dict]] = []
    for message in messages:
        if getattr(message, "type", "") != "tool":
            continue
        call_id = str(getattr(message, "tool_call_id", None) or "")
        call = last_calls.get(call_id)
        if call is not None and call[0] in _PLAN_AWARE_TOOL_NAMES:
            results.append((call_id, call[0], str(message.content), call[1]))
    return results


def _parsed_revision(report: str) -> list[str] | None:
    """Extract the proposed replacement list from a `revise_plan` tool result."""
    try:
        data = json.loads(report)
    except (TypeError, ValueError):
        return None
    if isinstance(data, dict):
        steps = data.get("requested_steps", data.get("remaining_steps"))
        if isinstance(steps, list) and all(isinstance(s, str) for s in steps):
            return steps
    return None


def _succeeded(report: str) -> bool:
    """Read the structured terminal state emitted by modification/verification tools."""
    try:
        payload = json.loads(report)
    except (TypeError, ValueError):
        return False
    return isinstance(payload, dict) and payload.get("state") in (SUCCESS, NO_TESTS)


def _delegation_payload(report: str) -> dict:
    try:
        payload = json.loads(report)
    except (TypeError, ValueError):
        return {"state": None, "report": report}
    return payload if isinstance(payload, dict) else {"state": None, "report": report}


class _SafeGraphWrapper:
    """Ensure per-invocation transaction checkpoints and per-repository locks
    are always finalized and released even if `.stream()` or `.invoke()` exits
    early due to an unhandled exception, generator close, or cancellation."""

    def __init__(self, compiled_graph, service: RepoService, config: AgentConfig) -> None:
        self._graph = compiled_graph
        self.service = service
        self.config = config

    def __getattr__(self, name: str):
        return getattr(self._graph, name)

    def with_config(self, *args, **kwargs) -> "_SafeGraphWrapper":
        return _SafeGraphWrapper(
            self._graph.with_config(*args, **kwargs), self.service, self.config
        )

    def stream(self, input_state, stream_mode="updates", **kwargs):
        tracker = {
            "tx_state": (input_state or {}).get("tx_state") if isinstance(input_state, dict) else None,
            "task": (input_state or {}).get("task", "") if isinstance(input_state, dict) else "",
            "concluded": False,
        }
        token = _INVOCATION_TRACKER.set(tracker)
        try:
            for chunk in self._graph.stream(input_state, stream_mode=stream_mode, **kwargs):
                if isinstance(chunk, dict):
                    for _node, upd in chunk.items():
                        if isinstance(upd, dict):
                            if "tx_state" in upd:
                                tracker["tx_state"] = upd["tx_state"]
                            if "task" in upd and upd["task"]:
                                tracker["task"] = upd["task"]
                            if upd.get("final_result") is not None:
                                tracker["concluded"] = True
                yield chunk
        finally:
            try:
                _INVOCATION_TRACKER.reset(token)
            except Exception:
                pass
            latest_tx_state = tracker.get("tx_state")
            latest_task = tracker.get("task")
            concluded = bool(tracker.get("concluded"))
            if isinstance(latest_tx_state, dict):
                tx = MutationTransaction.from_state_dict(
                    self.service, self.config, latest_tx_state, label=str(latest_task or "")
                )
                if not concluded and tx.in_repair_loop:
                    tx.finalize_if_unrepaired(
                        termination_reason="invocation stream terminated before completion"
                    )
                else:
                    tx.cleanup()

    def invoke(self, input_state, **kwargs):
        tracker = {
            "tx_state": (input_state or {}).get("tx_state") if isinstance(input_state, dict) else None,
            "task": (input_state or {}).get("task", "") if isinstance(input_state, dict) else "",
            "concluded": False,
        }
        token = _INVOCATION_TRACKER.set(tracker)
        try:
            result = self._graph.invoke(input_state, **kwargs)
            if isinstance(result, dict):
                if "tx_state" in result:
                    tracker["tx_state"] = result["tx_state"]
                if result.get("final_result") is not None:
                    tracker["concluded"] = True
            return result
        finally:
            try:
                _INVOCATION_TRACKER.reset(token)
            except Exception:
                pass
            latest_tx_state = tracker.get("tx_state")
            latest_task = tracker.get("task")
            concluded = bool(tracker.get("concluded"))
            if isinstance(latest_tx_state, dict):
                tx = MutationTransaction.from_state_dict(
                    self.service, self.config, latest_tx_state, label=str(latest_task or "")
                )
                if not concluded and tx.in_repair_loop:
                    tx.finalize_if_unrepaired(
                        termination_reason="invocation terminated before completion"
                    )
                else:
                    tx.cleanup()
