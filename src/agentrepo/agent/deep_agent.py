"""Plan-driven orchestration for repository investigation and changes.

Each invocation creates an initial plan and acceptance criteria, then loops
through tool selection, delegation, and state updates within configured
bounds. Recorded tool evidence gates completion; unsupported answers are
held for further work when budget remains. Intent and capability limits are
computed from the user's request, and repository-changing work retains the
delegation layer's checkpoint, verification, and rollback safeguards.
"""

from __future__ import annotations

import json
import hashlib
import time
import uuid

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

from ..service import RepoService
from .config import AgentConfig, load_config
from .deep_state import DeepAgentState
from .delegation import build_delegation_tools
from .model import build_llm
from . import evidence as ev
from . import intent as intent_mod
from . import plan as planning
from .prompts import CRITERIA_CHECK_PROMPT, DEEP_AGENT_SYSTEM_PROMPT, PLANNING_SYSTEM_PROMPT
from .safety import NO_TESTS, SUCCESS, UNAVAILABLE
from .tools import build_find_files_tool, build_find_symbol_tool, build_get_file_tool
from .followups import deterministic_followup, result_fingerprint, similar_action
from .utils import bounded_model_history, route_after_agent, wrap_up_prompt

# Delegation tool names, from delegation.py's build_delegation_tools().
_INVESTIGATE = "investigate_repository"
_CODE = "code_repository"
_VERIFY = "verify_repository"
# The Deep Agent's own plan-revision tool (see build_deep_agent
# below) - deliberately NOT one of delegation.py's tools: it never touches
# the repository, so it is not part of the safety-wrapped delegation set the
# Main Agent's graph also uses.
_REVISE_PLAN = "revise_plan"
# The single read-only repository tool exposed DIRECTLY to the
# Deep Agent (see build_deep_agent below) for a simple, known-path read -
# unlike investigate_repository/code_repository/verify_repository, this is
# not a sub-agent delegation, it is the same `get_file` tool
# build_repo_tools already gives the Explorer, called straight from this
# graph. Still plan-aware (its result must flow through update_state's
# evidence/grading machinery exactly like a delegation report), so it is
# part of _PLAN_AWARE_TOOL_NAMES alongside the three delegation tools.
_GET_FILE = "get_file"
_FIND_FILES = "find_files"
_FIND_SYMBOL = "find_symbol"
_READ_ONLY_PLAN_TOOLS = frozenset({_INVESTIGATE, _GET_FILE, _FIND_FILES, _FIND_SYMBOL})
_PROGRESS_TOOLS = _READ_ONLY_PLAN_TOOLS | frozenset({_CODE, _VERIFY})
_PLAN_AWARE_TOOL_NAMES = (_INVESTIGATE, _CODE, _VERIFY, _GET_FILE, _FIND_FILES, _FIND_SYMBOL, _REVISE_PLAN)
# Runtime safety boundary (see intent.capabilities_for_floor and `decide`
# below): delegation tools that can mutate the repository. Currently only
# code_repository - it is the sole gateway into the Coder/Debugger
# sub-agents (write_file, mutating run_command), so gating it here gates
# everything reachable through it. investigate_repository/get_file are
# read-only. verify_repository is gated separately below (EXECUTE
# capability) because its configured command can mutate the checkout.
_WRITE_DELEGATION_TOOLS = frozenset({_CODE})
# verify_repository takes no LLM-supplied arguments, but the command it runs
# (config.verification_command, normally the test suite) can still create or
# modify files in the checkout. It needs the EXECUTE capability - granted by
# a change request or an explicit "run the tests" request - so a read-only
# question can never reach it just because the model chose the tool.
_EXECUTE_DELEGATION_TOOLS = frozenset({_VERIFY})
# Bound on the verification report shown to the criteria check.
_CHECK_MAX_CHARS = 2000


class RevisePlanInput(BaseModel):
    steps: list[str] = Field(
        description=(
            "The FULL, updated list of remaining plan step descriptions - replaces whatever remaining "
            "(not-yet-attempted) steps were tracked before, not just the change. An EMPTY list is valid "
            "and means no planned work remains."
        ),
    )


def build_deep_agent(service: RepoService, config: AgentConfig | None = None, llm=None,
                      explorer_llm=None, coder_llm=None, debugger_llm=None):
    """
    Return a compiled Deep Agent graph. Use it like any LangGraph graph:
        deep_agent.invoke({"messages": [{"role": "user", "content": "..."}]})

    llm, explorer_llm, coder_llm and debugger_llm are optional (mainly for
    tests): without them, the Groq model from the configuration is used for
    all of them.
    """
    if config is None:
        config = load_config(require_api_key=llm is None)
    if llm is None:
        llm = build_llm(config)

    # Single-slot reference to the current invocation's plan, refreshed at
    # the top of every `decide()` call below (see there) and read by
    # `_investigated_hint`. Not a new state subsystem - it only lets the
    # delegation tools (built once, below) see the SAME plan/evidence
    # `decide` already has for this run; plan_task always runs before the
    # first `decide` of a new invocation, so a later run can never see a
    # stale plan left over from an earlier one.
    _latest_plan: list = [None]

    def _investigated_hint() -> str:
        """Compact, informational-only digest of this run's already-recorded
        read-only evidence, given to a freshly-started Explorer so it has
        some notion of what the Deep Agent already looked at without
        re-deriving it - purely a summary of existing plan/evidence state,
        not a restriction. The Explorer remains free to re-read or re-check
        anything listed here if its own task genuinely calls for it."""
        plan = _latest_plan[0]
        if not plan:
            return ""
        seen: list[str] = []
        for step in ev.evidence_steps(plan):
            if step.get("tool") in _READ_ONLY_PLAN_TOOLS and step.get("result") == ev.RESULT_OK:
                action = step.get("action")
                if action and action not in seen:
                    seen.append(action)
        if not seen:
            return ""
        return (
            "Already investigated earlier this run (system data, for context only - not a restriction; "
            "re-read or re-check any of these if verifying something specific genuinely requires it):\n"
            + "\n".join(f"- {item}" for item in seen[-10:])
        )

    # Reuse the existing Explorer/Coder/Debugger + verification/repair-loop
    # + checkpoint/rollback machinery exactly as the Main Agent does - the
    # Deep Agent is a new orchestration layer above it, not a reimplementation.
    delegation_tools = build_delegation_tools(
        service, config,
        explorer_llm=explorer_llm or llm,
        coder_llm=coder_llm or llm,
        debugger_llm=debugger_llm or coder_llm or llm,
        investigated_hint=_investigated_hint,
    )

    def revise_plan(steps: list[str]) -> str:
        # Pure Deep-Agent-state bookkeeping - no repository access, no LLM
        # call. The actual PENDING steps are updated in `update_state` from
        # THIS function's return value (the executed tool result), not from
        # the LLM's raw call arguments - the same "actual result is
        # authoritative" pattern the rest of this package uses for
        # write_file/run_command (see utils.written_files).
        cleaned = [s.strip() for s in steps if isinstance(s, str) and s.strip()]
        cleaned = cleaned[: config.max_plan_steps]
        # This is a proposal, not confirmation that state validation accepted
        # it. update_state applies it only after checking the current criteria.
        return json.dumps({"requested_steps": cleaned})

    revise_plan_tool = StructuredTool.from_function(
        func=revise_plan,
        name=_REVISE_PLAN,
        description=(
            "Replace the Deep Agent's list of remaining planned steps with a FULL, updated list (not just the "
            "change) - use when investigation shows the original plan needs to change (a step turned out "
            "unnecessary, or the repository needs different work than expected). Updates only the Deep Agent's "
            "own planning state; never touches the repository. An empty list requests dropping all pending "
            "steps, but is rejected while any acceptance criterion is still unsupported."
        ),
        args_schema=RevisePlanInput,
    )

    # Expose a small direct read-only set for targeted file/path/symbol
    # lookups. Broad repository tools stay behind investigate_repository/the
    # Explorer. All share the same service and observation-size bound.
    get_file_tool = build_get_file_tool(service, config.max_observation_chars)

    find_files_tool = build_find_files_tool(service, config.max_observation_chars)
    find_symbol_tool = build_find_symbol_tool(service, config.max_observation_chars)
    all_tools = delegation_tools + [revise_plan_tool, get_file_tool, find_files_tool, find_symbol_tool]
    llm_with_tools = llm.bind_tools(all_tools)
    assessment_cache: dict[str, dict] = {}
    assessment_metrics = {
        "grader_calls": 0,
        "grader_cache_hits": 0,
        "grader_cache_misses": 0,
        "grader_latency_ms": 0,
    }

    def invocation_metrics(state: DeepAgentState) -> dict:
        grader_delta = {
            key: value - (state.get("grader_metrics_start") or {}).get(key, 0)
            for key, value in assessment_metrics.items()
        }
        control = dict(state.get("control_metrics") or {})
        control["deterministic_followups"] = list(control.get("deterministic_followups") or [])
        return {**grader_delta, **control}

    def plan_task(state: DeepAgentState) -> dict:
        # Runs once per invocation, before the decide/delegate/update_state
        # loop starts. A plain LLM call (no tools bound) - it only needs to
        # produce text, not select a capability - so it does not consume
        # the delegation/revise_plan tool-calling budget the loop below
        # bounds via step_count/max_agent_steps.
        # The same call also drafts the acceptance criteria (no
        # extra LLM call); see evidence.parse_criteria.
        # The deterministic intent floor is a pure function of the
        # task text alone, so it is computed here, BEFORE the planning LLM
        # call - a planner failure below can therefore never lose it (see
        # intent.py's module docstring).
        task = state.get("task") or _task_from_messages(state["messages"])
        intent_floor = intent_mod.deterministic_floor(task)
        planning_error = None
        criteria: list[dict] = []
        model_intent_raw = None
        descriptions: list[str] = []
        if intent_mod.is_lookup_task(task, intent_floor):
            # Simple informational requests use the generic fallback plan
            # and criteria, with minimal depth and light rigor unless the
            # deterministic floor requires more.
            model_intent_raw = {"depth": intent_mod.DEPTH_MINIMAL, "rigor": intent_mod.RIGOR_LIGHT}
        else:
            try:
                reply = llm.invoke([SystemMessage(PLANNING_SYSTEM_PROMPT),
                                    HumanMessage(json.dumps({"objective": task,
                                                             "max_steps": config.max_plan_steps,
                                                             "max_criteria": ev.MAX_CRITERIA,
                                                             "intent_floor": intent_floor}, ensure_ascii=True))],
                                   config={"metadata": {"ari_stage": "planning"}})
                reply_text = _text(reply.content)
                descriptions = planning.parse_plan_steps(reply_text, config.max_plan_steps)
                criteria = ev.parse_criteria(reply_text)
                model_intent_raw = intent_mod.parse_intent_reply(reply_text)
            except Exception as exc:
                # A single generic step lets the bounded agent investigate the
                # objective safely; malformed output never becomes an action.
                descriptions = []
                planning_error = f"planning model call failed: {type(exc).__name__}: {exc}"
        if not descriptions:
            # Defensive fallback only (the planning call returned nothing
            # parseable) - a single generic step parameterised by the task
            # text itself, not a plan hard-coded for any particular kind of
            # task. decide/revise_plan can still change this once evidence
            # comes in.
            descriptions = [planning.fallback_step_description(task)]
        if not criteria:
            criteria = ev.default_criteria(task)
        # Resolve the final intent - the floor above always wins;
        # the planner may only raise depth/rigor, never lower a floor (see
        # intent.resolve_intent) - then use it to clamp the PLANNED step
        # count and the acceptance-criteria count. These are PLANNED-step/
        # criteria-count limits only; they say nothing about, and do not by
        # themselves bound, actual Explorer/Coder tool calls (out of scope
        # actual tool calls (see intent.py).
        intent = intent_mod.resolve_intent(task, model_intent_raw, floor=intent_floor)
        descriptions = descriptions[: intent_mod.plan_step_limit(intent, config.max_plan_steps)]
        criteria = criteria[: intent_mod.criteria_limit(intent, ev.MAX_CRITERIA)]
        # Runtime safety boundary (see intent.capabilities_for_floor):
        # computed ONLY from intent_floor - the deterministic keyword floor
        # over the raw task text - never from model_intent_raw/intent, so
        # nothing the planning LLM call (or, later, a delegation call) says
        # can widen it. Set once, here, before the decide/delegate loop
        # starts; `decide` below enforces it and never recomputes it.
        capabilities = intent_mod.capabilities_for_floor(intent_floor)
        return {
            "task": task,
            "intent": intent,
            "capabilities": capabilities,
            "plan": planning.seed_pending_steps(planning.new_plan(task), descriptions),
            "criteria": criteria,
            "completion": None,
            "completion_retries": 0,
            "model_error": planning_error,
            "evidence": [],
            "observations": [],
            "current_objective": None,
            "delegated_result": None,
            "verification_info": None,
            "failure_info": None,
            "final_result": None,
            "outcome": None,
            "step_count": 0,
            "grader_metrics_start": dict(assessment_metrics),
            "control_metrics": {"deterministic_followups": [], "deterministic_followup_count": 0,
                                "no_progress_stops": 0, "repeated_action_blocks": 0},
            "result_fingerprints": [],
            "progress_generation": 0,
            "no_progress_count": 0,
            "blocked_read_actions": [],
            "pending_followup": None,
        }

    def grade(state: DeepAgentState, task: str, plan) -> tuple[list[dict], dict, str]:
        """Grade the acceptance criteria against the evidence
        recorded on the plan's steps, then classify what that evidence
        supports. Returns (criteria, plan with criterion links, support).

        apply_assessment and classify_support remain the evidence gate. Once every criterion is
        satisfied by real, gate-accepted evidence, and the ONLY thing keeping
        classify_support's verdict at PARTIAL is a leftover PENDING
        read-only plan step, and the task is a light, read-only one (see
        _is_light_readonly_task), that leftover step cannot yield any
        further evidence a satisfied criterion still needs - so it is
        dropped (the exact effect revise_plan([]) has - see
        plan.revise_pending_steps) and classify_support is simply asked
        again against the now-empty pending list. No criterion's
        satisfied/evidence_ids are touched by this, and apply_assessment
        is not re-run or skipped."""
        plan = plan or planning.new_plan(task)
        criteria, plan = _assess(
            llm, task, state.get("criteria") or ev.default_criteria(task), plan,
            state.get("verification_info"), assessment_cache, assessment_metrics,
            repository_identity=f"{service.index.root}:{id(service.index)}",
            api_key=config.api_key,
        )
        support = ev.classify_support(criteria, plan, state.get("failure_info"), state.get("verification_info"))
        if ev.can_drop_pending_readonly_steps(
                support, criteria, plan,
                light_readonly=_is_light_readonly_task(intent_mod.get_intent(state), plan),
                failure_info=state.get("failure_info")):
            plan = planning.revise_pending_steps(plan, [])
            support = ev.classify_support(criteria, plan, state.get("failure_info"), state.get("verification_info"))
        return criteria, plan, support

    def conclude(state: DeepAgentState, plan, criteria, support: str, final_text: str,
                 step_limit_reached: bool, extra_unresolved: tuple = ()) -> dict:
        """The fields that end the invocation. `final_result` is
        built by evidence.build_final_response from the recorded state
        (Conclusion / Supporting evidence / Unresolved), so it cannot claim
        more than the completion state and only cites real recorded steps;
        the model's own text is used only as the conclusion of a SUPPORTED
        result. `outcome` follows the completion state (SUPPORTED <-> SUCCESS,
        BLOCKED -> FAILED); classify_outcome still applies its own gates, so
        the stricter of the two always wins."""
        # Everything the agent legitimately saw, to check file names against.
        corpus = "\n".join(list(state.get("observations", [])) + [s.get("evidence", "") for s in plan["steps"]])
        response, unresolved = ev.build_final_response(
            support, criteria, plan, final_text, state.get("failure_info"), state.get("verification_info"), corpus,
            extra_unresolved=extra_unresolved)
        outcome = planning.classify_outcome(
            plan, state.get("failure_info"), step_limit_reached=step_limit_reached,
            has_evidence=bool(state.get("observations")), objective_satisfied=(support == ev.SUPPORTED),
            verification_info=state.get("verification_info"))
        if support == ev.BLOCKED:
            outcome = planning.OUTCOME_FAILED
        return {"plan": plan, "criteria": criteria, "outcome": outcome, "current_objective": None,
                "final_result": response,
                "completion": ev.build_completion(support, criteria, unresolved),
                "runtime_metrics": invocation_metrics(state)}

    def decide(state: DeepAgentState) -> dict:
        task = state.get("task") or _task_from_messages(state["messages"])
        plan = state.get("plan")
        # Refresh before any tool call this turn can run, so
        # investigate_repository's hint (see _investigated_hint above)
        # reflects evidence recorded through the end of the previous turn.
        _latest_plan[0] = plan
        steps = state.get("step_count", 0)
        # Minimal-depth requests have a smaller tool-turn budget. Standard
        # and deep requests retain the configured maximum.
        step_budget = intent_mod.agent_step_budget(intent_mod.get_intent(state), config.max_agent_steps)
        if steps >= step_budget:
            # Out of budget: no more delegation, wrap up with whatever
            # evidence was gathered across the loop's previous iterations -
            # the forced-final-answer branch.
            wrap_up_error = None
            try:
                reply = llm.invoke([SystemMessage(DEEP_AGENT_SYSTEM_PROMPT),
                                    HumanMessage(wrap_up_prompt(state["messages"], step_budget,
                                                                task=task))],
                                   config={"metadata": {"ari_stage": "wrap_up"}})
                model_error = state.get("model_error")
            except Exception as exc:
                reply = AIMessage(content="The agent could not produce a wrap-up because the model call failed.")
                model_error = f"wrap-up model call failed: {_model_error_reason(exc, config.api_key)}"
                wrap_up_error = model_error
            final_text = _text(reply.content)
            criteria, plan, support = grade(state, task, plan)
            update = conclude(state, plan, criteria, support, final_text, step_limit_reached=True,
                              extra_unresolved=(wrap_up_error,) if wrap_up_error else ())
            # guarantee: no tool calls, so the loop ends
            update.update({"messages": [AIMessage(content=update["final_result"])], "task": task,
                           "step_count": steps + 1, "model_error": model_error})
            return update

        system_messages = [SystemMessage(DEEP_AGENT_SYSTEM_PROMPT)]
        plan_status = planning.describe_plan(plan)
        if plan_status:
            # Plan-aware step selection - shown as its own message
            # (not baked into DEEP_AGENT_SYSTEM_PROMPT, which is static)
            # so it always reflects the plan as it stands right now.
            system_messages.append(HumanMessage("Plan status data:\n" + plan_status))
        criteria_status = ev.describe_criteria(state.get("criteria", []), plan)
        if criteria_status:
            # What still has to be shown, and which attempts gave
            # no usable evidence (so they are not blindly repeated).
            system_messages.append(HumanMessage("Acceptance criteria and evidence data:\n" + criteria_status))
        # An unverified pointer from a prior, non-LLM retrieval
        # attempt (see DeepAgentState.retrieval_hint). Shown only on the
        # first turn (steps == 0) - once the agent has actually looked
        # anywhere, its own plan/criteria status is the relevant "what have
        # I found" summary, and re-showing a stale, un-acted-on hint every
        # turn would just invite it to be treated as more authoritative than
        # it is.
        if steps == 0 and state.get("retrieval_hint"):
            system_messages.append(HumanMessage(
                "Unverified retrieval hint (system data, not evidence): before you were invoked, a "
                "quick indexed search for this question found these possibly-relevant locations:\n"
                + state["retrieval_hint"]
                + "\nThey have not been read or confirmed. If one looks relevant, verify it with "
                "get_file or investigate_repository before relying on it - do not cite it directly. "
                "Disregard any that turn out unrelated or insufficient, and investigate normally instead."
            ))
        try:
            model_history = bounded_model_history(
                state["messages"], task=task,
                max_turns=min(config.max_agent_steps, 6),
            )
            reply = llm_with_tools.invoke(system_messages + model_history,
                                          config={"metadata": {"ari_stage": "decision"}})
        except Exception as exc:
            reason = _model_error_reason(exc, config.api_key)
            message = f"The agent stopped because the model request failed: {reason}."
            error = f"decision model call failed: {reason}"
            if any(s["result"] == ev.RESULT_OK for s in ev.evidence_steps(plan)):
                # Usable evidence exists: do not throw it away behind a bare
                # error string. Grade and build the normal grounded response,
                # but never claim SUPPORTED - the agent did not finish - so a
                # fully supported result is reported as PARTIAL.
                criteria, graded_plan, support = grade(state, task, plan)
                if support == ev.SUPPORTED:
                    support = ev.PARTIAL
                fields = conclude(state, graded_plan, criteria, support, "", step_limit_reached=False,
                                  extra_unresolved=(message,))
                return {**fields, "messages": [AIMessage(content=fields["final_result"])], "task": task,
                        "step_count": steps + 1, "model_error": error}
            criteria = state.get("criteria") or ev.default_criteria(task)
            plan = plan or planning.new_plan(task)
            fields = conclude(state, plan, criteria, ev.UNSUPPORTED, "", step_limit_reached=False,
                              extra_unresolved=(message,))
            return {**fields, "messages": [AIMessage(content=fields["final_result"])], "task": task,
                    "step_count": steps + 1, "current_objective": None, "model_error": error}

        update: dict = {"messages": [reply], "task": task, "step_count": steps + 1}
        calls = getattr(reply, "tool_calls", None) or []
        if len(calls) > 1:
            # ToolNode executes every call attached to one AIMessage. Keep
            # orchestration state deterministic by allowing exactly the first
            # requested action through; the model can choose another after it
            # sees that result on the next turn.
            note = "Only one orchestration action is run per turn; the first requested action is being run now."
            reply = AIMessage(content=f"{_text(reply.content)}\n\n{note}".strip(), tool_calls=[calls[0]])
            calls = [calls[0]]
            update["messages"] = [reply]
        delegate_calls = [c for c in calls if c.get("name") != _REVISE_PLAN]
        if delegate_calls:
            # Record what the Deep Agent is currently attempting, in plain
            # language, from the delegation call it just made - the "next
            # action" DeepAgentState.current_objective exists to track. With
            # more than one call in a turn (the system prompt asks for at
            # most one, but nothing enforces that on the model) this is the
            # first one; `update_state` below still folds every result in.
            args = delegate_calls[0].get("args") or {}
            objective_text = _action_arg_for_call(delegate_calls[0]["name"], args)
            intent = intent_mod.get_intent(state)
            if delegate_calls[0]["name"] == _FIND_SYMBOL and intent.get("sources"):
                note = ("Source restriction (system data): exact symbol lookup searches the full repository, "
                        "so it is unavailable for a task restricted to particular files. Use get_file only "
                        "for an explicitly allowed path.")
                return {"messages": [AIMessage(content=_text(reply.content) or "(restricted symbol lookup skipped)"),
                                     HumanMessage(note)],
                        "task": task, "step_count": steps + 1, "current_objective": None}
            if not intent_mod.delegation_allows(intent, delegate_calls[0]["name"]):
                note = (f"Source restriction (system data): this task is limited to "
                        f"{intent.get('sources')}. The requested capability "
                        f"{delegate_calls[0]['name']!r} is not permitted because it could access other "
                        "repository paths. Use direct get_file only for an explicitly allowed path.")
                return {"messages": [AIMessage(content=_text(reply.content) or "(restricted capability skipped)"),
                                     HumanMessage(note)],
                        "task": task, "step_count": steps + 1, "current_objective": None}

            if delegate_calls[0]["name"] in _WRITE_DELEGATION_TOOLS:
                # Runtime safety boundary (see intent.capabilities_for_floor
                # and DeepAgentState.capabilities): the LLM remains free to
                # choose code_repository for any task, including a
                # read-oriented one where it shouldn't - that is the
                # incident this guards against. The check here is on the
                # execution context's capability grant, computed once by
                # plan_task from the user's own original task text, NOT on
                # this call's `task` argument (which is LLM-generated,
                # untrusted content the LLM could phrase however it likes to
                # try to justify a mutation) and NOT on any second model
                # call - just a deterministic dict lookup. A state without a
                # capability grant at all (defensive default) is treated as
                # read-only, never as granting WRITE.
                capabilities = state.get("capabilities") or intent_mod.default_capabilities()
                if not capabilities.get("write"):
                    note = (
                        "MUTATION_BLOCKED (system data): this execution context does not have WRITE "
                        f"capability, so {delegate_calls[0]['name']!r} was not run and the repository was "
                        "NOT modified - no file was created, changed, deleted or renamed. The original "
                        "request was not recognized as a change/implementation task. If the user genuinely "
                        "wants a repository change, they need to ask for it explicitly; otherwise continue "
                        "with read-only tools (investigate_repository, get_file) or give a final answer."
                    )
                    return {"messages": [AIMessage(content=_text(reply.content) or "(mutation blocked - no WRITE capability)"),
                                         HumanMessage(note)],
                            "task": task, "step_count": steps + 1, "current_objective": None}

            if delegate_calls[0]["name"] in _EXECUTE_DELEGATION_TOOLS:
                # Same rule as the WRITE gate above: a deterministic lookup on
                # the capability grant computed from the user's own task text.
                # A grant without an "execute" key (older hand-built state)
                # falls back to its "write" value; no grant at all is
                # read-only.
                capabilities = state.get("capabilities") or intent_mod.default_capabilities()
                if not capabilities.get("execute", capabilities.get("write", False)):
                    note = (
                        "EXECUTION_BLOCKED (system data): this execution context does not have EXECUTE "
                        f"capability, so {delegate_calls[0]['name']!r} was not run and no repository command "
                        "was executed. The original request did not ask for tests or checks to be run. If "
                        "the user wants that, they need to ask explicitly (for example 'run the tests'); "
                        "otherwise continue with read-only tools (find_files, get_file, investigate_repository) "
                        "or give a final answer."
                    )
                    return {"messages": [AIMessage(content=_text(reply.content) or "(execution blocked - no EXECUTE capability)"),
                                         HumanMessage(note)],
                            "task": task, "step_count": steps + 1, "current_objective": None}

            if delegate_calls[0]["name"] == _GET_FILE:
                requested_path = args.get("path") or ""
                if not intent_mod.source_allows_path(intent, requested_path):
                    note = (f"Source restriction (system data): this task restricts direct file access to "
                            f"{intent.get('sources')}, so get_file was not run for {requested_path!r}. "
                            "Choose an explicitly allowed source or state that the restricted evidence could not "
                            "be established.")
                    return {"messages": [AIMessage(content=_text(reply.content) or "(restricted get_file skipped)"),
                                         HumanMessage(note)],
                            "task": task, "step_count": steps + 1, "current_objective": None}
            repeat_key = _repeated_read_action(
                plan, delegate_calls[0]["name"], objective_text,
                state.get("progress_generation", 0), id(service.index),
            ) if delegate_calls[0]["name"] in _READ_ONLY_PLAN_TOOLS else None
            if repeat_key is not None:
                criteria, plan, support = grade(state, task, plan)
                graded_state = dict(state, criteria=criteria, plan=plan)
                # A repeated action can be suppressed without ending the task
                # when the evidence already passes the ordinary evidence gate.
                if support == ev.SUPPORTED:
                    fields = conclude(graded_state, plan, criteria, support, "", step_limit_reached=False)
                    return {**fields, "messages": [AIMessage(content=fields["final_result"])],
                            "task": task, "step_count": steps + 1}
                blocked = list(state.get("blocked_read_actions") or [])
                metrics = dict(state.get("control_metrics") or {})
                if repeat_key in blocked:
                    metrics["no_progress_stops"] = metrics.get("no_progress_stops", 0) + 1
                    graded_state["control_metrics"] = metrics
                    fields = conclude(
                        graded_state, plan, criteria, support, "", step_limit_reached=False,
                        extra_unresolved=("Stopped after the same read-only action was proposed again without "
                                          "new evidence or a repository change.",),
                    )
                    return {**fields, "messages": [AIMessage(content=fields["final_result"])],
                            "task": task, "step_count": steps + 1, "control_metrics": metrics}
                blocked.append(repeat_key)
                metrics["repeated_action_blocks"] = metrics.get("repeated_action_blocks", 0) + 1
                note = ("Progress guard (system data): a same or near-identical read-only action already ran "
                        "against this unchanged repository state and added no new result. It was not repeated. "
                        "Choose a genuinely different source/action, or finish with the unresolved facts stated.")
                return {"messages": [AIMessage(content=_text(reply.content) or "(repeated action blocked)"),
                                     HumanMessage(note)],
                        "task": task, "step_count": steps + 1, "current_objective": None,
                        "criteria": criteria, "plan": plan,
                        "blocked_read_actions": blocked, "control_metrics": metrics,
                        "runtime_metrics": invocation_metrics(dict(graded_state, control_metrics=metrics))}
            if (delegate_calls[0]["name"] not in _READ_ONLY_PLAN_TOOLS and plan
                    and ev.is_blind_repeat(plan, delegate_calls[0]["name"], objective_text, _CODE)):
                # This exact action already
                # ran without usable evidence and nothing has changed since,
                # so running it again can only fail the same way. Skip it and
                # send the model back to `decide` (route "retry"); the turn
                # still counts against max_agent_steps, so this is bounded.
                note = ("Repeat check (system data): that exact delegation already ran without producing usable "
                        "evidence and the repository has not changed since, so it was not run again. Choose a "
                        "different, more targeted approach, or give a final answer stating what could not be "
                        "established.")
                return {"messages": [AIMessage(content=_text(reply.content) or "(repeated action skipped)"),
                                     HumanMessage(note)],
                        "task": task, "step_count": steps + 1, "current_objective": None}
            # _redundant_investigation_guard reads state["criteria"] as last
            # graded by whichever branch called grade() most recently (a
            # completion attempt, a forced wrap-up, etc. - see decide()'s
            # other grade() call sites). It never fires before a first
            # grading pass has actually happened this invocation (every
            # criterion starts satisfied=False - see its docstring), so no
            # extra grade() call is needed here just to evaluate it; that
            # would only re-ask the grader the same question an existing
            # grading pass already answered, for the single purpose of
            # deciding whether to skip a call that is itself already bounded
            # by _repeated_read_action/is_blind_repeat on the next attempt.
            if delegate_calls[0]["name"] == _INVESTIGATE and _redundant_investigation_guard(state, plan):
                # Every acceptance criterion is already supported
                # by recorded evidence for this light, read-only task - a
                # further investigate_repository call can only repeat
                # evidence apply_assessment already accepted. Skip it the
                # same way a blind repeat is skipped (route "retry", one
                # step_count) - never for a strict/deep/change task, and
                # never before a first grading pass has actually run (see
                # _redundant_investigation_guard).
                note = ("Sufficiency check (system data): every acceptance criterion is already supported by "
                        "recorded evidence, so another investigation was not run. Give your final answer, or use "
                        "code_repository/verify_repository if something other than more reading is still needed.")
                return {"messages": [AIMessage(content=_text(reply.content) or "(redundant investigation skipped)"),
                                     HumanMessage(note)],
                        "task": task, "step_count": steps + 1, "current_objective": None}
            update["current_objective"] = objective_text
            # Select against the plan (promote a matching PENDING
            # step to IN_PROGRESS) rather than always inventing a fresh one -
            # see plan.start_step. Falls back to appending
            # when the delegation does not match any planned step.
            current_plan = plan or planning.new_plan(task)
            update["plan"] = planning.start_step(current_plan, objective_text)
        elif calls:
            # Only revise_plan call(s) this turn, no delegation - nothing is
            # "currently being attempted". The plan mutation itself happens
            # in update_state, from revise_plan's own tool result.
            update["current_objective"] = None
        else:
            # The model gave a final answer. Grade the criteria
            # against recorded evidence first. If the evidence does not
            # support the answer and there is budget left, do not accept it:
            # send the model back with what is missing (route "retry").
            # Bounded by MAX_COMPLETION_RETRIES and the step budget.
            final_text = _text(reply.content)
            criteria, graded_plan, support = grade(state, task, plan)
            retries = state.get("completion_retries", 0)
            if (support != ev.SUPPORTED
                    and ev.completion_retry_allowed(steps, step_budget, retries)):
                update["messages"] = [reply, HumanMessage(ev.completion_feedback(support, criteria, graded_plan))]
                update.update({"plan": graded_plan, "criteria": criteria,
                               "completion_retries": retries + 1, "current_objective": None})
                return update
            fields = conclude(state, graded_plan, criteria, support, final_text, step_limit_reached=False)
            update["messages"] = [AIMessage(content=fields["final_result"])]  # the message shows the grounded response
            update.update(fields)
        return update

    def prepare_followup(state: DeepAgentState) -> dict:
        followup = state.get("pending_followup") or {}
        if followup.get("tool") != _GET_FILE:
            return {"pending_followup": None}
        tool_call = {
            "name": _GET_FILE,
            "args": dict(followup.get("arguments") or {}),
            "id": "auto_" + uuid.uuid4().hex,
            "type": "tool_call",
        }
        control_metrics = dict(state.get("control_metrics") or {})
        control_metrics["automatic_followup_pending"] = True
        return {"messages": [AIMessage(content="", tool_calls=[tool_call])],
                "pending_followup": None, "control_metrics": control_metrics}

    delegate = ToolNode(all_tools, handle_tool_errors=False)

    def update_state(state: DeepAgentState) -> dict:
        # Normally one delegation/revise_plan call per `decide` turn, but the
        # model can ask for more than one capability in a single turn (the
        # system prompt says "at most one", nothing enforces it) - fold
        # every result `delegate` just produced into state, not only the
        # last one, so no report is silently dropped from evidence/
        # observations.
        results = _pending_tool_results(state["messages"])
        if not results:
            return {}

        evidence = list(state.get("evidence", []))
        observations = list(state.get("observations", []))
        verification_info = state.get("verification_info")
        failure_info = state.get("failure_info")
        delegated_result = state.get("delegated_result")
        plan = state.get("plan")
        criteria = list(state.get("criteria") or [])
        fingerprints = list(state.get("result_fingerprints") or [])
        fingerprint_set = set(fingerprints)
        generation = int(state.get("progress_generation", 0))
        no_progress_count = int(state.get("no_progress_count", 0))
        control_metrics = dict(state.get("control_metrics") or {})
        was_automatic_followup = bool(control_metrics.pop("automatic_followup_pending", False))
        step_succeeded = True  # investigate_repository counts as successful unless it returned nothing usable (see below)
        delegated = False  # whether any *delegation* (not just revise_plan) result arrived this turn
        completed_read_actions: list[tuple[str, str, str]] = []
        revision_rejections: list[str] = []

        for name, report in results:
            if name == _REVISE_PLAN:
                # Apply the plan revision from the tool's own
                # (already-sanitised) return value, not the raw call args -
                # see build_deep_agent's revise_plan closure.
                new_steps = _parsed_revision(report)
                crit = state.get("criteria") or []
                if new_steps == [] and not (crit and all(c.get("satisfied") for c in crit)):
                    remaining = planning.pending_steps(plan)
                    descriptions = "; ".join(str(s.get("description", "")) for s in remaining) or "(none)"
                    unmet = "; ".join(str(c.get("description", c.get("id", "criterion")))
                                       for c in crit if not c.get("satisfied")) or "criteria are not yet supported"
                    revision_rejections.append(
                        "Plan revision REJECTED: your requested empty list was not applied because acceptance "
                        f"criteria remain unsupported ({unmet}). The actual pending plan is unchanged: "
                        f"{descriptions}. No pending plan steps does not mean the user's request is satisfied. "
                        "Gather targeted evidence for the unmet criterion, or give a final answer that states "
                        "what could not be established."
                    )
                    continue
                if new_steps is not None:
                    fallback_task = state.get("task") or _task_from_messages(state["messages"])
                    plan = planning.revise_pending_steps(plan or planning.new_plan(fallback_task), new_steps)
                continue
            delegated = True
            payload = _delegation_payload(report)
            readable_report = str(payload.get("report", report))
            observations.append(readable_report)
            delegated_result = readable_report
            # Record what this delegation actually established on
            # the current plan step - from the tool's own result, clipped, and
            # classified deterministically (see evidence.classify_result).
            action_arg = state.get("current_objective") or name
            # get_file's result shape ({"file"/"content"/...} or
            # {"error": ...}) is not what classify_result/summarize_evidence
            # expect (those assume either the code/verify {"state", ...}
            # shape or an Explorer-style free-text report) - use its own
            # dedicated, conservative classifier/summarizer instead so a
            # rejected/failed read is never misclassified as usable (ok)
            # evidence, and so the recorded excerpt is the actual file
            # content plus its path/line-range metadata rather than the raw
            # JSON wrapper.
            if name == _GET_FILE:
                result = ev.classify_get_file_result(readable_report)
                step_evidence_excerpt = ev.summarize_get_file_evidence(readable_report)
            elif name == _FIND_FILES:
                result = ev.classify_find_files_result(readable_report)
                step_evidence_excerpt = ev.summarize_evidence(name, readable_report, payload, (_CODE, _VERIFY))
            elif name == _FIND_SYMBOL:
                result = ev.classify_find_symbol_result(readable_report)
                step_evidence_excerpt = ev.summarize_evidence(name, readable_report, payload, (_CODE, _VERIFY))
            else:
                result = ev.classify_result(name, readable_report, payload, (_CODE, _VERIFY))
                step_evidence_excerpt = ev.summarize_evidence(name, readable_report, payload, (_CODE, _VERIFY))
            if name in _PROGRESS_TOOLS:
                fingerprint = result_fingerprint(name, readable_report)
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
            provenance = ev.evidence_provenance(name, payload, (_CODE, _VERIFY))
            if name == _GET_FILE:
                provenance = {**provenance, **ev.get_file_provenance(readable_report)}
            elif name == _FIND_FILES:
                provenance = {**provenance, **ev._find_files_provenance(payload)}
            elif name == _FIND_SYMBOL:
                symbol_matches = payload.get("matches") if isinstance(payload.get("matches"), list) else []
                provenance = {**provenance,
                              "match_type": payload.get("match_type"),
                              "ambiguous": bool(payload.get("ambiguous")),
                              "symbols": [m.get("symbol") for m in symbol_matches if isinstance(m, dict)],
                              "files": [m.get("file") for m in symbol_matches if isinstance(m, dict)]}
            provenance.update({"repository_epoch": id(service.index),
                               "progress_generation": generation})
            if plan is not None:
                plan = ev.record_step_evidence(plan, name, action_arg, result, step_evidence_excerpt, provenance=provenance)
            if name in (_INVESTIGATE, _GET_FILE, _FIND_FILES, _FIND_SYMBOL) and result != ev.RESULT_OK \
                    and not (name == _GET_FILE and result == ev.RESULT_INCONCLUSIVE):
                # An empty/failed investigation or read gave nothing to build
                # on - but a truncated get_file (RESULT_INCONCLUSIVE) is real,
                # partial evidence, not a failure: closing its plan step as
                # FAILED told the model "that approach failed" instead of
                # "keep reading with start_line" (see evidence.py's own
                # _read_gaps/_fully_read_files, which already treat a
                # truncated read as legitimate partial coverage, not a
                # failure). Leaving step_succeeded True here lets the step
                # close as COMPLETED while the read stays RESULT_INCONCLUSIVE
                # in evidence - so it still cannot satisfy a criterion on its
                # own (apply_assessment still requires RESULT_OK, or a
                # promotion via _fully_read_files) - it just stops being
                # misreported as an outright failure.
                step_succeeded = False
            if name in _READ_ONLY_PLAN_TOOLS:
                evidence.append(report)
                if result in (ev.RESULT_OK, ev.RESULT_INCONCLUSIVE):
                    completed_read_actions.append((name, readable_report, action_arg))
            elif name in (_CODE, _VERIFY):
                # Both end their report the same way ("Result: <state>.",
                # see safety.py/_succeeded below), so both drive
                # verification_info/failure_info identically - verify_repository
                # is just a way to get that same kind of outcome for the
                # repository's full command instead of code_repository's
                # scoped one.
                verification_info = {"tool": name, "state": payload.get("state"),
                                     "status": (payload.get("verification") or {}).get("status"),
                                     "report": readable_report}
                failure_info = (None if payload.get("state") in (SUCCESS, NO_TESTS, UNAVAILABLE)
                                else {"tool": name, "report": readable_report})
                # A code_repository/verify_repository step is only
                # "successful" (closes its plan step as COMPLETED rather
                # than FAILED) when its own report says so -
                # investigate_repository has no pass/fail outcome of its
                # own, so gathering its report at all is success for the
                # step_succeeded default above.
                if payload.get("state") not in (SUCCESS, NO_TESTS, UNAVAILABLE):
                    step_succeeded = False

        update: dict = {
            "evidence": evidence,
            "observations": observations,
            "delegated_result": delegated_result,
            "verification_info": verification_info,
            "failure_info": failure_info,
        }
        if revision_rejections:
            update["messages"] = [HumanMessage(message) for message in revision_rejections]
        # Close out the plan step `decide` opened for this turn's
        # delegation(s), now that their report(s) are in - a revise_plan-
        # only turn never started a step, so there is nothing to close.
        if plan is not None:
            if delegated and plan.get("current_step_id") is not None:
                plan = planning.finish_current_step(plan, step_succeeded)
            update["plan"] = plan
        update.update({"result_fingerprints": fingerprints,
                       "progress_generation": generation,
                       "no_progress_count": no_progress_count,
                       "control_metrics": control_metrics})

        # First take a deterministic, read-only step only when the lookup
        # returned one exact source and the task asks for its contents.
        candidate = None
        if completed_read_actions and plan is not None:
            name, report, action_arg = completed_read_actions[-1]
            candidate = deterministic_followup(name, report, state.get("task") or "")
            if candidate is not None:
                path = (candidate.get("arguments") or {}).get("path", "")
                request_intent = intent_mod.get_intent(state)
                if not intent_mod.source_allows_path(request_intent, path):
                    candidate = None
            if candidate is not None and candidate.get("tool") == _GET_FILE:
                args = candidate.get("arguments") or {}
                read_objective = f"Read {args.get('path')} source identified by {name}"
                plan = planning.start_step(plan, read_objective)
                update["plan"] = plan
                update["current_objective"] = _action_arg_for_call(_GET_FILE, args)
                update["pending_followup"] = candidate
                control_metrics["deterministic_followups"] = list(
                    control_metrics.get("deterministic_followups") or [])
                control_metrics["deterministic_followups"].append(f"{name}->get_file ({candidate['reason']})")
                control_metrics["deterministic_followup_count"] = len(control_metrics["deterministic_followups"])
                update["control_metrics"] = control_metrics
                update["runtime_metrics"] = invocation_metrics(dict(state, control_metrics=control_metrics))
                return update

        # Normal reads return directly to the decision loop. Grade only after
        # an automatically selected exact-source read whose content overlaps
        # an unmet criterion enough to plausibly establish it.
        if (plan is not None and was_automatic_followup and completed_read_actions
                and _likely_completing_followup(plan, criteria)):
            grading_state = dict(state, plan=plan, criteria=criteria, observations=observations,
                                 verification_info=verification_info, failure_info=failure_info)
            criteria, plan, support = grade(grading_state, state.get("task") or "", plan)
            update.update({"criteria": criteria, "plan": plan})
            if support == ev.SUPPORTED:
                grading_state.update({"criteria": criteria, "plan": plan})
                fields = conclude(grading_state, plan, criteria, support, "", step_limit_reached=False)
                update.update(fields)
                update["messages"] = [AIMessage(content=fields["final_result"])]
        update["runtime_metrics"] = invocation_metrics(dict(state, control_metrics=control_metrics))
        return update

    builder = StateGraph(DeepAgentState)
    builder.add_node("plan_task", plan_task)
    builder.add_node("decide", decide)
    builder.add_node("delegate", delegate)
    builder.add_node("update_state", update_state)
    builder.add_node("prepare_followup", prepare_followup)
    builder.add_edge(START, "plan_task")
    builder.add_edge("plan_task", "decide")
    # A "retry" route (decide -> decide) handles turns that
    # neither delegate nor finish: a final answer the evidence does not yet
    # support, or a skipped blind repeat. Each still costs one step_count.
    builder.add_conditional_edges("decide", _route_after_decide,
                                  {"tools": "delegate", "end": END, "retry": "decide"})
    builder.add_edge("delegate", "update_state")
    # Loop back to `decide` instead of ending here, so the Deep
    # Agent can keep delegating until it either gives a final answer or
    # `decide` itself forces a wrap-up at the step limit (see `decide`).
    builder.add_conditional_edges("update_state", _route_after_update,
                                  {"followup": "prepare_followup", "decide": "decide", "end": END})
    builder.add_edge("prepare_followup", "delegate")

    # plan_task (1) + decide/delegate/update_state (3) per delegated
    # iteration + 1 for the forced wrap-up call + margin.
    return builder.compile().with_config(recursion_limit=6 * config.max_agent_steps + 8)


def _likely_completing_followup(plan: dict, criteria: list[dict]) -> bool:
    """Whether the latest result is a relevant, automatic exact-source read."""
    steps = ev.evidence_steps(plan)
    if not steps:
        return False
    latest = steps[-1]
    if latest.get("tool") != _GET_FILE:
        return False
    if latest.get("result") != ev.RESULT_OK:
        return False
    content = str(latest.get("evidence", ""))
    if ":\n" in content:
        content = content.split(":\n", 1)[1]  # discard path/range provenance
    observed = ev._content_words(content)
    if not observed:
        return False
    for criterion in criteria:
        if criterion.get("satisfied"):
            continue
        wanted = ev._content_words(str(criterion.get("description", "")))
        if wanted and len(wanted & observed) >= min(2, len(wanted)):
            return True
    return False


def _route_after_decide(state: dict) -> str:
    """Delegate if the last message asks for a tool; end once `decide` has
    set a final result; otherwise `decide` deliberately held the answer back
    and runs again."""
    if route_after_agent(state) == "tools":
        return "tools"
    return "end" if state.get("final_result") is not None else "retry"


def _route_after_update(state: dict) -> str:
    if state.get("final_result") is not None:
        return "end"
    return "followup" if state.get("pending_followup") else "decide"


def _repeated_read_action(plan, tool: str, argument: str, generation: int, repository_epoch: int) -> str | None:
    """Find same-state read actions with no intervening new information."""
    if not plan or tool not in _READ_ONLY_PLAN_TOOLS:
        return None
    for step in reversed(ev.evidence_steps(plan)):
        if step.get("tool") != tool:
            continue
        provenance = step.get("provenance") or {}
        if (provenance.get("repository_epoch") != repository_epoch
                or provenance.get("progress_generation") != generation):
            continue
        action_key = str(step.get("action_key") or "")
        previous_argument = action_key.partition("|")[2]
        if tool == _GET_FILE:
            matches = planning._normalize(argument.replace("\\", "/")) == previous_argument
        else:
            matches = similar_action(argument, previous_argument)
        if matches:
            return f"{tool}|{previous_argument}|{generation}|{repository_epoch}"
    return None


def _action_arg_for_call(name: str, args: dict) -> str:
    """The text used both as `current_objective` and as the action-identity
    argument (ev.is_blind_repeat / ev.record_step_evidence) for one
    delegation/get_file call this turn.

    The three delegation tools take a single free-text query/task argument,
    so falling through to that (or the bare tool name, if neither is
    present) has always been the right identity for them. get_file's
    arguments are structured (`path`, optional `start_line`/`end_line`)
    instead - naively falling through to the bare tool name for it would
    collapse EVERY get_file call onto the same action_key ("get_file"),
    which would wrongly make a failed read of one path block a completely
    different path (or a failed range block a different range) via
    is_blind_repeat. Building the identity from
    the actual path/range keeps get_file:README.md and
    get_file:src/app.py:20-40 distinct, without touching _normalize's (or
    is_blind_repeat's) own semantics for the three delegation tools."""
    if name == _FIND_FILES:
        return args.get("pattern") or name
    if name == _GET_FILE:
        path = args.get("path") or ""
        start, end = args.get("start_line"), args.get("end_line")
        if start is not None or end is not None:
            return f"{path}:{start if start is not None else ''}-{end if end is not None else ''}"
        return path
    return args.get("query") or args.get("task") or name


def _is_light_readonly_task(intent: dict, plan) -> bool:
    """Whether this invocation is the narrow "light, read-only"
    case the sufficiency shortcuts below (grade()'s pending-step drop and
    _redundant_investigation_guard) are allowed to act on. Deliberately
    conservative - excludes anything that is not clearly both light and
    read-only:

    - kind == KIND_CHANGE: the task itself asks to modify the repository.
    - rigor == RIGOR_STRICT: the task explicitly wants rigorous checking.
    - depth == DEPTH_DEEP: a deep task must not stop early just because a
      coarse criterion happens to already be satisfied - "deep" is
      deliberately NOT treated as implying
      "strict" here, even though the two often go together in practice.
    - a code_repository step ran anywhere in this plan, whether it
      succeeded or failed: a real mutation attempt means this was not, in
      fact, a read-only task, regardless of what intent.kind guessed
      up front - never trust intent.kind alone for this.
    """
    if intent.get("kind") == intent_mod.KIND_CHANGE:
        return False
    if intent.get("rigor") == intent_mod.RIGOR_STRICT:
        return False
    if intent.get("depth") == intent_mod.DEPTH_DEEP:
        return False
    if plan and any(s.get("tool") == _CODE for s in ev.evidence_steps(plan)):
        return False
    return True


def _redundant_investigation_guard(state: DeepAgentState, plan) -> bool:
    """True when another investigate_repository delegation would
    only repeat evidence-gathering that is already sufficient, so `decide`
    should skip it (see its delegate_calls handling below).

    Only ever true once a PRIOR grade() call this invocation already
    marked every criterion satisfied - `state["criteria"]` carries those
    `satisfied` flags forward from whichever branch last called grade()
    (see `decide`). On the very first pass every criterion still has
    satisfied=False from evidence.new_criterion, so this can never fire
    before grading has actually happened once.
    Reads apply_assessment's own output; it never re-derives or bypasses
    it. Narrow the same way grade()'s pending-step drop is (see
    _is_light_readonly_task): never for a strict/deep/change task, never
    once a mutation was attempted, never when evidence has failed.
    """
    criteria = state.get("criteria") or []
    if not criteria or not all(c["satisfied"] for c in criteria):
        return False
    if _has_failed_evidence(state, plan):
        return False
    return _is_light_readonly_task(intent_mod.get_intent(state), plan)


def _has_successful_exact_investigation(plan, query: str) -> bool:
    """Whether this exact investigate_repository query already produced
    usable evidence in the current plan. This is deliberately scoped to
    successful investigations; failed attempts remain governed by the
    existing blind-repeat guard and its retry rules."""
    if not plan:
        return False
    action_key = f"{_INVESTIGATE}|{planning._normalize(query)}"[:300]
    return any(step.get("tool") == _INVESTIGATE and step.get("result") == ev.RESULT_OK
               and step.get("action_key") == action_key for step in ev.evidence_steps(plan))


def _has_failed_evidence(state: DeepAgentState, plan) -> bool:
    """The shared, conservative "did anything fail" check used by
    the redundant-investigation guard.

    `state["failure_info"]` is set and cleared by code_repository/
    verify_repository results (see update_state) and says nothing about a
    FAILED investigate_repository (or direct get_file) step recorded on the
    plan itself. Such a failed read must prevent the redundant-investigation
    guard from suppressing later exploration. Pending read-only plan cleanup
    uses a separate rule: a failed read does not outweigh later evidence that
    already supports every criterion."""
    if state.get("failure_info") is not None:
        return True
    return any(s.get("result") == ev.RESULT_FAILED and s.get("tool") in (_INVESTIGATE, _GET_FILE, _FIND_FILES, _CODE, _VERIFY)
               for s in ev.evidence_steps(plan))


def _assess(llm, task: str, criteria: list[dict], plan: dict, verification_info,
            cache: dict[str, dict] | None = None, metrics: dict | None = None,
            repository_identity: str = "", api_key: str | None = None) -> tuple[list[dict], dict]:
    """Ask the LLM (plain call, no tools) which acceptance criteria
    the recorded evidence supports, citing evidence item ids - see
    prompts.CRITERIA_CHECK_PROMPT. The reply is only a proposal:
    evidence.apply_assessment keeps a criterion satisfied only if it cites a
    real step with usable evidence. Conservative: no usable evidence, an
    LLM failure or an unparseable reply all mean "not satisfied"."""
    items = ev.evidence_for_grader(plan, criteria)
    verdicts: dict = {}
    grader_error = ""
    # Whether to call the grader at all is decided from the FULL recorded
    # evidence (unaffected by evidence_for_grader's payload trimming above),
    # exactly as before that trimming existed - otherwise narrowing `items`
    # to only already-cited evidence could make this check miss a RESULT_OK
    # step that just isn't cited by any criterion yet, and skip grading
    # when a fresh verdict was still called for.
    if any(s["result"] == ev.RESULT_OK for s in ev.evidence_steps(plan)):
        verification = _clip(verification_info["report"]) if verification_info else "(no verification was run)"
        payload = {"objective": task,
                   "criteria": [{k: c[k] for k in ("id", "description", "evidence_needed")} for c in criteria],
                   "evidence": items, "verification": verification}
        fingerprint = hashlib.sha256(json.dumps(
            {"repository": repository_identity, "assessment": payload},
            ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")).hexdigest()
        cached = cache.get(fingerprint) if cache is not None else None
        if cached is not None:
            verdicts = cached
            if metrics is not None:
                metrics["grader_cache_hits"] += 1
        else:
            if metrics is not None:
                metrics["grader_cache_misses"] += 1
                metrics["grader_calls"] += 1
            started = time.perf_counter()
            try:
                reply = llm.invoke([SystemMessage(CRITERIA_CHECK_PROMPT),
                                    HumanMessage(json.dumps(payload, ensure_ascii=True))],
                                   config={"metadata": {"ari_stage": "evidence_grading"}})
                verdicts = ev.parse_assessment(_text(reply.content))
                if not verdicts:
                    grader_error = "unparseable reply"
                elif cache is not None:
                    cache[fingerprint] = verdicts
                    if len(cache) > 128:
                        cache.pop(next(iter(cache)))
            except Exception as exc:
                verdicts = {}
                grader_error = _model_error_reason(exc, api_key)
            finally:
                if metrics is not None:
                    metrics["grader_latency_ms"] += int((time.perf_counter() - started) * 1000)
    new_criteria, new_plan = ev.apply_assessment(criteria, plan, verdicts)
    if grader_error:
        # Surface "the grader could not run" instead of letting it look like
        # "the evidence does not support it" (nothing is satisfied either way).
        new_criteria = [c if c["satisfied"] else dict(c, note=f"{ev.GRADER_FAILED_NOTE} ({grader_error})")
                        for c in new_criteria]
    return new_criteria, new_plan


def _clip(text: str, limit: int = _CHECK_MAX_CHARS) -> str:
    text = str(text)
    return text if len(text) <= limit else text[:limit] + " ...[truncated]"


def _model_error_reason(error: Exception, api_key: str | None = None) -> str:
    """Keep provider failure details visible without exposing configured secrets."""
    detail = " ".join(str(error).split())
    if api_key:
        detail = detail.replace(api_key, "[redacted]")
    detail = detail[:320]
    status = getattr(error, "status_code", None)
    response = getattr(error, "response", None)
    if status is None and response is not None:
        status = getattr(response, "status_code", None)
    rate_limited = status == 429 or "429" in detail or "ratelimit" in type(error).__name__.casefold()
    prefix = "provider rate limit (HTTP 429)" if rate_limited else type(error).__name__
    return f"{prefix}: {detail}" if detail else prefix


def _task_from_messages(messages: list) -> str:
    return next((str(m.content) for m in reversed(messages) if getattr(m, "type", "") == "human"), "")


def _text(content) -> str:
    if isinstance(content, list):  # some providers return content blocks
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return str(content)


def _pending_tool_results(messages: list) -> list[tuple[str, str]]:
    """The (tool name, report text) of every delegation/revise_plan tool
    result produced by the most recent `decide` -> `delegate` step, in
    message order. There is normally exactly one; a list (rather than a
    single result) is what lets `update_state` fold in every report when
    the model asked for more than one capability in the same turn instead
    of silently dropping all but the last one. It also recognizes
    `revise_plan` results.
    """
    last_calls: dict[str, str] = {}
    for message in reversed(messages):
        if getattr(message, "type", "") == "ai":
            for call in getattr(message, "tool_calls", None) or []:
                last_calls[call["id"]] = call["name"]
            if last_calls:
                break
    if not last_calls:
        return []

    results: list[tuple[str, str]] = []
    for message in messages:
        if getattr(message, "type", "") != "tool":
            continue
        call_id = getattr(message, "tool_call_id", None)
        name = last_calls.get(call_id)  # type: ignore
        if name in _PLAN_AWARE_TOOL_NAMES:
            results.append((name, str(message.content)))
    return results


def _parsed_revision(report: str) -> list[str] | None:
    """Extract the proposed replacement list from a `revise_plan` tool
    result (see build_deep_agent's revise_plan closure). Returns None (and
    `update_state` then leaves the plan's pending steps untouched) if the
    result is not the expected JSON shape - defensive only, since
    `revise_plan` always returns this shape itself."""
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
    """Read the structured terminal state emitted by delegation tools."""
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
