"""Shared application setup and query execution.

The module loads configuration, opens the indexed repository, builds the
agent, and selects direct retrieval or agent execution for each query.

Expected startup and runtime failures are raised as `AppError` with the
original exception preserved as `__cause__`.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Callable

from ..service import RepoService
from . import build_deep_agent
from .config import AgentConfig, ConfigError, MissingApiKeyError, load_config
from .model import build_llm
from .simple_retrieval import QueryRetrievalCache, retrieve_query
from .utils import describe_update, final_answer


class AppError(Exception):
    """A problem to report to the user as a single message, without a traceback."""


@dataclass
class AgentApp:
    agent: object          # the compiled Deep Agent graph
    service: RepoService
    config: AgentConfig
    source: str            # human-readable description of where the index came from


class LazyDeepAgent:
    """Construct the LLM-backed graph only if retrieval needs a fallback."""

    def __init__(self, service: RepoService, config: AgentConfig):
        self.service = service
        self.config = config
        self._graph = None

    def stream(self, input_state, stream_mode="updates", **kwargs):
        if self._graph is None:
            if not self.config.api_key:
                raise AppError("Configuration error: GROQ_API_KEY is required for deep investigation.")
            llm = build_llm(self.config)
            self._graph = build_deep_agent(self.service, self.config, llm=llm)
        return self._graph.stream(input_state, stream_mode=stream_mode, **kwargs)


def load_app_config(max_steps: int | None = None) -> AgentConfig:
    """Load the existing configuration (.env / environment). max_steps, if
    given, overrides MAX_AGENT_STEPS (validated by AgentConfig as usual)."""
    try:
        # Simple indexed lookups do not need an LLM credential. The key is
        # checked only if run_query falls back to the Deep Agent.
        config = load_config(require_api_key=False)
        if max_steps is not None:
            config = dataclasses.replace(config, max_agent_steps=max_steps)
    except (MissingApiKeyError, ConfigError) as error:
        raise AppError(f"Configuration error: {error}") from error
    return config


def open_repository(repo: str | Path, index_dir: str | Path | None = None,
                    repo_root: str | Path | None = None) -> tuple[RepoService, str]:
    """Return (RepoService, description of the index source).

    With index_dir, the saved index is loaded (it must already exist - build it
    with scripts/build_index.py); the repository itself is read from
    repo_root (default: repo); repo_root is ignored without index_dir. Without
    index_dir, repo is indexed in memory.
    """
    root = Path((repo_root if index_dir is not None else None) or repo).expanduser()
    if not root.is_dir():
        raise AppError(f"Repository path does not exist or is not a directory: {root}")

    if index_dir is None:
        try:
            return RepoService.from_repository(root), str(root)
        except (OSError, ValueError) as error:
            raise AppError(f"Could not index repository {root}: {error}") from error

    if not Path(index_dir).expanduser().is_dir():
        raise AppError(f"Index directory does not exist: {index_dir}\n"
                       f"Build it first: python scripts/build_index.py {root} {index_dir}")
    try:
        return RepoService.from_index_dir(index_dir, repo_root=root), f"{root} (saved index {index_dir})"
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise AppError(f"Could not load the index in {index_dir}: {error}\n"
                       f"Rebuild it with: python scripts/build_index.py {root} {index_dir}") from error


def build_app(service: RepoService, config: AgentConfig, source: str) -> AgentApp:
    """Prepare local services; defer model and graph initialization until needed."""
    agent = LazyDeepAgent(service, config)
    return AgentApp(agent=agent, service=service, config=config, source=source)


def run_query(agent, query: str, on_line: Callable[[str], None] | None = None,
              service: RepoService | None = None) -> tuple[str, dict | None]:
    """Use simple indexed retrieval when it can prove a direct fact; otherwise
    run the existing Deep Agent. Returns (response, completion state)."""
    query_started = time.perf_counter()
    service_metrics_before = _service_metrics(service)
    input_state: dict = {"messages": [{"role": "user", "content": query}], "task": query}
    query_service: QueryRetrievalCache | None = None
    if service is not None:
        direct_started = time.perf_counter()
        retrieval = retrieve_query(service, query)
        query_service = retrieval.cache
        direct = retrieval.answer
        if direct is not None:
            if on_line:
                for line in direct.trace:
                    on_line(line)
                _report_runtime_metrics(on_line, query_started, direct_ms=(time.perf_counter() - direct_started) * 1000,
                                        cache=query_service, messages=[],
                                        repository_metrics=_metric_delta(service_metrics_before,
                                                                         _service_metrics(service)))
            return direct.text, direct.completion

        # Simple retrieval could not itself answer (or the question did not
        # qualify for the shortcut at all) - hand the agent a pointer to
        # where the indexed search already looked, not the search's content.
        # This is deliberately weaker than passing along the retrieval
        # attempt's actual evidence: it travels as `retrieval_hint`, a plain
        # state field `decide` re-injects as its own system-data message
        # every first turn (see deep_agent.py) - never as a second message
        # in `messages`, which would both get mistaken for the task and get
        # silently dropped from the model-facing history. The agent must
        # still open the file and clear the normal evidence gate before it
        # can cite anything from a hinted location, so a bad or irrelevant
        # hint can only cost a wasted read, never a wrong answer stated as
        # fact.
        if retrieval.hints:
            input_state["retrieval_hint"] = "\n".join(retrieval.hints)

    messages: list = []
    completion = None
    node_times: dict[str, float] = {}
    node_counts: dict[str, int] = {}
    planned_steps = 0
    grader_metrics: dict = {}
    callback = _runtime_callback()

    def report_metrics() -> None:
        if on_line:
            _report_runtime_metrics(
                on_line, query_started,
                direct_ms=(time.perf_counter() - query_started) * 1000 - sum(node_times.values()),
                cache=query_service, messages=messages, node_times=node_times,
                node_counts=node_counts, planned_steps=planned_steps,
                grader_metrics=grader_metrics,
                callback=callback,
                repository_metrics=_metric_delta(service_metrics_before, _service_metrics(service)),
            )

    try:
        stream_started = time.perf_counter()
        for chunk in agent.stream(
            input_state,  # type: ignore
            stream_mode="updates",
            **({"config": {"callbacks": [callback]}} if callback is not None else {}),
        ):
            chunk_started = time.perf_counter()
            for node, update in chunk.items():
                node_times[node] = node_times.get(node, 0.0) + (chunk_started - stream_started) * 1000
                node_counts[node] = node_counts.get(node, 0) + 1
                if node == "plan_task":
                    planned_steps = len((update.get("plan") or {}).get("steps", []))
                if on_line:
                    for line in describe_update(node, update):
                        on_line(line)
                messages.extend(update.get("messages", []))
                completion = update.get("completion") or completion
                grader_metrics.update(update.get("runtime_metrics") or {})
            stream_started = time.perf_counter()
    except AppError:
        report_metrics()
        raise
    except ImportError as error:
        report_metrics()
        raise AppError(f"A required package is missing ({error}). "
                       "Install the dependencies with: pip install -r requirements.txt") from error
    except Exception as error:
        report_metrics()
        raise AppError(f"The agent run failed: {error}") from error
    report_metrics()
    return final_answer(messages), completion


def _report_runtime_metrics(on_line: Callable[[str], None], query_started: float,
                            direct_ms: float = 0.0, cache: QueryRetrievalCache | None = None,
                            messages: list | None = None, node_times: dict[str, float] | None = None,
                            node_counts: dict[str, int] | None = None, planned_steps: int = 0,
                            grader_metrics: dict | None = None, callback=None,
                            repository_metrics: dict | None = None) -> None:
    """Report invocation-local counters; provider token fields are optional."""
    llm_calls = tool_calls = duplicate_calls = tool_result_chars = 0
    input_tokens = output_tokens = 0
    token_data = False
    tool_counts: dict[str, int] = {}
    stages: dict[str, dict] = {}
    tool_ms = 0
    if callback is not None:
        llm_calls = callback.llm_calls
        tool_calls = callback.tool_calls
        duplicate_calls = callback.duplicate_calls
        tool_result_chars = callback.tool_result_chars
        input_tokens = callback.input_tokens
        output_tokens = callback.output_tokens
        token_data = callback.token_data
        tool_counts = callback.tool_counts
        stages = callback.stages
        tool_ms = callback.tool_ms
    else:
        for message in messages or []:
            if getattr(message, "type", "") == "tool":
                tool_result_chars += len(str(getattr(message, "content", "")))

    fields = [f"total_ms={int((time.perf_counter() - query_started) * 1000)}",
              f"llm_calls={llm_calls}", f"tool_calls={tool_calls}",
              f"decision_calls={stages.get('decision', {}).get('calls', 0)}",
              f"planning_calls={stages.get('planning', {}).get('calls', 0)}",
              f"wrap_up_calls={stages.get('wrap_up', {}).get('calls', 0)}",
              f"plan_steps={planned_steps}",
              "tool_counts=" + (",".join(f"{name}:{count}" for name, count in sorted(tool_counts.items())) or "none"),
              f"duplicate_tool_calls={duplicate_calls}", f"tool_result_chars={tool_result_chars}"]
    if token_data:
        fields.extend((f"input_tokens={input_tokens}", f"output_tokens={output_tokens}"))
    else:
        fields.append("token_usage=unavailable")
    if cache is not None:
        fields.extend((f"retrieval_searches={cache.search_calls}",
                       f"retrieval_cache_hits={cache.search_hits}",
                       f"retrieval_cache_misses={cache.search_calls - cache.search_hits}",
                       f"retrieval_search_ms={int(cache.search_ms)}",
                       f"retrieval_result_chars={cache.result_chars}",
                       f"retrieval_file_reads={cache.file_reads}",
                       f"retrieval_file_read_ms={int(cache.file_read_ms)}"))
    if node_times:
        fields.append("node_ms=" + ",".join(f"{name}:{int(duration)}" for name, duration in sorted(node_times.items())))
    if node_counts:
        fields.append("node_calls=" + ",".join(f"{name}:{count}" for name, count in sorted(node_counts.items())))
    grader_fields = {key: value for key, value in (grader_metrics or {}).items()
                     if key.startswith("grader_")}
    if grader_fields:
        fields.append("evidence_grader=" + ",".join(
            f"{key}:{value}" for key, value in sorted(grader_fields.items())))
    control = grader_metrics or {}
    followups = control.get("deterministic_followups") or []
    fields.extend(("deterministic_followups=" + ("|".join(map(str, followups)) or "none"),
                   f"deterministic_followup_count={control.get('deterministic_followup_count', 0)}",
                   f"no_progress_stops={control.get('no_progress_stops', 0)}",
                   f"repeated_action_blocks={control.get('repeated_action_blocks', 0)}"))
    if callback is not None:
        fields.append(f"tool_ms={tool_ms}")
        fields.append("model_stages=" + ",".join(
            f"{name}[calls={data['calls']},in={data['input_tokens']},out={data['output_tokens']},ms={data['latency_ms']}]"
            for name, data in sorted(stages.items())))
    if repository_metrics:
        fields.append("repository=" + ",".join(
            f"{name}[calls={data['calls']},ms={int(data['latency_ms'])}]"
            for name, data in sorted(repository_metrics.items())))
    fields.append(f"pre_agent_ms={int(direct_ms)}")
    on_line("[Metrics] " + " ".join(fields))


def _service_metrics(service) -> dict:
    snapshot = getattr(service, "runtime_metrics_snapshot", None)
    return snapshot() if callable(snapshot) else {}


def _metric_delta(before: dict, after: dict) -> dict:
    result = {}
    for name, current in after.items():
        previous = before.get(name, {})
        calls = current.get("calls", 0) - previous.get("calls", 0)
        duration = current.get("latency_ms", 0.0) - previous.get("latency_ms", 0.0)
        if calls or duration:
            result[name] = {"calls": calls, "latency_ms": duration}
    return result


def _runtime_callback():
    """Collect provider usage and real model/tool timings when available."""
    try:
        from langchain_core.callbacks import BaseCallbackHandler
    except ImportError:
        return None

    class RuntimeCallback(BaseCallbackHandler):
        def __init__(self):
            self._model_starts = {}
            self._tool_starts = {}
            self.llm_calls = self.tool_calls = self.duplicate_calls = 0
            self.tool_result_chars = self.input_tokens = self.output_tokens = self.tool_ms = 0
            self.token_data = False
            self.tool_counts = {}
            self._seen = set()
            self.stages = {}

        def _start_model(self, run_id, kwargs):
            self.llm_calls += 1
            stage = (kwargs.get("metadata") or {}).get("ari_stage", "other")
            self._model_starts[run_id] = (time.perf_counter(), stage)
            data = self.stages.setdefault(stage, {
                "calls": 0, "input_tokens": 0, "output_tokens": 0, "latency_ms": 0,
            })
            data["calls"] += 1

        def on_llm_start(self, serialized, prompts, *, run_id, **kwargs):
            self._start_model(run_id, kwargs)

        def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
            self._start_model(run_id, kwargs)

        def on_llm_end(self, response, *, run_id, **kwargs):
            started = self._model_starts.pop(run_id, None)
            if started is None:
                return
            stage = self.stages[started[1]]
            stage["latency_ms"] += int((time.perf_counter() - started[0]) * 1000)
            output = getattr(response, "llm_output", None) or {}
            usage = output.get("token_usage") or output.get("usage") or {}
            if not usage:
                for group in getattr(response, "generations", []) or []:
                    for generation in group:
                        message = getattr(generation, "message", None)
                        usage = getattr(message, "usage_metadata", None) or {}
                        if usage:
                            break
                    if usage:
                        break
            incoming = usage.get("input_tokens", usage.get("prompt_tokens"))
            outgoing = usage.get("output_tokens", usage.get("completion_tokens"))
            if isinstance(incoming, (int, float)) and isinstance(outgoing, (int, float)):
                incoming, outgoing = int(incoming), int(outgoing)
                self.input_tokens += incoming
                self.output_tokens += outgoing
                stage["input_tokens"] += incoming
                stage["output_tokens"] += outgoing
                self.token_data = True

        def on_llm_error(self, error, *, run_id, **kwargs):
            started = self._model_starts.pop(run_id, None)
            if started is not None:
                self.stages[started[1]]["latency_ms"] += int((time.perf_counter() - started[0]) * 1000)

        def on_tool_start(self, serialized, input_str, *, run_id, **kwargs):
            name = str(serialized.get("name") or "tool")
            self.tool_calls += 1
            self.tool_counts[name] = self.tool_counts.get(name, 0) + 1
            key = name + str(input_str)
            if key in self._seen:
                self.duplicate_calls += 1
            self._seen.add(key)
            self._tool_starts[run_id] = time.perf_counter()

        def on_tool_end(self, output, *, run_id, **kwargs):
            started = self._tool_starts.pop(run_id, None)
            if started is not None:
                self.tool_ms += int((time.perf_counter() - started) * 1000)
            self.tool_result_chars += len(str(output))

        def on_tool_error(self, error, *, run_id, **kwargs):
            started = self._tool_starts.pop(run_id, None)
            if started is not None:
                self.tool_ms += int((time.perf_counter() - started) * 1000)

    return RuntimeCallback()


def completion_line(completion: dict) -> str:
    """One-line, user-facing summary of the evidence check."""
    return f"Evidence status: {completion['support']}"
