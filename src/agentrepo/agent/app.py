"""Shared application setup and query execution.

The module loads configuration, opens the indexed repository, builds the
agent, and runs each query through the Main Agent.

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
from .observability import (
    RunTraceContext,
    format_attribution_metrics,
    format_run_diagnostic_summary,
    reset_run_context,
    start_run_context,
)
from .utils import describe_update, final_answer


class AppError(Exception):
    """A problem to report to the user as a single message, without a traceback."""


@dataclass
class AgentApp:
    agent: object          # the compiled Deep Agent graph (or LazyDeepAgent)
    service: RepoService
    config: AgentConfig
    source: str            # human-readable description of where the index came from


class LazyDeepAgent:
    """Construct the LLM-backed Main Agent graph on first query execution."""

    def __init__(self, service: RepoService, config: AgentConfig):
        self.service = service
        self.config = config
        self._graph = None

    def stream(self, input_state, stream_mode="updates", **kwargs):
        if self._graph is None:
            if not self.config.api_key:
                raise AppError(
                    "Configuration error: GEMINI_API_KEY is not set. Copy .env.example to .env, "
                    "add your Gemini API key (https://aistudio.google.com/apikey), or export "
                    "GEMINI_API_KEY in your shell."
                )
            llm = build_llm(self.config)
            self._graph = build_deep_agent(self.service, self.config, llm=llm)
        return self._graph.stream(input_state, stream_mode=stream_mode, **kwargs)


def load_app_config(max_steps: int | None = None) -> AgentConfig:
    """Load the existing configuration (.env / environment). max_steps, if
    given, overrides MAX_AGENT_STEPS (validated by AgentConfig as usual)."""
    try:
        # Defer the API key check until LazyDeepAgent initializes the Main Agent graph.
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
    """Run the query through the Main Agent.
    Returns (response, completion state)."""
    query_started = time.perf_counter()
    service_metrics_before = _service_metrics(service)
    api_key = getattr(getattr(agent, "config", None), "api_key", None)
    ctx, ctx_token = start_run_context(on_line=on_line, api_key=api_key)
    try:
        if on_line:
            on_line("[Main] run started")
        return _run_query_with_context(
            agent, query, on_line, service, query_started, service_metrics_before, ctx
        )
    finally:
        reset_run_context(ctx_token)


def _run_query_with_context(
    agent,
    query: str,
    on_line: Callable[[str], None] | None,
    service: RepoService | None,
    query_started: float,
    service_metrics_before: dict,
    ctx: RunTraceContext,
) -> tuple[str, dict | None]:
    input_state: dict = {"messages": [{"role": "user", "content": query}], "task": query}

    messages: list = []
    completion = None
    node_times: dict[str, float] = {}
    node_counts: dict[str, int] = {}
    planned_steps = 0
    grader_metrics: dict = {}
    callback = _runtime_callback()

    def report_metrics(run_exc: Exception | None = None) -> None:
        if on_line:
            on_line(format_run_diagnostic_summary(ctx, completion, run_exception=run_exc))

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
                    for line in describe_update(node, update, agent="Main"):
                        on_line(line)
                messages.extend(update.get("messages", []))
                completion = update.get("completion") or completion
                grader_metrics.update(update.get("runtime_metrics") or {})
            stream_started = time.perf_counter()
    except AppError as error:
        report_metrics(error)
        raise
    except ImportError as error:
        report_metrics(error)
        raise AppError(f"A required package is missing ({error}). "
                       "Install the dependencies with: pip install -r requirements.txt") from error
    except Exception as error:
        report_metrics(error)
        raise AppError(f"The agent run failed: {error}") from error
    report_metrics()
    return final_answer(messages), completion


def _report_runtime_metrics(on_line: Callable[[str], None], query_started: float,
                            messages: list | None = None, node_times: dict[str, float] | None = None,
                            node_counts: dict[str, int] | None = None, planned_steps: int = 0,
                            grader_metrics: dict | None = None, callback=None,
                            repository_metrics: dict | None = None,
                            ctx: RunTraceContext | None = None) -> None:
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

    if ctx is not None:
        ctx_llm = sum(ctx.agent_llm_calls.values())
        if ctx_llm > llm_calls:
            llm_calls = ctx_llm
        ctx_tools = sum(sum(t.values()) for t in ctx.agent_tool_calls.values())
        if ctx_tools > tool_calls:
            tool_calls = ctx_tools
        if not tool_counts and ctx.agent_tool_calls:
            tool_counts = {}
            for by_tool in ctx.agent_tool_calls.values():
                for t_name, cnt in by_tool.items():
                    tool_counts[t_name] = tool_counts.get(t_name, 0) + cnt
        if not token_data and ctx.stage_details:
            ctx_in = sum(d.get("input_tokens", 0) for d in ctx.stage_details.values())
            ctx_out = sum(d.get("output_tokens", 0) for d in ctx.stage_details.values())
            if ctx_in or ctx_out:
                input_tokens, output_tokens = ctx_in, ctx_out
                token_data = True

    fields = [f"total_ms={int((time.perf_counter() - query_started) * 1000)}",
              f"llm_calls={llm_calls}", f"tool_calls={tool_calls}",
              f"decision_calls={stages.get('decision', {}).get('calls', (ctx.stage_details.get('Main:decision', {}).get('calls', 0) if ctx else 0))}",
              f"planning_calls={stages.get('planning', {}).get('calls', (ctx.stage_details.get('Main:planning', {}).get('calls', 0) if ctx else 0))}",
              f"wrap_up_calls={stages.get('wrap_up', {}).get('calls', (ctx.stage_details.get('Main:wrap_up', {}).get('calls', 0) if ctx else 0))}",
              f"plan_steps={planned_steps}",
              "tool_counts=" + (",".join(f"{name}:{count}" for name, count in sorted(tool_counts.items())) or "none"),
              f"duplicate_tool_calls={duplicate_calls}", f"tool_result_chars={tool_result_chars}"]
    if token_data:
        fields.extend((f"input_tokens={input_tokens}", f"output_tokens={output_tokens}"))
    else:
        fields.append("token_usage=unavailable")
    if node_times:
        fields.append("node_ms=" + ",".join(f"{name}:{int(duration)}" for name, duration in sorted(node_times.items())))
    if node_counts:
        fields.append("node_calls=" + ",".join(f"{name}:{count}" for name, count in sorted(node_counts.items())))
    grader_fields = {key: value for key, value in (grader_metrics or {}).items()
                     if key.startswith(("grader_", "criterion_", "conclusion_"))}
    if grader_fields:
        fields.append("evidence_grader=" + ",".join(
            f"{key}:{value}" for key, value in sorted(grader_fields.items())))
    control = grader_metrics or {}
    followups = control.get("deterministic_followups") or []
    fields.extend(("deterministic_followups=" + ("|".join(map(str, followups)) or "none"),
                   f"deterministic_followup_count={control.get('deterministic_followup_count', 0)}",
                   f"planning_parse_errors={control.get('planning_parse_errors', 0)}",
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
    fields.extend(format_attribution_metrics(ctx))
    on_line("[Metrics] " + " ".join(fields))


def _service_metrics(service) -> dict:
    snapshot = getattr(service, "runtime_metrics_snapshot", None)
    return snapshot() if callable(snapshot) else {} # pyright: ignore[reportReturnType]


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


# ------------------------------------------------------------------
# Rich Terminal UI & Structured Answer Parser
# ------------------------------------------------------------------

import re
import sys
from typing import Any

try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich import box
    from rich.markup import escape

    RICH_AVAILABLE = True
except ImportError:  # pragma: no cover
    RICH_AVAILABLE = False
    Console = None  # type: ignore
    escape = lambda s: s  # type: ignore

_SECTION_HEADERS = (
    "Conclusion",
    "Confirmed findings",
    "Reasonable but unconfirmed possibilities",
    "Supporting evidence",
    "Evidence reviewed but not accepted as support",
    "Unknown / unresolved",
    "Unresolved",
    "Agent draft (unverified; not accepted as the answer)",
)

_STEP_BULLET_RE = re.compile(
    r"^-\s*\[(?P<step_id>step_\d+|c\d+)\]\s*"
    r"(?P<action>"
        r"[a-zA-Z_]+:\s*[^:\n]+:\d+(?:-\d+)?"
        r"|[a-zA-Z_]+:\s*[^:\n]+\s*\([^)]*\)"
        r'|[a-zA-Z_]+:\s*"[^"]*"'
        r"|[a-zA-Z_]+:\s*'[^']*'"
        r"|[a-zA-Z_]+:\s*[^:\n]+"
        r"|[a-zA-Z_]+\s+[^:\n]+:\d+(?:-\d+)?"
        r"|[a-zA-Z_]+\([^)]*\)"
        r"|[^:]+"
    r"):\s*(?P<excerpt>.*)$"
)


def parse_structured_answer(raw_answer: str) -> dict[str, Any]:
    """Parse an ARI final answer string into structured sections (`conclusion`,
    `supporting`, `reviewed`, `unresolved`, `draft`, etc.) for rich rendering."""
    text = (raw_answer or "").strip()
    if not text:
        return {"conclusion": "", "sections": []}

    lines = text.splitlines()
    sections: list[tuple[str, list[str]]] = []
    current_title = "Conclusion"
    current_lines: list[str] = []

    for idx, line in enumerate(lines):
        stripped = line.strip()
        clean_title = re.sub(r"^[#*\s_]+|[#*\s_:]+$", "", stripped).strip()
        if idx == 0 and clean_title == "Conclusion":
            current_title = "Conclusion"
            continue
        if (clean_title in _SECTION_HEADERS or stripped in _SECTION_HEADERS) and (idx == 0 or not lines[idx - 1].strip()):
            if current_lines or current_title != "Conclusion":
                sections.append((current_title, current_lines))
            current_title = clean_title if clean_title in _SECTION_HEADERS else stripped
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines or not sections:
        sections.append((current_title, current_lines))

    conclusion_body = ""
    other_sections: list[tuple[str, str]] = []
    for title, body_lines in sections:
        body = "\n".join(body_lines).strip()
        if title == "Conclusion" and not conclusion_body:
            conclusion_body = body
        elif body:
            other_sections.append((title, body))

    return {
        "conclusion": conclusion_body or text,
        "sections": other_sections,
    }


class RichTerminalUI:
    """Renders startup banners, live agent trace lines, and structured answers."""
    # pyright: ignore[reportMissingTypeStubs]

    def __init__(self, *, use_rich: bool = True) -> None:
        self.enabled = bool(use_rich and RICH_AVAILABLE)
        self.stdout = Console(soft_wrap=False) if self.enabled else None # pyright: ignore[reportOptionalCall]
        self.stderr = Console(stderr=True, soft_wrap=False) if self.enabled else None

    def print_banner(
        self,
        *,
        repo_path: str,
        index_dir: str | None,
        symbol_count: int,
        chunk_count: int,
        model_name: str,
        retriever_name: str = "BM25Plus",
    ) -> None:
        if not self.enabled or self.stdout is None:
            print("Repository Intelligence Agent", flush=True)
            print("=============================", flush=True)
            print(f"Repository: {repo_path}", flush=True)
            if index_dir:
                print(f"Index: {index_dir}", flush=True)
            print(f"Symbols: {symbol_count} | Chunks: {chunk_count} | Retriever: {retriever_name}", flush=True)
            print(f"Model: {model_name}\n", flush=True)
            return

        grid = Table.grid(padding=(0, 2)) # pyright: ignore[reportPossiblyUnboundVariable]
        grid.add_column(style="bold cyan", justify="right")
        grid.add_column(style="white")
        grid.add_row("Repository:", str(repo_path))
        if index_dir:
            grid.add_row("Index:", str(index_dir))
        grid.add_row("Indexed:", f"[bold]{symbol_count}[/bold] symbols, [bold]{chunk_count}[/bold] chunks")
        grid.add_row("Retriever:", f"[green]{retriever_name}[/green]")
        grid.add_row("Model:", f"[magenta]{model_name}[/magenta]")

        self.stdout.print(
            Panel( # type: ignore
                grid,
                title="[bold bright_cyan]Agentic Repository Intelligence (ARI)[/bold bright_cyan]",
                border_style="bright_blue",
                box=box.ROUNDED,
                expand=False,
            )
        )
        sys.stdout.flush()

    def print_trace(self, line: str) -> None:
        """Render a single live trace line to stderr with color-coded badges."""
        if not self.enabled or self.stderr is None:
            print(line, file=sys.stderr, flush=True)
            return

        raw = (line or "").strip()
        if not raw:
            return

        text = Text() # type: ignore
        # 1. Lifecycle start / end
        if raw.startswith("[Main] run started"):
            text.append("▶ ", style="bold bright_cyan")
            text.append(raw, style="bold cyan")
        elif "run completed ->" in raw:
            if "SUPPORTED" in raw:
                text.append("✔ ", style="bold bright_green")
                text.append(raw, style="bold green")
            elif "PARTIAL" in raw:
                text.append("◐ ", style="bold yellow")
                text.append(raw, style="bold yellow")
            else:
                text.append("✖ ", style="bold bright_red")
                text.append(raw, style="bold red")

        # 2. Errors, guard blocks, empty results, and context compaction retries
        elif "-> FAILED" in raw or "ERROR" in raw or "budget NOT met" in raw:
            text.append("✖ ", style="bold bright_red")
            text.append(raw, style="red")
        elif "-> BLOCKED" in raw:
            text.append("⚡ ", style="bold yellow")
            text.append(raw, style="yellow")
        elif "retrying with compacted context" in raw:
            text.append("♻ ", style="bold yellow")
            text.append(raw, style="yellow")
        elif "-> EMPTY" in raw:
            text.append("○ ", style="dim yellow")
            text.append(raw, style="dim yellow")

        # 3. LLM stages (planning, decision, wrap_up, grading, claim validation)
        elif "[Main] planning ->" in raw or "revise_plan" in raw:
            text.append("🧭 ", style="bold bright_cyan")
            text.append(raw, style="cyan")
        elif "[Main] decision" in raw or "[Main] model_call(wrap_up)" in raw:
            text.append("💭 ", style="bright_blue")
            text.append(raw, style="bright_white")
        elif "[Main] evidence_grading" in raw:
            text.append("⚖️  ", style="bright_yellow")
            text.append(raw, style="yellow")
        elif "[Main] conclusion_validation" in raw:
            text.append("🛡️  ", style="bright_green")
            text.append(raw, style="green")

        # 4. Repository tools by semantic category
        elif "search_repository" in raw:
            text.append("🔍 ", style="cyan")
            text.append(raw, style="bright_white")
        elif "find_files" in raw:
            text.append("📁 ", style="bright_cyan")
            text.append(raw, style="white")
        elif "find_symbol" in raw or "get_ast" in raw:
            text.append("🧩 ", style="bright_yellow")
            text.append(raw, style="bright_white")
        elif any(
            t in raw
            for t in ("find_callers", "find_callees", "find_importers", "get_dependencies")
        ):
            text.append("🔗 ", style="bright_blue")
            text.append(raw, style="bright_white")
        elif "get_file" in raw or "get_symbol" in raw:
            text.append("📄 ", style="green")
            text.append(raw, style="bright_white")
        elif "write_file" in raw or "verify_repository" in raw or "run_command" in raw:
            text.append("🛠  ", style="magenta")
            text.append(raw, style="bright_magenta")

        # 5. Fallback for any custom/future trace line
        else:
            text.append("• ", style="dim cyan")
            text.append(raw, style="dim white")

        self.stderr.print(text)

    def print_answer(self, answer: str, completion: dict | None = None) -> None:
        """Render the structured answer and evidence sections using Rich panels/tables."""
        if not self.enabled or self.stdout is None:
            print("\nAnswer\n------", flush=True)
            print(answer or "(the agent produced no answer)", flush=True)
            if completion:
                print(f"\nEvidence status: {completion.get('support', 'UNKNOWN')}", flush=True)
            return

        self.stdout.print()
        if not (answer or "").strip():
            self.stdout.print(
                Panel("(the agent produced no answer)", title="Answer", border_style="yellow")
            )
            return

        parsed = parse_structured_answer(answer)
        support = (completion or {}).get("support", "SUPPORTED")
        border_color = (
            "bright_green"
            if support == "SUPPORTED"
            else ("yellow" if support == "PARTIAL" else "bright_red")
        )

        # 1. Main Conclusion Panel rendered as Rich Markdown (supports code blocks, tables, bold, lists)
        self.stdout.print(
            Panel(
                Markdown(parsed["conclusion"]),
                title=f"[bold {border_color}]Conclusion[/bold {border_color}]",
                border_style=border_color,
                box=box.ROUNDED,
                padding=(1, 2),
            )
        )

        # 2. Supporting / Reviewed / Unresolved sections
        for sec_title, sec_body in parsed["sections"]:
            if sec_title in ("Supporting evidence", "Evidence reviewed but not accepted as support"):
                table = Table(
                    title=f"[bold cyan]{sec_title}[/bold cyan]",
                    box=box.SIMPLE_HEAVY,
                    show_lines=True,
                    expand=True,
                )
                table.add_column("Step", style="bold cyan", width=10, no_wrap=True)
                table.add_column("Tool & Location", style="bold green", width=34)
                table.add_column("Recorded Excerpt", style="white")

                any_rows = False
                for line in sec_body.splitlines():
                    m = _STEP_BULLET_RE.match(line.strip())
                    if m:
                        step_id = m.group("step_id")
                        action = m.group("action")
                        excerpt = m.group("excerpt")
                        table.add_row(
                            escape(f"[{step_id}]"),
                            escape(action.strip()),
                            escape(excerpt.strip()),
                        )
                        any_rows = True
                    elif line.strip():
                        table.add_row("", "", escape(line.strip().lstrip("- ")))
                        any_rows = True
                if any_rows:
                    self.stdout.print(table)
            elif "unresolved" in sec_title.lower() or "unverified" in sec_title.lower():
                self.stdout.print(
                    Panel(
                        Markdown(sec_body),
                        title=f"[bold yellow]{sec_title}[/bold yellow]",
                        border_style="yellow",
                        box=box.ROUNDED,
                    )
                )
            else:
                self.stdout.print(
                    Panel( # type: ignore
                        Markdown(sec_body),
                        title=f"[bold cyan]{sec_title}[/bold cyan]",
                        border_style="cyan",
                        box=box.ROUNDED,
                    )
                )

        # 3. Evidence status badge
        if completion:
            c_sup = completion.get("criteria_supported")
            c_tot = completion.get("criteria_total")
            ratio = f" ({c_sup}/{c_tot} criteria)" if c_sup is not None and c_tot is not None else ""
            badge = Text()
            badge.append("Evidence status: ", style="bold white")
            badge.append(f" {support} ", style=f"bold black on {border_color}")
            if ratio:
                badge.append(ratio, style="dim white")
            self.stdout.print(badge)
        sys.stdout.flush()
