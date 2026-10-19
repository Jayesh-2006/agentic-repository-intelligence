#!/usr/bin/env python3
"""
Agentic Repository Intelligence - application entry point.

One question, then exit:
    python main.py --repo /path/to/repository --query "Where is RepoService.refresh defined?"
    python main.py --repo /path/to/repository --index-dir data/index --query "Who calls index_repository?"

Interactive (one query per prompt; type help, status, refresh, or exit/quit):
    python main.py --repo /path/to/repository
    python main.py --repo /path/to/repository --index-dir data/index

Configuration (GEMINI_API_KEY, GEMINI_MODEL, ...) comes from .env or the
environment - see .env.example. --index-dir loads an index saved with
scripts/build_index.py; without it the repository is indexed in memory.
The progress trace goes to stderr, the final response to stdout.
"""

import argparse
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from agentrepo.agent.app import (  # noqa: E402
    AppError, RichTerminalUI, build_app, completion_line, load_app_config, open_repository, run_query,
)

EXIT_WORDS = {"exit", "quit", ":q"}
HELP_WORDS = {"help", "?", "/help"}
STATUS_WORDS = {"status", "/status"}
REFRESH_WORDS = {"refresh", "/refresh"}
_DEFAULT_UI = RichTerminalUI(use_rich=True)

INTERACTIVE_HELP = """Interactive Commands:
  help, ?       Show this help message and example questions
  status        Show the active repository, symbol/chunk counts, and model configuration
  refresh       Re-index the repository from disk
  exit, quit    Exit the interactive session

Example Questions:
  - Where is RepoService.refresh defined?
  - Who calls index_repository?
  - What does collect_module call?
  - Which modules import agentrepo.filesystem?
  - How is path_to_module implemented?
  - What is the purpose of this repository?
  - What environment variables are supported?
  - What are the main modules in this repository?
"""


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--repo", required=True, help="repository to operate on")
    parser.add_argument("--index-dir", help="load a saved index (scripts/build_index.py) instead of indexing --repo in memory")
    parser.add_argument("--query", help="question/task for the agent; omit to start interactive mode")
    parser.add_argument("--max-steps", type=int, help="override MAX_AGENT_STEPS")
    parser.add_argument("--plain", action="store_true", help="disable Rich formatting and output plain text")
    parser.add_argument("--debug", action="store_true", help="show full tracebacks on errors")
    args = parser.parse_args(argv)
    if args.query is not None and not args.query.strip():
        parser.error("--query must not be empty")
    return args


def print_trace(line: str) -> None:
    _DEFAULT_UI.print_trace(line)


def answer_query(app, query: str, ui: RichTerminalUI | None = None) -> None:
    active_ui = ui or _DEFAULT_UI
    answer, completion = run_query(
        app.agent, query, on_line=active_ui.print_trace, service=app.service
    )
    active_ui.print_answer(answer, completion)


def report_error(error: Exception, debug: bool) -> None:
    print(f"Error: {error}", file=sys.stderr, flush=True)
    if debug:
        traceback.print_exception(error, file=sys.stderr)
    elif not isinstance(error, AppError):
        print("(run with --debug for the full traceback)", file=sys.stderr, flush=True)


def interactive(app, debug: bool, ui: RichTerminalUI | None = None) -> int:
    active_ui = ui or _DEFAULT_UI
    print(
        "Type a question about the repository. Type 'help' for examples, 'status' for index info, or 'exit' to quit.\n",
        flush=True,
    )
    while True:
        try:
            query = input("Query > ").strip()
        except (EOFError, KeyboardInterrupt):
            print(flush=True)
            return 0
        if not query:
            continue
        lowered = query.lower()
        if lowered in EXIT_WORDS:
            return 0
        if lowered in HELP_WORDS:
            print(INTERACTIVE_HELP, flush=True)
            continue
        if lowered in STATUS_WORDS:
            idx = app.service.index
            print(
                f"Repository: {idx.root}\n"
                f"Indexed Python files: {len(idx.files)} | Total repo files: {len(idx.repo_files or [])}\n"
                f"Symbols: {len(idx.symbols)} | Edges: {len(idx.edges)} | Chunks: {len(idx.chunks)}\n"
                f"Model: {app.config.model} (API key configured: {'yes' if app.config.api_key else 'no'})\n",
                flush=True,
            )
            continue
        if lowered in REFRESH_WORDS:
            res = app.service.refresh()
            if "error" in res:
                print(f"Refresh failed: {res['error']}", file=sys.stderr, flush=True)
            else:
                print(
                    f"Refreshed repository index: {res.get('file_count')} files, "
                    f"{res.get('symbol_count')} symbols.\n",
                    flush=True,
                )
            continue
        try:
            answer_query(app, query, ui=active_ui)
        except KeyboardInterrupt:
            print("\nInterrupted.", file=sys.stderr, flush=True)
        except AppError as error:
            report_error(error, debug)  # one failed query does not end the session
        print(flush=True)


def main(argv=None) -> int:
    args = parse_args(argv)
    ui = RichTerminalUI(use_rich=not getattr(args, "plain", False))
    try:
        config = load_app_config(args.max_steps)
        service, source = open_repository(args.repo, args.index_dir)
        app = build_app(service, config, source)

        retriever_name = getattr(service.retriever, "backend_name", type(service.retriever).__name__)
        ui.print_banner(
            repo_path=str(Path(args.repo).expanduser().resolve()),
            index_dir=args.index_dir,
            symbol_count=len(service.index.symbols),
            chunk_count=len(service.index.chunks),
            model_name=config.model,
            retriever_name=retriever_name,
        )

        if args.query:
            answer_query(app, args.query, ui=ui)
            return 0
        return interactive(app, args.debug, ui=ui)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr, flush=True)
        return 130
    except AppError as error:
        report_error(error, args.debug)
        return 2
    except Exception as error:  # last resort: still no raw traceback for normal users
        report_error(error, args.debug)
        return 1


if __name__ == "__main__":
    sys.exit(main())
