#!/usr/bin/env python3
"""
Agentic Repository Intelligence - application entry point.

One question, then exit:
    python main.py --repo /path/to/repository --query "Who calls User.save()?"
    python main.py --repo /path/to/repository --index-dir data/index --query "Who calls User.save()?"

Interactive (one query per prompt; type exit or press Ctrl-D to quit):
    python main.py --repo /path/to/repository
    python main.py --repo /path/to/repository --index-dir data/index

Configuration (GROQ_API_KEY, GROQ_MODEL, ...) comes from .env or the
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
    AppError, build_app, completion_line, load_app_config, open_repository, run_query,
)

EXIT_WORDS = {"exit", "quit"}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--repo", required=True, help="repository to operate on")
    parser.add_argument("--index-dir", help="load a saved index (scripts/build_index.py) instead of indexing --repo in memory")
    parser.add_argument("--query", help="question/task for the agent; omit to start interactive mode")
    parser.add_argument("--max-steps", type=int, help="override MAX_AGENT_STEPS")
    parser.add_argument("--debug", action="store_true", help="show full tracebacks on errors")
    args = parser.parse_args(argv)
    if args.query is not None and not args.query.strip():
        parser.error("--query must not be empty")
    return args


def print_trace(line: str) -> None:
    print(line, file=sys.stderr, flush=True)


def answer_query(app, query: str) -> None:
    answer, completion = run_query(app.agent, query, on_line=print_trace, service=app.service)
    print("\nAnswer\n------")
    print(answer or "(the agent produced no answer)")
    if completion:
        print(f"\n{completion_line(completion)}")


def report_error(error: Exception, debug: bool) -> None:
    print(f"Error: {error}", file=sys.stderr)
    if debug:
        traceback.print_exception(error, file=sys.stderr)
    elif not isinstance(error, AppError):
        print("(run with --debug for the full traceback)", file=sys.stderr)


def interactive(app, debug: bool) -> int:
    print("Type a question about the repository. Type 'exit' or press Ctrl-D to quit.\n")
    while True:
        try:
            query = input("Query > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not query:
            continue
        if query.lower() in EXIT_WORDS:
            return 0
        try:
            answer_query(app, query)
        except KeyboardInterrupt:
            print("\nInterrupted.", file=sys.stderr)
        except AppError as error:
            report_error(error, debug)  # one failed query does not end the session
        print()


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        config = load_app_config(args.max_steps)
        service, source = open_repository(args.repo, args.index_dir)
        app = build_app(service, config, source)

        print("Repository Intelligence Agent")
        print("=============================")
        print(f"Repository: {Path(args.repo).expanduser().resolve()}")
        if args.index_dir:
            print(f"Index: {args.index_dir}")
        print(f"Symbols: {len(service.index.symbols)}")
        print(f"Model: {config.model}\n")

        if args.query:
            answer_query(app, args.query)
            return 0
        return interactive(app, args.debug)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except AppError as error:
        report_error(error, args.debug)
        return 2
    except Exception as error:  # last resort: still no raw traceback for normal users
        report_error(error, args.debug)
        return 1


if __name__ == "__main__":
    sys.exit(main())
