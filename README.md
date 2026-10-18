# Agentic Repository Intelligence (ARI)

**Understand a codebase. Trace how it works. Make carefully guarded
changes.**

Agentic Repository Intelligence (ARI) is a command-line AI assistant for
exploring software repositories. It combines Python AST indexing, a
static code relationship graph, BM25+ lexical search, and a single
LangGraph agent that investigates code, gathers source evidence, and can
perform guarded changes when permitted.

ARI is designed to make repository investigations more traceable:
answers are grounded in files read during the run, citations are checked
against recorded source ranges, and write or command operations are
subject to capability and safety checks.

## Highlights

-   **Repository-aware Q&A** --- ask about architecture, modules,
    functions, configuration, and execution flow.
-   **Code navigation** --- locate symbols, inspect implementations, and
    explore callers, callees, imports, and dependencies.
-   **Lexical retrieval** --- search indexed source and documentation
    using BM25+ with identifier-aware processing and a fallback
    retriever.
-   **Evidence-backed answers** --- track source provenance and validate
    file/line citations against inspected ranges.
-   **Single-agent workflow** --- one bounded LangGraph agent plans and
    executes repository investigations using tools.
-   **Guarded code changes** --- capability checks, command validation,
    checkpoints, verification, and rollback support for permitted
    operations.
-   **CLI-first experience** --- interactive terminal session, one-shot
    queries, readable progress traces, and evidence summaries.
-   **Token-conscious execution** --- bounded context, compacted older
    observations, and capability-scoped tool binding help limit
    unnecessary model input.

## How It Works

``` mermaid
flowchart TD
    A["Question or Task"] --> B["Index and Repository Service"]
    B --> C["Single LangGraph Agent"]
    C --> D["Plan and Choose Tools"]
    D --> E["Search / Read / Inspect Code"]
    D --> F["Guarded Write / Verify / Execute"]
    E --> G["Record Evidence and Progress"]
    F --> G
    G --> H["Assess Evidence and Citations"]
    H --> I{"Continue within step budget?"}
    I -->|Yes| D
    I -->|No| J["Answer with Supporting Evidence"]
```

ARI uses an agent-first workflow. It selects repository tools based on
the question, records tool results and evidence provenance, and produces
a conclusion within configured limits. For general purpose or overview
questions, it can prioritize the root `README.md` when one is available.

## Example Questions

**Understand the project** - "What is the purpose of this repository?" -
"Explain the flow from input to output." - "What are the main modules
and how do they interact?"

**Navigate code** - "Where is `RepoService.refresh` defined?" - "Show
the implementation of `collect_module`." - "Who calls
`index_repository`?" - "Which modules import `agentrepo.filesystem`?" -
"What are the dependencies of `agentrepo.service`?"

**Investigate a feature** - "Trace how `run_query`, `LazyDeepAgent`, and
`build_deep_agent` interact." - "Where is repository indexing
initialized?" - "How are source citations validated?"

**Request a guarded change** - "Add input validation to `pkg/core.py`
and run the configured verification."

Results depend on the repository contents, available evidence,
configured tools, and provider availability.

## Architecture and Main Components

  ------------------------------------------------------------------------------
  Component                                  Role
  ------------------------------------------ -----------------------------------
  `src/agentrepo/collector.py`, `indexer.py` Collect repository files and build
                                             symbol and text-chunk indexes.

  `src/agentrepo/resolver.py`, `graph.py`    Resolve and query static code
                                             relationships, including imports,
                                             inheritance, containment, and
                                             calls.

  `src/agentrepo/retrieval.py`               BM25+ lexical search, query
                                             processing, and fallback retrieval.

  `src/agentrepo/service.py`                 Unified repository service for
                                             search, symbol/graph queries, file
                                             operations, checkpoints, diffs,
                                             rollback, and command execution.

  `src/agentrepo/tools.py`                   Defines tools available to the
                                             agent.

  `src/agentrepo/agent/deep_agent.py`        Single-agent LangGraph
                                             orchestration, tool decisions,
                                             progress guards, evidence grading,
                                             and conclusion handling.

  `src/agentrepo/agent/plan.py`, `intent.py` Task planning and intent/capability
                                             classification.

  `src/agentrepo/agent/evidence.py`          Evidence provenance, acceptance
                                             criteria, citation/source-range
                                             checks, and deterministic fallback
                                             support.

  `src/agentrepo/agent/context_budget.py`,   Context budgeting and compaction of
  `utils.py`                                 older observations.

  `src/agentrepo/agent/safety.py`,           Capability enforcement, command
  `verification.py`                          safety, checkpoints, verification,
                                             and rollback.

  `src/agentrepo/agent/app.py`, `main.py`    CLI entry point, interactive
                                             session, and terminal output.
  ------------------------------------------------------------------------------

## Requirements

-   Python 3.10+
-   A Groq API key for live model-backed queries
-   Dependencies listed in `requirements.txt`
-   Git is optional and used for repository status metadata when
    available

## Installation

### 1. Clone the repository

``` bash
git clone https://github.com/Jayesh-2006/agentic-repository-intelligence.git
cd agentic-repository-intelligence
```

### 2. Create a virtual environment

**Windows PowerShell**

``` powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS**

``` bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

``` bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Configure credentials

Copy `.env.example` to `.env` and set your Groq API key:

``` dotenv
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-120b
GROQ_TEMPERATURE=0
```

## Usage

### Ask one question

``` bash
python main.py --repo /path/to/repository --query "What is the purpose of this project?"
```

### Start an interactive session

``` bash
python main.py --repo /path/to/repository
```

At the `Query >` prompt, enter questions about the repository. Use
`help` for examples, `status` to view index and model information,
`refresh` to re-index after external changes, and `exit` to quit.

### Reuse a saved index

Build an index outside the target repository:

``` bash
python scripts/build_index.py /path/to/repository /path/to/ari-index
```

Then query with it:

``` bash
python main.py --repo /path/to/repository --index-dir /path/to/ari-index --query "Who calls index_repository?"
```

### Useful flags

-   `--repo` --- repository directory to inspect.
-   `--index-dir` --- optional directory containing a saved index.
-   `--query` --- run one query and exit.
-   `--max-steps` --- override the maximum agent steps for a run.
-   `--plain` --- disable Rich terminal formatting.
-   `--debug` --- print full tracebacks when errors occur.

## Configuration

ARI reads settings from `.env` and the process environment. The exact
defaults and accepted ranges are defined in the project's configuration
code. Common settings include:

  -----------------------------------------------------------------------
  Variable                            Purpose
  ----------------------------------- -----------------------------------
  `GROQ_API_KEY`                      Groq API key required for live
                                      model calls.

  `GROQ_MODEL`                        Model identifier used by the Groq
                                      client.

  `GROQ_TEMPERATURE`                  Model sampling temperature.

  `MAX_AGENT_STEPS`                   Maximum tool-using agent turns.

  `MAX_PLAN_STEPS`                    Maximum planning steps.

  `MAX_OBSERVATION_CHARS`             Maximum size of an individual tool
                                      observation.

  `MAX_REPAIR_ATTEMPTS`               Maximum repair attempts for a
                                      guarded modification workflow.

  `VERIFICATION_COMMAND`              Optional repository verification
                                      command.

  `COMMAND_TIMEOUT_SECONDS`           Command execution timeout.

  `ENABLE_CHECKPOINTS`                Enables checkpoints before
                                      protected operations.

  `ROLLBACK_ON_REPAIR_LIMIT`          Controls rollback when repair
                                      attempts are exhausted.

  `MODEL_CONTEXT_TOKEN_BUDGET`        Estimated per-request context
                                      budget; not a provider quota.
  -----------------------------------------------------------------------

See `.env.example` and the configuration module for current defaults and
validation rules.

## Safety

ARI can inspect repositories and, when granted the required
capabilities, perform guarded write or command operations. Relevant
controls include capability-scoped tool availability, command-policy
checks, filesystem checkpoints, scoped verification, and rollback
behavior.

These are application-level safeguards, not an operating-system sandbox.
Review proposed changes and verification results. Use version control
and backups, and avoid running the agent against sensitive repositories
unless you understand the permissions and operations enabled for that
run.

## Token Budget and Provider Limits

ARI includes context budgeting, older-observation compaction,
capability-scoped tool binding, and bounded retry behavior intended to
reduce unnecessary model input. General repository-purpose questions can
use README-first guidance to avoid broad searches when the README
provides the answer.

These measures **do not eliminate provider rate limits**. Groq may
reject calls because of token-per-minute, token-per-day,
request-frequency, or other account limits. A failure can occur during
planning, a tool decision, evidence grading, wrap-up, or conclusion
validation.

When a model call fails, ARI may preserve recorded evidence or produce a
deterministic fallback, depending on the lifecycle stage. The resulting
answer may still be incomplete or unverified. Read the trace and inspect
the actual answer; an evidence status alone does not guarantee that
every finalization stage succeeded.

## Troubleshooting

**`GROQ_API_KEY` is missing** - Copy `.env.example` to `.env` and add
your key. - Ensure the application can load the `.env` file or that the
key is set in the process environment.

**HTTP 429 / rate limit** - Wait for the provider's retry window if one
is supplied. - Reduce `MODEL_CONTEXT_TOKEN_BUDGET`,
`MAX_OBSERVATION_CHARS`, or `MAX_AGENT_STEPS`. - Ask a narrower question
focused on a file, symbol, or specific part of the architecture. - Check
the limits and usage shown in the Groq console.

Lower budgets can reduce request size but may also reduce the context
available for reasoning. They cannot guarantee that a request will
succeed.

**HTTP 413 / request too large** - Reduce the context budget and
observation size. - Split broad questions into smaller investigations.

**Incomplete answer after a model error** - Inspect the trace to
identify which lifecycle stage failed. - Check whether the answer
actually explains the requested topic and whether its citations support
the claims. - Treat incomplete or unverified output as a signal to retry
with a narrower question or inspect the cited source files directly.

**Repository path or index errors** - Confirm `--repo` points to an
existing directory. - Store generated index files outside the target
repository. - Use quotes around paths that contain spaces.

**Import or dependency errors**

``` bash
python -m pip install -r requirements.txt
```

Activate the intended virtual environment before running the command.

## Tests

Run the automated test suite from the repository root:

``` bash
python -m unittest discover -s tests -v
```

Tests cover query answering, agent lifecycle behavior, evidence and
source-scope checks, context budgeting, command safety, checkpoints and
rollback, and concurrency/isolation. Passing tests validate covered
behaviors; they do not guarantee live provider availability or answer
quality for every repository.

## Project Layout

``` text
agentic-repository-intelligence/
├── main.py
├── requirements.txt
├── README.md
├── .env.example
├── .gitignore
├── docs/
├── scripts/
├── tests/
└── src/
    └── agentrepo/
        ├── collector.py
        ├── filesystem.py
        ├── graph.py
        ├── indexer.py
        ├── models.py
        ├── module_naming.py
        ├── persistence.py
        ├── resolver.py
        ├── retrieval.py
        ├── service.py
        ├── tools.py
        └── agent/
            ├── app.py
            ├── config.py
            ├── context_budget.py
            ├── deep_agent.py
            ├── deep_state.py
            ├── evidence.py
            ├── execution.py
            ├── followups.py
            ├── intent.py
            ├── model.py
            ├── observability.py
            ├── plan.py
            ├── prompts.py
            ├── safety.py
            ├── state.py
            ├── tools.py
            ├── utils.py
            └── verification.py
```

