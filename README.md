# Agentic Repository Intelligence

ARI indexes a software repository and answers questions using repository-grounded evidence. Narrow factual lookups use deterministic retrieval when the source supports an answer. Broader questions use a bounded LangGraph agent that can inspect files, search symbols and relationships, and delegate investigation. Requested changes use a separate write-capable path with repository safeguards.

## Requirements

Python plus the packages listed in `requirements.txt`. Deep Agent requests also require `GROQ_API_KEY`; set `GROQ_MODEL` to select a compatible tool-calling model.

## Index a repository

```text
python scripts/build_index.py <repository-path> <index-path>
```

The index stores Python symbols and relationships, ranked code chunks, supported text-file chunks, and a repository file listing. ARI can also index a repository in memory when no saved index is supplied.

## Ask a question

```text
python main.py --repo <repository-path>
python main.py --repo <repository-path> --index-dir <index-path> --query "Where is RepoService.refresh defined?"
```

Interactive mode accepts one question per prompt. Progress is written to stderr and the answer to stdout. Each query reports available runtime measurements, including per-stage model usage, deterministic follow-ups, repeated-action blocks, and no-progress stops.

## Request routing

- Narrow factual questions first try lexical retrieval and a direct source read. A direct answer is returned only when the source contains sufficiently specific evidence.
- If direct retrieval cannot establish the answer, ARI passes limited file-and-line hints to the Deep Agent. Hints are never treated as evidence.
- When the Deep Agent receives one exact file or symbol match for a source-content question, it may read that source directly before asking for another decision. The normal evidence grader still determines whether the answer is supported.
- Multi-file, architectural, tracing, or broad questions use the agentic investigation path.
- Explicit repository changes use the Coder and the configured validation safeguards.

## Repository tools

The read-only tool set supports content search, file discovery, symbol lookup, caller/callee and importer tracing, dependency inspection, AST structure, source reads, and focused symbol reads. `write_file` is available only to the Coder; command execution is available only to the Debugger or explicit verification path. All paths are constrained to the repository root and observations are bounded before being sent to a model.

## Evidence and safety

Retrieved text, repository files, and tool output are untrusted data, not instructions. Evidence records keep source paths, line ranges, result status, and provenance. Truncated reads cannot satisfy a complete-file requirement. Final responses pass through the evidence completion check; unsupported claims remain unsupported. Model-generated summaries are distinguished from raw repository observations.

Write and execute capabilities are determined from the user's request, not from model-generated tool arguments. Repository changes are protected by a filesystem checkpoint that preserves the existing working tree, including prior edits and untracked files. The index is refreshed after changes so later retrieval reflects the files on disk.

## Validation configuration

`VERIFICATION_COMMAND` is optional. ARI runs a configured repository-wide command only when verification is explicitly requested. If no command is configured, ARI reports that repository-wide verification is unavailable; it does not assume a universal command. Code changes retain a focused syntax check of changed Python files.

Do not add or rely on project test files. Use static inspection, targeted syntax/runtime checks, and manual queries against a real repository when dependencies and credentials are available.
