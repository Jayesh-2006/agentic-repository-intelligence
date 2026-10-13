"""Deterministic, citation-first answers for simple repository facts.

This path deliberately does not call an LLM. It only returns an answer when
the indexed search hit and a direct source read contain query-relevant text;
otherwise the caller should continue with the full Deep Agent.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import time

from ..service import RepoService
from ..retrieval import tokenize
from . import intent


@dataclass(frozen=True)
class SimpleAnswer:
    text: str
    completion: dict
    trace: tuple[str, ...]


@dataclass(frozen=True)
class RetrievalResult:
    """One query's direct answer and any safe pointer context for the agent."""

    answer: SimpleAnswer | None
    hints: tuple[str, ...]
    cache: "QueryRetrievalCache"


class QueryRetrievalCache:
    """Reuse ranked index results during one query without retaining stale data."""

    def __init__(self, service: RepoService):
        self._service = service
        self._index = service.index
        self._searches: dict[tuple[int, str, str | None], tuple[int, dict]] = {}
        self.search_calls = 0
        self.search_hits = 0
        self.search_ms = 0.0
        self.result_chars = 0
        self.file_reads = 0
        self.file_read_ms = 0.0

    def __getattr__(self, name):
        return getattr(self._service, name)

    def search_repository(self, query: str, k: int = 5, file_path: str | None = None) -> dict:
        self.search_calls += 1
        if self._service.index is not self._index:
            self._searches.clear()
            self._index = self._service.index
        key = (id(self._index), " ".join(query.split()).casefold(),
               file_path.replace("\\", "/").strip("/") if file_path else None)
        cached = self._searches.get(key)
        if cached is not None and k <= cached[0]:
            self.search_hits += 1
            data = dict(cached[1])
            data["query"] = query
            data["results"] = cached[1].get("results", [])[:k]
            data["count"] = len(data["results"])
        else:
            started = time.perf_counter()
            data = self._service.search_repository(query, k=k, file_path=file_path)
            self.search_ms += (time.perf_counter() - started) * 1000
            if "error" not in data:
                self._searches[key] = (k, data)
        self.result_chars += len(str(data))
        return data

    def get_file(self, *args, **kwargs) -> dict:
        started = time.perf_counter()
        result = self._service.get_file(*args, **kwargs)
        self.file_reads += 1
        self.file_read_ms += (time.perf_counter() - started) * 1000
        return result


def retrieve_query(service: RepoService, question: str) -> RetrievalResult:
    """Run retrieval once and retain only location hints if no direct fact is proven."""
    cache = QueryRetrievalCache(service)
    answer = try_simple_answer(cache, question)
    hints = () if answer is not None else tuple(gather_hint(cache, question) or ())
    return RetrievalResult(answer=answer, hints=hints, cache=cache)


_QUESTION_WORDS = {
    "what", "where", "which", "who", "is", "are", "does", "do", "can",
    "the", "a", "an", "of", "in", "on", "at", "for", "to", "from",
    "and", "or", "this", "that", "it", "with", "how",
    "according", "based", "file", "repository", "repo", "reported",
    "readme", "md",
    "required", "used", "value", "please", "tell", "me",
    # generic filler that must never be the only thing a shortcut matches on
    "use", "might", "main", "available", "different", "be", "mention", "mentioned", "many",
}
_VALUE_QUESTION_WORDS = {
    "accuracy", "score", "loss", "version", "rate", "precision", "recall",
    "f1", "metric", "count", "number", "percentage", "percent", "size",
}
_NAMED_FACT_WORDS = {"model", "dataset", "framework", "library", "package", "language"}
_NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)*(?:\s?%|\s?(?:ms|seconds?|GB|MB))?\b", re.IGNORECASE)
_LOCATION_QUESTION_RE = re.compile(r"\b(where|defined|definition|located|location)\b", re.IGNORECASE)
_QUESTION_TERM_STOPWORDS = {
    token for word in _QUESTION_WORDS for token in tokenize(word)
}
_METRIC_WORD_RE = "|".join(sorted((re.escape(word) for word in _VALUE_QUESTION_WORDS), key=len, reverse=True))
_SHARED_METRIC_PAIR_RE = re.compile(
    rf"\b([a-z][\w-]*)\s+and\s+([a-z][\w-]*)\s+({_METRIC_WORD_RE})\b",
    re.IGNORECASE,
)


def gather_hint(service: RepoService, question: str, max_candidates: int = 3) -> list[str] | None:
    """Best-effort pointers to where an answer likely lives, from the same
    indexed search the simple-retrieval shortcut itself uses - for when that
    shortcut could not (or was not allowed to) produce a supported answer.

    This is a hint for the Deep Agent's first move, never evidence: it is
    only file:line pointers, no snippet text, and it never enters
    `plan["steps"]` (the only thing `conclude()`/`build_final_response` ever
    cite from), so it cannot let an unverified location be reported as fact.
    The agent still has to call get_file/investigate_repository and pass the
    usual evidence gate before citing anything from a hinted location.
    """
    clean_question = intent.normalize_simple_question(question)
    terms = _question_terms(clean_question)
    if not terms:
        return None
    search = service.search_repository(clean_question, k=max_candidates)
    results = search.get("results", []) if "error" not in search else []
    lines = [
        f"- {result['file']}:{result.get('line_start')}-{result.get('line_end')}"
        for result in results[:max_candidates] if result.get("file")
    ]
    return lines or None


def try_simple_answer(service: RepoService, question: str) -> SimpleAnswer | None:
    """Return a concise evidence-backed answer, or None to invoke the agent."""
    if not intent.looks_simple(question):
        return None

    request_intent = intent.resolve_intent(question)
    sources = request_intent["sources"]
    restricted_file = sources[0] if len(sources) == 1 else None
    clean_question = intent.normalize_simple_question(question)
    terms = _question_terms(clean_question)
    if not terms:
        return None
    # A lone generic word is not a fact request; only a metric/named-fact word may stand alone.
    if len(terms) < 2 and not (terms & (_VALUE_QUESTION_WORDS | _NAMED_FACT_WORDS)):
        return None

    requested_facts = _shared_metric_facts(clean_question)
    if len(requested_facts) > 1:
        fact_evidence = []
        fact_traces = []
        for fact in requested_facts:
            fact_terms = _question_terms(fact)
            evidence, fact_trace = _find_supported_evidence(
                service, fact, fact_terms, fact, restricted_file
            )
            if evidence is None:
                trace = [
                    f"[Retrieval] could not establish requested fact: {fact}",
                    "[Retrieval] insufficient or ambiguous source evidence; using deep investigation",
                ]
                return None
            fact_evidence.append(evidence)
            fact_traces.extend(fact_trace[:1])
        return _supported_facts_answer(fact_evidence, fact_traces)

    evidence, trace = _find_supported_evidence(
        service, clean_question, terms, clean_question, restricted_file
    )
    if evidence is not None:
        return _supported_answer(evidence, trace, "search_repository + get_file")

    # If the user explicitly named one source, a direct read of that source is
    # still safe and useful when its chunks are missing or too coarse. This
    # never widens the search to another repository file.
    if restricted_file:
        report = service.get_file(restricted_file, max_lines=5000)
        if "error" not in report:
            answer = _answer_from_file(report, terms, clean_question)
            if answer is not None:
                trace.append(f"[Retrieval] get_file: read {restricted_file}")
                return _supported_answer(answer, trace, "get_file")

    return None


def _find_supported_evidence(
    service: RepoService,
    search_query: str,
    terms: set[str],
    answer_question: str,
    restricted_file: str | None,
) -> tuple[dict | None, list[str]]:
    """Search for one fact and accept only a sufficiently specific source read."""
    if not terms:
        return None, []

    search = service.search_repository(search_query, k=20, file_path=restricted_file)
    results = search.get("results", []) if "error" not in search else []
    trace = [
        f"[Retrieval] search_repository: {len(results)} indexed candidate(s)"
        + (f" in {restricted_file}" if restricted_file else "")
    ]

    candidates = []
    for result in results:
        if restricted_file and not _same_path(result.get("file", ""), restricted_file):
            continue
        snippet = str(result.get("snippet", ""))
        matched = terms & set(tokenize(snippet))
        if len(matched) < _minimum_match_count(terms):
            continue
        candidates.append((result, len(matched)))

    viable = []
    seen_ranges = set()
    # Read only the highest-ranked relevant windows. Continue past headings,
    # symbol names, and other lexical matches that do not contain a value.
    for result, matched_count in candidates[:8]:
        start = result.get("snippet_line_start") or result.get("line_start")
        snippet_line_count = max(1, len(str(result.get("snippet", "")).splitlines()))
        end = min(int(result.get("line_end") or start), int(start) + snippet_line_count - 1)
        key = (result.get("file"), int(start), int(end))
        if key in seen_ranges:
            continue
        seen_ranges.add(key)
        report = service.get_file(str(result["file"]), int(start), int(end))
        if "error" not in report:
            answer = _answer_from_file(report, terms, answer_question)
            if answer is not None:
                viable.append((result, matched_count, answer))

    selected = _select_viable_candidate(viable, restricted_file)
    if selected is not None:
        return selected, trace

    trace.append("[Retrieval] insufficient or ambiguous source evidence; using deep investigation")
    return None, trace


def _shared_metric_facts(question: str) -> list[str]:
    """Expand shorthand such as “strict and soft accuracy” into two facts."""
    match = _SHARED_METRIC_PAIR_RE.search(question)
    if not match:
        return []
    first, second, metric = match.groups()
    facts = [f"{first} {metric}", f"{second} {metric}"]
    requested_terms = _question_terms(question) - _NAMED_FACT_WORDS
    fact_terms = set().union(*(_question_terms(fact) for fact in facts))
    # Do not answer only the recognized pair if the query also asks for
    # another value/facet that this decomposition did not capture.
    return facts if requested_terms <= fact_terms else []


def _question_terms(question: str) -> set[str]:
    return {
        token for token in tokenize(question)
        if token not in _QUESTION_TERM_STOPWORDS and len(token) > 1
    }


def _minimum_match_count(terms: set[str]) -> int:
    """How many question terms one source line must contain. It scales with
    the question: a line that holds only half of a 4-term question ("complete
    flow input output" -> just "input", "output") states a different fact.
    One term may stand alone only when try_simple_answer already accepted it
    (a metric or named-fact word); two terms must both match."""
    if len(terms) <= 1:
        return 1
    return max(2, -(-len(terms) * 3 // 5))  # ceil(60%), at least 2


def _select_viable_candidate(viable: list[tuple[dict, int, dict]], restricted_file: str | None) -> dict | None:
    if not viable:
        return None
    viable.sort(key=lambda item: (-int(item[0].get("score", 0) * 1000), -item[1], item[0].get("file", "")))
    best_result, _, best_evidence = viable[0]
    if restricted_file:
        return best_evidence

    best_score = float(best_result.get("score", 0))
    best_path = best_result.get("file", "")
    for other, _, _ in viable[1:]:
        if _same_path(other.get("file", ""), best_path):
            continue
        other_score = float(other.get("score", 0))
        if best_score <= 0 or other_score >= best_score * 0.82:
            return None
        break
    return best_evidence


def _answer_from_file(report: dict, terms: set[str], question: str = "") -> dict | None:
    content = report.get("content")
    if not isinstance(content, str) or not content.strip():
        return None
    lines = content.splitlines()
    first = int(report.get("line_start") or 1)
    scores = [len(terms & set(tokenize(line))) for line in lines]
    if not scores or max(scores) < _minimum_match_count(terms):
        return None

    # Keep the matching source lines plus immediate context. Context helps
    # with labels/headings whose value is on the adjacent line, while retaining
    # the exact line range for citation.
    best_score = max(scores)
    best_indices = [index for index, score in enumerate(scores) if score == best_score]
    relevant = set(best_indices)
    needs_number = (
        not _LOCATION_QUESTION_RE.search(question)
        and bool(terms & _VALUE_QUESTION_WORDS)
    )
    for index in best_indices:
        line = lines[index]
        has_key_value = bool(re.search(r"[:=]\s*\S", line))
        has_inline_value = _is_labeled_number(line) or (not needs_number and has_key_value)
        if not has_inline_value:
            relevant.update(range(max(0, index - 1), min(len(lines), index + 2)))
    selected = [(first + i, lines[i]) for i in sorted(relevant) if lines[i].strip()]
    if not selected:
        return None
    needs_number = not _LOCATION_QUESTION_RE.search(question) and bool(terms & _VALUE_QUESTION_WORDS)
    if needs_number:
        matching_indices = [index for index, score in enumerate(scores) if score]
        value_terms = terms & _VALUE_QUESTION_WORDS
        source_terms = set(tokenize(str(report.get("file", ""))))
        required_label_terms = terms - _QUESTION_TERM_STOPWORDS - _NAMED_FACT_WORDS - source_terms
        has_matching_value = any(
            _is_labeled_number(lines[index])
            and (not value_terms or value_terms & set(tokenize(lines[index])))
            and required_label_terms <= set(tokenize(lines[index]))
            for index in matching_indices
        )
        has_adjacent_value = any(
            abs(value_index - match_index) == 1
            and _is_standalone_number(lines[value_index])
            and (not value_terms or value_terms & set(tokenize(lines[match_index])))
            and required_label_terms <= set(tokenize(lines[match_index]))
            for match_index in matching_indices
            for value_index in range(max(0, match_index - 1), min(len(lines), match_index + 2))
            if value_index != match_index
        )
        if not (has_matching_value or has_adjacent_value):
            return None
    if terms & _NAMED_FACT_WORDS:
        meaningful = set().union(*(set(tokenize(line)) for _, line in selected)) - terms - _QUESTION_WORDS
        if not meaningful and not any(":" in line or "=" in line for _, line in selected):
            return None
    return {
        "file": report.get("file", ""),
        "line_start": selected[0][0],
        "line_end": selected[-1][0],
        "content": "\n".join(line for _, line in selected),
        "retrieval_tool": "search_repository",
        "read_tool": "get_file",
    }


def _supported_answer(evidence: dict, trace: list[str], retrieval_tool: str) -> SimpleAnswer:
    evidence = dict(evidence, retrieval_tool=retrieval_tool)
    file_path = evidence["file"]
    start, end = evidence["line_start"], evidence["line_end"]
    excerpt = evidence["content"].strip()
    citation = f"{file_path}, line {start}" if start == end else f"{file_path}, lines {start}–{end}"
    text = f"The repository reports:\n> {excerpt.replace(chr(10), chr(10) + '> ')}\n\nSource: {citation}."
    trace = [*trace, f"[Evidence] simple_retrieval: {citation} -> supported", "[Completion] SUPPORTED"]
    return SimpleAnswer(
        text=text,
        completion={
            "support": "SUPPORTED",
            "criteria_supported": 1,
            "criteria_total": 1,
            "simple_retrieval": True,
            "evidence": evidence,
            "retrieval_tool": retrieval_tool,
        },
        trace=tuple(trace),
    )


def _supported_facts_answer(facts: list[dict], retrieval_trace: list[str]) -> SimpleAnswer:
    """Format independently supported facts, retaining per-fact citations."""
    lines = ["The repository reports:"]
    trace = list(retrieval_trace)
    for evidence in facts:
        file_path = evidence["file"]
        start, end = evidence["line_start"], evidence["line_end"]
        citation = f"{file_path}, line {start}" if start == end else f"{file_path}, lines {start}–{end}"
        excerpt = evidence["content"].strip().replace("\n", " / ")
        lines.append(f"- {excerpt} ({citation})")
        trace.append(f"[Evidence] simple_retrieval: {citation} -> supported")
    trace.append("[Completion] SUPPORTED")
    first = facts[0]
    return SimpleAnswer(
        text="\n".join(lines),
        completion={
            "support": "SUPPORTED",
            "criteria_supported": len(facts),
            "criteria_total": len(facts),
            "simple_retrieval": True,
            "evidence": first,
            "fact_evidence": facts,
            "retrieval_tool": "search_repository + get_file",
        },
        trace=tuple(trace),
    )


def _same_path(left: str, right: str) -> bool:
    return left.replace("\\", "/").casefold() == right.replace("\\", "/").casefold()


def _is_standalone_number(line: str) -> bool:
    match = _NUMBER_RE.search(line)
    if not match or not _is_labeled_number(line):
        return False
    value = _NUMBER_RE.sub("", line)
    return not value.strip(" |:*#`\t,.$()-")


def _is_labeled_number(line: str) -> bool:
    match = _NUMBER_RE.search(line)
    if not match:
        return False
    number = match.group(0)
    return any(mark in line for mark in ("%", ":", "=", "|")) or "." in number or "," in number
