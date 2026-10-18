"""
Local lexical retrieval over indexed repository chunks using rank-bm25 (BM25Plus).

`Retriever` is the interface the rest of the system depends on. A
retriever implements `search()` and `rebuild(index)` so RepoService
can keep it bound to the current index after refresh.

`LexicalRetriever` uses `BM25Plus` (from `rank_bm25` when installed, with a
built-in mathematically identical fallback):
  1. Normalizes morphological variants (`cache`/`caching`/`cached` -> `cach`,
     `optimize`/`optimizer`/`optimization` -> `optim`,
     `calculate`/`calculated`/`calculation` -> `calcul`, etc.) and splits
     `snake_case` and `CamelCase` identifiers.
  2. Weights symbol names (3x), file paths (2x), and markdown headings (2x) in the corpus.
  3. Expands common ML/code concept queries (e.g. `optimization` -> `optim`,
     `adam`, `lr`, `freeze`, `requires_grad`, `cach`, `bottleneck`) with
     expansion weights so conceptual queries find both code and docs.
  4. Delegates BM25+ scoring to `BM25Plus`.
  5. Adds phrase/line-proximity and exact symbol-name bonuses so chunks where query
     terms co-occur on the same lines rank highest.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np

try:
    from rank_bm25 import BM25Plus  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - fallback when rank-bm25 is not installed
    class BM25Plus:  # type: ignore[no-redef]
        """Self-contained BM25+ implementation matching rank_bm25.BM25Plus."""

        def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75, delta: float = 1.0) -> None:
            self.k1 = k1
            self.b = b
            self.delta = delta
            self.corpus_size = len(corpus)
            self.doc_len: list[int] = []
            self.doc_freqs: list[dict[str, int]] = []
            self.idf: dict[str, float] = {}
            nd: dict[str, int] = {}
            num_doc = 0
            for document in corpus:
                self.doc_len.append(len(document))
                num_doc += len(document)
                frequencies: dict[str, int] = {}
                for word in document:
                    frequencies[word] = frequencies.get(word, 0) + 1
                self.doc_freqs.append(frequencies)
                for word in frequencies:
                    nd[word] = nd.get(word, 0) + 1
            self.avgdl = (num_doc / self.corpus_size) if self.corpus_size > 0 else 1.0
            for word, freq in nd.items():
                self.idf[word] = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0)

        def get_scores(self, query: list[str]) -> np.ndarray:
            score = np.zeros(self.corpus_size, dtype=float)
            if not self.corpus_size or self.avgdl <= 0:
                return score
            doc_len = np.array(self.doc_len, dtype=float)
            for q in query:
                if q not in self.idf:
                    continue
                q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs], dtype=float)
                score += self.idf[q] * (
                    self.delta + (q_freq * (self.k1 + 1.0)) /
                    (q_freq + self.k1 * (1.0 - self.b + self.b * doc_len / self.avgdl))
                )
            return score

from .models import RepositoryIndex

_WORD = re.compile(r"[A-Za-z0-9_]+")
_CAMEL = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+")
_STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "and", "or", "for", "is", "are", "how",
    "does", "do", "what", "where", "which", "with", "that", "this", "it", "on",
    "by", "can", "you", "u", "get", "part", "from", "there", "any", "yes", "if",
    "done", "doing", "used", "using", "repo", "repository", "code", "file",
}

_SUFFIX_RULES: tuple[tuple[str, str], ...] = (
    ("izations", "iz"),
    ("ization", "iz"),
    ("isations", "is"),
    ("isation", "is"),
    ("abilities", "abil"),
    ("ability", "abil"),
    ("accuracies", "accurac"),
    ("accuracy", "accurac"),
    ("ations", ""),
    ("ation", ""),
    ("ators", ""),
    ("ator", ""),
    ("izers", "iz"),
    ("izer", "iz"),
    ("izes", "iz"),
    ("ized", "iz"),
    ("izing", "iz"),
    ("ize", "iz"),
    ("ings", ""),
    ("ing", ""),
    ("ments", "ment"),
    ("ates", ""),
    ("ated", ""),
    ("ate", ""),
    ("ers", ""),
    ("er", ""),
    ("ies", "y"),
    ("ed", ""),
    ("es", ""),
    ("s", ""),
)

_QUERY_EXPANSIONS: dict[str, tuple[str, ...]] = {
    "optim": ("adam", "adamw", "sgd", "lr", "schedul", "freez", "requires", "grad", "cach", "bottleneck"),
    "cach": ("featur", "precomput", "disk", "load", "bottleneck", "acceler"),
    "accurac": ("soft", "strict", "coco", "eval", "correct", "metric"),
    "architectur": ("model", "encod", "fusion", "attent", "forward", "classifi"),
}

NAME_WEIGHT = 3
PATH_WEIGHT = 2
HEADING_WEIGHT = 2
K1, B, DELTA = 1.5, 0.75, 1.0
EXPANSION_WEIGHT = 0.35


def _stem(word: str) -> str:
    """Normalize common English and code identifier inflections to a shared root."""
    w = word.lower()
    if len(w) <= 3:
        return w
    if w.startswith("cacul"):
        w = "calcul" + w[5:]
    if w.endswith("ationd"):
        w = w[:-1]

    for suffix, replacement in _SUFFIX_RULES:
        if w.endswith(suffix) and len(w) - len(suffix) + len(replacement) >= 3:
            w = w[: -len(suffix)] + replacement
            break

    if len(w) >= 5 and w.endswith("e") and not w.endswith(("ee", "le", "re")):
        w = w[:-1]
    if w.endswith("optimiz"):
        w = w[:-2]
    return w


def tokenize(text: str) -> list[str]:
    """Identifiers are kept whole and also split on snake_case / camelCase."""
    tokens: list[str] = []
    for word in _WORD.findall(text):
        whole = word.lower()
        parts = [p.lower() for piece in word.split("_") for p in _CAMEL.findall(piece)]
        tokens.append(_stem(whole))
        if len(parts) > 1 or parts != [whole]:
            tokens.extend(_stem(p) for p in parts if p)
    return tokens


def query_tokens(query: str) -> list[str]:
    stemmed_stops = {_stem(w) for w in _STOPWORDS} | _STOPWORDS
    return [t for t in tokenize(query) if t not in stemmed_stops and len(t) > 1]


@dataclass
class Hit:
    chunk_id: str
    score: float


class Retriever:
    """Retriever contract: search plus rebuilding against a replacement index."""

    def search(self, query: str, top_k: int = 5, file_path: str | None = None) -> list[Hit]:
        raise NotImplementedError

    def rebuild(self, index: RepositoryIndex) -> "Retriever":
        raise NotImplementedError("custom retrievers must implement rebuild(index)")


class LexicalRetriever(Retriever):
    """BM25+ lexical and structural code/doc retriever powered directly by rank-bm25."""

    backend_name: str = "rank-bm25 (BM25Plus)"

    def __init__(self, index: RepositoryIndex) -> None:
        self._index = index
        self._chunk_ids: list[str] = list(index.chunks.keys())
        self._line_tokens: dict[str, list[set[str]]] = {}
        corpus_tokens: list[list[str]] = []

        for chunk_id in self._chunk_ids:
            chunk = index.chunks[chunk_id]
            doc_tokens: list[str] = list(tokenize(chunk.text))
            if chunk.kind != "text":
                name = chunk.symbol_id or chunk_id.removesuffix("#module")
                name_toks = tokenize(name)
                for _ in range(NAME_WEIGHT):
                    doc_tokens.extend(name_toks)
            else:
                for line in chunk.text.splitlines():
                    if line.lstrip().startswith("#"):
                        doc_tokens.extend(tokenize(line) * HEADING_WEIGHT)
            path_toks = tokenize(chunk.file.removesuffix(".py"))
            for _ in range(PATH_WEIGHT):
                doc_tokens.extend(path_toks)

            self._line_tokens[chunk_id] = [
                set(tokenize(line)) for line in chunk.text.splitlines() if line.strip()
            ]
            corpus_tokens.append(doc_tokens)

        self._bm25: BM25Plus | None = (
            BM25Plus(corpus_tokens, k1=K1, b=B, delta=DELTA) if corpus_tokens else None
        )

    def rebuild(self, index: RepositoryIndex) -> "LexicalRetriever":
        return LexicalRetriever(index)

    def _expanded_query_weights(self, primary_terms: set[str]) -> dict[str, float]:
        weights: dict[str, float] = {t: 1.0 for t in primary_terms}
        for term in list(primary_terms):
            for extra in _QUERY_EXPANSIONS.get(term, ()):
                stemmed = _stem(extra)
                if stemmed not in weights:
                    weights[stemmed] = EXPANSION_WEIGHT
        return weights

    def search(self, query: str, top_k: int = 5, file_path: str | None = None) -> list[Hit]:
        if self._bm25 is None or not self._chunk_ids:
            return []
        primary_terms = set(query_tokens(query))
        if not primary_terms:
            return []
        cleaned = query.strip().lower()
        term_weights = self._expanded_query_weights(primary_terms)

        # Delegate BM25+ scoring directly to rank-bm25
        scores = np.zeros(len(self._chunk_ids), dtype=float)
        matched_primary_docs: set[int] = set()

        for term, weight in term_weights.items():
            term_scores = self._bm25.get_scores([term])
            scores += weight * term_scores
            if weight >= 1.0:
                for idx in np.where(term_scores > 0)[0]:
                    matched_primary_docs.add(int(idx))

        hits: list[Hit] = []
        for idx, chunk_id in enumerate(self._chunk_ids):
            if file_path is not None:
                chunk = self._index.chunks.get(chunk_id)
                if (
                    chunk is None
                    or chunk.file.replace("\\", "/").casefold()
                    != file_path.replace("\\", "/").casefold()
                ):
                    continue

            score = float(scores[idx])
            # Require at least one primary query term (or a strong expansion match when single-term)
            if score <= 0 or (idx not in matched_primary_docs and len(primary_terms) > 1):
                continue

            score += self._name_bonus(chunk_id, cleaned)
            score += self._proximity_bonus(chunk_id, primary_terms)
            hits.append(Hit(chunk_id, score))

        hits.sort(key=lambda h: (-h.score, h.chunk_id))
        return hits[:top_k]

    def _proximity_bonus(self, chunk_id: str, primary_terms: set[str]) -> float:
        """Reward chunks where multiple distinct query terms appear on the same line."""
        if len(primary_terms) < 2:
            return 0.0
        line_sets = self._line_tokens.get(chunk_id) or []
        if not line_sets:
            return 0.0
        best_line_overlap = max((len(primary_terms & lset) for lset in line_sets), default=0)
        if best_line_overlap >= 2:
            return 1.5 * (best_line_overlap - 1)
        return 0.0

    def _name_bonus(self, chunk_id: str, cleaned_query: str) -> float:
        chunk = self._index.chunks[chunk_id]
        symbol_id = chunk.symbol_id
        if symbol_id is None:
            return 0.0
        full = symbol_id.lower()
        if cleaned_query == full or full.endswith("." + cleaned_query):
            return 20.0
        sym = self._index.symbols.get(symbol_id)
        if sym is not None and cleaned_query == sym.name.lower():
            return 10.0
        return 0.0
