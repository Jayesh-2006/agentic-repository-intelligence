"""
Local lexical retrieval over indexed repository chunks.

`Retriever` is the small interface the rest of the system depends on. A
custom retriever implements `search()` and `rebuild(index)` so RepoService
can keep it bound to the current index after refresh.

`LexicalRetriever` is a compact BM25 scorer. Each chunk is turned into a bag
of tokens where the text counts once, the symbol's qualified name counts 3x
and the file path counts 2x, so a query that names a symbol or a file finds
it even if the body never repeats the name. Small bonuses reward exact symbol
name matches.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from .models import RepositoryIndex

_WORD = re.compile(r"[A-Za-z0-9_]+")
_CAMEL = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+")
_STOPWORDS = {"the", "a", "an", "of", "to", "in", "and", "or", "for", "is", "are", "how", "does", "do",
              "what", "where", "which", "with", "that", "this", "it", "on", "by"}
_SUFFIXES = ("ations", "ation", "ators", "ator", "ings", "ing", "ates", "ate", "ed", "es", "s")

NAME_WEIGHT = 3
PATH_WEIGHT = 2
K1, B = 1.5, 0.75


def _stem(word: str) -> str:
    """Very small suffix stripper so `validation`, `validate`, `validator`
    all become `valid`. Applied to documents and queries alike."""
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
    return word


def tokenize(text: str) -> list[str]:
    """Identifiers are kept whole and also split on snake_case / camelCase:
    `validateToken` -> validatetoken, validate, token."""
    tokens: list[str] = []
    for word in _WORD.findall(text):
        whole = word.lower()
        parts = [p.lower() for piece in word.split("_") for p in _CAMEL.findall(piece)]
        tokens.append(_stem(whole))
        if len(parts) > 1 or parts != [whole]:
            tokens.extend(_stem(p) for p in parts)
    return tokens


def query_tokens(query: str) -> list[str]:
    return [t for t in tokenize(query) if t not in _STOPWORDS]


@dataclass
class Hit:
    chunk_id: str
    score: float


class Retriever:
    """Retriever contract: search plus rebuilding against a replacement index."""

    def search(self, query: str, top_k: int = 5, file_path: str | None = None) -> list[Hit]:
        raise NotImplementedError

    def rebuild(self, index: RepositoryIndex) -> "Retriever":
        """Return a new retriever bound to ``index``; refresh relies on this."""
        raise NotImplementedError("custom retrievers must implement rebuild(index)")


class LexicalRetriever(Retriever):
    def __init__(self, index: RepositoryIndex) -> None:
        self._index = index
        self._tf: dict[str, Counter] = {}
        self._length: dict[str, int] = {}
        self._df: Counter = Counter()

        for chunk_id, chunk in index.chunks.items():
            tf: Counter = Counter(tokenize(chunk.text))
            if chunk.kind != "text":
                name = chunk.symbol_id or chunk_id.removesuffix("#module")
                for token in tokenize(name):
                    tf[token] += NAME_WEIGHT
            for token in tokenize(chunk.file.removesuffix(".py")):
                tf[token] += PATH_WEIGHT
            self._tf[chunk_id] = tf
            self._length[chunk_id] = sum(tf.values())
            self._df.update(tf.keys())

        n = max(len(self._tf), 1)
        self._avg_len = sum(self._length.values()) / n
        self._n = n

    def rebuild(self, index: RepositoryIndex) -> "LexicalRetriever":
        return LexicalRetriever(index)

    def search(self, query: str, top_k: int = 5, file_path: str | None = None) -> list[Hit]:
        terms = set(query_tokens(query))
        if not terms:
            return []
        cleaned = query.strip().lower()

        hits: list[Hit] = []
        for chunk_id, tf in self._tf.items():
            if file_path is not None:
                chunk = self._index.chunks.get(chunk_id)
                if chunk is None or chunk.file.replace("\\", "/").casefold() != file_path.replace("\\", "/").casefold():
                    continue
            score = 0.0
            for term in terms:
                freq = tf.get(term, 0)
                if freq:
                    idf = math.log(1 + (self._n - self._df[term] + 0.5) / (self._df[term] + 0.5))
                    norm = freq + K1 * (1 - B + B * self._length[chunk_id] / self._avg_len)
                    score += idf * freq * (K1 + 1) / norm
            if score > 0:
                score += self._name_bonus(chunk_id, cleaned)
                hits.append(Hit(chunk_id, score))

        hits.sort(key=lambda h: (-h.score, h.chunk_id))
        return hits[:top_k]

    def _name_bonus(self, chunk_id: str, cleaned_query: str) -> float:
        symbol_id = self._index.chunks[chunk_id].symbol_id
        if symbol_id is None:
            return 0.0
        full = symbol_id.lower()
        if cleaned_query == full or full.endswith("." + cleaned_query):
            return 20.0
        if cleaned_query == self._index.symbols[symbol_id].name.lower():
            return 10.0
        return 0.0
