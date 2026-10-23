"""BM25Plus lexical retrieval over indexed repository chunks."""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Plus

from .models import RepositoryIndex

_WORD = re.compile(r"[A-Za-z0-9_]+")
_CAMEL = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+")
_STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "and", "or", "for", "is", "are", "was",
    "were", "be", "been", "being", "how", "does", "do", "did", "what", "where",
    "when", "why", "who", "whom", "which", "with", "that", "this", "these",
    "those", "it", "its", "on", "by", "at", "as", "into", "can", "could",
    "should", "would", "you", "u", "get", "part", "from", "there", "any", "yes",
    "if", "done", "doing", "used", "using", "repo", "repository", "code", "file",
}

K1: float = 1.5
B: float = 0.75
DELTA: float = 0.0


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for word in _WORD.findall(text):
        whole = word.lower()
        tokens.append(whole)
        parts = [p.lower() for piece in word.split("_") for p in _CAMEL.findall(piece)]
        if len(parts) > 1 or parts != [whole]:
            tokens.extend(p for p in parts if p)
    return tokens


def query_tokens(query: str) -> list[str]:
    return list(
        dict.fromkeys(
            t for t in tokenize(query) if t not in _STOPWORDS and len(t) > 1
        )
    )


@dataclass
class Hit:
    chunk_id: str
    score: float


class Retriever:
    def search(self, query: str, top_k: int = 5, file_path: str | None = None) -> list[Hit]:
        raise NotImplementedError

    def rebuild(self, index: RepositoryIndex) -> "Retriever":
        raise NotImplementedError("custom retrievers must implement rebuild(index)")


class LexicalRetriever(Retriever):
    backend_name: str = "rank-bm25 (BM25Plus)"

    def __init__(self, index: RepositoryIndex) -> None:
        self._index = index
        self._chunk_ids: list[str] = list(index.chunks.keys())

        corpus: list[str] = []
        for chunk_id in self._chunk_ids:
            chunk = index.chunks[chunk_id]
            symbol_label = chunk.symbol_id or chunk_id.removesuffix("#module")
            header = f"{symbol_label} {chunk.file}"
            corpus.append(f"{header}\n{header}\n{chunk.text}")

        self._bm25: BM25Plus | None = (
            BM25Plus(corpus, tokenizer=tokenize, k1=K1, b=B, delta=DELTA)
            if corpus
            else None
        )

    def rebuild(self, index: RepositoryIndex) -> "LexicalRetriever":
        return LexicalRetriever(index)

    def search(self, query: str, top_k: int = 5, file_path: str | None = None) -> list[Hit]:
        if self._bm25 is None or not self._chunk_ids:
            return []
        tokens = query_tokens(query)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)
        target_path = file_path.replace("\\", "/").casefold() if file_path is not None else None

        hits: list[Hit] = []
        for idx in np.flatnonzero(scores > 0.0):
            chunk_id = self._chunk_ids[int(idx)]
            if target_path is not None:
                chunk = self._index.chunks.get(chunk_id)
                if chunk is None or chunk.file.replace("\\", "/").casefold() != target_path:
                    continue
            hits.append(Hit(chunk_id, float(scores[int(idx)])))

        hits.sort(key=lambda h: (-h.score, h.chunk_id))
        return hits[:top_k]
