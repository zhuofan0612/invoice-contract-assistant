"""Embedding and reranking interfaces, plus the factory that picks one.

Two stages, two different jobs:

* The **bi-encoder** embeds every clause once at ingest time and the query at
  search time, so candidate search is a cheap vector comparison. It is fast but
  compresses a whole clause into one vector, so it is recall-oriented.
* The **cross-encoder** sees the query and one clause *together* and scores the
  pair. It is far more accurate about which of two similar clauses actually
  answers the query, and far too slow to run over a whole corpus.

Hence: bi-encoder gets ~20 candidates, cross-encoder reorders them to ~4. That
is the standard precision/latency trade, and it is the thing that decides
between "§4.1 strategic advisory" and "§4.2 implementation".
"""

from __future__ import annotations

import logging
from typing import Protocol, Sequence, runtime_checkable

from app.config import Settings, get_settings

log = logging.getLogger(__name__)


@runtime_checkable
class Embedder(Protocol):
    """Maps text to a fixed-size L2-normalised vector."""

    name: str
    dim: int

    def encode(self, texts: Sequence[str]) -> list[list[float]]: ...

    def encode_one(self, text: str) -> list[float]: ...


@runtime_checkable
class Reranker(Protocol):
    """Scores (query, passage) pairs. Higher is more relevant."""

    name: str

    def score(self, query: str, passages: Sequence[str]) -> list[float]: ...


def get_embedder(settings: Settings | None = None) -> Embedder:
    settings = settings or get_settings()
    if settings.embedding_backend == "sbert":
        try:
            from app.embeddings.sbert import SentenceTransformerEmbedder

            return SentenceTransformerEmbedder(settings.embedding_model)
        except Exception as exc:
            log.warning(
                "sentence-transformers unavailable (%s); falling back to hashing embedder", exc
            )

    from app.embeddings.hashing import HashingEmbedder

    return HashingEmbedder(dim=settings.embedding_dim)


def get_reranker(settings: Settings | None = None) -> Reranker:
    settings = settings or get_settings()
    if settings.reranker_backend == "cross-encoder":
        try:
            from app.embeddings.sbert import CrossEncoderReranker

            return CrossEncoderReranker(settings.reranker_model)
        except Exception as exc:
            log.warning("cross-encoder unavailable (%s); falling back to lexical reranker", exc)

    from app.embeddings.rerank import LexicalReranker

    return LexicalReranker()
