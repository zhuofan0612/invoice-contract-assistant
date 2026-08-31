"""Two-stage clause retrieval.

    query -> [access + contract filter] -> bi-encoder top-N -> cross-encoder top-k

Three things worth noticing, because they are the design decisions:

1. **The filter comes first, in the database.** `SearchFilter` is part of the
   SQL predicate. A clause belonging to a department the caller is not in is
   never scored, so it cannot appear even as a near-miss candidate.

2. **Search is scoped to one contract.** The governing contract was already
   found deterministically from the invoice's supplier + contract ID, so there
   is no reason to semantically search 8 contracts and hope the right one wins.
   Semantic search is used only for the genuinely language-shaped question:
   *which clause inside this contract governs this line?*

3. **Reranking is not optional here.** Within one contract the candidate
   clauses are near-duplicates by design -- "§4.1 strategic advisory at 1 450"
   and "§4.2 implementation at 1 150" are topically almost identical and differ
   by 300 SEK/hour. A bi-encoder ranks those two nearly arbitrarily; the
   cross-encoder is what separates them.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from app.config import Settings, get_settings
from app.embeddings.base import Embedder, Reranker, get_embedder, get_reranker
from app.models import Principal, RetrievedChunk
from app.security.access import filter_for
from app.store.base import VectorStore, get_store


@dataclass
class RetrievalResult:
    chunks: list[RetrievedChunk] = field(default_factory=list)
    candidates_considered: int = 0
    timings_ms: dict[str, float] = field(default_factory=dict)

    @property
    def clause_ids(self) -> list[str]:
        return [c.clause_id for c in self.chunks if c.clause_id]


class ClauseRetriever:
    def __init__(
        self,
        store: VectorStore | None = None,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.store = store or get_store(self.settings)
        self.embedder = embedder or get_embedder(self.settings)
        self.reranker = reranker or get_reranker(self.settings)

    def retrieve(
        self,
        query: str,
        principal: Principal,
        contract_id: str | None = None,
        top_k: int | None = None,
    ) -> RetrievalResult:
        top_k = top_k or self.settings.rerank_k
        timings: dict[str, float] = {}

        # Stage 1: cheap vector search over the filtered candidate set.
        started = time.perf_counter()
        search_filter = filter_for(
            principal, contract_id=contract_id, settings=self.settings,
            limit=self.settings.candidate_k,
        )
        query_vector = self.embedder.encode_one(query)
        candidates = self.store.search(query_vector, search_filter)
        timings["candidate_search_ms"] = (time.perf_counter() - started) * 1000

        if not candidates:
            return RetrievalResult(candidates_considered=0, timings_ms=timings)

        # Stage 2: expensive pairwise scoring, over the shortlist only.
        started = time.perf_counter()
        rerank_scores = self.reranker.score(query, [c.text for c in candidates])
        timings["rerank_ms"] = (time.perf_counter() - started) * 1000

        ranked = sorted(
            zip(candidates, rerank_scores), key=lambda pair: pair[1], reverse=True
        )[:top_k]

        return RetrievalResult(
            chunks=[
                RetrievedChunk(
                    chunk_id=c.chunk_id,
                    contract_id=c.contract_id,
                    supplier=c.supplier,
                    clause_id=c.clause_id,
                    heading=c.heading,
                    text=c.text,
                    has_table=c.has_table,
                    candidate_score=c.score,
                    rerank_score=float(score),
                )
                for c, score in ranked
            ],
            candidates_considered=len(candidates),
            timings_ms=timings,
        )

    def contract_clauses(self, contract_id: str, principal: Principal) -> list[RetrievedChunk]:
        """All clauses of a contract the caller may read (used by the agent tools)."""
        stored = self.store.get_contract_chunks(
            contract_id, filter_for(principal, settings=self.settings)
        )
        return [
            RetrievedChunk(
                chunk_id=c.chunk_id, contract_id=c.contract_id, supplier=c.supplier,
                clause_id=c.clause_id, heading=c.heading, text=c.text,
                has_table=c.has_table,
            )
            for c in stored
        ]
