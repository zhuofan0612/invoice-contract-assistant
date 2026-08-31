"""Lexical reranker fallback (BM25-style scoring over the candidate set).

The cross-encoder is the intended second stage. This stands in when torch is
not installed. Unlike the bi-encoder it sees the query and passage together, so
it can reward exact overlap of the discriminating terms ("advisory",
"implementation", "senior technician") rather than overall topical similarity.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Sequence

from app.embeddings.hashing import TOKEN_RE

K1 = 1.5
B = 0.75


class LexicalReranker:
    name = "lexical"

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []

        query_terms = _tokenise(query)
        docs = [_tokenise(p) for p in passages]
        doc_lens = [len(d) for d in docs]
        avg_len = sum(doc_lens) / len(docs) if docs else 0.0

        # Document frequency is computed over the candidate set only. That is
        # the point of a reranker: it discriminates *within* the shortlist, so
        # a term shared by every candidate carries no signal here.
        df = Counter()
        for doc in docs:
            for term in set(doc):
                df[term] += 1
        n_docs = len(docs)

        scores: list[float] = []
        for doc, length in zip(docs, doc_lens):
            tf = Counter(doc)
            score = 0.0
            for term in query_terms:
                if term not in tf:
                    continue
                idf = math.log(1.0 + (n_docs - df[term] + 0.5) / (df[term] + 0.5))
                norm = 1.0 - B + B * (length / avg_len if avg_len else 1.0)
                score += idf * (tf[term] * (K1 + 1.0)) / (tf[term] + K1 * norm)
            scores.append(score)
        return scores


def _tokenise(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]
