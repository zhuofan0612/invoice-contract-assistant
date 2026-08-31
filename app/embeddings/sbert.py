"""The intended production embedding backend: local sentence-transformers.

Local, not hosted, on purpose. The design constraint is that procurement data
must not leave the customer's environment, so retrieval has to work with no
outbound calls. Running the bi-encoder and cross-encoder locally keeps that
story intact even while the interpretation step still calls a hosted LLM.

Enable with:  pip install -r requirements-ml.txt
"""

from __future__ import annotations

from typing import Sequence


class SentenceTransformerEmbedder:
    name = "sbert"

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())
        self.name = f"sbert:{model_name}"

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self._model.encode(
            list(texts), normalize_embeddings=True, show_progress_bar=False
        )
        return [v.tolist() for v in vectors]

    def encode_one(self, text: str) -> list[float]:
        return self.encode([text])[0]


class CrossEncoderReranker:
    name = "cross-encoder"

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name)
        self.name = f"cross-encoder:{model_name}"

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        pairs = [(query, p) for p in passages]
        return [float(s) for s in self._model.predict(pairs, show_progress_bar=False)]
