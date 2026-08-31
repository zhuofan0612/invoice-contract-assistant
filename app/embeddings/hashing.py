"""Dependency-free bi-encoder fallback (the hashing trick).

This exists so the whole retrieval pipeline runs, and is testable, without
downloading a ~90 MB transformer and its torch dependency. It is a real vector
embedding, just a lexical one: features are hashed straight into a fixed number
of dimensions with a signed hash, so there is no vocabulary to fit and no
training step.

What it can and cannot do, stated plainly because it matters for reading the
eval numbers:

* It matches shared wording and shared word-pieces, so "ventilation
  maintenance" finds the ventilation clause, and char n-grams give it partial
  credit on "snr tech" vs "senior technician".
* It has no concept of meaning, so "go-live" vs "implementation" is invisible
  to it. That is precisely the gap the sentence-transformer backend closes,
  and comparing the two backends on the same golden set shows the difference.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Iterable, Sequence

TOKEN_RE = re.compile(r"[a-zA-ZåäöÅÄÖéÉüÜ0-9]+")


class HashingEmbedder:
    name = "hashing"

    def __init__(self, dim: int = 384, char_ngram: int = 4) -> None:
        self.dim = dim
        self.char_ngram = char_ngram

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.encode_one(t) for t in texts]

    def encode_one(self, text: str) -> list[float]:
        counts = Counter(self._features(text))
        vector = [0.0] * self.dim
        for feature, tf in counts.items():
            index, sign = self._hash(feature)
            # Sublinear term frequency: a rate repeated five times in a clause
            # should not dominate the vector five-fold.
            vector[index] += sign * (1.0 + math.log(tf))
        return _l2_normalise(vector)

    def _features(self, text: str) -> Iterable[str]:
        tokens = [t.lower() for t in TOKEN_RE.findall(text)]
        for token in tokens:
            yield f"w:{token}"
            if len(token) > self.char_ngram:
                padded = f"^{token}$"
                for i in range(len(padded) - self.char_ngram + 1):
                    yield f"c:{padded[i : i + self.char_ngram]}"
        for a, b in zip(tokens, tokens[1:]):
            yield f"b:{a}_{b}"

    def _hash(self, feature: str) -> tuple[int, float]:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        # Low bit picks the sign, so colliding features cancel rather than
        # always reinforcing each other.
        return value % self.dim, 1.0 if (value >> 63) & 1 else -1.0


def _l2_normalise(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0.0:
        return vector
    return [v / norm for v in vector]
