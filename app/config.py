"""Runtime configuration.

Every heavy dependency in this project sits behind an interface with a
zero-dependency fallback, and this module decides which implementation is live:

    vector store   pgvector   if DATABASE_URL is set, else SQLite
    embeddings     sentence-transformers if importable, else hashing embedder
    LLM            Anthropic  if ANTHROPIC_API_KEY is set, else offline stub

That means `python -m app` runs end to end on a laptop with no Docker, no
Postgres and no API key, and upgrades to the real components the moment they
become available. The design doc calls for pgvector and a sentence-transformer
bi-encoder; those are the intended production configuration, and the fallbacks
exist so the pipeline logic stays testable without infrastructure.
"""

from __future__ import annotations

import functools
import importlib.util
import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


@dataclass(frozen=True)
class Settings:
    # --- data locations ---
    data_dir: Path = ROOT / "data"

    # --- vector store ---
    store_backend: str = "sqlite"
    database_url: str | None = None
    # Ingestion connects as a different Postgres role to the one that serves
    # requests: the serving role may only SELECT, and only rows its groups
    # allow. Falls back to database_url when unset, which is correct for SQLite
    # and merely undefended for Postgres.
    ingest_database_url: str | None = None
    sqlite_path: Path = ROOT / "var" / "index.sqlite3"

    # --- embeddings / rerank ---
    embedding_backend: str = "hashing"
    embedding_dim: int = 384
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_backend: str = "lexical"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # --- LLM ---
    llm_backend: str = "stub"
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-sonnet-4-6"
    llm_max_tokens: int = 2000

    # --- retrieval knobs ---
    candidate_k: int = 20
    rerank_k: int = 4

    # --- decision thresholds ---
    ambiguity_confidence_threshold: float = 0.75
    rate_tolerance: float = 0.01
    total_tolerance: float = 0.01

    # --- agent loop ---
    max_agent_iterations: int = 6

    # --- observability ---
    trace_path: Path = ROOT / "var" / "traces.jsonl"
    redact_pii: bool = True

    @property
    def contracts_dir(self) -> Path:
        return self.data_dir / "contracts"

    @property
    def invoices_dir(self) -> Path:
        return self.data_dir / "invoices"

    @property
    def labels_path(self) -> Path:
        return self.data_dir / "labels.json"

    @property
    def access_path(self) -> Path:
        return self.data_dir / "access.json"

    def describe(self) -> dict[str, str]:
        """Human-readable summary of which implementations are live."""
        return {
            "store_backend": self.store_backend,
            "embedding_backend": self.embedding_backend,
            "reranker_backend": self.reranker_backend,
            "llm_backend": self.llm_backend,
            "llm_model": self.anthropic_model if self.llm_backend == "anthropic" else "n/a",
        }


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    database_url = os.getenv("DATABASE_URL") or None
    ingest_database_url = os.getenv("INGEST_DATABASE_URL") or None
    api_key = os.getenv("ANTHROPIC_API_KEY") or None

    store_backend = os.getenv("STORE_BACKEND") or ("pgvector" if database_url else "sqlite")

    sbert_available = _module_available("sentence_transformers")
    embedding_backend = os.getenv("EMBEDDING_BACKEND") or ("sbert" if sbert_available else "hashing")
    reranker_backend = os.getenv("RERANKER_BACKEND") or ("cross-encoder" if sbert_available else "lexical")

    llm_backend = os.getenv("LLM_BACKEND") or ("anthropic" if api_key else "stub")

    data_dir = Path(os.getenv("DATA_DIR", str(ROOT / "data")))
    var_dir = Path(os.getenv("VAR_DIR", str(ROOT / "var")))
    var_dir.mkdir(parents=True, exist_ok=True)

    return Settings(
        data_dir=data_dir,
        store_backend=store_backend,
        database_url=database_url,
        ingest_database_url=ingest_database_url,
        sqlite_path=Path(os.getenv("SQLITE_PATH", str(var_dir / "index.sqlite3"))),
        embedding_backend=embedding_backend,
        embedding_dim=int(os.getenv("EMBEDDING_DIM", "384")),
        embedding_model=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        reranker_backend=reranker_backend,
        reranker_model=os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        llm_backend=llm_backend,
        anthropic_api_key=api_key,
        anthropic_model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
        candidate_k=int(os.getenv("CANDIDATE_K", "20")),
        rerank_k=int(os.getenv("RERANK_K", "4")),
        ambiguity_confidence_threshold=float(os.getenv("AMBIGUITY_THRESHOLD", "0.75")),
        max_agent_iterations=int(os.getenv("MAX_AGENT_ITERATIONS", "6")),
        trace_path=Path(os.getenv("TRACE_PATH", str(var_dir / "traces.jsonl"))),
        redact_pii=_flag("REDACT_PII", True),
    )
