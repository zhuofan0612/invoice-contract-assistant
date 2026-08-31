"""Request tracing to JSONL.

What gets recorded is chosen around one question: when a caseworker says "this
flag is wrong", can we reconstruct why the system said it? That needs the
inputs to each stage, not just the verdict:

  - which clauses were retrieved, with both stage scores, so a bad answer can
    be attributed to retrieval rather than to the model
  - the raw model output and any repairs the parser applied
  - what the deterministic checker concluded from those inputs
  - stage timings

Deliberately *not* recorded: full invoice line text is redacted when
REDACT_PII is on, because traces outlive requests and a trace file is a much
softer target than the database.

JSONL keeps it dependency-free. `to_langfuse_span()` shows the shape a real
tracing backend (Langfuse, Phoenix) would receive; wiring one up means
replacing the writer, not the instrumentation points.
"""

from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from app.security.redact import redact


@dataclass
class Span:
    name: str
    started_at: float
    duration_ms: float = 0.0
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class Trace:
    trace_id: str
    name: str
    started_at: float
    spans: list[Span] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    redact_pii: bool = True

    def set(self, **attributes: Any) -> None:
        self.attributes.update(attributes)

    def event(self, name: str, **data: Any) -> None:
        self.events.append({"name": name, "at_ms": self._elapsed(), **self._clean(data)})

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[Span]:
        span = Span(name=name, started_at=time.perf_counter(), attributes=self._clean(attributes))
        try:
            yield span
        finally:
            span.duration_ms = (time.perf_counter() - span.started_at) * 1000
            span.attributes = self._clean(span.attributes)
            self.spans.append(span)

    @property
    def timings_ms(self) -> dict[str, float]:
        return {s.name: round(s.duration_ms, 2) for s in self.spans}

    def _elapsed(self) -> float:
        return round((time.perf_counter() - self.started_at) * 1000, 2)

    def _clean(self, data: dict[str, Any]) -> dict[str, Any]:
        if not self.redact_pii:
            return data
        return {k: redact(v) if isinstance(v, str) else v for k, v in data.items()}

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "name": self.name,
            "timestamp": time.time(),
            "duration_ms": self._elapsed(),
            "attributes": self.attributes,
            "events": self.events,
            "spans": [
                {"name": s.name, "duration_ms": round(s.duration_ms, 2), "attributes": s.attributes}
                for s in self.spans
            ],
        }

    def to_langfuse_span(self) -> dict[str, Any]:
        """The same trace in the shape a hosted tracing backend expects."""
        payload = self.to_dict()
        return {
            "id": payload["trace_id"],
            "name": payload["name"],
            "metadata": payload["attributes"],
            "observations": [
                {"name": s["name"], "startTime": None, "endTime": None,
                 "metadata": s["attributes"]}
                for s in payload["spans"]
            ],
        }


class Tracer:
    def __init__(self, path: Path, redact_pii: bool = True) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.redact_pii = redact_pii
        self.last: Trace | None = None

    @contextmanager
    def trace(self, name: str, **attributes: Any) -> Iterator[Trace]:
        trace = Trace(
            trace_id=uuid.uuid4().hex[:16],
            name=name,
            started_at=time.perf_counter(),
            attributes=dict(attributes),
            redact_pii=self.redact_pii,
        )
        try:
            yield trace
        finally:
            self.last = trace
            self._write(trace)

    def _write(self, trace: Trace) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(trace.to_dict(), ensure_ascii=False, default=str) + "\n")

    def read_all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [
            json.loads(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
