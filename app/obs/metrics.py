"""In-process metrics, exposed at /metrics in Prometheus text format.

The counters are chosen to answer operational questions rather than to look
busy. The two that matter most in this system:

  `llm_output_repairs_total` -- how often the model's JSON needed fixing. This
  drifts when a model version changes, and it drifts *before* accuracy visibly
  drops, so it is an early warning rather than a postmortem statistic.

  `decisions_total{finding=...}` -- the mix of findings. A sudden collapse in
  flag rate usually means retrieval broke, not that suppliers got honest.

No Prometheus client dependency: the exposition format is a few lines of text
and this keeps the container thin.
"""

from __future__ import annotations

import threading
from collections import defaultdict


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[tuple[str, tuple[tuple[str, str], ...]], float] = defaultdict(float)
        self._histograms: dict[str, list[float]] = defaultdict(list)

    def increment(self, name: str, value: float = 1.0, **labels: str) -> None:
        key = (name, tuple(sorted(labels.items())))
        with self._lock:
            self._counters[key] += value

    def observe(self, name: str, value: float) -> None:
        with self._lock:
            self._histograms[name].append(value)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "counters": {
                    f"{name}{_render_labels(labels)}": value
                    for (name, labels), value in self._counters.items()
                },
                "histograms": {
                    name: _summarise(values) for name, values in self._histograms.items()
                },
            }

    def render_prometheus(self) -> str:
        lines: list[str] = []
        with self._lock:
            counters = dict(self._counters)
            histograms = {k: list(v) for k, v in self._histograms.items()}

        seen: set[str] = set()
        for (name, labels), value in sorted(counters.items()):
            if name not in seen:
                lines.append(f"# TYPE {name} counter")
                seen.add(name)
            lines.append(f"{name}{_render_labels(labels)} {value}")

        for name, values in sorted(histograms.items()):
            if not values:
                continue
            stats = _summarise(values)
            lines.append(f"# TYPE {name} summary")
            lines.append(f'{name}{{quantile="0.5"}} {stats["p50"]}')
            lines.append(f'{name}{{quantile="0.95"}} {stats["p95"]}')
            lines.append(f"{name}_count {stats['count']}")
            lines.append(f"{name}_sum {stats['sum']}")

        return "\n".join(lines) + "\n"

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._histograms.clear()


def _render_labels(labels: tuple[tuple[str, str], ...]) -> str:
    if not labels:
        return ""
    inner = ",".join(f'{k}="{v}"' for k, v in labels)
    return "{" + inner + "}"


def _summarise(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0, "sum": 0.0, "p50": 0.0, "p95": 0.0}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "sum": round(sum(ordered), 3),
        "p50": round(ordered[int(len(ordered) * 0.5)], 3),
        "p95": round(ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))], 3),
    }


METRICS = Metrics()
