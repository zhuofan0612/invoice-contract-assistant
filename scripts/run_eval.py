"""Run the evaluation and print the report.

    python scripts/run_eval.py                  # fixed pipeline
    python scripts/run_eval.py --mode agent     # tool-calling loop
    python scripts/run_eval.py --compare        # both, side by side
    python scripts/run_eval.py --json out.json  # machine-readable

`--compare` is the point of the whole file. Two retrieval strategies against
one labelled set and one deterministic checker means the difference between
them is attributable, which is the only way to say whether the agent is worth
its latency instead of guessing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.harness import run_eval  # noqa: E402
from eval.metrics import format_report  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the decision pipeline.")
    parser.add_argument("--mode", choices=["pipeline", "agent"], default="pipeline")
    parser.add_argument("--compare", action="store_true",
                        help="run both modes and print the difference")
    parser.add_argument("--json", type=Path, default=None,
                        help="also write the full report (including per-invoice rows)")
    parser.add_argument("--verbose", action="store_true",
                        help="list every invoice, not just the failures")
    args = parser.parse_args()

    modes = ["pipeline", "agent"] if args.compare else [args.mode]
    reports = {}

    for mode in modes:
        report = run_eval(mode=mode)
        reports[mode] = report
        print()
        print(format_report(report, title=f"MODE: {mode}"))

        rows = report["rows"]
        shown = rows if args.verbose else [r for r in rows if not r["category_correct"]]
        if shown:
            print()
            print("-- per invoice --" if args.verbose else "-- failures --")
            for row in shown:
                mark = "ok" if row["category_correct"] else "XX"
                retrieved = "" if row["recall"] else "  [GOLD CLAUSE NOT RETRIEVED]"
                print(
                    f"  {mark} {row['invoice_id']}  "
                    f"expected={row['expected_category']:<24} "
                    f"got={row['got_category']:<24} "
                    f"cited={row['cited_clause'] or '-'}{retrieved}"
                )
                for err in row["errors"]:
                    print(f"       ! {err}")

    if args.compare:
        a, b = reports["pipeline"], reports["agent"]
        print()
        print("COMPARISON")
        print("==========")
        for label, key in [
            ("category accuracy", "category_accuracy"),
            ("decision accuracy", "decision_accuracy"),
            ("retrieval recall", "retrieval_recall"),
        ]:
            delta = b[key] - a[key]
            print(f"  {label:<20} pipeline {a[key]:.2%}   agent {b[key]:.2%}   "
                  f"({delta:+.2%})")
        pa, ag = a["latency_ms"]["p50"], b["latency_ms"]["p50"]
        factor = f"{ag / pa:.1f}x" if pa else "n/a"
        print(f"  {'p50 latency':<20} pipeline {pa}ms   agent {ag}ms   ({factor})")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        payload = reports if args.compare else reports[args.mode]
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
