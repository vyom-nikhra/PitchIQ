"""Audit a generated report's numeric claims against its facts.json.

Thin CLI over :mod:`pitchiq.report.audit` — see that module for the method.

Usage:
    python scripts/audit_report.py data/demo/synthetic-derby [--min 0.9]

Exit code is non-zero when the grounded share falls below --min (default 0:
report-only). Output is ASCII (Windows cp1252 console).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from pitchiq.report.audit import audit  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("job_dir", help="job directory containing report/")
    ap.add_argument("--min", type=float, default=0.0,
                    help="fail (exit 1) below this grounded share")
    ap.add_argument("--show-all", action="store_true",
                    help="list grounded claims too, not just misses")
    args = ap.parse_args()

    job = Path(args.job_dir)
    report = (job / "report" / "report.md").read_text(encoding="utf-8")
    facts = json.loads((job / "report" / "facts.json").read_text(encoding="utf-8"))
    results, share = audit(report, facts)

    for tok, ctx, ok in results:
        if args.show_all or not ok:
            print(f"[{'ok' if ok else 'MISS':>4}] {tok:>8}  ...{ctx}...")
    print(f"\n{job.name}: {sum(ok for _, _, ok in results)}/{len(results)} "
          f"numeric claims grounded ({100 * share:.1f}%)")
    if share < args.min:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
