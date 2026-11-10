"""Run the benchmark with no camera and no Hermes: python benchmark/run.py [--scenario N ...] [--skip-could]
Uses the committed T2 answers in benchmark/cache/. To re-check against a real model, run
`hermes orders benchmark --live-vision` (it needs Hermes for model access)."""

from __future__ import annotations
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from standing_orders.benchmark import report, run_all  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--scenario", type=int, nargs="*")
ap.add_argument("--skip-could", action="store_true", help="skip the sticky-note scenarios (8, 9)")
args = ap.parse_args()
print(report(run_all(scenarios=args.scenario, include_could=not args.skip_could)))
