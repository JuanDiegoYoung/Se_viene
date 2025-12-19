#!/usr/bin/env python3
"""Delete CSV files created in the current iteration (iteration_1),
except those under any `pairwise_winners` folder.

Designed to be run as the last runner.
"""
from __future__ import annotations

import argparse
import os
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--strategy", required=True)
    p.add_argument("--asset", required=True)
    p.add_argument("--timeframe", required=True)
    p.add_argument("--window", type=int, required=True)
    p.add_argument("--rr", type=float, required=True)
    # strategy flags accepted for compatibility
    p.add_argument("--require-prior-swing", dest="require_prior_swing", action="store_true")
    p.add_argument("--no-require-prior-swing", dest="require_prior_swing", action="store_false")
    p.set_defaults(require_prior_swing=True)
    p.add_argument("--allow-countertrend", dest="allow_countertrend", action="store_true")
    p.add_argument("--no-allow-countertrend", dest="allow_countertrend", action="store_false")
    p.set_defaults(allow_countertrend=False)
    p.add_argument("--allow-micro-structure", dest="allow_micro_structure", action="store_true")
    p.add_argument("--no-allow-micro-structure", dest="allow_micro_structure", action="store_false")
    p.set_defaults(allow_micro_structure=True)

    args = p.parse_args()

    rr_str = f"{args.rr:.1f}"

    PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

    flags_dir = f"prior_{args.require_prior_swing}_counter_{args.allow_countertrend}_micro_{args.allow_micro_structure}"
    EXP_DIR = os.path.join(
        PROJECT_ROOT,
        "experiments",
        args.strategy,
        args.asset,
        args.timeframe,
        f"window_{args.window}",
        f"rr_{rr_str}",
    )

    ITER_DIR = os.path.join(EXP_DIR, flags_dir, "iteration_1")

    if not os.path.isdir(ITER_DIR):
        print(f"No iteration directory: {ITER_DIR}")
        return 0

    deleted = []
    skipped = []

    for root, _, files in os.walk(ITER_DIR):
        for fname in files:
            if not fname.lower().endswith(".csv"):
                continue
            full = os.path.join(root, fname)
            # normalize for consistent check
            norm = full.replace(os.sep, "/")
            if "/pairwise_winners/" in norm:
                skipped.append(full)
                continue
            try:
                os.remove(full)
                deleted.append(full)
            except Exception as e:
                print(f"Failed to remove {full}: {e}")

    print(f"Deleted {len(deleted)} CSV(s)")
    if deleted:
        for d in deleted:
            print(" -", d)
    if skipped:
        print(f"Skipped {len(skipped)} CSV(s) under pairwise_winners")

    return 0


if __name__ == "__main__":
    sys.exit(main())
