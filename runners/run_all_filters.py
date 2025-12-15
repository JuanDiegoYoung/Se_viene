#!/usr/bin/env python3
import argparse
import os
import subprocess
import glob

# ============================================================
# Args
# ============================================================
p = argparse.ArgumentParser()
p.add_argument("--strategy", required=True)
p.add_argument("--asset", required=True)
p.add_argument("--timeframe", required=True)
p.add_argument("--window", type=int, required=True)
p.add_argument("--rr", type=float, required=True)
args = p.parse_args()

rr_str = f"{args.rr:.1f}"

# ============================================================
# Paths
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

EXP_DIR = os.path.join(
    PROJECT_ROOT,
    "experiments",
    args.strategy,
    args.asset,
    args.timeframe,
    f"window_{args.window}",
    f"rr_{rr_str}",
)

CANONICAL_DIR = os.path.join(EXP_DIR, "canonical_output")
OUT_DIR = os.path.join(EXP_DIR, "iteration_1")

FILTERS_ROOT = os.path.join(PROJECT_ROOT, "filters")

# ============================================================
# Run all filters
# ============================================================
filter_scripts = sorted(
    glob.glob(os.path.join(FILTERS_ROOT, "**", "*_filters.py"), recursive=True)
)

if not filter_scripts:
    raise SystemExit("No filter scripts found")

for script in filter_scripts:
    print(f"Running filter: {os.path.basename(script)}")

    cmd = [
        "python",
        script,
        "--strategy", args.strategy,
        "--asset", args.asset,
        "--timeframe", args.timeframe,
        "--window", str(args.window),
        "--rr", rr_str,
    ]

    env = os.environ.copy()
    env["CANONICAL_DIR"] = CANONICAL_DIR
    env["OUT_DIR"] = OUT_DIR

    subprocess.run(cmd, check=True, env=env)

print("All filters executed successfully")