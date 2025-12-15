#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path
import os

# ============================================================
# Args
# ============================================================
p = argparse.ArgumentParser(description="Run a single filter over canonical data")
p.add_argument("--strategy", required=True)
p.add_argument("--asset", required=True)
p.add_argument("--timeframe", required=True)
p.add_argument("--window", type=int, required=True)
p.add_argument("--rr", type=float, required=True)
p.add_argument("--filter", required=True, help="Filter script path, e.g. filters/regime/rec_filters.py")
args = p.parse_args()

rr_str = f"{args.rr:.1f}"

# ============================================================
# Resolve paths
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

filter_script = PROJECT_ROOT / args.filter
if not filter_script.exists():
    print(f"ERROR: filter script not found: {filter_script}")
    sys.exit(2)

exp_root = (
    PROJECT_ROOT
    / "experiments"
    / args.strategy
    / args.asset
    / args.timeframe
    / f"window_{args.window}"
    / f"rr_{rr_str}"
)

canonical_dir = exp_root / "canonical_output"
if not canonical_dir.exists():
    print("ERROR: canonical_output not found. Run run_canonical.py first.")
    sys.exit(3)

iter1_dir = exp_root / "iteration_1"
logs_dir = iter1_dir / "logs"
logs_dir.mkdir(parents=True, exist_ok=True)

for d in ["plots", "equity", "info"]:
    (iter1_dir / d).mkdir(parents=True, exist_ok=True)

# ============================================================
# Run filter
# ============================================================
name = filter_script.stem
out_file = logs_dir / f"{name}.out"
err_file = logs_dir / f"{name}.err"

print("Running filter")
print(" Filter   :", filter_script)
print(" Canonical:", canonical_dir)
print(" Output   :", iter1_dir)

env = dict(os.environ)
env["CANONICAL_DIR"] = str(canonical_dir)
env["OUT_DIR"] = str(iter1_dir)
env["ASSET"] = args.asset
env["TIMEFRAME"] = args.timeframe
env["WINDOW"] = str(args.window)
env["RR"] = rr_str

with out_file.open("wb") as out_f, err_file.open("wb") as err_f:
    ret = subprocess.run(
        [
            sys.executable,
            str(filter_script),
            "--strategy", args.strategy,
            "--asset", args.asset,
            "--timeframe", args.timeframe,
            "--window", str(args.window),
            "--rr", str(args.rr),
        ],
        stdout=out_f,
        stderr=err_f,
        env=env,
    )

if ret.returncode == 0:
    print("OK")
else:
    print(f"FAIL (code {ret.returncode}) → check logs")

sys.exit(ret.returncode)