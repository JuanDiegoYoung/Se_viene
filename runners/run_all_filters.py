#!/usr/bin/env python3
import argparse
import os
import subprocess
import glob
import sys
import tempfile

# ============================================================
# Args
# ============================================================
p = argparse.ArgumentParser()
p.add_argument("--strategy", required=True)
p.add_argument("--asset", required=True)
p.add_argument("--timeframe", required=True)
p.add_argument("--window", type=int, required=True)
p.add_argument("--rr", type=float, required=True)
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
# ============================================================
# Paths
# ============================================================
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
    flags_dir
)

CANONICAL_DIR = os.path.join(EXP_DIR, "canonical_output")
OUT_DIR = os.path.join(EXP_DIR, "iteration_1")

# If parent requested no saving of equities, redirect OUT_DIR to a tempdir
if os.environ.get("NO_SAVE_EQUITIES"):
    tmp = tempfile.mkdtemp(prefix="pa_filters_out_")
    OUT_DIR = tmp

FILTERS_ROOT = os.path.join(PROJECT_ROOT, "filters")

# ============================================================
# Run all filters
# ============================================================
filter_scripts = sorted(
    glob.glob(os.path.join(FILTERS_ROOT, "**", "*_filters.py"), recursive=True)
)

if not filter_scripts:
    raise SystemExit("No filter scripts found")

total = len(filter_scripts)
# logs dir for individual filter outputs
LOGS_DIR = os.path.join(OUT_DIR, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

for i, script in enumerate(filter_scripts, start=1):
    name = os.path.basename(script)

    cmd = [
        sys.executable,
        script,
        "--strategy", args.strategy,
        "--asset", args.asset,
        "--timeframe", args.timeframe,
        "--window", str(args.window),
        "--rr", rr_str,
    ]
    # Propagar flags de estrategia actuales
    if args.require_prior_swing:
        cmd.append("--require-prior-swing")
    else:
        cmd.append("--no-require-prior-swing")
    if args.allow_countertrend:
        cmd.append("--allow-countertrend")
    else:
        cmd.append("--no-allow-countertrend")
    if args.allow_micro_structure:
        cmd.append("--allow-micro-structure")
    else:
        cmd.append("--no-allow-micro-structure")

    env = os.environ.copy()
    env["CANONICAL_DIR"] = CANONICAL_DIR
    env["OUT_DIR"] = OUT_DIR

    # redirect child stdout/stderr to per-filter log files to avoid interleaving
    logfile_base = os.path.join(LOGS_DIR, f"{name}")
    out_path = logfile_base + ".out"
    err_path = logfile_base + ".err"
    with open(out_path, "wb") as out_f, open(err_path, "wb") as err_f:
        subprocess.run(cmd, check=True, env=env, stdout=out_f, stderr=err_f)

    # per-filter logs are written to iteration_1/logs; keep console silent for each filter