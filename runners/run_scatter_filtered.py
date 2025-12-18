#!/usr/bin/env python3
import argparse
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("--strategy", required=True)
p.add_argument("--asset", required=True)
p.add_argument("--timeframe", required=True)
p.add_argument("--window", type=int, required=True)
p.add_argument("--rr", type=float, required=True)
# Accept extra strategy flags for compatibility
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

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
flags_dir = f"prior_{args.require_prior_swing}_counter_{args.allow_countertrend}_micro_{args.allow_micro_structure}"
SCATTERS_DIR = os.path.join(
    PROJECT_ROOT,
    "experiments",
    args.strategy,
    args.asset,
    args.timeframe,
    f"window_{args.window}",
    f"rr_{rr_str}",
    flags_dir,
    "scatters",
)

DATA_DIR = os.path.join(SCATTERS_DIR, "data")
IN_CSV = os.path.join(DATA_DIR, "filters_scatter_data.csv")
OUT_PNG = os.path.join(SCATTERS_DIR, "filters_scatter_filtered.png")

df = pd.read_csv(IN_CSV)

df = df[
    (df["final_equity"] > 100) &
    (df["maxdd"] < 50)
].copy()

df["score"] = df["final_equity"] / df["maxdd"]

df = df.sort_values("score", ascending=False)

top10 = df.head(10)

x = df["maxdd"].astype(float)
y = df["final_equity"].astype(float)
c = df["score"].astype(float)

plt.figure(figsize=(11, 8))

plt.scatter(
    x,
    y,
    c=c,
    s=80,
    alpha=0.75,
    edgecolors="k",
    linewidths=0.35,
)

for _, r in df.iterrows():
    plt.text(
        r["maxdd"],
        r["final_equity"],
        f"{r['score']:.2f}",
        fontsize=9,
        ha="left",
        va="bottom",
    )

plt.xlabel("Max Drawdown (absolute)")
plt.ylabel("Final Equity")
plt.title("Filtered strategies (equity > 100, drawdown < 50)")

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=160)
plt.close()

# quiet: filtered scatter saved to disk