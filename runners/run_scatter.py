#!/usr/bin/env python3
import argparse
import os
import glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
# Paths (FIX ROOT)
# ============================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

EXP_DIR = os.path.join(
    PROJECT_ROOT,
    "experiments",
    args.strategy,
    args.asset,
    args.timeframe,
    f"window_{args.window}",
    f"rr_{rr_str}",
)

INFO_DIR = os.path.join(EXP_DIR, "iteration_1", "info")

SCATTERS_DIR = os.path.join(EXP_DIR, "scatters")
os.makedirs(SCATTERS_DIR, exist_ok=True)
DATA_DIR = os.path.join(SCATTERS_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

OUT_CSV = os.path.join(DATA_DIR, "filters_scatter_data.csv")
OUT_PNG = os.path.join(SCATTERS_DIR, "filters_scatter.png")

if not os.path.isdir(INFO_DIR):
    raise SystemExit(f"INFO_DIR not found: {INFO_DIR}")

# ============================================================
# Collect summaries
# ============================================================
rows = []

summary_paths = glob.glob(os.path.join(INFO_DIR, "*summary*.csv"))

if not summary_paths:
    raise SystemExit(f"No summary files found in {INFO_DIR}")

for path in summary_paths:
    df = pd.read_csv(path)

    for col in ["filter", "final_equity", "maxdd", "n_trades"]:
        if col not in df.columns:
            continue

    for _, r in df.iterrows():
        rows.append({
            "filter": r.get("filter"),
            "final_equity": r.get("final_equity"),
            "maxdd": r.get("maxdd") if "maxdd" in df.columns else r.get("maxdd_units"),
            "win_rate": r.get("win_rate") if "win_rate" in df.columns else np.nan,
            "n_trades": r.get("n_trades"),
        })

df_all = pd.DataFrame(rows)

if df_all.empty:
    raise SystemExit("No valid rows collected from summaries")

df_all = df_all.dropna(subset=["final_equity", "maxdd"])

df_all.to_csv(OUT_CSV, index=False)

# ============================================================
# Scatter
# ============================================================
x = df_all["maxdd"].astype(float)
y = df_all["final_equity"].astype(float)
c = df_all["n_trades"].astype(float)

plt.figure(figsize=(11, 8))

sc = plt.scatter(
    x,
    y,
    c=c,
    s=80,
    cmap="viridis",
    alpha=0.75,
    edgecolors="k",
    linewidths=0.35,
)

plt.colorbar(sc, label="Win rate")
plt.xlabel("Max Drawdown (absolute)")
plt.ylabel("Final Equity")
plt.title("Filters comparison")

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=160)
plt.close()

# quiet: outputs saved to disk