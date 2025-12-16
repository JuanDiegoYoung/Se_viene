#!/usr/bin/env python3
import argparse
import os
import pandas as pd
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--strategy", required=True)
p.add_argument("--asset", required=True)
p.add_argument("--timeframe", required=True)
p.add_argument("--window", type=int, required=True)
p.add_argument("--rr", type=float, required=True)
p.add_argument("--top_csv", required=True)
args = p.parse_args()

rr_str = f"{args.rr:.1f}"

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

CANONICAL_DIR = os.path.join(EXP_DIR, "canonical_output")
OUT_DIR = os.path.join(EXP_DIR, "scatters")
os.makedirs(OUT_DIR, exist_ok=True)

trades = pd.read_csv(os.path.join(CANONICAL_DIR, "trades.csv"))
top = pd.read_csv(args.top_csv)

rows = []

# Try to load keep cache (preferred). If present, compute durations using keep masks.
SCATTERS_DIR = os.path.join(EXP_DIR, "scatters")
cache_path = os.path.join(SCATTERS_DIR, "filter_keep_cache.npz")
have_cache = os.path.exists(cache_path)
if have_cache:
    data = np.load(cache_path, allow_pickle=True)
    filter_names = [str(x) for x in data["filter_names"].tolist()]
    keep_matrix = data["keep_matrix"].astype(bool)
    name_to_idx = {n: i for i, n in enumerate(filter_names)}

for _, r in top.iterrows():
    a = str(r["filter_a"])
    b = str(r["filter_b"])

    if have_cache and a in name_to_idx and b in name_to_idx:
        idx_a = name_to_idx[a]
        idx_b = name_to_idx[b]
        keep_ab = keep_matrix[idx_a] & keep_matrix[idx_b]
        t = trades.loc[keep_ab]
    else:
        # fallback: expect trades to include filter_a/filter_b columns
        if "filter_a" in trades.columns and "filter_b" in trades.columns:
            mask = (trades["filter_a"] == a) & (trades["filter_b"] == b)
            t = trades.loc[mask]
        else:
            raise SystemExit("Cannot compute trade durations: missing keep cache and trades lack filter_a/filter_b columns")

    durations = (t["exit_idx"] - t["entry_idx"]).astype(float)

    rows.append({
        "filter_a": a,
        "filter_b": b,
        "mean_duration": float(durations.mean()) if len(durations) else float("nan"),
        "median_duration": float(durations.median()) if len(durations) else float("nan"),
        "p90_duration": float(durations.quantile(0.9)) if len(durations) else float("nan"),
        "n_trades": int(len(durations)),
    })

df_out = pd.DataFrame(rows)

df_out.to_csv(
    os.path.join(OUT_DIR, "trade_durations_top10.csv"),
    index=False,
)