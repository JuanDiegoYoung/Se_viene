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

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

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

for _, r in top.iterrows():
    mask = (trades["filter_a"] == r["filter_a"]) & (trades["filter_b"] == r["filter_b"])
    t = trades.loc[mask]

    durations = t["exit_idx"] - t["entry_idx"]

    rows.append({
        "filter_a": r["filter_a"],
        "filter_b": r["filter_b"],
        "mean_duration": durations.mean(),
        "median_duration": durations.median(),
        "p90_duration": durations.quantile(0.9),
        "n_trades": len(durations),
    })

df_out = pd.DataFrame(rows)

df_out.to_csv(
    os.path.join(OUT_DIR, "trade_durations_top10.csv"),
    index=False,
)