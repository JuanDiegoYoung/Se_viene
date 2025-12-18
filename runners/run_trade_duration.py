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
p.add_argument("--top_csv", required=False, help="Path to top CSV (optional). If omitted the script will try to detect top CSV in pairwise_winners or scatters directories.")
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

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

EXP_DIR = os.path.join(
    PROJECT_ROOT,
    "experiments",
    args.strategy,
    args.asset,
    args.timeframe,
    f"window_{args.window}",
        f"rr_{rr_str}",
        f"prior_{args.require_prior_swing}_counter_{args.allow_countertrend}_micro_{args.allow_micro_structure}",
)
CANONICAL_DIR = os.path.join(EXP_DIR, "canonical_output")
OUT_DIR = os.path.join(EXP_DIR, "scatters")
os.makedirs(OUT_DIR, exist_ok=True)

trades = pd.read_csv(os.path.join(CANONICAL_DIR, "trades.csv"))

# auto-detect top CSV if not provided
top_csv = args.top_csv
if top_csv is None:
    candidates = []
    pw = os.path.join(EXP_DIR, "pairwise_winners", "top_pairwise.csv")
    if os.path.exists(pw):
        candidates.append(pw)
    scat_data = os.path.join(EXP_DIR, "scatters", "data")
    if os.path.isdir(scat_data):
        for f in os.listdir(scat_data):
            if f.startswith("top10_pairs_by_") and f.endswith(".csv"):
                candidates.append(os.path.join(scat_data, f))
    scat_root = os.path.join(EXP_DIR, "scatters")
    if os.path.isdir(scat_root):
        for f in os.listdir(scat_root):
            if f.startswith("top10_pairs_by_") and f.endswith(".csv"):
                candidates.append(os.path.join(scat_root, f))

    pref = [pw,
            os.path.join(scat_data, "top10_pairs_by_win_rate.csv") if os.path.isdir(scat_data) else None,
            os.path.join(scat_data, "top10_pairs_by_n_trades.csv") if os.path.isdir(scat_data) else None]
    top_csv = None
    for p in pref:
        if p and os.path.exists(p):
            top_csv = p
            break
    if top_csv is None and candidates:
        top_csv = sorted(candidates)[0]
if top_csv is None or not os.path.exists(top_csv):
    raise SystemExit(f"Top CSV not found: {top_csv}")

top = pd.read_csv(top_csv)

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