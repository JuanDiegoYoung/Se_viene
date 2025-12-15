#!/usr/bin/env python3
import argparse
import os
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
# Paths
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

DATA_DIR = os.path.join(PROJECT_ROOT, "candle_data", args.asset, args.timeframe)

EXP_DIR = os.path.join(
    PROJECT_ROOT,
    "experiments",
    args.strategy,
    args.asset,
    args.timeframe,
    f"window_{args.window}",
    f"rr_{rr_str}",
)

CANONICAL_DIR = os.environ.get(
    "CANONICAL_DIR",
    os.path.join(EXP_DIR, "canonical_output")
)

OUT_DIR = os.environ.get(
    "OUT_DIR",
    os.path.join(EXP_DIR, "iteration_1")
)

PLOTS_DIR = os.path.join(OUT_DIR, "plots")
EQUITY_DIR = os.path.join(OUT_DIR, "equity")
INFO_DIR = os.path.join(OUT_DIR, "info")

for d in [PLOTS_DIR, EQUITY_DIR, INFO_DIR]:
    os.makedirs(d, exist_ok=True)

# ============================================================
# Load candles
# ============================================================
csvs = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
btc = pd.read_csv(os.path.join(DATA_DIR, csvs[0]))

if "open_time" in btc.columns:
    btc["date"] = pd.to_datetime(btc["open_time"], unit="ms")
else:
    btc["date"] = pd.to_datetime(btc["date"])

btc = btc.sort_values("date").reset_index(drop=True)
for c in ["open", "high", "low", "close", "volume"]:
    btc[c] = pd.to_numeric(btc[c], errors="coerce")

N = len(btc)

# ============================================================
# REC (Range Expansion / Compression)
# ============================================================
REC_PERIOD = 20

btc["range"] = btc["high"] - btc["low"]
btc["range_med"] = btc["range"].rolling(REC_PERIOD).median()
btc["range_prev"] = btc["range"].shift(1)
btc["range_med_prev"] = btc["range_med"].shift(1)
btc["range_ratio_prev"] = btc["range_prev"] / btc["range_med_prev"]

# ============================================================
# Load canonical trades & equity
# ============================================================
trades = pd.read_csv(os.path.join(CANONICAL_DIR, "trades.csv"))
eq_df = pd.read_csv(os.path.join(CANONICAL_DIR, "equity.csv"))

eq_col = [c for c in eq_df.columns if "equity" in c.lower()][0]
eq_all = eq_df[eq_col].astype(float).values[:N]

trades["entry_idx"] = trades["entry_idx"].astype(int)
trades["exit_idx"] = trades["exit_idx"].astype(int)
trades["side"] = trades["side"].astype(str).str.lower()

contrib_all = np.zeros(N)
for _, r in trades.iterrows():
    contrib_all[r["exit_idx"]] += r["R"]

# ============================================================
# Filters
# ============================================================
def rec_expand(idx, side, min_ratio):
    v = btc["range_ratio_prev"].iat[idx]
    if pd.isna(v):
        return False
    return v > min_ratio

def rec_compress(idx, side, max_ratio):
    v = btc["range_ratio_prev"].iat[idx]
    if pd.isna(v):
        return False
    return v < max_ratio

FILTERS = {}

for v in [1.3, 1.5, 1.7, 2.0]:
    FILTERS[f"rec_expand_{v:.2f}"] = lambda idx, side, v=v: rec_expand(idx, side, v)

for v in [0.7, 0.6, 0.5, 0.4]:
    FILTERS[f"rec_compress_{v:.2f}"] = lambda idx, side, v=v: rec_compress(idx, side, v)

# ============================================================
# Apply filters
# ============================================================
rows = []

for name, fn in FILTERS.items():
    contrib_filt = np.zeros(N)
    keep = []

    for _, r in trades.iterrows():
        keep.append(fn(r["entry_idx"], r["side"]))

    trades_f = trades.iloc[[i for i, k in enumerate(keep) if k]]

    for _, r in trades_f.iterrows():
        contrib_filt[r["exit_idx"]] += r["R"]

    eq_filt = eq_all - np.cumsum(contrib_all - contrib_filt)

    rows.append(pd.DataFrame({
        "date": btc["date"],
        "filter": name,
        "equity": eq_filt,
    }))

df_all = pd.concat(rows, ignore_index=True)

df_all.to_csv(
    os.path.join(EQUITY_DIR, "rec_equity.csv"),
    index=False,
)

# ============================================================
# Plot
# ============================================================
plt.figure(figsize=(14, 6))
plt.plot(btc["date"], eq_all, label="baseline", color="black", linewidth=1.5)

for name in FILTERS:
    df = df_all[df_all["filter"] == name]
    plt.plot(df["date"], df["equity"], label=name)

plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "rec_filters.png"), dpi=150)
plt.close()

# ============================================================
# Summary (FIX)
# ============================================================
summary = []

for name, fn in FILTERS.items():
    keep = []

    for _, r in trades.iterrows():
        keep.append(fn(r["entry_idx"], r["side"]))

    trades_f = trades.iloc[[i for i, k in enumerate(keep) if k]]

    df = df_all[df_all["filter"] == name]
    s = df["equity"].values
    dd = np.maximum.accumulate(s) - s

    win_rate = (
        float((trades_f["hit"] != "sl").mean())
        if "hit" in trades_f.columns and len(trades_f) > 0
        else np.nan
    )

    summary.append({
        "filter": name,
        "final_equity": float(s[-1]),
        "maxdd": float(dd.max()),
        "n_trades": int(len(trades_f)),
        "win_rate": win_rate,
    })

pd.DataFrame(summary).to_csv(
    os.path.join(INFO_DIR, "rec_summary.csv"),
    index=False,
)