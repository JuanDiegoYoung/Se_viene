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
# Accept extra strategy flags for compatibility
p.add_argument("--require-prior-swing", action="store_true", default=False, help="(optional, ignored)")
p.add_argument("--allow-countertrend", action="store_true", default=False, help="(optional, ignored)")
p.add_argument("--allow-micro-structure", action="store_true", default=False, help="(optional, ignored)")
args = p.parse_args()

rr_str = f"{args.rr:.1f}"

# ============================================================
# Paths
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

DATA_DIR = os.path.join(
    PROJECT_ROOT, "candle_data", args.asset, args.timeframe
)

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
    "CANONICAL_DIR", os.path.join(EXP_DIR, "canonical_output")
)
OUT_DIR = os.environ.get(
    "OUT_DIR", os.path.join(EXP_DIR, "iteration_1")
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
# MFI
# ============================================================
tp = (btc["high"] + btc["low"] + btc["close"]) / 3
rmf = tp * btc["volume"]

pos_mf = rmf.where(tp > tp.shift(1), 0.0)
neg_mf = rmf.where(tp < tp.shift(1), 0.0)

PER = 14
pos_sum = pos_mf.rolling(PER).sum()
neg_sum = neg_mf.rolling(PER).sum()

mfr = pos_sum / neg_sum.replace(0, np.nan)
btc["mfi"] = 100 - (100 / (1 + mfr))
btc["mfi_prev"] = btc["mfi"].shift(1)

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
def mfi_extreme(idx, side):
    v = btc["mfi_prev"].iat[idx]
    if pd.isna(v):
        return False
    return v > 80 if side.startswith("l") else v < 20

def mfi_center_momentum(idx, side):
    v = btc["mfi_prev"].iat[idx]
    if pd.isna(v):
        return False
    return v > 50 if side.startswith("l") else v < 50

def mfi_high_quantile(idx, side):
    v = btc["mfi_prev"].iat[idx]
    if pd.isna(v):
        return False
    hist = btc["mfi"].iloc[:idx].dropna()
    if len(hist) < 20:
        return False
    return v > hist.quantile(0.75)

FILTERS = {
    "mfi_extreme": mfi_extreme,
    "mfi_center_momentum": mfi_center_momentum,
    "mfi_high_quantile": mfi_high_quantile,
}

# ============================================================
# Apply filters (SINGLE equity file)
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
    os.path.join(EQUITY_DIR, "mfi_equity.csv"),
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
plt.savefig(os.path.join(PLOTS_DIR, "mfi_filters.png"), dpi=150)
plt.close()

# ============================================================
# Summary
# ============================================================
summary = []
for name, fn in FILTERS.items():
    keep = []

    for _, r in trades.iterrows():
        keep.append(fn(r["entry_idx"], r["side"]))

    trades_f = trades.iloc[[i for i, k in enumerate(keep) if k]]

    s = df_all[df_all["filter"] == name]["equity"].values
    dd = np.maximum.accumulate(s) - s

    win_rate = (
        (trades_f["hit"].astype(str).str.lower() != "sl").mean()
        if len(trades_f) > 0 and "hit" in trades_f.columns
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
    os.path.join(INFO_DIR, "mfi_summary.csv"),
    index=False,
)