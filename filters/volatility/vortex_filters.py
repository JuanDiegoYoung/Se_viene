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

CANONICAL_DIR = os.environ.get("CANONICAL_DIR", os.path.join(EXP_DIR, "canonical_output"))
OUT_DIR = os.environ.get("OUT_DIR", os.path.join(EXP_DIR, "iteration_1"))

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
# Vortex Indicator
# ============================================================
high = btc["high"]
low = btc["low"]
close = btc["close"]

vm_plus = (high - low.shift(1)).abs()
vm_minus = (low - high.shift(1)).abs()

tr = pd.concat([
    high - low,
    (high - close.shift(1)).abs(),
    (low - close.shift(1)).abs()
], axis=1).max(axis=1)

PER = 14
btc["vi_plus"] = vm_plus.rolling(PER).sum() / tr.rolling(PER).sum()
btc["vi_minus"] = vm_minus.rolling(PER).sum() / tr.rolling(PER).sum()

btc["vi_plus_prev"] = btc["vi_plus"].shift(1)
btc["vi_minus_prev"] = btc["vi_minus"].shift(1)
btc["vi_plus_prev2"] = btc["vi_plus"].shift(2)
btc["vi_minus_prev2"] = btc["vi_minus"].shift(2)

# ============================================================
# Trades & equity
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
def vortex_trend(idx, side):
    vp = btc["vi_plus_prev"].iat[idx]
    vm = btc["vi_minus_prev"].iat[idx]
    if pd.isna(vp) or pd.isna(vm):
        return False
    return vp > vm if side.startswith("l") else vp < vm

def vortex_cross(idx, side):
    vp1 = btc["vi_plus_prev"].iat[idx]
    vm1 = btc["vi_minus_prev"].iat[idx]
    vp2 = btc["vi_plus_prev2"].iat[idx]
    vm2 = btc["vi_minus_prev2"].iat[idx]
    if pd.isna(vp1) or pd.isna(vm1) or pd.isna(vp2) or pd.isna(vm2):
        return False
    if side.startswith("l"):
        return vp1 > vm1 and vp2 <= vm2
    else:
        return vp1 < vm1 and vp2 >= vm2

def vortex_strength(idx, side, thresh=1.1):
    vp = btc["vi_plus_prev"].iat[idx]
    vm = btc["vi_minus_prev"].iat[idx]
    if pd.isna(vp) or pd.isna(vm) or vm == 0:
        return False
    ratio = vp / vm
    return ratio > thresh if side.startswith("l") else ratio < 1 / thresh

FILTERS = {
    "vortex_trend": vortex_trend,
    "vortex_cross": vortex_cross,
    "vortex_strength": vortex_strength,
}

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
    os.path.join(EQUITY_DIR, "vortex_equity.csv"),
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
plt.savefig(os.path.join(PLOTS_DIR, "vortex_filters.png"), dpi=150)
plt.close()

# ============================================================
# Summary
# ============================================================
summary = []

for name in FILTERS:
    keep = []
    for _, r in trades.iterrows():
        keep.append(FILTERS[name](r["entry_idx"], r["side"]))

    trades_f = trades.iloc[[i for i, k in enumerate(keep) if k]]

    df = df_all[df_all["filter"] == name]
    s = df["equity"].values
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
        "win_rate": float(win_rate) if not np.isnan(win_rate) else np.nan,
    })

pd.DataFrame(summary).to_csv(
    os.path.join(INFO_DIR, "vortex_summary.csv"),
    index=False,
)