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
# Volume features
# ============================================================
btc["vol_prev"] = btc["volume"].shift(1)
btc["vol_ma20"] = btc["volume"].rolling(20).mean().shift(1)

btc["ret_prev"] = btc["close"].pct_change().shift(1)
btc["ema20"] = btc["close"].ewm(span=20, adjust=False).mean().shift(1)

# ============================================================
# Canonical trades & equity
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
def vol_spike_raw(idx, side, mult=1.5):
    v = btc["vol_prev"].iat[idx]
    m = btc["vol_ma20"].iat[idx]
    if pd.isna(v) or pd.isna(m):
        return False
    return v > m * mult

def vol_spike_breakout(idx, side, mult=1.5):
    v = btc["vol_prev"].iat[idx]
    m = btc["vol_ma20"].iat[idx]
    r = btc["ret_prev"].iat[idx]
    if pd.isna(v) or pd.isna(m) or pd.isna(r):
        return False
    return (v > m * mult) and (r > 0 if side.startswith("l") else r < 0)

def vol_spike_trend(idx, side, mult=1.5):
    v = btc["vol_prev"].iat[idx]
    m = btc["vol_ma20"].iat[idx]
    c = btc["close"].iat[idx - 1]
    e = btc["ema20"].iat[idx]
    if pd.isna(v) or pd.isna(m) or pd.isna(c) or pd.isna(e):
        return False
    return (v > m * mult) and (c > e if side.startswith("l") else c < e)

FILTERS = {
    "vol_spike_raw": vol_spike_raw,
    "vol_spike_breakout": vol_spike_breakout,
    "vol_spike_trend": vol_spike_trend,
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
    os.path.join(EQUITY_DIR, "volume_spike_equity.csv"),
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
plt.savefig(os.path.join(PLOTS_DIR, "volume_spike_filters.png"), dpi=150)
plt.close()

# ============================================================
# Summary (con win_rate y n_trades correctos)
# ============================================================
summary = []

for name in FILTERS:
    df = df_all[df_all["filter"] == name]
    s = df["equity"].values
    dd = np.maximum.accumulate(s) - s

    # trades realmente usados por el filtro
    keep = []
    for _, r in trades.iterrows():
        keep.append(FILTERS[name](r["entry_idx"], r["side"]))
    trades_f = trades.iloc[[i for i, k in enumerate(keep) if k]]

    win_rate = (
        float((trades_f["hit"].astype(str).str.lower() != "sl").mean())
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
    os.path.join(INFO_DIR, "volume_spike_summary.csv"),
    index=False,
)