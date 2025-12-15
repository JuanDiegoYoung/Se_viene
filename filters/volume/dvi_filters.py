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

DATA_DIR = os.path.join(
    PROJECT_ROOT,
    "candle_data",
    args.asset,
    args.timeframe,
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
    "CANONICAL_DIR",
    os.path.join(EXP_DIR, "canonical_output"),
)
OUT_DIR = os.environ.get(
    "OUT_DIR",
    os.path.join(EXP_DIR, "iteration_1"),
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
if not csvs:
    raise SystemExit(f"No candles found in {DATA_DIR}")

btc = pd.read_csv(os.path.join(DATA_DIR, csvs[0]))

if "open_time" in btc.columns:
    btc["date"] = pd.to_datetime(btc["open_time"], unit="ms")
elif "date" in btc.columns:
    btc["date"] = pd.to_datetime(btc["date"])
else:
    btc["date"] = pd.to_datetime(btc.iloc[:, 0])

btc = btc.sort_values("date").reset_index(drop=True)

for c in ["open", "high", "low", "close", "volume"]:
    btc[c] = pd.to_numeric(btc[c], errors="coerce")

N = len(btc)

# ============================================================
# Load canonical trades & equity
# ============================================================
trades = pd.read_csv(os.path.join(CANONICAL_DIR, "trades.csv"))
eq_df = pd.read_csv(os.path.join(CANONICAL_DIR, "equity.csv"))

eq_col = [c for c in eq_df.columns if "equity" in c.lower()][0]
eq_all = eq_df[eq_col].astype(float).values[:N]

for c in ["entry_idx", "exit_idx"]:
    trades[c] = trades[c].astype(int)

trades["side"] = trades["side"].astype(str).str.lower()

# ============================================================
# Directional Volume Imbalance
# ============================================================
DVI_PERIOD = 10

up_vol = np.where(btc["close"] > btc["open"], btc["volume"], 0.0)
down_vol = np.where(btc["close"] < btc["open"], btc["volume"], 0.0)

btc["up_vol_sum"] = pd.Series(up_vol).rolling(DVI_PERIOD).sum()
btc["down_vol_sum"] = pd.Series(down_vol).rolling(DVI_PERIOD).sum()

den = btc["up_vol_sum"] + btc["down_vol_sum"]
btc["up_ratio"] = btc["up_vol_sum"] / den
btc["down_ratio"] = btc["down_vol_sum"] / den

btc["up_ratio_prev"] = btc["up_ratio"].shift(1)
btc["down_ratio_prev"] = btc["down_ratio"].shift(1)

# ============================================================
# Contribution baseline
# ============================================================
contrib_all = np.zeros(N)
for _, r in trades.iterrows():
    ex = max(0, min(r["exit_idx"], N - 1))
    contrib_all[ex] += r["R"]

# ============================================================
# Filters
# ============================================================
def dvi_long(idx, side, thr):
    if idx <= DVI_PERIOD:
        return False
    v = btc["up_ratio_prev"].iat[idx]
    return (v > thr) if side.startswith("l") else True

def dvi_short(idx, side, thr):
    if idx <= DVI_PERIOD:
        return False
    v = btc["down_ratio_prev"].iat[idx]
    return (v > thr) if side.startswith("s") else True

DVI_GRID = [0.55, 0.60, 0.65, 0.70]

FILTERS = {}
for v in DVI_GRID:
    FILTERS[f"dvi_long_{v:.2f}"] = lambda i, s, v=v: dvi_long(i, s, v)
    FILTERS[f"dvi_short_{v:.2f}"] = lambda i, s, v=v: dvi_short(i, s, v)

# ============================================================
# Apply filters
# ============================================================
combined = pd.DataFrame({
    "date": btc["date"],
    "baseline": eq_all,
})

summary = []

plt.figure(figsize=(14, 6))
plt.plot(btc["date"], eq_all, label="baseline", color="black")

for name, fn in FILTERS.items():
    keep = []
    for _, r in trades.iterrows():
        idx = max(0, min(r["entry_idx"], N - 1))
        keep.append(fn(idx, r["side"]))

    trades_f = trades.iloc[[i for i, k in enumerate(keep) if k]]

    contrib_f = np.zeros(N)
    for _, r in trades_f.iterrows():
        ex = max(0, min(r["exit_idx"], N - 1))
        contrib_f[ex] += r["R"]

    eq_f = eq_all - np.cumsum(contrib_all - contrib_f)

    combined[name] = eq_f
    plt.plot(btc["date"], eq_f, label=name)

    s = pd.Series(eq_f).ffill().fillna(0).values
    dd = np.maximum.accumulate(s) - s

    summary.append({
        "filter": name,
        "n_trades": len(trades_f),
        "win_rate": (trades_f["hit"] != "sl").mean(),
        "maxdd": float(dd.max()),
        "final_equity": float(eq_f[-1]),
    })

# ============================================================
# Save outputs
# ============================================================
plt.legend(ncol=2)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "dvi_filters.png"), dpi=150)

combined.to_csv(
    os.path.join(EQUITY_DIR, "dvi_filters_equity.csv"),
    index=False,
)

pd.DataFrame(summary).to_csv(
    os.path.join(INFO_DIR, "dvi_filters_summary.csv"),
    index=False,
)