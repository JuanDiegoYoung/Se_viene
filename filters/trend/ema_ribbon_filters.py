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
# EMA ribbon
# ============================================================
SPANS = [5, 10, 20, 50, 100]

for s in SPANS:
    btc[f"ema_{s}"] = btc["close"].ewm(span=s, adjust=False).mean()

def strict_align(idx):
    if idx <= 0:
        return False
    v = [btc[f"ema_{s}"].iat[idx - 1] for s in SPANS]
    return all(v[i] > v[i + 1] for i in range(len(v) - 1))

def majority_align(idx):
    if idx <= 0:
        return False
    v = [btc[f"ema_{s}"].iat[idx - 1] for s in SPANS]
    return sum(v[i] > v[i + 1] for i in range(len(v) - 1)) >= 3

btc["ema_spread"] = np.mean(
    [btc[f"ema_{SPANS[i]}"] - btc[f"ema_{SPANS[i+1]}"] for i in range(len(SPANS) - 1)],
    axis=0,
)
btc["ema_spread_prev"] = btc["ema_spread"].shift(1)

# ============================================================
# Filters
# ============================================================
def f_all(idx, side):
    return strict_align(idx)

def f_majority(idx, side):
    return majority_align(idx)

def f_spread(idx, side):
    if idx <= 30:
        return False
    v = btc["ema_spread_prev"].iat[idx]
    hist = btc["ema_spread"].iloc[:idx].dropna()
    return v > hist.median()

FILTERS = {
    "ema_all_aligned": f_all,
    "ema_majority_aligned": f_majority,
    "ema_spread": f_spread,
}

# ============================================================
# Base contributions
# ============================================================
contrib_all = np.zeros(N)
for _, r in trades.iterrows():
    contrib_all[r["exit_idx"]] += r["R"]

# ============================================================
# Apply filters
# ============================================================
plt.figure(figsize=(14, 6))
plt.plot(btc["date"], eq_all, color="black", label="baseline")

combined = pd.DataFrame({"date": btc["date"], "baseline": eq_all})
summary = []

for name, fn in FILTERS.items():
    keep = [
        fn(r["entry_idx"], r["side"])
        for _, r in trades.iterrows()
    ]

    trades_f = trades.iloc[[i for i, k in enumerate(keep) if k]]

    contrib_f = np.zeros(N)
    for _, r in trades_f.iterrows():
        contrib_f[r["exit_idx"]] += r["R"]

    eq_f = eq_all - np.cumsum(contrib_all - contrib_f)

    combined[name] = eq_f
    plt.plot(btc["date"], eq_f, label=name)

    dd = np.maximum.accumulate(eq_f) - eq_f

    summary.append({
        "filter": name,
        "n_trades": len(trades_f),
        "final_equity": eq_f[-1],
        "maxdd": dd.max(),
        "win_rate": (trades_f["hit"] != "sl").mean()
    })

# ============================================================
# Save
# ============================================================
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "ema_ribbon_filters.png"), dpi=150)
plt.close()

combined.to_csv(os.path.join(EQUITY_DIR, "ema_ribbon_equity.csv"), index=False)
pd.DataFrame(summary).to_csv(os.path.join(INFO_DIR, "ema_ribbon_summary.csv"), index=False)