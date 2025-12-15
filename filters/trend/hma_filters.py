#!/usr/bin/env python3
import argparse
import os
import numpy as np
import pandas as pd
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
# Paths (new structure)
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
# HMA helpers
# ============================================================
def wma(series, period):
    weights = np.arange(1, period + 1)
    return series.rolling(period).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True
    )

def hma(series, period):
    half = int(period / 2)
    sqrtp = int(np.sqrt(period))
    return wma(2 * wma(series, half) - wma(series, period), sqrtp)

# ============================================================
# Indicators
# ============================================================
btc["hma_21"] = hma(btc["close"], 21)
btc["hma_55"] = hma(btc["close"], 55)
btc["hma_89"] = hma(btc["close"], 89)

for c in ["hma_21", "hma_55", "hma_89"]:
    btc[f"{c}_prev"] = btc[c].shift(1)
    btc[f"{c}_prev2"] = btc[c].shift(2)

# ============================================================
# Filters
# ============================================================
def hma_slope_pos(idx, side):
    if idx <= 1:
        return False
    v1 = btc["hma_55_prev"].iat[idx]
    v2 = btc["hma_55_prev2"].iat[idx]
    if np.isnan(v1) or np.isnan(v2):
        return False
    return (v1 - v2) > 0 if side.startswith("l") else (v1 - v2) < 0

def hma_price_above(idx, side):
    if idx <= 0:
        return False
    c_prev = btc["close"].iat[idx - 1]
    h_prev = btc["hma_55_prev"].iat[idx]
    if np.isnan(c_prev) or np.isnan(h_prev):
        return False
    return c_prev > h_prev if side.startswith("l") else c_prev < h_prev

def hma_ribbon(idx, side):
    if idx <= 0:
        return False
    f = btc["hma_21_prev"].iat[idx]
    m = btc["hma_55_prev"].iat[idx]
    s = btc["hma_89_prev"].iat[idx]
    if np.isnan(f) or np.isnan(m) or np.isnan(s):
        return False
    return (f > m > s) if side.startswith("l") else (f < m < s)

FILTERS = {
    "hma_slope_pos": hma_slope_pos,
    "hma_price_above": hma_price_above,
    "hma_ribbon": hma_ribbon,
}

# ============================================================
# Apply filters
# ============================================================
contrib_all = np.zeros(N)
for _, r in trades.iterrows():
    contrib_all[r["exit_idx"]] += r["R"]

combined = pd.DataFrame({"date": btc["date"], "baseline": eq_all})
summary = []

plt.figure(figsize=(14, 6))
plt.plot(btc["date"], eq_all, label="baseline", color="black", linewidth=1.5)

for name, fn in FILTERS.items():
    keep_mask = []

    for _, r in trades.iterrows():
        keep_mask.append(fn(r["entry_idx"], r["side"]))

    trades_f = trades.loc[keep_mask]

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
        "win_rate": (trades_f["hit"] != "sl").mean(),
        "maxdd": float(np.nanmax(dd)),
        "final_equity": float(eq_f[-1]),
    })

# ============================================================
# Save outputs
# ============================================================
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "hma_filters.png"), dpi=150)
plt.close()

combined.to_csv(
    os.path.join(EQUITY_DIR, "hma_filters_equity.csv"),
    index=False,
)

pd.DataFrame(summary).to_csv(
    os.path.join(INFO_DIR, "hma_filters_summary.csv"),
    index=False,
)