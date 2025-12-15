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
# Paths (FIX CLAVE)
# ============================================================
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

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

contrib_all = np.zeros(N)
for _, r in trades.iterrows():
    contrib_all[r["exit_idx"]] += r["R"]

# ============================================================
# Indicators (igual que antes)
# ============================================================
period = 20
sma = btc["close"].rolling(period).mean()
std = btc["close"].rolling(period).std()

btc["bb_mid"] = sma
btc["bb_upper"] = sma + 2 * std
btc["bb_lower"] = sma - 2 * std
btc["bb_width"] = btc["bb_upper"] - btc["bb_lower"]

for c in ["close", "bb_upper", "bb_lower", "bb_width"]:
    btc[f"{c}_prev"] = btc[c].shift(1)
    btc[f"{c}_prev2"] = btc[c].shift(2)

btc["ema20"] = btc["close"].ewm(span=20).mean()
btc["ema50"] = btc["close"].ewm(span=50).mean()

btc["ema20_prev"] = btc["ema20"].shift(1)
btc["ema20_prev2"] = btc["ema20"].shift(2)
btc["ema50_prev"] = btc["ema50"].shift(1)
btc["ema50_prev2"] = btc["ema50"].shift(2)

btc["volume_prev"] = btc["volume"].shift(1)

# ============================================================
# Filters
# ============================================================
def bb_breakout(idx, side):
    if idx <= 0:
        return False
    c = btc["close_prev"].iat[idx]
    up = btc["bb_upper_prev"].iat[idx]
    lo = btc["bb_lower_prev"].iat[idx]
    return c > up if side.startswith("l") else c < lo

def bb_squeeze_after(idx, side):
    if idx <= 10:
        return False
    hist = btc["bb_width"].iloc[idx-10:idx]
    if hist.dropna().empty:
        return False
    return btc["bb_width_prev"].iat[idx] < hist.quantile(0.10)

FILTERS = {
    "bb_breakout": bb_breakout,
    "bb_squeeze_after": bb_squeeze_after,
}

# ============================================================
# Apply
# ============================================================
summary = []
equity_df = pd.DataFrame({"date": btc["date"], "baseline": eq_all})

plt.figure(figsize=(14, 6))
plt.plot(btc["date"], eq_all, color="black", label="baseline")

for name, fn in FILTERS.items():
    keep = [fn(r["entry_idx"], r["side"]) for _, r in trades.iterrows()]
    trades_f = trades.iloc[[i for i, k in enumerate(keep) if k]]

    contrib_f = np.zeros(N)
    for _, r in trades_f.iterrows():
        contrib_f[r["exit_idx"]] += r["R"]

    eq_f = eq_all - np.cumsum(contrib_all - contrib_f)

    equity_df[name] = eq_f
    plt.plot(btc["date"], eq_f, label=name)

    s = pd.Series(eq_f).ffill().values
    dd = np.maximum.accumulate(s) - s

    summary.append({
        "filter": name,
        "final_equity": float(eq_f[-1]),
        "maxdd": float(dd.max()),
        "n_trades": len(trades_f),
    })

# ============================================================
# Summary (con win_rate y nombre estándar)
# ============================================================
summary = []

for name, fn in FILTERS.items():
    keep = [fn(r["entry_idx"], r["side"]) for _, r in trades.iterrows()]
    trades_f = trades.iloc[[i for i, k in enumerate(keep) if k]]

    contrib_f = np.zeros(N)
    for _, r in trades_f.iterrows():
        contrib_f[r["exit_idx"]] += r["R"]

    eq_f = eq_all - np.cumsum(contrib_all - contrib_f)

    s = pd.Series(eq_f).ffill().values
    dd = np.maximum.accumulate(s) - s

    win_rate = (
        (trades_f["hit"].astype(str).str.lower() != "sl").mean()
        if len(trades_f) > 0 and "hit" in trades_f.columns
        else np.nan
    )

    summary.append({
        "filter": name,
        "final_equity": float(eq_f[-1]),
        "maxdd": float(dd.max()),
        "n_trades": int(len(trades_f)),
        "win_rate": float(win_rate) if not np.isnan(win_rate) else np.nan,
    })

pd.DataFrame(summary).to_csv(
    os.path.join(INFO_DIR, "bb_filters_summary.csv"),
    index=False,
)