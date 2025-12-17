#!/usr/bin/env python3
import os
import argparse
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
    raise SystemExit(f"No candles in {DATA_DIR}")

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

dates = btc["date"].values
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
# EMA slopes (varias combinaciones razonables)
# ============================================================
EMA_SPANS = [10, 20, 50, 100]

for span in EMA_SPANS:
    btc[f"ema_{span}"] = btc["close"].ewm(
        span=span,
        adjust=False,
        min_periods=span
    ).mean()
    btc[f"ema_{span}_slope_prev"] = btc[f"ema_{span}"].diff().shift(1)

# ============================================================
# Filters
# ============================================================
def ema_slope_filter(span):
    col = f"ema_{span}_slope_prev"

    def fn(idx, side):
        if idx <= 0:
            return False
        v = btc[col].iat[idx]
        if pd.isna(v):
            return False
        return v > 0 if side.startswith("l") else v < 0

    return fn

FILTERS = {
    f"ema_slope_{span}": ema_slope_filter(span)
    for span in EMA_SPANS
}

# ============================================================
# Baseline contribution
# ============================================================
contrib_all = np.zeros(N)
for _, r in trades.iterrows():
    ex = max(0, min(r["exit_idx"], N - 1))
    contrib_all[ex] += float(r["R"])

# ============================================================
# Plot (UNA sola figura)
# ============================================================
plt.figure(figsize=(14, 6))
plt.plot(dates, eq_all, color="black", linewidth=1.5, label="baseline")

combined = {"date": dates, "baseline": eq_all}
summary = []

# ============================================================
# Apply filters
# ============================================================
for name, fn in FILTERS.items():
    keep = []

    for _, r in trades.iterrows():
        idx = max(0, min(r["entry_idx"], N - 1))
        keep.append(fn(idx, r["side"]))

    trades_f = trades.iloc[np.where(keep)[0]]

    contrib_f = np.zeros(N)
    for _, r in trades_f.iterrows():
        ex = max(0, min(r["exit_idx"], N - 1))
        contrib_f[ex] += float(r["R"])

    eq_f = eq_all - np.cumsum(contrib_all - contrib_f)

    combined[name] = eq_f

    plt.plot(dates, eq_f, label=name)

    s = pd.Series(eq_f).ffill().fillna(0).values
    dd = np.maximum.accumulate(s) - s

    summary.append({
        "filter": name,
        "n_trades": len(trades_f),
        "win_rate": float((trades_f["hit"] != "sl").mean()) if len(trades_f) else None,
        "maxdd": float(dd.max()),
        "final_equity": float(eq_f[-1]),
    })

# ============================================================
# Save outputs
# ============================================================
plt.legend(ncol=2)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "ema_filters.png"), dpi=150)
plt.close()

pd.DataFrame(combined).to_csv(
    os.path.join(EQUITY_DIR, "ema_filters_equity.csv"),
    index=False,
)

pd.DataFrame(summary).to_csv(
    os.path.join(INFO_DIR, "ema_filters_summary.csv"),
    index=False,
)