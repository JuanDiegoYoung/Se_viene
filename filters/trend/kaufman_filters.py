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

N = len(btc)

# ============================================================
# Kaufman Efficiency Ratio
# ============================================================
ER_PERIOD = 10
price = btc["close"]

net = price.diff(ER_PERIOD).abs()
gross = price.diff().abs().rolling(ER_PERIOD).sum()

btc["er"] = net / gross
btc["er_prev"] = btc["er"].shift(1)

# ============================================================
# Load trades & equity
# ============================================================
trades = pd.read_csv(os.path.join(CANONICAL_DIR, "trades.csv"))
eq_df = pd.read_csv(os.path.join(CANONICAL_DIR, "equity.csv"))

eq_col = [c for c in eq_df.columns if "equity" in c.lower()][0]
eq_all = eq_df[eq_col].astype(float).values[:N]

for c in ["entry_idx", "exit_idx"]:
    trades[c] = trades[c].astype(int)

trades["side"] = trades["side"].astype(str).str.lower()

# ============================================================
# Baseline contributions
# ============================================================
contrib_all = np.zeros(N)

for _, r in trades.iterrows():
    ex = max(0, min(int(r["exit_idx"]), N - 1))
    contrib_all[ex] += float(r["R"])

# ============================================================
# Filters
# ============================================================
def er_trend(idx, side, thr):
    if idx <= ER_PERIOD:
        return False
    v = btc["er_prev"].iat[idx]
    if pd.isna(v):
        return False
    return v > thr


ER_GRID = [0.25, 0.35, 0.40, 0.45]

filters = {
    f"er_trend_{v:.2f}": (lambda idx, side, v=v: er_trend(idx, side, v))
    for v in ER_GRID
}

# ============================================================
# Apply filters
# ============================================================
summary = []

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(btc["date"], eq_all, label="baseline", color="black", linewidth=1.5)

combined = pd.DataFrame({
    "date": btc["date"],
    "baseline": eq_all,
})

for name, fn in filters.items():
    keep = []

    for _, r in trades.iterrows():
        idx = max(0, min(int(r["entry_idx"]), N - 1))
        keep.append(fn(idx, r["side"]))

    trades_f = trades.iloc[[i for i, k in enumerate(keep) if k]]

    contrib_f = np.zeros(N)
    for _, r in trades_f.iterrows():
        ex = max(0, min(int(r["exit_idx"]), N - 1))
        contrib_f[ex] += float(r["R"])

    removed = contrib_all - contrib_f
    eq_filt = eq_all - np.cumsum(removed)

    df_eq = pd.DataFrame({
        "date": btc["date"],
        "equity": eq_filt,
    })

    df_eq.to_csv(
        os.path.join(EQUITY_DIR, f"kaufman_{name}.csv"),
        index=False,
    )

    ax.plot(btc["date"], eq_filt, label=name)
    combined[name] = eq_filt

    s = pd.Series(eq_filt).ffill().values
    dd = np.maximum.accumulate(s) - s

    summary.append({
        "filter": name,
        "n_trades": len(trades_f),
        "final_equity": eq_filt[-1],
        "maxdd": float(np.nanmax(dd)),
        "win_rate": (trades_f["hit"] != "sl").mean(),
    })

ax.legend(ncol=2)
ax.set_title("Kaufman ER filters")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "kaufman_filters.png"), dpi=150)
plt.close()

combined.to_csv(
    os.path.join(EQUITY_DIR, "kaufman_filters_all.csv"),
    index=False,
)

pd.DataFrame(summary).to_csv(
    os.path.join(INFO_DIR, "kaufman_filters_summary.csv"),
    index=False,
)