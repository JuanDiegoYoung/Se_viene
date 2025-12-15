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
    args.timeframe
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
# CMF
# ============================================================
PER = 20

mf_mult = (
    ((btc["close"] - btc["low"]) - (btc["high"] - btc["close"])) /
    (btc["high"] - btc["low"]).replace(0, np.nan)
)

mf = mf_mult.fillna(0) * btc["volume"]

btc["cmf"] = (
    mf.rolling(PER).sum() /
    btc["volume"].rolling(PER).sum().replace(0, np.nan)
)

btc["cmf_prev"] = btc["cmf"].shift(1)

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
# Filters
# ============================================================
def cmf_pos(idx, side):
    if idx <= 0:
        return False
    v = btc["cmf_prev"].iat[idx]
    if pd.isna(v):
        return False
    return v > 0 if side.startswith("l") else v < 0


def cmf_thresh(idx, side):
    if idx <= 0:
        return False
    v = btc["cmf_prev"].iat[idx]
    if pd.isna(v):
        return False
    return v > 0.05 if side.startswith("l") else v < -0.05


def cmf_quantile(idx, side):
    if idx <= 0:
        return False
    v = btc["cmf_prev"].iat[idx]
    if pd.isna(v):
        return False
    hist = btc["cmf"].iloc[:idx].dropna()
    if len(hist) < 30:
        return False
    q75 = hist.quantile(0.75)
    q25 = hist.quantile(0.25)
    return v > q75 if side.startswith("l") else v < q25


FILTERS = {
    "cmf_pos": cmf_pos,
    "cmf_thresh": cmf_thresh,
    "cmf_quantile": cmf_quantile,
}

# ============================================================
# Apply filters
# ============================================================
plt.figure(figsize=(14, 6))
plt.plot(btc["date"], eq_all, color="black", label="baseline")

out = pd.DataFrame({"date": btc["date"], "baseline": eq_all})
summary = []

for name, fn in FILTERS.items():
    keep = []
    for _, r in trades.iterrows():
        keep.append(fn(r["entry_idx"], r["side"]))

    trades_f = trades.iloc[[i for i, k in enumerate(keep) if k]]

    contrib_f = np.zeros(N)
    for _, r in trades_f.iterrows():
        contrib_f[r["exit_idx"]] += r["R"]

    eq_f = eq_all + np.cumsum(contrib_f)

    out[name] = eq_f
    plt.plot(btc["date"], eq_f, label=name)

    dd = np.maximum.accumulate(eq_f) - eq_f

    summary.append({
        "filter": name,
        "n_trades": len(trades_f),
        "final_equity": eq_f[-1],
        "maxdd": dd.max(),
        "win_rate": (trades_f["hit"] != "sl").mean() if len(trades_f) else None
    })

# ============================================================
# Save outputs
# ============================================================
out.to_csv(
    os.path.join(EQUITY_DIR, "cmf_filters_equity.csv"),
    index=False
)

plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "cmf_filters.png"), dpi=150)
plt.close()

pd.DataFrame(summary).to_csv(
    os.path.join(INFO_DIR, "cmf_filters_summary.csv"),
    index=False
)