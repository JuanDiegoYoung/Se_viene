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
# Accept extra strategy flags for compatibility and support explicit no-... flags
p.add_argument("--require-prior-swing", dest="require_prior_swing", action="store_true")
p.add_argument("--no-require-prior-swing", dest="require_prior_swing", action="store_false")
p.set_defaults(require_prior_swing=True)
p.add_argument("--allow-countertrend", dest="allow_countertrend", action="store_true")
p.add_argument("--no-allow-countertrend", dest="allow_countertrend", action="store_false")
p.set_defaults(allow_countertrend=False)
p.add_argument("--allow-micro-structure", dest="allow_micro_structure", action="store_true")
p.add_argument("--no-allow-micro-structure", dest="allow_micro_structure", action="store_false")
p.set_defaults(allow_micro_structure=True)
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

btc["date"] = pd.to_datetime(btc["open_time"], unit="ms") if "open_time" in btc.columns else pd.to_datetime(btc["date"])
btc = btc.sort_values("date").reset_index(drop=True)

for c in ["open", "high", "low", "close", "volume"]:
    btc[c] = pd.to_numeric(btc[c], errors="coerce")

N = len(btc)

# ============================================================
# ATR
# ============================================================
tr = pd.concat([
    btc["high"] - btc["low"],
    (btc["high"] - btc["close"].shift(1)).abs(),
    (btc["low"] - btc["close"].shift(1)).abs(),
], axis=1).max(axis=1)

btc["atr"] = tr.rolling(14).mean()
btc["atr_prev"] = btc["atr"].shift(1)

# ============================================================
# PSAR
# ============================================================
af_start = 0.02
af_step = 0.02
af_max = 0.2

psar = np.zeros(N)
trend = np.ones(N)

ep = btc["high"].iloc[0]
af = af_start
psar[0] = btc["low"].iloc[0]

for i in range(1, N):
    prev_psar = psar[i - 1]
    prev_trend = trend[i - 1]

    if prev_trend > 0:
        psar[i] = prev_psar + af * (ep - prev_psar)
        psar[i] = min(psar[i], btc["low"].iloc[i - 1], btc["low"].iloc[i])
        if btc["low"].iloc[i] < psar[i]:
            trend[i] = -1
            psar[i] = ep
            ep = btc["low"].iloc[i]
            af = af_start
        else:
            trend[i] = 1
            if btc["high"].iloc[i] > ep:
                ep = btc["high"].iloc[i]
                af = min(af + af_step, af_max)
    else:
        psar[i] = prev_psar + af * (ep - prev_psar)
        psar[i] = max(psar[i], btc["high"].iloc[i - 1], btc["high"].iloc[i])
        if btc["high"].iloc[i] > psar[i]:
            trend[i] = 1
            psar[i] = ep
            ep = btc["high"].iloc[i]
            af = af_start
        else:
            trend[i] = -1
            if btc["low"].iloc[i] < ep:
                ep = btc["low"].iloc[i]
                af = min(af + af_step, af_max)

btc["psar_prev"] = pd.Series(psar).shift(1)
btc["psar_prev2"] = pd.Series(psar).shift(2)
btc["close_prev"] = btc["close"].shift(1)
btc["close_prev2"] = btc["close"].shift(2)

# ============================================================
# Load canonical
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
def psar_trend(idx, side):
    c = btc["close_prev"].iat[idx]
    p = btc["psar_prev"].iat[idx]
    return c > p if side.startswith("l") else c < p

def psar_confirmed(idx, side):
    c1 = btc["close_prev"].iat[idx]
    c2 = btc["close_prev2"].iat[idx]
    p1 = btc["psar_prev"].iat[idx]
    p2 = btc["psar_prev2"].iat[idx]
    return (c1 > p1 and c2 > p2) if side.startswith("l") else (c1 < p1 and c2 < p2)

def psar_strength(idx, side, thresh=0.5):
    c = btc["close_prev"].iat[idx]
    p = btc["psar_prev"].iat[idx]
    atr = btc["atr_prev"].iat[idx]
    if atr == 0 or pd.isna(atr):
        return False
    v = (c - p) / atr
    return v > thresh if side.startswith("l") else v < -thresh

FILTERS = {
    "psar_trend": psar_trend,
    "psar_confirmed": psar_confirmed,
    "psar_strength": psar_strength,
}

# ============================================================
# Apply
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
df_all.to_csv(os.path.join(EQUITY_DIR, "psar_equity.csv"), index=False)

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
plt.savefig(os.path.join(PLOTS_DIR, "psar_filters.png"), dpi=150)
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

    win_rate = None
    if "hit" in trades_f.columns and len(trades_f) > 0:
        win_rate = float((trades_f["hit"].astype(str).str.lower() != "sl").mean())

    summary.append({
        "filter": name,
        "final_equity": float(s[-1]),
        "maxdd": float(dd.max()),
        "n_trades": int(len(trades_f)),
        "win_rate": win_rate,
    })

pd.DataFrame(summary).to_csv(
    os.path.join(INFO_DIR, "ichimoku_summary.csv"),
    index=False,
)