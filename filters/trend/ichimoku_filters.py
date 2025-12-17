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
# Ichimoku
# ============================================================
high = btc["high"]
low = btc["low"]
close = btc["close"]

tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
senkou_a = ((tenkan + kijun) / 2).shift(26)
senkou_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)

btc["tenkan_prev"] = tenkan.shift(1)
btc["kijun_prev"] = kijun.shift(1)
btc["tenkan_prev2"] = tenkan.shift(2)
btc["kijun_prev2"] = kijun.shift(2)
btc["senkou_a_prev"] = senkou_a.shift(1)
btc["senkou_b_prev"] = senkou_b.shift(1)
btc["close_prev"] = close.shift(1)

# ============================================================
# Load canonical trades & equity
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
def ichimoku_above_cloud(idx, side):
    c = btc["close_prev"].iat[idx]
    a = btc["senkou_a_prev"].iat[idx]
    b = btc["senkou_b_prev"].iat[idx]
    if pd.isna(c) or pd.isna(a) or pd.isna(b):
        return False
    top, bot = max(a, b), min(a, b)
    return c > top if side.startswith("l") else c < bot

def ichimoku_tk_cross(idx, side):
    t1 = btc["tenkan_prev"].iat[idx]
    k1 = btc["kijun_prev"].iat[idx]
    t2 = btc["tenkan_prev2"].iat[idx]
    k2 = btc["kijun_prev2"].iat[idx]
    if pd.isna(t1) or pd.isna(k1) or pd.isna(t2) or pd.isna(k2):
        return False
    return (t1 > k1 and t2 <= k2) if side.startswith("l") else (t1 < k1 and t2 >= k2)

def ichimoku_trend(idx, side):
    t = btc["tenkan_prev"].iat[idx]
    k = btc["kijun_prev"].iat[idx]
    c = btc["close_prev"].iat[idx]
    a = btc["senkou_a_prev"].iat[idx]
    b = btc["senkou_b_prev"].iat[idx]
    if pd.isna(t) or pd.isna(k) or pd.isna(c) or pd.isna(a) or pd.isna(b):
        return False
    top, bot = max(a, b), min(a, b)
    return (t > k and c > top) if side.startswith("l") else (t < k and c < bot)

FILTERS = {
    "ichimoku_above_cloud": ichimoku_above_cloud,
    "ichimoku_tk_cross": ichimoku_tk_cross,
    "ichimoku_trend": ichimoku_trend,
}

# ============================================================
# Apply filters
# ============================================================
rows = []
summary = []

for name, fn in FILTERS.items():
    keep = []
    contrib_filt = np.zeros(N)

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

    s = pd.Series(eq_filt).ffill().fillna(0).values
    dd = np.maximum.accumulate(s) - s

    if "hit" in trades_f.columns and len(trades_f) > 0:
        win_rate = (trades_f["hit"].astype(str).str.lower() != "sl").mean()
    else:
        win_rate = np.nan

    summary.append({
        "filter": name,
        "final_equity": float(s[-1]),
        "maxdd": float(dd.max()),
        "n_trades": int(len(trades_f)),
        "win_rate": float(win_rate) if not np.isnan(win_rate) else np.nan,
    })

df_all = pd.concat(rows, ignore_index=True)

df_all.to_csv(
    os.path.join(EQUITY_DIR, "ichimoku_equity.csv"),
    index=False,
)

pd.DataFrame(summary).to_csv(
    os.path.join(INFO_DIR, "ichimoku_summary.csv"),
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
plt.savefig(os.path.join(PLOTS_DIR, "ichimoku_filters.png"), dpi=150)
plt.close()