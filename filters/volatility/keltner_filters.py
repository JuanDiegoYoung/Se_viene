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
# Keltner Channel
# ============================================================
EMA_PER = 20
ATR_PER = 20
MULT = 1.5

btc["ema"] = btc["close"].ewm(span=EMA_PER, adjust=False).mean()

tr = pd.concat([
    btc["high"] - btc["low"],
    (btc["high"] - btc["close"].shift(1)).abs(),
    (btc["low"] - btc["close"].shift(1)).abs(),
], axis=1).max(axis=1)

btc["atr"] = tr.rolling(ATR_PER).mean()

btc["kc_up"] = btc["ema"] + MULT * btc["atr"]
btc["kc_dn"] = btc["ema"] - MULT * btc["atr"]
btc["kc_w"] = btc["kc_up"] - btc["kc_dn"]

btc["close_prev"] = btc["close"].shift(1)
btc["kc_up_prev"] = btc["kc_up"].shift(1)
btc["kc_dn_prev"] = btc["kc_dn"].shift(1)
btc["kc_w_prev"] = btc["kc_w"].shift(1)

btc["close_prev2"] = btc["close"].shift(2)
btc["kc_up_prev2"] = btc["kc_up"].shift(2)
btc["kc_dn_prev2"] = btc["kc_dn"].shift(2)

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
def keltner_breakout(idx, side):
    c = btc["close_prev"].iat[idx]
    up = btc["kc_up_prev"].iat[idx]
    dn = btc["kc_dn_prev"].iat[idx]
    if pd.isna(c) or pd.isna(up) or pd.isna(dn):
        return False
    return c > up if side.startswith("l") else c < dn

def keltner_confirmed(idx, side):
    c1 = btc["close_prev"].iat[idx]
    c2 = btc["close_prev2"].iat[idx]
    up1 = btc["kc_up_prev"].iat[idx]
    up2 = btc["kc_up_prev2"].iat[idx]
    dn1 = btc["kc_dn_prev"].iat[idx]
    dn2 = btc["kc_dn_prev2"].iat[idx]
    if pd.isna(c1) or pd.isna(c2):
        return False
    return (c1 > up1 and c2 > up2) if side.startswith("l") else (c1 < dn1 and c2 < dn2)

def keltner_wide(idx, side):
    w = btc["kc_w_prev"].iat[idx]
    hist = btc["kc_w"].iloc[:idx].dropna()
    if pd.isna(w) or len(hist) < 20:
        return False
    return w > hist.median()

FILTERS = {
    "keltner_breakout": keltner_breakout,
    "keltner_confirmed": keltner_confirmed,
    "keltner_wide": keltner_wide,
}

# ============================================================
# Apply filters
# ============================================================
rows = []

for name, fn in FILTERS.items():
    contrib_filt = np.zeros(N)

    for _, r in trades.iterrows():
        if fn(r["entry_idx"], r["side"]):
            contrib_filt[r["exit_idx"]] += r["R"]

    eq_filt = eq_all - np.cumsum(contrib_all - contrib_filt)

    rows.append(pd.DataFrame({
        "date": btc["date"],
        "filter": name,
        "equity": eq_filt,
    }))

df_all = pd.concat(rows, ignore_index=True)

df_all.to_csv(
    os.path.join(EQUITY_DIR, "keltner_equity.csv"),
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
plt.savefig(os.path.join(PLOTS_DIR, "keltner_filters.png"), dpi=150)