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

if "open_time" in btc.columns:
    btc["date"] = pd.to_datetime(btc["open_time"], unit="ms")
else:
    btc["date"] = pd.to_datetime(btc["date"])

btc = btc.sort_values("date").reset_index(drop=True)
for c in ["open", "high", "low", "close", "volume"]:
    btc[c] = pd.to_numeric(btc[c], errors="coerce")

N = len(btc)

# ============================================================
# RSI(14)
# ============================================================
delta = btc["close"].diff()
up = delta.clip(lower=0)
down = -delta.clip(upper=0)

ma_up = up.ewm(alpha=1/14, adjust=False).mean()
ma_down = down.ewm(alpha=1/14, adjust=False).mean()

btc["rsi_prev"] = (100 - (100 / (1 + ma_up / ma_down))).shift(1)

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
def rsi_center_50(idx, side):
    r = btc["rsi_prev"].iat[idx]
    if pd.isna(r):
        return False
    return r > 50 if side.startswith("l") else r < 50

def rsi_band_48_52(idx, side):
    r = btc["rsi_prev"].iat[idx]
    if pd.isna(r):
        return False
    return r > 52 if side.startswith("l") else r < 48

def rsi_band_46_54(idx, side):
    r = btc["rsi_prev"].iat[idx]
    if pd.isna(r):
        return False
    return r > 54 if side.startswith("l") else r < 46

FILTERS = {
    "rsi_center_50": rsi_center_50,
    "rsi_band_48_52": rsi_band_48_52,
    "rsi_band_46_54": rsi_band_46_54,
}

# ============================================================
# Apply filters (long format)
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

df_all.to_csv(
    os.path.join(EQUITY_DIR, "rsi_equity.csv"),
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
plt.savefig(os.path.join(PLOTS_DIR, "rsi_filters.png"), dpi=150)
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

    if len(trades_f) == 0:
        win_rate = np.nan
    else:
        win_rate = float((trades_f["hit"] != "sl").mean())

    s = df_all[df_all["filter"] == name]["equity"].values
    dd = np.maximum.accumulate(s) - s

    summary.append({
        "filter": name,
        "final_equity": float(s[-1]),
        "maxdd": float(dd.max()),
        "n_trades": int(len(trades_f)),
        "win_rate": win_rate,
    })

pd.DataFrame(summary).to_csv(
    os.path.join(INFO_DIR, "rsi_summary.csv"),
    index=False,
)