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
    raise FileNotFoundError(f"No CSV files in {DATA_DIR}")

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
# Klinger Oscillator (simplificado)
# ============================================================
high = btc["high"]
low = btc["low"]
close = btc["close"]
volume = btc["volume"]

trend = np.where(close > close.shift(1), 1, -1)
trend = pd.Series(trend, index=btc.index)

dm = high - low
dm_prev = dm.shift(1)
cm = dm.where(trend == trend.shift(1), dm_prev).cumsum()

vf = volume * trend * np.abs(2 * (dm / cm - 1))
vf = vf.replace([np.inf, -np.inf], np.nan).fillna(0)

btc["klinger_fast"] = vf.ewm(span=34, adjust=False).mean()
btc["klinger_slow"] = vf.ewm(span=55, adjust=False).mean()
btc["klinger"] = btc["klinger_fast"] - btc["klinger_slow"]

btc["klinger_prev"] = btc["klinger"].shift(1)
btc["klinger_prev2"] = btc["klinger"].shift(2)

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
def klinger_above_zero(idx, side):
    v = btc["klinger_prev"].iat[idx]
    if pd.isna(v):
        return False
    return v > 0 if side.startswith("l") else v < 0

def klinger_cross(idx, side):
    v1 = btc["klinger_prev"].iat[idx]
    v2 = btc["klinger_prev2"].iat[idx]
    if pd.isna(v1) or pd.isna(v2):
        return False
    return (v1 > 0 and v2 <= 0) if side.startswith("l") else (v1 < 0 and v2 >= 0)

def klinger_trend(idx, side):
    v1 = btc["klinger_prev"].iat[idx]
    v2 = btc["klinger_prev2"].iat[idx]
    if pd.isna(v1) or pd.isna(v2):
        return False
    return v1 > v2 if side.startswith("l") else v1 < v2

FILTERS = {
    "klinger_above_zero": klinger_above_zero,
    "klinger_cross": klinger_cross,
    "klinger_trend": klinger_trend,
}

# ============================================================
# Apply filters
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
    os.path.join(EQUITY_DIR, "klinger_equity.csv"),
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
plt.savefig(os.path.join(PLOTS_DIR, "klinger_filters.png"), dpi=150)
plt.close()

# ============================================================
# Summary
# ============================================================
summary = []

for name in FILTERS:
    df = df_all[df_all["filter"] == name]
    s = df["equity"].values
    dd = np.maximum.accumulate(s) - s

    # trades filtrados para este filtro
    keep = [FILTERS[name](r["entry_idx"], r["side"]) for _, r in trades.iterrows()]
    trades_f = trades.iloc[[i for i, k in enumerate(keep) if k]]

    if len(trades_f) > 0 and "hit" in trades_f.columns:
        win_rate = float((trades_f["hit"].astype(str).str.lower() != "sl").mean())
    else:
        win_rate = np.nan

    summary.append({
        "filter": name,
        "final_equity": float(s[-1]),
        "maxdd": float(dd.max()),
        "n_trades": int(len(trades_f)),
        "win_rate": win_rate,
    })

pd.DataFrame(summary).to_csv(
    os.path.join(INFO_DIR, "klinger_summary.csv"),
    index=False,
)