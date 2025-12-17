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
p.add_argument("--require-prior-swing", dest="require_prior_swing", action="store_true")
p.add_argument("--no-require-prior-swing", dest="require_prior_swing", action="store_false")
p.add_argument("--allow-countertrend", dest="allow_countertrend", action="store_true")
p.add_argument("--no-allow-countertrend", dest="allow_countertrend", action="store_false")
p.add_argument("--allow-micro-structure", dest="allow_micro_structure", action="store_true")
p.add_argument("--no-allow-micro-structure", dest="allow_micro_structure", action="store_false")
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
# Load canonical trades & equity
# ============================================================
trades = pd.read_csv(os.path.join(CANONICAL_DIR, "trades.csv"))
eq_df = pd.read_csv(os.path.join(CANONICAL_DIR, "equity.csv"))
eq_col = [c for c in eq_df.columns if "equity" in c.lower()][0]
eq_all = eq_df[eq_col].astype(float).values[:N]

for c in ["entry_idx", "exit_idx"]:
    trades[c] = trades[c].astype(int)
trades["side"] = trades["side"].astype(str).str.lower()

# contrib_all (baseline)
contrib_all = np.zeros(N)
for _, r in trades.iterrows():
    contrib_all[r["exit_idx"]] += r["R"]

# ============================================================
# CCI indicator (no lookahead)
# ============================================================
PERIOD = 20
typ = (btc["high"] + btc["low"] + btc["close"]) / 3.0
sma = typ.rolling(PERIOD).mean()
mad = typ.rolling(PERIOD).apply(
    lambda x: np.mean(np.abs(x - x.mean())), raw=True
)

btc["cci"] = (typ - sma) / (0.015 * mad)
btc["cci_prev"] = btc["cci"].shift(1)

# ============================================================
# Filters (Momentum)
# ============================================================
def cci_extreme(idx, side):
    v = btc["cci_prev"].iat[idx]
    if np.isnan(v):
        return False
    return v > 100 if side.startswith("l") else v < -100

def cci_center(idx, side):
    v = btc["cci_prev"].iat[idx]
    if np.isnan(v):
        return False
    return v > 0 if side.startswith("l") else v < 0

def cci_high_quantile(idx, side):
    v = btc["cci_prev"].iat[idx]
    hist = btc["cci"].iloc[:idx].dropna()
    if np.isnan(v) or len(hist) < 30:
        return False
    q75, q25 = hist.quantile([0.75, 0.25])
    return v > q75 if side.startswith("l") else v < q25

FILTERS = {
    "cci_extreme": cci_extreme,
    "cci_center": cci_center,
    "cci_high_quantile": cci_high_quantile,
}

# ============================================================
# Apply filters
# ============================================================
rows = []
equity_curves = {"date": btc["date"], "baseline": eq_all}

plt.figure(figsize=(14, 6))
plt.plot(btc["date"], eq_all, color="black", label="baseline")

for name, fn in FILTERS.items():
    keep = [
        fn(r.entry_idx, r.side)
        for r in trades.itertuples()
    ]

    trades_f = trades.iloc[[i for i, k in enumerate(keep) if k]]

    contrib_f = np.zeros(N)
    for _, r in trades_f.iterrows():
        contrib_f[r["exit_idx"]] += r["R"]

    eq_f = eq_all - np.cumsum(contrib_all - contrib_f)
    equity_curves[name] = eq_f

    dd = np.maximum.accumulate(eq_f) - eq_f

    rows.append({
        "filter": name,
        "n_trades": len(trades_f),
        "final_equity": eq_f[-1],
        "maxdd": dd.max(),
        "win_rate": (trades_f["hit"] != "sl").mean()
    })

    plt.plot(btc["date"], eq_f, label=name)

# ============================================================
# Save outputs
# ============================================================
pd.DataFrame(equity_curves).to_csv(
    os.path.join(EQUITY_DIR, "cci_equity_curves.csv"),
    index=False
)

pd.DataFrame(rows).to_csv(
    os.path.join(INFO_DIR, "cci_summary.csv"),
    index=False
)

plt.legend(ncol=2)
plt.title("CCI filters vs baseline")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "cci_filters.png"), dpi=150)
plt.close()