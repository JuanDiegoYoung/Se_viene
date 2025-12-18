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
    f"prior_{args.require_prior_swing}_counter_{args.allow_countertrend}_micro_{args.allow_micro_structure}",
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
# Fisher Transform
# ============================================================
PER = 10
hl2 = (btc["high"] + btc["low"]) / 2.0

min_h = hl2.rolling(PER, min_periods=PER).min()
max_h = hl2.rolling(PER, min_periods=PER).max()

val = 2 * ((hl2 - min_h) / (max_h - min_h).replace(0, np.nan) - 0.5)
val = val.clip(-0.999, 0.999).fillna(0)

fisher = np.zeros(N)
signal = np.zeros(N)

for i in range(N):
    signal[i] = 0.5 * val.iat[i] + (0.5 * signal[i - 1] if i > 0 else 0)
    fisher[i] = 0.5 * np.log((1 + signal[i]) / (1 - signal[i])) if abs(signal[i]) < 1 else fisher[i - 1]

btc["fisher"] = fisher
btc["fisher_prev"] = btc["fisher"].shift(1)

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
# Build contribution vector
# ============================================================
contrib_all = np.zeros(N)
for _, r in trades.iterrows():
    contrib_all[r["exit_idx"]] += r["R"]

# ============================================================
# Filters
# ============================================================
def fisher_pos(idx, side):
    v = btc["fisher_prev"].iat[idx]
    if pd.isna(v):
        return False
    return v > 0 if side.startswith("l") else v < 0

def fisher_thresh(idx, side):
    v = btc["fisher_prev"].iat[idx]
    if pd.isna(v):
        return False
    return v > 0.5 if side.startswith("l") else v < -0.5

def fisher_quantile(idx, side):
    v = btc["fisher_prev"].iat[idx]
    hist = btc["fisher"].iloc[:idx].dropna()
    if pd.isna(v) or len(hist) < 30:
        return False
    q75 = hist.quantile(0.75)
    q25 = hist.quantile(0.25)
    return v > q75 if side.startswith("l") else v < q25

FILTERS = {
    "fisher_pos": fisher_pos,
    "fisher_thresh": fisher_thresh,
    "fisher_quantile": fisher_quantile,
}

# ============================================================
# Apply filters
# ============================================================
equity_df = pd.DataFrame({"date": btc["date"], "baseline": eq_all})
summary = []

plt.figure(figsize=(14, 6))
plt.plot(btc["date"], eq_all, label="baseline", color="black")

for name, fn in FILTERS.items():
    keep_idx = []

    for _, r in trades.iterrows():
        if fn(r["entry_idx"], r["side"]):
            keep_idx.append(r.name)

    trades_f = trades.loc[keep_idx]

    contrib_f = np.zeros(N)
    for _, r in trades_f.iterrows():
        contrib_f[r["exit_idx"]] += r["R"]

    eq_f = eq_all - np.cumsum(contrib_all - contrib_f)

    equity_df[name] = eq_f
    plt.plot(btc["date"], eq_f, label=name)

    dd = np.maximum.accumulate(eq_f) - eq_f

    summary.append({
        "filter": name,
        "n_trades": len(trades_f),
        "win_rate": (trades_f["hit"] != "sl").mean() if len(trades_f) else np.nan,
        "maxdd": dd.max(),
        "final_equity": eq_f[-1],
    })

# ============================================================
# Save outputs
# ============================================================
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "fisher_filters.png"), dpi=150)
plt.close()

equity_df.to_csv(
    os.path.join(EQUITY_DIR, "fisher_equity_all.csv"),
    index=False,
)

pd.DataFrame(summary).to_csv(
    os.path.join(INFO_DIR, "fisher_summary.csv"),
    index=False,
)