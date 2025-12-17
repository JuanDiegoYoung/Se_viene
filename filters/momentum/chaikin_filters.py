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

DATA_DIR = os.path.join(
    PROJECT_ROOT,
    "candle_data",
    args.asset,
    args.timeframe,
)

CANONICAL_DIR = os.environ["CANONICAL_DIR"]
OUT_DIR = os.environ["OUT_DIR"]

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

# ============================================================
# ADL + Chaikin Oscillator
# ============================================================
mf_mult = (
    (btc["close"] - btc["low"]) - (btc["high"] - btc["close"])
) / (btc["high"] - btc["low"]).replace(0, np.nan)

mf_vol = mf_mult.fillna(0) * btc["volume"]
btc["adl"] = mf_vol.cumsum()

SHORT = 3
LONG = 10

btc["adl_ema_s"] = btc["adl"].ewm(span=SHORT, adjust=False).mean()
btc["adl_ema_l"] = btc["adl"].ewm(span=LONG, adjust=False).mean()
btc["chaikin"] = btc["adl_ema_s"] - btc["adl_ema_l"]

btc["chaikin_prev"] = btc["chaikin"].shift(1)
btc["chaikin_prev2"] = btc["chaikin"].shift(2)

# ============================================================
# Filters
# ============================================================
def chaikin_pos(idx, side):
    v = btc["chaikin_prev"].iat[idx]
    if pd.isna(v):
        return False
    return v > 0 if side.startswith("l") else v < 0


def chaikin_confirmed(idx, side):
    if idx <= 1:
        return False
    v1 = btc["chaikin_prev"].iat[idx]
    v2 = btc["chaikin_prev2"].iat[idx]
    if pd.isna(v1) or pd.isna(v2):
        return False
    return (v1 > 0 and v2 > 0) if side.startswith("l") else (v1 < 0 and v2 < 0)


def chaikin_top_quantile(idx, side):
    v = btc["chaikin_prev"].iat[idx]
    if pd.isna(v):
        return False
    hist = btc["chaikin"].iloc[:idx].dropna()
    if len(hist) < 30:
        return False
    q75 = hist.quantile(0.75)
    q25 = hist.quantile(0.25)
    return v > q75 if side.startswith("l") else v < q25


FILTERS = {
    "chaikin_pos": chaikin_pos,
    "chaikin_confirmed": chaikin_confirmed,
    "chaikin_top_quantile": chaikin_top_quantile,
}

# ============================================================
# Apply filters
# ============================================================
rows = []
plt.figure(figsize=(14, 6))
plt.plot(btc["date"], eq_all, color="black", label="baseline")

for name, fn in FILTERS.items():
    keep = []
    for _, r in trades.iterrows():
        keep.append(fn(r["entry_idx"], r["side"]))

    trades_f = trades.iloc[[i for i, k in enumerate(keep) if k]]

    contrib = np.zeros(N)
    for _, r in trades_f.iterrows():
        contrib[r["exit_idx"]] += r["R"]

    eq_f = eq_all + np.cumsum(contrib)

    plt.plot(btc["date"], eq_f, label=name)

    dd = np.maximum.accumulate(eq_f) - eq_f

    rows.append({
        "filter": name,
        "n_trades": len(trades_f),
        "final_equity": eq_f[-1],
        "maxdd": dd.max(),
        "win_rate": (trades_f["hit"] != "sl").mean() if len(trades_f) > 0 else np.nan,
    })

    rows_eq = pd.DataFrame({
        "date": btc["date"],
        "equity": eq_f,
        "filter": name,
    })
    rows_eq.to_csv(
        os.path.join(EQUITY_DIR, "chaikin_equity_all.csv"),
        mode="a",
        header=not os.path.exists(os.path.join(EQUITY_DIR, "chaikin_equity_all.csv")),
        index=False,
    )

# ============================================================
# Save outputs
# ============================================================
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "chaikin_filters.png"), dpi=150)

pd.DataFrame(rows).to_csv(
    os.path.join(INFO_DIR, "chaikin_filters_summary.csv"),
    index=False,
)