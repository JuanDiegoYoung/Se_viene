#!/usr/bin/env python3
import argparse
import os
import glob
import itertools
import math
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# Args
# ============================================================
p = argparse.ArgumentParser()
p.add_argument("--strategy", required=True)
p.add_argument("--asset", required=True)
p.add_argument("--timeframe", required=True)
p.add_argument("--window", type=int, required=True)
p.add_argument("--rr", type=float, required=True)
p.add_argument("--tol", type=float, default=1e-9, help="Tolerance for float matching when reconstructing keeps")
args = p.parse_args()

rr_str = f"{args.rr:.1f}"

# ============================================================
# Paths (robusto: runners/ -> project root)
# ============================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

EXP_DIR = os.path.join(
    PROJECT_ROOT,
    "experiments",
    args.strategy,
    args.asset,
    args.timeframe,
    f"window_{args.window}",
    f"rr_{rr_str}",
)

CANONICAL_DIR = os.path.join(EXP_DIR, "canonical_output")
ITER_DIR = os.path.join(EXP_DIR, "iteration_1")
EQUITY_DIR = os.path.join(ITER_DIR, "equity")
SCATTERS_DIR = os.path.join(EXP_DIR, "scatters")
os.makedirs(SCATTERS_DIR, exist_ok=True)

OUT_CSV = os.path.join(SCATTERS_DIR, "pairwise_filters_scatter_data.csv")
OUT_PNG = os.path.join(SCATTERS_DIR, "pairwise_filters_scatter.png")

# ============================================================
# Load candles (solo para N y dates)
# ============================================================
DATA_DIR = os.path.join(PROJECT_ROOT, "candle_data", args.asset, args.timeframe)
csvs = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
if not csvs:
    raise SystemExit(f"No candles found in {DATA_DIR}")
btc = pd.read_csv(os.path.join(DATA_DIR, csvs[0]))
if "open_time" in btc.columns:
    dates = pd.to_datetime(btc["open_time"], unit="ms")
elif "date" in btc.columns:
    dates = pd.to_datetime(btc["date"])
else:
    dates = pd.to_datetime(btc.iloc[:, 0])
dates = dates.sort_values().reset_index(drop=True)
N = len(dates)

# ============================================================
# Load canonical trades & baseline equity
# ============================================================
trades_path = os.path.join(CANONICAL_DIR, "trades.csv")
equity_path = os.path.join(CANONICAL_DIR, "equity.csv")

if not os.path.exists(trades_path):
    raise SystemExit(f"Missing canonical trades: {trades_path}")
if not os.path.exists(equity_path):
    raise SystemExit(f"Missing canonical equity: {equity_path}")

trades = pd.read_csv(trades_path)
eq_df = pd.read_csv(equity_path)

eq_col = [c for c in eq_df.columns if "equity" in c.lower()]
if not eq_col:
    raise SystemExit(f"No equity column found in {equity_path}. Columns={eq_df.columns.tolist()}")
eq_col = eq_col[0]

eq_all = eq_df[eq_col].astype(float).values[:N]
if len(eq_all) != N:
    eq_all = np.pad(eq_all, (0, max(0, N - len(eq_all))), mode="edge")[:N]

for c in ["entry_idx", "exit_idx"]:
    if c not in trades.columns:
        raise SystemExit(f"trades.csv missing column: {c}")
    trades[c] = trades[c].astype(int)

if "R" not in trades.columns:
    raise SystemExit("trades.csv missing column: R")

trades["R"] = pd.to_numeric(trades["R"], errors="coerce").fillna(0.0).astype(float)

# contrib_all por exit_idx
contrib_all = np.zeros(N, dtype=float)
for _, r in trades.iterrows():
    ei = int(r["exit_idx"])
    if 0 <= ei < N:
        contrib_all[ei] += float(r["R"])

# exit_idx -> lista de indices de trades (para detectar colisiones)
exit_to_trade_idxs = {}
for i, r in trades.iterrows():
    ei = int(r["exit_idx"])
    exit_to_trade_idxs.setdefault(ei, []).append(i)

# ============================================================
# Load all per-filter equities from iteration_1/equity
#   Soporta:
#   - long: date,filter,equity
#   - wide: date,baseline,<filter1>,<filter2>...
# ============================================================
equity_files = sorted(glob.glob(os.path.join(EQUITY_DIR, "*.csv")))
if not equity_files:
    raise SystemExit(f"No equity files found in {EQUITY_DIR}")

filter_to_eq = {}

def _align_to_N(arr):
    arr = np.asarray(arr, dtype=float)
    if len(arr) >= N:
        return arr[:N]
    return np.pad(arr, (0, N - len(arr)), mode="edge")

for path in equity_files:
    df = pd.read_csv(path)

    cols_lower = [c.lower() for c in df.columns]
    if "filter" in df.columns and "equity" in df.columns:
        for name, g in df.groupby("filter"):
            eq = _align_to_N(pd.to_numeric(g["equity"], errors="coerce").astype(float).values)
            filter_to_eq[str(name)] = eq
    else:
        # wide format: take any numeric columns except date/baseline
        drop_cols = set()
        for c in df.columns:
            cl = c.lower()
            if cl in ["date", "baseline"]:
                drop_cols.add(c)
        cand_cols = [c for c in df.columns if c not in drop_cols]
        for c in cand_cols:
            if c.lower() == "baseline":
                continue
            eq = _align_to_N(pd.to_numeric(df[c], errors="coerce").astype(float).values)
            filter_to_eq[str(c)] = eq

filter_names = sorted(filter_to_eq.keys())
if len(filter_names) < 2:
    raise SystemExit(f"Need at least 2 filters to do pairwise. Found: {filter_names}")

# ============================================================
# Reconstruct keep masks per filter from equity series
#   d[t] = eq_all[t] - eq_f[t] = cumsum(contrib_all - contrib_f)[t]
#   delta[t] = d[t]-d[t-1] = (contrib_all - contrib_f)[t]
#   => contrib_f[t] = contrib_all[t] - delta[t]
#   Luego inferimos keep por trade usando exit_idx (asumiendo 1 trade por exit_idx en la práctica)
# ============================================================
def contrib_from_equity(eq_f):
    d = eq_all - eq_f
    d0 = float(d[0]) if len(d) else 0.0
    delta = np.empty_like(d)
    delta[0] = d0
    if len(d) > 1:
        delta[1:] = d[1:] - d[:-1]
    return contrib_all - delta

def keep_mask_from_contrib(contrib_f, tol):
    keep = np.zeros(len(trades), dtype=bool)

    # fast path: si la mayoría de exit_idx tienen 1 trade, esto anda perfecto
    for i, r in trades.iterrows():
        ei = int(r["exit_idx"])
        Ri = float(r["R"])
        if not (0 <= ei < N):
            keep[i] = False
            continue

        # si hay colisión de trades en mismo exit_idx, tratamos casos obvios
        idxs_here = exit_to_trade_idxs.get(ei, [])
        if len(idxs_here) <= 1:
            keep[i] = abs(contrib_f[ei] - Ri) <= tol
        else:
            # casos obvios
            sum_all = float(trades.loc[idxs_here, "R"].sum())
            if abs(contrib_f[ei] - sum_all) <= tol:
                keep[i] = True
            elif abs(contrib_f[ei]) <= tol:
                keep[i] = False
            else:
                # ambiguo: decidimos por “presencia aproximada”
                keep[i] = abs(contrib_f[ei]) >= (abs(Ri) - tol)

    return keep

filter_to_keep = {}
filter_to_stats = {}

for fname in filter_names:
    eq_f = filter_to_eq[fname]
    cf = contrib_from_equity(eq_f)
    keep = keep_mask_from_contrib(cf, args.tol)
    filter_to_keep[fname] = keep

    kept_trades = trades[keep]
    n_tr = int(len(kept_trades))

    if n_tr > 0:
        if "hit" in kept_trades.columns:
            hit = kept_trades["hit"].astype(str).str.lower()
            win_rate = float((hit != "sl").mean())
        else:
            win_rate = float((kept_trades["R"] > 0).mean())
    else:
        win_rate = float("nan")

    s = pd.Series(eq_f).ffill().fillna(method="bfill").fillna(0.0).values
    dd = float((np.maximum.accumulate(s) - s).max()) if len(s) else float("nan")

    filter_to_stats[fname] = {
        "final_equity": float(eq_f[-1]) if len(eq_f) else float("nan"),
        "maxdd": dd,
        "n_trades": n_tr,
        "win_rate": win_rate,
    }

# Save keep cache so downstream runners (SL distribution, etc.) can use it
keep_matrix = np.vstack([filter_to_keep[name].astype(bool) for name in filter_names])
cache_path = os.path.join(SCATTERS_DIR, "filter_keep_cache.npz")
np.savez(cache_path, filter_names=np.array(filter_names, dtype=object), keep_matrix=keep_matrix)
print(f"Saved keep cache: {cache_path}")

# ============================================================
# Pairwise AND, equity resultante, scatter
# ============================================================
rows = []
pairs = list(itertools.combinations(filter_names, 2))

print(f"Loaded filters: {len(filter_names)}")
print(f"Pairwise combinations: {len(pairs)}")

for a, b in pairs:
    keep_ab = filter_to_keep[a] & filter_to_keep[b]
    kept = trades[keep_ab]

    contrib_ab = np.zeros(N, dtype=float)
    for _, r in kept.iterrows():
        ei = int(r["exit_idx"])
        if 0 <= ei < N:
            contrib_ab[ei] += float(r["R"])

    eq_ab = eq_all - np.cumsum(contrib_all - contrib_ab)

    s = pd.Series(eq_ab).ffill().fillna(method="bfill").fillna(0.0).values
    maxdd = float((np.maximum.accumulate(s) - s).max()) if len(s) else float("nan")

    n_tr = int(len(kept))
    if n_tr > 0:
        if "hit" in kept.columns:
            hit = kept["hit"].astype(str).str.lower()
            win_rate = float((hit != "sl").mean())
        else:
            win_rate = float((kept["R"] > 0).mean())
    else:
        win_rate = float("nan")

    rows.append({
        "filter_a": a,
        "filter_b": b,
        "pair": f"{a} & {b}",
        "final_equity": float(eq_ab[-1]),
        "maxdd": maxdd,
        "win_rate": win_rate,
        "n_trades": n_tr,
    })

df_pairs = pd.DataFrame(rows)
df_pairs = df_pairs.dropna(subset=["final_equity", "maxdd"])
df_pairs.to_csv(OUT_CSV, index=False)

# -----------------------------------------
# Filtros duros
# -----------------------------------------
df_filt = df_pairs[
    (df_pairs["n_trades"] > 1500) &
    (df_pairs["final_equity"] > 50) &
    (df_pairs["maxdd"] < 20)
].copy()

print(f"After hard filters: {len(df_filt)} pairs")

# -----------------------------------------
# Scatter
# -----------------------------------------
target_score = "n_trades"

x = df_filt["maxdd"].astype(float).values
y = df_filt["final_equity"].astype(float).values
c = pd.to_numeric(df_filt[target_score], errors="coerce").values
sizes = np.clip(df_filt["n_trades"].values / 10, 20, 300)

plt.figure(figsize=(11, 8))
sc = plt.scatter(
    x,
    y,
    c=c,
    s=sizes,
    cmap="viridis",
    alpha=0.75,
    edgecolors="k",
    linewidths=0.35,
)

plt.colorbar(sc, label=target_score)
plt.xlabel("Max Drawdown")
plt.ylabel("Final Equity")
plt.title("Filtered pairwise combinations")

plt.tight_layout()
plt.savefig(os.path.join(SCATTERS_DIR, f"pairwise_scatter_filtered_{target_score}.png"), dpi=160)
plt.close()

# -----------------------------------------
# Top 10 por win_rate
# -----------------------------------------
top10_score="win_rate"
top10 = (
    df_filt
    .sort_values(top10_score, ascending=False)
    .head(10)
    .reset_index(drop=True)
)

top10.to_csv(
    os.path.join(SCATTERS_DIR, f"top10_pairs_by_{top10_score}.csv"),
    index=False,
)

print("Saved:")
print(f"- pairwise_scatter_filtered_{target_score}.png")
print(f"- top10_pairs_by_{target_score}.csv")
print(top10[[
    "filter_a",
    "filter_b",
    "win_rate",
    "final_equity",
    "maxdd",
    "n_trades",
]])