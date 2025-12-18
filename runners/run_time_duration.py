#!/usr/bin/env python3
import argparse
import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _infer_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _align_to_N(arr, N):
    arr = np.asarray(arr, dtype=float)
    if len(arr) >= N:
        return arr[:N]
    return np.pad(arr, (0, N - len(arr)), mode="edge")


def contrib_from_equity(eq_all, eq_f, contrib_all):
    d = eq_all - eq_f
    d0 = float(d[0]) if len(d) else 0.0
    delta = np.empty_like(d)
    delta[0] = d0
    if len(d) > 1:
        delta[1:] = d[1:] - d[:-1]
    return contrib_all - delta


def keep_mask_from_contrib(trades, contrib_all, contrib_f, exit_to_trade_idxs, tol=1e-9):
    keep = np.zeros(len(trades), dtype=bool)

    for i, r in trades.iterrows():
        ei = int(r["exit_idx"]) if "exit_idx" in trades.columns else None
        Ri = float(r.get("R", 0.0))

        if ei is None or not (0 <= ei < len(contrib_all)):
            keep[i] = False
            continue

        idxs_here = exit_to_trade_idxs.get(ei, [])
        if len(idxs_here) <= 1:
            keep[i] = abs(contrib_f[ei] - Ri) <= tol
        else:
            sum_all = float(trades.loc[idxs_here, "R"].sum())
            if abs(contrib_f[ei] - sum_all) <= tol:
                keep[i] = True
            elif abs(contrib_f[ei]) <= tol:
                keep[i] = False
            else:
                keep[i] = abs(contrib_f[ei]) >= (abs(Ri) - tol)

    return keep


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--strategy", required=True)
    p.add_argument("--asset", required=True)
    p.add_argument("--timeframe", required=True)
    p.add_argument("--window", type=int, required=True)
    p.add_argument("--rr", type=float, required=True)
    p.add_argument("--top_csv", required=False, help="Path to top CSV (optional). If omitted the script will try to detect top CSV in pairwise_winners or scatters directories.")
    # Accept extra strategy flags for compatibility
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
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

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

    CANONICAL_DIR = os.path.join(EXP_DIR, "canonical_output")
    ITER_DIR = os.path.join(EXP_DIR, "iteration_1")
    EQUITY_DIR = os.path.join(ITER_DIR, "equity")
    SCATTERS_DIR = os.path.join(EXP_DIR, "scatters")
    OUT_DIR = os.path.join(EXP_DIR, "pairwise_winners", "trading_duration")
    OUT_DATA = os.path.join(OUT_DIR, "trading_duration_data")
    OUT_PLOTS = os.path.join(OUT_DIR, "trading_duration_histogram")
    os.makedirs(OUT_DATA, exist_ok=True)
    os.makedirs(OUT_PLOTS, exist_ok=True)

    top_csv = args.top_csv
    if top_csv is None:
        # auto-detect same places as other runners
        candidates = []
        pw = os.path.join(EXP_DIR, "pairwise_winners", "top_pairwise.csv")
        if os.path.exists(pw):
            candidates.append(pw)
        scat_data = os.path.join(EXP_DIR, "scatters", "data")
        if os.path.isdir(scat_data):
            for f in os.listdir(scat_data):
                if f.startswith("top10_pairs_by_") and f.endswith(".csv"):
                    candidates.append(os.path.join(scat_data, f))
        scat_root = os.path.join(EXP_DIR, "scatters")
        if os.path.isdir(scat_root):
            for f in os.listdir(scat_root):
                if f.startswith("top10_pairs_by_") and f.endswith(".csv"):
                    candidates.append(os.path.join(scat_root, f))

        pref = [pw,
                os.path.join(scat_data, "top10_pairs_by_win_rate.csv") if os.path.isdir(scat_data) else None,
                os.path.join(scat_data, "top10_pairs_by_n_trades.csv") if os.path.isdir(scat_data) else None]
        top_csv = None
        for p in pref:
            if p and os.path.exists(p):
                top_csv = p
                break
        if top_csv is None and candidates:
            top_csv = sorted(candidates)[0]
    if not os.path.exists(top_csv):
        raise SystemExit(f"Top CSV not found: {top_csv}")

    trades_path = os.path.join(CANONICAL_DIR, "trades.csv")
    if not os.path.exists(trades_path):
        raise SystemExit("Missing canonical trades.csv")

    trades = pd.read_csv(trades_path)
    top = pd.read_csv(top_csv).head(10)

    # infer duration column (exit_idx - entry_idx)
    if "entry_idx" not in trades.columns or "exit_idx" not in trades.columns:
        raise SystemExit("trades.csv missing entry_idx/exit_idx columns")

    # load canonical equity to reconstruct keeps
    eq_df = pd.read_csv(os.path.join(CANONICAL_DIR, "equity.csv"))
    eq_col = [c for c in eq_df.columns if "equity" in c.lower()]
    if not eq_col:
        raise SystemExit(f"No equity column found in {os.path.join(CANONICAL_DIR, 'equity.csv')}")
    eq_col = eq_col[0]

    # load candle dates to determine N
    data_dir = os.path.join(PROJECT_ROOT, "candle_data", args.asset, args.timeframe)
    csvs = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    if not csvs:
        raise SystemExit(f"No candle CSVs in {data_dir}")
    btc = pd.read_csv(os.path.join(data_dir, csvs[0]))
    if "open_time" in btc.columns:
        dates = pd.to_datetime(btc["open_time"], unit="ms")
    elif "date" in btc.columns:
        dates = pd.to_datetime(btc["date"])
    else:
        dates = pd.to_datetime(btc.iloc[:, 0])
    dates = dates.sort_values().reset_index(drop=True)
    N = len(dates)

    eq_all = eq_df[eq_col].astype(float).values[:N]
    if len(eq_all) != N:
        eq_all = np.pad(eq_all, (0, max(0, N - len(eq_all))), mode="edge")[:N]

    contrib_all = np.zeros(N, dtype=float)
    for _, r in trades.iterrows():
        ei = int(r["exit_idx"]) if "exit_idx" in trades.columns else None
        if ei is not None and 0 <= ei < N:
            contrib_all[ei] += float(r.get("R", 0.0))

    exit_to_trade_idxs = {}
    for i, r in trades.iterrows():
        ei = int(r["exit_idx"]) if "exit_idx" in trades.columns else None
        if ei is None:
            continue
        exit_to_trade_idxs.setdefault(ei, []).append(i)

    # load per-filter equities
    equity_files = sorted(glob.glob(os.path.join(EQUITY_DIR, "*.csv")))
    if not equity_files:
        raise SystemExit(f"No equity files found in {EQUITY_DIR}")

    filter_to_eq = {}
    for path in equity_files:
        df = pd.read_csv(path)
        if "filter" in df.columns and "equity" in df.columns:
            for name, g in df.groupby("filter"):
                eq = _align_to_N(pd.to_numeric(g["equity"], errors="coerce").astype(float).values, N)
                filter_to_eq[str(name)] = eq
        else:
            drop_cols = set()
            for c in df.columns:
                cl = c.lower()
                if cl in ["date", "baseline"]:
                    drop_cols.add(c)
            cand_cols = [c for c in df.columns if c not in drop_cols]
            for c in cand_cols:
                if c.lower() == "baseline":
                    continue
                eq = _align_to_N(pd.to_numeric(df[c], errors="coerce").astype(float).values, N)
                filter_to_eq[str(c)] = eq

    filter_names = sorted(filter_to_eq.keys())
    if not filter_names:
        raise SystemExit("No filters found from iteration_1/equity")

    filter_to_keep = {}
    for fname in filter_names:
        eq_f = filter_to_eq[fname]
        cf = contrib_from_equity(eq_all, eq_f, contrib_all)
        keep = keep_mask_from_contrib(trades, contrib_all, cf, exit_to_trade_idxs, tol=1e-9)
        filter_to_keep[fname] = keep

    summary = {}
    rows = []

    for _, r in top.iterrows():
        a = str(r["filter_a"])
        b = str(r["filter_b"])
        if a not in filter_to_keep or b not in filter_to_keep:
            continue

        keep_ab = filter_to_keep[a] & filter_to_keep[b]
        if not keep_ab.any():
            continue

        filtered = trades.loc[keep_ab]
        durations = (filtered["exit_idx"] - filtered["entry_idx"]).astype(float).replace([np.inf, -np.inf], np.nan).dropna().values
        if len(durations) == 0:
            continue

        # save CSV
        out_csv = os.path.join(OUT_DATA, f"{a}__{b}_durations.csv")
        pd.DataFrame({"duration": durations}).to_csv(out_csv, index=False)

        # histogram with mean line — finer bins, only 0..50 bars
        mean_val = float(np.nanmean(durations))
        try:
            durations_clipped = durations[(durations >= 0) & (durations <= 50)]
            if len(durations_clipped) > 0:
                mean_filtered = float(np.nanmean(durations_clipped))
                fig, ax = plt.subplots(figsize=(8, 3))
                ax.hist(durations_clipped, bins=50, range=(0, 50), alpha=0.7)
                ax.axvline(mean_filtered, color="red", linestyle="-", linewidth=1.2, label=f"mean={mean_filtered:.4g}")
                ax.legend(loc="upper right", fontsize=9)
                ax.set_title(f"Trade duration — {a} & {b} (0-50 bars)")
                ax.set_xlabel("Duration (bars)")
                ax.set_ylabel("Count")
                out_hist = os.path.join(OUT_PLOTS, f"{a}__{b}_duration_hist.png")
                plt.tight_layout()
                plt.savefig(out_hist, dpi=140)
                plt.close(fig)
            else:
                out_hist = None
        except Exception:
            out_hist = None

        rows.append({
            "filter_a": a,
            "filter_b": b,
            "n": int(len(durations)),
            "p50": float(np.nanpercentile(durations, 50)),
            "p75": float(np.nanpercentile(durations, 75)),
            "p90": float(np.nanpercentile(durations, 90)),
            "mean": mean_val,
            "csv": out_csv,
            "hist": out_hist,
        })

        summary[f"{a}__{b}"] = mean_val

    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "trading_duration_summary_top_pairs.csv"), index=False)
    # write summary json
    with open(os.path.join(OUT_DIR, "trading_duration_means.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # quiet: duration CSVs and histograms saved to OUT_DIR


if __name__ == "__main__":
    main()
