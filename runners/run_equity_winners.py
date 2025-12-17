#!/usr/bin/env python3
import argparse
import os
import glob
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


def _compute_sl_sizes(trades_df):
    entry_col = _infer_col(trades_df, ["entry", "entry_price", "Entry", "price_entry"])
    sl_col = _infer_col(trades_df, ["sl", "sl_price", "stop_loss", "stop", "StopLoss"])
    if entry_col is None or sl_col is None:
        return None

    entry = pd.to_numeric(trades_df[entry_col], errors="coerce").values
    sl = pd.to_numeric(trades_df[sl_col], errors="coerce").values
    dist_abs = np.abs(entry - sl)
    dist_abs = pd.Series(dist_abs).replace([np.inf, -np.inf], np.nan).dropna().values
    return dist_abs


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


def keep_mask_from_contrib(trades, contrib_all, contrib_f, exit_to_trade_idxs, tol):
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
    p.add_argument("--require-prior-swing", dest="require_prior_swing", action="store_true")
    p.add_argument("--no-require-prior-swing", dest="require_prior_swing", action="store_false")
    p.set_defaults(require_prior_swing=True)
    p.add_argument("--allow-countertrend", dest="allow_countertrend", action="store_true")
    p.add_argument("--no-allow-countertrend", dest="allow_countertrend", action="store_false")
    p.set_defaults(allow_countertrend=False)
    p.add_argument("--allow-micro-structure", dest="allow_micro_structure", action="store_true")
    p.add_argument("--no-allow-micro-structure", dest="allow_micro_structure", action="store_false")
    p.set_defaults(allow_micro_structure=True)
    p.add_argument("--top_csv", required=False, help="Path to top10 CSV inside scatters folder")
    p.add_argument("--tol", type=float, default=1e-9)
    p.add_argument("--save-equities", dest="save_equities", action="store_true", help="Save generated equity CSVs (default: enabled)")
    p.add_argument("--no-save-equities", dest="save_equities", action="store_false", help="Do not save generated equity CSVs")
    p.set_defaults(save_equities=True)
    args = p.parse_args()

    rr_str = f"{args.rr:.1f}"

    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    flags_dir = f"prior_{args.require_prior_swing}_counter_{args.allow_countertrend}_micro_{args.allow_micro_structure}"
    EXP_DIR = os.path.join(
        PROJECT_ROOT,
        "experiments",
        args.strategy,
        args.asset,
        args.timeframe,
        f"window_{args.window}",
        f"rr_{rr_str}",
        flags_dir
    )

    CANONICAL_DIR = os.path.join(EXP_DIR, "canonical_output")
    ITER_DIR = os.path.join(EXP_DIR, "iteration_1")
    EQUITY_DIR = os.path.join(ITER_DIR, "equity")
    SCATTERS_DIR = os.path.join(EXP_DIR, "scatters")
    OUT_DIR = os.path.join(EXP_DIR, "pairwise_winners")
    OUT_DATA_DIR = os.path.join(OUT_DIR, "data")
    OUT_EQUITY_DIR = os.path.join(OUT_DIR, "equity")
    os.makedirs(OUT_DATA_DIR, exist_ok=True)
    os.makedirs(OUT_EQUITY_DIR, exist_ok=True)
    OUT_SL_DIR = os.path.join(OUT_DIR, "stop_loss_distributions")
    OUT_SL_DATA = os.path.join(OUT_SL_DIR, "stop_loss_data")
    OUT_SL_HISTS = os.path.join(OUT_SL_DIR, "stop_loss_histograms")
    os.makedirs(OUT_SL_DATA, exist_ok=True)
    os.makedirs(OUT_SL_HISTS, exist_ok=True)

    top_csv = args.top_csv
    if top_csv is None:
        # auto-detect
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
    equity_path = os.path.join(CANONICAL_DIR, "equity.csv")
    if not os.path.exists(trades_path) or not os.path.exists(equity_path):
        raise SystemExit("Missing canonical trades/equity. Run run_canonical first.")

    trades = pd.read_csv(trades_path)
    eq_df = pd.read_csv(equity_path)

    eq_col = [c for c in eq_df.columns if "equity" in c.lower()]
    if not eq_col:
        raise SystemExit(f"No equity column found in {equity_path}")
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

    # contrib_all
    contrib_all = np.zeros(N, dtype=float)
    for _, r in trades.iterrows():
        ei = int(r["exit_idx"]) if "exit_idx" in trades.columns else None
        if ei is not None and 0 <= ei < N:
            contrib_all[ei] += float(r.get("R", 0.0))

    # exit_idx map
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
        cols_lower = [c.lower() for c in df.columns]
        if "filter" in df.columns and "equity" in df.columns:
            for name, g in df.groupby("filter"):
                eq = _align_to_N(pd.to_numeric(g["equity"], errors="coerce").astype(float).values, N)
                filter_to_eq[str(name)] = eq
        else:
            # wide format: take numeric columns except date/baseline
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

    # reconstruct keeps
    filter_to_keep = {}
    for fname in filter_names:
        eq_f = filter_to_eq[fname]
        cf = contrib_from_equity(eq_all, eq_f, contrib_all)
        keep = keep_mask_from_contrib(trades, contrib_all, cf, exit_to_trade_idxs, args.tol)
        filter_to_keep[fname] = keep

    # load top10
    top = pd.read_csv(top_csv).head(10)

    saved_data = []
    saved_plots = []
    skipped = 0

    sl_summary_rows = []
    for _, r in top.iterrows():
        a = str(r["filter_a"])
        b = str(r["filter_b"])
        if a not in filter_to_keep or b not in filter_to_keep:
            skipped += 1
            continue

        keep_ab = filter_to_keep[a] & filter_to_keep[b]

        contrib_ab = np.zeros(N, dtype=float)
        for idx in np.where(keep_ab)[0]:
            tr = trades.loc[idx]
            ei = int(tr["exit_idx"]) if "exit_idx" in trades.columns else None
            if ei is not None and 0 <= ei < N:
                contrib_ab[ei] += float(tr.get("R", 0.0))

        eq_ab = eq_all - np.cumsum(contrib_all - contrib_ab)

        out_name = f"{a}__{b}_equity.csv"
        out_path = os.path.join(OUT_DATA_DIR, out_name)

        # save with dates if available
        if args.save_equities:
            try:
                df_out = pd.DataFrame({"date": dates.reset_index(drop=True), "equity": eq_ab})
                df_out.to_csv(out_path, index=False)
            except Exception:
                pd.DataFrame({"equity": eq_ab}).to_csv(out_path, index=True)

        saved_data.append(out_path)

        # generate plot
        try:
            fig, ax = plt.subplots(figsize=(10, 3))
            if len(dates) == len(eq_ab):
                ax.plot(dates, eq_ab)
                fig.autofmt_xdate()
            else:
                ax.plot(eq_ab)
            ax.set_title(f"Equity — {a} & {b}")
            ax.grid(True, alpha=0.3)
            plot_name = f"{a}__{b}_equity.png"
            plot_path = os.path.join(OUT_EQUITY_DIR, plot_name)
            plt.tight_layout()
            plt.savefig(plot_path, dpi=160)
            plt.close(fig)
            saved_plots.append(plot_path)
        except Exception:
            pass

        # stop-loss distribution for this pair
        try:
            trades_ab = trades.loc[keep_ab].copy()
            sl_abs = _compute_sl_sizes(trades_ab)
            if sl_abs is not None and len(sl_abs) > 0:
                sl_csv = os.path.join(OUT_SL_DATA, f"{a}__{b}_sl_abs.csv")
                pd.DataFrame({"sl_abs": sl_abs}).to_csv(sl_csv, index=False)

                # save percentiles summary row
                p50 = float(np.nanpercentile(sl_abs, 50))
                p75 = float(np.nanpercentile(sl_abs, 75))
                p90 = float(np.nanpercentile(sl_abs, 90))

                # histogram
                try:
                    fig, ax = plt.subplots(figsize=(8, 3))
                    counts, bins, patches = ax.hist(sl_abs, bins=40, alpha=0.7)
                    mean_val = float(np.nanmean(sl_abs))
                    ax.axvline(mean_val, color="red", linestyle="-", linewidth=1.2, label=f"mean={mean_val:.4g}")
                    ax.legend(loc="upper right", fontsize=9)
                    ax.set_title(f"SL abs distribution — {a} & {b}")
                    ax.set_xlabel("SL size (abs)")
                    ax.set_ylabel("Count")
                    hist_path = os.path.join(OUT_SL_HISTS, f"{a}__{b}_sl_hist.png")
                    plt.tight_layout()
                    plt.savefig(hist_path, dpi=140)
                    plt.close(fig)
                except Exception:
                    hist_path = None

                sl_summary_rows.append({
                    "filter_a": a,
                    "filter_b": b,
                    "n_sl": int(len(sl_abs)),
                    "sl_p50": p50,
                    "sl_p75": p75,
                    "sl_p90": p90,
                    "sl_csv": sl_csv,
                    "sl_hist": hist_path,
                    "sl_mean": mean_val,
                })
        except Exception:
            # don't fail overall if SL calc fails
            pass

    # write SL summary
    sl_means = {}
    if sl_summary_rows:
        summary_df = pd.DataFrame(sl_summary_rows)
        summary_df.to_csv(os.path.join(OUT_SL_DIR, "sl_summary_top_pairs.csv"), index=False)
        # build JSON of means
        for r in sl_summary_rows:
            key = f"{r['filter_a']}__{r['filter_b']}"
            sl_means[key] = r.get("sl_mean")
        import json
        with open(os.path.join(OUT_SL_DIR, "sl_means_top_pairs.json"), "w") as f:
            json.dump(sl_means, f, indent=2)

    # quiet: saved equity CSVs, plots and SL summaries stored under OUT_DIR


if __name__ == "__main__":
    main()
