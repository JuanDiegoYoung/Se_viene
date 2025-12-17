#!/usr/bin/env python3
import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def _find_project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def _pick_first_existing(path_candidates):
    for p in path_candidates:
        if p is not None and os.path.exists(p):
            return p
    return None

def _infer_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _load_keep_cache(scatters_dir):
    """
    Expected npz schema:
      - filter_names: array of strings (n_filters,)
      - keep_matrix: boolean/int array (n_filters, n_trades)
    """
    cache_path = _pick_first_existing([
        os.path.join(scatters_dir, "filter_keep_cache.npz"),
        os.path.join(scatters_dir, "keeps_cache.npz"),
        os.path.join(scatters_dir, "keep_cache.npz"),
    ])
    if cache_path is None:
        return None, None, None

    data = np.load(cache_path, allow_pickle=True)
    if "filter_names" not in data or "keep_matrix" not in data:
        raise SystemExit(f"Keep cache inválido: {cache_path} (faltan filter_names/keep_matrix)")
    filter_names = [str(x) for x in data["filter_names"].tolist()]
    keep_matrix = data["keep_matrix"]
    keep_matrix = keep_matrix.astype(bool)
    return cache_path, filter_names, keep_matrix

def _compute_sl_sizes(trades):
    side_col = _infer_col(trades, ["side", "Side"])
    entry_col = _infer_col(trades, ["entry", "entry_price", "Entry", "price_entry"])
    sl_col = _infer_col(trades, ["sl", "sl_price", "stop_loss", "stop", "StopLoss"])

    if side_col is None or entry_col is None or sl_col is None:
        missing = []
        if side_col is None:
            missing.append("side")
        if entry_col is None:
            missing.append("entry")
        if sl_col is None:
            missing.append("sl")
        raise SystemExit(f"trades.csv no tiene columnas para SL: faltan {missing}. "
                         f"Columnas presentes: {trades.columns.tolist()}")

    side = trades[side_col].astype(str).str.lower().values
    entry = pd.to_numeric(trades[entry_col], errors="coerce").values
    sl = pd.to_numeric(trades[sl_col], errors="coerce").values

    dist_abs = np.abs(entry - sl)
    dist_pct = np.where(entry != 0, dist_abs / np.abs(entry), np.nan)

    return dist_abs, dist_pct

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--strategy", required=True)
    p.add_argument("--asset", required=True)
    p.add_argument("--timeframe", required=True)
    p.add_argument("--window", type=int, required=True)
    p.add_argument("--rr", type=float, required=True)
    p.add_argument("--pairs_csv", required=False, help="Path to pairs CSV (optional). If omitted the script will try to detect top CSV in pairwise_winners or scatters directories.")
    p.add_argument("--top_k", type=int, default=10)
    args = p.parse_args()

    rr_str = f"{args.rr:.1f}"

    PROJECT_ROOT = _find_project_root()

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
    SCATTERS_DIR = os.path.join(EXP_DIR, "scatters")
    os.makedirs(SCATTERS_DIR, exist_ok=True)

    # output into pairwise_winners/stop_loss_distributions by default
    OUT_DIR = os.path.join(EXP_DIR, "pairwise_winners", "stop_loss_distributions")
    OUT_DATA = os.path.join(OUT_DIR, "stop_loss_data")
    OUT_PLOTS = os.path.join(OUT_DIR, "stop_loss_histograms")
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(OUT_DATA, exist_ok=True)
    os.makedirs(OUT_PLOTS, exist_ok=True)

    trades_path = os.path.join(CANONICAL_DIR, "trades.csv")
    if not os.path.exists(trades_path):
        raise SystemExit(f"No encuentro trades.csv en: {trades_path}")

    pairs_path = args.pairs_csv
    if pairs_path is None:
        # try to auto-detect
        candidates = []
        # preferred: pairwise_winners/top_pairwise.csv
        pw = os.path.join(EXP_DIR, "pairwise_winners", "top_pairwise.csv")
        if os.path.exists(pw):
            candidates.append(pw)
        # scatters/data
        scat_data = os.path.join(EXP_DIR, "scatters", "data")
        if os.path.isdir(scat_data):
            for f in os.listdir(scat_data):
                if f.startswith("top10_pairs_by_") and f.endswith(".csv"):
                    candidates.append(os.path.join(scat_data, f))
        # scatters root
        scat_root = os.path.join(EXP_DIR, "scatters")
        if os.path.isdir(scat_root):
            for f in os.listdir(scat_root):
                if f.startswith("top10_pairs_by_") and f.endswith(".csv"):
                    candidates.append(os.path.join(scat_root, f))

        pairs_path = None
        pref = [pw,
                os.path.join(scat_data, "top10_pairs_by_win_rate.csv") if os.path.isdir(scat_data) else None,
                os.path.join(scat_data, "top10_pairs_by_n_trades.csv") if os.path.isdir(scat_data) else None]
        for p in pref:
            if p and os.path.exists(p):
                pairs_path = p
                break
        if pairs_path is None and candidates:
            pairs_path = sorted(candidates)[0]

    else:
        if not os.path.exists(pairs_path):
            alt = os.path.join(PROJECT_ROOT, pairs_path)
            if os.path.exists(alt):
                pairs_path = alt
            else:
                raise SystemExit(f"No encuentro pairs_csv: {args.pairs_csv}")

    pairs = pd.read_csv(pairs_path)
    if "filter_a" not in pairs.columns or "filter_b" not in pairs.columns:
        raise SystemExit(f"{pairs_path} debe tener columnas filter_a y filter_b. Tiene: {pairs.columns.tolist()}")

    pairs = pairs.head(int(args.top_k)).copy()

    trades = pd.read_csv(trades_path)

    cache_path, filter_names, keep_matrix = _load_keep_cache(SCATTERS_DIR)
    if cache_path is None:
        raise SystemExit(
            "No encuentro cache de keeps por filtro en scatters.\n"
            "Espero un .npz en:\n"
            f"  {os.path.join(SCATTERS_DIR, 'filter_keep_cache.npz')}\n"
            "con keys: filter_names, keep_matrix.\n"
            "Generalo desde tu runner de pairwise (cuando calculás keep[] por filtro, guardalo ahí)."
        )

    name_to_idx = {n: i for i, n in enumerate(filter_names)}
    n_trades_total = keep_matrix.shape[1]

    if len(trades) != n_trades_total:
        raise SystemExit(
            f"Mismatch: len(trades)={len(trades)} pero keep_matrix tiene n_trades={n_trades_total}. "
            f"Cache={cache_path}"
        )

    out_rows = []
    plt.figure(figsize=(12, 7))

    for i, row in pairs.iterrows():
        a = str(row["filter_a"])
        b = str(row["filter_b"])
        if a not in name_to_idx or b not in name_to_idx:
            continue

        keep_a = keep_matrix[name_to_idx[a]]
        keep_b = keep_matrix[name_to_idx[b]]
        keep_ab = keep_a & keep_b

        trades_ab = trades.loc[keep_ab].copy()
        if len(trades_ab) == 0:
            continue

        dist_abs, dist_pct = _compute_sl_sizes(trades_ab)

        dist_abs = pd.Series(dist_abs).replace([np.inf, -np.inf], np.nan).dropna().values
        dist_pct = pd.Series(dist_pct).replace([np.inf, -np.inf], np.nan).dropna().values

        if len(dist_abs) == 0:
            continue

        out_rows.append({
            "filter_a": a,
            "filter_b": b,
            "n_trades": int(len(trades_ab)),
            "sl_abs_p50": float(np.nanpercentile(dist_abs, 50)),
            "sl_abs_p75": float(np.nanpercentile(dist_abs, 75)),
            "sl_abs_p90": float(np.nanpercentile(dist_abs, 90)),
            "sl_pct_p50": float(np.nanpercentile(dist_pct, 50)) if len(dist_pct) else np.nan,
            "sl_pct_p75": float(np.nanpercentile(dist_pct, 75)) if len(dist_pct) else np.nan,
            "sl_pct_p90": float(np.nanpercentile(dist_pct, 90)) if len(dist_pct) else np.nan,
        })
        # save individual CSV and histogram per pair into pairwise_winners
        csv_out = os.path.join(OUT_DATA, f"{a}__{b}_sl_abs.csv")
        pd.DataFrame({"sl_abs": dist_abs}).to_csv(csv_out, index=False)
        try:
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.hist(dist_abs, bins=40, alpha=0.7)
            mean_val = float(np.nanmean(dist_abs))
            ax.axvline(mean_val, color="red", linestyle="-", linewidth=1.2, label=f"mean={mean_val:.4g}")
            ax.legend(loc="upper right", fontsize=9)
            ax.set_title(f"SL abs distribution — {a} & {b}")
            ax.set_xlabel("SL size (abs)")
            ax.set_ylabel("Count")
            hist_path = os.path.join(OUT_PLOTS, f"{a}__{b}_sl_hist.png")
            plt.tight_layout()
            plt.savefig(hist_path, dpi=140)
            plt.close(fig)
        except Exception:
            hist_path = None

        plt.hist(dist_abs, bins=40, alpha=0.35, label=f"{a} & {b} (n={len(trades_ab)})")

    if not out_rows:
        raise SystemExit("No pude calcular nada: pares no matchean con el cache o no hay trades filtrados.")

    out_df = pd.DataFrame(out_rows).sort_values(["sl_abs_p50"], ascending=True)
    out_csv = os.path.join(OUT_DIR, "sl_summary_top_pairs.csv")
    out_png = os.path.join(OUT_DIR, "sl_distribution_top_pairs.png")

    out_df.to_csv(out_csv, index=False)

    plt.title("Distribución de tamaño de SL (abs) para top pairs")
    plt.xlabel("SL size (abs)")
    plt.ylabel("Count")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

    print("OK")
        # Removed noisy print statements
        # print("Cache =", cache_path)
        # print("Saved CSV =", out_csv)
        # print("Saved PNG =", out_png)

if __name__ == "__main__":
    main()