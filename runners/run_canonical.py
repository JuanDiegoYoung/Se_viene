#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ============================================================
# Project root
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================
# Imports
# ============================================================
from strategies.boschonk import demo_from_csv

# ============================================================
# Helpers
# ============================================================
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# ============================================================
# Main
# ============================================================
def main(asset, timeframe, strategy, window, rr):

    # ----------------------------
    # Input data
    # ----------------------------
    data_csv = (
        PROJECT_ROOT
        / "candle_data"
        / asset
        / timeframe
        / f"{asset}_{timeframe}_last1825d.csv"
    )

    if not data_csv.exists():
        print(f"ERROR: data not found: {data_csv}")
        return 2

    # ----------------------------
    # Output path
    # ----------------------------
    out_dir = (
        PROJECT_ROOT
        / "Experiments"
        / strategy
        / asset
        / timeframe
        / f"window_{window}"
        / f"rr_{rr}"
        / "canonical_output"
    )
    ensure_dir(out_dir)

    print("Running canonical generation")
    print("Data:", data_csv)
    print("Out :", out_dir)

    # ----------------------------
    # Run strategy
    # ----------------------------
    res = demo_from_csv(
        path=str(data_csv),
        rr=rr,
        window=window
    )

    if res is None:
        print("ERROR: strategy returned nothing")
        return 3

    trades_df, equity = res

    # ----------------------------
    # Save trades
    # ----------------------------
    trades_path = out_dir / "trades.csv"
    trades_df.to_csv(trades_path, index=False)
    print(f"Saved trades ({len(trades_df)}):", trades_path)

    # ----------------------------
    # Save equity CSV
    # ----------------------------
    equity_path = out_dir / "equity.csv"
    equity.to_csv(equity_path, index=True, header=["equity"])
    print("Saved equity:", equity_path)

    # ----------------------------
    # Plot equity
    # ----------------------------
    df_raw = pd.read_csv(data_csv)
    date_col = df_raw.iloc[:, 0]

    dates = None
    for unit in ["ms", "s", "us"]:
        try:
            tmp = pd.to_datetime(date_col, unit=unit, errors="coerce")
            if tmp.notna().sum() > 0:
                dates = tmp
                break
        except Exception:
            pass

    fig, ax = plt.subplots(figsize=(12, 3))
    if dates is not None and len(dates) == len(equity):
        ax.plot(dates, equity.values)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    else:
        ax.plot(equity.values)

    ax.set_title(f"Equity – {strategy} | {asset} {timeframe} | rr={rr} | w={window}")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()

    plot_path = out_dir / "equity.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print("Saved equity plot:", plot_path)

    # ----------------------------
    # Canonical info
    # ----------------------------
    info = {
        "asset": asset,
        "timeframe": timeframe,
        "strategy": strategy,
        "window": window,
        "rr": rr,
        "n_trades": len(trades_df),
        "profit_total": float(trades_df["pnl"].sum()),
        "win_rate": float((trades_df["hit"] == "tp").mean()),
    }

    eq = equity.fillna(method="ffill").fillna(0.0)
    info["max_drawdown"] = float((eq.cummax() - eq).max())

    info_path = out_dir / "canonical_info.csv"
    pd.DataFrame([info]).to_csv(info_path, index=False)
    print("Saved canonical info:", info_path)

    return 0

# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--asset", required=True)
    p.add_argument("--timeframe", required=True)
    p.add_argument("--strategy", default="Bos")
    p.add_argument("--window", type=int, default=5)
    p.add_argument("--rr", type=float, default=1.0)
    args = p.parse_args()

    sys.exit(
        main(
            asset=args.asset,
            timeframe=args.timeframe,
            strategy=args.strategy,
            window=args.window,
            rr=args.rr,
        )
    )