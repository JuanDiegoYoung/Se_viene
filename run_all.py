#!/usr/bin/env python3
"""Run all runner scripts in logical order for a given asset/timeframe/window/rr.

Usage examples:
  python run_all.py --asset ETHUSDT --timeframe 1h --window 5 --rr 1.0
  python run_all.py --asset BTCUSDT --timeframe 1h --download

This script locates `runners/` scripts and executes them in a sensible order.
If a runner fails, by default it logs the error and continues; use
`--stop-on-error` to abort on the first failure.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import List


def find_runners() -> List[str]:
    base = os.path.join(os.path.dirname(__file__), "runners")
    all_py = [f for f in os.listdir(base) if f.endswith(".py")]
    # Preferred ordered list (only include if present)
    preferred = [
        "run_download_candles.py",
        "run_canonical.py",
        "run_all_filters.py",
        "run_pairwise_scatter.py",
        "run_scatter.py",
        "run_scatter_filtered.py",
        "run_sl_distribution.py",
        "run_trade_duration.py",
        "run_time_duration.py",
        "run_equity_winners.py",
    ]

    ordered = []
    for name in preferred:
        if name in all_py:
            ordered.append(name)

    # Append any remaining runner scripts alphabetically
    for name in sorted(all_py):
        if name not in ordered:
            ordered.append(name)

    return ordered


def build_cmd(
    script: str,
    asset: str,
    timeframe: str,
    window: int | None,
    rr: float | None,
    strategy: str | None = None,
) -> List[str]:
    script_path = os.path.join("runners", script)
    cmd = [sys.executable, script_path, "--asset", asset, "--timeframe", timeframe]
    if strategy is not None:
        cmd += ["--strategy", strategy]
    if window is not None:
        cmd += ["--window", str(window)]
    if rr is not None:
        cmd += ["--rr", str(rr)]
    return cmd


def main() -> None:
    p = argparse.ArgumentParser(description="Run all pipeline runners in sequence")
    p.add_argument("--asset", required=True)
    p.add_argument("--timeframe", required=True)
    p.add_argument("--window", type=int, default=None)
    p.add_argument("--rr", type=float, default=None)
    p.add_argument("--strategy", required=False, default="boschonk", help="strategy name to pass to runners (default: boschonk)")
    p.add_argument("--download", action="store_true", help="run download step first (if available)")
    p.add_argument("--include-single", action="store_true", help="also run run_single_filter.py (excluded by default)")
    p.add_argument("--min-n-trades", type=int, default=None, help="minimum n_trades to pass pairwise filters; forwarded to pairwise runner")
    p.add_argument("--min-final-equity", type=float, default=None, help="minimum final_equity to pass pairwise filters; forwarded to pairwise runner")
    p.add_argument("--max-maxdd", type=float, default=None, help="maximum maxdd to pass pairwise filters; forwarded to pairwise runner")
    p.add_argument("--save-equities", dest="save_equities", action="store_true", help="Save generated equity CSVs (default: enabled)")
    p.add_argument("--no-save-equities", dest="save_equities", action="store_false", help="Do not save generated equity CSVs")
    p.set_defaults(save_equities=True)
    p.add_argument("--stop-on-error", action="store_true", help="abort on first failing runner")
    p.add_argument("--require-prior-swing", dest="require_prior_swing", action="store_true", help="Require prior swing (default: enabled)")
    p.add_argument("--no-require-prior-swing", dest="require_prior_swing", action="store_false", help="Do not require prior swing")
    p.set_defaults(require_prior_swing=True)
    p.add_argument("--allow-countertrend", dest="allow_countertrend", action="store_true", help="Allow countertrend trades (default: disabled)")
    p.add_argument("--no-allow-countertrend", dest="allow_countertrend", action="store_false", help="Disallow countertrend trades")
    p.set_defaults(allow_countertrend=False)
    p.add_argument("--allow-micro-structure", dest="allow_micro_structure", action="store_true", help="Allow micro-structure (default: enabled)")
    p.add_argument("--no-allow-micro-structure", dest="allow_micro_structure", action="store_false", help="Disable micro-structure")
    p.set_defaults(allow_micro_structure=True)

    import itertools
    args = p.parse_args()
    runners = find_runners()
    # Filtrar runners una sola vez, fuera del bucle de combinaciones
    filtered_runners = runners
    if not args.download:
        filtered_runners = [r for r in filtered_runners if r != "run_download_candles.py"]
    if not args.include_single:
        filtered_runners = [r for r in filtered_runners if r != "run_single_filter.py"]

    # Si alguna flag no se especifica por CLI, iterar sobre todas las combinaciones posibles
    # Detect whether the CLI invocation explicitly provided the dashed flags
    flag_options = {
        'require_prior_swing': [True, False] if '--require-prior-swing' not in sys.argv and '--no-require-prior-swing' not in sys.argv else [args.require_prior_swing],
        'allow_countertrend': [True, False] if '--allow-countertrend' not in sys.argv and '--no-allow-countertrend' not in sys.argv else [args.allow_countertrend],
        'allow_micro_structure': [True, False] if '--allow-micro-structure' not in sys.argv and '--no-allow-micro-structure' not in sys.argv else [args.allow_micro_structure],
    }
    combinations = list(itertools.product(*flag_options.values()))

    for combo in combinations:
        combo_flags = dict(zip(flag_options.keys(), combo))
        print(f"\n=== Running with flags: {combo_flags} ===")
        # Sufijo de flags para carpetas únicas por combinación
        flags_dir = f"prior_{combo_flags['require_prior_swing']}_counter_{combo_flags['allow_countertrend']}_micro_{combo_flags['allow_micro_structure']}"
        for script in filtered_runners:
            cmd = build_cmd(
                script,
                args.asset,
                args.timeframe,
                args.window,
                args.rr,
                strategy=args.strategy,
            )
            # auto-detect top/pairs CSV produced by pairwise scatter
            base_exp = os.path.join("experiments", args.strategy, args.asset, args.timeframe, f"window_{args.window}", f"rr_{args.rr}", flags_dir)
            scatters_data = os.path.join(base_exp, "scatters", "data")
            scatters_root = os.path.join(base_exp, "scatters")
            selected_top = None
            candidates = []
            if os.path.isdir(scatters_data):
                candidates += [os.path.join(scatters_data, f) for f in os.listdir(scatters_data) if f.startswith("top10_pairs_by_") and f.endswith(".csv")]
            if os.path.isdir(scatters_root):
                candidates += [os.path.join(scatters_root, f) for f in os.listdir(scatters_root) if f.startswith("top10_pairs_by_") and f.endswith(".csv")]
            pref_order = [
                os.path.join(scatters_data, "top10_pairs_by_win_rate.csv"),
                os.path.join(scatters_data, "top10_pairs_by_n_trades.csv"),
                os.path.join(scatters_root, "top10_pairs_by_win_rate.csv"),
                os.path.join(scatters_root, "top10_pairs_by_n_trades.csv"),
            ]
            for p in pref_order:
                if os.path.exists(p):
                    selected_top = p
                    break
            if selected_top is None and candidates:
                selected_top = sorted(candidates)[0]
            if script == "run_pairwise_scatter.py":
                if args.min_n_trades is not None:
                    cmd += ["--min-n-trades", str(args.min_n_trades)]
                if args.min_final_equity is not None:
                    cmd += ["--min-final-equity", str(args.min_final_equity)]
                if args.max_maxdd is not None:
                    cmd += ["--max-maxdd", str(args.max_maxdd)]
            if selected_top is not None:
                if script == "run_sl_distribution.py":
                    cmd += ["--pairs_csv", selected_top]
                if script == "run_trade_duration.py":
                    cmd += ["--top_csv", selected_top]
                if script == "run_time_duration.py":
                    cmd += ["--top_csv", selected_top]
                if script == "run_equity_winners.py":
                    cmd += ["--top_csv", selected_top]
            if not args.save_equities and script == "run_equity_winners.py":
                cmd += ["--no-save-equities"]
            if not args.save_equities:
                os.environ["NO_SAVE_EQUITIES"] = "1"
            # Propagar flags de estrategia
            if combo_flags['require_prior_swing']:
                cmd += ["--require-prior-swing"]
            else:
                cmd += ["--no-require-prior-swing"]
            if combo_flags['allow_countertrend']:
                cmd += ["--allow-countertrend"]
            else:
                cmd += ["--no-allow-countertrend"]
            if combo_flags['allow_micro_structure']:
                cmd += ["--allow-micro-structure"]
            else:
                cmd += ["--no-allow-micro-structure"]
            try:
                proc = subprocess.run(cmd, check=False)
                rc = proc.returncode
                if rc != 0:
                    print(f"Runner {script} exited with code {rc}")
                    if args.stop_on_error:
                        raise SystemExit(rc)
                else:
                    stage_msgs = {
                        "run_canonical.py": "Canonical data generated",
                        "run_all_filters.py": "Filters computed",
                        "run_pairwise_scatter.py": "Pairwise computed",
                        "run_equity_winners.py": "Statistics computed",
                    }
                    msg = stage_msgs.get(script)
                    if msg:
                        print(msg)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"Failed to run {script}: {e}")
                if args.stop_on_error:
                    raise


if __name__ == "__main__":
    main()
