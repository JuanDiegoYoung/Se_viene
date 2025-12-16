#!/usr/bin/env python3
"""Download OHLCV candles from Binance and save to `candle_data/<symbol>/<interval>/`.

Saves CSV with columns matching existing files: open_time,open,high,low,close,volume,close_time,
quote_asset_volume,num_trades,taker_buy_base_asset_volume,taker_buy_quote_asset_volume,ignore,date

Usage example:
  python runners/download_candles.py --asset BTCUSDT --timeframe 1h --years 5
"""
import argparse
import os
import time
import requests
import math
import pandas as pd
from datetime import datetime, timedelta


def interval_to_millis(interval: str) -> int:
    mapping = {
        "1m": 60_000,
        "3m": 3 * 60_000,
        "5m": 5 * 60_000,
        "15m": 15 * 60_000,
        "30m": 30 * 60_000,
        "1h": 60 * 60_000,
        "2h": 2 * 60 * 60_000,
        "4h": 4 * 60 * 60_000,
        "6h": 6 * 60 * 60_000,
        "8h": 8 * 60 * 60_000,
        "12h": 12 * 60 * 60_000,
        "1d": 24 * 60 * 60_000,
        "3d": 3 * 24 * 60 * 60_000,
        "1w": 7 * 24 * 60 * 60_000,
        "1M": 30 * 24 * 60 * 60_000,
    }
    return mapping.get(interval)


def fetch_klines(symbol, interval, start_time_ms, end_time_ms, limit=1000):
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": int(start_time_ms),
        "endTime": int(end_time_ms),
        "limit": limit,
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--asset", required=True)
    p.add_argument("--timeframe", required=True)
    p.add_argument("--years", type=float, default=5.0, help="How many years of history (default 5)")
    p.add_argument("--out_dir", default="candle_data", help="Root candle data directory")
    args = p.parse_args()

    symbol = args.asset
    interval = args.timeframe

    ms_per_bar = interval_to_millis(interval)
    if ms_per_bar is None:
        raise SystemExit(f"Unsupported interval: {interval}")

    end_dt = datetime.utcnow()
    days = int(args.years * 365)
    start_dt = end_dt - timedelta(days=days)

    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    out_folder = os.path.join(args.out_dir, symbol, interval)
    os.makedirs(out_folder, exist_ok=True)
    out_path = os.path.join(out_folder, f"{symbol}_{interval}_last{days}d.csv")

    cols = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "num_trades",
        "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume",
        "ignore",
        "date",
    ]

    # If file exists, try to resume from last saved open_time
    last_saved_open = None
    if os.path.exists(out_path):
        try:
            with open(out_path, "r", newline="") as f:
                for row in f:
                    pass
                last = row.strip().split(",")
                # first column is open_time
                last_saved_open = int(last[0])
                print(f"Resuming from last saved open_time: {last_saved_open}")
        except Exception:
            last_saved_open = None

    # Open file in append mode and write header if needed
    is_new = not os.path.exists(out_path)
    f_out = open(out_path, "a", newline="")
    import csv
    writer = csv.writer(f_out)
    if is_new:
        writer.writerow(cols)
        f_out.flush()

    total_written = 0

    total_span = end_ms - start_ms

    # Decide number of outer segments: if timeframe < 1h, split proportionally
    ONE_HOUR_MS = 60 * 60 * 1000
    if ms_per_bar < ONE_HOUR_MS:
        ratio = int(ONE_HOUR_MS // ms_per_bar)
        segments = max(1, ratio)
    else:
        segments = 1

    for seg in range(segments):
        seg_start = start_ms + (total_span * seg) // segments
        seg_end = start_ms + (total_span * (seg + 1)) // segments

        # if we have a saved last_open, skip earlier segments
        cursor = seg_start
        if last_saved_open is not None and last_saved_open >= seg_start:
            cursor = last_saved_open + ms_per_bar

        while cursor < seg_end:
            try:
                data = fetch_klines(symbol, interval, cursor, seg_end, limit=1000)
            except Exception as e:
                print(f"Request failed: {e}; sleeping 1s and retrying")
                time.sleep(1)
                continue

            if not data:
                break

            for k in data:
                open_time = int(k[0])
                close_time = int(k[6])
                date = datetime.utcfromtimestamp(open_time / 1000.0).strftime("%Y-%m-%d %H:%M:%S")
                row = [
                    open_time,
                    k[1],
                    k[2],
                    k[3],
                    k[4],
                    k[5],
                    close_time,
                    k[7],
                    k[8],
                    k[9],
                    k[10],
                    k[11],
                    date,
                ]
                writer.writerow(row)
                total_written += 1

            f_out.flush()
            last_open = int(data[-1][0])
            cursor = last_open + ms_per_bar
            print(f"Segment {seg+1}/{segments}: fetched {total_written} bars so far, next start {datetime.utcfromtimestamp(cursor/1000).isoformat()}")
            time.sleep(0.2)

    f_out.close()
    if total_written == 0:
        raise SystemExit("No klines downloaded")
    print(f"Saved {total_written} rows to: {out_path}")

    # completed incremental download and writing
    # file closed; total_written indicates number of rows appended
    pass


if __name__ == "__main__":
    main()
