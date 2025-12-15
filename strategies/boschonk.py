#!/usr/bin/env python3
"""Strategy module extracted from `estrategia_smc.ipynb` and `functions-smc.ipynb`.

Provides parsing, BOS detection, trade simulation and simple plotting helpers.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_binance_files(folder_path):
    column_names = [
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_volume_base', 'taker_buy_volume_quote', 'ignore'
    ]

    dfs = []
    if not os.path.isdir(folder_path):
        print(f"parse_binance_files: folder not found: {folder_path}")
        return pd.DataFrame()

    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]

    for f in files:
        df = pd.read_csv(f, names=column_names)
        for unit in ['ms', 'us']:
            try:
                df['date'] = pd.to_datetime(df['open_time'], unit=unit)
                df['close_time'] = pd.to_datetime(df['close_time'], unit=unit)
                delta = df['date'].diff().dt.total_seconds().dropna()
                mode_delta = delta.mode().iloc[0]
                expected_timeframe_seconds = None
                if expected_timeframe_seconds is None or mode_delta == expected_timeframe_seconds:
                    break
            except Exception:
                continue
        else:
            print(f"Archivo descartado por timeframe: {f}")
            continue

        df = df.drop(columns='ignore')
        dfs.append(df)

    if not dfs:
        print('No se cargaron archivos.')
        return pd.DataFrame()

    df = pd.concat(dfs).sort_values('open_time').reset_index(drop=True)
    df['hora'] = df['date'].dt.time
    return df


# ==================== Funciones ====================
def build_swings_alternating(df, window=5):
    d = df.copy()
    d['swing_high'] = d['close'][(d['close'] == d['close'].rolling(window, center=True).max())]
    d['swing_low']  = d['close'][(d['close']  == d['close'].rolling(window, center=True).min())]
    swings, last = [], None
    for i, r in d.iterrows():
        if not np.isnan(r['swing_high']) and last != 'high':
            swings.append((i, float(r['swing_high']), 'high'))
            last = 'high'
        elif not np.isnan(r['swing_low']) and last != 'low':
            swings.append((i, float(r['swing_low']), 'low'))
            last = 'low'
    return d, swings


def compute_struct_levels_from_swings(df, swings):
    last_high_price = pd.Series(np.nan, index=df.index)
    last_high_idx   = pd.Series(np.nan, index=df.index)
    last_low_price  = pd.Series(np.nan, index=df.index)
    last_low_idx    = pd.Series(np.nan, index=df.index)
    for idx, val, t in swings:
        if t == 'high':
            last_high_price.iloc[idx] = val
            last_high_idx.iloc[idx] = idx
        else:
            last_low_price.iloc[idx]  = val
            last_low_idx.iloc[idx]  = idx
    return (
        last_high_price.ffill(),
        last_high_idx.ffill(),
        last_low_price.ffill(),
        last_low_idx.ffill(),
    )


def detect_bos_from_levels(df, last_high_price, last_high_idx, last_low_price, last_low_idx):
    bos = []
    broken_high_ref = broken_low_ref = None
    for i in range(1, len(df)):
        if not np.isnan(last_high_price.iloc[i]):
            ref_h = int(last_high_idx.iloc[i])
            if (
                broken_high_ref != ref_h
                and df['close'].iloc[i] > last_high_price.iloc[i]
                and df['close'].iloc[i - 1] <= last_high_price.iloc[i - 1]
            ):
                bos.append((i, 'up'))
                broken_high_ref = ref_h
        if not np.isnan(last_low_price.iloc[i]):
            ref_l = int(last_low_idx.iloc[i])
            if (
                broken_low_ref != ref_l
                and df['close'].iloc[i] < last_low_price.iloc[i]
                and df['close'].iloc[i - 1] >= last_low_price.iloc[i - 1]
            ):
                bos.append((i, 'down'))
                broken_low_ref = ref_l
    return bos


def build_spans_from_bos(bos, n_bars):
    if bos:
        cur_trend = bos[0][1]
        cur_start = 0
        spans = []
        for i in range(1, len(bos)):
            next_idx, next_trend = bos[i]
            spans.append((cur_start, next_idx, cur_trend))
            cur_trend, cur_start = next_trend, next_idx
        spans.append((cur_start, n_bars - 1, cur_trend))
    else:
        spans = [(0, n_bars - 1, 'up')]
    return spans


def plot_bos_swings(df, swings, bos, spans, title='Swings alternados, BOS y franjas (solo cambian en BOS)'):
    plt.figure(figsize=(14, 6))
    plt.plot(df['close'], color='black', alpha=0.75, label='Close')
    for s, e, tr in spans:
        plt.axvspan(s, e, facecolor=('green' if tr == 'up' else 'red'), alpha=0.10)
    for idx, val, t in swings:
        c = 'green' if t == 'high' else 'red'; m = '^' if t == 'high' else 'v'
        plt.scatter(idx, val, color=c, marker=m, s=90, zorder=3)
    for idx, tr in bos:
        col = 'blue' if tr == 'up' else 'orange'
        plt.scatter(idx, df['close'].iloc[idx], s=130, marker='o', edgecolor='k', linewidth=1.2, color=col, zorder=4, alpha=0.5)
    plt.title(title); plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()


def _last_swing_before(swings, i, t):
    for idx, val, typ in reversed(swings):
        if idx < i and typ == t:
            return idx, val
    return None, np.nan


def _simulate_trade_from(df, side, j_entry, px_sl, px_tp, *, allow_entry_bar_fill=True, touch_mode='strict', eps_frac=1e-4):
    for j in range(j_entry, len(df)):
        hi = float(df['high'].iloc[j]); lo = float(df['low'].iloc[j]); cl = float(df['close'].iloc[j])
        eps = float(eps_frac)

        if touch_mode == 'close':
            if side == 'long':
                if cl <= px_sl: return j, px_sl, 'sl'
                if cl >= px_tp: return j, px_tp, 'tp'
            else:
                if cl >= px_sl: return j, px_sl, 'sl'
                if cl <= px_tp: return j, px_tp, 'tp'
            continue

        if (j == j_entry) and (not allow_entry_bar_fill):
            continue

        if touch_mode == 'wick':
            if side == 'long':
                if lo <= px_sl: return j, px_sl, 'sl'
                if hi >= px_tp: return j, px_tp, 'tp'
            else:
                if hi >= px_sl: return j, px_sl, 'sl'
                if lo <= px_tp: return j, px_tp, 'tp'
        elif touch_mode == 'strict':
            if side == 'long':
                if lo < px_sl * (1 + eps): return j, px_sl, 'sl'
                if hi > px_tp * (1 + eps): return j, px_tp, 'tp'
            else:
                if hi > px_sl * (1 + eps): return j, px_sl, 'sl'
                if lo < px_tp * (1 + eps): return j, px_tp, 'tp'
        else:
            raise ValueError("touch_mode debe ser 'wick', 'strict' o 'close'")
    return len(df) - 1, float(df['close'].iloc[-1]), 'none'


def build_trades_from_bos(df, swings, bos, rr, use_next_open=True, fee=0.00075):
    rows = []
    for i, tr in bos:
        if tr == 'down':  # abrir short, SL en protected high
            prot = [s for s in swings if s[2] == 'high' and s[0] < i]
            if not prot:
                continue
            prot_idx, px_sl = prot[-1][0], float(prot[-1][1])
            i_entry = min(i + 1, len(df) - 1) if use_next_open else i
            px_entry = (
                float(df['open'].iloc[i + 1]) if (use_next_open and i + 1 < len(df)) else float(df['close'].iloc[i])
            )
            if px_sl <= px_entry:
                continue
            risk = px_sl - px_entry
            px_tp = px_entry - rr * risk
            start_j = i_entry if use_next_open else i_entry + 1
            i_exit, px_exit, hit = _simulate_trade_from(df, 'short', start_j, px_sl, px_tp)
            pnl_net = (px_entry - px_exit) - fee * px_entry - fee * px_exit
            R_net = pnl_net / risk
            R_clean = -1.0 if hit == 'sl' else (rr if hit == 'tp' else (px_entry - px_exit) / risk)
            rows.append(
                dict(
                    side='short',
                    bos_idx=i,
                    entry_idx=i_entry,
                    exit_idx=i_exit,
                    entry=px_entry,
                    sl=px_sl,
                    tp=px_tp,
                    exit=px_exit,
                    pnl=pnl_net,
                    R=R_net,
                    R_clean=R_clean,
                    hit=hit,
                    prot_idx=prot_idx,
                    prot_val=px_sl,
                )
            )
        else:  # tr == 'up' -> abrir long, SL en protected low
            prot = [s for s in swings if s[2] == 'low' and s[0] < i]
            if not prot:
                continue
            prot_idx, px_sl = prot[-1][0], float(prot[-1][1])
            i_entry = min(i + 1, len(df) - 1) if use_next_open else i
            px_entry = (
                float(df['open'].iloc[i + 1]) if (use_next_open and i + 1 < len(df)) else float(df['close'].iloc[i])
            )
            if px_sl >= px_entry:
                continue
            risk = px_entry - px_sl
            px_tp = px_entry + rr * risk
            start_j = i_entry if use_next_open else i_entry + 1
            i_exit, px_exit, hit = _simulate_trade_from(df, 'long', start_j, px_sl, px_tp)
            pnl_net = (px_exit - px_entry) - fee * px_entry - fee * px_exit
            R_net = pnl_net / risk
            R_clean = -1.0 if hit == 'sl' else (rr if hit == 'tp' else (px_exit - px_entry) / risk)
            rows.append(
                dict(
                    side='long',
                    bos_idx=i,
                    entry_idx=i_entry,
                    exit_idx=i_exit,
                    entry=px_entry,
                    sl=px_sl,
                    tp=px_tp,
                    exit=px_exit,
                    pnl=pnl_net,
                    R=R_net,
                    R_clean=R_clean,
                    hit=hit,
                    prot_idx=prot_idx,
                    prot_val=px_sl,
                )
            )

    trades_df = pd.DataFrame(rows)
    steps_Rnet = pd.Series(0.0, index=df.index, dtype=float)
    for _, r in trades_df.iterrows():
        steps_Rnet.iloc[int(r['exit_idx'])] += r['R']
    equity_Rnet = steps_Rnet.cumsum()
    return trades_df, equity_Rnet


def plot_bos_trades(df, bos, trades_df, title='BOS con SL/TP y operaciones'):
    plt.figure(figsize=(14, 6))
    plt.plot(df['close'], color='black', alpha=0.8, label='Close')

    shown = {'bos_up': False, 'bos_down': False, 'long': False, 'short': False, 'sl': False, 'tp': False, 'exit': False}

    if trades_df is not None and len(trades_df) > 0:
        for _, r in trades_df.iterrows():
            i0, i1 = int(r['entry_idx']), int(r['exit_idx'])
            entry, sl, tp, ex = float(r['entry']), float(r['sl']), float(r['tp']), float(r['exit'])
            is_long = (r['side'] == 'long')
            c = 'green' if is_long else 'red'
            m = '^' if is_long else 'v'

            plt.scatter(i0, entry, color=c, marker=m, s=90, edgecolor='k', linewidth=1.0,
                        zorder=5, label=('Entry long' if is_long else 'Entry short') if (not shown['long'] if is_long else not shown['short']) else None)
            if is_long:
                shown['long'] = True
            else:
                shown['short'] = True

            plt.plot([i0, i1], [sl, sl], linestyle='--', alpha=0.7, color='red',
                     label='Stop Loss' if not shown['sl'] else None)
            shown['sl'] = True
            plt.plot([i0, i1], [tp, tp], linestyle='--', alpha=0.7, color='green',
                     label='Take Profit' if not shown['tp'] else None)
            shown['tp'] = True

            plt.plot([i0, i1], [entry, ex], linestyle=':', alpha=0.7, color=c)

            plt.scatter(i1, ex, color=c, marker='x', s=90, zorder=5,
                        label='Exit' if not shown['exit'] else None)
            shown['exit'] = True

    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', ncol=2)
    plt.tight_layout()
    plt.show()


def plot_equity(equity):
    plt.figure(figsize=(12, 3))
    plt.plot(equity, color='blue')
    plt.title('Evolución del Equity')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def demo_from_csv(path='velas.csv', rr=1.0, window=5):
    if not os.path.exists(path):
        print(f"Demo CSV not found: {path}")
        return
    df = pd.read_csv(path)
    df = df.reset_index(drop=True)
    d, swings = build_swings_alternating(df, window=window)
    lh_p, lh_i, ll_p, ll_i = compute_struct_levels_from_swings(d, swings)
    bos = detect_bos_from_levels(d, lh_p, lh_i, ll_p, ll_i)
    trades_df, equity = build_trades_from_bos(df, swings, bos, rr=rr)
    print('Trades:', len(trades_df))
    return trades_df, equity


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument('--demo-csv', default='velas.csv', help='CSV de velas para demo')
    args = p.parse_args()
    demo_from_csv(args.demo_csv)
