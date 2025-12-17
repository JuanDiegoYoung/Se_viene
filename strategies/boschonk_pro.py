def stepper_bos(
    df,
    window=5,
    rr=1.0,
    use_next_open=True,
    fee=0.00075,
    require_prior_swing=True,
    allow_countertrend=False,
    allow_micro_structure=True,
    start=None
):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import ipywidgets as widgets
    from IPython.display import display

    df = df.reset_index(drop=True).copy()

    if start is None:
        start = max(2 * window + 5, 50)
    start = min(max(start, 5), len(df) - 1)

    state = {"t": start}

    out = widgets.Output()
    btn_next = widgets.Button(description="Next")
    btn_prev = widgets.Button(description="Prev")
    slider = widgets.IntSlider(value=start, min=5, max=len(df) - 1, step=1, description="t")

    def build_swings_confirmed_upto(dft, w, t):
        closes = dft["close"].values
        swings = []
        last = None
        for j in range(w, t + 1):
            i = j - w
            past = closes[i:j + 1]
            px = closes[i]
            if px == past.max() and last != "high":
                swings.append((i, float(px), "high", j))
                last = "high"
            elif px == past.min() and last != "low":
                swings.append((i, float(px), "low", j))
                last = "low"
        return swings

    def compute_levels_confirmed_upto(n, swings):
        lh_p = np.full(n, np.nan)
        lh_i = np.full(n, np.nan)
        ll_p = np.full(n, np.nan)
        ll_i = np.full(n, np.nan)

        for idx, val, typ, confirm_idx in swings:
            if confirm_idx >= n:
                continue
            if typ == "high":
                lh_p[confirm_idx] = val
                lh_i[confirm_idx] = idx
            else:
                ll_p[confirm_idx] = val
                ll_i[confirm_idx] = idx

        lh_p = pd.Series(lh_p).ffill().values
        lh_i = pd.Series(lh_i).ffill().values
        ll_p = pd.Series(ll_p).ffill().values
        ll_i = pd.Series(ll_i).ffill().values
        return lh_p, lh_i, ll_p, ll_i

    def detect_bos_upto(dft, lh_p, lh_i, ll_p, ll_i, t):
        bos = []
        broken_high_ref = None
        broken_low_ref = None

        for i in range(1, t + 1):
            cl = float(dft["close"].iloc[i])

            if not np.isnan(lh_p[i - 1]):
                ref_h = int(lh_i[i - 1])
                lvl_h = float(lh_p[i - 1])
                if cl > lvl_h and broken_high_ref != ref_h:
                    bos.append((i, "up"))
                    broken_high_ref = ref_h

            if not np.isnan(ll_p[i - 1]):
                ref_l = int(ll_i[i - 1])
                lvl_l = float(ll_p[i - 1])
                if cl < lvl_l and broken_low_ref != ref_l:
                    bos.append((i, "down"))
                    broken_low_ref = ref_l

        return bos

    def last_bos_dir_before(bos, idx):
        prev = None
        for bi, d in bos:
            if bi < idx:
                prev = d
            else:
                break
        return prev

    def filter_micro_structure(swings, bos):
        if allow_micro_structure:
            return swings

        out_sw = []
        last_up_high = None
        last_down_low = None
        prev_trend = None

        for i, px, typ, confirm in swings:
            trend = last_bos_dir_before(bos, confirm)

            if trend != prev_trend:
                if trend == "up":
                    last_up_high = None
                if trend == "down":
                    last_down_low = None
                prev_trend = trend

            if trend is None:
                out_sw.append((i, px, typ, confirm))
                continue

            if trend == "up" and typ == "high":
                if last_up_high is None or px >= last_up_high:
                    out_sw.append((i, px, typ, confirm))
                    last_up_high = px
                continue

            if trend == "down" and typ == "low":
                if last_down_low is None or px <= last_down_low:
                    out_sw.append((i, px, typ, confirm))
                    last_down_low = px
                continue

            out_sw.append((i, px, typ, confirm))

        return out_sw

    def stabilize_swings_and_bos(dft, swings_raw, t, iters=2):
        swings = swings_raw
        bos = []

        for _ in range(iters):
            lh_p, lh_i, ll_p, ll_i = compute_levels_confirmed_upto(t + 1, swings)
            bos = detect_bos_upto(dft, lh_p, lh_i, ll_p, ll_i, t)
            swings2 = filter_micro_structure(swings, bos)

            if len(swings2) == len(swings):
                swings = swings2
                break
            swings = swings2

        lh_p, lh_i, ll_p, ll_i = compute_levels_confirmed_upto(t + 1, swings)
        bos = detect_bos_upto(dft, lh_p, lh_i, ll_p, ll_i, t)
        return swings, bos

    def build_trades_upto(dft, swings, bos, t):
        rows = []

        def last_confirmed_swing_before(i):
            cand = [s for s in swings if s[3] <= i and s[0] < i]
            if not cand:
                return None
            return max(cand, key=lambda x: (x[3], x[0]))

        def last_bos_before(i):
            prev = [b for b in bos if b[0] < i]
            if not prev:
                return None
            return prev[-1][1]

        for i, tr in bos:
            if i > t:
                continue

            if not allow_countertrend:
                trend = last_bos_before(i)
                if trend is None or tr != trend:
                    continue

            last_sw = last_confirmed_swing_before(i)
            if require_prior_swing and last_sw is None:
                continue

            if tr == "down":
                if require_prior_swing and last_sw[2] != "high":
                    continue

                prot = [s for s in swings if s[2] == "high" and s[0] < i and s[3] <= i]
                if not prot:
                    continue

                prot_idx, px_sl = prot[-1][0], float(prot[-1][1])
                i_entry = min(i + 1, t) if use_next_open else i
                px_entry = float(dft["open"].iloc[i_entry]) if use_next_open else float(dft["close"].iloc[i])

                if px_sl <= px_entry:
                    continue

                risk = px_sl - px_entry
                px_tp = px_entry - rr * risk

                i_exit = t
                px_exit = float(dft["close"].iloc[t])
                hit = "none"

                for j in range(i_entry, t + 1):
                    hi = float(dft["high"].iloc[j])
                    lo = float(dft["low"].iloc[j])
                    if lo <= px_tp:
                        i_exit = j
                        px_exit = px_tp
                        hit = "tp"
                        break
                    if hi >= px_sl:
                        i_exit = j
                        px_exit = px_sl
                        hit = "sl"
                        break

                pnl = (px_entry - px_exit) - fee * px_entry - fee * px_exit

                rows.append(dict(
                    side="short",
                    bos_idx=i,
                    entry_idx=i_entry,
                    exit_idx=i_exit,
                    entry=px_entry,
                    sl=px_sl,
                    tp=px_tp,
                    exit=px_exit,
                    pnl=pnl,
                    R=pnl / risk,
                    hit=hit,
                    prot_idx=prot_idx
                ))

            else:
                if require_prior_swing and last_sw[2] != "low":
                    continue

                prot = [s for s in swings if s[2] == "low" and s[0] < i and s[3] <= i]
                if not prot:
                    continue

                prot_idx, px_sl = prot[-1][0], float(prot[-1][1])
                i_entry = min(i + 1, t) if use_next_open else i
                px_entry = float(dft["open"].iloc[i_entry]) if use_next_open else float(dft["close"].iloc[i])

                if px_sl >= px_entry:
                    continue

                risk = px_entry - px_sl
                px_tp = px_entry + rr * risk

                i_exit = t
                px_exit = float(dft["close"].iloc[t])
                hit = "none"

                for j in range(i_entry, t + 1):
                    hi = float(dft["high"].iloc[j])
                    lo = float(dft["low"].iloc[j])
                    if hi >= px_tp:
                        i_exit = j
                        px_exit = px_tp
                        hit = "tp"
                        break
                    if lo <= px_sl:
                        i_exit = j
                        px_exit = px_sl
                        hit = "sl"
                        break

                pnl = (px_exit - px_entry) - fee * px_entry - fee * px_exit

                rows.append(dict(
                    side="long",
                    bos_idx=i,
                    entry_idx=i_entry,
                    exit_idx=i_exit,
                    entry=px_entry,
                    sl=px_sl,
                    tp=px_tp,
                    exit=px_exit,
                    pnl=pnl,
                    R=pnl / risk,
                    hit=hit,
                    prot_idx=prot_idx
                ))

        return pd.DataFrame(rows)

    def render(t):
        with out:
            out.clear_output(wait=True)

            swings_raw = build_swings_confirmed_upto(df, window, t)
            swings, bos = stabilize_swings_and_bos(df, swings_raw, t, iters=2)
            trades = build_trades_upto(df, swings, bos, t)

            plt.figure(figsize=(14, 6))
            plt.plot(df["close"].iloc[:t + 1].values)

            for idx, val, typ, confirm in swings:
                if confirm <= t:
                    plt.scatter(idx, val, marker="^" if typ == "high" else "v", s=80)

            for i, _ in bos:
                plt.scatter(i, df["close"].iloc[i], s=120, alpha=0.5)

            for _, r in trades.iterrows():
                i0 = int(r["entry_idx"])
                i1 = int(r["exit_idx"])
                plt.plot([i0, i1], [r["sl"], r["sl"]], linestyle="--", alpha=0.6)
                plt.plot([i0, i1], [r["tp"], r["tp"]], linestyle="--", alpha=0.6)
                plt.plot([i0, i1], [r["entry"], r["exit"]], linestyle=":")

            plt.axvline(t)
            plt.title(
                f"t={t} | swings={sum(1 for s in swings if s[3] <= t)} | "
                f"bos={len(bos)} | trades={len(trades)}"
            )
            plt.grid(True)
            plt.tight_layout()
            plt.show()

    def on_next(_):
        state["t"] = min(state["t"] + 1, len(df) - 1)
        slider.value = state["t"]
        render(state["t"])

    def on_prev(_):
        state["t"] = max(state["t"] - 1, 5)
        slider.value = state["t"]
        render(state["t"])

    def on_slider(change):
        state["t"] = int(change["new"])
        render(state["t"])

    btn_next.on_click(on_next)
    btn_prev.on_click(on_prev)
    slider.observe(on_slider, names="value")

    display(widgets.HBox([btn_prev, btn_next, slider]))
    display(out)
    render(state["t"])
    return df