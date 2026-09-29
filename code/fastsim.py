"""
fastsim.py
==========
Compiled (Numba) implementation of the same T-DCA event model as src/engine.py,
used to run several simulations in ONE pass over the aggTrades data.

Mode 0  REFERENCE : operation-for-operation port of src/engine.py (Phase A). Must
                    reproduce the Python engine bit-exactly; the runner verifies this
                    against the delivered Phase A ledgers before any result is used.
                    Adds read-only recorders for the client's closure-time metrics
                    (per-cycle worst unrealised PnL and its price; closing Q, C, M, W).
Mode 1  F1 TICK   : ladder fills need a print at least one tick beyond the level price
                    (price <= level - tick); TP needs price >= TP + tick.
Mode 2  F2/F3 VOL : an order fills only when the cumulative aggTrade quantity printed at
                    or through its price since it became active reaches k x its size
                    (k = 1: no queue ahead; k = 2: queue ahead equal to own size).
                    All-or-nothing fills; ladder levels may fill out of order.
Mode 3  L1 LAT    : new cycle opens only on the first trade >= TP close + lat ms; ladder
                    and TP become active lat ms after the entry; after each additional
                    fill the TP is re-placed and active again lat ms later (no TP between).
Modes 1-3 are Phase B sensitivities (changed assumptions); they never alter mode 0.

Float state layout (fs[s, *]) and int state layout (ist[s, *]) are documented below.
"""

from __future__ import annotations

import numpy as np
from numba import njit

# ---- float state indices ----
F_WALLET, F_ANCHOR, F_P1, F_QTY, F_COST, F_DEP, F_NOTIONAL, F_ACC = 0, 1, 2, 3, 4, 5, 6, 7
F_PEAK, F_MDD, F_TFEE, F_TVOL, F_TFUND, F_TSLIP, F_WORST_U, F_WORST_P = 8, 9, 10, 11, 12, 13, 14, 15
F_VTP, F_TP_REF = 16, 17
NF = 18
# ---- int state indices ----
I_OPEN, I_NEXT, I_MAXR, I_TPC, I_LIQC, I_REB, I_MAXSTEP, I_MDDAT = 0, 1, 2, 3, 4, 5, 6, 7
I_TERM, I_FIDX, I_NCYC, I_OPENTS, I_LASTDAY, I_NEQ, I_TRADES = 8, 9, 10, 11, 12, 13, 14
I_REENTRY, I_RUNG_ACT, I_TP_ACT, I_OVERFLOW = 15, 16, 17, 18
NI = 19

# ---- cycle record columns (cf[s, c, *] floats, ci[s, c, *] ints) ----
C_WU, C_WP, C_Q, C_C, C_M, C_W, C_NOTIONAL = 0, 1, 2, 3, 4, 5, 6
NCF = 7
CI_OPEN, CI_CLOSE, CI_DEPTH, CI_LIQ = 0, 1, 2, 3
NCI = 4


@njit(cache=True)
def _record_cycle(s, ts, liq, fs, ist, cf, ci, qty, cost, dep, notional, wallet, depth):
    c = ist[s, I_NCYC]
    if c >= cf.shape[1]:
        ist[s, I_OVERFLOW] = 1
        return
    cf[s, c, C_WU] = fs[s, F_WORST_U]
    cf[s, c, C_WP] = fs[s, F_WORST_P]
    cf[s, c, C_Q] = qty
    cf[s, c, C_C] = cost
    cf[s, c, C_M] = dep
    cf[s, c, C_W] = wallet
    cf[s, c, C_NOTIONAL] = notional
    ci[s, c, CI_OPEN] = ist[s, I_OPENTS]
    ci[s, c, CI_CLOSE] = ts
    ci[s, c, CI_DEPTH] = depth
    ci[s, c, CI_LIQ] = liq
    ist[s, I_NCYC] = c + 1


@njit(cache=True)
def _mark(s, price, ts, fs, ist, eq_ts, eq_v):
    # identical to engine: equity = wallet + (price*qty - cost); mdd in percent
    unreal = price * fs[s, F_QTY] - fs[s, F_COST]
    eq = fs[s, F_WALLET] + unreal
    if eq > fs[s, F_PEAK]:
        fs[s, F_PEAK] = eq
    dd = (eq - fs[s, F_PEAK]) / fs[s, F_PEAK] * 100.0
    if dd < fs[s, F_MDD]:
        fs[s, F_MDD] = dd
        ist[s, I_MDDAT] = ts
    # client metric recorder: strict '<', first occurrence kept
    if unreal < fs[s, F_WORST_U]:
        fs[s, F_WORST_U] = unreal
        fs[s, F_WORST_P] = price
    day = ts // 86400000
    if day != ist[s, I_LASTDAY]:
        n = ist[s, I_NEQ]
        if n < eq_ts.shape[1]:
            eq_ts[s, n] = ts
            eq_v[s, n] = eq
            ist[s, I_NEQ] = n + 1
        ist[s, I_LASTDAY] = day


@njit(cache=True)
def _open_cycle(s, price, ts, fs, ist, margins, LEV, EF):
    m0 = margins[s, 0]
    q = (LEV * m0) / price
    fs[s, F_P1] = price
    fs[s, F_QTY] = q
    fs[s, F_COST] = price * q
    fs[s, F_DEP] = m0
    fs[s, F_NOTIONAL] = LEV * m0
    fs[s, F_ACC] = (LEV * m0) * EF
    fs[s, F_WORST_U] = 0.0
    fs[s, F_WORST_P] = 0.0
    fs[s, F_VTP] = 0.0
    ist[s, I_NEXT] = 1
    ist[s, I_MAXR] = 0
    ist[s, I_OPEN] = 1
    ist[s, I_OPENTS] = ts


@njit(cache=True)
def _fill_rung(s, r, rp, fs, ist, margins, LEV, EF):
    m = margins[s, r]
    qi = (LEV * m) / rp
    fs[s, F_QTY] += qi
    fs[s, F_COST] += rp * qi
    fs[s, F_DEP] += m
    fs[s, F_NOTIONAL] += LEV * m
    fs[s, F_ACC] += (LEV * m) * EF
    if r > ist[s, I_MAXR]:
        ist[s, I_MAXR] = r
    if r + 1 > ist[s, I_MAXSTEP]:
        ist[s, I_MAXSTEP] = r + 1


@njit(cache=True)
def _close_tp(s, ts, fs, ist, cf, ci, margins, MW, MW_SUM, TPM, TPP, XF, WU, REB, slip):
    qty = fs[s, F_QTY]
    cost = fs[s, F_COST]
    dep = fs[s, F_DEP]
    notional = fs[s, F_NOTIONAL]
    acc = fs[s, F_ACC]
    avg = cost / qty
    gross = dep * TPM
    exit_fee = notional * XF
    slip_cost = qty * avg * (1.0 + TPP) * slip
    net = gross - acc - exit_fee - slip_cost
    fs[s, F_WALLET] += net
    ist[s, I_TPC] += 1
    fs[s, F_TFEE] += acc + exit_fee
    fs[s, F_TVOL] += notional
    fs[s, F_TSLIP] += slip_cost
    _record_cycle(s, ts, 0, fs, ist, cf, ci, qty, cost, dep, notional, fs[s, F_WALLET], ist[s, I_MAXR] + 1)
    if fs[s, F_WALLET] >= fs[s, F_ANCHOR] * REB:
        fs[s, F_ANCHOR] = fs[s, F_WALLET]
        scale = (fs[s, F_WALLET] * WU) / MW_SUM
        for i in range(MW.shape[0]):
            margins[s, i] = MW[i] * scale
        ist[s, I_REB] += 1
    ist[s, I_OPEN] = 0
    fs[s, F_QTY] = 0.0


@njit(cache=True)
def _liq_check(s, price, ts, fs, ist, cf, ci, MMR):
    qty = fs[s, F_QTY]
    avg = fs[s, F_COST] / qty
    notional_q = qty * avg
    lp = avg - (fs[s, F_WALLET] - notional_q * MMR) / qty
    if lp > 0.0 and price <= lp:
        _record_cycle(s, ts, 1, fs, ist, cf, ci, qty, fs[s, F_COST], fs[s, F_DEP],
                      fs[s, F_NOTIONAL], fs[s, F_WALLET], ist[s, I_MAXR] + 1)
        ist[s, I_LIQC] += 1
        fs[s, F_WALLET] = 0.0
        ist[s, I_OPEN] = 0
        fs[s, F_QTY] = 0.0
        ist[s, I_TERM] = 1


@njit(cache=True)
def process_day(prices, stamps, qtys, fts, frates,
                CD, MW, MW_SUM, LEV, WU, TPM, TPP, EF, XF, MMR, REB,
                p_mode, p_fund, p_slip, p_k, p_tick, p_lat,
                fs, ist, margins, filled, vcount, cf, ci, eq_ts, eq_v):
    n = prices.shape[0]
    R = CD.shape[0]
    S = fs.shape[0]
    nf = fts.shape[0]
    for s in range(S):
        mode = p_mode[s]
        slip = p_slip[s]
        for i in range(n):
            if ist[s, I_TERM] == 1:
                break
            price = prices[i]
            ts = stamps[i]
            ist[s, I_TRADES] += 1

            # 1. funding (settled at this trade's price; skipped while flat)
            if p_fund[s] == 1:
                while ist[s, I_FIDX] < nf and fts[ist[s, I_FIDX]] <= ts:
                    if ist[s, I_OPEN] == 1:
                        c = fs[s, F_QTY] * price * frates[ist[s, I_FIDX]]
                        fs[s, F_WALLET] -= c
                        fs[s, F_TFUND] += c
                    ist[s, I_FIDX] += 1

            # 2a. flat -> open round 1, continue
            if ist[s, I_OPEN] == 0:
                if mode == 3 and ts < ist[s, I_REENTRY]:
                    continue
                _open_cycle(s, price, ts, fs, ist, margins, LEV, EF)
                if mode == 2:
                    for r in range(R):
                        filled[s, r] = 0
                        vcount[s, r] = 0.0
                    filled[s, 0] = 1
                if mode == 3:
                    ist[s, I_RUNG_ACT] = ts + p_lat[s]
                    ist[s, I_TP_ACT] = ts + p_lat[s]
                continue

            # 2b. ladder fills
            p1 = fs[s, F_P1]
            if mode == 0:
                while ist[s, I_NEXT] < R:
                    r = ist[s, I_NEXT]
                    rp = p1 * (1.0 - CD[r])
                    if price > rp:
                        break
                    _fill_rung(s, r, rp, fs, ist, margins, LEV, EF)
                    ist[s, I_NEXT] = r + 1
            elif mode == 1:
                while ist[s, I_NEXT] < R:
                    r = ist[s, I_NEXT]
                    rp = p1 * (1.0 - CD[r])
                    if price > rp - p_tick[s]:
                        break
                    _fill_rung(s, r, rp, fs, ist, margins, LEV, EF)
                    ist[s, I_NEXT] = r + 1
            elif mode == 2:
                changed = False
                for r in range(1, R):
                    if filled[s, r] == 1:
                        continue
                    rp = p1 * (1.0 - CD[r])
                    if price > rp:
                        break          # levels are descending: no deeper level is reached
                    vcount[s, r] += qtys[i]
                    own = (LEV * margins[s, r]) / rp
                    if vcount[s, r] >= p_k[s] * own:
                        _fill_rung(s, r, rp, fs, ist, margins, LEV, EF)
                        filled[s, r] = 1
                        changed = True
                if changed:
                    fs[s, F_VTP] = 0.0
            else:  # mode 3
                if ts >= ist[s, I_RUNG_ACT]:
                    filled_any = False
                    while ist[s, I_NEXT] < R:
                        r = ist[s, I_NEXT]
                        rp = p1 * (1.0 - CD[r])
                        if price > rp:
                            break
                        _fill_rung(s, r, rp, fs, ist, margins, LEV, EF)
                        ist[s, I_NEXT] = r + 1
                        filled_any = True
                    if filled_any:
                        ist[s, I_TP_ACT] = ts + p_lat[s]

            # 3. mark
            _mark(s, price, ts, fs, ist, eq_ts, eq_v)

            # 4. take-profit
            tp = (fs[s, F_COST] / fs[s, F_QTY]) * (1.0 + TPP)
            hit = False
            if mode == 0:
                hit = price >= tp
            elif mode == 1:
                hit = price >= tp + p_tick[s]
            elif mode == 2:
                if price >= tp:
                    fs[s, F_VTP] += qtys[i]
                    hit = fs[s, F_VTP] >= p_k[s] * fs[s, F_QTY]
            else:
                hit = ts >= ist[s, I_TP_ACT] and price >= tp
            if hit:
                _close_tp(s, ts, fs, ist, cf, ci, margins, MW, MW_SUM, TPM, TPP, XF, WU, REB, slip)
                if mode == 3:
                    ist[s, I_REENTRY] = ts + p_lat[s]
                continue

            # 5. liquidation proxy
            _liq_check(s, price, ts, fs, ist, cf, ci, MMR)