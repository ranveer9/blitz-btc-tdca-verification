"""
engine_generic.py - public generic implementation
==================================================
Independent re-implementation of the Blitz T-DCA reference event model
(BTCUSDT USD-M perpetual) used in the verification by Ranveer Verma
(github.com/ranveer9). Identical logic to the privately delivered engine,
except that EVERY strategy and cost parameter is read from a configuration
file at run time. No preset values are contained in this file.

    import engine_generic as E
    E.configure("path/to/presets.json")   # private file, not published

The exact configuration (ladder offsets, sizing weights, leverage, take-profit,
reset threshold, fee and maintenance settings) is private to Blitz Trading and
is NOT included in this repository. Without it these results cannot be
reproduced. See config_template.json for the expected structure.

Event order per aggTrade (reference model, Phase A):
  1. settle funding events due at or before this trade (funding scenarios)
  2a. flat -> open round 1 at the observed price, continue to next trade
  2b. open -> fill every crossed ladder level in full at its own level price
  3. mark equity (wallet + unrealised PnL) and update drawdown statistics
  4. price >= TP -> close cycle (fixed TP accounting), optional resize, continue
  5. otherwise evaluate the liquidation proxy (trade price, flat maintenance)

Limitations (see report): last price is not a conservative substitute for
mark price; a trade-through does not guarantee queue clearance or a full
fill; no partial fills, queue, latency or exchange maintenance tiers.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Window constants (milliseconds)
# ---------------------------------------------------------------------------
START_MS_INCLUSIVE: int = 1_609_459_200_000   # 2021-01-01T00:00:00Z
END_MS_EXCLUSIVE:   int = 1_787_616_000_000   # 2026-08-25T00:00:00Z

# ---------------------------------------------------------------------------
# Strategy / cost parameters: populated ONLY by configure(path)
# ---------------------------------------------------------------------------

INITIAL_WALLET:   float = float("nan")
LEVERAGE:         float = float("nan")
WALLET_USAGE:     float = float("nan")
ROUNDS:           int   = 0
REBAL_THRESHOLD:  float = float("nan")
TP_MARGIN_RETURN: float = float("nan")
TP_PRICE_RETURN:  float = float("nan")
ENTRY_FEE_RATE:   float = float("nan")
EXIT_FEE_RATE:    float = float("nan")
MAINT_FRACTION:   float = float("nan")
CUMULATIVE_DECLINES: list[float] = []
MARGIN_WEIGHTS:      list[float] = []
_CD_ARR = np.zeros(0, dtype=np.float64)
_MW_ARR = np.zeros(0, dtype=np.float64)
_MW_SUM = float("nan")
_CONFIGURED = False


def configure(path: str | Path) -> None:
    """Load all parameters from a presets.json-format file (private; not published)."""
    global INITIAL_WALLET, LEVERAGE, WALLET_USAGE, ROUNDS, REBAL_THRESHOLD
    global TP_MARGIN_RETURN, TP_PRICE_RETURN, ENTRY_FEE_RATE, EXIT_FEE_RATE, MAINT_FRACTION
    global CUMULATIVE_DECLINES, MARGIN_WEIGHTS, _CD_ARR, _MW_ARR, _MW_SUM, _CONFIGURED
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    p, a, lq = cfg["preset"], cfg["reference_accounting"], cfg["reference_liquidation"]
    INITIAL_WALLET = float(p["initial_wallet_usdt"])
    LEVERAGE = float(p["leverage"])
    WALLET_USAGE = float(p["wallet_usage_fraction"])
    ROUNDS = int(p["rounds"])
    REBAL_THRESHOLD = 1.0 + float(p["rebalance_wallet_growth_fraction"])
    TP_MARGIN_RETURN = float(p["take_profit_margin_return_fraction"])
    TP_PRICE_RETURN = float(p["take_profit_price_return_fraction"])
    ENTRY_FEE_RATE = float(a["entry_fee_fraction"])
    EXIT_FEE_RATE = float(a["exit_fee_fraction"])
    MAINT_FRACTION = float(lq["maintenance_fraction"])
    CUMULATIVE_DECLINES = [float(x) for x in p["cumulative_entry_decline_fractions"]]
    MARGIN_WEIGHTS = [float(x) for x in p["margin_weights"]]
    if len(CUMULATIVE_DECLINES) != ROUNDS or len(MARGIN_WEIGHTS) != ROUNDS:
        raise ValueError("ladder arrays do not match 'rounds'")
    _CD_ARR = np.array(CUMULATIVE_DECLINES, dtype=np.float64)
    _MW_ARR = np.array(MARGIN_WEIGHTS, dtype=np.float64)
    _MW_SUM = float(_MW_ARR.sum())
    SCENARIOS.clear()
    for s in cfg["scenarios"]:
        SCENARIOS.append(ScenarioConfig(s["id"], bool(s["funding_enabled"]),
                                        float(s["exit_slippage_fraction"])))
    _CONFIGURED = True


def _require_config() -> None:
    if not _CONFIGURED:
        raise RuntimeError("engine_generic: call configure(<private presets.json>) first. "
                           "The configuration is private and not included in this repository.")

# ---------------------------------------------------------------------------
# Fast date helpers using pure integer arithmetic (no datetime per trade)
# ---------------------------------------------------------------------------
# Days from Unix epoch (ms) → year, using integer division only.
# Accurate for 2021–2026 (no need for full Gregorian calendar edge cases).

_DAYS_PER_400Y = 146097
_DAYS_PER_100Y = 36524
_DAYS_PER_4Y   = 1461
_DAYS_PER_Y    = 365
_MS_PER_DAY    = 86_400_000

def _ms_to_year_int(ts_ms: int) -> int:
    """Extract UTC year from millisecond timestamp using integer arithmetic."""
    # Days since Unix epoch
    days = ts_ms // _MS_PER_DAY
    # Shift to year 1 (days since 0001-01-01 = days_since_epoch + 719162)
    n = days + 719162
    n400, n = divmod(n, _DAYS_PER_400Y)
    year = n400 * 400 + 1
    n100, n = divmod(n, _DAYS_PER_100Y)
    n4,   n = divmod(n, _DAYS_PER_4Y)
    n1,   n = divmod(n, _DAYS_PER_Y)
    year += n100 * 100 + n4 * 4 + n1
    if n1 == 4 or n100 == 4:
        year -= 1
    return year

def _ms_to_year(ts_ms: int) -> int:
    return _ms_to_year_int(ts_ms)

def _ms_to_date(ts_ms: int) -> str:
    """Return 'YYYY-MM-DD' string from millisecond timestamp."""
    import datetime
    return datetime.datetime.utcfromtimestamp(ts_ms / 1000.0).strftime("%Y-%m-%d")

# Cache for date strings — called ~2062 times (once per new day)
_date_cache: dict[int, str] = {}

def _ms_to_date_cached(ts_ms: int) -> str:
    day_key = ts_ms // _MS_PER_DAY
    if day_key not in _date_cache:
        _date_cache[day_key] = _ms_to_date(ts_ms)
    return _date_cache[day_key]


# ---------------------------------------------------------------------------
# Sizing helper
# ---------------------------------------------------------------------------

def compute_margins(wallet: float) -> list[float]:
    _require_config()
    scale = (wallet * WALLET_USAGE) / _MW_SUM
    return [w * scale for w in MARGIN_WEIGHTS]


# ---------------------------------------------------------------------------
# Scenario configuration
# ---------------------------------------------------------------------------

@dataclass
class ScenarioConfig:
    id:                     str
    funding_enabled:        bool
    exit_slippage_fraction: float


SCENARIOS: list[ScenarioConfig] = []   # populated by configure()


# ---------------------------------------------------------------------------
# Position state
# ---------------------------------------------------------------------------

@dataclass
class Position:
    anchor_price:    float = 0.0
    quantity:        float = 0.0
    entry_cost:      float = 0.0
    deployed_margin: float = 0.0
    entry_notional:  float = 0.0
    accrued_fees:    float = 0.0
    next_rung:       int   = 1
    max_rung:        int   = 0

    @property
    def is_open(self) -> bool:
        return self.quantity > 0.0

    @property
    def average_entry(self) -> float:
        if self.quantity == 0.0:
            return 0.0
        return self.entry_cost / self.quantity

    @property
    def tp_price(self) -> float:
        return self.average_entry * (1.0 + TP_PRICE_RETURN)

    def liquidation_price(self, wallet: float) -> float:
        avg      = self.average_entry
        notional = self.quantity * avg
        return avg - (wallet - notional * MAINT_FRACTION) / self.quantity


# ---------------------------------------------------------------------------
# Simulation state
# ---------------------------------------------------------------------------

@dataclass
class SimState:
    scenario:      ScenarioConfig
    wallet:        float
    reset_anchor:  float
    margins:       list[float]
    pos:           Position
    funding_idx:   int

    tp_count:      int   = 0
    liq_count:     int   = 0
    rebal_count:   int   = 0
    max_step:      int   = 0
    total_fee:     float = 0.0
    total_volume:  float = 0.0
    total_funding: float = 0.0
    total_slippage:float = 0.0

    equity_peak:    float = field(default=0.0)
    account_mdd_pct:float = 0.0
    account_mdd_at: int   = 0

    yearly: dict = field(default_factory=dict)
    cycle_log: list[dict] = field(default_factory=list)
    equity_samples: list[list] = field(default_factory=list)
    _last_sample_day: int = -1   # day key (ts_ms // MS_PER_DAY)

    _year_start_wallet: dict = field(default_factory=dict)
    _year_profit:       dict = field(default_factory=dict)
    _year_tp:           dict = field(default_factory=dict)
    _year_liq:          dict = field(default_factory=dict)
    _year_fee:          dict = field(default_factory=dict)
    _year_volume:       dict = field(default_factory=dict)
    _year_funding:      dict = field(default_factory=dict)
    _year_mdd_peak:     dict = field(default_factory=dict)
    _year_mdd_pct:      dict = field(default_factory=dict)

    open_at_end:    dict = field(default_factory=dict)
    terminated:     bool = False
    terminated_at_ms:int = 0

    @classmethod
    def new(cls, scenario: ScenarioConfig) -> "SimState":
        margins = compute_margins(INITIAL_WALLET)
        s = cls(
            scenario=scenario,
            wallet=INITIAL_WALLET,
            reset_anchor=INITIAL_WALLET,
            margins=margins,
            pos=Position(),
            funding_idx=0,
        )
        s.equity_peak = INITIAL_WALLET
        return s

    def current_equity(self, price: float) -> float:
        if self.pos.is_open:
            return self.wallet + (price * self.pos.quantity - self.pos.entry_cost)
        return self.wallet

    def _update_mdd(self, equity: float, ts_ms: int, year: str) -> None:
        if equity > self.equity_peak:
            self.equity_peak = equity
        dd_pct = (equity - self.equity_peak) / self.equity_peak * 100.0
        if dd_pct < self.account_mdd_pct:
            self.account_mdd_pct = dd_pct
            self.account_mdd_at  = ts_ms
        if year not in self._year_mdd_peak:
            self._year_mdd_peak[year] = equity
        if equity > self._year_mdd_peak[year]:
            self._year_mdd_peak[year] = equity
        ann_dd = (equity - self._year_mdd_peak[year]) / self._year_mdd_peak[year] * 100.0
        if year not in self._year_mdd_pct or ann_dd < self._year_mdd_pct[year]:
            self._year_mdd_pct[year] = ann_dd

    def _ensure_year(self, year: str) -> None:
        if year not in self._year_profit:
            self._year_profit[year]  = 0.0
            self._year_tp[year]      = 0
            self._year_liq[year]     = 0
            self._year_fee[year]     = 0.0
            self._year_volume[year]  = 0.0
            self._year_funding[year] = 0.0
            if year not in self._year_start_wallet:
                self._year_start_wallet[year] = self.wallet


# ---------------------------------------------------------------------------
# Funding rate loader
# ---------------------------------------------------------------------------

def load_funding_rates(path: Path) -> list[tuple[int, float]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    rates: list[tuple[int, float]] = []
    for item in raw:
        if isinstance(item, list):
            ts, rate = int(item[0]), float(item[1])
        elif isinstance(item, dict):
            ts   = int(item.get("fundingTime", item.get("timestamp", 0)))
            rate = float(item.get("fundingRate", item.get("rate", 0.0)))
        else:
            raise ValueError(f"Unexpected funding item format: {item!r}")
        if START_MS_INCLUSIVE <= ts < END_MS_EXCLUSIVE:
            rates.append((ts, rate))

    rates.sort(key=lambda x: x[0])
    log.info("Loaded %d funding events (window-filtered)", len(rates))
    return rates


# ---------------------------------------------------------------------------
# Fast aggTrade loader — pandas batch read per day
# ---------------------------------------------------------------------------

def load_day(csv_path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Load one daily CSV and return (prices, timestamps) as float64/int64 arrays,
    sorted stably by timestamp.

    Returns None if the file is missing or unreadable.
    """
    try:
        # Read only the two columns we need: price (col 1) and transact_time (col 5)
        df = pd.read_csv(
            csv_path,
            usecols=[1, 5],
            header=0,
            names=["price", "ts"],
            dtype={"price": np.float64, "ts": np.int64},
            engine="c",
        )
    except Exception as exc:
        log.error("Error reading %s: %s", csv_path, exc)
        return None

    if df.empty:
        return None

    # Filter invalid rows
    mask = (df["price"] > 0.0) & (df["ts"] >= START_MS_INCLUSIVE) & (df["ts"] < END_MS_EXCLUSIVE)
    prices = df["price"].to_numpy(dtype=np.float64)[mask.to_numpy()]
    stamps = df["ts"].to_numpy(dtype=np.int64)[mask.to_numpy()]
    if prices.size == 0:
        return None

    # Stable sort by timestamp (mergesort preserves original file order for equal ts)
    order = np.argsort(stamps, kind="mergesort")
    return prices[order], stamps[order]


# ---------------------------------------------------------------------------
# Core per-trade processor — unchanged logic, no datetime calls
# ---------------------------------------------------------------------------

def process_trade(
    state: SimState,
    price: float,
    ts_ms: int,
    funding_rates: list[tuple[int, float]],
) -> None:
    year = str(_ms_to_year_int(ts_ms))
    state._ensure_year(year)

    # 1. Funding
    if state.scenario.funding_enabled:
        while (state.funding_idx < len(funding_rates)
               and funding_rates[state.funding_idx][0] <= ts_ms):
            f_ts, f_rate = funding_rates[state.funding_idx]
            state.funding_idx += 1
            if state.pos.is_open:
                cost = state.pos.quantity * price * f_rate
                state.wallet        -= cost
                state.total_funding += cost
                state._year_funding[year] = state._year_funding.get(year, 0.0) + cost

    # 2a. Flat → Round 1 entry, then continue
    if not state.pos.is_open:
        p1  = price
        m0  = state.margins[0]
        qty = (LEVERAGE * m0) / p1
        fee = (LEVERAGE * m0) * ENTRY_FEE_RATE
        state.pos.anchor_price    = p1
        state.pos.quantity        = qty
        state.pos.entry_cost      = p1 * qty
        state.pos.deployed_margin = m0
        state.pos.entry_notional  = LEVERAGE * m0
        state.pos.accrued_fees    = fee
        state.pos.next_rung       = 1
        state.pos.max_rung        = 0
        return

    # 2b. DCA loop
    p1 = state.pos.anchor_price
    while state.pos.next_rung < ROUNDS:
        rung_idx   = state.pos.next_rung
        rung_price = p1 * (1.0 - CUMULATIVE_DECLINES[rung_idx])
        if price > rung_price:
            break
        m_i   = state.margins[rung_idx]
        qty_i = (LEVERAGE * m_i) / rung_price
        fee_i = (LEVERAGE * m_i) * ENTRY_FEE_RATE
        state.pos.quantity        += qty_i
        state.pos.entry_cost      += rung_price * qty_i
        state.pos.deployed_margin += m_i
        state.pos.entry_notional  += LEVERAGE * m_i
        state.pos.accrued_fees    += fee_i
        state.pos.max_rung         = rung_idx
        state.pos.next_rung        = rung_idx + 1
        if rung_idx + 1 > state.max_step:
            state.max_step = rung_idx + 1

    # 3. MTM equity observation
    equity  = state.current_equity(price)
    state._update_mdd(equity, ts_ms, year)

    day_key = ts_ms // _MS_PER_DAY
    if day_key != state._last_sample_day:
        state.equity_samples.append([ts_ms, round(equity, 2)])
        state._last_sample_day = day_key

    # 4. TP check
    if price >= state.pos.tp_price:
        _close_tp(state, price, ts_ms, year)
        return

    # 5. Liquidation check
    liq_price = state.pos.liquidation_price(state.wallet)
    if liq_price > 0.0 and price <= liq_price:
        _liquidate(state, ts_ms, year)


def _close_tp(state: SimState, price: float, ts_ms: int, year: str) -> None:
    pos  = state.pos
    scen = state.scenario

    gross_profit  = pos.deployed_margin * TP_MARGIN_RETURN
    exit_fee      = pos.entry_notional  * EXIT_FEE_RATE
    slippage_cost = (pos.quantity * pos.average_entry
                     * (1.0 + TP_PRICE_RETURN)
                     * scen.exit_slippage_fraction)
    net = gross_profit - pos.accrued_fees - exit_fee - slippage_cost

    state.wallet         += net
    state.tp_count       += 1
    state.total_fee      += pos.accrued_fees + exit_fee
    state.total_volume   += pos.entry_notional
    state.total_slippage += slippage_cost

    state._year_profit[year] = state._year_profit.get(year, 0.0) + net
    state._year_tp[year]     = state._year_tp.get(year, 0)        + 1
    state._year_fee[year]    = state._year_fee.get(year, 0.0)     + pos.accrued_fees + exit_fee
    state._year_volume[year] = state._year_volume.get(year, 0.0)  + pos.entry_notional

    state.cycle_log.append({
        "cycle_id":        state.tp_count,
        "open_ts_ms":      None,
        "close_ts_ms":     ts_ms,
        "close_date_utc":  _ms_to_date_cached(ts_ms),
        "max_rung_1based": pos.max_rung + 1,
        "anchor_price":    pos.anchor_price,
        "avg_entry":       pos.average_entry,
        "tp_price":        pos.tp_price,
        "quantity":        pos.quantity,
        "deployed_margin": pos.deployed_margin,
        "entry_notional":  pos.entry_notional,
        "gross_profit":    gross_profit,
        "entry_fees":      pos.accrued_fees,
        "exit_fee":        exit_fee,
        "slippage_cost":   slippage_cost,
        "net_profit":      net,
        "wallet_after":    state.wallet,
        "liquidated":      False,
    })

    if state.wallet >= state.reset_anchor * REBAL_THRESHOLD:
        state.reset_anchor = state.wallet
        state.margins      = compute_margins(state.wallet)
        state.rebal_count += 1

    state.pos = Position()


def _liquidate(state: SimState, ts_ms: int, year: str) -> None:
    pos = state.pos
    state.cycle_log.append({
        "cycle_id":        state.tp_count + state.liq_count + 1,
        "open_ts_ms":      None,
        "close_ts_ms":     ts_ms,
        "close_date_utc":  _ms_to_date_cached(ts_ms),
        "max_rung_1based": pos.max_rung + 1,
        "anchor_price":    pos.anchor_price,
        "avg_entry":       pos.average_entry,
        "tp_price":        pos.tp_price,
        "quantity":        pos.quantity,
        "deployed_margin": pos.deployed_margin,
        "entry_notional":  pos.entry_notional,
        "gross_profit":    0.0,
        "entry_fees":      pos.accrued_fees,
        "exit_fee":        0.0,
        "slippage_cost":   0.0,
        "net_profit":      -state.wallet,
        "wallet_after":    0.0,
        "liquidated":      True,
    })
    state._year_liq[year] = state._year_liq.get(year, 0) + 1
    state.liq_count       += 1
    state.wallet           = 0.0
    state.pos              = Position()
    state.terminated       = True
    state.terminated_at_ms = ts_ms


# ---------------------------------------------------------------------------
# Main simulation runner — pandas batch load per day
# ---------------------------------------------------------------------------

def run_scenario(
    scenario: ScenarioConfig,
    extracted_dir: Path,
    dates: list[str],
    funding_rates: list[tuple[int, float]],
) -> SimState:
    state = SimState.new(scenario)
    cycle_open_ts: int = 0

    log.info("Starting scenario: %s", scenario.id)

    trade_count = 0

    for date_str in dates:
        if state.terminated:
            break

        csv_path = extracted_dir / f"{date_str}.csv"
        day_data = load_day(csv_path)
        if day_data is None:
            log.warning("Missing or empty file for %s — skipping", date_str)
            continue

        prices, timestamps = day_data

        for i in range(len(prices)):
            if state.terminated:
                break

            price = float(prices[i])
            ts_ms = int(timestamps[i])

            was_flat  = not state.pos.is_open
            tp_before = state.tp_count

            process_trade(state, price, ts_ms, funding_rates)

            trade_count += 1

            if was_flat and state.pos.is_open:
                cycle_open_ts = ts_ms

            if state.tp_count > tp_before and state.cycle_log:
                state.cycle_log[-1]["open_ts_ms"] = cycle_open_ts

        if trade_count % 10_000_000 == 0:
            log.info(
                "  %s — %d trades processed, wallet=%.2f, tp=%d",
                scenario.id, trade_count, state.wallet, state.tp_count,
            )

    # Final open position record
    if state.pos.is_open:
        state.open_at_end = {
            "status":          "OPEN",
            "open_ts_ms":      cycle_open_ts,
            "round_1based":    state.pos.max_rung + 1,
            "average_entry":   state.pos.average_entry,
            "take_profit":     state.pos.tp_price,
            "quantity":        state.pos.quantity,
            "entry_cost":      state.pos.entry_cost,
            "deployed_margin": state.pos.deployed_margin,
            "entry_notional":  state.pos.entry_notional,
            "accrued_fees":    state.pos.accrued_fees,
        }
    else:
        state.open_at_end = {"status": "FLAT"}

    # Build yearly summary
    for yr, profit in state._year_profit.items():
        state.yearly[yr] = {
            "profit":       profit,
            "tp":           state._year_tp.get(yr, 0),
            "liq":          state._year_liq.get(yr, 0),
            "fee":          state._year_fee.get(yr, 0.0),
            "volume":       state._year_volume.get(yr, 0.0),
            "funding":      state._year_funding.get(yr, 0.0),
            "start_wallet": state._year_start_wallet.get(yr, 0.0),
            "mdd_pct":      state._year_mdd_pct.get(yr, 0.0),
        }

    log.info(
        "Scenario %s done — trades=%d, wallet=%.6f, tp=%d, liq=%d, rebal=%d",
        scenario.id, trade_count, state.wallet,
        state.tp_count, state.liq_count, state.rebal_count,
    )
    return state


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def build_date_list(extracted_dir: Path) -> list[str]:
    from datetime import date
    start = date(2021, 1, 1)
    end   = date(2026, 8, 24)
    dates = []
    for csv_file in sorted(extracted_dir.glob("*.csv")):
        stem = csv_file.stem
        try:
            d = date.fromisoformat(stem)
        except ValueError:
            continue
        if start <= d <= end:
            dates.append(stem)
    dates.sort()
    log.info("Found %d date files in %s", len(dates), extracted_dir)
    return dates