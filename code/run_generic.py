"""
run_generic.py - example runner for the public generic engine.

Requires the PRIVATE configuration file (not included in this repository):

    python code/run_generic.py --config presets.json --data data/extracted \
        --funding btc_funding_rates.json --scenario baseline

Prints the scenario totals. Runtime is several hours on the full 2021-2026 window.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import engine_generic as E


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="private presets.json (not published)")
    ap.add_argument("--data", required=True, help="folder with extracted daily aggTrades CSVs")
    ap.add_argument("--funding", required=True, help="funding-rate JSON")
    ap.add_argument("--scenario", required=True,
                    choices=["baseline", "funding_only", "funding_slip1bp", "funding_slip2bp"])
    a = ap.parse_args()

    E.configure(a.config)
    scenario = next(s for s in E.SCENARIOS if s.id == a.scenario)
    state = E.run_scenario(scenario, Path(a.data), E.build_date_list(Path(a.data)),
                           E.load_funding_rates(Path(a.funding)))
    print(json.dumps({
        "scenario": a.scenario, "final_wallet": state.wallet, "tp_count": state.tp_count,
        "liq_count": state.liq_count, "rebal_count": state.rebal_count, "max_step": state.max_step,
        "total_fee": state.total_fee, "total_volume": state.total_volume,
        "total_funding": state.total_funding, "total_slippage": state.total_slippage,
        "account_mdd_pct": state.account_mdd_pct,
    }, indent=2))


if __name__ == "__main__":
    main()