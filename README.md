# Independent Verification: Blitz T-DCA BTCUSDT Perpetual Backtest

**Author:** Ranveer Verma ([github.com/ranveer9](https://github.com/ranveer9)) · **Version:** v1.1.0 · **Report date:** 2026-09-29
**Commissioned by:** Blitz Trading. The work was commissioned; the method, results and conclusions are the author's own, including all limitations and negative findings.

This is a historical verification of a backtest. It is not a guarantee of live trading performance.

📄 **Full report:** [`blitz_btc_verification_report.pdf`](blitz_btc_verification_report.pdf) (public edition)

## Scope

| Item | Detail |
|---|---|
| Market | Binance BTCUSDT USD-M linear perpetual, long only |
| Preset | N18/U75 (`btc-futures-s14-40-n18-u75-20260824-public`), one preset |
| Cost scenarios | Baseline; historical funding; funding + 1 bp TP slippage; funding + 2 bp TP slippage |
| Window | 2021-01-01 00:00 UTC (inclusive) to 2026-08-25 00:00 UTC (exclusive) |
| Data | 2,062 daily aggTrades archives from data.binance.vision, each verified against its official checksum; 3,198,919,867 trades |
| Excluded | ETH, the 2026-08-25 to 2026-09-30 out-of-sample window, optimisation studies |

## Phase A: reproduction under the original assumptions (reproduced)

| Scenario | Independent final wallet (USD) | Published (USD) | Difference | TP cycles | Max drawdown | Result |
|---|---|---|---|---|---|---|
| Baseline (fees only) | 2,201,183.98 | 2,201,183.98 | 4.7e-10 | 14,018 | -81.3140% | Match |
| Historical funding | 1,801,697.35 | 1,801,697.35 | 0.0e+00 | 14,018 | -78.1736% | Match |
| Funding + 1 bp TP slippage | 1,593,044.04 | 1,593,044.04 | 0.0e+00 | 14,018 | -80.7259% | Match |
| Funding + 2 bp TP slippage | 1,410,768.97 | 1,410,768.97 | -4.7e-10 | 14,018 | -81.6940% | Match |

All twelve compared totals per scenario agree within the stated tolerances. So do the annual figures, the drawdown timestamps, all 14,018 of 14,018 baseline cycle open and close times and ladder depths, and 8,248 of 8,248 daily equity samples across the four scenarios. Boundary-case checks: 12 of 12 pass. The final realised wallet excludes an open position at the cutoff; the report gives a separately labelled net terminal equity.

## Phase B: adequacy of the assumptions (key findings)

Reproducing the model does **not** validate it as a description of exchange execution.

- **Liquidation:** modelled on trade (last) prices with a flat maintenance fraction. **Last price is not inherently a conservative substitute for mark price**, and Binance's notional-tiered maintenance is not modelled. The modelled path came within 6.04% of its liquidation price. Zero modelled liquidations do not prove exchange-level survival. The mark-price effect is not quantified.
- **Fills:** every ladder order and TP is assumed to fill in full on any print at or through its price. **A one-tick trade-through does not guarantee queue clearance or a full fill.** Partial fills, queue position, liquidity and latency are not modelled.
- **Latency:** 2,937 of 14,018 cycles lasted under one second (1,025 opened and closed in the same millisecond), contributing 15.2% of closed-cycle profit. They depend on zero-latency order placement and instantaneous TP replacement.
- **Fees:** charging a 0.05% taker rate on the immediate first entry would cost about 6.1% of closed-cycle net profit (first order). Maker exits would instead reduce modelled exit cost by about 13.0%. The net effect depends on the live order types.
- **Cost sensitivity:** funding, 1 bp and 2 bp exit slippage reduce the final wallet by 18.1%, 27.6% and 35.9%.
- **Drawdown and resets:** account MTM drawdown reached -81.3140%. Compounding assumes automatic capital resets that the live bot does not perform.

## Added in v1.1.0

**Client closure-time risk metrics** (MAE, capital utilisation, liquidation room), recomputed from definitions supplied after v1.0.0: **64 of 64** published values agree exactly (report Section 3.8).

**Quantified fill and latency sensitivities** on the baseline cost scenario (report Section 4.7), run with a compiled engine that reproduces all four Phase A runs bit-for-bit:

| Rule | Final wallet (USD) | vs reproduced baseline | TP cycles | Max drawdown | Liquidation |
|---|---|---|---|---|---|
| F1 one-tick trade-through | 2,249,550.34 | +2.20% | 14,158 | -86.30% | no |
| F2 volume at/through >= 1x size | 1,653,363.98 | -24.89% | 12,445 | -91.89% | no |
| F3 volume at/through >= 2x size | 1,620,936.43 | -26.36% | 12,311 | -92.81% | no |
| L1 1-second latency | 1,310,070.72 | -40.48% | 11,304 | -80.91% | no |

aggTrade quantity is a proxy for executable volume; queue ahead is assumed and fills are all-or-nothing.

## ⚠️ What this public repository does NOT contain

At the client's written instruction, the following are **private** and omitted here: exact ladder offsets, sizing weights, preset configuration values (leverage, take-profit, reset threshold, fee and maintenance settings) and all trade/cycle-level records. `code/engine_generic.py` contains the full event logic but reads every parameter from a private configuration file (structure in `code/config_template.json`).

**This public package therefore does not enable full public reproduction of the results.** The complete configuration, code and records were delivered privately to Blitz Trading. No material finding is withheld; all findings are in the report.

## Contents

```
blitz_btc_verification_report.pdf     public edition of the report
results/scalar_comparison.csv         scenario totals: independent vs published
results/annual_comparison.csv         annual figures: independent vs published
results/verification_summary.json     verdicts and aggregate comparison statistics
results/phase_b_additions.csv         v1.1.0: client risk metrics and fill/latency sensitivities
code/fastsim.py                       compiled multi-simulation engine used for v1.1.0 (no configuration inside)
code/engine_generic.py                generic engine (parameters loaded from private config)
code/run_generic.py                   example runner
code/downloader.py                    data retrieval with official checksum verification
code/config_template.json             structure of the required (private) configuration
requirements.txt                      Python 3.12 dependencies
```

## Data retrieval

`https://data.binance.vision/data/futures/um/daily/aggTrades/BTCUSDT/BTCUSDT-aggTrades-YYYY-MM-DD.zip` (+ `.zip.CHECKSUM`) for every day from 2021-01-01 to 2026-08-24 (2,062 files; about 3.2 billion trades, roughly 150 GB once extracted).

`code/downloader.py` takes a manifest CSV with one row per day and the columns `utc_date`, `archive_url`, `checksum_url` (build it from the URL pattern above; the client's own manifest is not part of this repository). It downloads each archive, verifies it against the official checksum, extracts it, and can be re-run safely: days already extracted are skipped.

```
python code/downloader.py --manifest manifest.csv --raw-dir data/raw --out-dir data/extracted --log download_log.csv --workers 4
```

## References

- Blitz Trading public evidence: https://github.com/BlitzTrading/blitz-backtest-evidence (release `snapshot-20260824`)
- Archived snapshot: https://doi.org/10.5281/zenodo.22809597

## License

Code: MIT (see `LICENSE`). Report: © 2026 Ranveer Verma. Blitz Trading may host the full report with attribution and quote it accurately under the agreed review terms.
