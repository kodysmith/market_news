# SPX Options Strategy Tracker

## Account: $700K total, $200K risk budget for options strategies

---

## FINAL STRATEGY: Regime-Aware Daily Cash Flow (2015-2025 backtest)

### 4-Tier Regime Engine + IC3 + IC6 + Quant Signals
- **Annual P&L: $397K** | Max DD: **-$87K (68% less than unfiltered)**
- **Sharpe: 3.13** | Zero losing years | 81% positive days
- **With compounding: BEATS unfiltered strategy over 11 years** (smaller DD preserves compound base)

### Regime Tiers
| Tier | % of Days | Action | Sizing |
|---|---|---|---|
| **BULL** | 22% | Full ICs | 10 IC3 + 5 IC6 |
| **NEUTRAL** | 53% | Full ICs | 10 IC3 + 5 IC6 |
| **CAUTION** | 15% | Reduced ICs, wider strikes | 2 IC3 + 1 IC6 |
| **BEAR** | 9% | Park in SGOV | No trades, protect capital |

### Transition Rules (asymmetric — fast down, slow up)
- **Drop to BEAR:** immediate on danger signal
- **Exit BEAR → CAUTION:** 3 consecutive recovery days
- **Exit CAUTION → NEUTRAL:** 5 consecutive recovery days
- **Enter BULL:** 20 consecutive strong days

### Regime Score (computed daily from T-1 data)
- +1: SPX > SMA50 | +1: SPX > SMA200 | +1: SMA200 slope up
- +1: VIX < 22 | +1: VIX contango | +1: VIX lower highs
- -1: SPX < SMA50 | -1: SPX < SMA200 | -1: VIX > 28
- -1: VIX backwardation | -1: GEX negative
- BEAR: score ≤ -2 | CAUTION: -1 to +1 | NEUTRAL: +2 to +4 | BULL: ≥ +5

### Additional Filters
- **Quant signals (danger ≥ 7):** Skip entry — GARCH vol expanding + high rv_percentile
- **Dynamic EVT stop:** Tail risk shapes stop multiplier (1.5x thin tails, 2.5x fat tails)
- **Portfolio risk:** SPX exposure < $150K, < 15% of account, < 6 concurrent spreads

---

## PREVIOUS APPROACH (superseded by regime engine)

### IC3(10cts) + IC6(5cts) + FLY(3cts) — Unfiltered
- **Annual P&L: $466,157** | Daily avg: $1,943
- **ROI on $200K: 233%** | ROI on $700K: 67%
- **90% positive days** (2362/2635)
- Margin needed: $51,150 (only 26% of risk budget used)
- Sharpe: 2.89 | Every year profitable (worst: 2018 at $136K, best: 2025 at $901K)
- Worst single day: -$78,702 (rare tail event)
- Trades in ALL market regimes — no directional bias

---

## TIER 1 — DAILY CASH FLOW (Highest Consistency)

### IC3: 1DTE Iron Condor, 10-delta, $30 wide, TP50%
- **Win rate: 90%** | Loss rate: 10% | Sharpe: 2.54
- **91% positive days** (1830/2003 trading days)
- Annual P&L per contract: $28,245
- At 10 contracts ($33K margin): **$282K/yr, $1,121/day avg, 141% ROI on $200K**
- Avg hold: 1.9 days | Exits: 55% expiry, 39% profit target, 6% stop loss
- Worst day: -$6,349 (per contract) | Best day: +$1,180
- **Trades in ALL regimes** — no SMA filter needed
- Every year profitable 2015-2025 (worst year: 2015 at $7,708/ct, best: 2025 at $59K/ct)

### IC6: 3DTE Iron Condor, 12-delta, $30 wide, TP40%
- **Win rate: 88%** | Loss rate: 12% | Sharpe: 2.56
- Annual P&L per contract: $28,493
- At 10 contracts ($33K margin): **$285K/yr, $1,131/day avg, 142% ROI on $200K**
- Avg hold: 2.7 days | Slightly more exposure than 1DTE but similar returns
- **Trades in ALL regimes**

### FLY14_D10_W10_R25: Funded Butterfly (14DTE, 10-delta funder, $10 wing)
- **Win rate: 99%** | Loss rate: 1% | PF: 9.1 | Sharpe: 3.65
- Annual P&L per contract: ~$13,600
- Butterfly hits 11% of the time for $545 avg payout (10:1 on fly cost)
- Net credit: $255 avg (you get PAID to take the trade)
- **Near-zero risk** — worst year was 2018 at $5,657/ct still profitable
- Good as add-on income layer

---

## TIER 2 — WEEKLY/BIWEEKLY INCOME (Current Bot Strategies)

### 15_VIX_SWEET_SPOT (SWEET): 14DTE IC, 15δ, $50w, VIX 16-24
- Win rate: 89% | Sharpe: 2.95 | Calmar: 3.38
- Annual P&L per contract: ~$5,145
- At 8 contracts: ~$41K/yr
- **Only trades when SPX > SMA50 AND VIX 16-24** (~11 trades/yr)

### SD1_7DTE_BASE: 7DTE IC, 15δ, $50w
- Win rate: 87% | Sharpe: 2.57 | Calmar: 3.21
- Annual P&L per contract: ~$2,980
- At 8 contracts: ~$24K/yr
- **Only trades when SPX > SMA50** (~34 trades/yr)

### AG2_VOLDET: 21DTE IC, 20δ, $75w, rv_ratio ≤ 1.05
- Win rate: 89% | Sharpe: 2.57
- Annual P&L per contract: ~$14,780
- At 6 contracts: ~$89K/yr
- **Only trades when SPX > SMA50** (~20 trades/yr)

### VD2_ENTRY_105: 14DTE IC, 15δ, $50w, compounds
- Win rate: 90% | Sharpe: 2.94
- Annual P&L per contract: ~$3,750
- At 6-10 contracts: ~$23-38K/yr
- **Compounds +1 contract every $30K banked profit**

### BC4_BEAR_CALL_CONSERVATIVE: 14DTE bear call, 10δ, $50w
- Win rate: 89% | Sharpe: 1.89
- Annual P&L per contract: ~$1,013
- At 6 contracts: ~$6K/yr
- **Only trades when SPX < SMA50** (bearish regime)

### BC2_BEAR_CALL_7DTE: 7DTE bear call, 15δ, $50w
- Win rate: 83% | Sharpe: 1.54
- Annual P&L per contract: ~$1,335
- At 6 contracts: ~$8K/yr
- **Only trades when SPX < SMA50** (bearish regime)

---

## TIER 2 COMBINED (Portfolio Backtest 2015-2025)
- Total P&L: $3,020,407 over 11 years
- Annual avg: $280,968
- CAGR: 16.8%
- Max DD: $238K (34% of starting capital)
- Sharpe: 2.77
- $700K → $3.72M
- Avg capital deployed: $87K (12.4% of account)

---

## COMPARISON: Which to Run?

| Approach | Annual P&L (10 cts) | Daily Avg | Win% | Capital Needed | ROI on $200K |
|---|---|---|---|---|---|
| **IC3 (1DTE daily)** | **$282K** | **$1,121** | **90%** | $33K | **141%** |
| **IC6 (3DTE daily)** | **$285K** | **$1,131** | **88%** | $33K | **142%** |
| Tier 2 combined (bot) | $281K | $1,115 | 87% | $87K avg | ~140% |
| Funded butterfly | $136K | $540 | 99% | $5.5K | 680% |

**Key insight:** The 1DTE and 3DTE daily strategies match the annual return of the full Tier 2 bot portfolio, but with less capital deployed and no directional bias. They trade every day regardless of SMA50 regime.

---

## RISKS & CAVEATS

- All backtests use BS pricing, not actual IBKR fills
- Slippage assumed $0.10/leg — real 0DTE/1DTE SPX spreads can have wider spreads
- 0DTE/1DTE strategies require execution during market hours (automatable via bot)
- Tail events (gaps, flash crashes) can blow through stops
- 2018 and 2022 were worst years but still profitable across all strategies
- Commission: $0.65/leg included in all P&L figures

---

## RESEARCH COMPLETED — Box Spread Conversion

**Tested:** Convert winning spreads to box spreads at 50% TP, park freed margin in SGOV/BOXX.

**Result:** Standard close wins by ~$5,850/yr. Box conversion adds ~$170/trade in extra slippage/commission. SGOV yields only ~$33/trade on 9 days of freed margin. Net loss of $137 per conversion.

**Conclusion:** Box conversion is NOT worth it as a systematic exit. Only use tactically:
- Before weekends with elevated risk
- When you need margin for a higher-conviction trade
- When position is at 70%+ profit and you want zero risk of reversal

**The only scenario where box conversion wins:** reinvesting freed margin into NEW TRADES at 50%+ annualized. But that requires a new opportunity at the exact right time — not reliable.

## NEXT STEPS
- [ ] Test with actual IBKR paper fills to validate slippage assumptions
- [ ] Explore GEX-wall butterfly targeting (needs historical options chain data)
- [ ] Start daily Massive API collection for real GEX training data
- [ ] Improve SSAN model with more data (intraday bars, options flow)
