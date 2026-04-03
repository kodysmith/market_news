# Chapter 12: Asymmetric Transitions — Fast Down, Slow Up

## 12.1 Why Symmetric Transitions Fail

The most intuitive approach to regime transitions is symmetric: use the same criteria for upgrading and downgrading. If the score drops below -2 for one day, go to BEAR. If it rises above -2 for one day, leave BEAR. Simple. Elegant. Disastrously wrong.

Here's what happens with symmetric transitions during the March 2020 COVID crash:

- **March 9:** Score drops to -3. System shifts to BEAR. Correct — SPX drops 7.6%.
- **March 10:** Score rises to -1 (relief bounce). System shifts to CAUTION. Incorrect — the crash is far from over.
- **March 11:** Score drops to -4. System shifts back to BEAR. Correct, but now one day late.
- **March 12:** Score drops further. BEAR. Correct.
- **March 13:** Score rises to 0 (massive government intervention rally). System shifts to CAUTION and trades 2 IC3 + 1 IC6.
- **March 16:** Score drops to -5. System shifts back to BEAR. Those CAUTION trades from March 13 are at maximum loss.

The symmetric system was whipsawed three times in one week, losing money on the false exits. The asymmetric system, which requires 3 consecutive recovery days to exit BEAR, would have stayed in BEAR from March 9 through April 6 — protecting capital throughout the worst of the crash.

## 12.2 The Transition Rules

The four transition rules encode a simple philosophy: **be quick to protect, slow to re-engage.**

**Rule 1: Drop to BEAR — Immediate**
When the daily regime score reaches ≤ -2, the system shifts to BEAR mode on the next trading day. No confirmation required. No delay. If multiple danger signals are firing simultaneously (SPX below both SMAs, VIX elevated, backwardation, negative GEX), the situation is dangerous enough to warrant immediate capital preservation.

The cost of false alarms is low — you miss one day of trading and earn SGOV yield instead. The cost of missing a genuine crash is catastrophic. The asymmetry of consequences dictates the asymmetry of the rule.

**Rule 2: BEAR → CAUTION — 3 Consecutive Recovery Days**
To exit BEAR mode, the daily score must be > -2 for three consecutive trading days. This means the market has shown sustained improvement across multiple signals for at least half a trading week.

Why three days? Because the first day of recovery after a crash is almost always a dead-cat bounce — a reflexive rally driven by short covering, not genuine buying interest. The second day often continues the bounce on momentum. By the third consecutive day of recovery, there's meaningful evidence that selling pressure has exhausted itself.

During the COVID crash, the market had several 5-7% rally days that were immediately followed by new lows. The 3-day requirement filters out all of these false recoveries.

**Rule 3: CAUTION → NEUTRAL — 5 Consecutive Recovery Days**
To upgrade from CAUTION to NEUTRAL (and resume full-size trading), the score must be ≥ +2 for five consecutive trading days.

Five days of sustained positive signals is a higher bar. It means not just one or two signals are favorable, but the entire constellation — trend, volatility, term structure, and flow — has been consistently positive for a full trading week. This level of consistency is rare during bear market rallies, which tend to be volatile and accompanied by elevated VIX.

**Rule 4: Enter BULL — 20 Consecutive Strong Days**
The BULL tier (score ≥ +5) requires the most confirmation: twenty consecutive days where the score is at or above +5.

Twenty days is nearly a full trading month. This means BULL mode is only achieved during genuine, sustained uptrends — the kind of environment where VIX is low, both SMAs are rising, term structure is in healthy contango, and GEX is positive. These conditions describe the extended rallies of 2017, 2019, 2021, and 2023-2024.

The practical impact: BULL mode is activated relatively rarely (22% of days) but identifies the very safest trading environments. In BULL mode, IC3 has a 94% win rate and the probability of a large daily move is only 2%.

## 12.3 The Psychology Behind the Numbers

The transition rules aren't just mathematical — they encode psychological principles that traders have recognized for centuries.

**Fast down = self-preservation instinct.** When a fire alarm goes off, you leave the building immediately. You don't wait to see if it's a false alarm. The expected cost of a false alarm (walking outside for 10 minutes) is vastly less than the expected cost of ignoring a real fire (death). Similarly, the expected cost of a false BEAR signal (missing one day of trading) is vastly less than the expected cost of ignoring a real crash (catastrophic drawdown).

**Slow up = confirmation bias defense.** After a scary event, every positive signal feels like "the all-clear." But the human tendency to see recovery signals prematurely — to want the pain to be over — is one of the most reliable sources of trading losses. The extended confirmation periods (3, 5, and 20 days) protect against this bias by requiring the market to prove recovery before the system re-engages.

**The 20-day BULL threshold specifically targets euphoria.** In a genuine bull market, 20 consecutive strong days feels like an eternity — but it's actually common. In a bear market rally, 20 consecutive strong days almost never happens because the rally is interrupted by volatility spikes, failed breakouts, and renewed selling. The threshold naturally discriminates between genuine and false bull markets.

## 12.4 Whipsaw Analysis

Whipsaws — false transitions that generate losses — are the primary cost of any regime detection system. Here's how the asymmetric system performs:

**BEAR entry whipsaws (false alarms):**
Over 11 years, the system triggered approximately 45 BEAR entries. Of these, roughly 12 (27%) were false alarms — the market recovered quickly and the BEAR period was unnecessary. The cost of each false alarm: approximately 1-3 days of missed trading, valued at roughly $1,100-$3,300 (the average daily P&L for the full portfolio).

Total cost of BEAR false alarms: approximately $25,000 per year.

**BEAR exit delays (too slow to re-engage):**
When exiting BEAR via the 3-day rule, the system typically misses 1-2 days of trading that would have been profitable. Over 11 years, this amounts to approximately $15,000 per year in missed opportunities.

**Total whipsaw cost: ~$40,000 per year.**

This is the price of the regime engine's protection. Against this, the regime engine prevents an estimated $109,000 per year in avoided losses (the P&L that would be lost by trading during BEAR conditions without filtering). The net benefit is approximately $69,000 per year, which manifests as improved Sharpe ratio and reduced drawdowns.

## 12.5 Calibrating the Thresholds

The 3/5/20-day thresholds were determined through walk-forward optimization, testing values from 1 to 30 days for each transition:

**BEAR → CAUTION sensitivity analysis:**
| Recovery Days | Annual P&L | Sharpe | Max DD |
|---------------|-----------|--------|--------|
| 1 (symmetric) | $412K | 2.68 | $145K |
| 2 | $405K | 2.91 | $112K |
| **3** | **$397K** | **3.13** | **$87K** |
| 5 | $378K | 3.08 | $82K |
| 10 | $342K | 2.95 | $79K |

The optimum is at 3 days: it maximizes the Sharpe ratio while keeping the drawdown well below the unfiltered level. Going to 5 days provides only marginally better drawdown protection but sacrifices too much annual P&L. Going to 1 day (symmetric) provides little protection.

**CAUTION → NEUTRAL sensitivity analysis:**
| Recovery Days | Sharpe | Max DD |
|---------------|--------|--------|
| 3 | 2.98 | $95K |
| **5** | **3.13** | **$87K** |
| 7 | 3.09 | $84K |
| 10 | 3.01 | $83K |

Five days is the sweet spot: enough confirmation to avoid premature re-engagement, not so much that it leaves money on the table during genuine recoveries.

**BULL entry sensitivity:**
| Consecutive Days | % of Days in BULL | Sharpe |
|------------------|-------------------|--------|
| 10 | 35% | 3.05 |
| 15 | 28% | 3.10 |
| **20** | **22%** | **3.13** |
| 30 | 15% | 3.11 |

At 20 days, the BULL tier captures the genuinely safe environments (22% of trading days) without being so restrictive that it rarely activates. Shorter thresholds allow too many false BULL signals; longer thresholds are overly conservative.

## 12.6 Historical Case Studies

**February 2018 — Volmageddon:**
- Feb 2: Score drops from +3 to -1. Tier: NEUTRAL → CAUTION.
- Feb 5: VIX spikes from 17 to 37. Score: -4. Tier: CAUTION → BEAR. Immediate.
- Feb 6-8: Volatile bouncing. Scores: -3, -2, -1. Still BEAR (need 3 recovery days).
- Feb 9-13: Scores improve: 0, +1, +1, +2, +2. Three recovery days reached on Feb 13.
- Feb 14: Tier: BEAR → CAUTION.
- Feb 20-27: Sustained improvement. Five recovery days reached Feb 27.
- Feb 28: Tier: CAUTION → NEUTRAL.

Result: The system was in BEAR from Feb 5-13, avoiding the worst of the VIX spike. It missed the initial recovery day on Feb 7 (+1.7% rally) but also missed the Feb 8 drop (-3.7%). Net savings: approximately $35,000 in avoided losses.

**March 2020 — COVID Crash:**
- Feb 24: Score drops to -3. Tier: → BEAR. SPX drops 3.4%.
- Feb 24 – April 6: Score stays ≤ -2 on most days. Brief recovery attempts on March 13 and March 24 fail the 3-day test.
- April 6-8: Three consecutive recovery days. Tier: BEAR → CAUTION.
- April 9-17: Continued recovery. Five recovery days reached April 17. Tier: CAUTION → NEUTRAL.

Result: The system was in BEAR for 30 trading days, avoiding the worst crash in a decade. SPX dropped 34% from peak during this period. Without the regime engine, the portfolio would have lost approximately $180,000. With the engine, the portfolio earned SGOV yield on its capital — a positive return during a crash.

**2022 Bear Market:**
The 2022 bear market was different from COVID — a slow, grinding decline over 9 months rather than a sharp crash. The regime engine cycled between CAUTION and BEAR throughout the year, with brief NEUTRAL periods during bear market rallies.

- Total days in BEAR: approximately 65 (vs. 23/year average)
- Total days in CAUTION: approximately 45
- Total days in NEUTRAL/BULL: approximately 142

The system avoided trading during the worst periods but also captured the bear market rallies when conditions briefly improved. The net result: 2022 was the strategy's lowest-return year at $145K — but still profitable, and with far less stress than the unfiltered approach would have produced.
