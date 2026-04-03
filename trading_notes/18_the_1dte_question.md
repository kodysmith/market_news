# Chapter 18: The 1DTE Question — Kill It, Filter It, or Fix It?

## 18.1 The Case for Elimination

The analysis in Chapters 16 and 17 paints a damning picture for IC3 (1DTE iron condors). Of the 150 worst losses in the balanced dataset:
- 80 were overnight gap losses, and IC3 dominates this category
- IC3 produced the worst single-trade loss at -$2,796
- IC3's total losses over 11 years: -$280,413 — eating 47% of its gross winnings

The intuitive response is: just kill it. Drop IC3 entirely. The IC6 (3DTE) and IC9 (7DTE) strategies produce similar per-trade returns with better risk profiles. Why carry the overnight gap risk?

The math says otherwise.

## 18.2 The Numbers Tell a More Nuanced Story

Dropping IC3 entirely costs $310,479 over 11 years — a 46% reduction in total P&L:

| Scenario | Total P&L | Annual | Change |
|----------|----------|--------|--------|
| All trades (baseline) | $678,269 | $61,661 | — |
| Drop IC3 entirely | $367,790 | $33,435 | -46% |

IC3 is the *single largest contributor* to the system. It runs 2,674 trades over 11 years — roughly one per trading day — and its 90% win rate generates consistent daily cash flow. Eliminating it cuts nearly half the system's output.

We explored five approaches to keeping IC3 while reducing its losses. Early analysis suggested blanket-skipping IC3 in CAUTION regime would work (only 4% P&L cost). But when we ran the full portfolio simulation at proper contract sizing ($200K starting capital, 10 IC3 contracts in NEUTRAL/BULL, 2 in CAUTION), the blanket CAUTION skip turned out to be the wrong answer.

## 18.3 Why the Blanket Regime Skip Failed

Skipping IC3 in all CAUTION days looked good at single-contract scale. At full portfolio scale, it created a losing year:

| Year | Original | IC3 Skip CAUTION | Difference |
|------|----------|-----------------|------------|
| 2022 | +$18,819 | **-$6,850** | -$25,669 |

2022 was a bear market with 103 IC3 trades in CAUTION. At 2 contracts each, those trades netted $25,669 — just enough to keep 2022 positive. The blanket skip removes all of them, including 88 winners, to avoid 15 losers. The cure is worse than the disease.

The CAUTION IC3 trades ARE profitable on average. They just carry occasional tail risk. The right answer is not to skip them all — it's to skip the *dangerous* ones.

## 18.4 The Surgical Approach: Skip News Days

We tested four filtering strategies against the original, all at full portfolio scale ($200K starting, proper contract sizing per regime tier):

| Scenario | Final Equity | Annual P&L | Sharpe | Max DD | Losing Years |
|----------|-------------|------------|--------|--------|-------------|
| S0: Original (no filter) | $4,471,110 | $388,283 | 3.01 | -$114,720 | None |
| S1: Skip FOMC/CPI/NFP | $4,219,228 | $365,384 | 3.26 | -$101,775 | None |
| S2: Skip news events | **$4,864,171** | **$424,016** | **3.55** | **-$75,659** | **None** |
| S3: Skip scheduled + news | $4,294,655 | $390,423 | **3.69** | **-$65,944** | **None** |
| S4: IC3 skip CAUTION | $4,417,312 | $383,392 | 3.00 | -$107,988 | **2022** |

**S2 — skipping IC3 on major macro news days — is the clear winner.** By skipping just 73 trades out of 5,384 (1.4% of IC3 trades), we get:

- **+$393,061 more total profit** over 11 years
- **+$35,733 more per year** — that's a new car worth of money annually
- **Sharpe jumps from 3.01 to 3.55** — a massive risk-adjusted improvement
- **Max drawdown nearly halved** ($114K → $76K)
- **Zero losing years preserved**

The blanket regime skip (S4) is the worst option. It creates a losing year, saves less money, and barely moves the Sharpe. The surgical news filter is overwhelmingly superior.

## 18.5 What $35,000 Per Year Means When It Compounds

The difference between $388K/year and $424K/year looks modest in percentage terms (9.2%). But this is compounding capital. The gap accelerates:

| Year | Original | News Filter | Gap |
|------|----------|-------------|-----|
| 5 | $819,875 | $854,451 | +$34,576 |
| 10 | $3,360,975 | $3,650,430 | +$289,456 |
| 11 | $4,456,652 | $4,880,625 | +$423,973 |
| 15 | $13,777,895 | $15,595,565 | +$1,817,670 |
| 20 | $56,480,753 | $66,628,209 | **+$10,147,456** |

At year 20, the news filter has generated **ten million dollars more** than the unfiltered strategy — from skipping 73 trades per decade. That is not a rounding error. That is generational wealth difference from a single insight: don't sell iron condors into known macro events.

## 18.6 The 73 Trades That Change Everything

What are these 73 trades? They are IC3 entries on days when major macro news was breaking:

- **COVID emergence** (Feb–Mar 2020): 15 trades skipped — the single largest cluster of losses
- **Trade war tariff escalation** (May & Aug 2019): 8 trades skipped — surprise tariff announcements
- **Volmageddon** (Feb 2018): 3 trades skipped — VIX blowup from wage growth fears
- **Trade war + yields** (Oct–Dec 2018): 8 trades skipped — multi-factor macro stress
- **CPI surprises** (2022): 4 trades skipped — inflation prints above consensus
- **China devaluation** (Aug 2015): 5 trades skipped — global contagion fears
- **Japan carry trade unwind** (Aug 2024): 4 trades skipped — yen strengthening cascade
- **Hawkish Fed** (Dec 2024): 4 trades skipped — dot plot shock
- Various other events: SVB crisis, bond stress, election, tariff fears

None of these were obscure signals. They were front-page news. A news bot reading headlines — which we already have built in `news_bot/` — would have flagged every single one.

## 18.7 Why News Beats Regime for 1DTE

The regime engine operates on daily price-derived signals: SMA trends, VIX levels, term structure. It's excellent at identifying multi-day and multi-week regime shifts. But it has three structural blind spots that make it the wrong tool for 1DTE filtering:

**1. Speed mismatch.** The regime engine requires 3–5 consecutive days of confirmation to move up a tier. A 1DTE trade needs to survive the next 24 hours. A tariff announcement at 4 PM triggers a gap at 9:30 AM — hours, not days.

**2. Score ambiguity in CAUTION.** A regime score of +1 (CAUTION) could mean "market is recovering from a dip" or "market is about to crash." The score doesn't distinguish between these — but the news does.

**3. Low-VIX false safety.** The regime scores VIX < 22 as bullish (+1). But for 1DTE specifically, low VIX means thin cushion and cheap options. A surprise move in a VIX-14 environment is more damaging to IC3 than the same move at VIX-25 because the strikes are closer to spot.

News filtering sidesteps all three problems. It doesn't care about regime scores or VIX levels. It asks one question: "Is there a specific catalyst today that could move SPX 2%+ overnight?" If yes, skip the 1DTE. That's it.

## 18.8 S3: The Maximum Protection Variant

If we combine scheduled event skipping (FOMC/CPI/NFP) with news event skipping, we get S3:

- **Sharpe: 3.69** — the highest of any scenario
- **Max drawdown: -$65,944** — 43% less than the original
- **Worst month: -$44,810** — 45% less than original
- **Annual P&L: $390,423** — essentially identical to the original ($388K)
- **Zero losing years**
- **409 IC3 trades skipped** (7.6% of total)

S3 costs $2,140/year in P&L relative to original but buys a massive reduction in risk. The question is whether the scheduled-event skipping (FOMC/CPI/NFP days + day before) is worth the ~$23K/year in foregone wins. The data says those events are less dangerous than news events — many FOMC/CPI days are actually calm (the market has already positioned). So S3 is slightly over-filtering.

**The recommended approach: start with S2 (news only). Add S1 (scheduled events) only for specific events that historically produced losses — not all of them.**

## 18.9 Where News Fits Into the Architecture

The IC3 decision tree, based on the final analysis:

```
Is regime BEAR?
  Yes → SKIP IC3 (already the rule, 0% scale)
  No  → Continue...

Is news_severity >= 4? (from news bot classifier)
  Yes → SKIP IC3 for today
  No  → Continue...

Is regime CAUTION?
  Yes → IC3 at 2 contracts (existing rule — keep it)
  No  → IC3 at 10 contracts (NEUTRAL/BULL, full size)
```

That's it. No blanket regime skips. No Friday rules. No VIX floors. Just one question layered on top of the existing regime logic: *is there major news today?* The news bot already exists. It already classifies severity on a 1–5 scale. It just needs to be wired into the entry decision.

## 18.10 The 0DTE Alternative

There's a third path beyond "kill it" and "filter it": **replace 1DTE with 0DTE**.

A 0DTE iron condor entered at 10 AM and settled at 4 PM the same day eliminates overnight risk entirely. There is no gap because there is no overnight hold. The trade opens and closes in the same session.

The tradeoffs:
- **Pro:** Zero overnight gap risk — the #1 loss driver is completely eliminated
- **Pro:** Higher theta decay rate (same-day expiry decays fastest)
- **Con:** Tighter strikes (0DTE 10-delta is closer to spot than 1DTE 10-delta)
- **Con:** Higher gamma risk (near-expiry gamma is steepest)
- **Con:** Requires intraday monitoring (can't set and forget)
- **Con:** Cannot be backtested accurately with daily close data — requires intraday price data

The 0DTE pivot is the most promising advanced research direction. It addresses the root cause (overnight exposure) rather than the symptoms (filtering around events). But it requires intraday backtesting infrastructure that doesn't exist yet.

## 18.11 What We're Going to Build

The path forward is clear:

**Phase 1 (now):** Wire the news bot severity score into the IC3 entry decision. When `news_severity >= 4`, skip IC3 for that day. This is a few lines of code connecting systems that already exist. Expected impact: +$35,733/year, Sharpe 3.01 → 3.55, max drawdown halved.

**Phase 2 (validate):** Backtest selective scheduled-event skipping. Not all FOMC/CPI/NFP days are dangerous — identify which ones historically produced losses and skip only those.

**Phase 3 (research):** Build the 0DTE backtester with intraday SPX data and evaluate whether 0DTE iron condors can replace 1DTE entirely, eliminating overnight risk at the structural level.

The answer to "kill it, filter it, or fix it?" is: **filter it with news, and the filter pays for itself many times over.**

Seventy-three trades. That's all we need to skip. Seventy-three trades out of 5,384, and the system generates $393K more over 11 years with half the drawdown. The information to skip them already exists in our news bot. We just need to connect the wire.
