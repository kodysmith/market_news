# Chapter 17: The Information Gap — What Our System Doesn't Know

## 17.1 Quantifying What's Missing

The trade autopsy in Chapter 16 revealed that our losses are not random. They cluster around macro events, they share common patterns, and — critically — most of them occurred when publicly available information would have changed the entry decision.

To quantify this, we tagged every losing trade in the balanced dataset with the signals that were *absent* from our entry decision but *would have flagged* the risk. We then ranked these missing signals by total dollars they would have saved:

| Missing Signal | Trades Saved | Total $ Saved | Avg $/Trade |
|----------------|-------------|---------------|-------------|
| Overnight Gap Protection | 80 | $190,388 | $2,380 |
| News Sentiment | 57 | $130,290 | $2,286 |
| Intraday Price Action | 38 | $92,367 | $2,431 |
| Volatility Regime Speed | 33 | $68,104 | $2,064 |
| Cross-Asset Stress | 29 | $61,585 | $2,124 |
| Economic Calendar | 25 | $57,053 | $2,282 |

The top six missing signals, if perfectly implemented, would have prevented $599,787 in losses — more than our annual P&L of $397,000. Even at 50% effectiveness, that's $300,000 saved. The cost of implementing these signals is effectively zero (API calls, data processing, a few hundred lines of code).

This is the most important finding of the research: **the biggest source of alpha available to us is not a better strategy — it's better information at the point of entry.**

## 17.2 Gap #1: Overnight Gap Protection ($190K)

**The problem:** 80 of our worst trades — more than half the loser dataset — were 1DTE positions (IC3) that held overnight. SPX gapped through the short strikes before the market opened, and no stop-loss could execute during the gap.

**What we already have:** Chapter 15 documents the overnight hedging strategy using ATM puts. The backtest shows it adds 59% to returns and reduces max drawdown by 33%.

**What's missing from the entry decision:** The backtest in Chapter 15 applies hedging uniformly to all overnight holdings. But the data shows that overnight gaps cluster around news events. Of the 80 overnight gap losses:
- 47 occurred on days with identifiable macro news catalysts
- 18 occurred on Fridays (weekend gap risk)
- 9 occurred before scheduled high-impact economic releases (CPI, FOMC, jobs)
- 6 occurred with no identifiable catalyst (truly random)

**The implementation path:**
1. **Immediate:** Flag 1DTE positions entered on Fridays for mandatory hedging or skip
2. **Short-term:** Wire the economic calendar into the entry decision — skip 1DTE entries before CPI, FOMC, jobs reports
3. **Medium-term:** Integrate the news bot's real-time sentiment — if negative macro news is trending, either skip the 1DTE or switch to 0DTE (same-day expiry, no overnight hold)
4. **Advanced:** Close 1DTE positions 15 minutes before close on elevated-risk days instead of holding to expiry

**Expected savings at 50% effectiveness:** $95,194

## 17.3 Gap #2: News Sentiment ($130K)

**The problem:** 57 of our worst losing trades occurred on days where publicly available news headlines would have signaled elevated risk. These weren't obscure signals — they were front-page stories: "US raises tariffs to 25%," "COVID cases surge in Italy," "CPI hits 8.6%."

**What we already have:** The project has a complete news intelligence system:
- `news_bot/news_aggregator.py`: Multi-source aggregation (FMP, NewsAPI, RSS, Alpha Vantage)
- `news_bot/classifier.py`: 5-level importance scoring with macro event detection
- `news_bot/llm_summarizer.py`: Claude-powered summarization and sentiment extraction
- `bot/market_data.py`: An `is_news_day` flag that exists but is rudimentary

**What's missing:** The news bot runs as a separate service. Its output is not connected to the entry decision in the backtest or in the entry engine in any systematic way. The `is_news_day` flag in `market_data.py` is a simple boolean that doesn't capture the *type* or *severity* of news.

**The specific events that would have been caught:**

| Date | News Event | Trades Lost | $ Lost |
|------|-----------|-------------|--------|
| 2020-02-19 to 02-28 | COVID spreading beyond China | 5 | $13,453 |
| 2019-05-10 | US-China tariffs raised to 25% | 2 | $5,571 |
| 2019-08-02 | Surprise tariff on $300B goods | 2 | $5,388 |
| 2018-02-02 | Wage growth → inflation fears | 1 | $2,761 |
| 2018-10-17 | Trade war + Fed + yields | 2 | $5,714 |
| 2022-06-10 | CPI 8.6% surprise | 2 | $5,269 |
| 2022-09-12 | CPI 8.3% surprise | 1 | $2,627 |
| 2024-08-02 | Jobs miss + recession fears | 1 | $2,602 |
| 2024-12-16 | Hawkish Fed expectations | 1 | $2,617 |

**The implementation path:**
1. **Immediate:** Add a `news_severity` field to the market snapshot (1–5 scale from the existing classifier)
2. **Short-term:** Entry engine checks `news_severity >= 4` before placing trades. Level 4+ (critical macro events) triggers SKIP or HALF_SIZE
3. **Medium-term:** Backtest the news filter historically by mapping known events to dates and simulating filtered entries
4. **Advanced:** Real-time LLM analysis of breaking news to assess directional risk to short premium positions specifically

**The key insight:** We don't need the news bot to predict the market direction. We just need it to identify *elevated uncertainty*. When a tariff announcement hits, we don't know if SPX will go up or down — but we know it will move *more than usual*. That's enough to skip the trade.

**Expected savings at 50% effectiveness:** $65,145

## 17.4 Gap #3: Intraday Price Action ($92K)

**The problem:** 38 losing trades showed zero max adverse excursion in the backtest data — meaning the backtest, which operates on daily closes, never saw an intermediate signal that the position was in trouble. The trade went from entry to expiry (or overnight gap) with no opportunity for a daily-resolution stop to trigger.

**What we already have:** The bot runs on a daily schedule (3:50 PM ET entry). Position management checks happen at open and close.

**What's missing:** For 1DTE positions, "daily" resolution means exactly one checkpoint between entry and expiry. If SPX drops 2% at the open and recovers by close, our system never saw it. If SPX is flat at 3:50 PM and drops 2% after hours, we don't know until the gap is already priced in.

**The implementation path:**
1. **Short-term:** Add a midday position check (12:00 PM ET) for all open positions. If unrealized loss exceeds 1x credit, close immediately
2. **Medium-term:** Intraday monitoring via IBKR's streaming API — set price alerts on the underlying at short strike levels
3. **Advanced:** 0DTE pivot — for the IC3 slot, switch from 1DTE to 0DTE entries after 10 AM. This eliminates overnight risk entirely (0DTE settles same day) while capturing the same theta decay

**Expected savings at 50% effectiveness:** $46,184

## 17.5 Gap #4: Volatility Regime Speed ($68K)

**The problem:** 33 losing trades occurred when VIX was below 18 at entry and then exploded (average VIX change of +7.3 points). The regime engine scored these days as NEUTRAL (score +4) because all price-based signals were positive. VIX was low, SPX was above both SMAs, SMA200 was rising. Everything looked perfect — until it wasn't.

**What we already have:**
- GARCH volatility forecasting in `quant_signals.py`
- `vol_expanding` flag when GARCH forecast exceeds realized vol
- Bayesian regime switching model with transition probability
- Danger score composite (0–10)

**What's missing:** These signals exist in the bot but are not used in the backtest or reflected in the training data. More importantly, the GARCH model forecasts *gradual* vol expansion — it doesn't handle *regime breaks* (sudden jumps from VIX 14 to VIX 30 in one day).

**The critical sub-problem: VIX < 16 is a risk factor, not a safety signal.**

When VIX is below 16:
- Options are cheap (less credit collected per trade)
- Strikes are closer to spot (less cushion)
- The market is priced for continued calm
- Any surprise produces a disproportionate move
- Hedge fund protection is cheap so nobody owns it — creating a "vacuum" of selling pressure when fear arrives

Our data proves this: of the 26 LOW_VOL_SURPRISE losses, 19 entered with VIX below 18 and the average loss was $2,121 — larger than the shock event average of $2,058. Low vol losses are *bigger* per trade because the credit cushion is smaller.

**The implementation path:**
1. **Immediate:** Add a VIX floor filter: when VIX < 15, reduce position size to 50% or widen to 8-delta
2. **Short-term:** Include GARCH forecast in the training data so the ML model can learn vol-expansion patterns
3. **Medium-term:** Implement a "complacency detector" — when VIX is below its 20th percentile AND the VIX of VIX (VVIX) is elevated, flag as "calm but fragile"
4. **Advanced:** Cross-reference VIX level with options market maker positioning (gamma exposure). When GEX is high and VIX is low, market makers are suppressing volatility — but if GEX flips negative, the suppression ends and vol explodes

**Expected savings at 50% effectiveness:** $34,052

## 17.6 Gap #5: Cross-Asset Stress Signals ($62K)

**The problem:** 29 losing trades occurred during periods where stress was visible in other markets before it hit SPX. The bond market (TLT), credit spreads (HYG), the dollar (DXY), and the Japanese yen were all signaling trouble before SPX dropped.

**What we already have:** Nothing. The regime engine and quant signals operate exclusively on SPX and VIX data. No cross-asset data is used.

**The specific cross-asset signals that preceded our losses:**

- **August 2024 (Japan carry trade unwind):** USD/JPY dropped from 153 to 142 in the week before SPX crashed. The yen strengthening was the *cause* of the equity selloff — leveraged carry trades unwinding. Our system had zero visibility into currency markets.
- **October 2018 (trade war + yields):** The 10-year Treasury yield spiked above 3.2% — the highest in 7 years. This was the *trigger* for the equity rotation. Bond market stress preceded equity stress by 2–3 days.
- **March 2020 (COVID + oil):** Oil crashed 30% on the Saudi-Russia price war. This amplified the COVID selloff through energy sector credit stress. The oil crash was a separate signal visible 24 hours before the equity circuit breaker.
- **September 2022 (UK pension crisis):** The UK gilt market crisis (post-Truss mini-budget) sent shockwaves through global bonds before hitting US equities.

**The implementation path:**
1. **Short-term:** Add daily readings of TLT, HYG, DXY, and VIX futures term structure to the market snapshot
2. **Medium-term:** Compute a cross-asset stress index: if 2+ of (TLT down >1%, HYG down >0.5%, DXY up >0.5%, VIX futures in backwardation) are true, flag as CROSS_ASSET_STRESS
3. **Advanced:** Include JPY and crude oil as early warning signals for specific tail scenarios

**Expected savings at 50% effectiveness:** $30,793

## 17.7 Gap #6: Economic Calendar Awareness ($57K)

**The problem:** 25 losing trades occurred on or immediately before scheduled high-impact economic releases — events whose dates are known weeks in advance.

**What we already have:**
- FRED economic calendar integration via `apis/routes_economic_calendar.py`
- The classifier in `news_bot/classifier.py` identifies macro event types (FOMC, CPI, PPI, GDP, employment)
- The entry engine has a `skip_news_days` flag (but it's coarse — a boolean, not a severity level)

**What's missing:** The economic calendar data exists in the API layer but is not connected to the entry decision. The `skip_news_days` flag in the entry engine is binary and doesn't distinguish between a routine housing starts report and an FOMC rate decision.

**The high-impact events that drive losses:**

| Event Type | Losses Attributed | Avg Loss | Frequency |
|-----------|-------------------|----------|-----------|
| FOMC rate decision | 8 | $2,412 | 8/year |
| CPI release | 7 | $2,534 | 12/year |
| Jobs report (NFP) | 5 | $2,198 | 12/year |
| Trade policy (tariffs) | 3 | $2,342 | Variable |
| Geopolitical shock | 2 | $2,381 | Variable |

**The implementation path:**
1. **Immediate:** Hard-code FOMC dates (published a year in advance) — skip 1DTE entries the day before and day of FOMC
2. **Short-term:** Integrate CPI and NFP dates — skip or reduce size for 1DTE on report days
3. **Medium-term:** Build a complete event calendar with impact scoring. Level 1 (housing starts): no change. Level 2 (PPI): reduce size. Level 3 (CPI/FOMC/NFP): skip 1DTE entirely, reduce 3DTE
4. **Advanced:** Track consensus estimates vs actuals for recurring releases. A CPI beat of +0.3% above consensus is much more dangerous than an in-line print

**Expected savings at 50% effectiveness:** $28,527

## 17.8 The Missing Signal We Haven't Built: Options Flow

One signal appears in nearly every loss narrative but has no existing implementation: **options flow data**. Before most major moves, large institutions reposition through the options market. This is visible as:

- Unusual put volume (put/call ratio spikes)
- Large block trades on far OTM puts
- Skew steepening (OTM puts becoming more expensive relative to ATM)
- Delta hedging flows from market makers

This data is available through:
- CBOE options volume data
- Dark pool activity monitors
- GEX (gamma exposure) — which we already partially implement

The challenge is that options flow data is noisy. Institutional hedging looks the same as directional betting at the flow level. Filtering signal from noise requires either a sophisticated model or significant domain expertise. This is a genuine unsolved problem, not a straightforward implementation.

We tagged every losing trade with "OPTIONS_FLOW would likely have helped" as a hypothesis, but we cannot quantify the savings without backtesting against historical flow data, which we do not currently possess.

## 17.9 Priority Implementation Roadmap

Based on the gap analysis, the implementation priority is clear. Ordered by expected savings per hour of development effort:

**Phase 1 — Immediate (< 1 day each):**
1. VIX floor filter: When VIX < 15, reduce IC3 to 50% size. Cost: 10 lines of code. Expected savings: ~$34K.
2. Hard-code FOMC/CPI/NFP dates into entry skip logic. Cost: a calendar lookup. Expected savings: ~$28K.
3. Friday skip for 1DTE: Don't enter IC3 on Fridays (weekend gap risk). Cost: one conditional. Expected savings: ~$15K.

**Phase 2 — Short-term (1–3 days each):**
4. Wire news bot severity into entry engine: `news_severity >= 4` → skip or half size. Already have the classifier.
5. Add midday position check for open positions. Requires a cron job at 12:00 PM.
6. Add TLT/HYG/DXY daily readings to market snapshot.

**Phase 3 — Medium-term (1–2 weeks each):**
7. Backtest the news filter historically by mapping curated events to dates.
8. Implement cross-asset stress index.
9. Build 0DTE variant of IC3 to eliminate overnight gap risk entirely.

**Phase 4 — Advanced (ongoing research):**
10. Real-time LLM news analysis for elevated-uncertainty detection.
11. Options flow integration (requires data source evaluation).
12. VVIX / complacency detector.

The total expected savings from Phases 1–3, at conservative 50% effectiveness, exceeds $200,000 annually — a 50% improvement over the current $397K annual P&L, achieved not through a new strategy but through *not entering trades we should never have entered*.

## 17.10 The Meta-Lesson

The gap analysis reveals something fundamental about systematic trading: **the edge is not just in the strategy. It's in the information that feeds the strategy.**

Our iron condor specifications (deltas, widths, DTEs, profit targets, stop levels) were optimized over 11 years and thousands of parameter combinations. They are mature. There is very little incremental improvement available from further parameter tuning.

But the *entry filter* — the decision of *whether to trade today* — is operating on incomplete information. It knows about price trends and volatility levels but not about tariff announcements, pandemic outbreaks, or scheduled FOMC meetings. This is like optimizing the recipe while ignoring whether the oven is on fire.

The system we have built — regime engine, quant signals, position sizing — is a good oven. The strategies are good recipes. What we need now is a smoke detector.
