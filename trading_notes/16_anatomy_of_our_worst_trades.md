# Chapter 16: Anatomy of Our Worst Trades

## 16.1 Why Study Losses Instead of Wins

Most trading books celebrate winners. They reverse-engineer the setup that produced the biggest gain, find a narrative that explains why it worked, and generalize from a sample of one. This is survivorship bias applied to education.

We took the opposite approach. We ran the complete regime-aware backtest — IC3 (1DTE, 10-delta, $30 wide), IC6 (3DTE, 12-delta, $30 wide), and IC9 (7DTE, 12-delta, $50 wide) — across 11 years of SPX data (2015–2025). That produced 5,949 trades: 5,278 winners and 671 losers. Then we selected the 150 biggest winners and 150 biggest losers, matched by time period (84% monthly overlap), so we could compare trades that succeeded with trades that failed in the same market environments.

The winners are boring. They look the same: theta decays, SPX stays in range, position hits profit target or expires worthless. The average winner made $262. The average loser cost $1,047 — a 4:1 asymmetry that defines short premium income trading. Understanding what drives that $1,047 average is where the edge lives.

## 16.2 The Five Loss Patterns

Every losing trade in the dataset falls into one of five distinct patterns. Classifying them matters because each pattern demands a different defense.

### SHOCK_EVENT (56 trades, $115,238 total losses)

The largest category. A shock event is a sudden market crash accompanied by a VIX explosion — SPX moves more than 3% and VIX spikes more than 5 points in the trade's lifetime.

Representative trades:
- **2018-10-17:** IC9, entered with VIX at 17.4 in CAUTION regime. SPX dropped 5.45% over 7 days. VIX spiked from 17.4 to 25.2. Loss: $3,012 (the worst single trade). Catalyst: US-China trade war escalation, hawkish Fed, treasury yields spiking.
- **2018-02-02:** IC3, entered with VIX at 17.3 in NEUTRAL regime. SPX dropped 4.1% overnight. VIX exploded from 17.3 to 37.3 (Volmageddon). Loss: $2,761.
- **2020-02-19:** IC9, entered with VIX at 14.4 in NEUTRAL regime. SPX dropped 4.73% as COVID fears emerged. VIX went from 14.4 to 25.0. Loss: $2,763.

The common thread: VIX was moderate (14–18) at entry, the regime engine rated conditions as NEUTRAL or CAUTION, and then a macro catalyst hit. The regime engine was correct based on available data — it just didn't know about the catalyst.

### OVERNIGHT_GAP (49 trades, $112,302 total losses)

The second-largest pattern and in some ways the most preventable. An overnight gap is a trade (usually IC3, 1DTE) where SPX gaps through the short strikes between market close and the next open. The stop-loss never triggers because there's no continuous market.

Representative trades:
- **2019-05-10:** IC3, SPX dropped 2.41% overnight after trade talks collapsed. Put cushion was only 1.26% — the gap blew right through. Loss: $2,791.
- **2019-08-02:** IC3, SPX dropped 2.98% after Trump announced surprise tariffs on $300B of Chinese goods. Put cushion was 1.43%. Loss: $2,780.
- **2018-12-03:** IC3, SPX dropped 3.24% as the "trade truce" unraveled and yield curve inverted. Loss: $2,796.

These trades share three properties: (1) 1DTE holding period that crosses an overnight boundary, (2) a macro news event that hits after hours or pre-market, and (3) a put/call cushion of 1.0–1.8% that was adequate for normal overnight moves but insufficient for event-driven gaps.

### LOW_VOL_SURPRISE (26 trades, $55,136 total losses)

This is the subtlest and most dangerous pattern. VIX is below 18 at entry — suggesting calm markets — but the market moves more than 2.5%. Low VIX means two things working against us: (1) the premium collected is small (low VIX = cheap options), and (2) the strike cushion is thin (strikes are closer to spot when implied vol is low).

Representative trades:
- **2015-08-17:** IC6, entered with VIX at 13.0. SPX dropped 3.17% in 3 days. The premium collected was small because VIX was low, but the loss was full-size. Loss: $2,738.
- **2020-02-21:** IC3, entered with VIX at 17.1. SPX dropped 3.35% (COVID Friday). Loss: $2,728.

The lesson is counterintuitive: low VIX is not a green light. It's a warning. When VIX is 13–16, the market is priced for perfection. Any surprise produces a disproportionate move because hedging is cheap and nobody owns protection. The risk/reward of selling premium in a sub-16 VIX environment is often negative — you collect less credit but face the same tail risk.

### TREND_MOVE (16 trades, $26,500 total losses)

A trend move is a sustained directional move that triggers the stop-loss on a longer-DTE position (typically IC9, 7DTE). Unlike shock events, these aren't single-day crashes — they're multi-day slides where each day chips away at the position.

Representative trades:
- **2018-10-22:** IC3, SPX in CAUTION regime (below both SMAs), dropped 3.62% in 2 days. Loss: $2,701.
- **2022-06-10:** IC3, CPI printed 8.6% (higher than expected), market dropped 3.88% over the weekend. Loss: $2,635.

Trend moves are less catastrophic per trade (average loss $1,656 vs $2,058 for shock events) because the stop-loss does engage. The stop doesn't prevent the loss, but it limits it. The damage comes from frequency — trend moves happen more often during extended bear markets (2022 had multiple).

### MODERATE_MOVE (3 trades, $6,003 total losses)

A small category: SPX moves 1.5–2.5% against the position but the strikes are too tight to survive even this moderate move. These are almost exclusively a delta problem — 12-delta strikes in a 14-VIX environment provide only 1.2% cushion, and a 2% move breaches them.

These trades are the easiest to prevent: wider deltas or a VIX floor filter would eliminate nearly all of them.

## 16.3 When Losses Cluster

Losses are not uniformly distributed across time. They cluster around macro events:

| Period | Event | Trades Lost | Total Loss |
|--------|-------|-------------|------------|
| Aug 2015 | China devaluation / Black Monday | 4 | $10,974 |
| Feb 2018 | Volmageddon | 3 | $7,830 |
| Oct–Dec 2018 | Trade war + Fed + yield curve | 5 | $14,053 |
| May & Aug 2019 | Trade war tariff escalation | 4 | $11,187 |
| Feb–Mar 2020 | COVID-19 crash | 12 | $31,594 |
| Jun 2022 | CPI shock / bear market | 4 | $10,670 |
| Sep 2022 | CPI surprise / Fed hawkish | 3 | $7,837 |
| Aug 2024 | Japan carry trade unwind | 3 | $7,616 |
| Dec 2024 | Hawkish Fed dot plot | 2 | $5,223 |

The concentration is striking. Ten discrete event clusters account for roughly $107,000 — nearly a third of all losses in the entire dataset. These are not random: they are all macro events with significant news signals that were publicly available before or during the trade.

## 16.4 The Regime Engine's Blind Spots

The regime engine correctly identified BEAR conditions during the worst crash periods (March 2020, mid-2022). It kept us out of positions during the deepest drawdowns. That's its job, and it does it well.

But the engine has systematic blind spots:

**Blind spot 1: The transition gap.** The regime moves DOWN immediately on a bad score but requires 3–5 consecutive days to move UP. This asymmetry is by design (fast down, slow up). The problem is that many of our worst losses occur in the 1–3 days *before* the regime downgrades. On 2020-02-19, the regime was NEUTRAL (score 4) — all signals positive. Two days later, the COVID crash had started. The regime dropped to CAUTION on 2020-02-24 and BEAR by 2020-02-28. But the three trades entered on 2/19, 2/21, and 2/24 in NEUTRAL collectively lost $8,204.

**Blind spot 2: Low VIX complacency.** When VIX is 14 and SPX is above both SMAs with a rising SMA200, the regime score is typically +4 or +5 — NEUTRAL or BULL. Every signal says "safe." But a VIX of 14 means options are cheap, strikes are close to spot, and any surprise will be amplified. The regime engine treats VIX < 22 as bullish (+1) rather than recognizing the hidden risk in extremely low volatility.

**Blind spot 3: No news awareness.** The regime engine uses price-derived signals exclusively — SMAs, VIX level, VIX term structure, VIX trend. It has no input for "the President just announced new tariffs on Twitter" or "Italy reported 650 COVID cases overnight." These catalysts move markets 3–5% in a day, but the regime engine only knows about them after the move has already happened.

## 16.5 What the Winners Tell Us

The 150 biggest winners are instructive by contrast. They cluster into three patterns:

### HIGH_VOL_PREMIUM (entered during VIX > 35)

The biggest winners — trades making $560–$596 — almost all entered during crisis periods: March 2020 (VIX 57–75), June 2022 (VIX 29), August 2024 (VIX 27). High VIX provides:
- Fat premium (more credit collected)
- Wide strikes (10-delta is further OTM when IV is high)
- Mean reversion tailwind (VIX tends to decline from elevated levels)

The IC6 trade entered on 2020-03-19 with VIX at 72 collected enough premium that put strikes were 8.5% OTM and call strikes were 10% OTM. Even in the chaos of COVID week, SPX stayed within that range and the position made $596 — maximum profit.

### CLEAN_THETA_DECAY (profit target hit early)

The IC9 (7DTE) winners that hit the 30% profit target within 2–3 days. These trades entered in NEUTRAL regime with VIX 18–21 (the sweet spot: enough premium to be worth the risk, but not so volatile that the position is in danger). SPX moved less than 1% during the hold, and time decay did the work.

### RANGE_BOUND (SPX barely moved)

Trades where SPX moved less than 0.5% during the hold period. Perfect conditions for iron condors — both sides decay together, and the position reaches max profit. These tend to occur in mid-week entries during low-catalyst periods.

## 16.6 The Asymmetry Problem, Quantified

Across all 5,949 trades:
- Average winner: $262
- Average loser: -$1,047
- Ratio: 4:1 against

Among the 300 balanced samples:
- Average top winner: $485
- Average top loser: -$2,101
- Ratio: 4.3:1 against

This is the fundamental challenge of short premium income trading. The strategy wins 89% of the time, but when it loses, each loss wipes out 4 winners. One bad week can erase a month of gains.

The regime engine helps by preventing trades during the worst environments (reducing the -$78K worst-day loss to -$87K max drawdown over a year, not a day). But the data shows there's meaningful additional alpha available by:
1. Identifying which *specific* trades to skip based on news and event awareness
2. Adjusting position size based on overnight gap risk
3. Treating low VIX as a risk factor rather than a safety signal

The next chapter quantifies exactly how much each of these missing signals would save.

## 16.7 The Training Dataset

The complete analysis produced three files for AI model training:

| File | Records | Description |
|------|---------|-------------|
| `data/training/all_trades_with_context.csv` | 5,949 | Every trade with 35+ market context features |
| `data/training/balanced_winners_losers.csv` | 300 | 150 winners + 150 losers, time-matched (84% monthly overlap) |
| `data/training/trade_narratives.jsonl` | 300 | Structured narratives with known signals, events, and lessons |

Each trade record includes: entry/exit dates, strategy config, strikes, credit, P&L, VIX at entry/exit, SMA50/200 positions, regime score/tier, SPX move, put/call distance, max adverse excursion, exit reason, and calendar features.

The balanced dataset ensures equal representation of wins and losses for training classifiers. The time-matching ensures the model learns from conditions where both outcomes were possible — not just "bear markets produce losses" but "on this specific day, one configuration won while another lost, and here's why."
