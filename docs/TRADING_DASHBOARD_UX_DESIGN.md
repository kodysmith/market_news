# Trading Dashboard UX Design & Critical Evaluation

This document evaluates the current Flutter app (Decision Cockpit + GEX) and proposes a clean dashboard UX for the requested data blocks, plus a **combined gravity metric** and backtest approach.

---

## 1. Critical Evaluation of Current Flutter App

### What Works Well
- **Cockpit** already shows: regime (positive/negative gamma), transition (near flip), GEX breakdown (C/P), price bar with put wall / flip / spot / call wall, de-pin risk, action bias.
- **Asset selection** is global (AssetSelectionProvider) and drives cockpit state.
- **Events** (cockpit/events) provide badges and expandable list (FOMC, CPI, earnings, OPEX) with date labels.
- **GEX Calculator** screen has full GEX by strike, max pain (separate mode), and summary across tickers.
- **Backend** exposes: `/cockpit/state`, `/cockpit/events`, `/gex/calculate`, `/gex/max-pain`, `/gex/summary`, multi-lens walls (today / tactical / regime).

### Gaps vs Your 5 Requirements

| Requirement | Current State | Gap |
|-------------|---------------|-----|
| **1. Gamma exposure & strategy (SPX, XSP, SPY, NDX)** | Cockpit is single-ticker; GEX summary is multi-ticker but not strategy-focused | No single view: "volatile vs pinning, how much, flip now/soon" for SPX/XSP/SPY/NDX. Need batch summary + explicit wording. |
| **2. Max pain (today/nearest expiry)** | Max pain exists in GEX screen only, separate flow | Not on main dashboard; not "today or nearest" by default. |
| **3. News today/tomorrow + impact on symbol** | Events show all events; impact is generic (high/med/low) | No filter "today & tomorrow AM"; no link "impact on *selected symbol*" (e.g. AAPL earnings → SPY/QQQ). |
| **4. Call/put walls (today + next OPEX)** | Structure has walls_today (0–2 DTE), walls_regime (0–60 DTE), OI walls | Not labeled as "today/nearest" vs "next major OPEX"; OI walls not surfaced as clearly as GEX walls. |
| **5. Price, yesterday close, today open** | Cockpit has `regime.spot` only | No previous close or today open in API or UI. |

---

## 2. Proposed Dashboard UX: One Screen, Six Blocks

**Principle:** One scrollable screen, symbol selector at top. Each block is a clear card. Priority order: Price context → Regime & strategy → **Volatility strategy** → Max pain → Events → Walls.

### Block A: Symbol + Price Strip (top)
- **Selection:** Single prominent symbol selector (SPY, SPX, XSP, NDX, QQQ, etc.).
- **Content:** Current price (large) | Yesterday close | Today open.
- **UX:** Single row: `SPY  612.34  │  Prev 611.20  │  Open 611.80` with small trend (e.g. vs prev in pts and %).
- **Data:** New or extended API: e.g. `/cockpit/quote?ticker=SPY` → `{ current, previous_close, open, ... }`. Yahoo chart `meta` has `regularMarketPrice`, `previousClose`; today open from first bar of session or a quote API.

### Block B: Gamma Exposure & Strategy (SPX, XSP, SPY, NDX)
- **Goal:** Answer: "Volatile or pinning? How much? Flip now or soon?"
- **Content:**
  - **Regime badge:** e.g. POSITIVE GAMMA / NEGATIVE GAMMA / NEAR FLIP (reuse current).
  - **One-line strategy:** e.g. "Sell premium / fade extremes" vs "Follow breaks / don’t fade" (already in cockpit as hero bias).
  - **Volatile vs Pinning:** Derive from regime + de-pin risk: e.g. "Pinning day" when positive gamma + low de-pin risk; "Volatile" when negative gamma or high de-pin risk. Show short label + optional 0–100 "pinning score" or reuse de-pin band.
  - **Flip:** "Flip at 611.2" + "2.1 pts away" or "Within 2 pts → flip soon."
- **Multi-index (SPX, XSP, SPY, NDX):** Either:
  - **Option 1:** One card per index (compact rows: ticker | regime | flip distance | 1-line advice).
  - **Option 2:** Single card "Index GEX" with a small table or chips: SPX / XSP / SPY / NDX each with regime + "Volatile/Pinning" + flip distance.
- **UX:** One card: "Gamma & strategy" with selected symbol emphasized; expandable or second card for "All indices" table.

### Block C: Volatility Strategy (new)
- **Goal:** Answer "What should I do with volatility?" — sell vol, buy vol, or stay neutral, with brief rationale.
- **Content:**
  - **Metrics:** VIX (and change), front IV, term structure (Contango / Flat / Inverted), direction (Rising / Falling / Flat).
  - **One-line strategy:** e.g. "Sell vol / premium" when state is CONTRACTING or COMPRESSED; "Buy vol / hedge" when EXPANDING or INVERTED; "Neutral / defined risk" when NORMAL.
  - **Optional:** 2–3 vol-specific allowed actions (subset of action_filter that are vol-related).
- **UX:** One card: "Volatility strategy" with VIX, term structure, direction, and headline + rationale; optionally vol-focused action chips.
- **Data:** Cockpit state already has `volatility` (front_iv, direction, term_structure, state, vix, vix_change). Add `volatility_strategy: { headline, rationale, vol_allowed?, vol_forbidden? }` to state (computed in decision_cockpit from volatility).

### Block D: Max Pain (Today / Nearest Expiry)
- **Content:** "Max pain: $611" for **nearest expiry** (0 DTE or 1 DTE). Show expiry date: "Exp: 2025-02-05."
- **UX:** Single row or small card: `Max pain 611  (Exp 2/5)  │  Spot 2.3 above`.
- **Data:** Include in cockpit state (e.g. from `/gex/max-pain?ticker=...&dte=0`) so dashboard needs one call.

### Block E: News & Events (Today & Tomorrow AM, Impact on Symbol)
- **Content:** Events with `date` = today or tomorrow and time = morning (e.g. before 12:00) or "BMO".
- **Impact on symbol:** 
  - **Earnings:** If event has `ticker`, show "High impact on [ticker]" when selected symbol is that ticker or a major index (SPY/QQQ) that heavily weights it.
  - **Macro (FOMC, CPI, NFP, etc.):** "High impact on SPY/NDX" when selected symbol is SPY, SPX, QQQ, etc.
  - **OPEX:** "Medium impact (gamma unwind)" for indices.
- **UX:** Card "Today & tomorrow" with list: time | title | impact badge (High/Med/Low) and optional "Relevant to SPY" tag.
- **Data:** Filter existing `/cockpit/events` by date ≤ tomorrow and (optional) time = AM. Backend adds `impact_on_symbol: "high" | "medium" | "low"` per event when `?symbol=SPY` is present.

### Block F: Call & Put Walls (Today/Nearest + Next Major OPEX)
- **Goal:** "What are we gravitating towards?"
- **Content:**
  - **Nearest expiry (0–2 DTE):** Put wall $605 | Call wall $618 (from `walls_today`). Label: "Nearest exp."
  - **Next major OPEX:** Put wall $602 | Call wall $620 (from a regime or dedicated "OPEX expiry" bucket). Label: "OPEX 2/21."
- **OI vs GEX:** Show both: e.g. "GEX walls: P 605 / C 618" and "OI walls: P 604 / C 619" so user sees both gravity types.
- **UX:** One card with two rows:
  - Row 1: **Nearest exp** — Put 605 | Spot 612.3 | Call 618.
  - Row 2: **OPEX [date]** — Put 602 | Call 620.
  - Optional: mini horizontal bar (spot between put and call) for each row.
- **Data:** Cockpit state already has `walls_today` and `walls_regime`. Add explicit "walls for next monthly OPEX expiry only" in backend (filter by `expiry == next_monthly_opex()`) if you want a clean OPEX-only bucket; otherwise use `walls_regime` and label it "Next major OPEX (0–60 DTE)."

### Layout Sketch (mobile-first)
```
[ Symbol: SPY ▼ ]
[ 612.34  |  Prev 611.20  |  Open 611.80 ]

[ Gamma & strategy ]
  POSITIVE GAMMA   Pinning day   Flip 611.2 (1.1 pts away)
  Sell premium • Fade extremes

[ Volatility strategy ]
  VIX 14.2 ↓   Contango   Sell vol / premium
  Rationale: IV contracting; premium selling favored.

[ Max pain ]
  $611  (Exp 2/5)   Spot 1.3 above

[ Today & tomorrow ]
  09:30  CPI         High impact on SPY
  10:00  AAPL Earns   High impact on SPY

[ Walls ]
  Nearest exp   P 605 —— ● 612.3 —— C 618
  OPEX 2/21     P 602 ———————— C 620
```

### Implementation Checklist
- **Doc:** Block C (Volatility Strategy) and block renumbering (D–F) as above.
- **Backend:** `/cockpit/quote` or `quote` in state; `volatility_strategy` in state; `max_pain` in state; `impact_on_symbol` in events when `?symbol=`; optional `walls_opex` in structure.
- **Flutter:** CockpitState models for quote, volatilityStrategy, maxPain, impact_on_symbol; six-block layout in decision_cockpit_screen (A: Price strip, B: Gamma, C: Volatility strategy card, D: Max pain, E: Events, F: Walls).

---

## 3. Combined Gravity Metric: Single Score from Call/Put Walls

**Idea:** Combine put wall, call wall, max pain, and spot into one "gravity center" or "pin score" that indicates where price is being pulled and how strongly.

### Possible Formulations

1. **Distance‑weighted center**
   - `gravity_center = (put_wall + call_wall) / 2` (or OI‑weighted).
   - `score = 100 - |spot - gravity_center| / wall_range * 100` so that score is high when spot is near the midpoint.

2. **Max‑pain‑centric**
   - `score = 100 - |spot - max_pain| / wall_range * 100` (how close spot is to max pain, normalized by put–call range).

3. **Multi‑anchor weighted average**
   - Anchors: put wall, call wall, max pain. Weight by OI or GEX strength.
   - `gravity_strike = (w1 * put + w2 * call + w3 * max_pain) / (w1 + w2 + w3)`.
   - Score = inverse distance of spot to `gravity_strike` (and optionally factor in "tightness" of walls).

4. **Pin probability (binary or 0–100)**
   - Heuristic: high when (a) spot near max pain, (b) spot inside put–call range, (c) positive gamma. Combine into a 0–100 "pin score" and backtest against "did price stay within X% of open at close?"

### Weighting and Accuracy
- **Weighting:** Start with equal weights for put wall, call wall, max pain; or weight by total OI at each level. GEX walls are more "behavioral" (dealer hedging), OI walls more "positioning." A hybrid could be: `gravity = 0.5 * (gex_center) + 0.5 * (oi_center)` and compare spot to that.
- **Accuracy:** Only backtesting will tell. Suggested target: "Does a high gravity score predict lower realized range (pinning) and does a low score predict larger range (volatile)?"

### Backtest Sketch

**Objective:** Test whether a single gravity/pin score (built from walls + max pain + spot) predicts end‑of‑day behavior (pinning vs volatile).

**Metrics to compute daily (per symbol):**
- Put wall, call wall (GEX and/or OI), max pain (nearest expiry), spot at open.
- **Gravity score** (e.g. 0–100): e.g. `score = 100 - min(100, |spot - gravity_strike| / range * 100)` where `gravity_strike` is midpoint of walls or a weighted combo with max pain.
- **Outcome:** Realized range = `high - low` for the day (or `|close - open|`).

**Backtest design:**
1. **Data:** Historical options snapshots (or stored GEX/walls/max pain) + daily OHLC. If full history is not available, start with a forward-looking log: each day store walls, max pain, spot open, then next day record range and close.
2. **Stratification:** Bucket days by gravity score (e.g. 0–25, 25–50, 50–75, 75–100).
3. **Hypothesis:** Higher gravity score → smaller realized range (pinning); lower score → larger range (volatile).
4. **Metrics:** Mean and median daily range by score bucket; correlation between score and range; optional regression of range on score (and controls: VIX, DTE to OPEX).
5. **Sensitivity:** Try different definitions of gravity_strike (midpoint vs max‑pain‑centric vs OI‑weighted) and different score formulas; compare which has the strongest predictive relationship.

**Implementation notes:**
- Backtest can live in `backtesting/` or `QuantEngine/` (e.g. `gravity_score_backtest.py`).
- Inputs: daily CSV or DB with columns `date, symbol, spot_open, put_wall_gex, call_wall_gex, put_wall_oi, call_wall_oi, max_pain, high, low, close`.
- If historical options data is limited, run a "forward log" for 2–4 weeks: each morning compute walls + max pain + score, then next day add that day’s OHLC and run the analysis.

---

## 4. API Contract Additions / Changes

| Endpoint | Change |
|----------|--------|
| **Cockpit state** | Add `quote: { current, previous_close, open }` (or new `/cockpit/quote`) so Block A and B have yesterday close and today open. |
| **Cockpit state** | Add `max_pain: { strike, expiration }` for nearest expiry (dte=0) so Block C is one call. |
| **Cockpit state** | Ensure `structure` exposes OI walls clearly: e.g. `oi_primary_walls` and `walls_today` / `walls_opex` (new) with both GEX and OI in one place. |
| **Cockpit events** | Add query `?days=2` and optional `?symbol=SPY`; response can include `impact_on_symbol: "high" | "medium" | "low"` for each event. |
| **GEX summary** | Already supports `?tickers=SPX,XSP,SPY,NDX`. Add optional field `strategy_advice` or `pinning_score` per ticker for Block B multi-index. |

---

## 5. Summary

- **Dashboard:** One screen, symbol at top, then: Price strip → Gamma & strategy (with optional SPX/XSP/SPY/NDX table) → Max pain → Today/tomorrow events (impact on symbol) → Call/put walls (nearest + OPEX).
- **Gravity metric:** Combine put wall, call wall, max pain, and spot into a single 0–100 score; backtest whether high score predicts pinning (small range) and low score predicts volatility (large range). Start with equal or OI-weighted center; refine with backtest results.
- **Backtest:** Stratify days by gravity score, compare realized range (high−low or |close−open|) across buckets; optionally regress range on score + VIX + DTE to OPEX.

This keeps the existing Flutter cockpit and GEX APIs as the backbone and adds a clear information architecture and a testable combined gravity metric.
