# Market News App — Product & UX Design

## Purpose
A cross-platform app for active traders and investors to:
- Instantly understand current market conditions
- See the top trading strategies for the day (ranked by probability of profit)
- Drill down into actionable trade ideas for each strategy
- Learn why each strategy is favored, with educational context

## Core User Experience
- **Fast, at-a-glance dashboard**: Users see the market's mood and the best ways to trade it today.
- **Actionable, not overwhelming**: Only the top 3 strategies are highlighted, with clear rationale and top tickers.
- **Educational**: Every card and detail screen explains the 'why', not just the 'what'.

## Main Screens & Information Architecture

### 1. **Home Dashboard**
- **Market Sentiment Card**
  - Shows overall market mood (Bullish/Bearish/Neutral)
  - Tap for a detailed breakdown of indicator contributions and explanations
- **Top 3 Strategy Cards**
  - Each card shows:
    - Strategy name (e.g., Gamma Scalping, Covered Calls, Bull Put Spread)
    - Short description
    - Probability of profit or strategy score
    - Icon and color reflecting risk/volatility
    - Tap to view details
- **Quick Navigation**
  - Easy access to market insights, economic calendar, and settings

### 2. **Market Sentiment Detail**
- Full explanation of why the market is leaning bullish/bearish/neutral
- Breakdown of each indicator (S&P 500, Nasdaq, VIX, Treasuries, USD)
- Educational context for each indicator's impact
- Trading implications and risk factors

### 3. **Strategy Detail Screen** (for each top strategy)
- List of top tickers for the strategy today (ranked)
- For each ticker:
  - Why it's a good candidate (metrics, e.g., IV/RV, trend, etc.)
  - Example trade setup (strikes, expiry, expected credit/debit)
  - Risk/reward summary
- Educational section: When/why to use this strategy

### 4. **Market Insights & Tools**
- Economic calendar
- Volatility dashboard
- Custom watchlists (future)

## User Flow Example
1. User opens app → sees dashboard with sentiment and top 3 strategies
2. Taps a strategy card → sees best tickers and trade setups for that strategy
3. Taps a ticker → sees trade details and educational context
4. Optionally, taps sentiment card for a deep dive into market mood

## Design Principles
- **Clarity**: No jargon without explanation
- **Actionability**: Every screen answers "What should I do today?"
- **Education**: Every recommendation is justified and explained
- **Responsiveness**: Works on mobile, tablet, and web 