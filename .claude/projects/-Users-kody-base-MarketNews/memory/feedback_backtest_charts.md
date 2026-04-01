---
name: backtest-chart-format
description: User wants all backtest results rendered as interactive HTML using Lightweight Charts (dark theme, scrollable, with trade markers)
type: feedback
---

Use the Lightweight Charts template from `data/daily_cashflow_backtest.html` as the standard format for all backtest result visualizations. Key requirements:
- Inline data directly into HTML (no external JSON fetch — local files block CORS)
- Dark theme (#0D1117 background) matching the app design
- Scrollable/zoomable equity curve with hover tooltips
- Trade markers (profit target / stop loss arrows)
- SPX overlay toggle
- Drawdown, daily P&L histogram, monthly P&L bars
- Time range buttons (All/5Y/3Y/1Y/YTD)
- Stats bar at top with key metrics

**Why:** User said "it is beautiful... we should use that view for all of our back test results"
**How to apply:** When generating any backtest visualization, produce a self-contained HTML file using this template pattern rather than static PNG charts.
