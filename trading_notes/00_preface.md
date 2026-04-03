# Preface

I started where most options traders start: watching a YouTube video about selling iron condors on SPX. The pitch was seductive — collect premium, let time decay work for you, win 90% of the time. What could go wrong?

Quite a lot, as it turns out.

The first few months were intoxicating. Wins stacked up. The account grew. I started sizing up. Then came a day where SPX gapped down 3% at the open. One trade wiped out two months of carefully collected premium. My 90% win rate was mathematically irrelevant because the 10% losses were three times larger than the wins. I was playing a game I didn't understand.

That day marked the beginning of a different kind of journey — one that would consume the better part of two years and produce the system documented in this book. Instead of trusting my gut about "where the market was heading," I started asking a different question: *Can I build a system that makes this decision for me, and prove that it works across eleven years of market history?*

The answer, eventually, was yes. But the path was anything but straight.

Along the way, I built a backtesting engine that simulates SPX options trades with realistic slippage and commissions across 2015–2025. I tested dozens of strategy variations — different deltas, different widths, different expiration dates, different profit targets. Some worked brilliantly. Many didn't. I learned that box spread recycling sounds elegant in theory but loses money to slippage in practice. I discovered that bear call spreads in neutral markets have win rates that look fine on paper but underperform iron condors on every risk-adjusted metric. I watched a parameter sweep produce "perfect" settings that collapsed the moment I tested them out of sample.

But I also found things that worked. A 1-day-to-expiration iron condor at 10-delta with $30 wings and a 50% profit target that wins 90% of the time with a 2.54 Sharpe ratio — and has been profitable every single year for eleven years. A funded butterfly structure with a 99% win rate that literally pays you a credit to enter. A regime detection engine that watches six market signals and tells you when to trade full size, when to reduce, and when to park everything in Treasury bills. That engine reduces maximum drawdown by 68% while only giving up 15% of returns. The math on that trade-off is overwhelming.

The system I built isn't theoretical. It runs live against Interactive Brokers, pricing options with real bid-ask spreads, managing positions through a cloud-deployed bot that executes at 9:35 AM every trading day. It tracks its own P&L, monitors regime transitions, and parks capital in SGOV when conditions turn dangerous.

Along the way, I also built something I didn't expect to build: a knowledge engine that reads books. Not trading books — or at least, not only trading books. I fed it over fifty works spanning classical economics, crowd psychology, monetary history, and speculative finance. Ricardo on value theory. Bastiat on market distortion. Le Bon on crowd behavior. The anonymous author of "Fifty Years in Wall Street" on the timeless patterns of speculation. What I found was that the insights from 1900 and the insights from 2025 converge on the same truths: markets are driven by human psychology, risk is non-linear, and the crowd is almost always wrong at extremes.

This book is the record of everything I learned. It is organized as a journey — from the naivety of my first trades through the systematic development of strategies, the construction of a regime engine, the integration of quantitative risk signals, and finally the technology that makes it all run autonomously. Every chapter includes the actual numbers: win rates, Sharpe ratios, annual P&L, maximum drawdowns. Nothing is rounded up or cherry-picked.

I've also included the failures. Chapter 9 is entirely devoted to strategies that didn't work, because I believe the dead ends taught me as much as the successes. If you can understand *why* box spread recycling loses money, you understand something important about market microstructure. If you can understand *why* parameter sweeps overfit, you'll avoid the most common mistake in quantitative trading.

**Who this book is for:** You should already understand what an option is, what a spread is, and roughly how the Greeks work. You don't need to be a quant — I'm not one by training. But you should be tired of discretionary trading and curious about whether there's a more systematic way. If you've ever thought "there must be a better approach than staring at charts and guessing," this book is for you.

**Who this book is not for:** If you're looking for a get-rich-quick system or a set of rules you can follow without understanding, you'll be disappointed. This book is about building understanding. The system works because of the reasoning behind it, not because of any single parameter setting.

Let's begin.
