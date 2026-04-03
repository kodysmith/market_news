# Chapter 19: Tax Strategy for Options Income

## 19.1 The Tax Bill Nobody Warns You About

The strategy generates $424,016 per year. At a 36.5% combined tax rate (federal 1256 blended + California 13.3%), the tax bill is $154,917. That's more than most Americans earn in a year — gone to taxes.

Most options trading books stop at backtesting and risk management. They never mention that a 36.5% effective tax rate turns a $424K strategy into a $269K strategy. The difference between keeping $269K and keeping $356K (with optimized tax strategy) compounds dramatically over a career.

Tax strategy is not about schemes or loopholes. It's about using the structures Congress explicitly created for people in this exact situation. The savings are large, legal, and well-established.

## 19.2 Section 1256: The Advantage You Already Have

SPX options receive Section 1256 treatment automatically. This is the single most valuable tax advantage in the entire system, and it's the reason we trade SPX instead of SPY.

**How it works:**
- 60% of gains taxed as long-term capital gains (20% federal rate)
- 40% of gains taxed as short-term / ordinary income (37% federal rate)
- Blended federal rate: 26.8%

**What this saves:**

| Treatment | Federal Rate | Tax on $424K | Savings |
|-----------|-------------|-------------|---------|
| Ordinary income (37%) | 37.0% | $156,886 | — |
| Section 1256 (60/40) | 26.8% | $113,636 | **$43,250/yr** |

If we traded SPY options instead of SPX, we would pay $43,250 more per year in federal taxes — for the exact same strategy. Over 11 years, that's $475,750 in additional taxes. Over 20 years, that difference compounds to well over a million dollars.

SPX has additional structural advantages beyond 1256 treatment: cash-settled (no assignment risk), European-style (no early exercise), and better liquidity at the strikes we trade. But the tax treatment alone justifies the choice.

**Section 1256 also provides a unique loss feature:** 3-year loss carryBACK. If you have a losing year, you can carry the 1256 losses back three years to offset prior gains and receive a tax refund. This is not available for ordinary capital losses, which can only carry forward.

## 19.3 The California Problem

California taxes all income — including 1256 gains — as ordinary income at the state level. California does not honor the federal 60/40 split. At the top bracket, the California rate is 13.3%.

The impact on our $424K strategy:

| Component | Rate | Tax |
|-----------|------|-----|
| Federal (1256 blended) | 26.8% | $113,636 |
| California | 13.3% | $56,394 |
| **Total** | **36.5%** | **$154,917** |

California alone costs $56,394 per year. That's $41,281 more than zero-state-tax alternatives (the remaining $15,113 is due to the interaction between federal and state deductions).

**States with no income tax:** Texas, Florida, Nevada, Washington, Tennessee, Wyoming, New Hampshire (no earned income tax), South Dakota, Alaska.

The math is unambiguous:

| Scenario | Annual Tax | After-Tax Income | 10-Year Savings |
|----------|-----------|-----------------|-----------------|
| California resident | $154,917 | $269,099 | — |
| No-state-tax state | $113,636 | $310,380 | **$412,805** |

Moving out of California saves $41,281 per year. Over 10 years, that's $412,805. Over 20 years, $825,610. This single decision saves more than any investment scheme, tax shelter, or capital structure optimization we evaluated.

The savings also compound: $41K/year reinvested at 32.5% CAGR (our strategy's rate) grows to $1.2M in 10 years and $12M in 20 years.

## 19.4 Entity Structure and Solo 401(k)

Trading through an S-Corp or single-member LLC (taxed as S-Corp) does not change the tax treatment of 1256 gains — they pass through to the personal return regardless. But the entity enables several deductions that are otherwise unavailable.

**Solo 401(k):**
- Employee contribution: $23,500/year (2025 limit, $31,000 if over 50)
- Employer contribution: up to 25% of net self-employment income
- Total possible: $69,000/year (under 50) or $76,500 (50+)
- Tax deduction at blended rate: $18,492–$20,502/year

The catch: 1256 gains from a personal brokerage account cannot be contributed to a 401(k). You would need to either trade inside the retirement account (complex but possible with some custodians) or have other earned income flowing through the entity to justify the contribution.

**Business expense deductions through the entity:**

| Deduction | Estimated Annual |
|-----------|-----------------|
| Solo 401(k) contribution | $69,000 |
| Health insurance (family) | $25,000 |
| Data feeds, APIs, software | $8,000 |
| Computing hardware | $3,000 |
| Home office | $5,000 |
| Professional development | $2,000 |
| Tax preparation / CPA | $2,000 |
| **Total deductions** | **$114,000** |
| **Tax saved (~30%)** | **$34,200/yr** |

The entity setup costs approximately $2,000 (formation, EIN, registered agent) plus $1,000–2,000/year in ongoing admin (annual filing, bookkeeping, payroll for S-Corp). The $34,200 annual savings pays for itself immediately.

**Important:** The Section 199A Qualified Business Income deduction does NOT apply to capital gains or 1256 income. This deduction only helps businesses with ordinary income.

## 19.5 Loss Harvesting Without Wash Sales

Section 1256 contracts are explicitly exempt from the wash sale rule. This is a massive advantage that most traders don't fully exploit.

**What this means in practice:**
- On December 31, close any underwater 1256 positions
- Book the capital loss (offsets 1256 gains dollar-for-dollar)
- On January 2, reopen the identical position
- You keep the same market exposure AND get the tax deduction

With stocks or ETFs, the wash sale rule prohibits repurchasing substantially identical securities within 30 days before or after a sale at a loss. The deduction is disallowed. Section 1256 contracts have no such restriction.

**Annual value estimate:** Varies by year. In flat years (like 2022), there are more underwater positions to harvest. In strong years, fewer. Conservative estimate: $5,000–15,000 per year in tax savings, at zero cost to the strategy's performance.

**The 3-year carryback amplifies this:** If you harvest enough 1256 losses to create a net loss for the year, you can carry that loss back to offset gains from up to 3 prior years and claim a refund. This is particularly powerful after a bad year like 2022.

## 19.6 Retirement Account Trading

The ultimate tax shelter for options income is trading inside a Roth IRA. All gains — forever — are tax-free. No 1256 needed because there's no tax at all.

**The constraints:**
- Roth IRA contribution limit: $7,000/year (2025), or $8,000 if over 50
- Direct Roth contributions phase out above $161K MAGI (single) / $240K (married)
- At $424K income, you must use the **backdoor Roth** (contribute to traditional IRA, convert to Roth)
- Backdoor Roth is currently legal but Congress periodically threatens to close it

**Options trading in a Roth IRA:**
- Some brokers allow defined-risk options strategies (spreads, iron condors) in IRAs
- No margin: all positions must be fully collateralized
- IC3 at $30 width = $3,000 max loss per contract, fully funded from IRA cash
- With a $50K Roth balance, you could run 10 IC3 contracts ($30K collateral)
- At $28,245/contract/year, that's $282K/year in tax-free gains growing inside the Roth

**The long game:** Even at $7,000/year contributions, the Roth grows quickly if you're generating 200%+ returns inside it. $7K → $21K after one year of trading → $63K after two years → the account compounds aggressively because the strategy's returns are so high relative to the required collateral.

**Self-directed IRA with SPX options:**
- Custodians like Schwab, Interactive Brokers, and Tastytrade allow options in IRAs
- IBKR specifically supports spread orders in IRA accounts
- The key limitation: no portfolio margin in IRA, so capital efficiency is lower than a regular account

## 19.7 Charitable Giving and Donor-Advised Funds

For traders who are already charitably inclined, a Donor-Advised Fund (DAF) optimizes the tax timing of giving.

**How a DAF works:**
1. Contribute cash or appreciated assets to the DAF
2. Receive an immediate tax deduction in the contribution year
3. Invest the funds inside the DAF (grow tax-free)
4. Distribute to charities over time, at your discretion

**Tax deduction limits:**
- Cash contributions: up to 60% of AGI
- Appreciated assets: up to 30% of AGI
- Excess carries forward 5 years

**Bunching strategy:** Instead of giving $20,000/year to charity, contribute $60,000 every three years. In the contribution year, itemize deductions (including the $60K charitable deduction). In the off years, take the standard deduction. This captures additional tax benefit that annual giving misses.

**At $50,000/year contribution:** Tax savings of approximately $18,268/year at the 36.5% combined rate.

This is only valuable if you would be giving the money anyway. Charitable giving to save on taxes is spending a dollar to save 37 cents — it reduces your tax bill but also reduces your wealth. The DAF structure optimizes the timing, not the fundamental economics.

## 19.8 What We Evaluated and Rejected

### Box Spread → Muni Bond Arbitrage

**The idea:** Borrow money via SPX box spread (creates a 1256 loss that offsets trading gains), invest in tax-free municipal bonds. The box loss reduces your tax bill, and the muni income is tax-free.

**Why it doesn't work:** The box spread borrow rate (~5.2%) exceeds muni bond yields (~3.0–3.5%) on a pre-tax basis. After the 1256 deduction, the effective borrow cost is 3.30% (in CA) or 3.81% (no state tax). In both cases, this exceeds the after-tax muni yield:

| Location | Effective Borrow | Muni Net Yield | Spread |
|----------|-----------------|----------------|--------|
| California | 3.30% | 3.03% | **-0.27%** |
| No state tax | 3.81% | 3.50% | **-0.31%** |

The arbitrage is negative in every scenario. You lose money on every dollar borrowed, regardless of state.

Even if the math were marginally positive, IRS Section 265(a)(2) specifically disallows deductions on debt incurred to purchase tax-exempt securities. The economic substance doctrine provides a second enforcement mechanism: if a transaction only makes money because of the tax benefit and has no pre-tax economic profit, the IRS can disallow the deduction and impose a 20% accuracy penalty.

**Verdict:** Don't do this. The pre-tax economics are negative, the legal risk is real, and the potential gain (even if it worked) is trivially small compared to simply moving out of California.

### Leveraging Contract Count

**The idea:** Use idle capital to fund more IC3/IC6 contracts, doubling the strategy's output.

**Why it doesn't help on tax:** Leverage multiplies both gains and losses proportionally. The tax rate stays the same. You make more pre-tax and pay more tax — the percentage doesn't change.

More importantly, leverage increases tail risk. A -$75K worst month becomes -$151K — 76% of a $200K account. The portfolio risk controls ($150K SPX exposure cap, <6 concurrent spreads) exist specifically to prevent this.

### BOXX ETF via Box Spread

**The idea:** Borrow via box spread, invest in BOXX (which also uses box spreads internally and gets 1256 treatment).

**Why it doesn't work:** You're borrowing at 5.2% to earn 4.4%. The 0.8% negative carry isn't overcome by the tax deduction. Both sides are 1256, so they partially cancel on the tax return, but the pre-tax loss dominates.

## 19.9 Qualified Opportunity Zone Funds

QOZ funds are a tax incentive Congress created in 2017 to encourage investment in economically distressed communities. They offer two benefits: deferral of capital gains tax on invested amounts, and exclusion of tax on appreciation if held 10+ years.

**Why we ranked it low:**

1. **Liquidity mismatch.** You'd be converting liquid, high-return trading capital into illiquid real estate or small business investments in distressed areas. The strategy returns 829% on margin deployed — no QOZ deal comes close.

2. **Deferral has mostly expired.** The original step-up basis incentives (10% reduction after 5 years, 15% after 7 years) ended in 2021. The remaining benefit is deferral until 2026 and the 10-year appreciation exclusion.

3. **Deal quality risk.** QOZ funds are only as good as the underlying investments. Many QOZ deals are marginal real estate projects that were designated as opportunity zones precisely because private capital wouldn't invest there without the tax incentive. The 8–12% projected returns often don't materialize.

4. **You still owe the deferred tax.** Deferral is not forgiveness. The original capital gains tax comes due in 2026 (or when you sell the QOZ investment, whichever is earlier). You're just delaying the bill.

5. **Management fees.** QOZ fund managers typically charge 1–2% annual management fees plus carried interest on profits. These eat into returns that are already lower than what the trading strategy produces.

The only scenario where a QOZ makes sense: you have a single large capital gain (selling a business, a property, a concentrated stock position) and you want to defer the tax for a decade while getting exposure to a specific real estate market you believe in. It's a tool for one-time events, not recurring trading income.

## 19.10 The Idle Capital Question

The strategy requires only $51,150 in margin for 10 IC3 + 5 IC6 contracts. On a $200K account, that leaves $148,850 (74%) idle. What to do with it?

**Best option by after-tax yield:**

In California:
- **National muni bond fund:** 3.5% gross, fed exempt, CA taxes at 13.3% → 3.03% net → $4,517/yr
- **CA muni fund (CMF):** 3.0% gross, double exempt → 3.00% net → $4,466/yr
- **SGOV (T-Bills):** 4.3% gross, fed taxed 37%, state exempt → 2.71% net → $4,032/yr
- **BOXX:** 4.4% gross, 1256 fed + CA taxed → 2.64% net → $3,923/yr

Outside California:
- **National muni bond fund:** 3.5% gross, fully exempt → 3.50% net → $5,210/yr
- **BOXX:** 4.4% gross, 1256 taxed 26.8% → 3.22% net → $4,794/yr
- **SGOV:** 4.3% gross, fed taxed 37% → 2.71% net → $4,032/yr

The idle capital yield is $4–5K/year. It's free money and should be captured, but it's a rounding error on a $424K strategy. Don't over-optimize here — pick one, park it, and focus energy on things that actually move the needle.

## 19.11 The Priority Ranking

Ordered by annual savings:

| Rank | Strategy | Annual Savings | Effort | Status |
|------|----------|---------------|--------|--------|
| 1 | Section 1256 (SPX not SPY) | $58,363 | None | **Already done** |
| 2 | Leave California | $41,281 | High (move) | Decision pending |
| 3 | S-Corp + Solo 401(k) + deductions | $34,200 | Medium (setup) | **Action item** |
| 4 | Charitable giving (DAF) | $18,268 | Low (if giving) | Personal choice |
| 5 | 1256 loss harvesting (Dec) | $5,000–15,000 | Low (annual) | **Action item** |
| 6 | Backdoor Roth IRA | $1,876 | Low (annual) | **Action item** |
| 7 | Park idle cash in munis/SGOV | $4,000–5,200 | None | **Action item** |

**The realistic annual savings from actionable items (#3, #5, #6, #7):** approximately $45,000–55,000/year, reducing the effective tax rate from 36.5% to roughly 24%.

**If California exit is added (#2):** total savings rise to $87,000–96,000/year, reducing the effective rate to approximately 15.9%.

**Items evaluated and rejected:** box spread borrowing to fund munis (negative spread in all scenarios), leveraging contract count (same tax rate, more risk), BOXX via box spread (negative carry), QOZ funds (liquidity mismatch, marginal deal quality, deferral not elimination).

## 19.12 The Compound Impact

The $35K/year news filter alpha and the tax optimization are surprisingly similar in magnitude:

| Improvement | Annual Value | 10-Year Compound |
|------------|-------------|------------------|
| News filter (skip 73 trades) | +$35,733 | +$423,973 |
| Leave CA | +$41,281 | +$412,805 |
| S-Corp + 401(k) | +$34,200 | +$342,000 |
| Loss harvesting + Roth + idle cash | +$11,876 | +$118,760 |

The news filter and California exit are nearly identical in value — both worth ~$40K/year. One requires connecting a news bot to an entry decision. The other requires a moving truck. Both are one-time actions with permanent, compounding benefits.

Together with the entity structure, the total improvement is $123,090/year over the unoptimized baseline. That's not incremental. That's a second income stream, created entirely by not losing money to bad trades and not losing money to avoidable taxes.
