"""
Account profiles for the scalp bot.

Each profile defines position sizing, risk limits, and execution mode
for a specific brokerage account.
"""

# ─── Robinhood ($2k, commission-free, PDT-restricted) ────────────────────────
ROBINHOOD = {
    "name": "Robinhood",
    "mode": "spot",               # spot share trading
    "ticker": "SPY",              # most liquid, commission-free
    "account_size": 2000,
    "commission_per_side": 0.0,   # commission-free
    "slippage_pct": 0.01,         # tighter slippage (limit orders)

    # Position sizing
    "max_position_dollars": 600,  # ~1 SPY share, leaves room for drawdown
    "max_shares": 1,              # conservative start

    # Risk limits
    "max_loss_per_trade": 15,     # dollars
    "max_daily_loss": 40,         # dollars — hard stop for the day
    "max_drawdown": 200,          # dollars — pause bot if hit (10% of account)

    # PDT management (3 day trades per 5 rolling business days)
    "pdt_limited": True,
    "max_day_trades_per_week": 3,
    "target_trades_per_day": 1,   # save day trades for best signals only

    # Signal filters — 0.60 matches POS_GAMMA mean-reversion signals
    "min_confidence": 0.60,       # tested: catches trail-stop winners at 0.60
    "require_gex_regime": True,   # skip if GEX regime is UNKNOWN
    "cooldown_sec": 600,          # 10 min cooldown

    # ATR risk params
    "sl_atr_mult": 2.0,
    "tp_atr_mult": 4.0,          # wider TP for fewer but bigger wins
    "trail_activate_atr": 2.5,
    "trail_distance_atr": 1.0,
    "max_hold_minutes": 20,
}

# ─── PM Options Account ($500k, 0DTE SPX) ────────────────────────────────────
PM_OPTIONS = {
    "name": "PM Options",
    "mode": "0dte",               # 0DTE SPX option buying
    "ticker": "SPX",
    "account_size": 500_000,
    "commission_per_side": 0.65,  # per contract
    "slippage_pct": 0.0,          # options use limit orders

    # Position sizing
    "contracts": 1,               # start with 1 contract
    "max_premium": 1500,          # max dollars to spend per entry
    "multiplier": 100,            # SPX options = $100 per point

    # Scale-on-trail: add 1 contract when trade reaches +20% premium gain
    # Only scales winners — losers stay at 1 contract (fixed initial risk)
    "scale_on_trail": True,
    "scale_trigger_pct": 20,      # % premium gain to trigger scale-up
    "scale_add_contracts": 1,     # add this many contracts on trigger

    # Risk limits
    "max_loss_per_trade": 1500,   # max = premium paid (long options)
    "max_daily_loss": 3000,       # dollars
    "max_drawdown": 10_000,       # dollars — pause if hit (2% of account)

    # Signal filters
    "min_confidence": 0.60,       # 0.60 catches mean-reversion + momentum
    "require_gex_regime": True,
    "cooldown_sec": 300,          # 5 min
    "max_trades_per_day": 6,

    # Option-specific
    "option_tp_pct": 50,          # take profit at 50% premium gain
    "option_sl_pct": 35,          # stop at 35% premium loss
    "option_trail_pct": 25,       # trail 25% from peak
    "max_hold_minutes": 30,       # 0DTE needs time for directional move
    "min_tte_minutes": 30,        # don't enter if < 30 min to expiry
    "min_premium": 2.00,          # skip if option premium < $2
}

# ─── E*Trade ($261k, reg-T margin, Level 4 options) ──────────────────────────
ETRADE = {
    "name": "E*Trade",
    "mode": "0dte",               # 0DTE SPX option buying (Level 4)
    "ticker": "SPX",
    "account_size": 261_000,
    "commission_per_side": 0.65,  # $0.65/contract E*Trade options
    "slippage_pct": 0.0,          # options use limit orders

    # Position sizing — reg-T, buying options = premium is max risk
    "contracts": 1,               # start conservative
    "max_premium": 1500,          # max dollars per entry
    "multiplier": 100,            # SPX options = $100 per point

    # Scale-on-trail: add 1 contract when trade reaches +20% premium gain
    "scale_on_trail": True,
    "scale_trigger_pct": 20,      # % premium gain to trigger scale-up
    "scale_add_contracts": 1,     # add this many contracts on trigger

    # Risk limits
    "max_loss_per_trade": 1500,   # max = premium paid
    "max_daily_loss": 2500,       # dollars
    "max_drawdown": 5000,         # dollars — pause if hit (~2% of account)

    # No PDT restriction (account > $25k)
    "pdt_limited": False,
    "max_day_trades_per_week": 999,
    "max_trades_per_day": 6,

    # Signal filters
    "min_confidence": 0.60,       # 0.60 catches mean-reversion + momentum
    "require_gex_regime": True,
    "cooldown_sec": 300,

    # Option-specific
    "option_tp_pct": 50,          # take profit at 50% premium gain
    "option_sl_pct": 35,          # stop at 35% premium loss
    "option_trail_pct": 25,       # trail 25% from peak
    "max_hold_minutes": 30,       # give directional move time to develop
    "min_tte_minutes": 30,        # don't enter if < 30 min to expiry
    "min_premium": 2.00,          # skip if option premium < $2
}

# ─── E*Trade XSP ($261k, 0DTE Mini-SPX — 1/10th capital per contract) ───────
ETRADE_XSP = {
    "name": "E*Trade XSP",
    "mode": "0dte",
    "ticker": "XSP",              # Mini-SPX = SPX/10, same $100 multiplier
    "account_size": 261_000,
    "commission_per_side": 0.65,
    "slippage_pct": 0.0,

    # Position sizing — XSP premium ~$120/contract vs SPX ~$1,200
    # Can run 10 contracts for the same capital as 1 SPX
    "contracts": 5,               # 5 XSP ≈ 0.5 SPX in dollar risk
    "max_premium": 300,           # ~$120-150 per XSP contract, cap at $300
    "multiplier": 100,            # same $100 multiplier as SPX

    # Scale-on-trail
    "scale_on_trail": True,
    "scale_trigger_pct": 20,
    "scale_add_contracts": 5,     # double to 10 on trail activation

    # Risk limits — XSP is smaller per contract
    "max_loss_per_trade": 300,    # 5 contracts × ~$42 SL = ~$210 typical
    "max_daily_loss": 1500,
    "max_drawdown": 5000,

    "pdt_limited": False,
    "max_day_trades_per_week": 999,
    "max_trades_per_day": 6,

    "min_confidence": 0.60,
    "require_gex_regime": True,
    "cooldown_sec": 300,

    "option_tp_pct": 50,
    "option_sl_pct": 35,
    "option_trail_pct": 25,
    "max_hold_minutes": 30,
    "min_tte_minutes": 30,
    "min_premium": 0.20,          # XSP premiums are 1/10th — lower threshold
}

# ─── PM XSP ($500k, 0DTE Mini-SPX) ──────────────────────────────────────────
PM_XSP = {
    "name": "PM XSP",
    "mode": "0dte",
    "ticker": "XSP",
    "account_size": 500_000,
    "commission_per_side": 0.65,
    "slippage_pct": 0.0,

    "contracts": 10,              # 10 XSP ≈ 1 SPX in dollar risk
    "max_premium": 300,
    "multiplier": 100,

    "scale_on_trail": True,
    "scale_trigger_pct": 20,
    "scale_add_contracts": 10,    # double to 20 on trail

    "max_loss_per_trade": 600,    # 10 contracts × ~$42 SL
    "max_daily_loss": 3000,
    "max_drawdown": 10_000,

    "pdt_limited": False,
    "max_day_trades_per_week": 999,
    "max_trades_per_day": 6,

    "min_confidence": 0.60,
    "require_gex_regime": True,
    "cooldown_sec": 300,

    "option_tp_pct": 50,
    "option_sl_pct": 35,
    "option_trail_pct": 25,
    "max_hold_minutes": 30,
    "min_tte_minutes": 30,
    "min_premium": 0.20,
}

ACCOUNTS = {
    "robinhood": ROBINHOOD,
    "pm": PM_OPTIONS,
    "etrade": ETRADE,
    "etrade_xsp": ETRADE_XSP,
    "pm_xsp": PM_XSP,
}
