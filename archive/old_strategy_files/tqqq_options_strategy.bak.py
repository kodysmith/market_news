#!/usr/bin/env python3
"""
TQQQ Options Protection Strategy Backtest

Strategy:
- Buy 100 shares of TQQQ every 2 weeks
- Buy put options to protect 66% of position size
- Put options 90 days out, 4 standard deviations OTM
- Roll options when they reach ~30 days to expiration
- Never sell shares, only manage options for protection
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class TQQQOptionsStrategy:
    def __init__(self,
                 initial_capital: float = 1000000,
                 share_purchase_amount: int = 100,
                 purchase_frequency_days: int = 14,  # Optimized: 14-day DCA for more buys
                 protection_ratio: float = 0.66,
                 put_dte: int = 90,
                 roll_dte: int = 30,
                 otm_deviations: float = 4.0,
                 start_date: str = '2020-01-01',
                 end_date: str = '2024-01-01',
                 tqqq_data: Optional[pd.DataFrame] = None,
                 qqq_data: Optional[pd.DataFrame] = None):
        """
        Initialize the TQQQ options protection strategy
        
        Args:
            initial_capital: Starting capital
            share_purchase_amount: Number of shares to buy each period
            purchase_frequency_days: Days between purchases
            protection_ratio: Percentage of position to protect (0.66 = 66%)
            put_dte: Days to expiration for puts
            roll_dte: Days to expiration when to roll options
            otm_deviations: Standard deviations OTM for puts
            start_date: Backtest start date
            end_date: Backtest end date
        """
        self.initial_capital = initial_capital
        self.share_purchase_amount = share_purchase_amount
        self.purchase_frequency_days = purchase_frequency_days
        self.protection_ratio = protection_ratio
        self.put_dte = put_dte
        self.roll_dte = roll_dte
        self.otm_deviations = otm_deviations
        self.start_date = start_date
        self.end_date = end_date
        
        # Strategy state
        self.cash = initial_capital
        self.shares_owned = 0
        self.total_invested = 0
        self.options_positions = []  # List of dicts with option details
        self.trades = []
        self.portfolio_values = []
        self.dates = []
        # Track call positions for collar financing
        self.call_positions = []  # dicts similar to put options
        # Track vertical spreads (put/bull/bear)
        self.spread_positions = []  # list of dicts; each has type: put_spread | bear_call_spread | bull_put_spread

        # Config and risk controls - OPTIMIZED CONFIG FROM WALK-FORWARD
        self.config = {
            'risk_on_put_delta': -0.15,        # Top config: deeper puts in ON for better protection
            'risk_on_call_delta': 0.05,        # Modest calls for income without capping upside
            'risk_off_put_delta': -0.35,       # Aggressive puts in OFF
            'risk_off_call_delta': 0.07,       # Slightly wider calls in OFF
            'ladder_tenors': [35, 70, 105],    # 35/70/105 DTE ladder
            'cover_frac_risk_on': 0.60,
            'cover_frac_risk_off': 0.80,
            'protection_ratio_risk_on': 0.60,  # 60% protection in ON
            'protection_ratio_risk_off': 0.80, # 80% protection in OFF
            'btfd_sigma': 1.5,
            'budget_tiers': {'calm': 0.001, 'normal': 0.003, 'stress': 0.01},  # IVR-tiered budgets
            'put_roll_delta': 0.03,
            'call_roll_delta': 0.35,
            'call_roll_pnl_pct': 0.60,
            'min_hedge': 0.6,                  # Never-naked: always 60% protected
            'use_put_spreads': True,           # Use debit put spreads (cheaper)
            'use_call_spreads': True           # Use bear call spreads (capped risk)
        }
        self.monthly_hedge_budget_pct = 0.01  # 1% of portfolio per month for hedges
        self.hedge_spend_month = {}

        # Download or reuse TQQQ data
        self.tqqq_data = tqqq_data.copy() if tqqq_data is not None else self._download_tqqq_data()

        # Config used by QQQ loader
        self.ivr_window = 252  # ~1y trading days

        # Download or reuse QQQ data for hedging and IV rank
        self.qqq_data = qqq_data.copy() if qqq_data is not None else self._download_qqq_data()

        # Monthly 10-month SMA trend toggle (end-of-month)
        self._compute_monthly_trend_flags()

    def _current_protected_shares(self) -> int:
        """Shares protected via long puts and long legs of put spreads."""
        direct = sum(p.get('contracts', 0) * 100 for p in self.options_positions if p.get('type') == 'put')
        spread_long = 0
        for sp in getattr(self, 'spread_positions', []):
            if sp.get('type') == 'put_spread':
                spread_long += sp.get('contracts', 0) * 100
        return int(direct + spread_long)

    def _hedge_ratio(self) -> float:
        if self.shares_owned <= 0:
            return 1.0
        return min(1.0, self._current_protected_shares() / max(1, self.shares_owned))

    def set_config(self, **overrides):
        for k, v in overrides.items():
            if k in self.config:
                self.config[k] = v
        
    def _download_tqqq_data(self) -> pd.DataFrame:
        """Download TQQQ historical data"""
        print("📊 Downloading TQQQ data...")
        ticker = yf.Ticker("TQQQ")
        data = ticker.history(start=self.start_date, end=self.end_date)
        
        if data.empty:
            raise ValueError("No TQQQ data found for the specified date range")
        
        # Calculate daily returns and volatility
        data['Returns'] = data['Close'].pct_change()
        data['Volatility'] = data['Returns'].rolling(window=20).std() * np.sqrt(252)
        
        # Calculate rolling 20-day volatility for option pricing
        data['20d_Vol'] = data['Returns'].rolling(window=20).std() * np.sqrt(252)
        data['20d_Vol'] = data['20d_Vol'].fillna(method='bfill')

        # Trend regime features
        data['MA200'] = data['Close'].rolling(200).mean()
        data['Regime'] = np.where(data['Close'] > data['MA200'], 'RISK_ON', 'RISK_OFF')
        
        print(f"✅ Downloaded {len(data)} days of TQQQ data")
        return data

    def _download_qqq_data(self) -> pd.DataFrame:
        """Download QQQ historical data and compute vol + IV rank proxy."""
        print("📊 Downloading QQQ data (for hedge and IVR)...")
        ticker = yf.Ticker("QQQ")
        data = ticker.history(start=self.start_date, end=self.end_date)
        if data.empty:
            raise ValueError("No QQQ data found for the specified date range")
        data['Returns'] = data['Close'].pct_change()
        data['20d_Vol'] = data['Returns'].rolling(window=20).std() * np.sqrt(252)
        data['20d_Vol'] = data['20d_Vol'].fillna(method='bfill')
        # IV proxy for QQQ (less noisy than TQQQ); clamp
        data['IV_proxy'] = np.clip(1.1 * data['20d_Vol'], 0.18, 0.55)
        # IV Rank over rolling window
        roll = data['IV_proxy'].rolling(self.ivr_window)
        data['IVR'] = (data['IV_proxy'] - roll.min()) / (roll.max() - roll.min() + 1e-9) * 100.0
        print(f"✅ Downloaded {len(data)} days of QQQ data")
        return data

    def _ivr_tier(self, date) -> str:
        """Map IV rank to tiers: calm/normal/stress."""
        if date in self.qqq_data.index:
            ivr = float(self.qqq_data.loc[date, 'IVR']) if 'IVR' in self.qqq_data.columns else 50.0
        else:
            prev_dates = self.qqq_data.index[self.qqq_data.index <= date]
            ivr = float(self.qqq_data.loc[prev_dates[-1], 'IVR']) if len(prev_dates) and 'IVR' in self.qqq_data.columns else 50.0
        if ivr < 20:
            return 'calm'
        if ivr >= 60:
            return 'stress'
        return 'normal'

    def _monthly_hedge_budget_allowed(self, date, reference_value: float) -> float:
        """Dynamic monthly hedge budget by IVR tier (configurable)."""
        tier = self._ivr_tier(date)
        pct = self.config['budget_tiers'].get(tier, 0.003)
        return reference_value * pct

    def _compute_monthly_trend_flags(self):
        """Compute 10-month SMA (monthly close) and a buy-the-dip trigger on QQQ to gate DCA."""
        q = self.qqq_data.copy()
        # Monthly resample
        monthly_close = q['Close'].resample('M').last()
        ma_10m = monthly_close.rolling(10).mean()
        trend_on = monthly_close > ma_10m
        # Map back to daily dates using forward-fill
        trend_daily = trend_on.reindex(self.tqqq_data.index, method='ffill')
        self.tqqq_data['TREND_ON'] = trend_daily.astype(bool)
        # Buy-the-dip: QQQ below 20DMA by >1.5 std of 20d returns
        q['20dma'] = q['Close'].rolling(20).mean()
        q['ret20sd'] = q['Returns'].rolling(20).std()
        dip = (q['Close'] < (q['20dma'] - 1.5 * q['ret20sd'] * q['Close'].shift(1))).reindex(self.tqqq_data.index, method='ffill')
        self.tqqq_data['BTFD'] = dip.fillna(False)
    
    def _calculate_put_strike(self, current_price: float, volatility: float, dte: int) -> float:
        """
        Calculate put strike price using Black-Scholes approximation
        Strike = S * exp(-volatility * sqrt(dte/365) * deviations)
        """
        if volatility <= 0 or dte <= 0:
            return current_price * 0.8  # Fallback to 20% OTM
        
        # Calculate strike using normal distribution approximation
        strike_factor = np.exp(-volatility * np.sqrt(dte/365) * self.otm_deviations)
        strike = current_price * strike_factor
        
        # Ensure strike is reasonable (not too far OTM)
        min_strike = current_price * 0.5  # At least 50% of current price
        max_strike = current_price * 0.95  # At most 5% OTM
        
        return max(min_strike, min(strike, max_strike))
    
    def _estimate_put_premium(self, current_price: float, strike: float, volatility: float, dte: int) -> float:
        """
        Estimate put option premium using simplified Black-Scholes
        This is a rough approximation for backtesting purposes
        """
        if dte <= 0 or volatility <= 0:
            return 0.01
        
        # Simplified Black-Scholes for puts
        S = current_price
        K = strike
        r = 0.02  # Assume 2% risk-free rate
        sigma = volatility
        T = dte / 365.0
        
        # Calculate d1 and d2
        d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Black-Scholes put formula
        from scipy.stats import norm
        put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        
        # Add some realistic bid-ask spread and minimum price
        put_price = max(put_price * 1.1, 0.01)  # 10% spread, minimum $0.01
        
        return put_price

    def _estimate_call_premium(self, current_price: float, strike: float, volatility: float, dte: int) -> float:
        """
        Estimate call option premium using simplified Black-Scholes
        """
        if dte <= 0 or volatility <= 0:
            return 0.01
        S = current_price
        K = strike
        r = 0.02
        sigma = volatility
        T = dte / 365.0
        d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        from scipy.stats import norm
        call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        call_price = max(call_price * 1.1, 0.01)
        return call_price

    # ===================== Vertical Spread Builders =====================
    def _buy_put_spread(self, date, price, iv, long_delta=-0.20, short_delta=-0.05, dte=35) -> Optional[Dict]:
        """Debit put spread: long higher strike put, short farther OTM put. Sized by protection_ratio."""
        if self.shares_owned <= 0:
            return None
        # Solve strikes by delta
        k_long = self._strike_for_put_delta(price, iv, dte, target_put_delta=long_delta)
        k_short = self._strike_for_put_delta(price, iv, dte, target_put_delta=short_delta)
        # enforce long strike > short strike for put spread
        k_long, k_short = max(k_long, k_short), min(k_long, k_short)
        prem_long = self._estimate_put_premium(price, k_long, iv, dte)
        prem_short = self._estimate_put_premium(price, k_short, iv, dte)
        debit_per = max(prem_long - prem_short, 0.01)
        # size by remaining protection need
        existing = self._current_protected_shares()
        target = int(self.shares_owned * self.protection_ratio)
        shortfall = max(0, target - existing)
        contracts = int(np.ceil(shortfall / 100))
        if contracts <= 0:
            return None
        total_cost = debit_per * 100 * contracts
        # monthly budget check
        mkey = (date.year, date.month)
        if mkey not in self.hedge_spend_month:
            self.hedge_spend_month[mkey] = 0.0
        ref_value = self.portfolio_values[-1] if self.portfolio_values else self.initial_capital
        budget = self._monthly_hedge_budget_allowed(date, ref_value) if hasattr(self, '_monthly_hedge_budget_allowed') else ref_value * self.monthly_hedge_budget_pct
        remaining = max(0.0, budget - self.hedge_spend_month[mkey])
        if total_cost > self.cash or total_cost > remaining:
            return None
        # execute
        self.cash -= total_cost
        self.hedge_spend_month[mkey] += total_cost
        pos = {
            'type': 'put_spread',
            'date_opened': date,
            'dte': dte,
            'contracts': contracts,
            'long_strike': k_long,
            'short_strike': k_short,
            'debit_per': debit_per
        }
        self.spread_positions.append(pos)
        self.trades.append({
            'date': date, 'action': 'BUY_PUT_SPREAD', 'contracts': contracts,
            'long_strike': k_long, 'short_strike': k_short, 'debit_per': debit_per, 'dte': dte
        })
        return pos

    def _sell_bear_call_spread(self, date, price, iv, short_delta=0.15, long_delta=0.30, dte=35, cover_ratio=0.6) -> Optional[Dict]:
        """Credit bear call vertical: short lower strike call, long higher strike call to cap risk."""
        if self.shares_owned < 100:
            return None
        # Only allow if current hedge ratio meets minimum
        if self._hedge_ratio() + 1e-9 < self.config.get('min_hedge', 0.6):
            return None
        ks = self._strike_for_call_delta(price, iv, dte, target_call_delta=short_delta)
        kl = self._strike_for_call_delta(price, iv, dte, target_call_delta=long_delta)
        short_k, long_k = min(ks, kl), max(ks, kl)
        prem_short = self._estimate_call_premium(price, short_k, iv, dte)
        prem_long = self._estimate_call_premium(price, long_k, iv, dte)
        credit_per = max(prem_short - prem_long, 0.01)
        max_cover_shares = int(self.shares_owned * cover_ratio)
        contracts = max(0, max_cover_shares // 100)
        if contracts <= 0:
            return None
        proceeds = credit_per * 100 * contracts
        self.cash += proceeds
        pos = {
            'type': 'bear_call_spread',
            'date_opened': date,
            'dte': dte,
            'contracts': contracts,
            'short_strike': short_k,
            'long_strike': long_k,
            'credit_per': credit_per
        }
        self.spread_positions.append(pos)
        self.trades.append({
            'date': date, 'action': 'SELL_BEAR_CALL_SPREAD', 'contracts': contracts,
            'short_strike': short_k, 'long_strike': long_k, 'credit_per': credit_per, 'dte': dte
        })
        return pos

    def _sell_bull_put_spread(self, date, price, iv, short_delta=-0.15, long_delta=-0.30, dte=35, contracts: Optional[int]=None) -> Optional[Dict]:
        """Credit bull put vertical for income with defined risk."""
        ks = self._strike_for_put_delta(price, iv, dte, target_put_delta=short_delta)
        kl = self._strike_for_put_delta(price, iv, dte, target_put_delta=long_delta)
        short_k, long_k = max(ks, kl), min(ks, kl)
        prem_short = self._estimate_put_premium(price, short_k, iv, dte)
        prem_long = self._estimate_put_premium(price, long_k, iv, dte)
        credit_per = max(prem_short - prem_long, 0.01)
        n = contracts if contracts is not None else max(1, self.shares_owned // 100)
        proceeds = credit_per * 100 * n
        self.cash += proceeds
        pos = {
            'type': 'bull_put_spread',
            'date_opened': date,
            'dte': dte,
            'contracts': n,
            'short_strike': short_k,
            'long_strike': long_k,
            'credit_per': credit_per
        }
        self.spread_positions.append(pos)
        self.trades.append({
            'date': date, 'action': 'SELL_BULL_PUT_SPREAD', 'contracts': n,
            'short_strike': short_k, 'long_strike': long_k, 'credit_per': credit_per, 'dte': dte
        })
        return pos

    def _put_delta(self, S: float, K: float, sigma: float, dte: int, r: float = 0.02) -> float:
        """Approximate Black–Scholes put delta."""
        T = max(dte / 365.0, 1/365)
        if sigma <= 0:
            return -1.0 if K > S else 0.0
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        from scipy.stats import norm
        return norm.cdf(d1) - 1.0  # put delta

    def _call_delta(self, S: float, K: float, sigma: float, dte: int, r: float = 0.02) -> float:
        """Approximate Black–Scholes call delta."""
        T = max(dte / 365.0, 1/365)
        if sigma <= 0:
            return 1.0 if S > K else 0.0
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        from scipy.stats import norm
        return norm.cdf(d1)

    def _strike_for_put_delta(self, S: float, sigma: float, dte: int, target_put_delta: float = -0.20, r: float = 0.02) -> float:
        """Solve for strike K that yields approximately the target put delta."""
        from scipy.optimize import brentq
        def f(K):
            return self._put_delta(S, K, sigma, dte, r) - target_put_delta
        # bracket between 40% and 99% of spot
        K_low, K_high = max(0.01, S * 0.40), S * 0.99
        try:
            return brentq(f, K_low, K_high, maxiter=100)
        except Exception:
            # fallback to percentage OTM if solver fails
            return max(S * 0.5, min(S * 0.95, S * np.exp(-sigma * np.sqrt(max(dte/365.0, 1/365)) * 2.0)))

    def _strike_for_call_delta(self, S: float, sigma: float, dte: int, target_call_delta: float = 0.15, r: float = 0.02) -> float:
        """Solve for strike K that yields approximately the target call delta."""
        from scipy.optimize import brentq
        def f(K):
            return self._call_delta(S, K, sigma, dte, r) - target_call_delta
        # bracket slightly OTM to 200% of spot for safety
        K_low, K_high = S * 0.9, S * 2.0
        try:
            return brentq(f, K_low, K_high, maxiter=100)
        except Exception:
            # fallback: move up by moneyness from delta heuristic
            return S * (1.0 + 0.15)

    def _proxy_iv(self, date) -> float:
        """Proxy IV from realized vol with bounds for TQQQ; adds a simple skew bump later if needed."""
        base = float(self.tqqq_data.loc[date, '20d_Vol']) if '20d_Vol' in self.tqqq_data.columns else 0.5
        iv = np.clip(1.2 * base, 0.35, 1.10)  # scale & clamp
        return float(iv)
    
    def _buy_shares(self, date: datetime, price: float) -> Dict:
        """Buy shares and record transaction"""
        cost = self.share_purchase_amount * price
        
        if cost > self.cash:
            print(f"⚠️  Insufficient cash to buy shares on {date.strftime('%Y-%m-%d')}")
            return None
        
        self.cash -= cost
        self.shares_owned += self.share_purchase_amount
        self.total_invested += cost
        
        trade = {
            'date': date,
            'action': 'BUY_SHARES',
            'quantity': self.share_purchase_amount,
            'price': price,
            'cost': cost,
            'shares_owned': self.shares_owned,
            'cash_remaining': self.cash
        }
        
        self.trades.append(trade)
        print(f"📈 Bought {self.share_purchase_amount} TQQQ shares at ${price:.2f} on {date.strftime('%Y-%m-%d')}")
        
        return trade
    
    def _buy_put_protection(self, date: datetime, current_price: float, base_iv: float, target_delta: float, dte: Optional[int] = None) -> Optional[Dict]:
        """Buy put options sized to protection_ratio using delta-targeted strikes and a monthly hedge budget."""
        if self.shares_owned == 0:
            return None

        dte = dte if dte is not None else self.put_dte
        sigma = max(base_iv, 1e-6)

        # Strike via delta target
        strike = self._strike_for_put_delta(current_price, sigma, dte, target_put_delta=target_delta)

        # Premium via BS proxy
        premium = self._estimate_put_premium(current_price, strike, sigma, dte)

        # Contracts sized by protection_ratio minus existing protection
        existing_protected = sum(p['contracts'] * 100 for p in self.options_positions if p.get('type') == 'put')
        target_protected = int(self.shares_owned * self.protection_ratio)
        shortfall_shares = max(0, target_protected - existing_protected)
        contracts_needed = int(np.ceil(shortfall_shares / 100))
        if contracts_needed <= 0:
            return None

        total_cost = contracts_needed * premium * 100

        # Monthly hedge budget enforcement
        mkey = (date.year, date.month)
        if mkey not in self.hedge_spend_month:
            self.hedge_spend_month[mkey] = 0.0
        # Use last portfolio value if available; else initial capital
        ref_value = self.portfolio_values[-1] if self.portfolio_values else self.initial_capital
        # Dynamic budget by IVR tier
        budget = self._monthly_hedge_budget_allowed(date, ref_value) if hasattr(self, '_monthly_hedge_budget_allowed') else ref_value * self.monthly_hedge_budget_pct
        remaining_budget = max(0.0, budget - self.hedge_spend_month[mkey])

        if total_cost > self.cash or total_cost > remaining_budget:
            # If over budget or cash, skip buying new protection
            print(f"⚠️  Skipping put buy on {date.strftime('%Y-%m-%d')} (cost ${total_cost:,.0f} exceeds budget/cash)")
            return None

        # Execute
        self.cash -= total_cost
        self.hedge_spend_month[mkey] += total_cost

        option_position = {
            'date_opened': date,
            'strike': strike,
            'premium': premium,
            'contracts': contracts_needed,
            'total_cost': total_cost,
            'dte': dte,
            'shares_protected': contracts_needed * 100,
            'type': 'put'
        }
        self.options_positions.append(option_position)

        trade = {
            'date': date,
            'action': 'BUY_PUTS',
            'strike': strike,
            'premium': premium,
            'contracts': contracts_needed,
            'cost': total_cost,
            'shares_protected': contracts_needed * 100,
            'dte': dte,
            'target_delta': target_delta
        }
        self.trades.append(trade)
        print(f"🛡️  Bought {contracts_needed} put contracts at ${strike:.2f} strike (≈{abs(target_delta)*100:.0f}Δ) for ${premium:.2f} on {date.strftime('%Y-%m-%d')}")
        return trade

    def _sell_covered_calls(self, date: datetime, current_price: float, base_iv: float, target_delta: float, dte: int = 35, cover_fraction: Optional[float] = None) -> Optional[Dict]:
        """Sell covered calls on a subset of shares to finance puts."""
        if self.shares_owned < 100:
            return None
        # Never-naked: require minimum hedge before selling calls
        if self._hedge_ratio() + 1e-9 < self.config.get('min_hedge', 0.6):
            return None
        sigma = max(base_iv, 1e-6)
        strike = self._strike_for_call_delta(current_price, sigma, dte, target_call_delta=target_delta)
        premium = self._estimate_call_premium(current_price, strike, sigma, dte)
        # cover up to 60% of shares minus existing covered
        existing_covered = sum(c['contracts'] * 100 for c in self.call_positions if c.get('type') == 'call')
        frac = cover_fraction if cover_fraction is not None else 0.6
        max_cover_shares = int(self.shares_owned * frac)
        shortfall_cover = max(0, max_cover_shares - existing_covered)
        contracts = max(0, shortfall_cover // 100)
        if contracts <= 0:
            return None
        notional = contracts * 100 * current_price
        premium_received = contracts * premium * 100
        self.cash += premium_received
        position = {
            'date_opened': date,
            'strike': strike,
            'premium': premium,
            'contracts': contracts,
            'dte': dte,
            'type': 'call'
        }
        self.call_positions.append(position)
        trade = {
            'date': date,
            'action': 'SELL_CALLS',
            'strike': strike,
            'premium': premium,
            'contracts': contracts,
            'proceeds': premium_received,
            'dte': dte,
            'target_delta': target_delta
        }
        self.trades.append(trade)
        print(f"💵 Sold {contracts} covered call contracts at ${strike:.2f} (≈{target_delta*100:.0f}Δ) for ${premium:.2f} on {date.strftime('%Y-%m-%d')}")
        return trade

    def _roll_options_smart(self, date: datetime, current_price: float, base_iv: float) -> List[Dict]:
        """Smarter rolling:
        - Puts: roll if DTE < 25, or (abs(Δ) < 0.03 and IVR < 20). Otherwise let tails decay.
        - Calls: roll if Δ >= 0.35 or if MTM loss > 60% of collected premium; extend tenor and lift strike back to target Δ.
        """
        rolled_trades = []
        keep_positions = []
        # Handle puts
        for position in self.options_positions:
            days_held = (date - position['date_opened']).days
            current_dte = max(position['dte'] - days_held, 0)
            sigma = max(base_iv, 1e-6)

            # MTM value of current position
            mtm = self._estimate_put_premium(current_price, position['strike'], sigma, max(current_dte, 1)) * 100 * position['contracts']

            # Compute delta of existing put
            put_delta_now = self._put_delta(current_price, position['strike'], sigma, max(current_dte, 1))

            # IVR tier
            if date in self.qqq_data.index:
                ivr_now = float(self.qqq_data.loc[date, 'IVR']) if 'IVR' in self.qqq_data.columns else 50.0
            else:
                prev_dates = self.qqq_data.index[self.qqq_data.index <= date]
                ivr_now = float(self.qqq_data.loc[prev_dates[-1], 'IVR']) if len(prev_dates) and 'IVR' in self.qqq_data.columns else 50.0

            need_roll = (current_dte <= 25) or ((abs(put_delta_now) < 0.03) and (ivr_now < 20))

            if need_roll:
                # Close: receive MTM
                self.cash += mtm

                # Determine regime to set target delta
                regime = self.tqqq_data.loc[date, 'Regime'] if 'Regime' in self.tqqq_data.columns else 'RISK_ON'
                target_delta = -0.20 if regime == 'RISK_ON' else -0.10

                # Rebuy new protection (budget will be enforced inside)
                new_trade = self._buy_put_protection(date, current_price, base_iv, target_delta, dte=self.put_dte)
                if new_trade:
                    rolled_trades.append({
                        'date': date,
                        'action': 'ROLL_PUTS',
                        'old_strike': position['strike'],
                        'new_strike': new_trade['strike'],
                        'old_dte': current_dte,
                        'new_dte': self.put_dte,
                        'contracts': position['contracts'],
                        'close_value': mtm,
                        'target_delta': target_delta
                    })
                else:
                    # If unable to rebuy (budget), we simply close and drop protection
                    rolled_trades.append({
                        'date': date,
                        'action': 'CLOSE_PUTS_NO_REBUY',
                        'old_strike': position['strike'],
                        'old_dte': current_dte,
                        'contracts': position['contracts'],
                        'close_value': mtm
                    })
            else:
                keep_positions.append(position)

        self.options_positions = keep_positions

        # Handle calls
        keep_calls = []
        for cpos in self.call_positions:
            days_held = (date - cpos['date_opened']).days
            current_dte = max(cpos['dte'] - days_held, 0)
            sigma = max(base_iv, 1e-6)
            call_price_now = self._estimate_call_premium(current_price, cpos['strike'], sigma, max(current_dte, 1))
            pnl_per_contract = (cpos['premium'] - call_price_now) * 100  # we are short; negative means loss
            delta_now = self._call_delta(current_price, cpos['strike'], sigma, max(current_dte, 1))

            roll_call = (delta_now >= self.config['call_roll_delta']) or (pnl_per_contract < -self.config['call_roll_pnl_pct'] * cpos['premium'] * 100) or (current_dte <= 10)
            if roll_call:
                # Close short call
                close_cost = call_price_now * cpos['contracts'] * 100
                self.cash -= close_cost
                # Open new at target ~15Δ, 35D
                new_strike = self._strike_for_call_delta(current_price, sigma, 35, target_call_delta=self.config['risk_on_call_delta'])
                new_prem = self._estimate_call_premium(current_price, new_strike, sigma, 35)
                proceeds = new_prem * cpos['contracts'] * 100
                self.cash += proceeds
                new_cpos = {
                    'date_opened': date,
                    'strike': new_strike,
                    'premium': new_prem,
                    'contracts': cpos['contracts'],
                    'dte': 35,
                    'type': 'call'
                }
                keep_calls.append(new_cpos)
                rolled_trades.append({
                    'date': date,
                    'action': 'ROLL_CALLS',
                    'old_strike': cpos['strike'],
                    'new_strike': new_strike,
                    'old_dte': current_dte,
                    'new_dte': 35,
                    'contracts': cpos['contracts']
                })
            else:
                keep_calls.append(cpos)

        self.call_positions = keep_calls
        if rolled_trades:
            self.trades.extend(rolled_trades)
        # NOTE: Spread roll logic can be added here if desired; currently held to expiry/DTE logic above.
        return rolled_trades
    
    def _roll_options(self, date: datetime, current_price: float, volatility: float) -> List[Dict]:
        """Roll options that are close to expiration"""
        rolled_trades = []
        positions_to_remove = []
        
        for i, position in enumerate(self.options_positions):
            days_held = (date - position['date_opened']).days
            current_dte = position['dte'] - days_held
            
            if current_dte <= self.roll_dte:
                # Close current position (simplified - assume we get back some value)
                remaining_value = position['total_cost'] * (current_dte / position['dte']) * 0.5
                self.cash += remaining_value
                
                # Buy new protection
                new_strike = self._calculate_put_strike(current_price, volatility, self.put_dte)
                new_premium = self._estimate_put_premium(current_price, new_strike, volatility, self.put_dte)
                new_contracts = position['contracts']
                new_cost = new_contracts * new_premium * 100
                
                if new_cost <= self.cash:
                    self.cash -= new_cost
                    
                    # Create new position
                    new_position = {
                        'date_opened': date,
                        'strike': new_strike,
                        'premium': new_premium,
                        'contracts': new_contracts,
                        'total_cost': new_cost,
                        'dte': self.put_dte,
                        'shares_protected': new_contracts * 100
                    }
                    
                    self.options_positions.append(new_position)
                    positions_to_remove.append(i)
                    
                    trade = {
                        'date': date,
                        'action': 'ROLL_PUTS',
                        'old_strike': position['strike'],
                        'new_strike': new_strike,
                        'old_premium': position['premium'],
                        'new_premium': new_premium,
                        'contracts': new_contracts,
                        'net_cost': new_cost - remaining_value,
                        'dte': self.put_dte
                    }
                    
                    self.trades.append(trade)
                    rolled_trades.append(trade)
                    print(f"🔄 Rolled {new_contracts} put contracts from ${position['strike']:.2f} to ${new_strike:.2f} on {date.strftime('%Y-%m-%d')}")
        
        # Remove rolled positions
        for i in reversed(positions_to_remove):
            self.options_positions.pop(i)
        
        return rolled_trades
    
    def _calculate_portfolio_value(self, date: datetime, current_price: float) -> float:
        """Calculate total portfolio value"""
        shares_value = self.shares_owned * current_price
        
        # Calculate options value (mark-to-model)
        options_value = 0
        for position in self.options_positions:
            days_held = (date - position['date_opened']).days
            current_dte = position['dte'] - days_held

            if current_dte > 0:
                # Mark-to-model using the same BS pricer used for premium estimation
                option_price = self._estimate_put_premium(current_price, position['strike'],
                                                          max(self.tqqq_data.loc[date, '20d_Vol'], 0.35) if '20d_Vol' in self.tqqq_data.columns else 0.5,
                                                          current_dte)
                option_value = option_price * position['contracts'] * 100
                options_value += option_value

        # Mark call positions (liability due to cap). For simplicity, use BS price as negative value (we are short).
        for position in self.call_positions:
            days_held = (date - position['date_opened']).days
            current_dte = max(position['dte'] - days_held, 0)
            if current_dte > 0:
                call_price = self._estimate_call_premium(current_price, position['strike'],
                                                         max(self.tqqq_data.loc[date, '20d_Vol'], 0.35) if '20d_Vol' in self.tqqq_data.columns else 0.5,
                                                         current_dte)
                options_value -= call_price * position['contracts'] * 100

        # Value vertical spreads (two-leg MTM)
        for sp in getattr(self, 'spread_positions', []):
            days_held = (date - sp['date_opened']).days
            dte = max(sp['dte'] - days_held, 0)
            if dte <= 0:
                continue
            iv = max(self.tqqq_data.loc[date, '20d_Vol'], 0.35) if '20d_Vol' in self.tqqq_data.columns else 0.5
            if sp['type'] in ('put_spread', 'bull_put_spread'):
                long_leg = self._estimate_put_premium(current_price, sp['long_strike'], iv, dte)
                short_leg = self._estimate_put_premium(current_price, sp['short_strike'], iv, dte)
                if sp['type'] == 'put_spread':
                    spread_val = max((long_leg - short_leg) * 100 * sp['contracts'], 0.0)
                else:
                    spread_val = -max((short_leg - long_leg) * 100 * sp['contracts'], 0.0)
                options_value += spread_val
            elif sp['type'] == 'bear_call_spread':
                short_leg = self._estimate_call_premium(current_price, sp['short_strike'], iv, dte)
                long_leg  = self._estimate_call_premium(current_price, sp['long_strike'], iv, dte)
                spread_val = -max((short_leg - long_leg) * 100 * sp['contracts'], 0.0)
                options_value += spread_val
        
        return self.cash + shares_value + options_value
    
    def run_backtest(self) -> Dict:
        """Run the complete backtest"""
        print("🚀 Starting TQQQ Options Protection Strategy Backtest")
        print("=" * 60)
        
        # Get trading dates
        trading_dates = self.tqqq_data.index.tolist()
        last_purchase_date = None

        for i, date in enumerate(trading_dates):
            current_price = self.tqqq_data.loc[date, 'Close']
            base_iv = self._proxy_iv(date)
            regime = self.tqqq_data.loc[date, 'Regime'] if 'Regime' in self.tqqq_data.columns else 'RISK_ON'
            # QQQ IV rank (use nearest available date index)
            if date in self.qqq_data.index:
                ivr = float(self.qqq_data.loc[date, 'IVR']) if 'IVR' in self.qqq_data.columns else 50.0
            else:
                # nearest previous
                prev_dates = self.qqq_data.index[self.qqq_data.index <= date]
                ivr = float(self.qqq_data.loc[prev_dates[-1], 'IVR']) if len(prev_dates) and 'IVR' in self.qqq_data.columns else 50.0

            # Rebalance buying cadence by regime
            should_buy_today = (last_purchase_date is None or (date - last_purchase_date).days >= self.purchase_frequency_days)
            # DCA when trend is ON and buy-the-dip triggers; avoid random buys
            if should_buy_today and self.tqqq_data.get('TREND_ON', pd.Series(True, index=self.tqqq_data.index)).loc[date]:
                if regime == 'RISK_ON' and (self.tqqq_data.get('BTFD', pd.Series(False, index=self.tqqq_data.index)).loc[date]):
                    self._buy_shares(date, current_price)
                    # Ladder collars: 30/60/90 DTE thirds
                    self.protection_ratio = self.config['protection_ratio_risk_on']
                    put_delta = self.config['risk_on_put_delta']
                    call_delta = self.config['risk_on_call_delta']
                    for tenor in self.config['ladder_tenors']:
                        # Protection first (either put or put spread depending on config)
                        if self.config.get('use_put_spreads', True):
                            self._buy_put_spread(date, current_price, base_iv, long_delta=put_delta, short_delta=-0.05, dte=tenor)
                        else:
                            self._buy_put_protection(date, current_price, base_iv, put_delta, dte=tenor)
                        # Only after hedge is established, sell the call leg (covered or spread)
                        if self.config.get('use_call_spreads', True):
                            self._sell_bear_call_spread(date, current_price, base_iv, short_delta=call_delta, long_delta=0.30, dte=tenor, cover_ratio=self.config['cover_frac_risk_on'])
                        else:
                            self._sell_covered_calls(date, current_price, base_iv, call_delta, dte=tenor, cover_fraction=self.config['cover_frac_risk_on'])
                    last_purchase_date = date
                elif regime == 'RISK_OFF':
                    # No new shares; add protection ladder and tighter puts, smaller calls
                    self.protection_ratio = self.config['protection_ratio_risk_off']
                    for tenor in self.config['ladder_tenors']:
                        if self.config.get('use_put_spreads', True):
                            self._buy_put_spread(date, current_price, base_iv, long_delta=self.config['risk_off_put_delta'], short_delta=-0.10, dte=tenor)
                        else:
                            self._buy_put_protection(date, current_price, base_iv, self.config['risk_off_put_delta'], dte=tenor)
                        if self.config.get('use_call_spreads', True):
                            self._sell_bear_call_spread(date, current_price, base_iv, short_delta=self.config['risk_off_call_delta'], long_delta=0.25, dte=tenor, cover_ratio=self.config['cover_frac_risk_off'])
                        else:
                            self._sell_covered_calls(date, current_price, base_iv, self.config['risk_off_call_delta'], dte=tenor, cover_fraction=self.config['cover_frac_risk_off'])
                    last_purchase_date = date

            # TODO: smarter roll for calls and puts using delta and IVR
            self._roll_options_smart(date, current_price, base_iv)

            # Calculate portfolio value
            portfolio_value = self._calculate_portfolio_value(date, current_price)
            self.portfolio_values.append(portfolio_value)
            self.dates.append(date)
        
        # Calculate final results
        final_value = self.portfolio_values[-1]
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        # Calculate benchmark (buy and hold TQQQ)
        initial_price = self.tqqq_data['Close'].iloc[0]
        final_price = self.tqqq_data['Close'].iloc[-1]
        benchmark_return = (final_price - initial_price) / initial_price
        
        results = {
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'benchmark_return': benchmark_return,
            'excess_return': total_return - benchmark_return,
            'shares_owned': self.shares_owned,
            'total_invested': self.total_invested,
            'cash_remaining': self.cash,
            'total_trades': len(self.trades),
            'options_positions': len(self.options_positions),
            'portfolio_values': self.portfolio_values,
            'dates': self.dates,
            'trades': self.trades
        }
        
        return results
    
    def plot_results(self, results: Dict):
        """Plot backtest results"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Portfolio value over time
        ax1.plot(results['dates'], results['portfolio_values'], label='Strategy', linewidth=2)
        
        # Benchmark (buy and hold TQQQ)
        initial_price = self.tqqq_data['Close'].iloc[0]
        benchmark_values = [initial_price * (price / initial_price) * (self.initial_capital / initial_price) 
                           for price in self.tqqq_data['Close']]
        ax1.plot(self.tqqq_data.index, benchmark_values, label='Buy & Hold TQQQ', linewidth=2, alpha=0.7)
        
        # Add markers for trading events
        trades_df = pd.DataFrame(results['trades'])
        if not trades_df.empty:
            # Convert trade dates to datetime if they're not already
            trades_df['date'] = pd.to_datetime(trades_df['date'])
            
            # Find portfolio values at trade dates
            trade_markers = []
            trade_values = []
            trade_actions = []
            
            for _, trade in trades_df.iterrows():
                trade_date = trade['date']
                # Find closest portfolio value date
                closest_idx = min(range(len(results['dates'])), 
                                key=lambda i: abs((results['dates'][i] - trade_date).total_seconds()))
                trade_markers.append(results['dates'][closest_idx])
                trade_values.append(results['portfolio_values'][closest_idx])
                trade_actions.append(trade['action'])
            
            # Plot markers for different actions
            buy_puts_mask = [action == 'BUY_PUTS' for action in trade_actions]
            roll_puts_mask = [action == 'ROLL_PUTS' for action in trade_actions]
            buy_shares_mask = [action == 'BUY_SHARES' for action in trade_actions]
            close_puts_mask = [action == 'CLOSE_PUTS_NO_REBUY' for action in trade_actions]
            sell_calls_mask = [action == 'SELL_CALLS' for action in trade_actions]
            roll_calls_mask = [action == 'ROLL_CALLS' for action in trade_actions]
            
            if any(buy_puts_mask):
                buy_puts_dates = [d for d, mask in zip(trade_markers, buy_puts_mask) if mask]
                buy_puts_values = [v for v, mask in zip(trade_values, buy_puts_mask) if mask]
                ax1.scatter(buy_puts_dates, buy_puts_values, color='green', marker='^', 
                           s=60, label='Buy Puts', alpha=0.8, zorder=5)
            
            if any(roll_puts_mask):
                roll_puts_dates = [d for d, mask in zip(trade_markers, roll_puts_mask) if mask]
                roll_puts_values = [v for v, mask in zip(trade_values, roll_puts_mask) if mask]
                ax1.scatter(roll_puts_dates, roll_puts_values, color='orange', marker='s', 
                           s=60, label='Roll Puts', alpha=0.8, zorder=5)
            
            if any(buy_shares_mask):
                buy_shares_dates = [d for d, mask in zip(trade_markers, buy_shares_mask) if mask]
                buy_shares_values = [v for v, mask in zip(trade_values, buy_shares_mask) if mask]
                ax1.scatter(buy_shares_dates, buy_shares_values, color='blue', marker='o', 
                           s=60, label='Buy Shares', alpha=0.8, zorder=5)
            
            if any(close_puts_mask):
                close_puts_dates = [d for d, mask in zip(trade_markers, close_puts_mask) if mask]
                close_puts_values = [v for v, mask in zip(trade_values, close_puts_mask) if mask]
                ax1.scatter(close_puts_dates, close_puts_values, color='red', marker='x', 
                           s=60, label='Close Puts', alpha=0.8, zorder=5)
            
            if any(sell_calls_mask):
                sell_call_dates = [d for d, mask in zip(trade_markers, sell_calls_mask) if mask]
                sell_call_values = [v for v, mask in zip(trade_values, sell_calls_mask) if mask]
                ax1.scatter(sell_call_dates, sell_call_values, color='purple', marker='v', 
                           s=60, label='Sell Calls', alpha=0.8, zorder=5)
            if any(roll_calls_mask):
                roll_call_dates = [d for d, mask in zip(trade_markers, roll_calls_mask) if mask]
                roll_call_values = [v for v, mask in zip(trade_values, roll_calls_mask) if mask]
                ax1.scatter(roll_call_dates, roll_call_values, color='magenta', marker='P', 
                           s=70, label='Roll Calls', alpha=0.9, zorder=6)
        
        ax1.set_title('Portfolio Value Over Time with Trading Events')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # TQQQ price with trading events
        ax2.plot(self.tqqq_data.index, self.tqqq_data['Close'], label='TQQQ Price', linewidth=2, alpha=0.7)
        
        # Add trading event markers on price chart
        if not trades_df.empty:
            # Plot markers for different actions on price chart
            if any(buy_puts_mask):
                buy_puts_prices = [self.tqqq_data.loc[d, 'Close'] for d in buy_puts_dates if d in self.tqqq_data.index]
                buy_puts_dates_valid = [d for d in buy_puts_dates if d in self.tqqq_data.index]
                ax2.scatter(buy_puts_dates_valid, buy_puts_prices, color='green', marker='^', 
                           s=60, label='Buy Puts', alpha=0.8, zorder=5)
            
            if any(roll_puts_mask):
                roll_puts_prices = [self.tqqq_data.loc[d, 'Close'] for d in roll_puts_dates if d in self.tqqq_data.index]
                roll_puts_dates_valid = [d for d in roll_puts_dates if d in self.tqqq_data.index]
                ax2.scatter(roll_puts_dates_valid, roll_puts_prices, color='orange', marker='s', 
                           s=60, label='Roll Puts', alpha=0.8, zorder=5)
            
            if any(buy_shares_mask):
                buy_shares_prices = [self.tqqq_data.loc[d, 'Close'] for d in buy_shares_dates if d in self.tqqq_data.index]
                buy_shares_dates_valid = [d for d in buy_shares_dates if d in self.tqqq_data.index]
                ax2.scatter(buy_shares_dates_valid, buy_shares_prices, color='blue', marker='o', 
                           s=60, label='Buy Shares', alpha=0.8, zorder=5)
            
            if any(close_puts_mask):
                close_puts_prices = [self.tqqq_data.loc[d, 'Close'] for d in close_puts_dates if d in self.tqqq_data.index]
                close_puts_dates_valid = [d for d in close_puts_dates if d in self.tqqq_data.index]
                ax2.scatter(close_puts_dates_valid, close_puts_prices, color='red', marker='x', 
                           s=60, label='Close Puts', alpha=0.8, zorder=5)
            if any(sell_calls_mask):
                sell_call_prices = [self.tqqq_data.loc[d, 'Close'] for d in sell_call_dates if d in self.tqqq_data.index]
                sell_call_dates_valid = [d for d in sell_call_dates if d in self.tqqq_data.index]
                ax2.scatter(sell_call_dates_valid, sell_call_prices, color='purple', marker='v', 
                           s=60, label='Sell Calls', alpha=0.8, zorder=5)
            if any(roll_calls_mask):
                roll_call_prices = [self.tqqq_data.loc[d, 'Close'] for d in roll_call_dates if d in self.tqqq_data.index]
                roll_call_dates_valid = [d for d in roll_call_dates if d in self.tqqq_data.index]
                ax2.scatter(roll_call_dates_valid, roll_call_prices, color='magenta', marker='P', 
                           s=70, label='Roll Calls', alpha=0.9, zorder=6)
        
        ax2.set_title('TQQQ Price with Trading Events')
        ax2.set_ylabel('TQQQ Price ($)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Drawdown analysis
        portfolio_series = pd.Series(results['portfolio_values'], index=results['dates'])
        rolling_max = portfolio_series.expanding().max()
        drawdown = (portfolio_series - rolling_max) / rolling_max
        
        ax3.fill_between(results['dates'], drawdown, 0, alpha=0.3, color='red')
        ax3.set_title('Portfolio Drawdown')
        ax3.set_ylabel('Drawdown (%)')
        ax3.grid(True, alpha=0.3)
        
        # Trade analysis
        if not trades_df.empty:
            trade_counts = trades_df['action'].value_counts()
            ax4.pie(trade_counts.values, labels=trade_counts.index, autopct='%1.1f%%')
            ax4.set_title('Trade Distribution')
        
        # Performance metrics
        metrics = [
            f"Total Return: {results['total_return']:.1%}",
            f"Benchmark Return: {results['benchmark_return']:.1%}",
            f"Excess Return: {results['excess_return']:.1%}",
            f"Final Value: ${results['final_value']:,.0f}",
            f"Shares Owned: {results['shares_owned']:,}",
            f"Total Trades: {results['total_trades']}"
        ]
        
        # Create a text box for metrics in the drawdown subplot
        ax3.text(0.02, 0.98, '\n'.join(metrics), transform=ax3.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
    
    def print_summary(self, results: Dict):
        """Print backtest summary"""
        print("\n" + "="*60)
        print("📊 TQQQ OPTIONS PROTECTION STRATEGY RESULTS")
        print("="*60)
        print(f"Initial Capital: ${results['initial_capital']:,.0f}")
        print(f"Final Value: ${results['final_value']:,.0f}")
        print(f"Total Return: {results['total_return']:.1%}")
        print(f"Benchmark Return: {results['benchmark_return']:.1%}")
        print(f"Excess Return: {results['excess_return']:.1%}")
        print(f"Shares Owned: {results['shares_owned']:,}")
        print(f"Total Invested: ${results['total_invested']:,.0f}")
        print(f"Cash Remaining: ${results['cash_remaining']:,.0f}")
        print(f"Total Trades: {results['total_trades']}")
        print(f"Active Options Positions: {results['options_positions']}")
        
        # Calculate some additional metrics
        portfolio_series = pd.Series(results['portfolio_values'], index=results['dates'])
        returns = portfolio_series.pct_change().dropna()
        
        if len(returns) > 0:
            volatility = returns.std() * np.sqrt(252)
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
            max_drawdown = (portfolio_series / portfolio_series.expanding().max() - 1).min()

            # Approximate monthly hedge spend % (using initial cap if no history)
            total_spend = sum(self.hedge_spend_month.values()) if hasattr(self, 'hedge_spend_month') else 0.0
            ref_value = results['final_value'] if results.get('final_value') else self.initial_capital
            avg_months = max(1, len(self.hedge_spend_month)) if hasattr(self, 'hedge_spend_month') else 1
            avg_monthly_spend_pct = (total_spend / avg_months) / ref_value if ref_value else 0.0
            
            print(f"\nRisk Metrics:")
            print(f"Annualized Volatility: {volatility:.1%}")
            print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
            print(f"Maximum Drawdown: {max_drawdown:.1%}")
            print(f"Avg Monthly Hedge Spend: {avg_monthly_spend_pct:.2%}")
            print(f"Current Hedge Ratio: {self._hedge_ratio():.0%}")

def main():
    """Run the TQQQ options strategy backtest"""
    print("🏦 TQQQ Options Protection Strategy Backtest")
    print("=" * 60)
    
    # Strategy parameters - OPTIMIZED
    strategy = TQQQOptionsStrategy(
        initial_capital=1000000,
        share_purchase_amount=100,
        purchase_frequency_days=14,  # Every 2 weeks; buys gated by trend + dip
        protection_ratio=0.50,  # Base; overridden per regime inside
        put_dte=120,  # Not used directly when laddering (35/70/105)
        roll_dte=30,  # Roll when 30 days left
        otm_deviations=2.0,  # delta-targeted in code (legacy σ input unused)
        start_date='2020-01-01',
        end_date='2024-01-01'
    )
    
    # Run backtest
    results = strategy.run_backtest()
    
    # Print results
    strategy.print_summary(results)
    
    # Plot results
    strategy.plot_results(results)
    
    # Save detailed results
    results_df = pd.DataFrame({
        'Date': results['dates'],
        'Portfolio_Value': results['portfolio_values']
    })
    results_df.to_csv('tqqq_options_strategy_results.csv', index=False)
    print(f"\n💾 Detailed results saved to 'tqqq_options_strategy_results.csv'")

if __name__ == "__main__":
    main()
