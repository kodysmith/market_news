#!/usr/bin/env python3
"""
Aggressive Growth Strategy Backtest

Strategy for scaling small accounts fast with controlled risk:
- Core position: Long TQQQ for leveraged exposure
- Income Layer: Bull put spreads on QQQ/SPY (high prob, defined risk)
- Momentum Layer: Bull call spreads on tech names (NVDA, TSLA, META)
- Moonshot Layer: OTM options around catalysts
- Weekly trading cadence
- Target >60% win rate with small losses
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

# Numba for fast BS pricing
from numba import njit
import numba as nb

# Fast normal CDF approximation (Abramowitz & Stegun)
@njit(fastmath=True)
def erf_approx(x):
    # Abramowitz & Stegun approximation for erf
    a1 =  0.254829592
    a2 = -0.284496736
    a3 =  1.421413741
    a4 = -1.453152027
    a5 =  1.061405429
    p =  0.3275911
    y = abs(x)
    t = 1.0 / (1.0 + p * y)
    z = t * (a1 + t * (a2 + t * (a3 + t * (a4 + t * a5))))
    return np.sign(x) * (1.0 - z * np.exp(-y * y))

@njit(fastmath=True)
def norm_cdf(x):
    return 0.5 * (1.0 + erf_approx(x))

# Abramowitz-Stegun style inverse CDF approximation
@njit(fastmath=True)
def norm_ppf(p):
    p = max(1e-9, min(1 - 1e-9, p))
    a1, a2, a3 = -39.6968302866538, 220.946098424521, -275.928510446969
    a4, a5, a6 = 138.357751867269, -30.6647980661472, 2.50662827745924
    b1, b2, b3 = -54.4760987982241, 161.585836858041, -155.698979859887
    b4, b5 = 66.8013118877197, -13.2806815528857
    c1, c2, c3 = -0.00778489400243029, -0.322396458041136, -2.40075827716184
    c4, c5, c6 = -2.54973253934373, 4.37466414146497, 2.93816398269878
    d1, d2, d3, d4 = 0.00778469570904146, 0.32246712907004, 2.445134137143, 3.75440866190742
    plow = 0.02425
    phigh = 1 - plow
    tail = p < plow
    q = np.sqrt(-2 * np.log(np.where(tail, p, 1 - p)))
    # Tail approximation
    num = ((c1 * q + c2) * q + c3) * q + c4
    num = (num * q + c5) * q + c6
    den = (((d1 * q + d2) * q + d3) * q + d4)
    z_tail = num / den
    # Central approximation
    q = p - 0.5
    r = q * q
    num = (((a1 * r + a2) * r + a3) * r + a4) * r + a5
    num = num * r + a6
    den = ((((b1 * r + b2) * r + b3) * r + b4) * r + b5)
    z_cen = q * (num / den)
    return np.where(tail, z_tail, z_cen)

# Closed-form strike from target delta (no Brentq)
@njit(fastmath=True)
def strike_for_put_delta(S, sigma, dte, target_put_delta, r=0.02):
    T = max(dte / 365.0, 1 / 365)
    d1 = norm_ppf(target_put_delta + 1.0)
    return S * np.exp(-(d1 * sigma * np.sqrt(T)) + (r + 0.5 * sigma**2) * T)

@njit(fastmath=True)
def strike_for_call_delta(S, sigma, dte, target_call_delta, r=0.02):
    T = max(dte / 365.0, 1 / 365)
    d1 = norm_ppf(target_call_delta)
    return S * np.exp(-(d1 * sigma * np.sqrt(T)) + (r + 0.5 * sigma**2) * T)

# Numba BS pricing kernels
@njit(fastmath=True)
def bs_put_price(S, K, sigma, T, r=0.02):
    if T <= 0 or sigma <= 0:
        return 0.01
    d1 = (np.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put = K * np.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)
    return max(put, 0.01)

@njit(fastmath=True)
def bs_call_price(S, K, sigma, T, r=0.02):
    if T <= 0 or sigma <= 0:
        return 0.01
    d1 = (np.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call = S * norm_cdf(d1) - K * np.exp(-r * T) * norm_cdf(d2)
    return max(call, 0.01)

class AggressiveGrowthStrategy:
    def __init__(self,
                 initial_capital: float = 100000,  # Smaller account for aggressive growth
                 core_allocation: float = 0.4,      # 40% in core TQQQ position
                 income_allocation: float = 0.3,    # 30% in income strategies (bull put spreads)
                 momentum_allocation: float = 0.2,  # 20% in momentum plays
                 moonshot_allocation: float = 0.1,  # 10% in high-risk moonshots
                 trade_frequency_days: int = 7,     # Weekly trading
                 max_loss_per_trade: float = 0.05,  # 5% max loss per trade
                 target_win_rate: float = 0.6,      # Target 60% win rate
                 start_date: str = '2020-01-01',
                 end_date: str = '2024-01-01'):
        """
        Initialize the aggressive growth strategy

        Args:
            initial_capital: Starting capital (smaller accounts)
            core_allocation: % allocated to core TQQQ position
            income_allocation: % allocated to income strategies (bull put spreads)
            momentum_allocation: % allocated to momentum plays
            moonshot_allocation: % allocated to high-risk moonshots
            trade_frequency_days: Days between trades
            max_loss_per_trade: Maximum loss allowed per trade (5%)
            target_win_rate: Target win rate for trade selection
            start_date: Backtest start date
            end_date: Backtest end date
        """
        self.initial_capital = initial_capital
        self.core_allocation = core_allocation
        self.income_allocation = income_allocation
        self.momentum_allocation = momentum_allocation
        self.moonshot_allocation = moonshot_allocation
        self.trade_frequency_days = trade_frequency_days
        self.max_loss_per_trade = max_loss_per_trade
        self.target_win_rate = target_win_rate
        self.start_date = start_date
        self.end_date = end_date
        
        # Strategy state - Multi-asset, multi-layer approach
        self.cash = initial_capital
        self.shares_owned = 0  # For backward compatibility
        self.total_invested = 0  # For backward compatibility
        self.portfolio_values = []
        self.dates = []
        self.trades = []

        # Core positions (TQQQ)
        self.core_positions = []  # Long TQQQ shares

        # Income layer positions (Bull put spreads on QQQ/SPY)
        self.income_positions = []  # Bull put spreads for steady income

        # Momentum layer positions (Bull call spreads on tech stocks)
        self.momentum_positions = []  # Bull call spreads on NVDA, TSLA, META

        # Moonshot layer positions (OTM options around catalysts)
        self.moonshot_positions = []  # High-risk, high-reward options

        # Asset data caches
        self.asset_data_cache = {}  # Cache for different tickers

        # Trading parameters
        self.last_trade_date = None
        self.win_count = 0
        self.total_trades_count = 0

        # Strategy allocation amounts
        self.core_capital = initial_capital * core_allocation
        self.income_capital = initial_capital * income_allocation
        self.momentum_capital = initial_capital * momentum_allocation
        self.moonshot_capital = initial_capital * moonshot_allocation

        # Initialize data caches
        self.idx = None
        self.S = None
        self.iv_raw = None
        self.regime_on = None
        self.asset_dates = None
        self.asset_ma200 = None

        # Config used by QQQ loader
        self.ivr_window = 252  # ~1y trading days

        # Download data
        self.asset_data = self._download_asset_data()
        self.qqq_data = self._download_qqq_data()

        # Monthly 10-month SMA trend toggle (end-of-month)
        self._compute_monthly_trend_flags()

        # Precompute numpy arrays for fast access (no Pandas lookups in loop)
        self.idx = self.asset_data.index.to_numpy()
        self.S = self.asset_data["Close"].to_numpy(dtype=np.float64)
        self.iv_raw = self.asset_data["20d_Vol"].to_numpy(dtype=np.float64) if '20d_Vol' in self.asset_data.columns else np.full(len(self.idx), 0.5, dtype=np.float64)
        self.regime_on = (self.asset_data["Regime"].to_numpy() == "RISK_ON")
        self.asset_dates = self.asset_data.index.to_numpy()
        self.qqq_iv = self.qqq_data["IV_proxy"].to_numpy(dtype=np.float64) if 'IV_proxy' in self.qqq_data.columns else np.full(len(self.idx), 0.18, dtype=np.float64)
        self.qqq_ivr = self.qqq_data["IVR"].to_numpy(dtype=np.float64) if 'IVR' in self.qqq_data.columns else np.full(len(self.idx), 50.0, dtype=np.float64)
        self.asset_ma200 = self.asset_data["MA200"].to_numpy(dtype=np.float64) if 'MA200' in self.asset_data.columns else self.S

        # Date to index map for fast lookups
        self.i_of = {d: i for i, d in enumerate(self.idx)}

    def _get_asset_data(self, ticker: str) -> pd.DataFrame:
        """Get cached asset data or download it"""
        if ticker not in self.asset_data_cache:
            print(f"📊 Downloading {ticker} data...")
            data = yf.download(ticker, start=self.start_date, end=self.end_date, progress=False)
            if data.empty:
                raise ValueError(f"No data found for {ticker}")
            # Add volatility column
            data['Returns'] = data['Close'].pct_change()
            data['20d_Vol'] = data['Returns'].rolling(20).std() * np.sqrt(252)
            self.asset_data_cache[ticker] = data
        return self.asset_data_cache[ticker]

    def _download_asset_data(self) -> pd.DataFrame:
        """Download TQQQ historical data"""
        print(f"📊 Downloading TQQQ data...")
        ticker = yf.Ticker('TQQQ')
        data = ticker.history(start=self.start_date, end=self.end_date)

        if data.empty:
            raise ValueError("No TQQQ data found for the specified date range")

        # Calculate daily returns and volatility
        data['Returns'] = data['Close'].pct_change()
        data['20d_Vol'] = data['Returns'].rolling(window=20).std() * np.sqrt(252)
        data['20d_Vol'] = data['20d_Vol'].fillna(method='bfill')

        # Trend regime features
        data['MA200'] = data['Close'].rolling(200).mean()
        data['Regime'] = np.where(data['Close'] > data['MA200'], 'RISK_ON', 'RISK_OFF')

        # Precompute numpy arrays for speed
        self.idx = data.index.to_numpy()
        self.S = data["Close"].to_numpy(dtype=np.float64)
        self.iv_raw = data["20d_Vol"].to_numpy(dtype=np.float64)
        self.regime_on = (data["Regime"].to_numpy() == "RISK_ON")
        self.asset_dates = data.index.to_numpy()
        self.asset_ma200 = data["MA200"].to_numpy(dtype=np.float64)

        # Date to index map for fast lookups
        self.i_of = {d: i for i, d in enumerate(self.idx)}

        return data

    def _download_qqq_data(self) -> pd.DataFrame:
        """Download QQQ data for IV reference and spreads"""
        print(f"📊 Downloading QQQ data...")
        ticker = yf.Ticker('QQQ')
        data = ticker.history(start=self.start_date, end=self.end_date)

        if data.empty:
            raise ValueError("No QQQ data found for the specified date range")

        # Calculate IV rank and proxy
        data['Returns'] = data['Close'].pct_change()
        data['20d_Vol'] = data['Returns'].rolling(window=20).std() * np.sqrt(252)
        data['20d_Vol'] = data['20d_Vol'].fillna(method='bfill')

        # IV rank (simplified)
        data['IV_proxy'] = data['20d_Vol'] * 1.2
        data['IV_proxy'] = np.clip(data['IV_proxy'], 0.15, 0.35)

        # IVR calculation
        vol_mean = data['20d_Vol'].rolling(self.ivr_window).mean()
        vol_std = data['20d_Vol'].rolling(self.ivr_window).std()
        data['IVR'] = ((data['20d_Vol'] - vol_mean) / vol_std).clip(-3, 3)
        data['IVR'] = (data['IVR'] + 3) / 6  # Normalize to 0-1

        return data

    def _buy_core_position(self, date: datetime) -> bool:
        """Buy TQQQ core position if not already owned"""
        if self.core_positions:
            return False  # Already have core position

        tqqq_data = self._get_asset_data('TQQQ')
        if date not in tqqq_data.index:
            return False

        current_price = float(tqqq_data.loc[date, 'Close'])
        shares_to_buy = int(self.core_capital // current_price)

        if shares_to_buy > 0:
            cost = shares_to_buy * current_price
            self.cash -= cost

            position = {
                'date_opened': date,
                'shares': shares_to_buy,
                'avg_price': current_price,
                'current_value': cost
            }
            self.core_positions.append(position)

            self.trades.append({
                'date': date,
                'action': 'BUY_CORE_TQQQ',
                'quantity': shares_to_buy,
                'price': current_price,
                'cost': cost
            })
            print(f"🏗️  Bought {shares_to_buy} TQQQ shares at ${current_price:.2f} for core position")
            return True
        return False

    def _trade_income_layer(self, date: datetime) -> bool:
        """Trade bull put spreads on QQQ/SPY for steady income"""
        # Use QQQ for more stability than SPY
        qqq_data = self._get_asset_data('QQQ')
        if date not in qqq_data.index:
            return False

        current_price = float(qqq_data.loc[date, 'Close'])
        vol = float(qqq_data.loc[date, '20d_Vol'])

        # Bull put spread: Sell ~15Δ put, Buy ~5Δ put (30-45 DTE)
        # This has high probability of profit in bull markets
        dte = 35  # 5 weeks

        # Find strikes for bull put spread
        # Bull put spread: Sell higher strike put (more OTM), Buy lower strike put (less OTM)
        # More negative delta = more OTM = higher strike for puts
        short_delta = -0.15  # Sell more OTM put (higher strike)
        long_delta = -0.05   # Buy less OTM put (lower strike)

        short_strike = self._strike_for_put_delta(current_price, vol, dte, short_delta)
        long_strike = self._strike_for_put_delta(current_price, vol, dte, long_delta)

        # Ensure correct order for bull put spread
        if short_strike < long_strike:
            short_strike, long_strike = long_strike, short_strike

        # Calculate premiums
        short_premium = self._estimate_put_premium(current_price, short_strike, vol, dte)
        long_premium = self._estimate_put_premium(current_price, long_strike, vol, dte)
        net_premium = short_premium - long_premium

        if net_premium <= 0:
            return False

        # Risk management: Max loss is the spread width minus net premium received
        # For bull put spread: max_loss = (short_strike - long_strike) - net_premium
        max_loss = (short_strike - long_strike) - net_premium
        risk_amount = max_loss * 100  # Per contract

        if risk_amount > self.income_capital * self.max_loss_per_trade:
            return False  # Too much risk

        # Calculate win probability (both puts expire worthless)
        prob_win = self._calculate_spread_win_prob(current_price, short_strike, long_strike, vol, dte)
        if prob_win < self.target_win_rate:
            return False

        # Trade the spread
        contracts = min(5, int((self.income_capital * 0.1) // risk_amount))  # Small position sizing

        if contracts <= 0:
            return False

        total_premium = net_premium * 100 * contracts
        self.cash += total_premium

        position = {
            'date_opened': date,
            'type': 'bull_put_spread',
            'underlying': 'QQQ',
            'short_strike': short_strike,
            'long_strike': long_strike,
            'contracts': contracts,
            'dte': dte,
            'net_premium': net_premium,
            'max_loss': risk_amount,
            'prob_win': prob_win
        }
        self.income_positions.append(position)

        self.trades.append({
            'date': date,
            'action': 'SELL_BULL_PUT_SPREAD_QQQ',
            'short_strike': short_strike,
            'long_strike': long_strike,
            'contracts': contracts,
            'premium': total_premium,
            'prob_win': prob_win
        })
        print(f"💰 Sold {contracts} QQQ bull put spreads (${short_strike:.1f}/$:{long_strike:.1f}) for ${total_premium:.0f} (Win prob: {prob_win:.1%})")
        return True

    def _trade_momentum_layer(self, date: datetime) -> bool:
        """Trade bull call spreads on momentum tech stocks"""
        momentum_stocks = ['NVDA', 'TSLA', 'META']
        selected_stock = np.random.choice(momentum_stocks)  # Rotate between stocks

        stock_data = self._get_asset_data(selected_stock)
        if date not in stock_data.index:
            return False

        current_price = float(stock_data.loc[date, 'Close'])
        vol = float(stock_data.loc[date, '20d_Vol'])

        # Bull call spread: Buy ~10Δ call, Sell ~25Δ call (21-28 DTE)
        dte = 21

        long_delta = 0.10   # Buy 10Δ call (slightly OTM)
        short_delta = 0.25  # Sell 25Δ call (further OTM)

        long_strike = self._strike_for_call_delta(current_price, vol, dte, long_delta)
        short_strike = self._strike_for_call_delta(current_price, vol, dte, short_delta)

        if long_strike >= short_strike:
            return False

        # Calculate net debit
        long_premium = self._estimate_call_premium(current_price, long_strike, vol, dte)
        short_premium = self._estimate_call_premium(current_price, short_strike, vol, dte)
        net_debit = long_premium - short_premium

        if net_debit >= 0:
            return False  # Should be a debit spread

        max_loss = abs(net_debit) * 100
        if max_loss > self.momentum_capital * self.max_loss_per_trade:
            return False

        contracts = min(3, int((self.momentum_capital * 0.15) // max_loss))

        if contracts <= 0:
            return False

        total_debit = abs(net_debit) * 100 * contracts
        self.cash -= total_debit

        position = {
            'date_opened': date,
            'type': 'bull_call_spread',
            'underlying': selected_stock,
            'long_strike': long_strike,
            'short_strike': short_strike,
            'contracts': contracts,
            'dte': dte,
            'net_debit': net_debit,
            'max_loss': max_loss
        }
        self.momentum_positions.append(position)

        self.trades.append({
            'date': date,
            'action': f'BUY_BULL_CALL_SPREAD_{selected_stock}',
            'long_strike': long_strike,
            'short_strike': short_strike,
            'contracts': contracts,
            'debit': total_debit
        })
        print(f"📈 Bought {contracts} {selected_stock} bull call spreads (${long_strike:.1f}/$:{short_strike:.1f}) for ${total_debit:.0f}")
        return True

    def _calculate_spread_win_prob(self, S: float, short_strike: float, long_strike: float, vol: float, dte: int) -> float:
        """Calculate win probability for bull put spread"""
        # Simplified: both puts expire worthless (price stays above short_strike)
        T = dte / 365.0
        d1 = (np.log(S / short_strike) + (0.02 + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
        prob_both_expire = norm_cdf(d1)  # Probability price > short_strike
        return prob_both_expire

    def _manage_positions(self, date: datetime) -> None:
        """Manage existing positions - close profitable ones, roll if needed"""
        # Close income positions (bull put spreads) at 50-70% profit target
        self._manage_income_positions(date)

        # Close momentum positions (bull call spreads) when profitable or stop loss
        self._manage_momentum_positions(date)

        # Manage moonshot positions (if implemented)
        # self._manage_moonshot_positions(date)

    def _manage_income_positions(self, date: datetime) -> None:
        """Manage bull put spread positions"""
        positions_to_close = []

        for j, pos in enumerate(self.income_positions):
            days_held = (date - pos['date_opened']).days
            current_dte = max(pos['dte'] - days_held, 0)

            if current_dte <= 0:
                # Position expired - calculate final P&L
                qqq_data = self._get_asset_data('QQQ')
                if date in qqq_data.index:
                    final_price = qqq_data.loc[date, 'Close']

                    # Bull put spread P&L: Keep full premium if price > short_strike
                    # Lose partial amount if price between strikes
                    if final_price > pos['short_strike']:
                        pnl = pos['net_premium'] * 100 * pos['contracts']  # Full win
                        self.win_count += 1
                    elif final_price > pos['long_strike']:
                        loss = (pos['short_strike'] - final_price) * 100 * pos['contracts']
                        pnl = pos['net_premium'] * 100 * pos['contracts'] - loss
                    else:
                        pnl = -pos['max_loss'] * pos['contracts']  # Max loss

                    self.cash += pnl

                    self.trades.append({
                        'date': date,
                        'action': 'CLOSE_BULL_PUT_SPREAD',
                        'underlying': pos['underlying'],
                        'pnl': pnl
                    })

                    if pnl > 0:
                        self.win_count += 1

                    print(f"💰 Closed QQQ bull put spread for ${pnl:.0f} P&L")
                positions_to_close.append(j)

            elif days_held >= 7:  # Check after 1 week
                # Close early if 50%+ profit target reached
                current_value = self._calculate_income_position_value(pos, date)
                initial_cost = -pos['net_premium'] * 100 * pos['contracts']  # Credit received
                current_pnl = current_value - initial_cost

                if current_pnl >= abs(initial_cost) * 0.5:  # 50% profit target
                    self.cash += current_pnl
                    self.trades.append({
                        'date': date,
                        'action': 'CLOSE_BULL_PUT_SPREAD_EARLY',
                        'underlying': pos['underlying'],
                        'pnl': current_pnl
                    })
                    self.win_count += 1
                    print(f"💰 Closed QQQ bull put spread early for ${current_pnl:.0f} P&L (50% target)")
                    positions_to_close.append(j)

        # Remove closed positions
        for j in reversed(positions_to_close):
            self.income_positions.pop(j)

    def _manage_momentum_positions(self, date: datetime) -> None:
        """Manage bull call spread positions"""
        positions_to_close = []

        for j, pos in enumerate(self.momentum_positions):
            days_held = (date - pos['date_opened']).days
            current_dte = max(pos['dte'] - days_held, 0)

            if current_dte <= 0:
                # Position expired - calculate final P&L
                stock_data = self._get_asset_data(pos['underlying'])
                if date in stock_data.index:
                    final_price = stock_data.loc[date, 'Close']

                    # Bull call spread P&L calculation
                    intrinsic_value = max(0, final_price - pos['long_strike']) - max(0, final_price - pos['short_strike'])
                    pnl = (intrinsic_value - abs(pos['net_debit'])) * 100 * pos['contracts']

                    self.cash += pnl

                    self.trades.append({
                        'date': date,
                        'action': f'CLOSE_BULL_CALL_SPREAD_{pos["underlying"]}',
                        'pnl': pnl
                    })

                    if pnl > 0:
                        self.win_count += 1

                    print(f"📈 Closed {pos['underlying']} bull call spread for ${pnl:.0f} P&L")
                positions_to_close.append(j)

            else:
                # Check stop loss or profit target
                current_value = self._calculate_momentum_position_value(pos, date)
                initial_debit = abs(pos['net_debit']) * 100 * pos['contracts']
                current_pnl = current_value + initial_debit  # Value is positive for profit

                # Stop loss at 100% of debit paid
                if current_pnl <= -initial_debit:
                    self.cash += current_value
                    self.trades.append({
                        'date': date,
                        'action': f'STOP_LOSS_BULL_CALL_SPREAD_{pos["underlying"]}',
                        'pnl': current_value
                    })
                    print(f"📈 Stop loss on {pos['underlying']} bull call spread for ${current_value:.0f} P&L")
                    positions_to_close.append(j)

                # Take profit at 200% return
                elif current_pnl >= initial_debit * 2:
                    self.cash += current_value
                    self.trades.append({
                        'date': date,
                        'action': f'TAKE_PROFIT_BULL_CALL_SPREAD_{pos["underlying"]}',
                        'pnl': current_value
                    })
                    self.win_count += 1
                    print(f"📈 Take profit on {pos['underlying']} bull call spread for ${current_value:.0f} P&L")
                    positions_to_close.append(j)

        # Remove closed positions
        for j in reversed(positions_to_close):
            self.momentum_positions.pop(j)

    def _calculate_income_position_value(self, pos: Dict, date: datetime) -> float:
        """Calculate current value of bull put spread"""
        qqq_data = self._get_asset_data('QQQ')
        if date not in qqq_data.index:
            return 0

        current_price = float(qqq_data.loc[date, 'Close'])
        days_held = (date - pos['date_opened']).days
        current_dte = max(pos['dte'] - days_held, 0)

        if current_dte <= 0:
            return 0  # Will be handled by expiration logic

        vol = float(qqq_data.loc[date, '20d_Vol'])

        # Recalculate current spread value
        short_premium = self._estimate_put_premium(current_price, pos['short_strike'], vol, current_dte)
        long_premium = self._estimate_put_premium(current_price, pos['long_strike'], vol, current_dte)
        current_value = (short_premium - long_premium) * 100 * pos['contracts']

        return current_value

    def _calculate_momentum_position_value(self, pos: Dict, date: datetime) -> float:
        """Calculate current value of bull call spread"""
        stock_data = self._get_asset_data(pos['underlying'])
        if date not in stock_data.index:
            return 0

        current_price = float(stock_data.loc[date, 'Close'])
        days_held = (date - pos['date_opened']).days
        current_dte = max(pos['dte'] - days_held, 0)

        if current_dte <= 0:
            return 0  # Will be handled by expiration logic

        vol = float(stock_data.loc[date, '20d_Vol'])

        # Recalculate current spread value
        long_premium = self._estimate_call_premium(current_price, pos['long_strike'], vol, current_dte)
        short_premium = self._estimate_call_premium(current_price, pos['short_strike'], vol, current_dte)
        current_value = (long_premium - short_premium) * 100 * pos['contracts']

        return current_value

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
        trend_daily = trend_on.reindex(self.asset_data.index, method='ffill')
        self.asset_data['TREND_ON'] = trend_daily.astype(bool)
        # Buy-the-dip: QQQ below 20DMA by >1.5 std of 20d returns
        q['20dma'] = q['Close'].rolling(20).mean()
        q['ret20sd'] = q['Returns'].rolling(20).std()
        dip = (q['Close'] < (q['20dma'] - 1.5 * q['ret20sd'] * q['Close'].shift(1))).reindex(self.asset_data.index, method='ffill')
        self.asset_data['BTFD'] = dip.fillna(False)
    
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
        """Fast BS put pricing with Numba."""
        return bs_put_price(current_price, strike, volatility, dte / 365.0)

    def _estimate_call_premium(self, current_price: float, strike: float, volatility: float, dte: int) -> float:
        """Fast BS call pricing with Numba."""
        return bs_call_price(current_price, strike, volatility, dte / 365.0)

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
        i = self.i_of[date]
        ref_value = self.portfolio_values[i-1] if i > 0 else self.initial_capital
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
        """Closed-form strike from target put delta (no Brentq)."""
        return strike_for_put_delta(S, sigma, dte, target_put_delta, r)

    def _strike_for_call_delta(self, S: float, sigma: float, dte: int, target_call_delta: float = 0.15, r: float = 0.02) -> float:
        """Closed-form strike from target call delta (no Brentq)."""
        return strike_for_call_delta(S, sigma, dte, target_call_delta, r)

    def _proxy_iv(self, date) -> float:
        """Proxy IV from realized vol with bounds for TQQQ; adds a simple skew bump later if needed."""
        base = float(self.asset_data.loc[date, '20d_Vol']) if '20d_Vol' in self.asset_data.columns else 0.5
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
        print(f"📈 Bought {self.share_purchase_amount} {self.ticker} shares at ${price:.2f} on {date.strftime('%Y-%m-%d')}")
        
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
        i = self.i_of[date]
        ref_value = self.portfolio_values[i-1] if i > 0 else self.initial_capital
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
                regime = self.asset_data.loc[date, 'Regime'] if 'Regime' in self.asset_data.columns else 'RISK_ON'
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
    
    def _calculate_portfolio_value(self, date: datetime) -> float:
        """Calculate total portfolio value for aggressive growth strategy"""
        portfolio_value = self.cash

        # Core TQQQ positions
        for pos in self.core_positions:
            tqqq_data = self._get_asset_data('TQQQ')
            if date in tqqq_data.index:
                current_price = tqqq_data.loc[date, 'Close']
                pos['current_value'] = pos['shares'] * current_price
                portfolio_value += pos['current_value']

        # Income positions (bull put spreads)
        for pos in self.income_positions:
            value = self._calculate_income_position_value(pos, date)
            portfolio_value += value

        # Momentum positions (bull call spreads)
        for pos in self.momentum_positions:
            value = self._calculate_momentum_position_value(pos, date)
            portfolio_value += value

        # Moonshot positions (if implemented)
        # portfolio_value += sum(self._calculate_moonshot_position_value(pos, date) for pos in self.moonshot_positions)

        return portfolio_value
    
    def run_backtest(self) -> Dict:
        """Run the aggressive growth strategy backtest"""
        print("🚀 Starting Aggressive Growth Strategy Backtest")
        print("=" * 60)

        # Initialize with core position - find first valid trading date
        tqqq_data = self._get_asset_data('TQQQ')
        first_valid_date = tqqq_data.index[0]  # First date with data
        self._buy_core_position(first_valid_date)

        # Pre-allocate arrays
        N = len(self.idx)
        self.portfolio_values = np.full(N, self.initial_capital, dtype=np.float64)
        self.dates = self.idx.copy()

        for i in range(N):
            date = self.idx[i]

            # Weekly trading decisions
            should_trade = (self.last_trade_date is None or
                          (date - self.last_trade_date).days >= self.trade_frequency_days)

            if should_trade:
                # Trade across different layers
                trades_made = 0

                # Income layer: Bull put spreads (most consistent)
                if self._trade_income_layer(date):
                    trades_made += 1

                # Momentum layer: Bull call spreads (higher risk/reward)
                if np.random.random() < 0.7:  # 70% chance to trade momentum
                    if self._trade_momentum_layer(date):
                        trades_made += 1

                # Moonshot layer: OTM options (only occasionally)
                if np.random.random() < 0.2:  # 20% chance for moonshots
                    # TODO: Implement moonshot trading
                    pass

                if trades_made > 0:
                    self.last_trade_date = date
                    self.total_trades_count += trades_made

            # Manage existing positions (roll/close when profitable)
            self._manage_positions(date)

            # Calculate portfolio value
            portfolio_value = self._calculate_portfolio_value(date)
            self.portfolio_values[i] = portfolio_value
        
        # Calculate final results
        final_value = float(self.portfolio_values[-1])
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        # Calculate benchmark (buy and hold asset)
        initial_price = self.asset_data['Close'].iloc[0]
        final_price = self.asset_data['Close'].iloc[-1]
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
            'income_positions': len(self.income_positions),
            'momentum_positions': len(self.momentum_positions),
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
        
        # Benchmark (buy and hold asset)
        initial_price = self.asset_data['Close'].iloc[0]
        benchmark_values = [initial_price * (price / initial_price) * (self.initial_capital / initial_price)
                           for price in self.asset_data['Close']]
        ax1.plot(self.asset_data.index, benchmark_values, label=f'Buy & Hold {self.ticker}', linewidth=2, alpha=0.7)
        
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
        
        # Asset price with trading events
        ax2.plot(self.asset_data.index, self.asset_data['Close'], label=f'{self.ticker} Price', linewidth=2, alpha=0.7)
        
        # Add trading event markers on price chart
        if not trades_df.empty:
            # Plot markers for different actions on price chart
            if any(buy_puts_mask):
                buy_puts_prices = [self.asset_data.loc[d, 'Close'] for d in buy_puts_dates if d in self.asset_data.index]
                buy_puts_dates_valid = [d for d in buy_puts_dates if d in self.asset_data.index]
                ax2.scatter(buy_puts_dates_valid, buy_puts_prices, color='green', marker='^',
                           s=60, label='Buy Puts', alpha=0.8, zorder=5)

            if any(roll_puts_mask):
                roll_puts_prices = [self.asset_data.loc[d, 'Close'] for d in roll_puts_dates if d in self.asset_data.index]
                roll_puts_dates_valid = [d for d in roll_puts_dates if d in self.asset_data.index]
                ax2.scatter(roll_puts_dates_valid, roll_puts_prices, color='orange', marker='s',
                           s=60, label='Roll Puts', alpha=0.8, zorder=5)

            if any(buy_shares_mask):
                buy_shares_prices = [self.asset_data.loc[d, 'Close'] for d in buy_shares_dates if d in self.asset_data.index]
                buy_shares_dates_valid = [d for d in buy_shares_dates if d in self.asset_data.index]
                ax2.scatter(buy_shares_dates_valid, buy_shares_prices, color='blue', marker='o',
                           s=60, label='Buy Shares', alpha=0.8, zorder=5)

            if any(close_puts_mask):
                close_puts_prices = [self.asset_data.loc[d, 'Close'] for d in close_puts_dates if d in self.asset_data.index]
                close_puts_dates_valid = [d for d in close_puts_dates if d in self.asset_data.index]
                ax2.scatter(close_puts_dates_valid, close_puts_prices, color='red', marker='x',
                           s=60, label='Close Puts', alpha=0.8, zorder=5)
            if any(sell_calls_mask):
                sell_call_prices = [self.asset_data.loc[d, 'Close'] for d in sell_call_dates if d in self.asset_data.index]
                sell_call_dates_valid = [d for d in sell_call_dates if d in self.asset_data.index]
                ax2.scatter(sell_call_dates_valid, sell_call_prices, color='purple', marker='v',
                           s=60, label='Sell Calls', alpha=0.8, zorder=5)
            if any(roll_calls_mask):
                roll_call_prices = [self.asset_data.loc[d, 'Close'] for d in roll_call_dates if d in self.asset_data.index]
                roll_call_dates_valid = [d for d in roll_call_dates if d in self.asset_data.index]
                ax2.scatter(roll_call_dates_valid, roll_call_prices, color='magenta', marker='P',
                           s=70, label='Roll Calls', alpha=0.9, zorder=6)
        
        ax2.set_title(f'{self.ticker} Price with Trading Events')
        ax2.set_ylabel(f'{self.ticker} Price ($)')
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
        print(f"📊 AGGRESSIVE GROWTH STRATEGY RESULTS")
        print("="*60)
        print(f"Initial Capital: ${results['initial_capital']:,.0f}")
        print(f"Final Value: ${results['final_value']:,.0f}")
        print(f"Total Return: {results['total_return']:.1%}")
        print(f"Benchmark Return: {results['benchmark_return']:.1%}")
        print(f"Excess Return: {results['excess_return']:.1%}")
        print(f"Cash Remaining: ${results['cash_remaining']:,.0f}")
        print(f"Total Trades: {results['total_trades']}")
        win_rate = self.win_count / max(1, self.total_trades_count)
        print(f"Win Rate: {win_rate:.1%}")
        print(f"Active Income Positions: {len(self.income_positions)}")
        print(f"Active Momentum Positions: {len(self.momentum_positions)}")
        print(f"Core Position Value: ${sum(p.get('current_value', 0) for p in self.core_positions):,.0f}")
        
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
            
            # Calculate premium collected from income trades
            premium_collected = sum(t.get('premium', 0) for t in self.trades if 'PUT_SPREAD' in t.get('action', ''))
            momentum_pnl = sum(t.get('pnl', 0) for t in self.trades if 'CALL_SPREAD' in t.get('action', ''))

            print(f"\nRisk Metrics:")
            print(f"Annualized Volatility: {volatility:.1%}")
            print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
            print(f"Maximum Drawdown: {max_drawdown:.1%}")
            print(f"Premium Collected (Income): ${premium_collected:,.0f}")
            print(f"Momentum P&L: ${momentum_pnl:,.0f}")
            print(f"Total Income Generated: ${premium_collected + momentum_pnl:,.0f}")

def main():
    """Run the aggressive growth strategy backtest"""
    print("🚀 Aggressive Growth Strategy Backtest")
    print("=" * 60)

    # Strategy parameters - Aggressive Growth Setup
    strategy = AggressiveGrowthStrategy(
        initial_capital=100000,  # Smaller account for scaling
        core_allocation=0.4,      # 40% in core TQQQ position
        income_allocation=0.3,    # 30% in income strategies
        momentum_allocation=0.2,  # 20% in momentum plays
        moonshot_allocation=0.1,  # 10% in high-risk moonshots
        trade_frequency_days=7,   # Weekly trading
        max_loss_per_trade=0.05,  # 5% max loss per trade
        target_win_rate=0.6       # Target 60% win rate
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
    filename = 'aggressive_growth_strategy_results.csv'
    results_df.to_csv(filename, index=False)
    print(f"\n💾 Detailed results saved to '{filename}'")

if __name__ == "__main__":
    main()
