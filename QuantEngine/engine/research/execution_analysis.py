"""
Optimal Execution Algorithms and Transaction Cost Analysis

Implements sophisticated execution analysis including:
- Almgren-Chriss optimal execution framework
- Square-root and linear market impact models
- Post-trade transaction cost analysis (TCA)
- Implementation shortfall decomposition
- VWAP/TWAP benchmark comparison
- Execution strategy evaluation and risk estimation

Uses realistic default parameters calibrated to US equity markets.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from scipy.optimize import minimize_scalar

logger = logging.getLogger(__name__)


class ExecutionAnalyzer:
    """
    Comprehensive execution analysis and optimal trading framework
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # Default US equity market parameters
        self.default_spread_bps = config.get('default_spread_bps', 2.0)
        self.default_commission_per_share = config.get('commission_per_share', 0.005)
        self.annual_trading_days = config.get('annual_trading_days', 252)
        self.trading_minutes_per_day = config.get('trading_minutes_per_day', 390)

        # Default impact model calibration (US large-cap equities)
        self.eta_default = config.get('permanent_impact_coeff', 0.1)
        self.gamma_default = config.get('temporary_impact_coeff', 0.1)

        # Risk-free rate for opportunity cost calculations
        self.risk_free_rate = config.get('risk_free_rate', 0.05)

    # ------------------------------------------------------------------ #
    #  1. Almgren-Chriss Optimal Execution
    # ------------------------------------------------------------------ #

    def optimal_execution_schedule(
        self,
        total_shares: int,
        n_periods: int,
        daily_volume: float,
        volatility: float,
        risk_aversion: float = 1e-6,
        permanent_impact: float = None,
        temporary_impact: float = None,
    ) -> Dict[str, Any]:
        """
        Compute Almgren-Chriss optimal execution trajectory.

        Minimises E[cost] + lambda * Var[cost] over a deterministic
        trading schedule, producing a VWAP-like trajectory that balances
        urgency (risk aversion) against market impact.

        Args:
            total_shares: Total shares to execute
            n_periods: Number of trading intervals
            daily_volume: Average daily volume (shares)
            volatility: Daily return volatility (e.g. 0.02 for 2%)
            risk_aversion: Risk-aversion parameter (lambda)
            permanent_impact: Permanent impact coefficient (eta)
            temporary_impact: Temporary impact coefficient (gamma)

        Returns:
            Dict with trade_schedule, holdings, expected_cost,
            cost_variance, and per-period details
        """
        eta = permanent_impact if permanent_impact is not None else self.eta_default
        gamma = temporary_impact if temporary_impact is not None else self.gamma_default

        tau = 1.0 / n_periods  # fraction of trading day per period
        sigma = volatility * np.sqrt(tau)

        # Almgren-Chriss kappa
        kappa_sq = (risk_aversion * sigma ** 2) / (eta * (1.0 / tau))
        kappa = np.sqrt(max(kappa_sq, 1e-20))

        # Optimal holdings trajectory: x_j = X * sinh(kappa*(N-j)) / sinh(kappa*N)
        holdings = np.zeros(n_periods + 1)
        sinh_kN = np.sinh(kappa * n_periods)
        if abs(sinh_kN) < 1e-30:
            # Fallback to linear (TWAP) when kappa ~ 0
            holdings = np.array([
                total_shares * (1.0 - j / n_periods) for j in range(n_periods + 1)
            ])
        else:
            for j in range(n_periods + 1):
                holdings[j] = total_shares * np.sinh(kappa * (n_periods - j)) / sinh_kN

        # Trade sizes per period (n_j = x_{j-1} - x_j)
        trade_sizes = np.diff(-holdings)  # positive means selling
        trade_sizes = -np.diff(holdings)

        # Expected cost components
        # Permanent impact cost
        permanent_cost = 0.5 * eta * total_shares ** 2

        # Temporary impact cost
        temp_cost = gamma * np.sum(trade_sizes ** 2) / tau

        expected_cost = permanent_cost + temp_cost

        # Cost variance
        cost_variance = (sigma ** 2) * np.sum(holdings[:-1] ** 2)

        # Build per-period detail
        period_details = []
        for j in range(n_periods):
            period_details.append({
                'period': j + 1,
                'holdings_start': float(holdings[j]),
                'holdings_end': float(holdings[j + 1]),
                'trade_size': float(trade_sizes[j]),
                'participation_rate': float(trade_sizes[j] / (daily_volume / n_periods))
                if daily_volume > 0 else 0.0,
            })

        logger.info(
            f"Optimal schedule: {total_shares} shares over {n_periods} periods, "
            f"E[cost]={expected_cost:.2f}, Var[cost]={cost_variance:.2f}"
        )

        return {
            'trade_schedule': trade_sizes.tolist(),
            'holdings': holdings.tolist(),
            'expected_cost': float(expected_cost),
            'cost_variance': float(cost_variance),
            'cost_std': float(np.sqrt(max(cost_variance, 0.0))),
            'risk_aversion': risk_aversion,
            'n_periods': n_periods,
            'period_details': period_details,
        }

    def calculate_impact_params(self, ohlcv: pd.DataFrame) -> Dict[str, Any]:
        """
        Estimate permanent and temporary market impact parameters
        from historical OHLCV data.

        Uses Kyle's lambda for permanent impact and the Amihud
        illiquidity ratio for temporary impact calibration.

        Args:
            ohlcv: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']

        Returns:
            Dict with estimated eta (permanent), gamma (temporary),
            and supporting statistics
        """
        df = ohlcv.copy()

        if len(df) < 20:
            logger.warning("Insufficient data for impact estimation, using defaults")
            return {
                'permanent_impact': self.eta_default,
                'temporary_impact': self.gamma_default,
                'method': 'default',
            }

        # Daily returns and dollar volume
        df['returns'] = df['close'].pct_change()
        df['dollar_volume'] = df['close'] * df['volume']
        df = df.dropna()

        # Amihud illiquidity ratio: |r_t| / DollarVolume_t
        amihud_values = np.abs(df['returns'].values) / df['dollar_volume'].values
        amihud_ratio = float(np.mean(amihud_values))

        # Kyle's lambda estimate via regression |r_t| = lambda * sqrt(V_t) + eps
        abs_returns = np.abs(df['returns'].values)
        sqrt_volume = np.sqrt(df['volume'].values)
        if np.std(sqrt_volume) > 0:
            cov_rv = np.cov(abs_returns, sqrt_volume)[0, 1]
            var_v = np.var(sqrt_volume)
            kyle_lambda = cov_rv / var_v if var_v > 0 else self.eta_default
        else:
            kyle_lambda = self.eta_default

        # Map to Almgren-Chriss parameters
        avg_price = float(df['close'].mean())
        avg_volume = float(df['volume'].mean())
        avg_volatility = float(df['returns'].std())

        # Permanent impact: eta calibrated from Kyle's lambda
        eta = abs(kyle_lambda) * avg_price

        # Temporary impact: gamma from Amihud ratio scaled
        gamma = amihud_ratio * avg_price * avg_volume * 0.5

        # Floor at small positive values
        eta = max(eta, 1e-8)
        gamma = max(gamma, 1e-8)

        logger.info(
            f"Impact params estimated: eta={eta:.6f}, gamma={gamma:.6f}, "
            f"amihud={amihud_ratio:.2e}, kyle_lambda={kyle_lambda:.6f}"
        )

        return {
            'permanent_impact': float(eta),
            'temporary_impact': float(gamma),
            'amihud_ratio': float(amihud_ratio),
            'kyle_lambda': float(kyle_lambda),
            'avg_daily_volume': float(avg_volume),
            'avg_volatility': float(avg_volatility),
            'avg_price': float(avg_price),
            'method': 'estimated',
        }

    def execution_frontier(
        self,
        total_shares: int,
        daily_volume: float,
        volatility: float,
        n_points: int = 20,
    ) -> pd.DataFrame:
        """
        Compute the efficient frontier of expected cost vs cost variance
        for different risk-aversion levels.

        Args:
            total_shares: Total shares to execute
            daily_volume: Average daily volume
            volatility: Daily return volatility
            n_points: Number of frontier points

        Returns:
            DataFrame with columns: risk_aversion, expected_cost,
            cost_variance, cost_std, n_periods
        """
        # Sweep risk aversion from very patient to very urgent
        lambdas = np.logspace(-10, -2, n_points)
        n_periods = max(10, int(total_shares / (daily_volume * 0.05)))
        n_periods = min(n_periods, 100)

        frontier_rows = []
        for lam in lambdas:
            result = self.optimal_execution_schedule(
                total_shares=total_shares,
                n_periods=n_periods,
                daily_volume=daily_volume,
                volatility=volatility,
                risk_aversion=lam,
            )
            frontier_rows.append({
                'risk_aversion': lam,
                'expected_cost': result['expected_cost'],
                'cost_variance': result['cost_variance'],
                'cost_std': result['cost_std'],
                'n_periods': n_periods,
            })

        frontier_df = pd.DataFrame(frontier_rows)

        logger.info(
            f"Execution frontier computed: {n_points} points, "
            f"cost range [{frontier_df['expected_cost'].min():.2f}, "
            f"{frontier_df['expected_cost'].max():.2f}]"
        )

        return frontier_df

    # ------------------------------------------------------------------ #
    #  2. Market Impact Models
    # ------------------------------------------------------------------ #

    def calculate_temporary_impact(
        self,
        trade_size: float,
        daily_volume: float,
        volatility: float,
    ) -> float:
        """
        Square-root temporary impact model.

        impact = sigma * sqrt(Q / V)

        Args:
            trade_size: Number of shares traded
            daily_volume: Average daily volume
            volatility: Daily return volatility

        Returns:
            Estimated temporary price impact as a fraction
        """
        if daily_volume <= 0:
            logger.warning("Daily volume <= 0, returning 0 impact")
            return 0.0

        participation = abs(trade_size) / daily_volume
        impact = volatility * np.sqrt(participation)

        return float(impact)

    def calculate_permanent_impact(
        self,
        trade_size: float,
        daily_volume: float,
    ) -> float:
        """
        Linear permanent impact estimate.

        impact = eta * (Q / V)

        Args:
            trade_size: Number of shares traded
            daily_volume: Average daily volume

        Returns:
            Estimated permanent price impact as a fraction
        """
        if daily_volume <= 0:
            logger.warning("Daily volume <= 0, returning 0 impact")
            return 0.0

        participation = abs(trade_size) / daily_volume
        impact = self.eta_default * participation

        return float(impact)

    def estimate_slippage(
        self,
        order_size: float,
        daily_volume: float,
        spread: float,
        volatility: float,
    ) -> Dict[str, Any]:
        """
        Comprehensive slippage estimate combining spread cost,
        market impact, and timing risk.

        Args:
            order_size: Number of shares
            daily_volume: Average daily volume
            spread: Bid-ask spread as a fraction (e.g. 0.0002 for 2 bps)
            volatility: Daily return volatility

        Returns:
            Dict with spread_cost, temporary_impact, permanent_impact,
            timing_risk, total_slippage (all as fractions)
        """
        # Half-spread cost (crossing the spread)
        spread_cost = spread / 2.0

        # Market impact
        temp_impact = self.calculate_temporary_impact(order_size, daily_volume, volatility)
        perm_impact = self.calculate_permanent_impact(order_size, daily_volume)

        # Timing risk: volatility over expected execution duration
        participation_rate = abs(order_size) / daily_volume if daily_volume > 0 else 1.0
        execution_days = max(participation_rate / 0.10, 1.0 / self.annual_trading_days)
        timing_risk = volatility * np.sqrt(execution_days)

        total_slippage = spread_cost + temp_impact + perm_impact + timing_risk

        logger.info(
            f"Slippage estimate: spread={spread_cost:.4f}, "
            f"temp_impact={temp_impact:.4f}, perm_impact={perm_impact:.4f}, "
            f"timing_risk={timing_risk:.4f}, total={total_slippage:.4f}"
        )

        return {
            'spread_cost': float(spread_cost),
            'temporary_impact': float(temp_impact),
            'permanent_impact': float(perm_impact),
            'timing_risk': float(timing_risk),
            'total_slippage': float(total_slippage),
            'participation_rate': float(participation_rate),
            'estimated_execution_days': float(execution_days),
        }

    # ------------------------------------------------------------------ #
    #  3. Transaction Cost Analysis (TCA)
    # ------------------------------------------------------------------ #

    def post_trade_tca(
        self,
        trades: pd.DataFrame,
        market_data: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Full post-trade transaction cost analysis.

        Args:
            trades: DataFrame with columns ['timestamp', 'price', 'size',
                     'side', 'decision_price']
            market_data: DataFrame with columns ['timestamp', 'open', 'high',
                         'low', 'close', 'volume', 'vwap']

        Returns:
            Dict with implementation_shortfall, vwap_slippage,
            market_impact_cost, timing_cost, opportunity_cost,
            and summary statistics
        """
        if trades.empty or market_data.empty:
            logger.warning("Empty trades or market data for TCA")
            return {'error': 'insufficient_data'}

        # Execution prices and sizes
        exec_prices = trades['price'].tolist()
        exec_sizes = trades['size'].tolist()
        sides = trades['side'].tolist() if 'side' in trades.columns else ['buy'] * len(trades)

        # Derive sign-adjusted sizes (positive for buys)
        signed_sizes = [
            s if side.lower() == 'buy' else -s
            for s, side in zip(exec_sizes, sides)
        ]

        total_shares = sum(abs(s) for s in exec_sizes)
        total_value = sum(p * abs(s) for p, s in zip(exec_prices, exec_sizes))
        avg_exec_price = total_value / total_shares if total_shares > 0 else 0.0

        # Benchmarks
        decision_price = float(trades['decision_price'].iloc[0]) if 'decision_price' in trades.columns else float(market_data['open'].iloc[0])
        benchmark_close = float(market_data['close'].iloc[-1])
        market_vwap = float(market_data['vwap'].mean()) if 'vwap' in market_data.columns else avg_exec_price

        # Implementation shortfall
        is_result = self.calculate_implementation_shortfall(
            decision_price=decision_price,
            execution_prices=exec_prices,
            execution_sizes=[abs(s) for s in exec_sizes],
            benchmark_price=benchmark_close,
        )

        # VWAP slippage
        vwap_slip = self.calculate_vwap_slippage(
            execution_prices=exec_prices,
            execution_sizes=[abs(s) for s in exec_sizes],
            market_vwap=market_vwap,
        )

        # Market impact estimate
        avg_daily_volume = float(market_data['volume'].mean()) if 'volume' in market_data.columns else 1e6
        avg_volatility = float(market_data['close'].pct_change().std()) if len(market_data) > 1 else 0.02
        market_impact = self.calculate_temporary_impact(total_shares, avg_daily_volume, avg_volatility)

        # Timing cost: price drift from decision to execution
        timing_cost = (avg_exec_price - decision_price) / decision_price

        # Opportunity cost: unexecuted portion (assume fully filled here)
        opportunity_cost = 0.0

        logger.info(
            f"Post-trade TCA: IS={is_result['total_shortfall_bps']:.1f}bps, "
            f"VWAP slip={vwap_slip * 10000:.1f}bps, "
            f"impact={market_impact * 10000:.1f}bps"
        )

        return {
            'implementation_shortfall': is_result,
            'vwap_slippage': float(vwap_slip),
            'vwap_slippage_bps': float(vwap_slip * 10000),
            'market_impact_cost': float(market_impact),
            'market_impact_bps': float(market_impact * 10000),
            'timing_cost': float(timing_cost),
            'timing_cost_bps': float(timing_cost * 10000),
            'opportunity_cost': float(opportunity_cost),
            'avg_execution_price': float(avg_exec_price),
            'decision_price': float(decision_price),
            'market_vwap': float(market_vwap),
            'total_shares': int(total_shares),
            'total_value': float(total_value),
            'num_fills': len(trades),
        }

    def calculate_implementation_shortfall(
        self,
        decision_price: float,
        execution_prices: List[float],
        execution_sizes: List[int],
        benchmark_price: float,
    ) -> Dict[str, Any]:
        """
        Implementation shortfall decomposition.

        IS = delay_cost + trading_impact + opportunity_cost

        Args:
            decision_price: Price at time of trading decision
            execution_prices: List of fill prices
            execution_sizes: List of fill sizes (positive)
            benchmark_price: End-of-period benchmark price

        Returns:
            Dict with total_shortfall, delay_cost, trading_impact,
            opportunity_cost, all in absolute and basis-point terms
        """
        total_shares_ordered = sum(execution_sizes)
        total_shares_filled = sum(execution_sizes)

        if total_shares_ordered == 0 or decision_price == 0:
            return {
                'total_shortfall': 0.0,
                'total_shortfall_bps': 0.0,
                'delay_cost': 0.0,
                'trading_impact': 0.0,
                'opportunity_cost': 0.0,
            }

        # Weighted average execution price
        total_cost = sum(p * s for p, s in zip(execution_prices, execution_sizes))
        avg_exec = total_cost / total_shares_filled

        # First fill price as arrival price
        arrival_price = execution_prices[0]

        # Delay cost: slippage between decision and first execution
        delay_cost = (arrival_price - decision_price) / decision_price

        # Trading impact: slippage between arrival and average execution
        trading_impact = (avg_exec - arrival_price) / decision_price

        # Opportunity cost: gain/loss on unfilled portion
        unfilled = total_shares_ordered - total_shares_filled
        if unfilled > 0 and total_shares_ordered > 0:
            opportunity_cost = (
                (benchmark_price - decision_price) / decision_price
                * (unfilled / total_shares_ordered)
            )
        else:
            opportunity_cost = 0.0

        total_shortfall = delay_cost + trading_impact + opportunity_cost

        return {
            'total_shortfall': float(total_shortfall),
            'total_shortfall_bps': float(total_shortfall * 10000),
            'delay_cost': float(delay_cost),
            'delay_cost_bps': float(delay_cost * 10000),
            'trading_impact': float(trading_impact),
            'trading_impact_bps': float(trading_impact * 10000),
            'opportunity_cost': float(opportunity_cost),
            'opportunity_cost_bps': float(opportunity_cost * 10000),
            'decision_price': float(decision_price),
            'arrival_price': float(arrival_price),
            'avg_execution_price': float(avg_exec),
            'benchmark_price': float(benchmark_price),
        }

    def calculate_vwap_slippage(
        self,
        execution_prices: List[float],
        execution_sizes: List[int],
        market_vwap: float,
    ) -> float:
        """
        Calculate execution price vs market VWAP slippage.

        Positive value means execution was worse than VWAP (for buys).

        Args:
            execution_prices: List of fill prices
            execution_sizes: List of fill sizes (positive)
            market_vwap: Market VWAP over the execution window

        Returns:
            Slippage as a fraction (e.g. 0.0005 = 5 bps)
        """
        total_value = sum(p * s for p, s in zip(execution_prices, execution_sizes))
        total_shares = sum(execution_sizes)

        if total_shares == 0 or market_vwap == 0:
            return 0.0

        avg_exec_price = total_value / total_shares
        slippage = (avg_exec_price - market_vwap) / market_vwap

        return float(slippage)

    def cost_attribution(
        self,
        trades: pd.DataFrame,
        market_data: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Break down total execution costs into components:
        spread, impact, timing, and opportunity.

        Args:
            trades: DataFrame with columns ['timestamp', 'price', 'size', 'side']
            market_data: DataFrame with OHLCV + 'vwap' columns

        Returns:
            Dict with cost breakdown in absolute and basis-point terms
        """
        if trades.empty or market_data.empty:
            logger.warning("Empty data for cost attribution")
            return {'error': 'insufficient_data'}

        exec_prices = trades['price'].values
        exec_sizes = trades['size'].values
        total_shares = float(np.sum(np.abs(exec_sizes)))
        total_value = float(np.sum(exec_prices * np.abs(exec_sizes)))

        if total_value == 0:
            return {'error': 'zero_value'}

        avg_exec_price = total_value / total_shares
        avg_daily_volume = float(market_data['volume'].mean()) if 'volume' in market_data.columns else 1e6

        # Spread cost
        if 'bid' in market_data.columns and 'ask' in market_data.columns:
            avg_spread = float((market_data['ask'] - market_data['bid']).mean())
            spread_frac = avg_spread / avg_exec_price
        else:
            spread_frac = self.default_spread_bps / 10000.0

        spread_cost = spread_frac / 2.0  # half-spread

        # Impact cost
        volatility = float(market_data['close'].pct_change().std()) if len(market_data) > 1 else 0.02
        impact_cost = self.calculate_temporary_impact(total_shares, avg_daily_volume, volatility)

        # Timing cost
        decision_price = float(market_data['open'].iloc[0])
        timing_cost = (avg_exec_price - decision_price) / decision_price if decision_price > 0 else 0.0

        # Opportunity cost (residual)
        market_vwap = float(market_data['vwap'].mean()) if 'vwap' in market_data.columns else avg_exec_price
        total_slippage = (avg_exec_price - market_vwap) / market_vwap if market_vwap > 0 else 0.0
        opportunity_cost = max(0.0, total_slippage - spread_cost - impact_cost - timing_cost)

        total_cost = spread_cost + impact_cost + abs(timing_cost) + opportunity_cost

        logger.info(
            f"Cost attribution: spread={spread_cost * 10000:.1f}bps, "
            f"impact={impact_cost * 10000:.1f}bps, "
            f"timing={timing_cost * 10000:.1f}bps, "
            f"opportunity={opportunity_cost * 10000:.1f}bps"
        )

        return {
            'spread_cost': float(spread_cost),
            'spread_cost_bps': float(spread_cost * 10000),
            'impact_cost': float(impact_cost),
            'impact_cost_bps': float(impact_cost * 10000),
            'timing_cost': float(timing_cost),
            'timing_cost_bps': float(timing_cost * 10000),
            'opportunity_cost': float(opportunity_cost),
            'opportunity_cost_bps': float(opportunity_cost * 10000),
            'total_cost': float(total_cost),
            'total_cost_bps': float(total_cost * 10000),
            'avg_execution_price': float(avg_exec_price),
            'total_shares': float(total_shares),
            'total_value': float(total_value),
        }

    # ------------------------------------------------------------------ #
    #  4. Execution Strategy Evaluation
    # ------------------------------------------------------------------ #

    def compare_execution_strategies(
        self,
        trades: pd.DataFrame,
        market_data: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Compare actual execution against VWAP, TWAP, and Almgren-Chriss
        benchmarks.

        Args:
            trades: DataFrame with ['timestamp', 'price', 'size', 'side']
            market_data: DataFrame with OHLCV + optional 'vwap'

        Returns:
            Dict with actual vs benchmark performance for each strategy
        """
        if trades.empty or market_data.empty:
            logger.warning("Empty data for strategy comparison")
            return {'error': 'insufficient_data'}

        exec_prices = trades['price'].values
        exec_sizes = np.abs(trades['size'].values)
        total_shares = int(np.sum(exec_sizes))
        total_value = float(np.sum(exec_prices * exec_sizes))
        avg_exec_price = total_value / total_shares if total_shares > 0 else 0.0

        n_periods = len(trades)
        avg_daily_volume = float(market_data['volume'].mean()) if 'volume' in market_data.columns else 1e6
        volatility = float(market_data['close'].pct_change().std()) if len(market_data) > 1 else 0.02

        # --- VWAP benchmark ---
        if 'vwap' in market_data.columns:
            market_vwap = float(market_data['vwap'].mean())
        else:
            # Approximate VWAP from OHLCV
            typical = (market_data['high'] + market_data['low'] + market_data['close']) / 3.0
            if 'volume' in market_data.columns:
                market_vwap = float(np.average(typical, weights=market_data['volume']))
            else:
                market_vwap = float(typical.mean())

        vwap_slippage = (avg_exec_price - market_vwap) / market_vwap if market_vwap > 0 else 0.0

        # --- TWAP benchmark ---
        if len(market_data) > 0:
            twap_price = float(market_data['close'].mean())
        else:
            twap_price = avg_exec_price
        twap_slippage = (avg_exec_price - twap_price) / twap_price if twap_price > 0 else 0.0

        # --- Almgren-Chriss benchmark ---
        ac_result = self.optimal_execution_schedule(
            total_shares=total_shares,
            n_periods=max(n_periods, 2),
            daily_volume=avg_daily_volume,
            volatility=volatility,
        )
        ac_expected_cost_frac = ac_result['expected_cost'] / total_value if total_value > 0 else 0.0

        # Arrival price benchmark
        arrival_price = float(market_data['open'].iloc[0])
        arrival_slippage = (avg_exec_price - arrival_price) / arrival_price if arrival_price > 0 else 0.0

        logger.info(
            f"Strategy comparison: VWAP={vwap_slippage * 10000:.1f}bps, "
            f"TWAP={twap_slippage * 10000:.1f}bps, "
            f"arrival={arrival_slippage * 10000:.1f}bps"
        )

        return {
            'actual': {
                'avg_price': float(avg_exec_price),
                'total_value': float(total_value),
                'total_shares': int(total_shares),
            },
            'vwap_benchmark': {
                'benchmark_price': float(market_vwap),
                'slippage': float(vwap_slippage),
                'slippage_bps': float(vwap_slippage * 10000),
            },
            'twap_benchmark': {
                'benchmark_price': float(twap_price),
                'slippage': float(twap_slippage),
                'slippage_bps': float(twap_slippage * 10000),
            },
            'arrival_benchmark': {
                'benchmark_price': float(arrival_price),
                'slippage': float(arrival_slippage),
                'slippage_bps': float(arrival_slippage * 10000),
            },
            'almgren_chriss_benchmark': {
                'expected_cost_frac': float(ac_expected_cost_frac),
                'expected_cost_bps': float(ac_expected_cost_frac * 10000),
                'cost_std': float(ac_result['cost_std']),
            },
        }

    def optimal_trade_schedule_vwap(
        self,
        target_shares: int,
        historical_volume_profile: pd.Series,
    ) -> pd.Series:
        """
        Compute a VWAP participation schedule from an intraday
        volume profile.

        Args:
            target_shares: Total shares to execute
            historical_volume_profile: Series indexed by time-of-day
                with average intraday volume at each interval

        Returns:
            Series of shares to trade in each interval
        """
        if historical_volume_profile.empty or historical_volume_profile.sum() <= 0:
            logger.warning("Empty or zero volume profile, returning uniform schedule")
            n = max(len(historical_volume_profile), 1)
            return pd.Series(
                [target_shares / n] * n,
                index=historical_volume_profile.index if len(historical_volume_profile) > 0 else range(n),
            )

        # Normalise volume profile to participation weights
        volume_weights = historical_volume_profile / historical_volume_profile.sum()

        # Allocate shares proportionally
        schedule = (volume_weights * target_shares).round().astype(int)

        # Adjust rounding error
        diff = target_shares - schedule.sum()
        if diff != 0:
            max_idx = schedule.idxmax()
            schedule.loc[max_idx] += diff

        logger.info(
            f"VWAP schedule: {target_shares} shares across "
            f"{len(schedule)} intervals, max participation "
            f"{schedule.max()} shares"
        )

        return schedule

    def estimate_execution_risk(
        self,
        order_size: float,
        daily_volume: float,
        holding_period: int,
        volatility: float,
    ) -> Dict[str, Any]:
        """
        Estimate total expected execution cost and risk for a planned
        trade, combining impact, spread, and price risk.

        Args:
            order_size: Number of shares to trade
            daily_volume: Average daily volume
            holding_period: Expected execution duration (trading days)
            volatility: Daily return volatility

        Returns:
            Dict with expected_cost, cost_std, participation_rate,
            and risk breakdown
        """
        participation_rate = abs(order_size) / daily_volume if daily_volume > 0 else 1.0

        # Expected costs
        spread_cost = self.default_spread_bps / 10000.0 / 2.0
        temp_impact = self.calculate_temporary_impact(order_size, daily_volume, volatility)
        perm_impact = self.calculate_permanent_impact(order_size, daily_volume)
        commission = self.default_commission_per_share * abs(order_size)

        expected_cost_frac = spread_cost + temp_impact + perm_impact
        expected_cost_total = expected_cost_frac + commission

        # Price risk during execution
        execution_vol = volatility * np.sqrt(holding_period)
        price_risk_1std = execution_vol * abs(order_size)

        # Value at Risk (95%) for execution
        var_95 = expected_cost_frac + 1.645 * execution_vol

        logger.info(
            f"Execution risk: E[cost]={expected_cost_frac * 10000:.1f}bps, "
            f"vol={execution_vol * 10000:.1f}bps, "
            f"VaR95={var_95 * 10000:.1f}bps, "
            f"participation={participation_rate:.2%}"
        )

        return {
            'expected_cost_frac': float(expected_cost_frac),
            'expected_cost_bps': float(expected_cost_frac * 10000),
            'spread_cost_bps': float(spread_cost * 10000),
            'temporary_impact_bps': float(temp_impact * 10000),
            'permanent_impact_bps': float(perm_impact * 10000),
            'commission_total': float(commission),
            'execution_volatility': float(execution_vol),
            'execution_volatility_bps': float(execution_vol * 10000),
            'var_95_frac': float(var_95),
            'var_95_bps': float(var_95 * 10000),
            'participation_rate': float(participation_rate),
            'holding_period_days': int(holding_period),
            'order_size': float(order_size),
        }
