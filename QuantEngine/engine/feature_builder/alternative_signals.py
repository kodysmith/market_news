"""
Alternative Data Signals for AI Quant Trading System

Implements:
- Insider trading signal analysis (Form 4 parsing, sentiment, timing)
- Short interest analysis and squeeze detection
- Dark pool and off-exchange activity signals
- ETF flow analysis and sector rotation detection
- Unusual options activity detection and scoring
- Cross-asset signals (bond-equity, FX-equity, commodity-equity)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta
from scipy import stats

logger = logging.getLogger(__name__)


class AlternativeSignalGenerator:
    """Generate trading signals from alternative data sources"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # Insider trading config
        self.insider_lookback_days = config.get('insider_lookback_days', 90)
        self.insider_cluster_window = config.get('insider_cluster_window', 7)
        self.insider_cluster_min_count = config.get('insider_cluster_min_count', 3)

        # Title-based weighting for insider trades
        self.insider_title_weights = config.get('insider_title_weights', {
            'CEO': 2.0, 'CFO': 1.8, 'COO': 1.6, 'CTO': 1.5,
            'President': 1.7, 'VP': 1.3, 'Director': 1.0,
            'Officer': 1.4, 'Chairman': 1.9,
        })

        # Short interest config
        self.short_squeeze_si_threshold = config.get('short_squeeze_si_threshold', 20.0)
        self.short_squeeze_dtc_threshold = config.get('short_squeeze_dtc_threshold', 5.0)

        # Dark pool config
        self.dark_pool_unusual_threshold = config.get('dark_pool_unusual_threshold', 2.0)
        self.block_trade_min_shares = config.get('block_trade_min_shares', 10000)

        # Options config
        self.options_unusual_volume_mult = config.get('options_unusual_volume_mult', 3.0)
        self.options_block_premium_min = config.get('options_block_premium_min', 100000)

        # Cross-asset config
        self.correlation_window = config.get('correlation_window', 60)
        self.momentum_windows = config.get('momentum_windows', [5, 10, 21, 63])

    # ---------------------------------------------------------------
    # 1. Insider Trading Signals
    # ---------------------------------------------------------------

    def parse_form4_signals(self, insider_data: pd.DataFrame) -> Dict:
        """
        Analyze insider buy/sell patterns from Form 4 filings

        Args:
            insider_data: DataFrame with columns: ticker, insider_name,
                transaction_type, shares, price, date, insider_title

        Returns:
            Dictionary of insider trading signals and summary statistics
        """

        if insider_data.empty:
            logger.warning("Empty insider data provided")
            return {'signals': {}, 'summary': {}}

        results = {}
        insider_data = insider_data.copy()

        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(insider_data['date']):
            insider_data['date'] = pd.to_datetime(insider_data['date'])

        # Calculate trade value
        insider_data['trade_value'] = insider_data['shares'] * insider_data['price']

        # Classify transactions
        buy_mask = insider_data['transaction_type'].str.lower().isin(
            ['purchase', 'buy', 'p', 'a']
        )
        sell_mask = insider_data['transaction_type'].str.lower().isin(
            ['sale', 'sell', 's', 'd']
        )

        buys = insider_data[buy_mask]
        sells = insider_data[sell_mask]

        # Per-ticker aggregation
        tickers = insider_data['ticker'].unique()
        ticker_signals = {}

        for ticker in tickers:
            ticker_buys = buys[buys['ticker'] == ticker]
            ticker_sells = sells[sells['ticker'] == ticker]

            total_buy_value = ticker_buys['trade_value'].sum()
            total_sell_value = ticker_sells['trade_value'].sum()
            total_value = total_buy_value + total_sell_value

            buy_ratio = total_buy_value / total_value if total_value > 0 else 0.5

            ticker_signals[ticker] = {
                'buy_count': len(ticker_buys),
                'sell_count': len(ticker_sells),
                'total_buy_value': float(total_buy_value),
                'total_sell_value': float(total_sell_value),
                'buy_ratio': float(buy_ratio),
                'net_signal': 1 if buy_ratio > 0.6 else (-1 if buy_ratio < 0.4 else 0),
                'unique_insiders_buying': ticker_buys['insider_name'].nunique(),
                'unique_insiders_selling': ticker_sells['insider_name'].nunique(),
            }

        results['ticker_signals'] = ticker_signals

        # Overall summary
        results['summary'] = {
            'total_transactions': len(insider_data),
            'total_buys': len(buys),
            'total_sells': len(sells),
            'tickers_with_net_buying': sum(
                1 for s in ticker_signals.values() if s['net_signal'] == 1
            ),
            'tickers_with_net_selling': sum(
                1 for s in ticker_signals.values() if s['net_signal'] == -1
            ),
        }

        logger.info(
            f"Parsed Form 4 signals for {len(tickers)} tickers: "
            f"{len(buys)} buys, {len(sells)} sells"
        )

        return results

    def insider_sentiment_score(self, insider_data: pd.DataFrame, ticker: str,
                                lookback_days: int = 90) -> Dict:
        """
        Calculate insider sentiment score for a specific ticker

        Incorporates net buying ratio, cluster buying detection,
        and officer vs director weighting.

        Args:
            insider_data: DataFrame with insider transaction data
            ticker: Ticker symbol to analyze
            lookback_days: Number of days to look back

        Returns:
            Dictionary with sentiment score and component breakdowns
        """

        if insider_data.empty:
            return {'sentiment_score': 0.0, 'components': {}}

        insider_data = insider_data.copy()
        if not pd.api.types.is_datetime64_any_dtype(insider_data['date']):
            insider_data['date'] = pd.to_datetime(insider_data['date'])

        # Filter to ticker and lookback window
        cutoff_date = insider_data['date'].max() - timedelta(days=lookback_days)
        mask = (insider_data['ticker'] == ticker) & (insider_data['date'] >= cutoff_date)
        filtered = insider_data[mask].copy()

        if filtered.empty:
            return {'sentiment_score': 0.0, 'components': {}, 'ticker': ticker}

        # Classify transactions
        buy_mask = filtered['transaction_type'].str.lower().isin(
            ['purchase', 'buy', 'p', 'a']
        )
        sell_mask = filtered['transaction_type'].str.lower().isin(
            ['sale', 'sell', 's', 'd']
        )

        filtered['trade_value'] = filtered['shares'] * filtered['price']

        # --- Component 1: Net buying ratio (weighted by trade value) ---
        buy_value = filtered.loc[buy_mask, 'trade_value'].sum()
        sell_value = filtered.loc[sell_mask, 'trade_value'].sum()
        total_value = buy_value + sell_value

        net_buying_ratio = (buy_value - sell_value) / total_value if total_value > 0 else 0.0

        # --- Component 2: Cluster buying detection ---
        buy_dates = filtered.loc[buy_mask, 'date'].sort_values()
        cluster_score = 0.0

        if len(buy_dates) >= self.insider_cluster_min_count:
            # Check for clusters of buys within the cluster window
            for i in range(len(buy_dates) - self.insider_cluster_min_count + 1):
                window_end = buy_dates.iloc[i] + timedelta(days=self.insider_cluster_window)
                cluster_count = ((buy_dates >= buy_dates.iloc[i]) &
                                 (buy_dates <= window_end)).sum()
                if cluster_count >= self.insider_cluster_min_count:
                    cluster_score = min(cluster_count / self.insider_cluster_min_count, 2.0)
                    break

        # --- Component 3: Officer vs director weighting ---
        weighted_signal = 0.0
        total_weight = 0.0

        for _, row in filtered.iterrows():
            title = str(row.get('insider_title', ''))
            weight = 1.0
            for title_key, title_weight in self.insider_title_weights.items():
                if title_key.lower() in title.lower():
                    weight = title_weight
                    break

            direction = 1.0 if str(row['transaction_type']).lower() in ['purchase', 'buy', 'p', 'a'] else -1.0
            weighted_signal += direction * row['trade_value'] * weight
            total_weight += row['trade_value'] * weight

        officer_weighted_score = weighted_signal / total_weight if total_weight > 0 else 0.0

        # --- Composite sentiment score ---
        sentiment_score = (
            0.4 * net_buying_ratio +
            0.3 * np.clip(cluster_score - 1.0, -1.0, 1.0) +
            0.3 * officer_weighted_score
        )
        sentiment_score = float(np.clip(sentiment_score, -1.0, 1.0))

        return {
            'sentiment_score': sentiment_score,
            'ticker': ticker,
            'lookback_days': lookback_days,
            'transaction_count': len(filtered),
            'components': {
                'net_buying_ratio': float(net_buying_ratio),
                'cluster_score': float(cluster_score),
                'officer_weighted_score': float(officer_weighted_score),
            },
        }

    def insider_timing_analysis(self, insider_data: pd.DataFrame,
                                price_data: pd.DataFrame) -> Dict:
        """
        Measure how well insiders time their trades

        Calculates forward returns after insider buy/sell transactions
        to assess insider timing ability.

        Args:
            insider_data: DataFrame with insider transaction data
            price_data: DataFrame with columns including 'close' and 'ticker',
                indexed by date

        Returns:
            Dictionary with timing analysis results per ticker
        """

        if insider_data.empty or price_data.empty:
            return {'timing_results': {}}

        insider_data = insider_data.copy()
        if not pd.api.types.is_datetime64_any_dtype(insider_data['date']):
            insider_data['date'] = pd.to_datetime(insider_data['date'])

        horizons = [5, 10, 21, 63]  # 1 week, 2 weeks, 1 month, 3 months
        timing_results = {}

        tickers = insider_data['ticker'].unique()

        for ticker in tickers:
            ticker_insider = insider_data[insider_data['ticker'] == ticker]

            # Get price series for the ticker
            if 'ticker' in price_data.columns:
                ticker_prices = price_data[price_data['ticker'] == ticker]['close']
            elif isinstance(price_data, pd.Series):
                ticker_prices = price_data
            else:
                ticker_prices = price_data.get('close', pd.Series(dtype=float))

            if ticker_prices.empty:
                continue

            buy_returns = {h: [] for h in horizons}
            sell_returns = {h: [] for h in horizons}

            for _, row in ticker_insider.iterrows():
                trade_date = row['date']
                is_buy = str(row['transaction_type']).lower() in ['purchase', 'buy', 'p', 'a']

                # Find price on trade date and forward dates
                future_prices = ticker_prices[ticker_prices.index >= trade_date]
                if len(future_prices) < 2:
                    continue

                base_price = future_prices.iloc[0]

                for h in horizons:
                    if len(future_prices) > h:
                        fwd_return = (future_prices.iloc[h] - base_price) / base_price
                        if is_buy:
                            buy_returns[h].append(fwd_return)
                        else:
                            sell_returns[h].append(fwd_return)

            # Aggregate results
            ticker_timing = {}
            for h in horizons:
                buy_avg = float(np.mean(buy_returns[h])) if buy_returns[h] else None
                sell_avg = float(np.mean(sell_returns[h])) if sell_returns[h] else None

                ticker_timing[f'{h}d'] = {
                    'avg_return_after_buy': buy_avg,
                    'avg_return_after_sell': sell_avg,
                    'buy_count': len(buy_returns[h]),
                    'sell_count': len(sell_returns[h]),
                }

                # Timing skill: insiders buy before positive returns, sell before negative
                if buy_avg is not None and sell_avg is not None:
                    ticker_timing[f'{h}d']['timing_skill'] = float(buy_avg - sell_avg)
                else:
                    ticker_timing[f'{h}d']['timing_skill'] = None

            timing_results[ticker] = ticker_timing

        logger.info(f"Insider timing analysis completed for {len(timing_results)} tickers")

        return {'timing_results': timing_results}

    # ---------------------------------------------------------------
    # 2. Short Interest Analysis
    # ---------------------------------------------------------------

    def analyze_short_interest(self, short_data: pd.DataFrame,
                               volume_data: pd.Series) -> Dict:
        """
        Analyze short interest metrics

        Args:
            short_data: DataFrame with columns: date, short_interest,
                shares_outstanding, float_shares
            volume_data: Series of daily trading volume

        Returns:
            Dictionary with short interest analysis
        """

        if short_data.empty or volume_data.empty:
            return {'short_interest_ratio': None, 'days_to_cover': None}

        short_data = short_data.copy()

        latest = short_data.iloc[-1]
        short_interest = latest.get('short_interest', 0)
        shares_outstanding = latest.get('shares_outstanding', 1)
        float_shares = latest.get('float_shares', shares_outstanding)

        # Short interest ratio (% of outstanding)
        si_ratio = (short_interest / shares_outstanding * 100) if shares_outstanding > 0 else 0.0

        # Short interest as % of float
        si_float_pct = (short_interest / float_shares * 100) if float_shares > 0 else 0.0

        # Days to cover (short interest / average daily volume)
        avg_volume = volume_data.tail(20).mean()
        days_to_cover = (short_interest / avg_volume) if avg_volume > 0 else 0.0

        # Short interest trend
        if len(short_data) >= 2:
            si_change = (
                short_data['short_interest'].iloc[-1] - short_data['short_interest'].iloc[-2]
            )
            si_change_pct = (
                si_change / short_data['short_interest'].iloc[-2] * 100
                if short_data['short_interest'].iloc[-2] > 0 else 0.0
            )
        else:
            si_change = 0.0
            si_change_pct = 0.0

        result = {
            'short_interest': int(short_interest),
            'short_interest_ratio': float(si_ratio),
            'short_interest_float_pct': float(si_float_pct),
            'days_to_cover': float(days_to_cover),
            'avg_daily_volume': float(avg_volume),
            'si_change': float(si_change),
            'si_change_pct': float(si_change_pct),
            'signal': 'bearish' if si_float_pct > 10 else ('neutral' if si_float_pct > 5 else 'low'),
        }

        logger.info(
            f"Short interest analysis: SI ratio={si_ratio:.1f}%, "
            f"DTC={days_to_cover:.1f}, Float%={si_float_pct:.1f}%"
        )

        return result

    def detect_short_squeeze_setup(self, short_data: pd.DataFrame,
                                   price_data: pd.DataFrame,
                                   volume_data: pd.Series) -> Dict:
        """
        Score short squeeze potential

        Evaluates: high short interest + rising price + increasing volume + low float

        Args:
            short_data: DataFrame with short interest data
            price_data: DataFrame with price data (close column)
            volume_data: Series of daily trading volume

        Returns:
            Dictionary with squeeze score and component breakdown
        """

        if short_data.empty or price_data.empty or volume_data.empty:
            return {'squeeze_score': 0.0, 'components': {}}

        # Component 1: Short interest level
        latest_si = short_data.iloc[-1]
        float_shares = latest_si.get('float_shares', latest_si.get('shares_outstanding', 1))
        si_pct = (latest_si.get('short_interest', 0) / float_shares * 100) if float_shares > 0 else 0.0
        si_score = np.clip(si_pct / self.short_squeeze_si_threshold, 0.0, 1.0)

        # Component 2: Days to cover
        avg_vol = volume_data.tail(20).mean()
        dtc = (latest_si.get('short_interest', 0) / avg_vol) if avg_vol > 0 else 0.0
        dtc_score = np.clip(dtc / self.short_squeeze_dtc_threshold, 0.0, 1.0)

        # Component 3: Price momentum (rising price)
        close = price_data['close'] if isinstance(price_data, pd.DataFrame) else price_data
        if len(close) >= 10:
            price_return_5d = (close.iloc[-1] - close.iloc[-6]) / close.iloc[-6] if close.iloc[-6] > 0 else 0.0
            price_momentum_score = np.clip(price_return_5d * 10, 0.0, 1.0)
        else:
            price_momentum_score = 0.0

        # Component 4: Volume surge
        if len(volume_data) >= 20:
            recent_vol = volume_data.tail(5).mean()
            historical_vol = volume_data.tail(20).mean()
            vol_ratio = recent_vol / historical_vol if historical_vol > 0 else 1.0
            volume_score = np.clip((vol_ratio - 1.0) / 2.0, 0.0, 1.0)
        else:
            volume_score = 0.0

        # Composite squeeze score (weighted average)
        squeeze_score = float(
            0.35 * si_score +
            0.25 * dtc_score +
            0.25 * price_momentum_score +
            0.15 * volume_score
        )

        result = {
            'squeeze_score': squeeze_score,
            'squeeze_alert': squeeze_score > 0.7,
            'components': {
                'short_interest_score': float(si_score),
                'days_to_cover_score': float(dtc_score),
                'price_momentum_score': float(price_momentum_score),
                'volume_surge_score': float(volume_score),
            },
            'raw_metrics': {
                'si_pct_of_float': float(si_pct),
                'days_to_cover': float(dtc),
            },
        }

        logger.info(f"Short squeeze score: {squeeze_score:.2f} (alert={squeeze_score > 0.7})")

        return result

    def short_interest_momentum(self, short_data: pd.DataFrame) -> pd.Series:
        """
        Calculate rate of change of short interest

        Increasing shorts is a bearish signal; decreasing shorts is bullish.

        Args:
            short_data: DataFrame with short_interest column, indexed by date

        Returns:
            Series of short interest momentum values
        """

        if short_data.empty or 'short_interest' not in short_data.columns:
            return pd.Series(dtype=float)

        si = short_data['short_interest']

        # Rate of change over available periods
        si_momentum = si.pct_change().fillna(0.0)

        # Smooth with rolling average to reduce noise
        if len(si_momentum) >= 3:
            si_momentum = si_momentum.rolling(window=min(3, len(si_momentum)), min_periods=1).mean()

        si_momentum.name = 'short_interest_momentum'

        return si_momentum

    # ---------------------------------------------------------------
    # 3. Dark Pool & Off-Exchange Analysis
    # ---------------------------------------------------------------

    def analyze_dark_pool_activity(self, trade_data: pd.DataFrame) -> Dict:
        """
        Analyze dark pool and off-exchange trading activity

        Args:
            trade_data: DataFrame with columns: date, volume, dark_pool_volume,
                total_volume, trade_size (optional), venue (optional)

        Returns:
            Dictionary with dark pool activity analysis
        """

        if trade_data.empty:
            return {'dark_pool_ratio': None, 'unusual_activity': False}

        trade_data = trade_data.copy()

        # Dark pool volume ratio
        if 'dark_pool_volume' in trade_data.columns and 'total_volume' in trade_data.columns:
            trade_data['dp_ratio'] = (
                trade_data['dark_pool_volume'] / trade_data['total_volume']
            ).fillna(0.0)
        else:
            return {'dark_pool_ratio': None, 'unusual_activity': False}

        latest_ratio = float(trade_data['dp_ratio'].iloc[-1])
        avg_ratio = float(trade_data['dp_ratio'].mean())
        std_ratio = float(trade_data['dp_ratio'].std()) if len(trade_data) > 1 else 0.0

        # Unusual activity detection (z-score based)
        z_score = (latest_ratio - avg_ratio) / std_ratio if std_ratio > 0 else 0.0
        unusual_activity = abs(z_score) > self.dark_pool_unusual_threshold

        # Block trade identification
        block_trades = []
        if 'trade_size' in trade_data.columns:
            block_mask = trade_data['trade_size'] >= self.block_trade_min_shares
            block_trades_df = trade_data[block_mask]
            block_trade_count = len(block_trades_df)
            block_volume = int(block_trades_df['trade_size'].sum()) if not block_trades_df.empty else 0
        else:
            block_trade_count = 0
            block_volume = 0

        # Trend in dark pool ratio
        if len(trade_data) >= 5:
            recent_avg = trade_data['dp_ratio'].tail(5).mean()
            prior_avg = trade_data['dp_ratio'].iloc[:-5].mean() if len(trade_data) > 5 else avg_ratio
            dp_trend = 'increasing' if recent_avg > prior_avg * 1.05 else (
                'decreasing' if recent_avg < prior_avg * 0.95 else 'stable'
            )
        else:
            dp_trend = 'insufficient_data'

        result = {
            'dark_pool_ratio': latest_ratio,
            'avg_dark_pool_ratio': avg_ratio,
            'dp_ratio_z_score': float(z_score),
            'unusual_activity': unusual_activity,
            'dp_trend': dp_trend,
            'block_trade_count': block_trade_count,
            'block_volume': block_volume,
        }

        logger.info(
            f"Dark pool analysis: ratio={latest_ratio:.3f}, "
            f"unusual={unusual_activity}, trend={dp_trend}"
        )

        return result

    def dark_pool_divergence(self, dark_pool_ratio: pd.Series,
                             price: pd.Series) -> Dict:
        """
        Detect divergence between dark pool activity and price direction

        Divergence may indicate smart money positioning counter to price trend.

        Args:
            dark_pool_ratio: Series of dark pool volume ratios
            price: Series of asset prices

        Returns:
            Dictionary with divergence analysis
        """

        if dark_pool_ratio.empty or price.empty:
            return {'divergence_detected': False, 'signal': 'neutral'}

        # Align series
        common_idx = dark_pool_ratio.index.intersection(price.index)
        if len(common_idx) < 10:
            return {'divergence_detected': False, 'signal': 'neutral',
                    'reason': 'insufficient_data'}

        dp = dark_pool_ratio.loc[common_idx]
        px = price.loc[common_idx]

        # Calculate trends over recent window
        window = min(20, len(common_idx) // 2)
        dp_recent = dp.tail(window)
        px_recent = px.tail(window)

        # Linear trend via polyfit
        x = np.arange(len(dp_recent))
        dp_slope = np.polyfit(x, dp_recent.values, 1)[0] if len(dp_recent) > 1 else 0.0
        px_slope = np.polyfit(x, px_recent.values, 1)[0] if len(px_recent) > 1 else 0.0

        # Normalize slopes
        dp_slope_norm = dp_slope / dp_recent.mean() if dp_recent.mean() != 0 else 0.0
        px_slope_norm = px_slope / px_recent.mean() if px_recent.mean() != 0 else 0.0

        # Correlation between dark pool ratio and price
        correlation = float(dp.corr(px))

        # Divergence: dark pool ratio rising while price falling (or vice versa)
        divergence_detected = (dp_slope_norm > 0.01 and px_slope_norm < -0.01) or \
                              (dp_slope_norm < -0.01 and px_slope_norm > 0.01)

        # Signal interpretation
        if dp_slope_norm > 0.01 and px_slope_norm < -0.01:
            signal = 'bullish_divergence'  # Smart money accumulating on dip
        elif dp_slope_norm < -0.01 and px_slope_norm > 0.01:
            signal = 'bearish_divergence'  # Smart money distributing on rally
        else:
            signal = 'no_divergence'

        result = {
            'divergence_detected': divergence_detected,
            'signal': signal,
            'dp_trend_slope': float(dp_slope_norm),
            'price_trend_slope': float(px_slope_norm),
            'dp_price_correlation': correlation,
            'analysis_window': window,
        }

        if divergence_detected:
            logger.info(f"Dark pool divergence detected: {signal}")

        return result

    # ---------------------------------------------------------------
    # 4. ETF Flow Analysis
    # ---------------------------------------------------------------

    def analyze_etf_flows(self, flow_data: pd.DataFrame) -> Dict:
        """
        Analyze ETF creation/redemption flows

        Args:
            flow_data: DataFrame with columns: date, etf_ticker, flow_amount,
                shares_outstanding, nav, aum (optional), sector (optional)

        Returns:
            Dictionary with ETF flow analysis
        """

        if flow_data.empty:
            return {'flow_summary': {}, 'flow_momentum': {}}

        flow_data = flow_data.copy()

        # Aggregate flows by ETF
        etf_summary = {}
        for etf in flow_data['etf_ticker'].unique():
            etf_flows = flow_data[flow_data['etf_ticker'] == etf]
            total_flow = etf_flows['flow_amount'].sum()
            avg_flow = etf_flows['flow_amount'].mean()
            flow_std = etf_flows['flow_amount'].std() if len(etf_flows) > 1 else 0.0

            # Recent flow momentum
            recent_flow = etf_flows['flow_amount'].tail(5).sum()
            prior_flow = etf_flows['flow_amount'].iloc[:-5].tail(5).sum() if len(etf_flows) > 5 else 0.0

            # Creation vs redemption
            creations = etf_flows[etf_flows['flow_amount'] > 0]['flow_amount'].sum()
            redemptions = abs(etf_flows[etf_flows['flow_amount'] < 0]['flow_amount'].sum())

            etf_summary[etf] = {
                'total_flow': float(total_flow),
                'avg_daily_flow': float(avg_flow),
                'flow_volatility': float(flow_std),
                'recent_flow_5d': float(recent_flow),
                'flow_acceleration': float(recent_flow - prior_flow),
                'creation_total': float(creations),
                'redemption_total': float(redemptions),
                'net_creation_ratio': float(
                    creations / (creations + redemptions)
                    if (creations + redemptions) > 0 else 0.5
                ),
                'signal': 'bullish' if total_flow > 0 else 'bearish',
            }

        # Overall market flow
        total_market_flow = sum(s['total_flow'] for s in etf_summary.values())

        result = {
            'etf_flow_summary': etf_summary,
            'total_market_flow': float(total_market_flow),
            'net_bullish_etfs': sum(1 for s in etf_summary.values() if s['signal'] == 'bullish'),
            'net_bearish_etfs': sum(1 for s in etf_summary.values() if s['signal'] == 'bearish'),
        }

        logger.info(f"ETF flow analysis: {len(etf_summary)} ETFs, net flow={total_market_flow:.0f}")

        return result

    def etf_flow_momentum_signal(self, flow_data: pd.DataFrame,
                                 lookback: int = 20) -> Dict:
        """
        Generate signal based on ETF flow trends

        Persistent inflows are bullish; persistent outflows are bearish.

        Args:
            flow_data: DataFrame with columns: date, etf_ticker, flow_amount
            lookback: Number of periods for momentum calculation

        Returns:
            Dictionary with flow momentum signals per ETF
        """

        if flow_data.empty:
            return {'signals': {}}

        signals = {}

        for etf in flow_data['etf_ticker'].unique():
            etf_flows = flow_data[flow_data['etf_ticker'] == etf].copy()
            etf_flows = etf_flows.sort_values('date')

            flows = etf_flows['flow_amount']

            if len(flows) < lookback:
                continue

            # Cumulative flow over lookback
            recent_flows = flows.tail(lookback)
            cumulative_flow = recent_flows.sum()

            # Flow consistency (% of days with inflows)
            inflow_pct = (recent_flows > 0).sum() / len(recent_flows)

            # Flow trend (slope of cumulative flows)
            cumsum = recent_flows.cumsum()
            x = np.arange(len(cumsum))
            slope = np.polyfit(x, cumsum.values, 1)[0] if len(cumsum) > 1 else 0.0

            # Momentum score: combines magnitude, consistency, and trend
            magnitude_score = np.clip(cumulative_flow / (recent_flows.std() * np.sqrt(lookback) + 1e-10), -2, 2)
            consistency_score = (inflow_pct - 0.5) * 2  # Scale to [-1, 1]

            momentum_score = float(0.5 * magnitude_score + 0.5 * consistency_score)

            signals[etf] = {
                'momentum_score': momentum_score,
                'cumulative_flow': float(cumulative_flow),
                'inflow_consistency': float(inflow_pct),
                'flow_trend_slope': float(slope),
                'signal': 'bullish' if momentum_score > 0.3 else (
                    'bearish' if momentum_score < -0.3 else 'neutral'
                ),
            }

        logger.info(f"ETF flow momentum signals generated for {len(signals)} ETFs")

        return {'signals': signals}

    def sector_rotation_from_flows(self, sector_etf_flows: pd.DataFrame) -> Dict:
        """
        Detect sector rotation patterns from relative ETF flows

        Args:
            sector_etf_flows: DataFrame with columns: date, sector, flow_amount

        Returns:
            Dictionary with sector rotation analysis
        """

        if sector_etf_flows.empty or 'sector' not in sector_etf_flows.columns:
            return {'rotation_signal': {}, 'sector_rankings': []}

        # Aggregate flows by sector
        sector_flows = sector_etf_flows.groupby('sector')['flow_amount'].sum()
        total_abs_flow = sector_flows.abs().sum()

        # Relative flow strength per sector
        sector_strength = {}
        for sector in sector_flows.index:
            flow = sector_flows[sector]
            relative_flow = flow / total_abs_flow if total_abs_flow > 0 else 0.0
            sector_strength[sector] = float(relative_flow)

        # Rank sectors by flow
        sector_ranking = sorted(sector_strength.items(), key=lambda x: x[1], reverse=True)

        # Recent vs prior period comparison for rotation detection
        flow_data = sector_etf_flows.copy()
        if not pd.api.types.is_datetime64_any_dtype(flow_data['date']):
            flow_data['date'] = pd.to_datetime(flow_data['date'])

        midpoint = flow_data['date'].median()
        recent = flow_data[flow_data['date'] >= midpoint]
        prior = flow_data[flow_data['date'] < midpoint]

        rotation_signals = {}
        for sector in sector_flows.index:
            recent_flow = recent[recent['sector'] == sector]['flow_amount'].sum()
            prior_flow = prior[prior['sector'] == sector]['flow_amount'].sum()

            if prior_flow != 0:
                flow_change = (recent_flow - prior_flow) / abs(prior_flow)
            else:
                flow_change = 1.0 if recent_flow > 0 else (-1.0 if recent_flow < 0 else 0.0)

            rotation_signals[sector] = {
                'recent_flow': float(recent_flow),
                'prior_flow': float(prior_flow),
                'flow_change_pct': float(flow_change * 100),
                'rotation_direction': 'inflow' if flow_change > 0.1 else (
                    'outflow' if flow_change < -0.1 else 'stable'
                ),
            }

        # Identify leaders and laggards
        leaders = [s for s, info in rotation_signals.items() if info['rotation_direction'] == 'inflow']
        laggards = [s for s, info in rotation_signals.items() if info['rotation_direction'] == 'outflow']

        result = {
            'sector_strength': sector_strength,
            'sector_rankings': [{'sector': s, 'strength': v} for s, v in sector_ranking],
            'rotation_signals': rotation_signals,
            'leaders': leaders,
            'laggards': laggards,
            'rotation_detected': len(leaders) > 0 and len(laggards) > 0,
        }

        logger.info(
            f"Sector rotation analysis: leaders={leaders}, laggards={laggards}"
        )

        return result

    # ---------------------------------------------------------------
    # 5. Options Unusual Activity
    # ---------------------------------------------------------------

    def detect_unusual_options_activity(self, options_data: pd.DataFrame) -> List[Dict]:
        """
        Find unusual options volume/OI spikes, large block trades, sweep orders

        Args:
            options_data: DataFrame with columns: ticker, date, strike, expiry,
                option_type (call/put), volume, open_interest, premium,
                trade_type (optional: block/sweep/regular)

        Returns:
            List of unusual activity alerts
        """

        if options_data.empty:
            return []

        alerts = []
        options_data = options_data.copy()

        # Group by ticker and option contract
        for ticker in options_data['ticker'].unique():
            ticker_data = options_data[options_data['ticker'] == ticker]

            for _, row in ticker_data.iterrows():
                is_unusual = False
                reasons = []

                # Check volume vs open interest
                volume = row.get('volume', 0)
                oi = row.get('open_interest', 1)
                vol_oi_ratio = volume / oi if oi > 0 else 0.0

                if vol_oi_ratio > self.options_unusual_volume_mult:
                    is_unusual = True
                    reasons.append(f'volume/OI ratio={vol_oi_ratio:.1f}x')

                # Check for block trades
                premium = row.get('premium', 0)
                if premium >= self.options_block_premium_min:
                    is_unusual = True
                    reasons.append(f'block_premium=${premium:,.0f}')

                # Check for sweep orders
                trade_type = str(row.get('trade_type', '')).lower()
                if trade_type == 'sweep':
                    is_unusual = True
                    reasons.append('sweep_order')

                if is_unusual:
                    alert = {
                        'ticker': ticker,
                        'date': str(row.get('date', '')),
                        'strike': float(row.get('strike', 0)),
                        'expiry': str(row.get('expiry', '')),
                        'option_type': str(row.get('option_type', '')),
                        'volume': int(volume),
                        'open_interest': int(oi),
                        'vol_oi_ratio': float(vol_oi_ratio),
                        'premium': float(premium),
                        'reasons': reasons,
                        'sentiment': 'bullish' if str(row.get('option_type', '')).lower() == 'call' else 'bearish',
                    }
                    alerts.append(alert)

        # Sort by premium descending (most significant first)
        alerts.sort(key=lambda x: x['premium'], reverse=True)

        logger.info(f"Detected {len(alerts)} unusual options activity alerts")

        return alerts

    def options_sentiment_from_flow(self, options_data: pd.DataFrame) -> Dict:
        """
        Calculate options sentiment from order flow

        Analyzes net call vs put flow, premium-weighted sentiment,
        and attempts to identify smart money options flow.

        Args:
            options_data: DataFrame with columns: ticker, option_type,
                volume, premium, trade_type (optional)

        Returns:
            Dictionary with options sentiment metrics
        """

        if options_data.empty:
            return {'sentiment': 'neutral', 'put_call_ratio': None}

        calls = options_data[options_data['option_type'].str.lower() == 'call']
        puts = options_data[options_data['option_type'].str.lower() == 'put']

        # Volume-based put/call ratio
        call_volume = calls['volume'].sum()
        put_volume = puts['volume'].sum()
        total_volume = call_volume + put_volume

        put_call_ratio = put_volume / call_volume if call_volume > 0 else float('inf')

        # Premium-weighted sentiment
        call_premium = calls['premium'].sum() if 'premium' in calls.columns else 0.0
        put_premium = puts['premium'].sum() if 'premium' in puts.columns else 0.0
        total_premium = call_premium + put_premium

        premium_sentiment = (
            (call_premium - put_premium) / total_premium if total_premium > 0 else 0.0
        )

        # Smart money flow (block trades and sweeps)
        smart_money_sentiment = 0.0
        if 'trade_type' in options_data.columns:
            smart_mask = options_data['trade_type'].str.lower().isin(['block', 'sweep'])
            smart_trades = options_data[smart_mask]

            if not smart_trades.empty:
                smart_calls = smart_trades[smart_trades['option_type'].str.lower() == 'call']
                smart_puts = smart_trades[smart_trades['option_type'].str.lower() == 'put']

                smart_call_premium = smart_calls['premium'].sum() if 'premium' in smart_calls.columns else 0.0
                smart_put_premium = smart_puts['premium'].sum() if 'premium' in smart_puts.columns else 0.0
                smart_total = smart_call_premium + smart_put_premium

                smart_money_sentiment = (
                    (smart_call_premium - smart_put_premium) / smart_total
                    if smart_total > 0 else 0.0
                )

        # Overall sentiment classification
        if put_call_ratio < 0.7 and premium_sentiment > 0.2:
            sentiment = 'bullish'
        elif put_call_ratio > 1.3 and premium_sentiment < -0.2:
            sentiment = 'bearish'
        else:
            sentiment = 'neutral'

        result = {
            'sentiment': sentiment,
            'put_call_ratio': float(put_call_ratio) if put_call_ratio != float('inf') else None,
            'call_volume': int(call_volume),
            'put_volume': int(put_volume),
            'premium_sentiment': float(premium_sentiment),
            'call_premium': float(call_premium),
            'put_premium': float(put_premium),
            'smart_money_sentiment': float(smart_money_sentiment),
        }

        logger.info(
            f"Options sentiment: {sentiment}, P/C ratio={put_call_ratio:.2f}, "
            f"premium_sentiment={premium_sentiment:.3f}"
        )

        return result

    def unusual_activity_score(self, options_data: pd.DataFrame,
                               historical_avg: Optional[pd.DataFrame] = None) -> float:
        """
        Compute composite score of how unusual current options activity is

        Args:
            options_data: Current options activity DataFrame
            historical_avg: Optional historical average DataFrame with columns:
                ticker, avg_volume, avg_premium, avg_oi

        Returns:
            Float score from 0.0 (normal) to 1.0 (extremely unusual)
        """

        if options_data.empty:
            return 0.0

        scores = []

        # Volume unusualness
        total_volume = options_data['volume'].sum()
        if historical_avg is not None and 'avg_volume' in historical_avg.columns:
            hist_vol = historical_avg['avg_volume'].sum()
            if hist_vol > 0:
                vol_ratio = total_volume / hist_vol
                vol_score = np.clip((vol_ratio - 1.0) / 3.0, 0.0, 1.0)
                scores.append(vol_score)

        # Premium unusualness
        if 'premium' in options_data.columns:
            total_premium = options_data['premium'].sum()
            if historical_avg is not None and 'avg_premium' in historical_avg.columns:
                hist_premium = historical_avg['avg_premium'].sum()
                if hist_premium > 0:
                    prem_ratio = total_premium / hist_premium
                    prem_score = np.clip((prem_ratio - 1.0) / 3.0, 0.0, 1.0)
                    scores.append(prem_score)

        # Volume/OI ratio unusualness
        if 'open_interest' in options_data.columns:
            total_oi = options_data['open_interest'].sum()
            if total_oi > 0:
                vol_oi = total_volume / total_oi
                vol_oi_score = np.clip(vol_oi / 3.0, 0.0, 1.0)
                scores.append(vol_oi_score)

        # Sweep/block proportion
        if 'trade_type' in options_data.columns:
            smart_count = options_data['trade_type'].str.lower().isin(['block', 'sweep']).sum()
            smart_ratio = smart_count / len(options_data) if len(options_data) > 0 else 0.0
            scores.append(np.clip(smart_ratio * 2.0, 0.0, 1.0))

        if not scores:
            # Fallback: simple volume z-score if per-ticker volumes available
            if 'volume' in options_data.columns and len(options_data) > 1:
                vol_mean = options_data['volume'].mean()
                vol_std = options_data['volume'].std()
                if vol_std > 0:
                    max_z = (options_data['volume'].max() - vol_mean) / vol_std
                    return float(np.clip(max_z / 4.0, 0.0, 1.0))
            return 0.0

        return float(np.mean(scores))

    # ---------------------------------------------------------------
    # 6. Cross-Asset Signals
    # ---------------------------------------------------------------

    def bond_equity_signal(self, bond_yields: pd.Series,
                           equity_returns: pd.Series) -> Dict:
        """
        Analyze bond-equity correlation regime and divergence signals

        Args:
            bond_yields: Series of bond yield levels (e.g., 10Y Treasury)
            equity_returns: Series of equity returns

        Returns:
            Dictionary with bond-equity signal analysis
        """

        if bond_yields.empty or equity_returns.empty:
            return {'signal': 'neutral', 'correlation': None}

        # Align series
        common_idx = bond_yields.index.intersection(equity_returns.index)
        if len(common_idx) < 20:
            return {'signal': 'neutral', 'correlation': None,
                    'reason': 'insufficient_data'}

        yields = bond_yields.loc[common_idx]
        eq_ret = equity_returns.loc[common_idx]

        # Yield changes (not levels) for correlation
        yield_changes = yields.diff().dropna()
        eq_ret_aligned = eq_ret.loc[yield_changes.index]

        # Rolling correlation
        window = min(self.correlation_window, len(yield_changes) // 2)
        if window < 10:
            window = 10

        if len(yield_changes) >= window:
            rolling_corr = yield_changes.rolling(window).corr(eq_ret_aligned)
            current_corr = float(rolling_corr.iloc[-1]) if not np.isnan(rolling_corr.iloc[-1]) else 0.0

            # Historical correlation for regime detection
            hist_corr = float(yield_changes.corr(eq_ret_aligned))
        else:
            current_corr = float(yield_changes.corr(eq_ret_aligned))
            hist_corr = current_corr

        # Regime classification
        if current_corr > 0.3:
            regime = 'risk_on'  # Yields and equities moving together
        elif current_corr < -0.3:
            regime = 'flight_to_safety'  # Inverse relationship
        else:
            regime = 'decorrelated'

        # Divergence detection: compare recent regime to historical
        divergence = abs(current_corr - hist_corr) > 0.4

        # Signal: yield curve moves that predict equity direction
        recent_yield_change = yields.diff(5).iloc[-1] if len(yields) > 5 else 0.0

        if regime == 'risk_on':
            signal = 'bullish' if recent_yield_change > 0 else 'bearish'
        elif regime == 'flight_to_safety':
            signal = 'bearish' if recent_yield_change < 0 else 'bullish'
        else:
            signal = 'neutral'

        result = {
            'signal': signal,
            'regime': regime,
            'current_correlation': current_corr,
            'historical_correlation': hist_corr,
            'divergence_detected': divergence,
            'recent_yield_change': float(recent_yield_change),
        }

        logger.info(f"Bond-equity signal: {signal}, regime={regime}, corr={current_corr:.3f}")

        return result

    def fx_equity_signal(self, fx_data: pd.Series,
                         equity_returns: pd.Series) -> Dict:
        """
        Analyze currency-equity lead-lag relationship

        For example, weak USD often correlates with strong equities.

        Args:
            fx_data: Series of FX rates (e.g., DXY or USD index)
            equity_returns: Series of equity returns

        Returns:
            Dictionary with FX-equity signal analysis
        """

        if fx_data.empty or equity_returns.empty:
            return {'signal': 'neutral', 'correlation': None}

        # Align series
        common_idx = fx_data.index.intersection(equity_returns.index)
        if len(common_idx) < 20:
            return {'signal': 'neutral', 'correlation': None,
                    'reason': 'insufficient_data'}

        fx = fx_data.loc[common_idx]
        eq = equity_returns.loc[common_idx]

        fx_returns = fx.pct_change().dropna()
        eq_aligned = eq.loc[fx_returns.index]

        # Contemporaneous correlation
        contemporaneous_corr = float(fx_returns.corr(eq_aligned))

        # Lead-lag analysis: does FX lead equities?
        lead_lag = {}
        for lag in range(1, 6):
            if len(fx_returns) > lag:
                lagged_fx = fx_returns.shift(lag).dropna()
                eq_for_lag = eq_aligned.loc[lagged_fx.index]
                if len(eq_for_lag) > 5:
                    lead_lag[f'fx_leads_{lag}d'] = float(lagged_fx.corr(eq_for_lag))

        # Find strongest lead-lag
        best_lag = None
        best_corr = 0.0
        for lag_name, corr_val in lead_lag.items():
            if abs(corr_val) > abs(best_corr):
                best_corr = corr_val
                best_lag = lag_name

        # Recent FX trend
        recent_fx_return = float(fx.pct_change(5).iloc[-1]) if len(fx) > 5 else 0.0

        # Signal generation
        if contemporaneous_corr < -0.3:
            # Inverse relationship (typical USD-equity)
            signal = 'bullish' if recent_fx_return < -0.005 else (
                'bearish' if recent_fx_return > 0.005 else 'neutral'
            )
        elif contemporaneous_corr > 0.3:
            # Positive relationship
            signal = 'bullish' if recent_fx_return > 0.005 else (
                'bearish' if recent_fx_return < -0.005 else 'neutral'
            )
        else:
            signal = 'neutral'

        result = {
            'signal': signal,
            'contemporaneous_correlation': contemporaneous_corr,
            'lead_lag_correlations': lead_lag,
            'strongest_lead_lag': best_lag,
            'strongest_lead_lag_corr': best_corr,
            'recent_fx_return_5d': recent_fx_return,
        }

        logger.info(f"FX-equity signal: {signal}, corr={contemporaneous_corr:.3f}")

        return result

    def commodity_equity_signal(self, commodity_prices: pd.DataFrame,
                                equity_returns: pd.Series) -> Dict:
        """
        Analyze commodity-equity transmission effects

        E.g., oil → energy stocks, gold → miners, copper → industrials.

        Args:
            commodity_prices: DataFrame with commodity price columns
                (e.g., 'oil', 'gold', 'copper')
            equity_returns: Series of equity returns

        Returns:
            Dictionary with commodity-equity signal analysis
        """

        if commodity_prices.empty or equity_returns.empty:
            return {'signals': {}, 'dominant_commodity': None}

        common_idx = commodity_prices.index.intersection(equity_returns.index)
        if len(common_idx) < 20:
            return {'signals': {}, 'dominant_commodity': None,
                    'reason': 'insufficient_data'}

        commodities = commodity_prices.loc[common_idx]
        eq = equity_returns.loc[common_idx]

        commodity_signals = {}
        best_commodity = None
        best_abs_corr = 0.0

        for col in commodities.columns:
            comm_returns = commodities[col].pct_change().dropna()
            eq_aligned = eq.loc[comm_returns.index]

            if len(eq_aligned) < 10:
                continue

            # Correlation
            corr = float(comm_returns.corr(eq_aligned))

            # Recent commodity trend
            recent_return = float(commodities[col].pct_change(5).iloc[-1]) if len(commodities[col]) > 5 else 0.0

            # Lead-lag (commodity leading equity by 1 day)
            if len(comm_returns) > 1:
                lagged = comm_returns.shift(1).dropna()
                eq_for_lag = eq_aligned.loc[lagged.index]
                lead_corr = float(lagged.corr(eq_for_lag)) if len(eq_for_lag) > 5 else 0.0
            else:
                lead_corr = 0.0

            # Signal based on recent commodity move and correlation direction
            if abs(corr) > 0.3:
                if corr > 0:
                    signal = 'bullish' if recent_return > 0.01 else (
                        'bearish' if recent_return < -0.01 else 'neutral'
                    )
                else:
                    signal = 'bearish' if recent_return > 0.01 else (
                        'bullish' if recent_return < -0.01 else 'neutral'
                    )
            else:
                signal = 'neutral'

            commodity_signals[col] = {
                'correlation': corr,
                'lead_correlation': lead_corr,
                'recent_return_5d': recent_return,
                'signal': signal,
            }

            if abs(corr) > best_abs_corr:
                best_abs_corr = abs(corr)
                best_commodity = col

        result = {
            'commodity_signals': commodity_signals,
            'dominant_commodity': best_commodity,
            'dominant_correlation': float(best_abs_corr) if best_commodity else None,
        }

        logger.info(
            f"Commodity-equity analysis: dominant={best_commodity}, "
            f"corr={best_abs_corr:.3f}"
        )

        return result

    def cross_asset_momentum(self, asset_returns: pd.DataFrame) -> Dict:
        """
        Multi-asset momentum signal across equities, bonds, commodities, FX

        Args:
            asset_returns: DataFrame with return series for multiple asset classes.
                Column names should indicate asset class (e.g., 'equity', 'bonds',
                'gold', 'oil', 'usd_index')

        Returns:
            Dictionary with cross-asset momentum signals
        """

        if asset_returns.empty:
            return {'momentum_signals': {}, 'composite_signal': 'neutral'}

        asset_returns = asset_returns.dropna()
        if len(asset_returns) < max(self.momentum_windows):
            # Use what we have
            usable_windows = [w for w in self.momentum_windows if w < len(asset_returns)]
            if not usable_windows:
                return {'momentum_signals': {}, 'composite_signal': 'neutral',
                        'reason': 'insufficient_data'}
        else:
            usable_windows = self.momentum_windows

        momentum_signals = {}

        for col in asset_returns.columns:
            series = asset_returns[col].dropna()
            if len(series) < min(usable_windows):
                continue

            col_momentum = {}

            for window in usable_windows:
                if len(series) >= window:
                    # Cumulative return over window
                    cum_return = float((1 + series.tail(window)).prod() - 1)

                    # Sharpe-like ratio for the window
                    mean_ret = series.tail(window).mean()
                    std_ret = series.tail(window).std()
                    sharpe = float(mean_ret / std_ret * np.sqrt(252)) if std_ret > 0 else 0.0

                    col_momentum[f'{window}d'] = {
                        'cumulative_return': cum_return,
                        'annualized_sharpe': sharpe,
                        'signal': 'bullish' if cum_return > 0 and sharpe > 0.5 else (
                            'bearish' if cum_return < 0 and sharpe < -0.5 else 'neutral'
                        ),
                    }

            # Composite per-asset signal (majority vote across windows)
            window_signals = [v['signal'] for v in col_momentum.values()]
            bullish_count = window_signals.count('bullish')
            bearish_count = window_signals.count('bearish')

            if bullish_count > bearish_count and bullish_count > len(window_signals) / 2:
                asset_signal = 'bullish'
            elif bearish_count > bullish_count and bearish_count > len(window_signals) / 2:
                asset_signal = 'bearish'
            else:
                asset_signal = 'neutral'

            momentum_signals[col] = {
                'windows': col_momentum,
                'composite_signal': asset_signal,
            }

        # Overall cross-asset composite
        asset_composites = [v['composite_signal'] for v in momentum_signals.values()]
        bullish_assets = asset_composites.count('bullish')
        bearish_assets = asset_composites.count('bearish')
        total_assets = len(asset_composites)

        if bullish_assets > bearish_assets and bullish_assets > total_assets / 2:
            composite_signal = 'bullish'
        elif bearish_assets > bullish_assets and bearish_assets > total_assets / 2:
            composite_signal = 'bearish'
        else:
            composite_signal = 'mixed'

        # Cross-asset correlation matrix
        corr_matrix = asset_returns.corr()
        avg_correlation = float(
            corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
        ) if len(corr_matrix) > 1 else 0.0

        result = {
            'momentum_signals': momentum_signals,
            'composite_signal': composite_signal,
            'bullish_asset_count': bullish_assets,
            'bearish_asset_count': bearish_assets,
            'avg_cross_asset_correlation': avg_correlation,
            'high_correlation_regime': avg_correlation > 0.5,
        }

        logger.info(
            f"Cross-asset momentum: {composite_signal}, "
            f"bullish={bullish_assets}/{total_assets}, avg_corr={avg_correlation:.3f}"
        )

        return result
