import yfinance as yf
import pandas as pd
import numpy as np
import json
from TickerProvider import get_most_active_tickers
from scipy.stats import norm
import datetime

# Load configuration
with open('config.json', 'r') as f:
    config = json.load(f)

CREDIT_SPREAD_OTM_PERCENT = config['CREDIT_SPREAD_OTM_PERCENT']
STRANGLE_OTM_PERCENT = config['STRANGLE_OTM_PERCENT']
MIN_PROB_SUCCESS = config['MIN_PROB_SUCCESS']
MIN_CREDIT_RECEIVED = config['MIN_CREDIT_RECEIVED']

def black_scholes_cdf(S, K, T, r, sigma, option_type):
    """
    Calculates the cumulative distribution function (CDF) for Black-Scholes.
    Used to estimate probability of expiring in-the-money.
    S: current stock price
    K: strike price
    T: time to expiration (in years)
    r: risk-free rate (e.g., 10-year treasury yield)
    sigma: implied volatility
    option_type: 'call' or 'put'
    """
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        return norm.cdf(d2)
    elif option_type == 'put':
        return norm.cdf(-d2)
    return 0

def get_trade_ideas():
    """
    Scans for various option strategies based on configured risk parameters.
    """
    dynamic_tickers = get_most_active_tickers(num_tickers=config['NUM_MOST_ACTIVE_TICKERS'])
    TICKERS = list(set(config['FIXED_TICKERS'] + dynamic_tickers))
    TICKERS = [ticker.strip() for ticker in TICKERS] # Clean up tickers

    if not TICKERS:
        print("No tickers to scan. Exiting.")
        return []
        
    trade_ideas = []

    # Fetch risk-free rate (using 10-year treasury yield)
    try:
        tnx = yf.Ticker('^TNX').history(period='1d')['Close'].iloc[-1]
        risk_free_rate = tnx / 100  # Convert to decimal
    except Exception:
        risk_free_rate = 0.02 # Default to 2% if cannot fetch

    for ticker in TICKERS:
        try:
            stock = yf.Ticker(ticker)
            history = stock.history(period='1d')
            if history.empty:
                print(f"Skipping {ticker}: No historical data found.")
                continue
            current_price = history['Close'].iloc[-1]
            expirations = stock.options[:2]

            for expiry in expirations:
                chain = stock.option_chain(expiry)
                calls = chain.calls
                puts = chain.puts

                # Calculate time to expiration in years
                days_to_expiry = (datetime.datetime.strptime(expiry, '%Y-%m-%d').date() - datetime.datetime.now().date()).days
                T = days_to_expiry / 365.0
                if T <= 0: # Skip expired options
                    continue

                # --- Strategy 1: Bull Put Spread (Bullish) ---
                target_put_strike = current_price * (1 - (CREDIT_SPREAD_OTM_PERCENT / 100))
                otm_puts = puts[puts['strike'] < target_put_strike]
                if not otm_puts.empty:
                    candidate = otm_puts.iloc[(otm_puts['strike'] - target_put_strike).abs().argsort()[:1]].iloc[0]
                    
                    # Estimate probability of success (option expiring OTM)
                    prob_otm = 1 - black_scholes_cdf(current_price, float(candidate['strike']), T, risk_free_rate, float(candidate['impliedVolatility']), 'put')

                    # For a simple credit spread, we need a second leg (bought put) to define max loss.
                    # For now, we'll assume a 5-point wide spread for max loss calculation for simplicity.
                    # In a real scenario, you'd scan for the second leg.
                    assumed_spread_width = 5.0 # Example: assuming a $5 wide spread
                    max_profit = float(candidate['bid'])
                    max_loss = assumed_spread_width - max_profit
                    risk_reward_ratio = float(max_profit / max_loss) if max_loss > 0 else float('inf')

                    if prob_otm * 100 >= MIN_PROB_SUCCESS and float(candidate['bid']) >= MIN_CREDIT_RECEIVED:
                        trade_ideas.append({
                            "ticker": ticker, "strategy": "Bull Put Spread", "expiry": expiry,
                            "details": f"Sell {float(candidate['strike'])} Put",
                            "cost": float(candidate['bid']),
                            "metric_name": "Prob. of Success",
                            "metric_value": f"{prob_otm * 100:.1f}%",
                            "max_profit": float(max_profit),
                            "max_loss": float(max_loss),
                            "risk_reward_ratio": float(risk_reward_ratio)
                        })

                # --- Strategy 2: Bear Call Spread (Bearish) ---
                target_call_strike = current_price * (1 + (CREDIT_SPREAD_OTM_PERCENT / 100))
                otm_calls = calls[calls['strike'] > target_call_strike]
                if not otm_calls.empty:
                    candidate = otm_calls.iloc[(otm_calls['strike'] - target_call_strike).abs().argsort()[:1]].iloc[0]

                    # Estimate probability of success (option expiring OTM)
                    prob_otm = 1 - black_scholes_cdf(current_price, float(candidate['strike']), T, risk_free_rate, float(candidate['impliedVolatility']), 'call')

                    # For a simple credit spread, we need a second leg (bought call) to define max loss.
                    # For now, we'll assume a 5-point wide spread for max loss calculation for simplicity.
                    # In a real scenario, you'd scan for the second leg.
                    assumed_spread_width = 5.0 # Example: assuming a $5 wide spread
                    max_profit = float(candidate['bid'])
                    max_loss = assumed_spread_width - max_profit
                    risk_reward_ratio = float(max_profit / max_loss) if max_loss > 0 else float('inf')

                    if prob_otm * 100 >= MIN_PROB_SUCCESS and float(candidate['bid']) >= MIN_CREDIT_RECEIVED:
                        trade_ideas.append({
                            "ticker": ticker, "strategy": "Bear Call Spread", "expiry": expiry,
                            "details": f"Sell {float(candidate['strike'])} Call",
                            "cost": float(candidate['bid']),
                            "metric_name": "Prob. of Success",
                            "metric_value": f"{prob_otm * 100:.1f}%",
                            "max_profit": float(max_profit),
                            "max_loss": float(max_loss),
                            "risk_reward_ratio": float(risk_reward_ratio)
                        })

                # --- Strategy 3: Long Straddle (High Volatility) ---
                atm_strike_row = puts.iloc[(puts['strike'] - current_price).abs().argsort()[:1]]
                if atm_strike_row.empty:
                    continue
                atm_strike = float(atm_strike_row['strike'].iloc[0])
                atm_call = calls[calls['strike'] == atm_strike]
                atm_put = puts[puts['strike'] == atm_strike]
                if not atm_call.empty and not atm_put.empty:
                    cost = float(atm_call['ask'].iloc[0]) + float(atm_put['ask'].iloc[0])
                    implied_move = (cost / current_price) * 100
                    trade_ideas.append({
                        "ticker": ticker, "strategy": "Long Straddle", "expiry": expiry,
                        "details": f"Buy {atm_strike} Call & {atm_strike} Put", 
                        "cost": -float(cost),
                        "metric_name": "Implied Move",
                        "metric_value": f"+/- {implied_move:.2f}%"
                    })

                # --- Strategy 4: Long Strangle (High Volatility) ---
                target_strangle_put_strike = current_price * (1 - (STRANGLE_OTM_PERCENT / 100))
                target_strangle_call_strike = current_price * (1 + (STRANGLE_OTM_PERCENT / 100))
                otm_put_candidate = puts[puts['strike'] < target_strangle_put_strike]
                otm_call_candidate = calls[calls['strike'] > target_strangle_call_strike]
                if not otm_put_candidate.empty and not otm_call_candidate.empty:
                    otm_put = otm_put_candidate.iloc[(otm_put_candidate['strike'] - target_strangle_put_strike).abs().argsort()[:1]].iloc[0]
                    otm_call = otm_call_candidate.iloc[(otm_call_candidate['strike'] - target_strangle_call_strike).abs().argsort()[:1]].iloc[0]
                    cost = float(otm_call['ask']) + float(otm_put['ask'])
                    implied_move = (cost / current_price) * 100
                    trade_ideas.append({
                        "ticker": ticker, "strategy": "Long Strangle", "expiry": expiry,
                        "details": f"Buy {float(otm_call['strike'])} Call & {float(otm_put['strike'])} Put", 
                        "cost": -float(cost),
                        "metric_name": "Implied Move",
                        "metric_value": f"+/- {implied_move:.2f}%"
                    })

        except Exception as e:
            trade_ideas.append({"ticker": ticker, "strategy": f"Error: {e}", "details": "", "cost": 0.0, "metric_name": "", "metric_value": ""})
            
    return trade_ideas

if __name__ == "__main__":
    trades = get_trade_ideas()
    for trade in trades:
        cost_type = "Credit" if trade['cost'] > 0 else "Debit"
        print(f"{trade['ticker']} {trade['strategy']} ({trade.get('expiry', 'N/A')})")
        print(f"- {trade['details']} | {cost_type}: ${abs(trade['cost']):.2f}")
        if trade['metric_name']:
            print(f"- {trade['metric_name']}: {trade['metric_value']}")
