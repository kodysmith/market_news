#!/usr/bin/env python3
"""
Display current actionable trade ideas from the scanner
"""

import json
import sys
from datetime import datetime

def load_report():
    """Load the latest report.json"""
    try:
        with open('report.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("❌ No report.json found. Run the scanner first.")
        return None
    except Exception as e:
        print(f"❌ Error loading report: {e}")
        return None

def format_trade_idea(idea):
    """Format a trade idea for display"""
    ticker = idea.get('ticker', 'N/A')
    strategy = idea.get('strategy', 'N/A')
    expiry = idea.get('expiry', 'N/A')
    short_k = idea.get('shortK', 0)
    long_k = idea.get('longK', 0)
    width = idea.get('width', 0)
    credit = idea.get('credit', 0)
    max_loss = idea.get('maxLoss', 0)
    dte = idea.get('dte', 0)
    pop = idea.get('pop', 0)
    ev = idea.get('ev', 0)
    fill_score = idea.get('fillScore', 0)
    
    # Calculate risk/reward
    max_profit = credit * 100
    max_loss_amount = max_loss * 100
    risk_reward = max_profit / max_loss_amount if max_loss_amount > 0 else 0
    
    return f"""
🎯 {ticker} {strategy} - {expiry}
   Strikes: {short_k:.0f}/{long_k:.0f} (${width:.0f}-point spread)
   Credit: ${credit:.2f} per contract (${max_profit:.0f} max profit)
   Max Loss: {max_loss:.2f} per contract (${max_loss_amount:.0f} max loss)
   DTE: {dte} days
   Probability of Profit: {pop:.1%}
   Expected Value: ${ev:.2f} per contract
   Fill Score: {fill_score:.1f}/10
   Risk/Reward: 1:{risk_reward:.1f}
"""

def main():
    print("🔍 Current Actionable Trade Ideas")
    print("=" * 50)
    
    report = load_report()
    if not report:
        return
    
    # Show scanner info
    scanner = report.get('scanner', {})
    universe = scanner.get('universe', [])
    dte_window = scanner.get('dteWindow', [])
    thresholds = scanner.get('thresholds', {})
    
    print(f"📊 Scanner Status:")
    print(f"   Universe: {', '.join(universe)}")
    print(f"   DTE Window: {dte_window[0]}-{dte_window[1]} days")
    print(f"   Min POP: {thresholds.get('minPOP', 0):.1%}")
    print(f"   Min EV per $100: ${thresholds.get('minEVPer100', 0):.2f}")
    print()
    
    # Show trade ideas
    ideas = report.get('topIdeas', [])
    if not ideas:
        print("❌ No actionable trade ideas found")
        print("   This is common in low-volatility environments")
        print("   Consider waiting for higher IV or adjusting criteria")
        return
    
    print(f"✅ Found {len(ideas)} actionable trade idea(s):")
    print()
    
    for i, idea in enumerate(ideas, 1):
        print(f"Trade #{i}:")
        print(format_trade_idea(idea))
    
    # Show alerts
    alerts = report.get('alertsSentThisRun', [])
    if alerts:
        print(f"🔔 {len(alerts)} alert(s) sent this run")
    
    # Show timestamp
    as_of = report.get('asOf', '')
    if as_of:
        try:
            dt = datetime.fromisoformat(as_of.replace('Z', '+00:00'))
            print(f"📅 Last updated: {dt.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        except:
            print(f"📅 Last updated: {as_of}")

if __name__ == "__main__":
    main()

