#!/usr/bin/env python3
"""
Opportunity Analytics Script

Analyzes logged arbitrage opportunities to answer:
1. How many opportunities per day?
2. How long do opportunities stay open?
3. What pairs/exchanges have most opportunities?
4. What are the spread distributions?
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from decimal import Decimal
from collections import defaultdict
import json

from src.common.config import load_config


class OpportunityAnalyzer:
    """Analyzes arbitrage opportunities from database"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
    
    def analyze_opportunities_per_day(self, days: int = 30) -> dict:
        """
        Analyze how many opportunities per day
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Dictionary with daily counts
        """
        cursor = self.conn.cursor()
        
        query = '''
            SELECT 
                DATE(detected_at) as date,
                COUNT(*) as total_opportunities,
                COUNT(DISTINCT pair) as unique_pairs,
                COUNT(DISTINCT buy_exchange || '-' || sell_exchange) as unique_routes,
                AVG(net_spread_pct) as avg_spread,
                MAX(net_spread_pct) as max_spread,
                MIN(net_spread_pct) as min_spread,
                SUM(CASE WHEN is_executed = 1 THEN 1 ELSE 0 END) as executed_count
            FROM opportunities
            WHERE detected_at >= DATE('now', '-' || ? || ' days')
            GROUP BY DATE(detected_at)
            ORDER BY date DESC
        '''
        
        cursor.execute(query, (days,))
        rows = cursor.fetchall()
        
        results = {
            'daily_breakdown': [],
            'summary': {}
        }
        
        total_opps = 0
        total_executed = 0
        
        for row in rows:
            daily_data = {
                'date': row['date'],
                'total_opportunities': row['total_opportunities'],
                'unique_pairs': row['unique_pairs'],
                'unique_routes': row['unique_routes'],
                'avg_spread_pct': float(row['avg_spread']) if row['avg_spread'] else 0,
                'max_spread_pct': float(row['max_spread']) if row['max_spread'] else 0,
                'min_spread_pct': float(row['min_spread']) if row['min_spread'] else 0,
                'executed': row['executed_count']
            }
            results['daily_breakdown'].append(daily_data)
            total_opps += row['total_opportunities']
            total_executed += row['executed_count']
        
        # Calculate summary statistics
        if results['daily_breakdown']:
            avg_per_day = total_opps / len(results['daily_breakdown'])
            results['summary'] = {
                'total_opportunities': total_opps,
                'total_executed': total_executed,
                'days_analyzed': len(results['daily_breakdown']),
                'avg_opportunities_per_day': avg_per_day,
                'execution_rate': (total_executed / total_opps * 100) if total_opps > 0 else 0
            }
        
        return results
    
    def analyze_opportunity_durations(self) -> dict:
        """
        Analyze how long opportunities stay open
        
        Returns:
            Statistics on opportunity lifecycles
        """
        cursor = self.conn.cursor()
        
        query = '''
            SELECT 
                opportunity_duration_ms,
                net_spread_pct,
                pair,
                buy_exchange,
                sell_exchange,
                times_seen,
                first_seen_at,
                expired_at
            FROM opportunities
            WHERE opportunity_duration_ms IS NOT NULL
            ORDER BY opportunity_duration_ms DESC
        '''
        
        cursor.execute(query)
        rows = cursor.fetchall()
        
        if not rows:
            return {
                'total_with_duration': 0,
                'message': 'No opportunities with duration data yet. Run paper trading first.'
            }
        
        durations_ms = [row['opportunity_duration_ms'] for row in rows]
        durations_seconds = [d / 1000 for d in durations_ms]
        
        # Calculate statistics
        sorted_durations = sorted(durations_ms)
        n = len(sorted_durations)
        
        avg_duration = sum(durations_ms) / n
        min_duration = min(durations_ms)
        max_duration = max(durations_ms)
        
        # Percentiles
        p10 = sorted_durations[int(n * 0.10)] if n > 0 else 0
        p25 = sorted_durations[int(n * 0.25)] if n > 0 else 0
        p50 = sorted_durations[int(n * 0.50)] if n > 0 else 0
        p75 = sorted_durations[int(n * 0.75)] if n > 0 else 0
        p90 = sorted_durations[int(n * 0.90)] if n > 0 else 0
        p95 = sorted_durations[int(n * 0.95)] if n > 0 else 0
        p99 = sorted_durations[int(n * 0.99)] if n > 0 else 0
        
        # Duration buckets (in seconds)
        buckets = {
            '< 100ms': sum(1 for d in durations_ms if d < 100),
            '100ms - 500ms': sum(1 for d in durations_ms if 100 <= d < 500),
            '500ms - 1s': sum(1 for d in durations_ms if 500 <= d < 1000),
            '1s - 2s': sum(1 for d in durations_ms if 1000 <= d < 2000),
            '2s - 5s': sum(1 for d in durations_ms if 2000 <= d < 5000),
            '5s - 10s': sum(1 for d in durations_ms if 5000 <= d < 10000),
            '> 10s': sum(1 for d in durations_ms if d >= 10000)
        }
        
        # Longest lasting opportunities
        longest_opps = []
        for row in rows[:10]:  # Top 10
            longest_opps.append({
                'duration_ms': row['opportunity_duration_ms'],
                'duration_seconds': row['opportunity_duration_ms'] / 1000,
                'pair': row['pair'],
                'route': f"{row['buy_exchange']} -> {row['sell_exchange']}",
                'net_spread_pct': float(row['net_spread_pct']),
                'times_seen': row['times_seen'],
                'first_seen': row['first_seen_at'],
                'expired': row['expired_at']
            })
        
        return {
            'total_with_duration': n,
            'statistics': {
                'avg_duration_ms': avg_duration,
                'avg_duration_seconds': avg_duration / 1000,
                'min_duration_ms': min_duration,
                'min_duration_seconds': min_duration / 1000,
                'max_duration_ms': max_duration,
                'max_duration_seconds': max_duration / 1000,
                'median_duration_ms': p50,
                'median_duration_seconds': p50 / 1000
            },
            'percentiles': {
                'p10_ms': p10,
                'p10_seconds': p10 / 1000,
                'p25_ms': p25,
                'p25_seconds': p25 / 1000,
                'p50_ms': p50,
                'p50_seconds': p50 / 1000,
                'p75_ms': p75,
                'p75_seconds': p75 / 1000,
                'p90_ms': p90,
                'p90_seconds': p90 / 1000,
                'p95_ms': p95,
                'p95_seconds': p95 / 1000,
                'p99_ms': p99,
                'p99_seconds': p99 / 1000
            },
            'duration_buckets': buckets,
            'longest_opportunities': longest_opps
        }
    
    def analyze_by_pair(self) -> dict:
        """
        Analyze opportunities by trading pair
        
        Returns:
            Statistics grouped by pair
        """
        cursor = self.conn.cursor()
        
        query = '''
            SELECT 
                pair,
                COUNT(*) as total_opportunities,
                AVG(net_spread_pct) as avg_spread,
                MAX(net_spread_pct) as max_spread,
                AVG(opportunity_duration_ms) as avg_duration_ms,
                SUM(CASE WHEN is_executed = 1 THEN 1 ELSE 0 END) as executed_count
            FROM opportunities
            GROUP BY pair
            ORDER BY total_opportunities DESC
        '''
        
        cursor.execute(query)
        rows = cursor.fetchall()
        
        pairs_data = []
        for row in rows:
            pairs_data.append({
                'pair': row['pair'],
                'total_opportunities': row['total_opportunities'],
                'avg_spread_pct': float(row['avg_spread']) if row['avg_spread'] else 0,
                'max_spread_pct': float(row['max_spread']) if row['max_spread'] else 0,
                'avg_duration_ms': float(row['avg_duration_ms']) if row['avg_duration_ms'] else None,
                'avg_duration_seconds': float(row['avg_duration_ms']) / 1000 if row['avg_duration_ms'] else None,
                'executed_count': row['executed_count']
            })
        
        return {
            'total_pairs': len(pairs_data),
            'pairs': pairs_data
        }
    
    def analyze_by_exchange_route(self) -> dict:
        """
        Analyze opportunities by exchange routing (buy -> sell)
        
        Returns:
            Statistics grouped by exchange pairs
        """
        cursor = self.conn.cursor()
        
        query = '''
            SELECT 
                buy_exchange,
                sell_exchange,
                COUNT(*) as total_opportunities,
                AVG(net_spread_pct) as avg_spread,
                MAX(net_spread_pct) as max_spread,
                AVG(opportunity_duration_ms) as avg_duration_ms,
                SUM(CASE WHEN is_executed = 1 THEN 1 ELSE 0 END) as executed_count
            FROM opportunities
            GROUP BY buy_exchange, sell_exchange
            ORDER BY total_opportunities DESC
        '''
        
        cursor.execute(query)
        rows = cursor.fetchall()
        
        routes_data = []
        for row in rows:
            routes_data.append({
                'route': f"{row['buy_exchange']} -> {row['sell_exchange']}",
                'buy_exchange': row['buy_exchange'],
                'sell_exchange': row['sell_exchange'],
                'total_opportunities': row['total_opportunities'],
                'avg_spread_pct': float(row['avg_spread']) if row['avg_spread'] else 0,
                'max_spread_pct': float(row['max_spread']) if row['max_spread'] else 0,
                'avg_duration_ms': float(row['avg_duration_ms']) if row['avg_duration_ms'] else None,
                'executed_count': row['executed_count']
            })
        
        return {
            'total_routes': len(routes_data),
            'routes': routes_data
        }
    
    def analyze_spread_distribution(self) -> dict:
        """
        Analyze distribution of spread percentages
        
        Returns:
            Spread distribution statistics
        """
        cursor = self.conn.cursor()
        
        query = '''
            SELECT net_spread_pct
            FROM opportunities
            ORDER BY net_spread_pct DESC
        '''
        
        cursor.execute(query)
        rows = cursor.fetchall()
        
        if not rows:
            return {'message': 'No opportunities found'}
        
        spreads = [float(row['net_spread_pct']) for row in rows]
        n = len(spreads)
        
        # Statistics
        avg_spread = sum(spreads) / n
        min_spread = min(spreads)
        max_spread = max(spreads)
        
        sorted_spreads = sorted(spreads)
        median_spread = sorted_spreads[n // 2]
        
        # Spread buckets
        buckets = {
            '0.5% - 0.7%': sum(1 for s in spreads if 0.5 <= s < 0.7),
            '0.7% - 1.0%': sum(1 for s in spreads if 0.7 <= s < 1.0),
            '1.0% - 1.5%': sum(1 for s in spreads if 1.0 <= s < 1.5),
            '1.5% - 2.0%': sum(1 for s in spreads if 1.5 <= s < 2.0),
            '2.0% - 3.0%': sum(1 for s in spreads if 2.0 <= s < 3.0),
            '> 3.0%': sum(1 for s in spreads if s >= 3.0)
        }
        
        return {
            'total_opportunities': n,
            'statistics': {
                'avg_spread_pct': avg_spread,
                'min_spread_pct': min_spread,
                'max_spread_pct': max_spread,
                'median_spread_pct': median_spread
            },
            'spread_buckets': buckets
        }
    
    def get_recent_opportunities(self, limit: int = 20) -> list:
        """Get most recent opportunities"""
        cursor = self.conn.cursor()
        
        query = '''
            SELECT 
                detected_at,
                pair,
                buy_exchange,
                sell_exchange,
                net_spread_pct,
                estimated_profit_usd,
                opportunity_duration_ms,
                times_seen,
                is_executed
            FROM opportunities
            ORDER BY detected_at DESC
            LIMIT ?
        '''
        
        cursor.execute(query, (limit,))
        rows = cursor.fetchall()
        
        opportunities = []
        for row in rows:
            opportunities.append({
                'detected_at': row['detected_at'],
                'pair': row['pair'],
                'route': f"{row['buy_exchange']} -> {row['sell_exchange']}",
                'net_spread_pct': float(row['net_spread_pct']),
                'estimated_profit_usd': float(row['estimated_profit_usd']),
                'duration_ms': row['opportunity_duration_ms'],
                'duration_seconds': row['opportunity_duration_ms'] / 1000 if row['opportunity_duration_ms'] else None,
                'times_seen': row['times_seen'],
                'executed': bool(row['is_executed'])
            })
        
        return opportunities
    
    def close(self):
        """Close database connection"""
        self.conn.close()


def print_analysis(analyzer: OpportunityAnalyzer, output_format: str = 'text'):
    """
    Print comprehensive analysis
    
    Args:
        analyzer: OpportunityAnalyzer instance
        output_format: 'text' or 'json'
    """
    print("=" * 80)
    print("CRYPTO ARBITRAGE OPPORTUNITY ANALYSIS")
    print("=" * 80)
    print()
    
    # 1. Opportunities per day
    print("📊 OPPORTUNITIES PER DAY (Last 30 Days)")
    print("-" * 80)
    daily_analysis = analyzer.analyze_opportunities_per_day(days=30)
    
    if daily_analysis['summary']:
        summary = daily_analysis['summary']
        print(f"Total opportunities: {summary['total_opportunities']}")
        print(f"Average per day: {summary['avg_opportunities_per_day']:.1f}")
        print(f"Total executed: {summary['total_executed']}")
        print(f"Execution rate: {summary['execution_rate']:.1f}%")
        print()
        
        print("Recent days:")
        for day in daily_analysis['daily_breakdown'][:7]:
            print(f"  {day['date']}: {day['total_opportunities']:3d} opportunities, "
                  f"avg spread: {day['avg_spread_pct']:.2f}%, "
                  f"{day['executed']} executed")
    else:
        print("No opportunities logged yet.")
    
    print()
    
    # 2. Opportunity durations
    print("⏱️  OPPORTUNITY DURATION ANALYSIS")
    print("-" * 80)
    duration_analysis = analyzer.analyze_opportunity_durations()
    
    if duration_analysis.get('total_with_duration', 0) > 0:
        stats = duration_analysis['statistics']
        print(f"Total opportunities with duration data: {duration_analysis['total_with_duration']}")
        print()
        print("Duration Statistics:")
        print(f"  Average: {stats['avg_duration_seconds']:.2f} seconds ({stats['avg_duration_ms']:.0f} ms)")
        print(f"  Median:  {stats['median_duration_seconds']:.2f} seconds ({stats['median_duration_ms']:.0f} ms)")
        print(f"  Min:     {stats['min_duration_seconds']:.2f} seconds ({stats['min_duration_ms']:.0f} ms)")
        print(f"  Max:     {stats['max_duration_seconds']:.2f} seconds ({stats['max_duration_ms']:.0f} ms)")
        print()
        
        percentiles = duration_analysis['percentiles']
        print("Duration Percentiles:")
        print(f"  P10: {percentiles['p10_seconds']:.2f}s")
        print(f"  P25: {percentiles['p25_seconds']:.2f}s")
        print(f"  P50: {percentiles['p50_seconds']:.2f}s")
        print(f"  P75: {percentiles['p75_seconds']:.2f}s")
        print(f"  P90: {percentiles['p90_seconds']:.2f}s")
        print(f"  P95: {percentiles['p95_seconds']:.2f}s")
        print(f"  P99: {percentiles['p99_seconds']:.2f}s")
        print()
        
        print("Duration Distribution:")
        for bucket, count in duration_analysis['duration_buckets'].items():
            pct = (count / duration_analysis['total_with_duration'] * 100)
            bar = '█' * int(pct / 2)
            print(f"  {bucket:15s}: {count:4d} ({pct:5.1f}%) {bar}")
        print()
        
        if duration_analysis['longest_opportunities']:
            print("Longest-lasting opportunities:")
            for i, opp in enumerate(duration_analysis['longest_opportunities'][:5], 1):
                print(f"  {i}. {opp['pair']} via {opp['route']}")
                print(f"     Duration: {opp['duration_seconds']:.2f}s, Spread: {opp['net_spread_pct']:.2f}%, "
                      f"Seen: {opp['times_seen']} times")
    else:
        print(duration_analysis.get('message', 'No duration data available'))
    
    print()
    
    # 3. By pair
    print("💰 TOP OPPORTUNITIES BY TRADING PAIR")
    print("-" * 80)
    pair_analysis = analyzer.analyze_by_pair()
    
    if pair_analysis['pairs']:
        print(f"Total pairs with opportunities: {pair_analysis['total_pairs']}")
        print()
        for i, pair_data in enumerate(pair_analysis['pairs'][:10], 1):
            print(f"{i:2d}. {pair_data['pair']:12s} - {pair_data['total_opportunities']:4d} opportunities, "
                  f"avg spread: {pair_data['avg_spread_pct']:.2f}%", end='')
            if pair_data['avg_duration_seconds']:
                print(f", avg duration: {pair_data['avg_duration_seconds']:.2f}s")
            else:
                print()
    else:
        print("No pair data available")
    
    print()
    
    # 4. By exchange route
    print("🔀 TOP EXCHANGE ROUTES")
    print("-" * 80)
    route_analysis = analyzer.analyze_by_exchange_route()
    
    if route_analysis['routes']:
        print(f"Total unique routes: {route_analysis['total_routes']}")
        print()
        for i, route_data in enumerate(route_analysis['routes'][:10], 1):
            print(f"{i:2d}. {route_data['route']:25s} - {route_data['total_opportunities']:4d} opportunities, "
                  f"avg spread: {route_data['avg_spread_pct']:.2f}%")
    else:
        print("No route data available")
    
    print()
    
    # 5. Spread distribution
    print("📈 SPREAD DISTRIBUTION")
    print("-" * 80)
    spread_analysis = analyzer.analyze_spread_distribution()
    
    if 'statistics' in spread_analysis:
        stats = spread_analysis['statistics']
        print(f"Average spread: {stats['avg_spread_pct']:.2f}%")
        print(f"Median spread:  {stats['median_spread_pct']:.2f}%")
        print(f"Min spread:     {stats['min_spread_pct']:.2f}%")
        print(f"Max spread:     {stats['max_spread_pct']:.2f}%")
        print()
        
        print("Spread buckets:")
        for bucket, count in spread_analysis['spread_buckets'].items():
            pct = (count / spread_analysis['total_opportunities'] * 100)
            bar = '█' * int(pct / 2)
            print(f"  {bucket:15s}: {count:4d} ({pct:5.1f}%) {bar}")
    else:
        print(spread_analysis.get('message', 'No spread data available'))
    
    print()
    
    # 6. Recent opportunities
    print("🕐 RECENT OPPORTUNITIES (Last 20)")
    print("-" * 80)
    recent = analyzer.get_recent_opportunities(limit=20)
    
    if recent:
        for opp in recent[:10]:
            exec_marker = "✅" if opp['executed'] else "  "
            duration_str = f"{opp['duration_seconds']:.2f}s" if opp['duration_seconds'] else "N/A"
            print(f"{exec_marker} {opp['detected_at']:19s} | {opp['pair']:12s} | {opp['route']:25s} | "
                  f"{opp['net_spread_pct']:5.2f}% | ${opp['estimated_profit_usd']:6.2f} | "
                  f"Duration: {duration_str}")
    else:
        print("No recent opportunities")
    
    print()
    print("=" * 80)


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze Arbitrage Opportunities')
    parser.add_argument('--db', type=str, default=None,
                       help='Path to database (default: from config)')
    parser.add_argument('--format', type=str, default='text', choices=['text', 'json'],
                       help='Output format')
    args = parser.parse_args()
    
    # Get database path
    if args.db:
        db_path = args.db
    else:
        config = load_config()
        db_path = config.database.sqlite_path
    
    if not Path(db_path).exists():
        print(f"❌ Database not found: {db_path}")
        print("Run paper trading first to generate data.")
        return 1
    
    # Run analysis
    analyzer = OpportunityAnalyzer(db_path)
    
    try:
        print_analysis(analyzer, output_format=args.format)
    finally:
        analyzer.close()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())


