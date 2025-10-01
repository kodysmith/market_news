#!/usr/bin/env python3
"""
Opportunity Database Query Interface

Quick query interface for the opportunity database.
Allows fast querying without API costs.
"""

import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import argparse
import sys


class OpportunityQuerier:
    """Query interface for opportunity database"""
    
    def __init__(self, db_path: str = "opportunity_database.db"):
        self.db_path = db_path
        self.connection = sqlite3.connect(db_path)
    
    def query(self, 
              ticker: str = None,
              sector: str = None,
              opportunity_type: str = None,
              min_score: float = None,
              max_score: float = None,
              days_back: int = 7,
              limit: int = 100,
              format_output: str = 'table') -> pd.DataFrame:
        """Query opportunities with various filters"""
        
        query = "SELECT * FROM opportunities WHERE 1=1"
        params = []
        
        if ticker:
            query += " AND ticker = ?"
            params.append(ticker.upper())
        
        if sector:
            query += " AND sector = ?"
            params.append(sector.title())
        
        if opportunity_type:
            query += " AND opportunity_type = ?"
            params.append(opportunity_type.upper())
        
        if min_score is not None:
            query += " AND overall_score >= ?"
            params.append(min_score)
        
        if max_score is not None:
            query += " AND overall_score <= ?"
            params.append(max_score)
        
        if days_back:
            query += " AND scan_date >= datetime('now', '-{} days')".format(days_back)
        
        query += " ORDER BY scan_date DESC, overall_score DESC LIMIT ?"
        params.append(limit)
        
        df = pd.read_sql_query(query, self.connection, params=params)
        
        if format_output == 'csv':
            return df
        elif format_output == 'json':
            return df.to_json(orient='records', indent=2)
        else:
            return df
    
    def get_top_opportunities(self, limit: int = 20, min_score: float = 70) -> pd.DataFrame:
        """Get top opportunities by score"""
        return self.query(min_score=min_score, limit=limit)
    
    def get_sector_opportunities(self, sector: str, limit: int = 20) -> pd.DataFrame:
        """Get opportunities for specific sector"""
        return self.query(sector=sector, limit=limit)
    
    def get_ticker_history(self, ticker: str, days_back: int = 30) -> pd.DataFrame:
        """Get historical opportunities for a ticker"""
        return self.query(ticker=ticker, days_back=days_back, limit=1000)
    
    def get_sector_summary(self, days_back: int = 7) -> pd.DataFrame:
        """Get sector performance summary"""
        query = '''
            SELECT sector, 
                   COUNT(*) as total_opportunities,
                   AVG(overall_score) as avg_score,
                   MAX(overall_score) as max_score,
                   MIN(overall_score) as min_score,
                   SUM(CASE WHEN opportunity_type = 'STRONG_BUY' THEN 1 ELSE 0 END) as strong_buy,
                   SUM(CASE WHEN opportunity_type = 'BUY' THEN 1 ELSE 0 END) as buy,
                   SUM(CASE WHEN opportunity_type = 'HOLD' THEN 1 ELSE 0 END) as hold,
                   SUM(CASE WHEN opportunity_type = 'SELL' THEN 1 ELSE 0 END) as sell,
                   SUM(CASE WHEN opportunity_type = 'STRONG_SELL' THEN 1 ELSE 0 END) as strong_sell
            FROM opportunities 
            WHERE scan_date >= datetime('now', '-{} days')
            GROUP BY sector
            ORDER BY avg_score DESC
        '''.format(days_back)
        
        return pd.read_sql_query(query, self.connection)
    
    def get_training_data(self, days_back: int = 30) -> pd.DataFrame:
        """Get training data for ML models"""
        query = '''
            SELECT ticker, sector, scan_date, overall_score, opportunity_type,
                   technical_score, fundamental_score, sector_score,
                   rsi, pe_ratio, revenue_growth, profit_margin, debt_equity,
                   relative_strength, market_cap, current_price,
                   target_price, stop_loss, risk_reward_ratio
            FROM opportunities 
            WHERE scan_date >= datetime('now', '-{} days')
            ORDER BY scan_date DESC
        '''.format(days_back)
        
        return pd.read_sql_query(query, self.connection)
    
    def get_opportunity_stats(self) -> dict:
        """Get database statistics"""
        stats = {}
        
        # Total opportunities
        cursor = self.connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM opportunities")
        stats['total_opportunities'] = cursor.fetchone()[0]
        
        # Opportunities by type
        cursor.execute("""
            SELECT opportunity_type, COUNT(*) 
            FROM opportunities 
            GROUP BY opportunity_type
        """)
        stats['by_type'] = dict(cursor.fetchall())
        
        # Opportunities by sector
        cursor.execute("""
            SELECT sector, COUNT(*) 
            FROM opportunities 
            GROUP BY sector
            ORDER BY COUNT(*) DESC
        """)
        stats['by_sector'] = dict(cursor.fetchall())
        
        # Date range
        cursor.execute("SELECT MIN(scan_date), MAX(scan_date) FROM opportunities")
        min_date, max_date = cursor.fetchone()
        stats['date_range'] = {'min': min_date, 'max': max_date}
        
        # Average scores
        cursor.execute("""
            SELECT AVG(overall_score), AVG(technical_score), 
                   AVG(fundamental_score), AVG(sector_score)
            FROM opportunities
        """)
        avg_scores = cursor.fetchone()
        stats['avg_scores'] = {
            'overall': round(avg_scores[0], 2) if avg_scores[0] else 0,
            'technical': round(avg_scores[1], 2) if avg_scores[1] else 0,
            'fundamental': round(avg_scores[2], 2) if avg_scores[2] else 0,
            'sector': round(avg_scores[3], 2) if avg_scores[3] else 0
        }
        
        return stats
    
    def close(self):
        """Close database connection"""
        self.connection.close()


def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description='Query Opportunity Database')
    parser.add_argument('--ticker', help='Filter by ticker symbol')
    parser.add_argument('--sector', help='Filter by sector')
    parser.add_argument('--type', help='Filter by opportunity type (STRONG_BUY, BUY, HOLD, SELL, STRONG_SELL)')
    parser.add_argument('--min-score', type=float, help='Minimum overall score')
    parser.add_argument('--max-score', type=float, help='Maximum overall score')
    parser.add_argument('--days', type=int, default=7, help='Days back to search (default: 7)')
    parser.add_argument('--limit', type=int, default=20, help='Maximum results (default: 20)')
    parser.add_argument('--format', choices=['table', 'csv', 'json'], default='table', help='Output format')
    parser.add_argument('--top', action='store_true', help='Show top opportunities only')
    parser.add_argument('--sector-summary', action='store_true', help='Show sector summary')
    parser.add_argument('--stats', action='store_true', help='Show database statistics')
    parser.add_argument('--training-data', action='store_true', help='Export training data')
    
    args = parser.parse_args()
    
    querier = OpportunityQuerier()
    
    try:
        if args.stats:
            # Show database statistics
            stats = querier.get_opportunity_stats()
            print("📊 DATABASE STATISTICS")
            print("=" * 50)
            print(f"Total Opportunities: {stats['total_opportunities']:,}")
            print(f"Date Range: {stats['date_range']['min']} to {stats['date_range']['max']}")
            print(f"\nBy Type:")
            for opp_type, count in stats['by_type'].items():
                print(f"  {opp_type}: {count:,}")
            print(f"\nBy Sector:")
            for sector, count in stats['by_sector'].items():
                print(f"  {sector}: {count:,}")
            print(f"\nAverage Scores:")
            for score_type, value in stats['avg_scores'].items():
                print(f"  {score_type.title()}: {value}")
        
        elif args.sector_summary:
            # Show sector summary
            df = querier.get_sector_summary(args.days)
            print(f"📊 SECTOR SUMMARY (Last {args.days} days)")
            print("=" * 80)
            print(df.to_string(index=False))
        
        elif args.training_data:
            # Export training data
            df = querier.get_training_data(args.days)
            filename = f"training_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(filename, index=False)
            print(f"📁 Training data exported to: {filename}")
            print(f"Records: {len(df):,}")
        
        elif args.top:
            # Show top opportunities
            df = querier.get_top_opportunities(args.limit, args.min_score or 70)
            print(f"🏆 TOP OPPORTUNITIES (Score >= {args.min_score or 70})")
            print("=" * 100)
            if not df.empty:
                # Show key columns
                display_cols = ['ticker', 'sector', 'opportunity_type', 'overall_score', 
                              'current_price', 'target_price', 'risk_reward_ratio', 'scan_date']
                available_cols = [col for col in display_cols if col in df.columns]
                print(df[available_cols].to_string(index=False))
            else:
                print("No opportunities found")
        
        else:
            # Regular query
            df = querier.query(
                ticker=args.ticker,
                sector=args.sector,
                opportunity_type=args.type,
                min_score=args.min_score,
                max_score=args.max_score,
                days_back=args.days,
                limit=args.limit,
                format_output=args.format
            )
            
            if args.format == 'csv':
                print(df.to_csv(index=False))
            elif args.format == 'json':
                print(df.to_json(orient='records', indent=2))
            else:
                print(f"🔍 OPPORTUNITIES QUERY RESULTS")
                print("=" * 100)
                if not df.empty:
                    # Show key columns
                    display_cols = ['ticker', 'sector', 'opportunity_type', 'overall_score', 
                                  'current_price', 'target_price', 'risk_reward_ratio', 'scan_date']
                    available_cols = [col for col in display_cols if col in df.columns]
                    print(df[available_cols].to_string(index=False))
                    print(f"\nTotal: {len(df)} opportunities found")
                else:
                    print("No opportunities found matching criteria")
    
    finally:
        querier.close()


if __name__ == '__main__':
    main()
