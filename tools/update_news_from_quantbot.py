#!/usr/bin/env python3
"""
Update data/news.json with live data from QuantBot database

This script merges QuantBot's enhanced news feed with existing news data
and updates news.json for the market news app.
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta

def update_news_json():
    """Update news.json with QuantBot database data"""
    try:
        # Add QuantEngine to path
        quant_engine_path = Path(__file__).parent / 'QuantEngine'
        if str(quant_engine_path) not in sys.path:
            sys.path.insert(0, str(quant_engine_path))

        from data_broker import QuantBotDataBroker

        # Get news from QuantBot database
        broker = QuantBotDataBroker()
        quantbot_news = broker.get_news_feed(limit=20)

        # Load existing news.json if it exists
        existing_news = []
        if os.path.exists('data/news.json'):
            try:
                with open('data/news.json', 'r', encoding='utf-8') as f:
                    existing_news = json.load(f)
                    # Filter out old news (keep last 7 days)
                    cutoff_date = datetime.now() - timedelta(days=7)
                    existing_news = [
                        item for item in existing_news
                        if datetime.fromisoformat(item.get('published_date', '2000-01-01').replace('Z', '+00:00')) > cutoff_date
                    ]
            except Exception as e:
                print(f"Warning: Could not load existing news.json: {e}")
                existing_news = []

        # Convert QuantBot news to news.json format
        quantbot_formatted = []
        for item in quantbot_news:
            quantbot_formatted.append({
                'headline': item['headline'],
                'source': item['source'],
                'url': item.get('url', ''),
                'summary': item.get('summary', ''),
                'sentiment': item.get('sentiment', 'neutral'),
                'tickers': item.get('tickers', []),
                'type': item.get('type', 'financial_news'),
                'impact': item.get('impact', 'medium'),
                'published_date': item.get('published_date', datetime.now().isoformat())
            })

        # Merge news feeds (QuantBot news first, then existing news)
        merged_news = quantbot_formatted + existing_news

        # Remove duplicates based on headline
        seen_headlines = set()
        deduplicated_news = []
        for item in merged_news:
            headline = item.get('headline', '').strip()
            if headline and headline not in seen_headlines:
                seen_headlines.add(headline)
                deduplicated_news.append(item)

        # Sort by published date (most recent first)
        deduplicated_news.sort(
            key=lambda x: x.get('published_date', '2000-01-01'),
            reverse=True
        )

        # Keep only the most recent 50 items
        final_news = deduplicated_news[:50]

        # Save updated data/news.json
        with open('data/news.json', 'w', encoding='utf-8') as f:
            json.dump(final_news, f, indent=2, ensure_ascii=False)

        print(f"✅ Updated data/news.json with {len(final_news)} items")
        print(f"   • QuantBot news: {len(quantbot_formatted)}")
        print(f"   • Existing news: {len(existing_news)}")
        print(f"   • Total merged: {len(merged_news)}")
        print(f"   • After deduplication: {len(deduplicated_news)}")
        print(f"   • Final count: {len(final_news)}")

        # Show sample of updated news
        if final_news:
            print(f"\n📈 Latest headlines:")
            for i, item in enumerate(final_news[:3]):
                sentiment = item.get('sentiment', 'neutral')
                sentiment_emoji = {'bullish': '🟢', 'bearish': '🔴', 'neutral': '🟡'}.get(sentiment, '🟡')
                print(f"   {i+1}. {sentiment_emoji} {item['headline'][:60]}...")

        return True

    except Exception as e:
        print(f"❌ Failed to update data/news.json: {e}")
        import traceback
        traceback.print_exc()
        return False

def get_news_stats():
    """Get statistics about the current news feed"""
    try:
        if not os.path.exists('data/news.json'):
            print("❌ data/news.json not found")
            return

        with open('data/news.json', 'r', encoding='utf-8') as f:
            news_data = json.load(f)

        print(f"📊 News Feed Statistics:")
        print(f"   Total articles: {len(news_data)}")

        # Sentiment breakdown
        sentiments = {}
        sources = {}
        types = {}

        for item in news_data:
            sentiment = item.get('sentiment', 'unknown')
            source = item.get('source', 'unknown')
            news_type = item.get('type', 'unknown')

            sentiments[sentiment] = sentiments.get(sentiment, 0) + 1
            sources[source] = sources.get(source, 0) + 1
            types[news_type] = types.get(news_type, 0) + 1

        print(f"   Sentiment breakdown: {sentiments}")
        print(f"   Top sources: {dict(sorted(sources.items(), key=lambda x: x[1], reverse=True)[:5])}")
        print(f"   Content types: {types}")

        # Date range
        if news_data:
            dates = [item.get('published_date') for item in news_data if item.get('published_date')]
            if dates:
                dates.sort()
                print(f"   Date range: {dates[0]} to {dates[-1]}")

    except Exception as e:
        print(f"❌ Failed to get news stats: {e}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Update data/news.json with QuantBot data')
    parser.add_argument('--stats', action='store_true', help='Show news feed statistics instead of updating')

    args = parser.parse_args()

    if args.stats:
        get_news_stats()
    else:
        success = update_news_json()
        if success:
            print("\n🎉 data/news.json successfully updated with QuantBot data!")
            print("The market news app will now serve enhanced news with AI insights.")
        else:
            print("\n❌ Failed to update data/news.json")
            sys.exit(1)

