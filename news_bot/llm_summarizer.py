"""
LLM-based news summarizer using Anthropic Claude
Generates actionable intelligence from news articles
"""
import anthropic
from typing import Dict, List, Optional
import logging
import json
from . import config

logger = logging.getLogger(__name__)


class LLMSummarizer:
    """Generate intelligent summaries using Claude"""
    
    def __init__(self):
        self.api_key = config.ANTHROPIC_API_KEY
        self.client = None
        
        if self.api_key:
            try:
                self.client = anthropic.Anthropic(api_key=self.api_key)
                logger.info("Claude API initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Claude API: {e}")
                self.client = None
        else:
            logger.warning("Anthropic API key not configured - LLM summarization disabled")
    
    def summarize_article(self, article: Dict) -> Optional[Dict]:
        """
        Summarize a single article with actionable intelligence
        Returns dict with: summary, affected_tickers, time_sensitivity, actionable_insight, sentiment
        """
        if not self.client:
            return self._fallback_summary(article)
        
        try:
            headline = article.get('headline', '')
            text = article.get('raw_summary', '') or article.get('full_text', '')
            mentioned_tickers = article.get('mentioned_tickers', [])
            
            prompt = f"""Analyze this financial news article and provide actionable trading intelligence:

Headline: {headline}

Content: {text}

Mentioned Tickers: {', '.join(mentioned_tickers) if mentioned_tickers else 'None specified'}

Provide your analysis in JSON format with these fields:
1. summary: A concise 2-3 sentence summary focused on market impact
2. affected_tickers: List of stock tickers that will be most affected (include sector if relevant)
3. time_sensitivity: One of: "pre-market", "intraday", "multi-day", "long-term"
4. actionable_insight: Specific trading implication (bullish/bearish bias and why)
5. sentiment: One of: "bullish", "bearish", "neutral"

Focus on what matters for traders making decisions in the morning."""

            message = self.client.messages.create(
                model=config.LLM_MODEL,
                max_tokens=config.LLM_MAX_TOKENS,
                temperature=config.LLM_TEMPERATURE,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            response_text = message.content[0].text
            
            # Try to parse JSON response
            try:
                result = json.loads(response_text)
            except json.JSONDecodeError:
                # If not JSON, try to extract structured data
                result = self._parse_unstructured_response(response_text, article)
            
            return result
        
        except Exception as e:
            logger.error(f"Error in LLM summarization: {e}")
            return self._fallback_summary(article)
    
    def summarize_batch(self, articles: List[Dict], max_articles: int = 50) -> List[Dict]:
        """
        Summarize a batch of articles
        Returns list of summaries matched to articles
        """
        summaries = []
        
        # Limit batch size to avoid costs
        articles_to_process = articles[:max_articles]
        
        logger.info(f"Starting LLM summarization for {len(articles_to_process)} articles")
        
        for i, article in enumerate(articles_to_process):
            try:
                summary = self.summarize_article(article)
                
                if summary:
                    summary['article_id'] = article.get('id')
                    summaries.append(summary)
                
                # Log progress
                if (i + 1) % 10 == 0:
                    logger.info(f"Summarized {i + 1}/{len(articles_to_process)} articles")
            
            except Exception as e:
                logger.error(f"Error summarizing article {article.get('id')}: {e}")
                continue
        
        logger.info(f"Completed {len(summaries)} summaries")
        return summaries
    
    def _fallback_summary(self, article: Dict) -> Dict:
        """
        Generate a basic summary when LLM is not available
        Uses rule-based approach
        """
        headline = article.get('headline', '')
        text = article.get('raw_summary', '')
        mentioned_tickers = article.get('mentioned_tickers', [])
        importance = article.get('importance_level', 2)
        
        # Determine sentiment from keywords
        sentiment = 'neutral'
        text_lower = (headline + ' ' + text).lower()
        
        positive_words = ['beat', 'surge', 'gains', 'rally', 'upgrade', 'approval', 'growth', 'record']
        negative_words = ['miss', 'plunge', 'falls', 'downgrade', 'reject', 'decline', 'loss', 'crash']
        
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count > neg_count:
            sentiment = 'bullish'
        elif neg_count > pos_count:
            sentiment = 'bearish'
        
        # Determine time sensitivity
        time_sensitivity = 'multi-day'
        if any(word in text_lower for word in ['breaking', 'just', 'alert', 'now']):
            time_sensitivity = 'intraday'
        elif importance >= 4:
            time_sensitivity = 'pre-market'
        
        # Generate basic insight
        actionable_insight = f"Monitor {', '.join(mentioned_tickers[:3])} for market reaction" if mentioned_tickers else "General market awareness"
        
        if importance >= 4:
            actionable_insight = f"High priority - {actionable_insight}"
        
        return {
            'summary': text[:300] if text else headline,
            'affected_tickers': mentioned_tickers[:5],
            'time_sensitivity': time_sensitivity,
            'actionable_insight': actionable_insight,
            'sentiment': sentiment
        }
    
    def _parse_unstructured_response(self, text: str, article: Dict) -> Dict:
        """Parse non-JSON LLM response into structured format"""
        # Basic parsing of free-form text
        lines = text.split('\n')
        
        result = {
            'summary': '',
            'affected_tickers': article.get('mentioned_tickers', []),
            'time_sensitivity': 'multi-day',
            'actionable_insight': '',
            'sentiment': 'neutral'
        }
        
        # Try to extract key information
        for line in lines:
            line_lower = line.lower()
            
            if 'summary' in line_lower and ':' in line:
                result['summary'] = line.split(':', 1)[1].strip()
            elif 'sentiment' in line_lower and ':' in line:
                sentiment_text = line.split(':', 1)[1].strip().lower()
                if 'bull' in sentiment_text:
                    result['sentiment'] = 'bullish'
                elif 'bear' in sentiment_text:
                    result['sentiment'] = 'bearish'
            elif 'time' in line_lower and ':' in line:
                time_text = line.split(':', 1)[1].strip().lower()
                if 'pre-market' in time_text:
                    result['time_sensitivity'] = 'pre-market'
                elif 'intraday' in time_text or 'today' in time_text:
                    result['time_sensitivity'] = 'intraday'
            elif 'insight' in line_lower or 'action' in line_lower:
                if ':' in line:
                    result['actionable_insight'] = line.split(':', 1)[1].strip()
        
        # Use fallback if parsing failed
        if not result['summary']:
            result['summary'] = article.get('raw_summary', '')[:300]
        
        if not result['actionable_insight']:
            result['actionable_insight'] = "Monitor for market developments"
        
        return result


if __name__ == "__main__":
    # Test the summarizer
    logging.basicConfig(level=logging.INFO)
    
    test_article = {
        'id': 1,
        'headline': 'Federal Reserve Announces 50 Basis Point Rate Cut',
        'raw_summary': 'The Federal Reserve announced a surprise 50 basis point interest rate cut today, citing concerns about economic growth. Markets rallied on the news with technology stocks leading gains.',
        'mentioned_tickers': ['SPY', 'QQQ', 'AAPL', 'MSFT'],
        'importance_level': 5
    }
    
    summarizer = LLMSummarizer()
    
    if summarizer.client:
        print("Testing Claude summarization...")
        summary = summarizer.summarize_article(test_article)
        print(f"\nSummary: {summary}")
    else:
        print("Testing fallback summarization...")
        summary = summarizer._fallback_summary(test_article)
        print(f"\nFallback Summary: {summary}")


