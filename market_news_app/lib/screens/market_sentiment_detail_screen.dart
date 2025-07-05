import 'package:flutter/material.dart';
import 'package:market_news_app/models/report_data.dart';

class MarketSentimentDetailScreen extends StatelessWidget {
  final MarketSentiment marketSentiment;

  const MarketSentimentDetailScreen({super.key, required this.marketSentiment});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Market Sentiment Analysis'),
        backgroundColor: _getSentimentColor(marketSentiment.sentiment),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            _buildOverallSentimentCard(context),
            const SizedBox(height: 20),
            _buildIndicatorAnalysis(context),
            const SizedBox(height: 20),
            _buildMarketContextCard(context),
            const SizedBox(height: 20),
            _buildTradingImplicationsCard(context),
          ],
        ),
      ),
    );
  }

  Widget _buildOverallSentimentCard(BuildContext context) {
    final sentimentColor = _getSentimentColor(marketSentiment.sentiment);
    final sentimentIcon = _getSentimentIcon(marketSentiment.sentiment);
    
    return Card(
      elevation: 4,
      color: sentimentColor.withOpacity(0.1),
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(sentimentIcon, color: sentimentColor, size: 32),
                const SizedBox(width: 12),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Text(
                        'Overall Market Sentiment',
                        style: TextStyle(
                          fontSize: 20,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      Text(
                        marketSentiment.sentiment,
                        style: TextStyle(
                          fontSize: 24,
                          fontWeight: FontWeight.w600,
                          color: sentimentColor,
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            ),
            const SizedBox(height: 16),
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: Colors.white.withOpacity(0.7),
                borderRadius: BorderRadius.circular(8),
                border: Border.all(color: sentimentColor.withOpacity(0.3)),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'What This Means:',
                    style: TextStyle(
                      fontSize: 16,
                      fontWeight: FontWeight.w600,
                      color: sentimentColor,
                    ),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    _getSentimentExplanation(marketSentiment.sentiment),
                    style: TextStyle(
                      fontSize: 14,
                      color: sentimentColor.withOpacity(0.8),
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildIndicatorAnalysis(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text(
          'Indicator Analysis',
          style: TextStyle(
            fontSize: 20,
            fontWeight: FontWeight.bold,
          ),
        ),
        const SizedBox(height: 16),
        ...marketSentiment.indicators.map((indicator) => _buildIndicatorCard(indicator)),
      ],
    );
  }

  Widget _buildIndicatorCard(Indicator indicator) {
    final indicatorColor = _getIndicatorColor(indicator.direction);
    final indicatorIcon = _getIndicatorIcon(indicator.direction);
    
    return Card(
      elevation: 2,
      margin: const EdgeInsets.only(bottom: 12),
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(indicatorIcon, color: indicatorColor, size: 24),
                const SizedBox(width: 12),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        indicator.name,
                        style: const TextStyle(
                          fontSize: 16,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                      Text(
                        '${indicator.ticker} • ${indicator.price}',
                        style: TextStyle(
                          fontSize: 14,
                          color: Colors.grey.shade600,
                        ),
                      ),
                    ],
                  ),
                ),
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                  decoration: BoxDecoration(
                    color: indicatorColor.withOpacity(0.1),
                    borderRadius: BorderRadius.circular(12),
                    border: Border.all(color: indicatorColor.withOpacity(0.3)),
                  ),
                  child: Text(
                    indicator.direction,
                    style: TextStyle(
                      fontSize: 12,
                      fontWeight: FontWeight.w600,
                      color: indicatorColor,
                    ),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 12),
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: indicatorColor.withOpacity(0.05),
                borderRadius: BorderRadius.circular(8),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'Market Impact:',
                    style: TextStyle(
                      fontSize: 14,
                      fontWeight: FontWeight.w600,
                      color: indicatorColor,
                    ),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    _getIndicatorExplanation(indicator.name, indicator.direction),
                    style: TextStyle(
                      fontSize: 13,
                      color: indicatorColor.withOpacity(0.8),
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildMarketContextCard(BuildContext context) {
    return Card(
      elevation: 4,
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(Icons.analytics_outlined, color: Colors.blue.shade700, size: 28),
                const SizedBox(width: 12),
                const Text(
                  'Market Context',
                  style: TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 16),
            _buildContextSection('Current Environment', _getMarketEnvironmentDescription()),
            const SizedBox(height: 12),
            _buildContextSection('Key Drivers', _getKeyDriversDescription()),
            const SizedBox(height: 12),
            _buildContextSection('Risk Factors', _getRiskFactorsDescription()),
          ],
        ),
      ),
    );
  }

  Widget _buildContextSection(String title, String description) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          title,
          style: TextStyle(
            fontSize: 15,
            fontWeight: FontWeight.w600,
            color: Colors.blue.shade700,
          ),
        ),
        const SizedBox(height: 4),
        Text(
          description,
          style: TextStyle(
            fontSize: 13,
            color: Colors.grey.shade700,
          ),
        ),
      ],
    );
  }

  Widget _buildTradingImplicationsCard(BuildContext context) {
    final sentimentColor = _getSentimentColor(marketSentiment.sentiment);
    
    return Card(
      elevation: 4,
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(Icons.trending_up, color: sentimentColor, size: 28),
                const SizedBox(width: 12),
                const Text(
                  'Trading Implications',
                  style: TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 16),
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: sentimentColor.withOpacity(0.1),
                borderRadius: BorderRadius.circular(8),
                border: Border.all(color: sentimentColor.withOpacity(0.3)),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'Strategy Recommendations:',
                    style: TextStyle(
                      fontSize: 15,
                      fontWeight: FontWeight.w600,
                      color: sentimentColor,
                    ),
                  ),
                  const SizedBox(height: 8),
                  ...(_getTradingImplications().map((implication) => Padding(
                    padding: const EdgeInsets.only(bottom: 4),
                    child: Row(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          '• ',
                          style: TextStyle(
                            fontSize: 14,
                            fontWeight: FontWeight.bold,
                            color: sentimentColor,
                          ),
                        ),
                        Expanded(
                          child: Text(
                            implication,
                            style: TextStyle(
                              fontSize: 13,
                              color: sentimentColor.withOpacity(0.8),
                            ),
                          ),
                        ),
                      ],
                    ),
                  ))),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Color _getSentimentColor(String sentiment) {
    if (sentiment.contains('BULLISH')) return Colors.green.shade700;
    if (sentiment.contains('BEARISH')) return Colors.red.shade700;
    return Colors.orange.shade700;
  }

  IconData _getSentimentIcon(String sentiment) {
    if (sentiment.contains('BULLISH')) return Icons.trending_up;
    if (sentiment.contains('BEARISH')) return Icons.trending_down;
    return Icons.trending_flat;
  }

  Color _getIndicatorColor(String direction) {
    if (direction.contains('⬆') || direction.contains('UP')) return Colors.green.shade600;
    if (direction.contains('⬇') || direction.contains('DOWN')) return Colors.red.shade600;
    return Colors.orange.shade600;
  }

  IconData _getIndicatorIcon(String direction) {
    if (direction.contains('⬆') || direction.contains('UP')) return Icons.arrow_upward;
    if (direction.contains('⬇') || direction.contains('DOWN')) return Icons.arrow_downward;
    return Icons.remove;
  }

  String _getSentimentExplanation(String sentiment) {
    if (sentiment.contains('BULLISH')) {
      return 'The market indicators suggest a positive outlook with upward momentum. This typically indicates investor confidence and potential for continued growth in the near term.';
    } else if (sentiment.contains('BEARISH')) {
      return 'The market indicators suggest a negative outlook with downward pressure. This typically indicates investor caution, potential selling pressure, and possible market declines ahead.';
    } else {
      return 'The market indicators are showing mixed signals with no clear directional bias. This suggests uncertainty and potential sideways movement in the near term.';
    }
  }

  String _getIndicatorExplanation(String name, String direction) {
    final isPositive = direction.contains('⬆') || direction.contains('UP');
    
    switch (name) {
      case 'S&P 500 Futures':
        return isPositive 
          ? 'Rising S&P 500 futures indicate optimism about large-cap U.S. stocks and overall market direction.'
          : 'Declining S&P 500 futures suggest pessimism about large-cap U.S. stocks and potential market weakness.';
      case 'Nasdaq 100 Futures':
        return isPositive 
          ? 'Rising Nasdaq futures indicate optimism about technology and growth stocks.'
          : 'Declining Nasdaq futures suggest concerns about technology and growth stocks, often leading market declines.';
      case 'VIX (Fear Index)':
        return isPositive 
          ? 'Rising VIX indicates increasing market fear and uncertainty, often accompanying market declines.'
          : 'Declining VIX suggests decreasing market fear and increasing investor confidence.';
      case '10-Year Treasury Yield':
        return isPositive 
          ? 'Rising yields may indicate economic optimism but can pressure stock valuations, especially growth stocks.'
          : 'Declining yields may suggest economic concerns but can support stock valuations through lower discount rates.';
      case 'US Dollar Index':
        return isPositive 
          ? 'A stronger dollar can pressure U.S. exports and multinational companies but may indicate economic strength.'
          : 'A weaker dollar can benefit U.S. exports and multinational companies but may indicate economic weakness.';
      default:
        return isPositive 
          ? 'This indicator is showing positive momentum, contributing to overall market optimism.'
          : 'This indicator is showing negative momentum, contributing to overall market pessimism.';
    }
  }

  String _getMarketEnvironmentDescription() {
    if (marketSentiment.sentiment.contains('BEARISH')) {
      return 'Current market conditions are characterized by risk-off sentiment, with investors showing increased caution and defensive positioning. Multiple indicators are signaling potential downward pressure.';
    } else if (marketSentiment.sentiment.contains('BULLISH')) {
      return 'Current market conditions are characterized by risk-on sentiment, with investors showing increased confidence and growth-oriented positioning. Multiple indicators are signaling potential upward momentum.';
    } else {
      return 'Current market conditions are mixed, with conflicting signals from various indicators. This suggests a period of uncertainty and potential consolidation.';
    }
  }

  String _getKeyDriversDescription() {
    // Count bearish vs bullish indicators
    final bearishCount = marketSentiment.indicators.where((i) => i.direction.contains('⬇') || i.direction.contains('DOWN')).length;
    final bullishCount = marketSentiment.indicators.where((i) => i.direction.contains('⬆') || i.direction.contains('UP')).length;
    
    if (bearishCount > bullishCount) {
      return 'Primary drivers include declining futures markets, elevated volatility concerns, and defensive positioning in bonds and currencies. These factors are creating headwinds for risk assets.';
    } else if (bullishCount > bearishCount) {
      return 'Primary drivers include rising futures markets, improving risk appetite, and optimistic positioning across asset classes. These factors are creating tailwinds for risk assets.';
    } else {
      return 'Market drivers are balanced between growth optimism and risk concerns, creating a tug-of-war between bulls and bears with no clear winner emerging.';
    }
  }

  String _getRiskFactorsDescription() {
    if (marketSentiment.sentiment.contains('BEARISH')) {
      return 'Key risks include potential market volatility, declining investor confidence, and possible economic headwinds. Consider defensive positioning and risk management strategies.';
    } else if (marketSentiment.sentiment.contains('BULLISH')) {
      return 'Key risks include potential overextension, complacency, and unexpected negative catalysts. Consider taking profits and maintaining diversification despite positive momentum.';
    } else {
      return 'Key risks include continued uncertainty, lack of clear direction, and potential for sudden moves in either direction. Consider maintaining balanced positioning and staying flexible.';
    }
  }

  List<String> _getTradingImplications() {
    if (marketSentiment.sentiment.contains('BEARISH')) {
      return [
        'Consider defensive strategies such as bear put spreads or protective puts',
        'Focus on high-quality dividend stocks and defensive sectors',
        'Reduce exposure to high-beta and growth stocks',
        'Consider hedging strategies for existing long positions',
        'Monitor support levels for potential entry points'
      ];
    } else if (marketSentiment.sentiment.contains('BULLISH')) {
      return [
        'Consider bullish strategies such as bull call spreads or covered calls',
        'Focus on growth stocks and cyclical sectors',
        'Look for breakout opportunities in trending stocks',
        'Consider momentum-based strategies',
        'Monitor resistance levels for potential profit-taking'
      ];
    } else {
      return [
        'Consider range-bound strategies such as straddles or strangles',
        'Focus on stocks with strong fundamentals regardless of sector',
        'Look for oversold/overbought conditions for mean reversion',
        'Consider income-generating strategies like covered calls',
        'Monitor key support and resistance levels for directional moves'
      ];
    }
  }
} 