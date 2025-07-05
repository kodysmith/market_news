import 'package:flutter/material.dart';
import 'package:market_news_app/models/report_data.dart';

class GammaScalpingScreen extends StatelessWidget {
  final GammaAnalysis gammaAnalysis;

  const GammaScalpingScreen({super.key, required this.gammaAnalysis});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Gamma Scalping Analysis'),
        backgroundColor: Colors.blue.shade700,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            _buildMarketOverview(context),
            const SizedBox(height: 20),
            _buildStrategyGuide(context),
            const SizedBox(height: 20),
            _buildTickerAnalysis(context),
          ],
        ),
      ),
    );
  }

  Widget _buildMarketOverview(BuildContext context) {
    Color backgroundColor;
    Color textColor;
    IconData icon;
    String title;
    
    switch (gammaAnalysis.marketRecommendation) {
      case 'GAMMA_SCALPING_FAVORED':
        backgroundColor = Colors.blue.shade50;
        textColor = Colors.blue.shade800;
        icon = Icons.trending_up;
        title = 'Gamma Scalping Favored';
        break;
      case 'PREMIUM_SELLING_FAVORED':
        backgroundColor = Colors.orange.shade50;
        textColor = Colors.orange.shade800;
        icon = Icons.trending_down;
        title = 'Premium Selling Favored';
        break;
      default:
        backgroundColor = Colors.purple.shade50;
        textColor = Colors.purple.shade800;
        icon = Icons.balance;
        title = 'Mixed Conditions';
    }

    return Card(
      elevation: 4,
      color: backgroundColor,
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(icon, color: textColor, size: 32),
                const SizedBox(width: 12),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'Market Overview',
                        style: TextStyle(
                          fontSize: 20,
                          fontWeight: FontWeight.bold,
                          color: textColor,
                        ),
                      ),
                      Text(
                        title,
                        style: TextStyle(
                          fontSize: 16,
                          color: textColor.withOpacity(0.8),
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            ),
            const SizedBox(height: 16),
            Row(
              children: [
                Expanded(
                  child: _buildMetricCard(
                    'Gamma Score',
                    '${gammaAnalysis.avgGammaScore.toStringAsFixed(1)}/100',
                    textColor,
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: _buildMetricCard(
                    'Gamma Scalping',
                    '${gammaAnalysis.gammaScalpingCount} stocks',
                    textColor,
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: _buildMetricCard(
                    'Premium Selling',
                    '${gammaAnalysis.premiumSellingCount} stocks',
                    textColor,
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildMetricCard(String label, String value, Color color) {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.7),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: color.withOpacity(0.3)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            label,
            style: TextStyle(
              fontSize: 12,
              color: color.withOpacity(0.7),
            ),
          ),
          const SizedBox(height: 4),
          Text(
            value,
            style: TextStyle(
              fontSize: 16,
              fontWeight: FontWeight.bold,
              color: color,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildStrategyGuide(BuildContext context) {
    return Card(
      elevation: 4,
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(Icons.lightbulb_outline, color: Colors.amber.shade700, size: 28),
                const SizedBox(width: 12),
                const Text(
                  'Strategy Guide',
                  style: TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 16),
            _buildStrategySection(
              'Gamma Scalping',
              'Buy straddles/strangles and delta hedge with shares',
              'Best when options are cheap relative to realized volatility',
              Colors.blue.shade600,
              Icons.trending_up,
            ),
            const SizedBox(height: 12),
            _buildStrategySection(
              'Premium Selling',
              'Sell straddles/strangles to collect premium',
              'Best when options are expensive relative to realized volatility',
              Colors.orange.shade600,
              Icons.trending_down,
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildStrategySection(String title, String strategy, String description, Color color, IconData icon) {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: color.withOpacity(0.1),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: color.withOpacity(0.3)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(icon, color: color, size: 20),
              const SizedBox(width: 8),
              Text(
                title,
                style: TextStyle(
                  fontSize: 16,
                  fontWeight: FontWeight.w600,
                  color: color,
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          Text(
            strategy,
            style: TextStyle(
              fontSize: 14,
              fontWeight: FontWeight.w500,
              color: color.withOpacity(0.8),
            ),
          ),
          const SizedBox(height: 4),
          Text(
            description,
            style: TextStyle(
              fontSize: 12,
              color: color.withOpacity(0.7),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildTickerAnalysis(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text(
          'Individual Ticker Analysis',
          style: TextStyle(
            fontSize: 20,
            fontWeight: FontWeight.bold,
          ),
        ),
        const SizedBox(height: 16),
        ...gammaAnalysis.individualAnalysis.map((analysis) => _buildTickerCard(analysis)),
      ],
    );
  }

  Widget _buildTickerCard(TickerGammaAnalysis analysis) {
    Color backgroundColor;
    Color textColor;
    IconData icon;
    
    switch (analysis.recommendation) {
      case 'GAMMA_SCALPING':
        backgroundColor = Colors.blue.shade50;
        textColor = Colors.blue.shade800;
        icon = Icons.trending_up;
        break;
      case 'PREMIUM_SELLING':
        backgroundColor = Colors.orange.shade50;
        textColor = Colors.orange.shade800;
        icon = Icons.trending_down;
        break;
      default:
        backgroundColor = Colors.grey.shade50;
        textColor = Colors.grey.shade800;
        icon = Icons.balance;
    }

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
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                  decoration: BoxDecoration(
                    color: textColor,
                    borderRadius: BorderRadius.circular(16),
                  ),
                  child: Text(
                    analysis.ticker,
                    style: const TextStyle(
                      fontSize: 14,
                      fontWeight: FontWeight.bold,
                      color: Colors.white,
                    ),
                  ),
                ),
                const SizedBox(width: 12),
                Icon(icon, color: textColor, size: 20),
                const SizedBox(width: 4),
                Text(
                  analysis.recommendation.replaceAll('_', ' '),
                  style: TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.w600,
                    color: textColor,
                  ),
                ),
                const Spacer(),
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                  decoration: BoxDecoration(
                    color: backgroundColor,
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Text(
                    'Score: ${analysis.gammaScore.toStringAsFixed(0)}',
                    style: TextStyle(
                      fontSize: 12,
                      fontWeight: FontWeight.w600,
                      color: textColor,
                    ),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 12),
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: backgroundColor,
                borderRadius: BorderRadius.circular(8),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'Strategy: ${analysis.strategy}',
                    style: TextStyle(
                      fontSize: 14,
                      fontWeight: FontWeight.w500,
                      color: textColor,
                    ),
                  ),
                  const SizedBox(height: 8),
                  Text(
                    'Key Factors:',
                    style: TextStyle(
                      fontSize: 13,
                      fontWeight: FontWeight.w600,
                      color: textColor,
                    ),
                  ),
                  const SizedBox(height: 4),
                  ...analysis.reasons.take(2).map((reason) => Padding(
                    padding: const EdgeInsets.only(bottom: 2),
                    child: Text(
                      '• $reason',
                      style: TextStyle(
                        fontSize: 12,
                        color: textColor.withOpacity(0.8),
                      ),
                    ),
                  )),
                ],
              ),
            ),
            const SizedBox(height: 12),
            Row(
              children: [
                Expanded(
                  child: _buildAnalysisMetric(
                    'IV/RV Ratio',
                    analysis.analysis.ivRvRatio.toStringAsFixed(2),
                    textColor,
                  ),
                ),
                const SizedBox(width: 8),
                Expanded(
                  child: _buildAnalysisMetric(
                    'VIX %ile',
                    '${analysis.analysis.vixPercentile.toStringAsFixed(0)}%',
                    textColor,
                  ),
                ),
                const SizedBox(width: 8),
                Expanded(
                  child: _buildAnalysisMetric(
                    'Vol Accel',
                    analysis.analysis.rvAcceleration.toStringAsFixed(2),
                    textColor,
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildAnalysisMetric(String label, String value, Color color) {
    return Container(
      padding: const EdgeInsets.all(8),
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.7),
        borderRadius: BorderRadius.circular(6),
        border: Border.all(color: color.withOpacity(0.2)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            label,
            style: TextStyle(
              fontSize: 10,
              color: color.withOpacity(0.7),
            ),
          ),
          const SizedBox(height: 2),
          Text(
            value,
            style: TextStyle(
              fontSize: 14,
              fontWeight: FontWeight.bold,
              color: color,
            ),
          ),
        ],
      ),
    );
  }
} 