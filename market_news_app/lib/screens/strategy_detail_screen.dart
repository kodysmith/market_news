import 'package:flutter/material.dart';
import 'package:market_news_app/models/report_data.dart';

class StrategyDetailScreen extends StatelessWidget {
  final TopStrategy strategy;

  const StrategyDetailScreen({super.key, required this.strategy});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(strategy.name),
        backgroundColor: _getStrategyColor(strategy.name),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            _buildHeader(context),
            const SizedBox(height: 20),
            _buildTickerList(context),
            const SizedBox(height: 20),
            _buildEducationSection(context),
          ],
        ),
      ),
    );
  }

  Widget _buildHeader(BuildContext context) {
    return Card(
      elevation: 4,
      color: _getStrategyColor(strategy.name).withOpacity(0.08),
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(_getStrategyIcon(strategy.name), color: _getStrategyColor(strategy.name), size: 28),
                const SizedBox(width: 12),
                Expanded(
                  child: Text(
                    strategy.name,
                    style: TextStyle(
                      fontSize: 22,
                      fontWeight: FontWeight.bold,
                      color: _getStrategyColor(strategy.name),
                    ),
                  ),
                ),
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
                  decoration: BoxDecoration(
                    color: _getStrategyColor(strategy.name).withOpacity(0.15),
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Text(
                    'Score: ${strategy.score.toStringAsFixed(1)}',
                    style: TextStyle(
                      fontSize: 14,
                      fontWeight: FontWeight.w600,
                      color: _getStrategyColor(strategy.name),
                    ),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 12),
            Text(
              strategy.description,
              style: TextStyle(fontSize: 15, color: Colors.grey.shade800),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildTickerList(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text(
          'Top Stocks for This Strategy',
          style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
        ),
        const SizedBox(height: 12),
        ...strategy.topTickers.map((ticker) => _buildTickerCard(ticker)),
      ],
    );
  }

  Widget _buildTickerCard(StrategyTicker ticker) {
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
                    color: _getStrategyColor(strategy.name),
                    borderRadius: BorderRadius.circular(16),
                  ),
                  child: Text(
                    ticker.ticker,
                    style: const TextStyle(
                      fontSize: 14,
                      fontWeight: FontWeight.bold,
                      color: Colors.white,
                    ),
                  ),
                ),
                const SizedBox(width: 12),
                Text(
                  'Score: ${ticker.score.toStringAsFixed(1)}',
                  style: TextStyle(
                    fontSize: 14,
                    fontWeight: FontWeight.w600,
                    color: _getStrategyColor(strategy.name),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 10),
            Text(
              'Reason: ${ticker.reason}',
              style: TextStyle(fontSize: 13, color: Colors.grey.shade800),
            ),
            const SizedBox(height: 8),
            _buildSetupSection(ticker.setup),
          ],
        ),
      ),
    );
  }

  Widget _buildSetupSection(Map<String, dynamic> setup) {
    List<Widget> rows = [];
    setup.forEach((key, value) {
      rows.add(Row(
        children: [
          Text(
            '${_formatKey(key)}: ',
            style: const TextStyle(fontWeight: FontWeight.w600, fontSize: 13),
          ),
          Text(
            value.toString(),
            style: const TextStyle(fontSize: 13),
          ),
        ],
      ));
    });
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text('Trade Setup:', style: TextStyle(fontWeight: FontWeight.bold, fontSize: 14)),
        const SizedBox(height: 4),
        ...rows,
      ],
    );
  }

  Widget _buildEducationSection(BuildContext context) {
    return Card(
      elevation: 2,
      color: Colors.blueGrey.shade50,
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'When to Use This Strategy',
              style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16),
            ),
            const SizedBox(height: 8),
            Text(_getStrategyEducation(strategy.name), style: const TextStyle(fontSize: 13)),
          ],
        ),
      ),
    );
  }

  Color _getStrategyColor(String name) {
    if (name.toLowerCase().contains('scalp')) return Colors.blue.shade700;
    if (name.toLowerCase().contains('covered')) return Colors.orange.shade700;
    if (name.toLowerCase().contains('bull')) return Colors.green.shade700;
    if (name.toLowerCase().contains('bear')) return Colors.red.shade700;
    return Colors.purple.shade700;
  }

  IconData _getStrategyIcon(String name) {
    if (name.toLowerCase().contains('scalp')) return Icons.show_chart;
    if (name.toLowerCase().contains('covered')) return Icons.call;
    if (name.toLowerCase().contains('bull')) return Icons.trending_up;
    if (name.toLowerCase().contains('bear')) return Icons.trending_down;
    return Icons.auto_graph;
  }

  String _formatKey(String key) {
    return key.replaceAll('_', ' ').split(' ').map((w) => w[0].toUpperCase() + w.substring(1)).join(' ');
  }

  String _getStrategyEducation(String name) {
    if (name.toLowerCase().contains('scalp')) {
      return 'Gamma scalping is best when options are cheap (low IV/RV ratio) and volatility may expand. It involves buying straddles/strangles and actively hedging with shares.';
    } else if (name.toLowerCase().contains('covered')) {
      return 'Covered calls are ideal when you expect neutral to slightly bullish movement. They generate income from call premiums while holding the underlying stock.';
    } else if (name.toLowerCase().contains('bull')) {
      return 'Bull put spreads are used when you expect the stock to stay above the short put strike. They offer limited risk and reward, and work best in bullish or neutral markets.';
    } else if (name.toLowerCase().contains('bear')) {
      return 'Bear put spreads are used when you expect the stock to decline. They offer limited risk and reward, and work best in bearish or neutral markets.';
    } else {
      return 'This strategy is best used when its risk/reward profile matches your market outlook and risk tolerance.';
    }
  }
} 