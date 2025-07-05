import 'package:flutter/material.dart';
import 'package:market_news_app/models/economic_event.dart';
import 'package:market_news_app/models/vix_data.dart';
import 'package:market_news_app/services/fmp_api_service.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';
import 'package:market_news_app/widgets/daily_strategy_guide.dart';
import 'package:market_news_app/models/report_data.dart';

class MarketInsightsScreen extends StatefulWidget {
  final ReportData reportData;

  const MarketInsightsScreen({super.key, required this.reportData});

  @override
  State<MarketInsightsScreen> createState() => _MarketInsightsScreenState();
}

class _MarketInsightsScreenState extends State<MarketInsightsScreen> {
  List<VixData> _vixData = [];
  String? _newsError;

  @override
  void initState() {
    super.initState();
    _vixData = widget.reportData.vixData;
    _newsError = "Economic calendar data is not available on the current API plan.";
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Market Insights'),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const DailyStrategyGuide(),
            const SizedBox(height: 20),
            _buildVolatilityCard(context),
            const SizedBox(height: 20),
            _buildNewsCard(context),
          ],
        ),
      ),
    );
  }

  Widget _buildVolatilityCard(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Volatility Forecast', style: Theme.of(context).textTheme.headlineSmall),
            const SizedBox(height: 10),
            if (_vixData.isNotEmpty)
              Text('Today\'s VIX: ${_vixData.first.close.toStringAsFixed(2)}', style: Theme.of(context).textTheme.titleLarge),
            const SizedBox(height: 5),
            if (_vixData.length > 2)
              Text('Next 3 Days: ${_vixData.sublist(0, 3).map((v) => v.close.toStringAsFixed(2)).join(', ')}', style: Theme.of(context).textTheme.bodyMedium),
          ],
        ),
      ),
    );
  }

  Widget _buildNewsCard(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Upcoming News', style: Theme.of(context).textTheme.headlineSmall),
            const SizedBox(height: 10),
            Text(
              _newsError!,
              style: const TextStyle(color: Colors.orange),
            ),
          ],
        ),
      ),
    );
  }
}

class MarketInsightsScreenPlaceholder extends StatelessWidget {
  const MarketInsightsScreenPlaceholder({super.key});

  @override
  Widget build(BuildContext context) {
    // This would normally get the report data from a provider or state management
    // For now, just show a placeholder
    return Scaffold(
      appBar: AppBar(title: const Text('Market Insights')),
      body: const Center(
        child: Text('Market Insights coming soon!'),
      ),
    );
  }
}
