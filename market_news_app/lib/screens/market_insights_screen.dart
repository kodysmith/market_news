import 'package:flutter/material.dart';
import 'package:market_news_app/models/economic_event.dart';
import 'package:market_news_app/models/vix_data.dart';
import 'package:market_news_app/services/fmp_api_service.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';
import 'package:market_news_app/widgets/daily_strategy_guide.dart';

class MarketInsightsScreen extends StatefulWidget {
  const MarketInsightsScreen({super.key});

  @override
  State<MarketInsightsScreen> createState() => _MarketInsightsScreenState();
}

class _MarketInsightsScreenState extends State<MarketInsightsScreen> {
  late final FmpApiService _fmpApiService;
  List<VixData> _vixData = [];
  List<EconomicEvent> _economicEvents = [];
  bool _isLoading = true;
  String? _error;
  String? _newsError;

  @override
  void initState() {
    super.initState();
    _fmpApiService = FmpApiService(dotenv.env['FMP_API_KEY']!);
    _fetchData();
  }

  Future<void> _fetchData() async {
    try {
      final vix = await _fmpApiService.fetchVixData();
      setState(() {
        _vixData = vix;
      });
    } catch (e) {
      setState(() {
        _error = e.toString();
      });
    }

    try {
      final events = await _fmpApiService.fetchEconomicCalendar();
      setState(() {
        _economicEvents = events;
      });
    } catch (e) {
      setState(() {
        _newsError = e.toString();
      });
    }

    setState(() {
      _isLoading = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Market Insights'),
      ),
      body: _isLoading
          ? const Center(child: CircularProgressIndicator())
          : _error != null
              ? Center(child: Text('Error: $_error'))
              : SingleChildScrollView(
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
    if (_newsError != null) {
      return Card(
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text('Upcoming News', style: Theme.of(context).textTheme.headlineSmall),
              const SizedBox(height: 10),
              Text(
                'Could not load economic events: $_newsError',
                style: const TextStyle(color: Colors.red),
              ),
            ],
          ),
        ),
      );
    }

    final today = DateTime.now();
    final todayEvents = _economicEvents.where((e) => DateTime.parse(e.date).day == today.day).toList();
    final weekEvents = _economicEvents.where((e) => DateTime.parse(e.date).day != today.day).toList();

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Upcoming News', style: Theme.of(context).textTheme.headlineSmall),
            const SizedBox(height: 10),
            Text('Today\'s High-Impact Events', style: Theme.of(context).textTheme.titleLarge),
            if (todayEvents.isNotEmpty)
              ...todayEvents.map((event) => ListTile(
                    title: Text(event.event),
                    subtitle: Text(event.impact),
                  ))
            else
              const Text('No high-impact events scheduled for today.'),
            const SizedBox(height: 20),
            Text('This Week\'s Outlook', style: Theme.of(context).textTheme.titleLarge),
            if (weekEvents.isNotEmpty)
              ...weekEvents.map((event) => ListTile(
                    title: Text(event.event),
                    subtitle: Text('${event.date} - ${event.impact}'),
                  ))
            else
              const Text('No other major events scheduled for this week.'),
          ],
        ),
      ),
    );
  }
}
