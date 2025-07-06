import 'package:flutter/material.dart';
import 'package:market_news_app/models/vix_data.dart';
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
    final report = widget.reportData;
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
            _buildSectionCard(
              context,
              title: 'Earnings Calendar',
              child: _buildEarningsList(report.earningsCalendar),
            ),
            const SizedBox(height: 20),
            _buildSectionCard(
              context,
              title: 'Top Gainers',
              child: _buildMoversList(report.topGainers, gainers: true),
            ),
            const SizedBox(height: 20),
            _buildSectionCard(
              context,
              title: 'Top Losers',
              child: _buildMoversList(report.topLosers, gainers: false),
            ),
            const SizedBox(height: 20),
            _buildSectionCard(
              context,
              title: 'Major Indices',
              child: _buildIndicesList(report.indices),
            ),
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

  Widget _buildSectionCard(BuildContext context, {required String title, required Widget child}) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(title, style: Theme.of(context).textTheme.headlineSmall),
            const SizedBox(height: 10),
            child,
          ],
        ),
      ),
    );
  }

  Widget _buildEarningsList(List<Map<String, dynamic>> earnings) {
    if (earnings.isEmpty) {
      return const Text('No earnings data available.');
    }
    return Column(
      children: earnings.take(5).map((e) => ListTile(
        leading: const Icon(Icons.calendar_today, color: Colors.amber),
        title: Text(e['symbol'] ?? '', style: const TextStyle(fontWeight: FontWeight.bold)),
        subtitle: Text('Date: ${e['date'] ?? ''}  Time: ${e['time'] ?? ''}'),
        trailing: e['epsEstimated'] != null ? Text('Est. EPS: ${e['epsEstimated']}', style: const TextStyle(color: Colors.blue)) : null,
        onTap: () {
          showDialog(
            context: context,
            builder: (context) => AlertDialog(
              title: Text(e['symbol'] ?? ''),
              content: Column(
                mainAxisSize: MainAxisSize.min,
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text('Date: ${e['date'] ?? ''}'),
                  Text('Time: ${e['time'] ?? ''}'),
                  Text('Fiscal End: ${e['fiscalDateEnding'] ?? ''}'),
                  if (e['eps'] != null) Text('EPS: ${e['eps']}'),
                  if (e['epsEstimated'] != null) Text('EPS Est.: ${e['epsEstimated']}'),
                  if (e['revenue'] != null) Text('Revenue: ${e['revenue']}'),
                  if (e['revenueEstimated'] != null) Text('Revenue Est.: ${e['revenueEstimated']}'),
                ],
              ),
              actions: [
                TextButton(onPressed: () => Navigator.of(context).pop(), child: const Text('Close')),
              ],
            ),
          );
        },
      )).toList(),
    );
  }

  Widget _buildMoversList(List<Map<String, dynamic>> movers, {required bool gainers}) {
    if (movers.isEmpty) {
      return Text(gainers ? 'No gainers data.' : 'No losers data.');
    }
    return Column(
      children: movers.take(5).map((m) => ListTile(
        leading: Icon(gainers ? Icons.trending_up : Icons.trending_down, color: gainers ? Colors.green : Colors.red),
        title: Text(m['symbol'] ?? '', style: const TextStyle(fontWeight: FontWeight.bold)),
        subtitle: Text(m['name'] ?? ''),
        trailing: Text('${m['changesPercentage']?.toStringAsFixed(2) ?? ''}%', style: TextStyle(color: gainers ? Colors.green : Colors.red, fontWeight: FontWeight.bold)),
        onTap: () {
          showDialog(
            context: context,
            builder: (context) => AlertDialog(
              title: Text(m['symbol'] ?? ''),
              content: Column(
                mainAxisSize: MainAxisSize.min,
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text('Name: ${m['name'] ?? ''}'),
                  Text('Price: ${m['price'] ?? ''}'),
                  Text('Change: ${m['change'] ?? ''}'),
                  Text('Change %: ${m['changesPercentage'] ?? ''}%'),
                ],
              ),
              actions: [
                TextButton(onPressed: () => Navigator.of(context).pop(), child: const Text('Close')),
              ],
            ),
          );
        },
      )).toList(),
    );
  }

  Widget _buildIndicesList(List<Map<String, dynamic>> indices) {
    if (indices.isEmpty) {
      return const Text('No indices data available.');
    }
    return Column(
      children: indices.take(5).map((i) => ListTile(
        leading: const Icon(Icons.show_chart, color: Colors.purple),
        title: Text(i['name'] ?? '', style: const TextStyle(fontWeight: FontWeight.bold)),
        subtitle: Text(i['symbol'] ?? ''),
        trailing: Text('Price: ${i['price'] ?? ''}', style: const TextStyle(color: Colors.purple)),
        onTap: () {
          showDialog(
            context: context,
            builder: (context) => AlertDialog(
              title: Text(i['name'] ?? ''),
              content: Column(
                mainAxisSize: MainAxisSize.min,
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text('Symbol: ${i['symbol'] ?? ''}'),
                  Text('Price: ${i['price'] ?? ''}'),
                  Text('Change: ${i['change'] ?? ''}'),
                  Text('Change %: ${i['changesPercentage'] ?? ''}%'),
                  Text('Day Low: ${i['dayLow'] ?? ''}'),
                  Text('Day High: ${i['dayHigh'] ?? ''}'),
                  Text('Year Low: ${i['yearLow'] ?? ''}'),
                  Text('Year High: ${i['yearHigh'] ?? ''}'),
                ],
              ),
              actions: [
                TextButton(onPressed: () => Navigator.of(context).pop(), child: const Text('Close')),
              ],
            ),
          );
        },
      )).toList(),
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
