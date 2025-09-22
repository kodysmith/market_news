import 'package:flutter/material.dart';
import 'package:market_news_app/models/vix_data.dart';
import 'package:market_news_app/widgets/daily_strategy_guide.dart';
import 'package:market_news_app/models/report_data.dart';
import 'dart:math' as math;
import 'package:http/http.dart' as http;
import 'dart:convert';
import '../main.dart' show apiBaseUrl, apiSecretKey;

class TradeRecommendation {
  final String symbol;
  final int ivRank;
  final double currentIv;
  final String trend;
  final String volatilityLevel;
  final String recommendation;
  final String rationale;

  TradeRecommendation({
    required this.symbol,
    required this.ivRank,
    required this.currentIv,
    required this.trend,
    required this.volatilityLevel,
    required this.recommendation,
    required this.rationale,
  });

  factory TradeRecommendation.fromJson(Map<String, dynamic> json) {
    return TradeRecommendation(
      symbol: json['symbol'],
      ivRank: json['iv_rank'],
      currentIv: (json['current_iv'] as num).toDouble(),
      trend: json['trend'],
      volatilityLevel: json['volatility_level'],
      recommendation: json['recommendation'],
      rationale: json['rationale'],
    );
  }
}

class MarketInsightsScreen extends StatefulWidget {
  final ReportData reportData;

  const MarketInsightsScreen({super.key, required this.reportData});

  @override
  State<MarketInsightsScreen> createState() => _MarketInsightsScreenState();
}

class _MarketInsightsScreenState extends State<MarketInsightsScreen> {
  List<VixData> _vixData = [];
  String? _newsError;
  String _selectedStrategy = 'All';
  final List<String> _strategyOptions = [
    'All',
    'Covered Call',
    'Bull Put Spread',
    'Bear Call Spread',
    'Long Straddle',
    'Long Strangle',
    // Add more as needed
  ];
  int _currentPage = 0;
  static const int _ideasPerPage = 10;
  late Future<List<TradeRecommendation>> _futureRecs;
  final List<String> _symbols = ['TQQQ', 'TSLA', 'AAPL', 'META', 'SPY', 'NVDA', 'MSFT'];

  @override
  void initState() {
    super.initState();
    _vixData = widget.reportData.vixData;
    _newsError = "Economic calendar data is not available on the current API plan.";
    _futureRecs = fetchTradeRecommendations();
  }

  Future<List<TradeRecommendation>> fetchTradeRecommendations() async {
    final response = await http.post(
      Uri.parse('$apiBaseUrl/trade_recommendations'),
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': apiSecretKey,
      },
      body: json.encode({'symbols': _symbols}),
    );
    if (response.statusCode == 200) {
      final List<dynamic> data = json.decode(response.body);
      return data.map((json) => TradeRecommendation.fromJson(json)).toList();
    } else {
      throw Exception('Failed to load trade recommendations: ${response.body}');
    }
  }

  @override
  Widget build(BuildContext context) {
    final report = widget.reportData;
    final filteredIdeas = _selectedStrategy == 'All'
        ? report.tradeIdeas
        : report.tradeIdeas.where((t) => t.strategy == _selectedStrategy).toList();
    final totalPages = (filteredIdeas.length / _ideasPerPage).ceil();
    final pageIdeas = filteredIdeas.skip(_currentPage * _ideasPerPage).take(_ideasPerPage).toList();
    return Scaffold(
      appBar: AppBar(
        title: const Text('Market Insights'),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Strategy Filter Bar
            Card(
              child: Padding(
                padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                child: Row(
                  children: [
                    const Text('Filter by strategy:'),
                    const SizedBox(width: 12),
                    DropdownButton<String>(
                      value: _selectedStrategy,
                      items: _strategyOptions.map((s) => DropdownMenuItem(value: s, child: Text(s))).toList(),
                      onChanged: (val) => setState(() => _selectedStrategy = val!),
                    ),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 12),
            // Trade Ideas List with Pagination
            _buildSectionCard(
              context,
              title: 'Actionable Trade Ideas',
              child: Column(
                children: [
                  if (pageIdeas.isEmpty)
                    const Text('No trade ideas available for this strategy.'),
                  if (pageIdeas.isNotEmpty)
                    ...pageIdeas.map((t) => ListTile(
                          leading: const Icon(Icons.lightbulb, color: Colors.blue),
                          title: Text('${t.ticker} - ${t.strategy}', style: const TextStyle(fontWeight: FontWeight.bold)),
                          subtitle: Text('Expiry: ${t.expiry} | ${t.details}'),
                          trailing: Row(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              _buildProbabilityBadge(t.metricName, t.metricValue),
                            ],
                          ),
                          onTap: () {
                            showDialog(
                              context: context,
                              builder: (context) => AlertDialog(
                                title: Text('${t.ticker} - ${t.strategy}'),
                                content: Column(
                                  mainAxisSize: MainAxisSize.min,
                                  crossAxisAlignment: CrossAxisAlignment.start,
                                  children: [
                                    Text('Expiry: ${t.expiry}'),
                                    Text('Details: ${t.details}'),
                                    Text('Cost: ${t.cost > 0 ? 'Credit' : 'Debit'} ${t.cost.abs().toStringAsFixed(2)}'),
                                    if (t.maxProfit != null) Text('Max Profit: ${t.maxProfit}'),
                                    if (t.maxLoss != null) Text('Max Loss: ${t.maxLoss}'),
                                    if (t.riskRewardRatio != null) Text('Risk/Reward: ${t.riskRewardRatio!.toStringAsFixed(2)}'),
                                    if (t.metricName.isNotEmpty) Text('${t.metricName}: ${t.metricValue}'),
                                    const SizedBox(height: 10),
                                    Text('Why this strategy?', style: const TextStyle(fontWeight: FontWeight.bold)),
                                    Text(_explainStrategy(t)),
                                  ],
                                ),
                                actions: [
                                  TextButton(onPressed: () => Navigator.of(context).pop(), child: const Text('Close')),
                                ],
                              ),
                            );
                          },
                        )),
                  if (pageIdeas.isNotEmpty && totalPages > 1)
                    Padding(
                      padding: const EdgeInsets.symmetric(vertical: 8.0),
                      child: Row(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          IconButton(
                            icon: const Icon(Icons.chevron_left),
                            onPressed: _currentPage > 0
                                ? () => setState(() => _currentPage--)
                                : null,
                          ),
                          Text('Page ${_currentPage + 1} of $totalPages'),
                          IconButton(
                            icon: const Icon(Icons.chevron_right),
                            onPressed: _currentPage < totalPages - 1
                                ? () => setState(() => _currentPage++)
                                : null,
                          ),
                        ],
                      ),
                    ),
                ],
              ),
            ),
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
            const SizedBox(height: 20),
            _buildSectionCard(
              context,
              title: 'Trade Recommendations',
              child: FutureBuilder<List<TradeRecommendation>>(
                future: _futureRecs,
                builder: (context, snapshot) {
                  if (snapshot.connectionState == ConnectionState.waiting) {
                    return const Center(child: CircularProgressIndicator());
                  } else if (snapshot.hasError) {
                    return Center(
                      child: Column(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          Text('Error: \\${snapshot.error}', textAlign: TextAlign.center, style: TextStyle(color: Colors.red)),
                          SizedBox(height: 16),
                          ElevatedButton(
                            onPressed: () {
                              setState(() {
                                _futureRecs = fetchTradeRecommendations();
                              });
                            },
                            child: const Text('Retry'),
                          ),
                        ],
                      ),
                    );
                  } else if (!snapshot.hasData || snapshot.data!.isEmpty) {
                    return const Center(child: Text('No trade recommendations available.'));
                  }
                  final recs = snapshot.data!;
                  return Expanded(
                    child: ListView.separated(
                      itemCount: recs.length,
                      separatorBuilder: (context, i) => const Divider(),
                      itemBuilder: (context, i) {
                      final rec = recs[i];
                      return Card(
                        margin: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                        child: ListTile(
                          title: Text(
                            '${rec.symbol}: ${rec.recommendation}',
                            style: TextStyle(
                              fontWeight: FontWeight.bold,
                              color: rec.recommendation == 'Sell Options'
                                  ? Colors.green
                                  : rec.recommendation == 'Buy Options'
                                      ? Colors.blue
                                      : rec.recommendation.contains('Scalp')
                                          ? Colors.orange
                                          : Colors.grey,
                            ),
                          ),
                          subtitle: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text('IV Rank: ${rec.ivRank}  |  IV: ${rec.currentIv.toStringAsFixed(2)}  |  Trend: ${rec.trend}'),
                              const SizedBox(height: 4),
                              Text('Volatility: ${rec.volatilityLevel}'),
                              const SizedBox(height: 4),
                              Text(rec.rationale),
                            ],
                          ),
                        ),
                      );
                    },
                    ),
                  );
                },
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildVolatilityCard(BuildContext context) {
    double vix = _vixData.isNotEmpty ? _vixData.first.close : 0;
    String vixLevel;
    Color gaugeColor;
    if (vix < 15) {
      vixLevel = 'Low';
      gaugeColor = Colors.green;
    } else if (vix < 25) {
      vixLevel = 'Moderate';
      gaugeColor = Colors.orange;
    } else {
      vixLevel = 'High';
      gaugeColor = Colors.red;
    }
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Volatility Forecast', style: Theme.of(context).textTheme.headlineSmall),
            const SizedBox(height: 10),
            if (_vixData.isNotEmpty)
              Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text("Today's VIX: ${vix.toStringAsFixed(2)} ($vixLevel)", style: Theme.of(context).textTheme.titleLarge),
                  const SizedBox(height: 8),
                  // Gauge graphic
                  _buildVixGauge(vix, gaugeColor),
                  const SizedBox(height: 6),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: const [
                      Text('<15: Low', style: TextStyle(color: Colors.green)),
                      Text('15-25: Moderate', style: TextStyle(color: Colors.orange)),
                      Text('>25: High', style: TextStyle(color: Colors.red)),
                    ],
                  ),
                ],
              ),
            const SizedBox(height: 5),
            if (_vixData.length > 2)
              Text('Next 3 Days: ${_vixData.sublist(0, 3).map((v) => v.close.toStringAsFixed(2)).join(', ')}', style: Theme.of(context).textTheme.bodyMedium),
          ],
        ),
      ),
    );
  }

  Widget _buildVixGauge(double vix, Color color) {
    // Gauge: horizontal bar, 0-40 scale
    double percent = (vix / 40).clamp(0.0, 1.0);
    return Container(
      height: 24,
      width: double.infinity,
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(12),
        color: Colors.grey[200],
      ),
      child: Stack(
        children: [
          FractionallySizedBox(
            widthFactor: percent,
            child: Container(
              decoration: BoxDecoration(
                color: color,
                borderRadius: BorderRadius.circular(12),
              ),
            ),
          ),
          Positioned.fill(
            child: Center(
              child: Text('${vix.toStringAsFixed(2)}', style: TextStyle(color: color, fontWeight: FontWeight.bold)),
            ),
          ),
        ],
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

  String _explainStrategy(TradeIdea t) {
    // Simple explanations based on strategy type
    if (t.strategy.contains('Covered Call')) {
      return 'A covered call generates income from stocks you already own. It is best used in neutral to slightly bullish markets.';
    } else if (t.strategy.contains('Bull Put Spread')) {
      return 'A bull put spread profits if the stock stays above the short put strike. It is a bullish, limited-risk strategy.';
    } else if (t.strategy.contains('Bear Call Spread')) {
      return 'A bear call spread profits if the stock stays below the short call strike. It is a bearish, limited-risk strategy.';
    } else if (t.strategy.contains('Long Straddle')) {
      return 'A long straddle profits from large moves in either direction. Use when expecting high volatility.';
    } else if (t.strategy.contains('Long Strangle')) {
      return 'A long strangle profits from large moves in either direction, but is cheaper than a straddle. Use when expecting high volatility.';
    }
    return 'This strategy is selected based on current market conditions and risk/reward profile.';
  }

  Widget _buildProbabilityBadge(String metricName, String metricValue) {
    if (metricName.toLowerCase().contains('prob')) {
      // Extract numeric value for color cue
      final percent = double.tryParse(metricValue.replaceAll(RegExp(r'[^0-9\.]'), '')) ?? 0;
      Color color;
      if (percent >= 80) {
        color = Colors.green;
      } else if (percent >= 65) {
        color = Colors.orange;
      } else {
        color = Colors.red;
      }
      return Container(
        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
        decoration: BoxDecoration(
          color: color.withOpacity(0.15),
          border: Border.all(color: color, width: 1.5),
          borderRadius: BorderRadius.circular(12),
        ),
        child: Text(
          metricValue,
          style: TextStyle(
            color: color,
            fontWeight: FontWeight.bold,
            fontSize: 14,
          ),
        ),
      );
    }
    return const SizedBox.shrink();
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
