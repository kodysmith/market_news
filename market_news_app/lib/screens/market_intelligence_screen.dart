import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import '../main.dart' show apiBaseUrl, apiSecretKey;

class MarketRegime {
  final String regime;
  final double confidence;
  final String reasoning;
  final String vixLevel;

  MarketRegime({
    required this.regime,
    required this.confidence,
    required this.reasoning,
    required this.vixLevel,
  });

  factory MarketRegime.fromJson(Map<String, dynamic> json) {
    return MarketRegime(
      regime: json['regime'],
      confidence: json['confidence'].toDouble(),
      reasoning: json['reasoning'],
      vixLevel: json['vix_level'],
    );
  }
}

class TradeExample {
  final String symbol;
  final String strategy;
  final String setup;
  final String entry;
  final String target;
  final String stopLoss;
  final String rationale;

  TradeExample({
    required this.symbol,
    required this.strategy,
    required this.setup,
    required this.entry,
    required this.target,
    required this.stopLoss,
    required this.rationale,
  });

  factory TradeExample.fromJson(Map<String, dynamic> json) {
    return TradeExample(
      symbol: json['symbol'],
      strategy: json['strategy'],
      setup: json['setup'],
      entry: json['entry'],
      target: json['target'],
      stopLoss: json['stop_loss'],
      rationale: json['rationale'],
    );
  }
}

class QuantEngineRecommendation {
  final String strategy;
  final String priority;
  final String rationale;
  final List<String> specificStrategies;
  final List<String> targetAssets;
  final String riskLevel;
  final String timeHorizon;
  final List<TradeExample> exampleTrades;

  QuantEngineRecommendation({
    required this.strategy,
    required this.priority,
    required this.rationale,
    required this.specificStrategies,
    required this.targetAssets,
    required this.riskLevel,
    required this.timeHorizon,
    required this.exampleTrades,
  });

  factory QuantEngineRecommendation.fromJson(Map<String, dynamic> json) {
    return QuantEngineRecommendation(
      strategy: json['strategy'],
      priority: json['priority'],
      rationale: json['rationale'],
      specificStrategies: List<String>.from(json['specific_strategies']),
      targetAssets: List<String>.from(json['target_assets']),
      riskLevel: json['risk_level'],
      timeHorizon: json['time_horizon'],
      exampleTrades: (json['example_trades'] as List? ?? [])
          .map((t) => TradeExample.fromJson(t))
          .toList(),
    );
  }
}

class MarketIntelligenceData {
  final MarketRegime marketRegime;
  final List<QuantEngineRecommendation> recommendations;
  final String generatedAt;

  MarketIntelligenceData({
    required this.marketRegime,
    required this.recommendations,
    required this.generatedAt,
  });

  factory MarketIntelligenceData.fromJson(Map<String, dynamic> json) {
    return MarketIntelligenceData(
      marketRegime: MarketRegime.fromJson(json['market_regime']),
      recommendations: (json['recommendations'] as List)
          .map((r) => QuantEngineRecommendation.fromJson(r))
          .toList(),
      generatedAt: json['generated_at'],
    );
  }
}

class MarketIntelligenceScreen extends StatefulWidget {
  const MarketIntelligenceScreen({super.key});

  @override
  State<MarketIntelligenceScreen> createState() => _MarketIntelligenceScreenState();
}

class _MarketIntelligenceScreenState extends State<MarketIntelligenceScreen> {
  late Future<MarketIntelligenceData> _futureIntelligence;

  @override
  void initState() {
    super.initState();
    _futureIntelligence = fetchMarketIntelligence();
  }

  Future<MarketIntelligenceData> fetchMarketIntelligence() async {
    final response = await http.get(
      Uri.parse('$apiBaseUrl/quantengine/recommendations'),
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': apiSecretKey,
      },
    );

    if (response.statusCode == 200) {
      return MarketIntelligenceData.fromJson(json.decode(response.body));
    } else {
      throw Exception('Failed to load market intelligence: ${response.body}');
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Market Intelligence'),
        backgroundColor: Colors.indigo,
        foregroundColor: Colors.white,
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh),
            onPressed: () {
              setState(() {
                _futureIntelligence = fetchMarketIntelligence();
              });
            },
          ),
        ],
      ),
      body: FutureBuilder<MarketIntelligenceData>(
        future: _futureIntelligence,
        builder: (context, snapshot) {
          if (snapshot.connectionState == ConnectionState.waiting) {
            return const Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  CircularProgressIndicator(),
                  SizedBox(height: 16),
                  Text('Analyzing market conditions...'),
                ],
              ),
            );
          } else if (snapshot.hasError) {
            return Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  const Icon(Icons.error, size: 64, color: Colors.red),
                  const SizedBox(height: 16),
                  Text('Error: ${snapshot.error}'),
                  const SizedBox(height: 16),
                  ElevatedButton(
                    onPressed: () {
                      setState(() {
                        _futureIntelligence = fetchMarketIntelligence();
                      });
                    },
                    child: const Text('Retry'),
                  ),
                ],
              ),
            );
          } else if (!snapshot.hasData) {
            return const Center(child: Text('No market intelligence available'));
          }

          final intelligence = snapshot.data!;
          return SingleChildScrollView(
            padding: const EdgeInsets.all(16.0),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                _buildMarketRegimeCard(intelligence.marketRegime),
                const SizedBox(height: 20),
                _buildRecommendationsSection(intelligence.recommendations),
                const SizedBox(height: 20),
                _buildLastUpdated(intelligence.generatedAt),
              ],
            ),
          );
        },
      ),
    );
  }

  Widget _buildMarketRegimeCard(MarketRegime regime) {
    Color regimeColor;
    IconData regimeIcon;
    String regimeTitle;

    switch (regime.regime) {
      case 'high_volatility':
        regimeColor = Colors.red;
        regimeIcon = Icons.trending_up;
        regimeTitle = 'High Volatility Regime';
        break;
      case 'low_volatility':
        regimeColor = Colors.blue;
        regimeIcon = Icons.trending_flat;
        regimeTitle = 'Low Volatility Regime';
        break;
      case 'trending_up':
        regimeColor = Colors.green;
        regimeIcon = Icons.arrow_upward;
        regimeTitle = 'Bullish Trend';
        break;
      case 'trending_down':
        regimeColor = Colors.orange;
        regimeIcon = Icons.arrow_downward;
        regimeTitle = 'Bearish Trend';
        break;
      default:
        regimeColor = Colors.grey;
        regimeIcon = Icons.remove;
        regimeTitle = 'Neutral Market';
    }

    return Card(
      elevation: 8,
      child: Container(
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(12),
          gradient: LinearGradient(
            colors: [regimeColor.withOpacity(0.1), Colors.white],
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
          ),
        ),
        child: Padding(
          padding: const EdgeInsets.all(20.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  Icon(regimeIcon, size: 32, color: regimeColor),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Text(
                      regimeTitle,
                      style: TextStyle(
                        fontSize: 24,
                        fontWeight: FontWeight.bold,
                        color: regimeColor,
                      ),
                    ),
                  ),
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                    decoration: BoxDecoration(
                      color: regimeColor.withOpacity(0.2),
                      borderRadius: BorderRadius.circular(16),
                    ),
                    child: Text(
                      '${(regime.confidence * 100).toInt()}% confidence',
                      style: TextStyle(
                        fontWeight: FontWeight.bold,
                        color: regimeColor,
                      ),
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 16),
              Container(
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: Colors.grey[50],
                  borderRadius: BorderRadius.circular(8),
                  border: Border(left: BorderSide(width: 4, color: regimeColor)),
                ),
                child: Text(
                  regime.reasoning,
                  style: const TextStyle(fontSize: 16, height: 1.4),
                ),
              ),
              const SizedBox(height: 12),
              Row(
                children: [
                  const Icon(Icons.speed, size: 20, color: Colors.grey),
                  const SizedBox(width: 8),
                  Text(
                    'VIX Level: ${regime.vixLevel.toUpperCase()}',
                    style: const TextStyle(
                      fontSize: 14,
                      fontWeight: FontWeight.w500,
                      color: Colors.grey,
                    ),
                  ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildRecommendationsSection(List<QuantEngineRecommendation> recommendations) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text(
          'QuantEngine Recommendations',
          style: TextStyle(
            fontSize: 22,
            fontWeight: FontWeight.bold,
            color: Colors.indigo,
          ),
        ),
        const SizedBox(height: 16),
        ...recommendations.map((rec) => _buildRecommendationCard(rec)),
      ],
    );
  }

  Widget _buildRecommendationCard(QuantEngineRecommendation recommendation) {
    Color priorityColor;
    switch (recommendation.priority) {
      case 'critical':
        priorityColor = Colors.red;
        break;
      case 'high':
        priorityColor = Colors.orange;
        break;
      case 'medium':
        priorityColor = Colors.blue;
        break;
      default:
        priorityColor = Colors.grey;
    }

    Color riskColor;
    switch (recommendation.riskLevel) {
      case 'high':
        riskColor = Colors.red;
        break;
      case 'medium':
        riskColor = Colors.orange;
        break;
      default:
        riskColor = Colors.green;
    }

    return Card(
      margin: const EdgeInsets.only(bottom: 16),
      elevation: 4,
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                  decoration: BoxDecoration(
                    color: priorityColor.withOpacity(0.2),
                    borderRadius: BorderRadius.circular(12),
                    border: Border.all(color: priorityColor),
                  ),
                  child: Text(
                    recommendation.priority.toUpperCase(),
                    style: TextStyle(
                      fontSize: 12,
                      fontWeight: FontWeight.bold,
                      color: priorityColor,
                    ),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: Text(
                    recommendation.strategy,
                    style: const TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 12),
            Text(
              recommendation.rationale,
              style: const TextStyle(fontSize: 16, height: 1.4),
            ),
            const SizedBox(height: 16),
            _buildStrategyChips('Strategies', recommendation.specificStrategies, Colors.blue),
            const SizedBox(height: 12),
            _buildStrategyChips('Target Assets', recommendation.targetAssets, Colors.green),
            const SizedBox(height: 16),
            Row(
              children: [
                _buildInfoChip('Risk', recommendation.riskLevel, riskColor),
                const SizedBox(width: 12),
                _buildInfoChip('Time Horizon', recommendation.timeHorizon, Colors.grey),
              ],
            ),
            if (recommendation.exampleTrades.isNotEmpty) ...[
              const SizedBox(height: 20),
              const Text(
                'Specific Trade Examples:',
                style: TextStyle(
                  fontSize: 16,
                  fontWeight: FontWeight.bold,
                  color: Colors.indigo,
                ),
              ),
              const SizedBox(height: 12),
              ...recommendation.exampleTrades.map((trade) => _buildTradeExampleCard(trade)),
            ],
          ],
        ),
      ),
    );
  }

  Widget _buildStrategyChips(String label, List<String> items, Color color) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          '$label:',
          style: const TextStyle(
            fontSize: 14,
            fontWeight: FontWeight.w600,
            color: Colors.grey,
          ),
        ),
        const SizedBox(height: 6),
        Wrap(
          spacing: 8,
          runSpacing: 4,
          children: items.map((item) => Chip(
            label: Text(item),
            backgroundColor: color.withOpacity(0.1),
            labelStyle: TextStyle(color: color, fontSize: 12),
            side: BorderSide(color: color.withOpacity(0.3)),
          )).toList(),
        ),
      ],
    );
  }

  Widget _buildInfoChip(String label, String value, Color color) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
      decoration: BoxDecoration(
        color: color.withOpacity(0.1),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: color.withOpacity(0.3)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Text(
            '$label: ',
            style: const TextStyle(fontSize: 12, fontWeight: FontWeight.w500),
          ),
          Text(
            value,
            style: TextStyle(
              fontSize: 12,
              fontWeight: FontWeight.bold,
              color: color,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildTradeExampleCard(TradeExample trade) {
    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      decoration: BoxDecoration(
        color: Colors.grey[50],
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Colors.grey[300]!),
      ),
      child: ExpansionTile(
        tilePadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
        childrenPadding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
        title: Row(
          children: [
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
              decoration: BoxDecoration(
                color: Colors.blue.withOpacity(0.1),
                borderRadius: BorderRadius.circular(8),
                border: Border.all(color: Colors.blue.withOpacity(0.3)),
              ),
              child: Text(
                trade.symbol,
                style: const TextStyle(
                  fontSize: 12,
                  fontWeight: FontWeight.bold,
                  color: Colors.blue,
                ),
              ),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: Text(
                trade.strategy,
                style: const TextStyle(
                  fontSize: 14,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ),
          ],
        ),
        subtitle: Padding(
          padding: const EdgeInsets.only(top: 4),
          child: Text(
            trade.setup,
            style: TextStyle(
              fontSize: 12,
              color: Colors.grey[600],
            ),
            maxLines: 2,
            overflow: TextOverflow.ellipsis,
          ),
        ),
        children: [
          Container(
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
              color: Colors.white,
              borderRadius: BorderRadius.circular(8),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                _buildTradeDetailRow('Entry', trade.entry, Icons.login),
                const SizedBox(height: 8),
                _buildTradeDetailRow('Target', trade.target, Icons.flag),
                const SizedBox(height: 8),
                _buildTradeDetailRow('Stop Loss', trade.stopLoss, Icons.stop_circle),
                const SizedBox(height: 12),
                Container(
                  padding: const EdgeInsets.all(12),
                  decoration: BoxDecoration(
                    color: Colors.blue.withOpacity(0.05),
                    borderRadius: BorderRadius.circular(8),
                    border: Border(left: BorderSide(width: 3, color: Colors.blue)),
                  ),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Text(
                        'Rationale:',
                        style: TextStyle(
                          fontSize: 12,
                          fontWeight: FontWeight.bold,
                          color: Colors.blue,
                        ),
                      ),
                      const SizedBox(height: 4),
                      Text(
                        trade.rationale,
                        style: const TextStyle(
                          fontSize: 12,
                          height: 1.4,
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildTradeDetailRow(String label, String value, IconData icon) {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Icon(icon, size: 16, color: Colors.grey[600]),
        const SizedBox(width: 8),
        Text(
          '$label: ',
          style: const TextStyle(
            fontSize: 12,
            fontWeight: FontWeight.w600,
            color: Colors.grey,
          ),
        ),
        Expanded(
          child: Text(
            value,
            style: const TextStyle(
              fontSize: 12,
              fontWeight: FontWeight.w500,
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildLastUpdated(String timestamp) {
    final DateTime dateTime = DateTime.parse(timestamp);
    final String formattedTime = '${dateTime.hour.toString().padLeft(2, '0')}:${dateTime.minute.toString().padLeft(2, '0')}';
    
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.grey[100],
        borderRadius: BorderRadius.circular(8),
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          const Icon(Icons.access_time, size: 16, color: Colors.grey),
          const SizedBox(width: 8),
          Text(
            'Last updated: $formattedTime',
            style: const TextStyle(
              fontSize: 12,
              color: Colors.grey,
            ),
          ),
        ],
      ),
    );
  }
}
