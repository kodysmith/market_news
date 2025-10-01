import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

class OpportunitiesScreen extends StatefulWidget {
  const OpportunitiesScreen({Key? key}) : super(key: key);

  @override
  _OpportunitiesScreenState createState() => _OpportunitiesScreenState();
}

class _OpportunitiesScreenState extends State<OpportunitiesScreen> {
  List<dynamic> opportunities = [];
  List<dynamic> sectorSummaries = [];
  bool isLoading = true;
  String error = '';
  String selectedFilter = 'All';
  double minScore = 60.0;

  final List<String> filters = [
    'All',
    'Technology',
    'Healthcare',
    'Financials',
    'Consumer Discretionary',
    'Consumer Staples',
    'Industrials',
    'Energy',
    'Utilities',
    'Real Estate',
    'Materials',
    'Communication Services'
  ];

  @override
  void initState() {
    super.initState();
    _loadOpportunities();
  }

  Future<void> _loadOpportunities() async {
    setState(() {
      isLoading = true;
      error = '';
    });

    try {
      // Load opportunities
      final oppResponse = await http.get(
        Uri.parse('https://us-central1-kardova-capital.cloudfunctions.net/api/latest-opportunities?limit=50&minScore=${minScore.toInt()}'),
        headers: {'Content-Type': 'application/json'},
      );

      if (oppResponse.statusCode == 200) {
        final oppData = json.decode(oppResponse.body);
        setState(() {
          opportunities = oppData['opportunities'] ?? [];
        });
      } else {
        throw Exception('Failed to load opportunities');
      }

      // Load sector summaries
      final sectorResponse = await http.get(
        Uri.parse('https://us-central1-kardova-capital.cloudfunctions.net/api/sector-summaries?limit=12'),
        headers: {'Content-Type': 'application/json'},
      );

      if (sectorResponse.statusCode == 200) {
        final sectorData = json.decode(sectorResponse.body);
        setState(() {
          sectorSummaries = sectorData['summaries'] ?? [];
        });
      }

    } catch (e) {
      setState(() {
        error = 'Failed to load opportunities: $e';
      });
    } finally {
      setState(() {
        isLoading = false;
      });
    }
  }

  List<dynamic> get filteredOpportunities {
    if (selectedFilter == 'All') {
      return opportunities;
    }
    return opportunities.where((opp) => opp['sector'] == selectedFilter).toList();
  }

  Color getOpportunityColor(String type) {
    switch (type) {
      case 'STRONG_BUY':
        return Colors.green.shade700;
      case 'BUY':
        return Colors.green.shade500;
      case 'HOLD':
        return Colors.orange.shade500;
      case 'SELL':
        return Colors.red.shade500;
      case 'STRONG_SELL':
        return Colors.red.shade700;
      default:
        return Colors.grey.shade500;
    }
  }

  IconData getOpportunityIcon(String type) {
    switch (type) {
      case 'STRONG_BUY':
        return Icons.trending_up;
      case 'BUY':
        return Icons.arrow_upward;
      case 'HOLD':
        return Icons.horizontal_rule;
      case 'SELL':
        return Icons.arrow_downward;
      case 'STRONG_SELL':
        return Icons.trending_down;
      default:
        return Icons.help_outline;
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Trading Opportunities'),
        backgroundColor: Colors.blue.shade800,
        foregroundColor: Colors.white,
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh),
            onPressed: _loadOpportunities,
          ),
        ],
      ),
      body: Column(
        children: [
          // Filter and controls
          Container(
            padding: const EdgeInsets.all(16),
            color: Colors.grey.shade100,
            child: Column(
              children: [
                // Sector filter
                DropdownButton<String>(
                  value: selectedFilter,
                  isExpanded: true,
                  items: filters.map((String filter) {
                    return DropdownMenuItem<String>(
                      value: filter,
                      child: Text(filter),
                    );
                  }).toList(),
                  onChanged: (String? newValue) {
                    setState(() {
                      selectedFilter = newValue!;
                    });
                  },
                ),
                const SizedBox(height: 16),
                // Score filter
                Row(
                  children: [
                    const Text('Min Score: '),
                    Expanded(
                      child: Slider(
                        value: minScore,
                        min: 0,
                        max: 100,
                        divisions: 20,
                        label: minScore.round().toString(),
                        onChanged: (double value) {
                          setState(() {
                            minScore = value;
                          });
                          _loadOpportunities();
                        },
                      ),
                    ),
                    Text('${minScore.round()}'),
                  ],
                ),
              ],
            ),
          ),
          
          // Content
          Expanded(
            child: isLoading
                ? const Center(child: CircularProgressIndicator())
                : error.isNotEmpty
                    ? Center(
                        child: Column(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            Icon(Icons.error, size: 64, color: Colors.red.shade300),
                            const SizedBox(height: 16),
                            Text(
                              error,
                              style: TextStyle(color: Colors.red.shade700),
                              textAlign: TextAlign.center,
                            ),
                            const SizedBox(height: 16),
                            ElevatedButton(
                              onPressed: _loadOpportunities,
                              child: const Text('Retry'),
                            ),
                          ],
                        ),
                      )
                    : filteredOpportunities.isEmpty
                        ? const Center(
                            child: Column(
                              mainAxisAlignment: MainAxisAlignment.center,
                              children: [
                                Icon(Icons.search_off, size: 64, color: Colors.grey),
                                SizedBox(height: 16),
                                Text(
                                  'No opportunities found',
                                  style: TextStyle(fontSize: 18, color: Colors.grey),
                                ),
                              ],
                            ),
                          )
                        : ListView.builder(
                            itemCount: filteredOpportunities.length,
                            itemBuilder: (context, index) {
                              final opp = filteredOpportunities[index];
                              return _buildOpportunityCard(opp);
                            },
                          ),
          ),
        ],
      ),
    );
  }

  Widget _buildOpportunityCard(Map<String, dynamic> opp) {
    final score = opp['overall_score']?.toDouble() ?? 0.0;
    final type = opp['opportunity_type'] ?? 'UNKNOWN';
    final ticker = opp['ticker'] ?? 'N/A';
    final sector = opp['sector'] ?? 'N/A';
    final currentPrice = opp['current_price']?.toDouble() ?? 0.0;
    final targetPrice = opp['target_price']?.toDouble() ?? 0.0;
    final riskReward = opp['risk_reward_ratio']?.toDouble() ?? 0.0;
    final rsi = opp['rsi']?.toDouble() ?? 0.0;
    final peRatio = opp['pe_ratio']?.toDouble() ?? 0.0;
    final revenueGrowth = opp['revenue_growth']?.toDouble() ?? 0.0;

    return Card(
      margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      elevation: 4,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Header
            Row(
              children: [
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                  decoration: BoxDecoration(
                    color: getOpportunityColor(type),
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Icon(
                        getOpportunityIcon(type),
                        color: Colors.white,
                        size: 16,
                      ),
                      const SizedBox(width: 4),
                      Text(
                        type,
                        style: const TextStyle(
                          color: Colors.white,
                          fontWeight: FontWeight.bold,
                          fontSize: 12,
                        ),
                      ),
                    ],
                  ),
                ),
                const Spacer(),
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                  decoration: BoxDecoration(
                    color: Colors.blue.shade100,
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Text(
                    'Score: ${score.toStringAsFixed(1)}',
                    style: TextStyle(
                      color: Colors.blue.shade800,
                      fontWeight: FontWeight.bold,
                      fontSize: 12,
                    ),
                  ),
                ),
              ],
            ),
            
            const SizedBox(height: 12),
            
            // Ticker and Sector
            Row(
              children: [
                Text(
                  ticker,
                  style: const TextStyle(
                    fontSize: 24,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                const SizedBox(width: 8),
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                  decoration: BoxDecoration(
                    color: Colors.grey.shade200,
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: Text(
                    sector,
                    style: TextStyle(
                      fontSize: 12,
                      color: Colors.grey.shade700,
                    ),
                  ),
                ),
              ],
            ),
            
            const SizedBox(height: 12),
            
            // Price Information
            Row(
              children: [
                Expanded(
                  child: _buildInfoItem(
                    'Current Price',
                    '\$${currentPrice.toStringAsFixed(2)}',
                    Colors.black87,
                  ),
                ),
                Expanded(
                  child: _buildInfoItem(
                    'Target Price',
                    '\$${targetPrice.toStringAsFixed(2)}',
                    Colors.green.shade700,
                  ),
                ),
                Expanded(
                  child: _buildInfoItem(
                    'Risk/Reward',
                    '${riskReward.toStringAsFixed(1)}:1',
                    Colors.blue.shade700,
                  ),
                ),
              ],
            ),
            
            const SizedBox(height: 12),
            
            // Technical Indicators
            Row(
              children: [
                Expanded(
                  child: _buildInfoItem(
                    'RSI',
                    rsi.toStringAsFixed(1),
                    rsi > 70 ? Colors.red : rsi < 30 ? Colors.green : Colors.grey,
                  ),
                ),
                Expanded(
                  child: _buildInfoItem(
                    'P/E Ratio',
                    peRatio.toStringAsFixed(1),
                    peRatio > 25 ? Colors.red : peRatio < 15 ? Colors.green : Colors.grey,
                  ),
                ),
                Expanded(
                  child: _buildInfoItem(
                    'Revenue Growth',
                    '${revenueGrowth.toStringAsFixed(1)}%',
                    revenueGrowth > 10 ? Colors.green : revenueGrowth < 0 ? Colors.red : Colors.grey,
                  ),
                ),
              ],
            ),
            
            const SizedBox(height: 12),
            
            // Additional Info
            if (opp['trend_direction'] != null)
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                decoration: BoxDecoration(
                  color: Colors.grey.shade100,
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Icon(
                      Icons.trending_up,
                      size: 16,
                      color: Colors.grey.shade600,
                    ),
                    const SizedBox(width: 4),
                    Text(
                      'Trend: ${opp['trend_direction']}',
                      style: TextStyle(
                        fontSize: 12,
                        color: Colors.grey.shade700,
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

  Widget _buildInfoItem(String label, String value, Color valueColor) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          label,
          style: TextStyle(
            fontSize: 12,
            color: Colors.grey.shade600,
            fontWeight: FontWeight.w500,
          ),
        ),
        const SizedBox(height: 2),
        Text(
          value,
          style: TextStyle(
            fontSize: 14,
            color: valueColor,
            fontWeight: FontWeight.bold,
          ),
        ),
      ],
    );
  }
}
