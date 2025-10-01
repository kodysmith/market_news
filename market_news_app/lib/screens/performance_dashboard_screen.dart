import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import '../main.dart' show apiBaseUrl, apiSecretKey;

class PerformanceSummary {
  final int totalRecommendations;
  final int openPositions;
  final int closedPositions;
  final double winRate;
  final double avgReturnPerTrade;
  final double totalPnl;
  final double bestTrade;
  final double worstTrade;
  final double avgHoldTime;
  final double sharpeRatio;

  PerformanceSummary({
    required this.totalRecommendations,
    required this.openPositions,
    required this.closedPositions,
    required this.winRate,
    required this.avgReturnPerTrade,
    required this.totalPnl,
    required this.bestTrade,
    required this.worstTrade,
    required this.avgHoldTime,
    required this.sharpeRatio,
  });

  factory PerformanceSummary.fromJson(Map<String, dynamic> json) {
    return PerformanceSummary(
      totalRecommendations: json['total_recommendations'],
      openPositions: json['open_positions'],
      closedPositions: json['closed_positions'],
      winRate: json['win_rate'].toDouble(),
      avgReturnPerTrade: json['avg_return_per_trade'].toDouble(),
      totalPnl: json['total_pnl'].toDouble(),
      bestTrade: json['best_trade'].toDouble(),
      worstTrade: json['worst_trade'].toDouble(),
      avgHoldTime: json['avg_hold_time'].toDouble(),
      sharpeRatio: json['sharpe_ratio'].toDouble(),
    );
  }
}

class OpenPosition {
  final String id;
  final String symbol;
  final String strategy;
  final String entryDate;
  final double entryPrice;
  final double currentPrice;
  final double currentPnl;
  final int daysHeld;
  final String status;

  OpenPosition({
    required this.id,
    required this.symbol,
    required this.strategy,
    required this.entryDate,
    required this.entryPrice,
    required this.currentPrice,
    required this.currentPnl,
    required this.daysHeld,
    required this.status,
  });

  factory OpenPosition.fromJson(Map<String, dynamic> json) {
    return OpenPosition(
      id: json['id'],
      symbol: json['symbol'],
      strategy: json['strategy'],
      entryDate: json['entry_date'],
      entryPrice: json['entry_price'].toDouble(),
      currentPrice: json['current_price'].toDouble(),
      currentPnl: json['current_pnl'].toDouble(),
      daysHeld: json['days_held'],
      status: json['status'],
    );
  }
}

class StrategyPerformance {
  final String strategy;
  final int totalTrades;
  final double winRate;
  final double avgReturn;
  final double bestReturn;
  final String status;

  StrategyPerformance({
    required this.strategy,
    required this.totalTrades,
    required this.winRate,
    required this.avgReturn,
    required this.bestReturn,
    required this.status,
  });

  factory StrategyPerformance.fromJson(Map<String, dynamic> json) {
    return StrategyPerformance(
      strategy: json['strategy'],
      totalTrades: json['total_trades'],
      winRate: json['win_rate'].toDouble(),
      avgReturn: json['avg_return'].toDouble(),
      bestReturn: json['best_return'].toDouble(),
      status: json['status'],
    );
  }
}

class Milestone {
  final String title;
  final String description;
  final String achievedDate;
  final String type;

  Milestone({
    required this.title,
    required this.description,
    required this.achievedDate,
    required this.type,
  });

  factory Milestone.fromJson(Map<String, dynamic> json) {
    return Milestone(
      title: json['title'],
      description: json['description'],
      achievedDate: json['achieved_date'],
      type: json['type'],
    );
  }
}

class PerformanceData {
  final PerformanceSummary summary;
  final List<OpenPosition> openPositions;
  final List<StrategyPerformance> strategyPerformance;
  final List<Milestone> milestones;

  PerformanceData({
    required this.summary,
    required this.openPositions,
    required this.strategyPerformance,
    required this.milestones,
  });

  factory PerformanceData.fromJson(Map<String, dynamic> json) {
    return PerformanceData(
      summary: PerformanceSummary.fromJson(json['summary']),
      openPositions: (json['open_positions'] as List)
          .map((p) => OpenPosition.fromJson(p))
          .toList(),
      strategyPerformance: (json['strategy_performance'] as List)
          .map((s) => StrategyPerformance.fromJson(s))
          .toList(),
      milestones: (json['milestones'] as List)
          .map((m) => Milestone.fromJson(m))
          .toList(),
    );
  }
}

class PerformanceDashboardScreen extends StatefulWidget {
  const PerformanceDashboardScreen({super.key});

  @override
  State<PerformanceDashboardScreen> createState() => _PerformanceDashboardScreenState();
}

class _PerformanceDashboardScreenState extends State<PerformanceDashboardScreen> {
  late Future<PerformanceData> _futurePerformance;

  @override
  void initState() {
    super.initState();
    _futurePerformance = fetchPerformanceData();
  }

  Future<PerformanceData> fetchPerformanceData() async {
    final response = await http.get(
      Uri.parse('$apiBaseUrl/performance/dashboard'),
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': apiSecretKey,
      },
    );

    if (response.statusCode == 200) {
      return PerformanceData.fromJson(json.decode(response.body));
    } else {
      throw Exception('Failed to load performance data: ${response.body}');
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Our Track Record'),
        backgroundColor: Colors.green,
        foregroundColor: Colors.white,
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh),
            onPressed: () {
              setState(() {
                _futurePerformance = fetchPerformanceData();
              });
            },
          ),
        ],
      ),
      body: FutureBuilder<PerformanceData>(
        future: _futurePerformance,
        builder: (context, snapshot) {
          if (snapshot.connectionState == ConnectionState.waiting) {
            return const Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  CircularProgressIndicator(),
                  SizedBox(height: 16),
                  Text('Loading performance data...'),
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
                        _futurePerformance = fetchPerformanceData();
                      });
                    },
                    child: const Text('Retry'),
                  ),
                ],
              ),
            );
          } else if (!snapshot.hasData) {
            return const Center(child: Text('No performance data available'));
          }

          final performance = snapshot.data!;
          return SingleChildScrollView(
            padding: const EdgeInsets.all(16.0),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                _buildPerformanceSummaryCard(performance.summary),
                const SizedBox(height: 20),
                _buildMilestonesSection(performance.milestones),
                const SizedBox(height: 20),
                _buildOpenPositionsSection(performance.openPositions),
                const SizedBox(height: 20),
                _buildStrategyPerformanceSection(performance.strategyPerformance),
              ],
            ),
          );
        },
      ),
    );
  }

  Widget _buildPerformanceSummaryCard(PerformanceSummary summary) {
    return Card(
      elevation: 8,
      child: Container(
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(12),
          gradient: LinearGradient(
            colors: [Colors.green.withOpacity(0.1), Colors.white],
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
                  const Icon(Icons.trending_up, size: 32, color: Colors.green),
                  const SizedBox(width: 12),
                  const Text(
                    'Overall Performance',
                    style: TextStyle(
                      fontSize: 24,
                      fontWeight: FontWeight.bold,
                      color: Colors.green,
                    ),
                  ),
                  const Spacer(),
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                    decoration: BoxDecoration(
                      color: Colors.green.withOpacity(0.2),
                      borderRadius: BorderRadius.circular(16),
                    ),
                    child: Text(
                      'Sharpe: ${summary.sharpeRatio.toStringAsFixed(2)}',
                      style: const TextStyle(
                        fontWeight: FontWeight.bold,
                        color: Colors.green,
                      ),
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 20),
              Row(
                children: [
                  Expanded(
                    child: _buildStatCard(
                      'Win Rate',
                      '${summary.winRate.toStringAsFixed(1)}%',
                      Colors.green,
                      Icons.check_circle,
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: _buildStatCard(
                      'Avg Return',
                      '${summary.avgReturnPerTrade.toStringAsFixed(1)}%',
                      Colors.blue,
                      Icons.show_chart,
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 12),
              Row(
                children: [
                  Expanded(
                    child: _buildStatCard(
                      'Total P&L',
                      '\$${summary.totalPnl.toStringAsFixed(0)}',
                      Colors.purple,
                      Icons.account_balance_wallet,
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: _buildStatCard(
                      'Open Trades',
                      '${summary.openPositions}',
                      Colors.orange,
                      Icons.pending,
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
                ),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.spaceAround,
                  children: [
                    Column(
                      children: [
                        Text(
                          'Best Trade',
                          style: TextStyle(fontSize: 12, color: Colors.grey[600]),
                        ),
                        Text(
                          '+${summary.bestTrade.toStringAsFixed(1)}%',
                          style: const TextStyle(
                            fontSize: 16,
                            fontWeight: FontWeight.bold,
                            color: Colors.green,
                          ),
                        ),
                      ],
                    ),
                    Column(
                      children: [
                        Text(
                          'Worst Trade',
                          style: TextStyle(fontSize: 12, color: Colors.grey[600]),
                        ),
                        Text(
                          '${summary.worstTrade.toStringAsFixed(1)}%',
                          style: const TextStyle(
                            fontSize: 16,
                            fontWeight: FontWeight.bold,
                            color: Colors.red,
                          ),
                        ),
                      ],
                    ),
                    Column(
                      children: [
                        Text(
                          'Avg Hold Time',
                          style: TextStyle(fontSize: 12, color: Colors.grey[600]),
                        ),
                        Text(
                          '${summary.avgHoldTime.toStringAsFixed(1)} days',
                          style: const TextStyle(
                            fontSize: 16,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                      ],
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildStatCard(String title, String value, Color color, IconData icon) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: color.withOpacity(0.1),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: color.withOpacity(0.3)),
      ),
      child: Column(
        children: [
          Icon(icon, color: color, size: 24),
          const SizedBox(height: 8),
          Text(
            title,
            style: TextStyle(
              fontSize: 12,
              color: Colors.grey[600],
            ),
          ),
          const SizedBox(height: 4),
          Text(
            value,
            style: TextStyle(
              fontSize: 18,
              fontWeight: FontWeight.bold,
              color: color,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildMilestonesSection(List<Milestone> milestones) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text(
          'Recent Achievements',
          style: TextStyle(
            fontSize: 22,
            fontWeight: FontWeight.bold,
            color: Colors.indigo,
          ),
        ),
        const SizedBox(height: 16),
        Container(
          height: 120,
          child: ListView.builder(
            scrollDirection: Axis.horizontal,
            itemCount: milestones.length,
            itemBuilder: (context, index) {
              final milestone = milestones[index];
              return Container(
                width: 200,
                margin: const EdgeInsets.only(right: 16),
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  gradient: LinearGradient(
                    colors: [Colors.purple.withOpacity(0.1), Colors.blue.withOpacity(0.1)],
                    begin: Alignment.topLeft,
                    end: Alignment.bottomRight,
                  ),
                  borderRadius: BorderRadius.circular(12),
                  border: Border.all(color: Colors.purple.withOpacity(0.3)),
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      milestone.title,
                      style: const TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    const SizedBox(height: 8),
                    Text(
                      milestone.description,
                      style: TextStyle(
                        fontSize: 12,
                        color: Colors.grey[600],
                      ),
                    ),
                    const Spacer(),
                    Text(
                      milestone.achievedDate,
                      style: TextStyle(
                        fontSize: 10,
                        color: Colors.grey[500],
                      ),
                    ),
                  ],
                ),
              );
            },
          ),
        ),
      ],
    );
  }

  Widget _buildOpenPositionsSection(List<OpenPosition> positions) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            const Text(
              'Live Positions',
              style: TextStyle(
                fontSize: 22,
                fontWeight: FontWeight.bold,
                color: Colors.indigo,
              ),
            ),
            const SizedBox(width: 12),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
              decoration: BoxDecoration(
                color: Colors.orange.withOpacity(0.2),
                borderRadius: BorderRadius.circular(12),
              ),
              child: Text(
                '${positions.length} open',
                style: const TextStyle(
                  fontSize: 12,
                  fontWeight: FontWeight.bold,
                  color: Colors.orange,
                ),
              ),
            ),
          ],
        ),
        const SizedBox(height: 16),
        ...positions.map((position) => Container(
          margin: const EdgeInsets.only(bottom: 12),
          padding: const EdgeInsets.all(16),
          decoration: BoxDecoration(
            color: position.currentPnl >= 0 ? Colors.green.withOpacity(0.05) : Colors.red.withOpacity(0.05),
            borderRadius: BorderRadius.circular(12),
            border: Border.all(
              color: position.currentPnl >= 0 ? Colors.green.withOpacity(0.3) : Colors.red.withOpacity(0.3),
            ),
          ),
          child: Row(
            children: [
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                decoration: BoxDecoration(
                  color: Colors.blue.withOpacity(0.1),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Text(
                  position.symbol,
                  style: const TextStyle(
                    fontSize: 12,
                    fontWeight: FontWeight.bold,
                    color: Colors.blue,
                  ),
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      position.strategy,
                      style: const TextStyle(
                        fontSize: 14,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                    Text(
                      '${position.daysHeld} days held',
                      style: TextStyle(
                        fontSize: 12,
                        color: Colors.grey[600],
                      ),
                    ),
                  ],
                ),
              ),
              Column(
                crossAxisAlignment: CrossAxisAlignment.end,
                children: [
                  Text(
                    '${position.currentPnl >= 0 ? '+' : ''}${position.currentPnl.toStringAsFixed(1)}%',
                    style: TextStyle(
                      fontSize: 16,
                      fontWeight: FontWeight.bold,
                      color: position.currentPnl >= 0 ? Colors.green : Colors.red,
                    ),
                  ),
                  Text(
                    '\$${position.currentPrice.toStringAsFixed(2)}',
                    style: TextStyle(
                      fontSize: 12,
                      color: Colors.grey[600],
                    ),
                  ),
                ],
              ),
            ],
          ),
        )),
      ],
    );
  }

  Widget _buildStrategyPerformanceSection(List<StrategyPerformance> strategies) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text(
          'Strategy Performance',
          style: TextStyle(
            fontSize: 22,
            fontWeight: FontWeight.bold,
            color: Colors.indigo,
          ),
        ),
        const SizedBox(height: 16),
        ...strategies.map((strategy) {
          Color statusColor;
          switch (strategy.status) {
            case 'hot':
              statusColor = Colors.red;
              break;
            case 'good':
              statusColor = Colors.green;
              break;
            default:
              statusColor = Colors.blue;
          }

          return Container(
            margin: const EdgeInsets.only(bottom: 12),
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
              color: statusColor.withOpacity(0.05),
              borderRadius: BorderRadius.circular(12),
              border: Border.all(color: statusColor.withOpacity(0.3)),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Expanded(
                      child: Text(
                        strategy.strategy,
                        style: const TextStyle(
                          fontSize: 16,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ),
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                      decoration: BoxDecoration(
                        color: statusColor.withOpacity(0.2),
                        borderRadius: BorderRadius.circular(12),
                      ),
                      child: Text(
                        strategy.status.toUpperCase(),
                        style: TextStyle(
                          fontSize: 10,
                          fontWeight: FontWeight.bold,
                          color: statusColor,
                        ),
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 12),
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceAround,
                  children: [
                    Column(
                      children: [
                        Text(
                          'Win Rate',
                          style: TextStyle(fontSize: 12, color: Colors.grey[600]),
                        ),
                        Text(
                          '${strategy.winRate.toStringAsFixed(1)}%',
                          style: const TextStyle(
                            fontSize: 14,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                      ],
                    ),
                    Column(
                      children: [
                        Text(
                          'Avg Return',
                          style: TextStyle(fontSize: 12, color: Colors.grey[600]),
                        ),
                        Text(
                          '${strategy.avgReturn.toStringAsFixed(1)}%',
                          style: const TextStyle(
                            fontSize: 14,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                      ],
                    ),
                    Column(
                      children: [
                        Text(
                          'Best Return',
                          style: TextStyle(fontSize: 12, color: Colors.grey[600]),
                        ),
                        Text(
                          '${strategy.bestReturn.toStringAsFixed(1)}%',
                          style: const TextStyle(
                            fontSize: 14,
                            fontWeight: FontWeight.bold,
                            color: Colors.green,
                          ),
                        ),
                      ],
                    ),
                    Column(
                      children: [
                        Text(
                          'Total Trades',
                          style: TextStyle(fontSize: 12, color: Colors.grey[600]),
                        ),
                        Text(
                          '${strategy.totalTrades}',
                          style: const TextStyle(
                            fontSize: 14,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                      ],
                    ),
                  ],
                ),
              ],
            ),
          );
        }),
      ],
    );
  }
}

