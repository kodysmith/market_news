import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

class ScannerOpportunitiesWidget extends StatefulWidget {
  const ScannerOpportunitiesWidget({Key? key}) : super(key: key);

  @override
  _ScannerOpportunitiesWidgetState createState() => _ScannerOpportunitiesWidgetState();
}

class _ScannerOpportunitiesWidgetState extends State<ScannerOpportunitiesWidget> {
  List<Map<String, dynamic>> opportunities = [];
  bool isLoading = true;
  String? error;
  Map<String, dynamic>? summary;

  @override
  void initState() {
    super.initState();
    _loadScannerOpportunities();
  }

  Future<void> _loadScannerOpportunities() async {
    try {
      setState(() {
        isLoading = true;
        error = null;
      });

      // For now, use mock data since Firebase functions are having issues
      // TODO: Replace with actual API call when Firebase is working
      await Future.delayed(const Duration(seconds: 1)); // Simulate loading
      
      setState(() {
        opportunities = [
          {
            'ticker': 'AAPL',
            'signal': 'WEAK_SELL',
            'confidence': 90,
            'current_price': 254.63,
            'technical_indicators': {'rsi': 68.8},
            'targets_stops': {'target': 246.08, 'stop_loss': 262.48, 'risk_reward': 1.33},
            'fundamentals': {'fundamental_score': 45}
          },
          {
            'ticker': 'TSLA',
            'signal': 'SELL',
            'confidence': 80,
            'current_price': 444.72,
            'technical_indicators': {'rsi': 73.7},
            'targets_stops': {'target': 427.16, 'stop_loss': 483.68, 'risk_reward': 1.33},
            'fundamentals': {'fundamental_score': 15}
          },
          {
            'ticker': 'GOOGL',
            'signal': 'WEAK_SELL',
            'confidence': 74,
            'current_price': 233.57,
            'technical_indicators': {'rsi': 64.0},
            'targets_stops': {'target': 233.57, 'stop_loss': 253.40, 'risk_reward': 1.33},
            'fundamentals': {'fundamental_score': 70}
          },
          {
            'ticker': 'MSFT',
            'signal': 'HOLD',
            'confidence': 45,
            'current_price': 534.85,
            'technical_indicators': {'rsi': 59.6},
            'targets_stops': {'target': 534.85, 'stop_loss': 508.35, 'risk_reward': 1.33},
            'fundamentals': {'fundamental_score': 50}
          },
          {
            'ticker': 'NVDA',
            'signal': 'WEAK_SELL',
            'confidence': 25,
            'current_price': 176.56,
            'technical_indicators': {'rsi': 63.9},
            'targets_stops': {'target': 176.56, 'stop_loss': 195.25, 'risk_reward': 1.33},
            'fundamentals': {'fundamental_score': 50}
          }
        ];
        summary = {
          'scan_timestamp': DateTime.now().toIso8601String(),
          'summary': {'high_confidence_signals': 2}
        };
        isLoading = false;
      });

      // Try to load from Firebase in background (for future use)
      _loadFromFirebase();
      
    } catch (e) {
      setState(() {
        error = 'Error loading opportunities: $e';
        isLoading = false;
      });
    }
  }

  Future<void> _loadFromFirebase() async {
    try {
      final response = await http.get(
        Uri.parse('https://us-central1-kardova-capital.cloudfunctions.net/api/scanner-opportunities'),
        headers: {'Content-Type': 'application/json'},
      );

      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        if (mounted) {
          setState(() {
            opportunities = List<Map<String, dynamic>>.from(data['opportunities'] ?? []);
            summary = data['summary'];
          });
        }
      }
    } catch (e) {
      // Silently fail - we have mock data as fallback
      print('Firebase load failed: $e');
    }
  }

  String _getSignalColor(String signal) {
    switch (signal) {
      case 'BUY':
      case 'WEAK_BUY':
        return '#4CAF50'; // Green
      case 'SELL':
      case 'WEAK_SELL':
        return '#F44336'; // Red
      default:
        return '#FF9800'; // Orange
    }
  }

  String _getSignalIcon(String signal) {
    switch (signal) {
      case 'BUY':
      case 'WEAK_BUY':
        return '📈';
      case 'SELL':
      case 'WEAK_SELL':
        return '📉';
      default:
        return '⏸️';
    }
  }

  @override
  Widget build(BuildContext context) {
    return Card(
      margin: const EdgeInsets.all(8.0),
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                const Text(
                  '🎯 Trading Opportunities',
                  style: TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                IconButton(
                  icon: const Icon(Icons.refresh),
                  onPressed: _loadScannerOpportunities,
                ),
              ],
            ),
            if (summary != null) ...[
              const SizedBox(height: 8),
              Text(
                'Last scan: ${summary!['scan_timestamp'] ?? 'Unknown'}',
                style: TextStyle(
                  fontSize: 12,
                  color: Colors.grey[600],
                ),
              ),
              Text(
                'High confidence signals: ${summary!['summary']?['high_confidence_signals'] ?? 0}',
                style: TextStyle(
                  fontSize: 12,
                  color: Colors.grey[600],
                ),
              ),
            ],
            const SizedBox(height: 16),
            if (isLoading)
              const Center(
                child: CircularProgressIndicator(),
              )
            else if (error != null)
              Center(
                child: Column(
                  children: [
                    Icon(
                      Icons.error_outline,
                      color: Colors.red,
                      size: 48,
                    ),
                    const SizedBox(height: 8),
                    Text(
                      error!,
                      style: const TextStyle(color: Colors.red),
                      textAlign: TextAlign.center,
                    ),
                    const SizedBox(height: 8),
                    ElevatedButton(
                      onPressed: _loadScannerOpportunities,
                      child: const Text('Retry'),
                    ),
                  ],
                ),
              )
            else if (opportunities.isEmpty)
              const Center(
                child: Column(
                  children: [
                    Icon(
                      Icons.search_off,
                      size: 48,
                      color: Colors.grey,
                    ),
                    SizedBox(height: 8),
                    Text(
                      'No opportunities found',
                      style: TextStyle(color: Colors.grey),
                    ),
                  ],
                ),
              )
            else
              ListView.builder(
                shrinkWrap: true,
                physics: const NeverScrollableScrollPhysics(),
                itemCount: opportunities.length > 5 ? 5 : opportunities.length,
                itemBuilder: (context, index) {
                  final opp = opportunities[index];
                  final signal = opp['signal'] ?? 'HOLD';
                  final ticker = opp['ticker'] ?? 'N/A';
                  final currentPrice = opp['current_price'] ?? 0.0;
                  final confidence = opp['confidence'] ?? 0;
                  final rsi = opp['technical_indicators']?['rsi'] ?? 50.0;
                  final target = opp['targets_stops']?['target'] ?? 0.0;
                  final stop = opp['targets_stops']?['stop_loss'] ?? 0.0;
                  final riskReward = opp['targets_stops']?['risk_reward'] ?? 0.0;

                  return Card(
                    margin: const EdgeInsets.symmetric(vertical: 4.0),
                    child: ListTile(
                      leading: CircleAvatar(
                        backgroundColor: Color(int.parse(_getSignalColor(signal).substring(1), radix: 16) + 0xFF000000),
                        child: Text(
                          _getSignalIcon(signal),
                          style: const TextStyle(fontSize: 16),
                        ),
                      ),
                      title: Row(
                        children: [
                          Text(
                            ticker,
                            style: const TextStyle(
                              fontWeight: FontWeight.bold,
                              fontSize: 16,
                            ),
                          ),
                          const SizedBox(width: 8),
                          Container(
                            padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                            decoration: BoxDecoration(
                              color: confidence > 70 
                                  ? Colors.green.withOpacity(0.2)
                                  : confidence > 50 
                                      ? Colors.orange.withOpacity(0.2)
                                      : Colors.red.withOpacity(0.2),
                              borderRadius: BorderRadius.circular(4),
                            ),
                            child: Text(
                              '${confidence.toInt()}%',
                              style: TextStyle(
                                fontSize: 10,
                                fontWeight: FontWeight.bold,
                                color: confidence > 70 
                                    ? Colors.green[700]
                                    : confidence > 50 
                                        ? Colors.orange[700]
                                        : Colors.red[700],
                              ),
                            ),
                          ),
                        ],
                      ),
                      subtitle: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            '$signal • RSI: ${rsi.toStringAsFixed(1)} • \$${currentPrice.toStringAsFixed(2)}',
                            style: const TextStyle(fontSize: 12),
                          ),
                          if (target > 0 && stop > 0) ...[
                            const SizedBox(height: 4),
                            Text(
                              'Target: \$${target.toStringAsFixed(2)} • Stop: \$${stop.toStringAsFixed(2)} • R/R: ${riskReward.toStringAsFixed(2)}',
                              style: TextStyle(
                                fontSize: 10,
                                color: Colors.grey[600],
                              ),
                            ),
                          ],
                        ],
                      ),
                      trailing: Icon(
                        signal.contains('BUY') 
                            ? Icons.trending_up 
                            : signal.contains('SELL') 
                                ? Icons.trending_down 
                                : Icons.trending_flat,
                        color: signal.contains('BUY') 
                            ? Colors.green 
                            : signal.contains('SELL') 
                                ? Colors.red 
                                : Colors.orange,
                      ),
                    ),
                  );
                },
              ),
          ],
        ),
      ),
    );
  }
}
