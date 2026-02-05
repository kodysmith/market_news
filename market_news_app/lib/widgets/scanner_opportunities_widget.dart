import 'package:flutter/material.dart';

import '../models/opportunity_v1.dart';
import '../services/opportunity_scanner_service.dart';

class ScannerOpportunitiesWidget extends StatefulWidget {
  const ScannerOpportunitiesWidget({Key? key}) : super(key: key);

  @override
  _ScannerOpportunitiesWidgetState createState() => _ScannerOpportunitiesWidgetState();
}

class _ScannerOpportunitiesWidgetState extends State<ScannerOpportunitiesWidget> {
  List<OpportunityV1> opportunities = [];
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

      // Small universe to start; expand later or load from remote config.
      const tickers = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'NVDA', 'TSLA', 'AMZN', 'META', 'GOOGL', 'IWM', 'TQQQ'];
      final results = await OpportunityScannerService.scan(tickers: tickers);

      setState(() {
        opportunities = results;
        summary = {
          'scan_timestamp': DateTime.now().toIso8601String(),
          'summary': {
            'tickers_scanned': tickers.length,
            'opportunities_found': results.length,
          }
        };
        isLoading = false;
      });
      
    } catch (e) {
      setState(() {
        error = 'Error loading opportunities: $e';
        isLoading = false;
      });
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
                'Opportunities found: ${summary!['summary']?['opportunities_found'] ?? 0}',
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
                  final signal = opp.opportunityType;
                  final ticker = opp.ticker;
                  final currentPrice = opp.currentPrice;
                  final confidence = opp.overallScore;
                  final rsi = opp.rsi ?? 50.0;
                  final target = opp.targetPrice ?? 0.0;
                  final stop = opp.stopLoss ?? 0.0;
                  final riskReward = opp.riskRewardRatio ?? 0.0;

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
                              confidence.toStringAsFixed(0),
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
