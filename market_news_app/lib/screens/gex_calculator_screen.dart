import 'package:flutter/material.dart';
import 'package:market_news_app/services/gex_service.dart';
import 'package:market_news_app/models/gex_data.dart';
import '../main.dart' show apiBaseUrl;

class GexCalculatorScreen extends StatefulWidget {
  const GexCalculatorScreen({super.key});

  @override
  State<GexCalculatorScreen> createState() => _GexCalculatorScreenState();
}

class _GexCalculatorScreenState extends State<GexCalculatorScreen> {
  String _selectedTicker = 'SPY';
  List<String> _availableTickers = ['SPY', 'SPX', 'QQQ'];
  GexCalculation? _gexData;
  GexSummary? _summaryData;
  bool _isLoading = false;
  bool _isLoadingTickers = false;
  String? _error;
  bool _showSummary = false;

  @override
  void initState() {
    super.initState();
    _loadTickers();
    _loadGexData();
  }

  Future<void> _loadTickers() async {
    setState(() {
      _isLoadingTickers = true;
    });
    try {
      final tickers = await GexService.getGexTickers();
      if (mounted) {
        setState(() {
          _availableTickers = tickers;
          if (tickers.isNotEmpty && !tickers.contains(_selectedTicker)) {
            _selectedTicker = tickers.first;
          }
          _isLoadingTickers = false;
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _isLoadingTickers = false;
        });
      }
    }
  }

  Future<void> _loadGexData() async {
    setState(() {
      _isLoading = true;
      _error = null;
    });

    try {
      if (_showSummary) {
        final summary = await GexService.getGexSummary();
        if (mounted) {
          setState(() {
            _summaryData = summary;
            _isLoading = false;
            if (summary == null) {
              _error = 'Failed to load GEX summary';
            }
          });
        }
      } else {
        final data = await GexService.calculateGex(_selectedTicker);
        if (mounted) {
          setState(() {
            _gexData = data;
            _isLoading = false;
            if (data == null) {
              _error = 'Failed to load GEX data for $_selectedTicker. Check API connection and ensure GEX endpoints are available.';
            } else {
              print('✅ GEX data loaded successfully for $_selectedTicker');
            }
          });
        }
      }
    } catch (e, stackTrace) {
      print('❌ Exception in _loadGexData: $e');
      print('Stack trace: $stackTrace');
      if (mounted) {
        setState(() {
          _isLoading = false;
          _error = 'Error loading GEX data: $e';
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('⚡ GEX Calculator'),
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh),
            onPressed: _loadGexData,
            tooltip: 'Refresh',
          ),
        ],
      ),
      // Debug banner showing API URL
      persistentFooterButtons: [
        TextButton.icon(
          onPressed: () {
            showDialog(
              context: context,
              builder: (context) => AlertDialog(
                title: const Text('API Configuration'),
                content: Text(
                  'Current API URL: ${apiBaseUrl}\n\n'
                  'For local GEX testing, update apiBaseUrl in lib/main.dart to:\n'
                  'http://localhost:5000 (for web)\n'
                  'or\n'
                  'http://192.168.1.31:5000 (for mobile/network)',
                ),
                actions: [
                  TextButton(
                    onPressed: () => Navigator.pop(context),
                    child: const Text('OK'),
                  ),
                ],
              ),
            );
          },
          icon: const Icon(Icons.info, size: 16),
          label: const Text('API Info', style: TextStyle(fontSize: 12)),
        ),
      ],
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Controls Section
            _buildControlsSection(),
            const SizedBox(height: 20),

            // Error Display
            if (_error != null) _buildErrorCard(),

            // Loading Indicator
            if (_isLoading) const Center(child: CircularProgressIndicator()),

            // Content
            if (!_isLoading && _error == null)
              _showSummary ? _buildSummaryView() : _buildDetailView(),
          ],
        ),
      ),
    );
  }

  Widget _buildControlsSection() {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            Row(
              children: [
                Expanded(
                  child: DropdownButtonFormField<String>(
                    value: _selectedTicker,
                    decoration: const InputDecoration(
                      labelText: 'Ticker',
                      border: OutlineInputBorder(),
                    ),
                    items: _availableTickers
                        .map((ticker) => DropdownMenuItem(
                              value: ticker,
                              child: Text(ticker),
                            ))
                        .toList(),
                    onChanged: (value) {
                      if (value != null) {
                        setState(() {
                          _selectedTicker = value;
                          _showSummary = false;
                        });
                        _loadGexData();
                      }
                    },
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: TextFormField(
                    decoration: const InputDecoration(
                      labelText: 'Custom Ticker',
                      border: OutlineInputBorder(),
                      hintText: 'Enter ticker',
                    ),
                    onFieldSubmitted: (value) {
                      if (value.isNotEmpty) {
                        setState(() {
                          _selectedTicker = value.toUpperCase();
                          _showSummary = false;
                        });
                        _loadGexData();
                      }
                    },
                  ),
                ),
              ],
            ),
            const SizedBox(height: 12),
            Row(
              children: [
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: _loadGexData,
                    icon: const Icon(Icons.calculate),
                    label: const Text('Calculate GEX'),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: () {
                      setState(() {
                        _showSummary = true;
                      });
                      _loadGexData();
                    },
                    icon: const Icon(Icons.list),
                    label: const Text('View Summary'),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildErrorCard() {
    return Card(
      color: Colors.red.shade50,
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Row(
          children: [
            const Icon(Icons.error, color: Colors.red),
            const SizedBox(width: 12),
            Expanded(
              child: Text(
                _error!,
                style: const TextStyle(color: Colors.red),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildDetailView() {
    if (_gexData == null) {
      return const Center(
        child: Text('Select a ticker and click Calculate GEX'),
      );
    }

    final metrics = _gexData!.metrics;
    final breakdown = _gexData!.breakdown;
    final annotations = _gexData!.chartAnnotations;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // Metrics Cards
        _buildMetricsCards(metrics),
        const SizedBox(height: 20),

        // Breakdown Card
        _buildBreakdownCard(breakdown),
        const SizedBox(height: 20),

        // Chart Section
        _buildChartSection(_gexData!.gexByStrike, annotations),
        const SizedBox(height: 20),

        // Data Table
        _buildDataTable(_gexData!.gexByStrike),
        const SizedBox(height: 20),

        // Debug Section
        _buildDebugSection(),
      ],
    );
  }

  Widget _buildMetricsCards(GexMetrics metrics) {
    return GridView.count(
      crossAxisCount: MediaQuery.of(context).size.width > 600 ? 3 : 2,
      shrinkWrap: true,
      physics: const NeverScrollableScrollPhysics(),
      crossAxisSpacing: 12,
      mainAxisSpacing: 12,
      childAspectRatio: 1.5,
      children: [
        _buildMetricCard('Spot Price', '\$${metrics.spotPrice.toStringAsFixed(2)}', Colors.blue),
        _buildMetricCard('Total GEX', '${_formatLargeNumber(metrics.totalGex)}', Colors.green),
        _buildMetricCard('Put Wall', '\$${metrics.putWall.toStringAsFixed(2)}', Colors.red),
        _buildMetricCard('Call Wall', '\$${metrics.callWall.toStringAsFixed(2)}', Colors.orange),
        _buildMetricCard(
          'Flip Line',
          metrics.flipLine != null ? '\$${metrics.flipLine!.toStringAsFixed(2)}' : 'N/A',
          Colors.purple,
        ),
        _buildMetricCard(
          'Regime',
          metrics.regime.toUpperCase(),
          metrics.regime == 'positive' ? Colors.green : Colors.red,
        ),
      ],
    );
  }

  Widget _buildMetricCard(String label, String value, Color color) {
    return Card(
      elevation: 2,
      child: Padding(
        padding: const EdgeInsets.all(12.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
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
            const SizedBox(height: 8),
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
      ),
    );
  }

  Widget _buildBreakdownCard(GexBreakdown breakdown) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'Breakdown',
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 12),
            Row(
              children: [
                Expanded(
                  child: _buildBreakdownItem('Call GEX', breakdown.callGex, Colors.green),
                ),
                Expanded(
                  child: _buildBreakdownItem('Put GEX', breakdown.putGex, Colors.red),
                ),
              ],
            ),
            const SizedBox(height: 12),
            Row(
              children: [
                Expanded(
                  child: _buildBreakdownItem('Call Contracts', breakdown.callContracts.toDouble(), Colors.blue),
                ),
                Expanded(
                  child: _buildBreakdownItem('Put Contracts', breakdown.putContracts.toDouble(), Colors.orange),
                ),
              ],
            ),
            const SizedBox(height: 8),
            Text(
              'Total Contracts: ${breakdown.totalContracts} | Skipped: ${breakdown.skippedContracts}',
              style: TextStyle(fontSize: 12, color: Colors.grey.shade600),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildBreakdownItem(String label, double value, Color color) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          label,
          style: TextStyle(fontSize: 12, color: Colors.grey.shade600),
        ),
        const SizedBox(height: 4),
        Text(
          _formatLargeNumber(value),
          style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold, color: color),
        ),
      ],
    );
  }

  Widget _buildChartSection(List<GexByStrike> data, ChartAnnotations annotations) {
    if (data.isEmpty) {
      return const Card(
        child: Padding(
          padding: EdgeInsets.all(16.0),
          child: Text('No chart data available'),
        ),
      );
    }

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'GEX by Strike',
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 16),
            SizedBox(
              height: 300,
              child: CustomPaint(
                painter: GexChartPainter(data, annotations),
                child: Container(),
              ),
            ),
            const SizedBox(height: 12),
            _buildChartLegend(annotations),
          ],
        ),
      ),
    );
  }

  Widget _buildChartLegend(ChartAnnotations annotations) {
    return Wrap(
      spacing: 16,
      runSpacing: 8,
      children: [
        _buildLegendItem('Spot', Colors.green, Icons.circle),
        if (annotations.flipLine != null)
          _buildLegendItem('Flip Line', Colors.red, Icons.remove),
        _buildLegendItem('Put Wall', Colors.orange, Icons.arrow_downward),
        _buildLegendItem('Call Wall', Colors.blue, Icons.arrow_upward),
      ],
    );
  }

  Widget _buildLegendItem(String label, Color color, IconData icon) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Icon(icon, color: color, size: 16),
        const SizedBox(width: 4),
        Text(label, style: TextStyle(fontSize: 12, color: color)),
      ],
    );
  }

  Widget _buildDataTable(List<GexByStrike> data) {
    if (data.isEmpty) {
      return const SizedBox.shrink();
    }

    return Card(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Padding(
            padding: EdgeInsets.all(16.0),
            child: Text(
              'GEX by Strike (Detailed)',
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
          ),
          SingleChildScrollView(
            scrollDirection: Axis.horizontal,
            child: DataTable(
              columns: const [
                DataColumn(label: Text('Strike')),
                DataColumn(label: Text('GEX')),
                DataColumn(label: Text('Cumulative GEX')),
              ],
              rows: data.take(50).map((item) {
                return DataRow(
                  cells: [
                    DataCell(Text('\$${item.strike.toStringAsFixed(2)}')),
                    DataCell(Text(_formatLargeNumber(item.gex))),
                    DataCell(Text(_formatLargeNumber(item.cumulativeGex))),
                  ],
                );
              }).toList(),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildSummaryView() {
    if (_summaryData == null || _summaryData!.tickers.isEmpty) {
      return const Center(
        child: Text('No summary data available'),
      );
    }

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text(
          'GEX Summary (All Tickers)',
          style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
        ),
        const SizedBox(height: 16),
        ..._summaryData!.tickers.map((ticker) => _buildSummaryCard(ticker)),
        if (_summaryData!.errors != null && _summaryData!.errors!.isNotEmpty) ...[
          const SizedBox(height: 20),
          Card(
            color: Colors.orange.shade50,
            child: Padding(
              padding: const EdgeInsets.all(16.0),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text(
                    'Errors',
                    style: TextStyle(fontWeight: FontWeight.bold),
                  ),
                  ..._summaryData!.errors!.map((error) => Text(
                        '${error['ticker']}: ${error['error']}',
                        style: const TextStyle(color: Colors.orange),
                      )),
                ],
              ),
            ),
          ),
        ],
      ],
    );
  }

  Widget _buildSummaryCard(GexTickerSummary ticker) {
    return Card(
      margin: const EdgeInsets.only(bottom: 12),
      child: InkWell(
        onTap: () {
          setState(() {
            _selectedTicker = ticker.ticker;
            _showSummary = false;
          });
          _loadGexData();
        },
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Text(
                    ticker.ticker,
                    style: const TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
                  ),
                  Chip(
                    label: Text(ticker.regime.toUpperCase()),
                    backgroundColor: ticker.regime == 'positive' ? Colors.green.shade100 : Colors.red.shade100,
                  ),
                ],
              ),
              const SizedBox(height: 12),
              Row(
                children: [
                  Expanded(
                    child: _buildSummaryItem('Spot', '\$${ticker.spot.toStringAsFixed(2)}'),
                  ),
                  Expanded(
                    child: _buildSummaryItem(
                      'Flip Line',
                      ticker.flipLine != null ? '\$${ticker.flipLine!.toStringAsFixed(2)}' : 'N/A',
                    ),
                  ),
                  Expanded(
                    child: _buildSummaryItem('Total GEX', _formatLargeNumber(ticker.totalGex)),
                  ),
                ],
              ),
              const SizedBox(height: 8),
              Row(
                children: [
                  Expanded(
                    child: _buildSummaryItem('Put Wall', '\$${ticker.putWall.toStringAsFixed(2)}'),
                  ),
                  Expanded(
                    child: _buildSummaryItem('Call Wall', '\$${ticker.callWall.toStringAsFixed(2)}'),
                  ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildSummaryItem(String label, String value) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          label,
          style: TextStyle(fontSize: 12, color: Colors.grey.shade600),
        ),
        const SizedBox(height: 4),
        Text(
          value,
          style: const TextStyle(fontSize: 14, fontWeight: FontWeight.bold),
        ),
      ],
    );
  }

  Widget _buildDebugSection() {
    if (_gexData == null) return const SizedBox.shrink();

    return ExpansionTile(
      title: const Text('🔍 Debug Information'),
      children: [
        Padding(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text('Ticker: ${_gexData!.ticker}'),
              Text('Spot Price: \$${_gexData!.spotPrice.toStringAsFixed(2)}'),
              Text('Timestamp: ${DateTime.fromMillisecondsSinceEpoch((_gexData!.timestamp * 1000).toInt())}'),
              Text('Total Strikes: ${_gexData!.gexByStrike.length}'),
            ],
          ),
        ),
      ],
    );
  }

  String _formatLargeNumber(double value) {
    if (value.abs() >= 1000000) {
      return '${(value / 1000000).toStringAsFixed(2)}M';
    } else if (value.abs() >= 1000) {
      return '${(value / 1000).toStringAsFixed(2)}K';
    } else {
      return value.toStringAsFixed(0);
    }
  }
}

// Simple chart painter for GEX visualization
class GexChartPainter extends CustomPainter {
  final List<GexByStrike> data;
  final ChartAnnotations annotations;

  GexChartPainter(this.data, this.annotations);

  @override
  void paint(Canvas canvas, Size size) {
    if (data.isEmpty) return;

    final padding = 40.0;
    final chartWidth = size.width - (padding * 2);
    final chartHeight = size.height - (padding * 2);

    // Find min/max values
    double minGex = data.map((d) => d.gex).reduce((a, b) => a < b ? a : b);
    double maxGex = data.map((d) => d.gex).reduce((a, b) => a > b ? a : b);
    double minStrike = data.map((d) => d.strike).reduce((a, b) => a < b ? a : b);
    double maxStrike = data.map((d) => d.strike).reduce((a, b) => a > b ? a : b);

    // Normalize to include zero
    minGex = minGex < 0 ? minGex * 1.1 : 0;
    maxGex = maxGex > 0 ? maxGex * 1.1 : 0;

    final gexRange = maxGex - minGex;
    final strikeRange = maxStrike - minStrike;

    // Draw zero line
    final zeroY = padding + chartHeight - ((0 - minGex) / gexRange * chartHeight);
    final zeroPaint = Paint()
      ..color = Colors.grey
      ..strokeWidth = 1;
    canvas.drawLine(
      Offset(padding, zeroY),
      Offset(size.width - padding, zeroY),
      zeroPaint,
    );

    // Draw bars
    final barWidth = chartWidth / data.length;
    for (int i = 0; i < data.length; i++) {
      final item = data[i];
      final x = padding + (i * barWidth);
      final barHeight = (item.gex.abs() / gexRange * chartHeight);
      final barY = item.gex >= 0
          ? zeroY - barHeight
          : zeroY;

      final barPaint = Paint()
        ..color = item.gex >= 0 ? Colors.green : Colors.red
        ..style = PaintingStyle.fill;
      canvas.drawRect(
        Rect.fromLTWH(x, barY, barWidth * 0.8, barHeight),
        barPaint,
      );
    }

    // Draw annotations
    final annotationPaint = Paint()
      ..strokeWidth = 2
      ..style = PaintingStyle.stroke;

    // Spot price
    if (annotations.spotPrice >= minStrike && annotations.spotPrice <= maxStrike) {
      final spotX = padding + ((annotations.spotPrice - minStrike) / strikeRange * chartWidth);
      annotationPaint.color = Colors.green;
      canvas.drawLine(
        Offset(spotX, padding),
        Offset(spotX, size.height - padding),
        annotationPaint,
      );
    }

    // Flip line
    if (annotations.flipLine != null &&
        annotations.flipLine! >= minStrike &&
        annotations.flipLine! <= maxStrike) {
      final flipX = padding + ((annotations.flipLine! - minStrike) / strikeRange * chartWidth);
      annotationPaint.color = Colors.red;
      annotationPaint.style = PaintingStyle.stroke;
      canvas.drawLine(
        Offset(flipX, padding),
        Offset(flipX, size.height - padding),
        annotationPaint,
      );
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => true;
}
