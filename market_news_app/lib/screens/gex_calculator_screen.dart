import 'package:flutter/material.dart';
import 'package:market_news_app/services/gex_service.dart';
import 'package:market_news_app/models/gex_data.dart';
import '../widgets/asset_selector_widget.dart';
import '../widgets/asset_selection_provider.dart';
import '../main.dart' show apiBaseUrl;

class GexCalculatorScreen extends StatefulWidget {
  const GexCalculatorScreen({super.key});

  @override
  State<GexCalculatorScreen> createState() => _GexCalculatorScreenState();
}

class _GexCalculatorScreenState extends State<GexCalculatorScreen> {
  GexCalculation? _gexData;
  GexSummary? _summaryData;
  bool _isLoading = false;
  String? _error;
  bool _showSummary = false;

  @override
  void initState() {
    super.initState();
    _loadGexData();
    // Load GEX tickers from API and add to service
    _loadGexTickers();
  }
  
  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    // Set up listener after context is available
    final service = AssetSelectionProvider.of(context);
    service.addListener(_onAssetChanged);
  }
  
  @override
  void dispose() {
    try {
      final service = AssetSelectionProvider.of(context);
      service.removeListener(_onAssetChanged);
    } catch (e) {
      // Context might not be available during dispose
    }
    super.dispose();
  }
  
  void _onAssetChanged() {
    // Reload GEX data when asset changes
    if (mounted) {
      setState(() {
        _showSummary = false;
      });
      _loadGexData();
    }
  }
  
  String get _selectedTicker {
    try {
      return AssetSelectionProvider.of(context).selectedAsset;
    } catch (e) {
      return 'SPY'; // Fallback
    }
  }
  
  Future<void> _loadGexTickers() async {
    try {
      final tickers = await GexService.getGexTickers();
      // Add API tickers to service (access in didChangeDependencies or build)
      WidgetsBinding.instance.addPostFrameCallback((_) {
        try {
          final service = AssetSelectionProvider.of(context);
          for (final ticker in tickers) {
            service.addAsset(ticker);
          }
        } catch (e) {
          // Context not available yet
        }
      });
    } catch (e) {
      // Silently fail - service already has default tickers
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
            AssetSelectorWidget(
              onAssetChanged: (asset) {
                setState(() {
                  _showSummary = false;
                });
                _loadGexData();
              },
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
                painter: GexChartPainter(
                  data, 
                  annotations,
                  _gexData?.cumulativeGex ?? [],
                ),
                child: Container(),
              ),
            ),
            const SizedBox(height: 12),
            _buildChartLegend(annotations),
            if (_gexData?.gammaSlope != null) ...[
              const SizedBox(height: 16),
              _buildGammaSlopeIndicator(_gexData!.gammaSlope!),
            ],
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
        if (_gexData?.cumulativeGex.isNotEmpty ?? false)
          _buildLegendItem('Cumulative GEX', Colors.purple, Icons.show_chart),
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

  Widget _buildGammaSlopeIndicator(GammaSlope slope) {
    Color bucketColor;
    IconData bucketIcon;
    
    switch (slope.slopeBucket) {
      case 'STABILIZING':
        bucketColor = Colors.green;
        bucketIcon = Icons.trending_up;
        break;
      case 'ACCELERATIVE':
        bucketColor = Colors.red;
        bucketIcon = Icons.trending_down;
        break;
      default:
        bucketColor = Colors.orange;
        bucketIcon = Icons.trending_flat;
    }

    return Container(
      padding: const EdgeInsets.all(12.0),
      decoration: BoxDecoration(
        color: bucketColor.withOpacity(0.1),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: bucketColor.withOpacity(0.3)),
      ),
      child: Row(
        children: [
          Icon(bucketIcon, color: bucketColor, size: 24),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Text(
                      'Gamma Slope: ',
                      style: TextStyle(
                        fontSize: 14,
                        fontWeight: FontWeight.w500,
                        color: Colors.grey.shade700,
                      ),
                    ),
                    Text(
                      slope.slopeAtSpot.toStringAsFixed(2),
                      style: TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.bold,
                        color: bucketColor,
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 4),
                Row(
                  children: [
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                      decoration: BoxDecoration(
                        color: bucketColor.withOpacity(0.2),
                        borderRadius: BorderRadius.circular(12),
                      ),
                      child: Text(
                        slope.slopeBucket,
                        style: TextStyle(
                          fontSize: 12,
                          fontWeight: FontWeight.w600,
                          color: bucketColor,
                        ),
                      ),
                    ),
                    const SizedBox(width: 8),
                    Expanded(
                      child: Text(
                        slope.interpretation,
                        style: TextStyle(
                          fontSize: 12,
                          color: Colors.grey.shade600,
                        ),
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ),
          Tooltip(
            message: 'Gamma Slope measures how dealer hedging pressure changes as price moves.\n'
                'Negative slope = moves accelerate.\n'
                'Positive slope = moves stabilize.',
            child: Icon(Icons.info_outline, size: 18, color: Colors.grey.shade600),
          ),
        ],
      ),
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
          final service = AssetSelectionProvider.of(context);
          service.setAsset(ticker.ticker);
          setState(() {
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
  final List<CumulativeGexPoint> cumulativeGex;

  GexChartPainter(this.data, this.annotations, this.cumulativeGex);

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

    // Find min/max for cumulative GEX to scale the curve
    double minCumGex = 0.0;
    double maxCumGex = 0.0;
    if (cumulativeGex.isNotEmpty) {
      minCumGex = cumulativeGex.map((d) => d.cumulativeGex).reduce((a, b) => a < b ? a : b);
      maxCumGex = cumulativeGex.map((d) => d.cumulativeGex).reduce((a, b) => a > b ? a : b);
    }
    final cumGexRange = maxCumGex - minCumGex;
    final zeroCumY = cumGexRange > 0 
        ? padding + chartHeight - ((0 - minCumGex) / cumGexRange * chartHeight)
        : padding + chartHeight / 2;

    // Draw cumulative gamma curve
    // Uses its own scale (cumulative values are typically much larger)
    if (cumulativeGex.isNotEmpty && cumGexRange > 0) {
      final segments = <Path>[];
      final colors = <Color>[];
      
      for (int i = 0; i < cumulativeGex.length - 1; i++) {
        final point1 = cumulativeGex[i];
        final point2 = cumulativeGex[i + 1];
        
        if (point1.strike < minStrike || point2.strike > maxStrike) continue;
        
        // Calculate positions using cumulative GEX's own scale
        final x1 = padding + ((point1.strike - minStrike) / strikeRange * chartWidth);
        final y1 = padding + chartHeight - ((point1.cumulativeGex - minCumGex) / cumGexRange * chartHeight);
        final x2 = padding + ((point2.strike - minStrike) / strikeRange * chartWidth);
        final y2 = padding + chartHeight - ((point2.cumulativeGex - minCumGex) / cumGexRange * chartHeight);
        
        final segmentPath = Path()
          ..moveTo(x1, y1)
          ..lineTo(x2, y2);
        segments.add(segmentPath);
        
        // Color based on sign of cumulative GEX
        final avgCumGex = (point1.cumulativeGex + point2.cumulativeGex) / 2;
        colors.add(avgCumGex >= 0 ? Colors.green.shade700 : Colors.red.shade700);
      }
      
      // Draw segments with appropriate colors (thicker line for visibility)
      for (int i = 0; i < segments.length; i++) {
        final segmentPaint = Paint()
          ..style = PaintingStyle.stroke
          ..strokeWidth = 3.0
          ..color = colors[i];
        canvas.drawPath(segments[i], segmentPaint);
      }
      
      // Draw zero line for cumulative GEX (if it crosses zero)
      if (minCumGex < 0 && maxCumGex > 0) {
        final zeroCumY = padding + chartHeight - ((0 - minCumGex) / cumGexRange * chartHeight);
        final zeroCumPaint = Paint()
          ..color = Colors.grey.withOpacity(0.5)
          ..strokeWidth = 1
          ..style = PaintingStyle.stroke;
        canvas.drawLine(
          Offset(padding, zeroCumY),
          Offset(size.width - padding, zeroCumY),
          zeroCumPaint,
        );
      }
    }

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
        ..color = item.gex >= 0 ? Colors.green.withOpacity(0.6) : Colors.red.withOpacity(0.6)
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
