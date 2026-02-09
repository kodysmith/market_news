import 'package:flutter/material.dart';
import 'package:market_news_app/services/gex_service.dart';
import 'package:market_news_app/models/gex_data.dart';
import '../widgets/asset_selector_widget.dart';
import '../widgets/asset_selection_provider.dart';
import '../widgets/gex_price_bar_widget.dart';
import '../services/compute_queue_service.dart';

class GexCalculatorScreen extends StatefulWidget {
  const GexCalculatorScreen({super.key});

  @override
  State<GexCalculatorScreen> createState() => _GexCalculatorScreenState();
}

class _GexCalculatorScreenState extends State<GexCalculatorScreen> {
  GexCalculation? _gexData;
  GexSummary? _summaryData;
  MaxPainResult? _maxPainData;
  DateTime? _lastCacheUpdate;
  bool _isLoading = false;
  String? _error;
  bool _showSummary = false;
  bool _showMaxPain = false;
  int _maxPainDte = 0;

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
    // Reload GEX data when asset changes (or clear max pain result)
    if (mounted) {
      setState(() {
        _showSummary = false;
        if (_showMaxPain) _maxPainData = null;
      });
      if (!_showMaxPain) _loadGexData();
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
      if (_showMaxPain) {
        final result = await GexService.getMaxPain(_selectedTicker, dte: _maxPainDte);
        if (mounted) {
          setState(() {
            _maxPainData = result;
            _isLoading = false;
            _error = result == null
                ? 'Failed to load max pain for $_selectedTicker. Check API connection.'
                : null;
          });
        }
        return;
      }
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
        final result = await GexService.calculateGex(_selectedTicker);
        if (mounted) {
          setState(() {
            _gexData = result.calculation;
            _lastCacheUpdate = result.updatedAt;
            _isLoading = false;
            if (result.calculation == null) {
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
                  'GEX data is read from Supabase (compute_result_cache).\n\n'
                  'Set SUPABASE_URL and SUPABASE_ANON_KEY in .env. '
                  'Core symbols: ${ComputeQueueService.coreSymbols.join(", ")}.',
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
            const SizedBox(height: 12),
            if (!_showSummary && !_showMaxPain) _buildCacheStalenessBanner(),
            const SizedBox(height: 20),

            // Error Display
            if (_error != null) _buildErrorCard(),

            // Loading Indicator
            if (_isLoading) const Center(child: CircularProgressIndicator()),

            // Content
            if (!_isLoading && _error == null)
              _showMaxPain
                  ? _buildMaxPainView()
                  : (_showSummary ? _buildSummaryView() : _buildDetailView()),
          ],
        ),
      ),
    );
  }

  bool get _isCacheStale {
    if (_lastCacheUpdate == null) return false;
    return DateTime.now().difference(_lastCacheUpdate!) > cacheStalenessThreshold;
  }

  String get _cacheAgeLabel {
    if (_lastCacheUpdate == null) return '';
    final diff = DateTime.now().difference(_lastCacheUpdate!);
    if (diff.inMinutes < 1) return 'Updated just now';
    if (diff.inMinutes == 1) return 'Updated 1 min ago';
    return 'Updated ${diff.inMinutes} min ago';
  }

  Widget _buildCacheStalenessBanner() {
    if (_lastCacheUpdate == null) return const SizedBox.shrink();
    final stale = _isCacheStale;
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: stale ? const Color(0xFF3D1F00) : const Color(0xFF161B22),
        borderRadius: BorderRadius.circular(6),
        border: stale ? Border.all(color: const Color(0xFFD29922), width: 1) : null,
      ),
      child: Row(
        children: [
          Icon(
            stale ? Icons.warning_amber_rounded : Icons.schedule,
            size: 20,
            color: stale ? const Color(0xFFD29922) : Colors.white54,
          ),
          const SizedBox(width: 8),
          Expanded(
            child: Text(
              stale
                  ? 'Data may be stale ($_cacheAgeLabel). Use other sources if the service has stopped.'
                  : _cacheAgeLabel,
              style: TextStyle(
                color: stale ? const Color(0xFFD29922) : Colors.white54,
                fontSize: 13,
              ),
            ),
          ),
        ],
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
                  if (_showMaxPain) _maxPainData = null;
                });
                if (!_showMaxPain) _loadGexData();
              },
            ),
            const SizedBox(height: 12),
            Row(
              children: [
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: () {
                      setState(() {
                        _showMaxPain = false;
                        _showSummary = false;
                      });
                      _loadGexData();
                    },
                    icon: const Icon(Icons.calculate),
                    label: const Text('Calculate GEX'),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: () {
                      setState(() {
                        _showMaxPain = false;
                        _showSummary = true;
                      });
                      _loadGexData();
                    },
                    icon: const Icon(Icons.list),
                    label: const Text('View Summary'),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: () {
                      setState(() {
                        _showMaxPain = true;
                        _showSummary = false;
                        _maxPainData = null;
                        _error = null;
                      });
                      _loadGexData();
                    },
                    icon: const Icon(Icons.show_chart),
                    label: const Text('Max Pain'),
                  ),
                ),
              ],
            ),
            if (_showMaxPain) ...[
              const SizedBox(height: 12),
              Row(
                children: [
                  const Text('DTE: ', style: TextStyle(fontWeight: FontWeight.w500)),
                  DropdownButton<int>(
                    value: _maxPainDte,
                    items: [0, 1, 5]
                        .map((dte) => DropdownMenuItem(
                              value: dte,
                              child: Text('$dte DTE'),
                            ))
                        .toList(),
                    onChanged: (v) {
                      if (v != null) setState(() => _maxPainDte = v);
                    },
                  ),
                  const SizedBox(width: 12),
                  ElevatedButton.icon(
                    onPressed: _isLoading ? null : _loadGexData,
                    icon: const Icon(Icons.calculate),
                    label: const Text('Get Max Pain'),
                  ),
                ],
              ),
            ],
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

  Widget _buildMaxPainView() {
    if (_maxPainData == null) {
      return Center(
        child: Padding(
          padding: const EdgeInsets.all(24.0),
          child: Text(
            'Select DTE above and tap Get Max Pain for $_selectedTicker',
            textAlign: TextAlign.center,
            style: TextStyle(color: Colors.grey.shade600),
          ),
        ),
      );
    }
    final d = _maxPainData!;
    final diff = d.spotPrice - d.maxPainStrike;
    final diffText = diff >= 0
        ? 'Spot is \$${diff.toStringAsFixed(2)} above max pain'
        : 'Spot is \$${(-diff).toStringAsFixed(2)} below max pain';

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        _buildMetricCard('Max Pain Strike', '\$${d.maxPainStrike.toStringAsFixed(2)}', Colors.orange),
        const SizedBox(height: 12),
        _buildMetricCard('Spot Price', '\$${d.spotPrice.toStringAsFixed(2)}', Colors.blue),
        const SizedBox(height: 12),
        _buildMetricCard('Expiration', d.expiration, Colors.purple),
        const SizedBox(height: 12),
        Card(
          child: Padding(
            padding: const EdgeInsets.all(16.0),
            child: Row(
              children: [
                Icon(
                  diff >= 0 ? Icons.arrow_upward : Icons.arrow_downward,
                  color: diff >= 0 ? Colors.green : Colors.red,
                  size: 28,
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: Text(
                    diffText,
                    style: TextStyle(
                      fontSize: 16,
                      fontWeight: FontWeight.w500,
                      color: diff >= 0 ? Colors.green.shade700 : Colors.red.shade700,
                    ),
                  ),
                ),
              ],
            ),
          ),
        ),
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
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'Price & Walls',
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 16),
            Container(
              decoration: BoxDecoration(
                color: Colors.grey.shade900,
                borderRadius: BorderRadius.circular(8),
              ),
              padding: const EdgeInsets.symmetric(vertical: 8),
              child: GexPriceBarWidget(
                spot: annotations.spotPrice,
                flipLine: annotations.flipLine,
                putWall: annotations.putWall > 0 ? annotations.putWall : null,
                callWall: annotations.callWall > 0 ? annotations.callWall : null,
                height: 120,
                darkBackground: true,
              ),
            ),
            if (_gexData?.gammaSlope != null) ...[
              const SizedBox(height: 16),
              _buildGammaSlopeIndicator(_gexData!.gammaSlope!),
            ],
          ],
        ),
      ),
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

