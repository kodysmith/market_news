import 'package:flutter/material.dart';
import '../services/valuation_service.dart';
import '../models/valuation_data.dart';
import '../main.dart' show apiBaseUrl;

class IntrinsicValueScreen extends StatefulWidget {
  const IntrinsicValueScreen({super.key});

  @override
  State<IntrinsicValueScreen> createState() => _IntrinsicValueScreenState();
}

class _IntrinsicValueScreenState extends State<IntrinsicValueScreen>
    with SingleTickerProviderStateMixin {
  late TabController _tabController;
  final TextEditingController _tickerController = TextEditingController();
  
  bool _isLoading = false;
  ValuationResult? _currentResult;
  List<ValuationHistory> _history = [];
  List<WatchlistItem> _watchlist = [];
  String _errorMessage = '';
  
  // Popular tickers for quick access
  final List<String> _popularTickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B'];

  @override
  void initState() {
    super.initState();
    _tabController = TabController(length: 3, vsync: this);
    _loadWatchlist();
  }

  @override
  void dispose() {
    _tabController.dispose();
    _tickerController.dispose();
    super.dispose();
  }

  Future<void> _loadWatchlist() async {
    final watchlist = await ValuationService.getWatchlist();
    setState(() {
      _watchlist = watchlist;
    });
  }

  Future<void> _calculateValuation(String ticker) async {
    if (ticker.isEmpty) return;
    
    setState(() {
      _isLoading = true;
      _errorMessage = '';
    });
    
    try {
      final result = await ValuationService.calculateIntrinsicValue(ticker);
      
      if (result != null) {
        final history = await ValuationService.getHistory(ticker, days: 365);
        
        setState(() {
          _currentResult = result;
          _history = history;
          _isLoading = false;
        });
      } else {
        setState(() {
          _errorMessage = 'Failed to calculate intrinsic value for $ticker';
          _isLoading = false;
        });
      }
    } catch (e) {
      setState(() {
        _errorMessage = 'Error: $e';
        _isLoading = false;
      });
    }
  }

  Future<void> _addToWatchlist(String ticker) async {
    final success = await ValuationService.addToWatchlist(ticker);
    if (success) {
      _loadWatchlist();
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Added $ticker to watchlist')),
      );
    }
  }

  Future<void> _removeFromWatchlist(String ticker) async {
    final success = await ValuationService.removeFromWatchlist(ticker);
    if (success) {
      _loadWatchlist();
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Removed $ticker from watchlist')),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Intrinsic Value Calculator'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        bottom: TabBar(
          controller: _tabController,
          tabs: const [
            Tab(text: 'Calculate', icon: Icon(Icons.calculate)),
            Tab(text: 'Watchlist', icon: Icon(Icons.star)),
            Tab(text: 'History', icon: Icon(Icons.history)),
          ],
        ),
      ),
      body: TabBarView(
        controller: _tabController,
        children: [
          _buildCalculateTab(),
          _buildWatchlistTab(),
          _buildHistoryTab(),
        ],
      ),
    );
  }

  Widget _buildCalculateTab() {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // API URL indicator
          Container(
            padding: const EdgeInsets.all(8),
            decoration: BoxDecoration(
              color: Colors.grey[200],
              borderRadius: BorderRadius.circular(8),
            ),
            child: Text(
              'API: $apiBaseUrl',
              style: TextStyle(fontSize: 12, color: Colors.grey[600]),
            ),
          ),
          const SizedBox(height: 16),
          
          // Ticker input
          Row(
            children: [
              Expanded(
                child: TextField(
                  controller: _tickerController,
                  decoration: InputDecoration(
                    labelText: 'Enter Ticker Symbol',
                    hintText: 'e.g., AAPL',
                    border: const OutlineInputBorder(),
                    suffixIcon: IconButton(
                      icon: const Icon(Icons.search),
                      onPressed: () => _calculateValuation(_tickerController.text.trim().toUpperCase()),
                    ),
                  ),
                  textCapitalization: TextCapitalization.characters,
                  onSubmitted: (value) => _calculateValuation(value.trim().toUpperCase()),
                ),
              ),
              const SizedBox(width: 8),
              ElevatedButton(
                onPressed: _isLoading
                    ? null
                    : () => _calculateValuation(_tickerController.text.trim().toUpperCase()),
                child: _isLoading
                    ? const SizedBox(
                        width: 20,
                        height: 20,
                        child: CircularProgressIndicator(strokeWidth: 2),
                      )
                    : const Text('Calculate'),
              ),
            ],
          ),
          const SizedBox(height: 16),
          
          // Popular tickers
          Wrap(
            spacing: 8,
            runSpacing: 8,
            children: _popularTickers.map((ticker) {
              return ActionChip(
                label: Text(ticker),
                onPressed: () {
                  _tickerController.text = ticker;
                  _calculateValuation(ticker);
                },
              );
            }).toList(),
          ),
          const SizedBox(height: 24),
          
          // Error message
          if (_errorMessage.isNotEmpty)
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: Colors.red[50],
                borderRadius: BorderRadius.circular(8),
                border: Border.all(color: Colors.red[200]!),
              ),
              child: Text(_errorMessage, style: TextStyle(color: Colors.red[700])),
            ),
          
          // Results
          if (_currentResult != null) ...[
            _buildResultsSection(),
          ],
        ],
      ),
    );
  }

  Widget _buildResultsSection() {
    final result = _currentResult!;
    final composite = result.composite;
    
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // Company header
        Card(
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            result.ticker,
                            style: const TextStyle(fontSize: 24, fontWeight: FontWeight.bold),
                          ),
                          Text(
                            result.companyName ?? '',
                            style: TextStyle(fontSize: 16, color: Colors.grey[600]),
                          ),
                          Text(
                            '${result.sector ?? ''} • ${result.industry ?? ''}',
                            style: TextStyle(fontSize: 12, color: Colors.grey[500]),
                          ),
                        ],
                      ),
                    ),
                    IconButton(
                      icon: const Icon(Icons.star_border),
                      onPressed: () => _addToWatchlist(result.ticker),
                      tooltip: 'Add to Watchlist',
                    ),
                  ],
                ),
              ],
            ),
          ),
        ),
        const SizedBox(height: 16),
        
        // Price and composite value
        Row(
          children: [
            Expanded(
              child: _buildMetricCard(
                'Current Price',
                '\$${result.currentPrice.toStringAsFixed(2)}',
                Colors.blue,
              ),
            ),
            const SizedBox(width: 8),
            Expanded(
              child: _buildMetricCard(
                'Intrinsic Value',
                composite.value != null
                    ? '\$${composite.value!.toStringAsFixed(2)}'
                    : 'N/A',
                Colors.purple,
              ),
            ),
          ],
        ),
        const SizedBox(height: 8),
        Row(
          children: [
            Expanded(
              child: _buildMetricCard(
                'Divergence',
                composite.divergencePct != null
                    ? '${composite.divergencePct! >= 0 ? '+' : ''}${composite.divergencePct!.toStringAsFixed(1)}%'
                    : 'N/A',
                _getDivergenceColor(composite.divergencePct),
              ),
            ),
            const SizedBox(width: 8),
            Expanded(
              child: _buildMetricCard(
                'Verdict',
                composite.verdict,
                _getVerdictColor(composite.verdict),
              ),
            ),
          ],
        ),
        const SizedBox(height: 24),
        
        // Valuation methods breakdown
        const Text(
          'Valuation Methods',
          style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
        ),
        const SizedBox(height: 8),
        ...result.applicableMethods.map((entry) {
          return _buildMethodCard(entry.key, entry.value, result.currentPrice);
        }).toList(),
        
        // Non-applicable methods
        if (result.valuations.values.any((v) => !v.applicable)) ...[
          const SizedBox(height: 16),
          ExpansionTile(
            title: Text(
              'Non-Applicable Methods',
              style: TextStyle(color: Colors.grey[600]),
            ),
            children: result.valuations.entries
                .where((e) => !e.value.applicable)
                .map((entry) {
              return ListTile(
                title: Text(_getMethodName(entry.key)),
                subtitle: Text(entry.value.reason ?? 'Not applicable'),
                leading: const Icon(Icons.info_outline, color: Colors.grey),
              );
            }).toList(),
          ),
        ],
      ],
    );
  }

  Widget _buildMetricCard(String label, String value, Color color) {
    return Card(
      color: color.withOpacity(0.1),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              label,
              style: TextStyle(fontSize: 12, color: color.withOpacity(0.8)),
            ),
            const SizedBox(height: 4),
            Text(
              value,
              style: TextStyle(
                fontSize: 20,
                fontWeight: FontWeight.bold,
                color: color,
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildMethodCard(String methodKey, ValuationMethod method, double currentPrice) {
    final divergence = method.value != null
        ? ((currentPrice - method.value!) / method.value!) * 100
        : null;
    
    return Card(
      margin: const EdgeInsets.symmetric(vertical: 4),
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Text(
                  _getMethodName(methodKey),
                  style: const TextStyle(fontWeight: FontWeight.bold),
                ),
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
                  decoration: BoxDecoration(
                    color: _getConfidenceColor(method.confidence).withOpacity(0.2),
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Text(
                    '${(method.confidence * 100).toInt()}% confidence',
                    style: TextStyle(
                      fontSize: 12,
                      color: _getConfidenceColor(method.confidence),
                    ),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 8),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Text(
                  'Value: \$${method.value?.toStringAsFixed(2) ?? 'N/A'}',
                  style: const TextStyle(fontSize: 16),
                ),
                if (divergence != null)
                  Text(
                    '${divergence >= 0 ? '+' : ''}${divergence.toStringAsFixed(1)}%',
                    style: TextStyle(
                      fontSize: 16,
                      fontWeight: FontWeight.bold,
                      color: _getDivergenceColor(divergence),
                    ),
                  ),
              ],
            ),
            // Visual divergence bar
            if (method.value != null) ...[
              const SizedBox(height: 8),
              _buildDivergenceBar(currentPrice, method.value!),
            ],
          ],
        ),
      ),
    );
  }

  Widget _buildDivergenceBar(double price, double intrinsicValue) {
    final ratio = price / intrinsicValue;
    final barPosition = (ratio - 0.5).clamp(0.0, 1.0); // 0.5x to 1.5x range
    
    return SizedBox(
      height: 20,
      child: Stack(
        children: [
          // Background gradient
          Container(
            decoration: BoxDecoration(
              gradient: const LinearGradient(
                colors: [Colors.green, Colors.yellow, Colors.red],
              ),
              borderRadius: BorderRadius.circular(4),
            ),
          ),
          // Position marker
          Positioned(
            left: barPosition * (MediaQuery.of(context).size.width - 80),
            top: 0,
            bottom: 0,
            child: Container(
              width: 4,
              color: Colors.black,
            ),
          ),
          // Labels
          const Positioned(
            left: 4,
            top: 2,
            child: Text('Undervalued', style: TextStyle(fontSize: 10, color: Colors.white)),
          ),
          const Positioned(
            right: 4,
            top: 2,
            child: Text('Overvalued', style: TextStyle(fontSize: 10, color: Colors.white)),
          ),
        ],
      ),
    );
  }

  Widget _buildWatchlistTab() {
    return RefreshIndicator(
      onRefresh: _loadWatchlist,
      child: _watchlist.isEmpty
          ? const Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Icon(Icons.star_border, size: 64, color: Colors.grey),
                  SizedBox(height: 16),
                  Text(
                    'No stocks in watchlist',
                    style: TextStyle(fontSize: 18, color: Colors.grey),
                  ),
                  SizedBox(height: 8),
                  Text(
                    'Add stocks from the Calculate tab',
                    style: TextStyle(color: Colors.grey),
                  ),
                ],
              ),
            )
          : ListView.builder(
              padding: const EdgeInsets.all(16),
              itemCount: _watchlist.length,
              itemBuilder: (context, index) {
                final item = _watchlist[index];
                return _buildWatchlistCard(item);
              },
            ),
    );
  }

  Widget _buildWatchlistCard(WatchlistItem item) {
    return Card(
      margin: const EdgeInsets.only(bottom: 8),
      child: ListTile(
        leading: CircleAvatar(
          backgroundColor: _getDivergenceColor(item.divergencePct).withOpacity(0.2),
          child: Text(
            item.ticker.substring(0, item.ticker.length > 2 ? 2 : item.ticker.length),
            style: TextStyle(
              fontWeight: FontWeight.bold,
              color: _getDivergenceColor(item.divergencePct),
            ),
          ),
        ),
        title: Text(item.ticker),
        subtitle: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(item.companyName ?? ''),
            if (item.price != null && item.compositeValue != null)
              Text(
                '\$${item.price!.toStringAsFixed(2)} vs \$${item.compositeValue!.toStringAsFixed(2)}',
                style: TextStyle(color: Colors.grey[600]),
              ),
          ],
        ),
        trailing: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          crossAxisAlignment: CrossAxisAlignment.end,
          children: [
            Text(
              item.divergencePct != null
                  ? '${item.divergencePct! >= 0 ? '+' : ''}${item.divergencePct!.toStringAsFixed(1)}%'
                  : 'N/A',
              style: TextStyle(
                fontWeight: FontWeight.bold,
                fontSize: 16,
                color: _getDivergenceColor(item.divergencePct),
              ),
            ),
            Text(
              item.verdict ?? '',
              style: TextStyle(fontSize: 12, color: Colors.grey[600]),
            ),
          ],
        ),
        onTap: () {
          _tickerController.text = item.ticker;
          _calculateValuation(item.ticker);
          _tabController.animateTo(0);
        },
        onLongPress: () {
          showDialog(
            context: context,
            builder: (context) => AlertDialog(
              title: Text('Remove ${item.ticker}?'),
              content: const Text('Remove this stock from your watchlist?'),
              actions: [
                TextButton(
                  onPressed: () => Navigator.pop(context),
                  child: const Text('Cancel'),
                ),
                TextButton(
                  onPressed: () {
                    Navigator.pop(context);
                    _removeFromWatchlist(item.ticker);
                  },
                  child: const Text('Remove', style: TextStyle(color: Colors.red)),
                ),
              ],
            ),
          );
        },
      ),
    );
  }

  Widget _buildHistoryTab() {
    if (_currentResult == null) {
      return const Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(Icons.history, size: 64, color: Colors.grey),
            SizedBox(height: 16),
            Text(
              'No history available',
              style: TextStyle(fontSize: 18, color: Colors.grey),
            ),
            SizedBox(height: 8),
            Text(
              'Calculate a stock first to see history',
              style: TextStyle(color: Colors.grey),
            ),
          ],
        ),
      );
    }

    return SingleChildScrollView(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            '${_currentResult!.ticker} Historical Data',
            style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
          ),
          const SizedBox(height: 8),
          Text(
            '${_history.length} records',
            style: TextStyle(color: Colors.grey[600]),
          ),
          const SizedBox(height: 16),
          
          // History chart placeholder
          if (_history.isNotEmpty)
            Container(
              height: 200,
              decoration: BoxDecoration(
                color: Colors.grey[100],
                borderRadius: BorderRadius.circular(8),
              ),
              child: CustomPaint(
                painter: _HistoryChartPainter(_history),
                size: const Size(double.infinity, 200),
              ),
            ),
          
          const SizedBox(height: 16),
          
          // History table
          if (_history.isNotEmpty)
            SingleChildScrollView(
              scrollDirection: Axis.horizontal,
              child: DataTable(
                columns: const [
                  DataColumn(label: Text('Date')),
                  DataColumn(label: Text('Price')),
                  DataColumn(label: Text('Intrinsic')),
                  DataColumn(label: Text('Divergence')),
                ],
                rows: _history.reversed.take(30).map((h) {
                  return DataRow(cells: [
                    DataCell(Text(h.date)),
                    DataCell(Text('\$${h.price.toStringAsFixed(2)}')),
                    DataCell(Text(h.compositeValue != null
                        ? '\$${h.compositeValue!.toStringAsFixed(2)}'
                        : 'N/A')),
                    DataCell(Text(
                      h.divergencePct != null
                          ? '${h.divergencePct! >= 0 ? '+' : ''}${h.divergencePct!.toStringAsFixed(1)}%'
                          : 'N/A',
                      style: TextStyle(color: _getDivergenceColor(h.divergencePct)),
                    )),
                  ]);
                }).toList(),
              ),
            ),
        ],
      ),
    );
  }

  String _getMethodName(String key) {
    switch (key) {
      case 'dcf':
        return 'DCF (Discounted Cash Flow)';
      case 'graham':
        return 'Graham Number';
      case 'ddm':
        return 'Dividend Discount Model';
      case 'epv':
        return 'Earnings Power Value';
      case 'peer_comps':
        return 'Peer Comparable Analysis';
      case 'nav':
        return 'Net Asset Value';
      default:
        return key.toUpperCase();
    }
  }

  Color _getDivergenceColor(double? divergence) {
    if (divergence == null) return Colors.grey;
    if (divergence < -20) return Colors.green;
    if (divergence < -10) return Colors.lightGreen;
    if (divergence <= 10) return Colors.orange;
    if (divergence <= 20) return Colors.deepOrange;
    return Colors.red;
  }

  Color _getVerdictColor(String verdict) {
    if (verdict.contains('Undervalued')) return Colors.green;
    if (verdict.contains('Overvalued')) return Colors.red;
    if (verdict.contains('Fair')) return Colors.orange;
    return Colors.grey;
  }

  Color _getConfidenceColor(double confidence) {
    if (confidence >= 0.8) return Colors.green;
    if (confidence >= 0.6) return Colors.orange;
    return Colors.red;
  }
}

/// Simple line chart painter for history
class _HistoryChartPainter extends CustomPainter {
  final List<ValuationHistory> history;

  _HistoryChartPainter(this.history);

  @override
  void paint(Canvas canvas, Size size) {
    if (history.isEmpty) return;

    final pricePaint = Paint()
      ..color = Colors.blue
      ..strokeWidth = 2
      ..style = PaintingStyle.stroke;

    final intrinsicPaint = Paint()
      ..color = Colors.purple
      ..strokeWidth = 2
      ..style = PaintingStyle.stroke;

    // Find min/max values
    double minPrice = double.infinity;
    double maxPrice = 0;
    for (var h in history) {
      if (h.price < minPrice) minPrice = h.price;
      if (h.price > maxPrice) maxPrice = h.price;
      if (h.compositeValue != null) {
        if (h.compositeValue! < minPrice) minPrice = h.compositeValue!;
        if (h.compositeValue! > maxPrice) maxPrice = h.compositeValue!;
      }
    }

    final padding = (maxPrice - minPrice) * 0.1;
    minPrice -= padding;
    maxPrice += padding;
    final range = maxPrice - minPrice;

    if (range == 0) return;

    final pricePath = Path();
    final intrinsicPath = Path();

    for (var i = 0; i < history.length; i++) {
      final x = (i / (history.length - 1)) * size.width;
      final priceY = size.height - ((history[i].price - minPrice) / range) * size.height;

      if (i == 0) {
        pricePath.moveTo(x, priceY);
      } else {
        pricePath.lineTo(x, priceY);
      }

      if (history[i].compositeValue != null) {
        final intrinsicY = size.height - ((history[i].compositeValue! - minPrice) / range) * size.height;
        if (i == 0 || history[i - 1].compositeValue == null) {
          intrinsicPath.moveTo(x, intrinsicY);
        } else {
          intrinsicPath.lineTo(x, intrinsicY);
        }
      }
    }

    canvas.drawPath(pricePath, pricePaint);
    canvas.drawPath(intrinsicPath, intrinsicPaint);

    // Legend
    final textPainter = TextPainter(
      text: const TextSpan(
        text: '— Price  ',
        style: TextStyle(color: Colors.blue, fontSize: 12),
        children: [
          TextSpan(
            text: '— Intrinsic Value',
            style: TextStyle(color: Colors.purple),
          ),
        ],
      ),
      textDirection: TextDirection.ltr,
    );
    textPainter.layout();
    textPainter.paint(canvas, Offset(size.width - textPainter.width - 8, 8));
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => true;
}
