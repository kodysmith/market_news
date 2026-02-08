import 'package:flutter/material.dart';
import 'package:market_news_app/constants/fisher_points.dart';
import 'package:market_news_app/models/fisher_snapshot.dart';
import 'package:market_news_app/models/valuation_data.dart';
import 'package:market_news_app/services/fisher_service.dart';
import 'package:market_news_app/services/valuation_service.dart';
import 'package:market_news_app/widgets/asset_selector_widget.dart';
import 'package:market_news_app/widgets/asset_selection_provider.dart';

/// Combined Fisher & Valuation: high growth + profitable list, Fisher score, and DCF buy price.
class FisherValuationScreen extends StatefulWidget {
  const FisherValuationScreen({super.key});

  @override
  State<FisherValuationScreen> createState() => _FisherValuationScreenState();
}

class _FisherValuationScreenState extends State<FisherValuationScreen> {
  List<FisherGrowthProfitableItem> _growthList = [];
  bool _growthListLoading = false;
  String? _growthListError; // connection or DB error from growth-profitable API
  FisherSnapshot? _snapshot;
  ValuationResult? _valuation;
  String? _error;
  bool _detailLoading = false;

  String get _selectedTicker {
    try {
      return AssetSelectionProvider.of(context).selectedAsset;
    } catch (_) {
      return 'AAPL';
    }
  }

  Future<void> _loadGrowthList() async {
    setState(() {
      _growthListLoading = true;
      _growthListError = null;
    });
    try {
      final result = await FisherService.getGrowthProfitableWithError(minGrowth: 6.0, minFinancials: 6.0, limit: 100);
      if (!mounted) return;
      setState(() {
        _growthList = result.companies;
        _growthListError = result.error;
        _growthListLoading = false;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _growthList = [];
        _growthListError = 'Failed to load: $e';
        _growthListLoading = false;
      });
    }
  }

  Future<void> _loadDetail(String ticker) async {
    if (ticker.isEmpty) return;
    setState(() {
      _detailLoading = true;
      _error = null;
      _snapshot = null;
      _valuation = null;
    });
    try {
      final snapshot = await FisherService.getSnapshot(ticker);
      final valuation = await ValuationService.calculateIntrinsicValue(ticker);
      if (!mounted) return;
      setState(() {
        _snapshot = snapshot;
        _valuation = valuation;
        _detailLoading = false;
        if (snapshot == null && valuation == null) _error = 'No data for $ticker.';
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _detailLoading = false;
        _error = e.toString();
      });
    }
  }

  @override
  void initState() {
    super.initState();
    _loadGrowthList();
    WidgetsBinding.instance.addPostFrameCallback((_) => _loadDetail(_selectedTicker));
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF0D1117),
      appBar: AppBar(
        title: const Text('Fisher & Valuation', style: TextStyle(fontFamily: 'JetBrains Mono')),
        backgroundColor: const Color(0xFF161B22),
        actions: [
          IconButton(icon: const Icon(Icons.refresh), onPressed: () {
            _loadGrowthList();
            _loadDetail(_selectedTicker);
          }),
        ],
      ),
      body: Column(
        children: [
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
            child: Row(
              children: [
                Expanded(
                  child: AssetSelectorWidget(
                    compact: true,
                    showQuickButtons: false,
                    onAssetChanged: (t) => _loadDetail(t),
                  ),
                ),
                const SizedBox(width: 8),
                ElevatedButton(
                  onPressed: _detailLoading ? null : () => _loadDetail(_selectedTicker),
                  child: _detailLoading
                      ? const SizedBox(width: 20, height: 20, child: CircularProgressIndicator(strokeWidth: 2))
                      : const Text('Refresh'),
                ),
              ],
            ),
          ),
          Expanded(
            child: RefreshIndicator(
              onRefresh: () async {
                await _loadGrowthList();
                await _loadDetail(_selectedTicker);
              },
              child: SingleChildScrollView(
                physics: const AlwaysScrollableScrollPhysics(),
                padding: const EdgeInsets.all(12),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    _buildGrowthListSection(),
                    const SizedBox(height: 20),
                    if (_error != null) _buildError(),
                    if (_snapshot != null) _buildFisherCard(_snapshot!),
                    if (_snapshot != null) const SizedBox(height: 16),
                    if (_snapshot != null) _buildFisher15PointsCard(_snapshot!),
                    if (_snapshot != null) const SizedBox(height: 16),
                    if (_valuation != null) _buildValuationCard(_valuation!),
                  ],
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildGrowthListSection() {
    return Card(
      color: const Color(0xFF161B22),
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Text('High growth + profitable', style: Theme.of(context).textTheme.titleMedium?.copyWith(color: Colors.grey.shade400)),
                const SizedBox(width: 8),
                if (_growthListLoading) const SizedBox(width: 16, height: 16, child: CircularProgressIndicator(strokeWidth: 2)),
              ],
            ),
            const SizedBox(height: 8),
            Text('Tap a row to load Fisher score and DCF valuation.', style: TextStyle(fontSize: 12, color: Colors.grey.shade500)),
            const SizedBox(height: 8),
            if (_growthListError != null)
              Padding(
                padding: const EdgeInsets.all(12),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Icon(Icons.error_outline, size: 20, color: Colors.orange.shade300),
                        const SizedBox(width: 8),
                        Expanded(
                          child: Text(
                            _growthListError!,
                            style: TextStyle(fontSize: 13, color: Colors.orange.shade300),
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 8),
                    Text(
                      'On the server: set DATABASE_URL (e.g. postgresql://user:pass@host:5433/db), then run the Fisher pipeline and scan.',
                      style: TextStyle(fontSize: 11, color: Colors.grey.shade500),
                    ),
                  ],
                ),
              )
            else if (_growthList.isEmpty && !_growthListLoading)
              Padding(
                padding: const EdgeInsets.all(8),
                child: Text('No companies yet. Run Fisher pipeline and scripts/scan_growth_profitable.py.', style: TextStyle(fontSize: 12, color: Colors.grey.shade500)),
              )
            else
              ConstrainedBox(
                constraints: const BoxConstraints(maxHeight: 220),
                child: ListView.builder(
                  shrinkWrap: true,
                  itemCount: _growthList.length,
                  itemBuilder: (context, i) {
                    final item = _growthList[i];
                    final selected = item.ticker == _selectedTicker;
                    return Material(
                      color: selected ? Colors.blue.shade900.withOpacity(0.3) : Colors.transparent,
                      child: ListTile(
                        dense: true,
                        title: Text('${item.ticker}  ${item.name}', style: TextStyle(fontSize: 13, fontWeight: selected ? FontWeight.bold : null)),
                        subtitle: Text('Growth ${item.growth?.toStringAsFixed(1) ?? "—"}  Fin ${item.financials?.toStringAsFixed(1) ?? "—"}  Score ${item.totalScore?.toStringAsFixed(1) ?? "—"}', style: TextStyle(fontSize: 11, color: Colors.grey.shade400)),
                        onTap: () {
                          AssetSelectionProvider.of(context).setAsset(item.ticker);
                          _loadDetail(item.ticker);
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

  Widget _buildError() {
    return Card(
      color: const Color(0xFF161B22),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Row(
          children: [
            Icon(Icons.warning_amber, color: Colors.orange.shade300),
            const SizedBox(width: 12),
            Expanded(child: Text(_error!, style: TextStyle(color: Colors.orange.shade300, fontSize: 13))),
          ],
        ),
      ),
    );
  }

  Widget _buildFisherCard(FisherSnapshot s) {
    final total = s.totalScore ?? 0.0;
    final color = total >= 7 ? Colors.green : total >= 5 ? Colors.orange : Colors.red;
    return Card(
      color: const Color(0xFF161B22),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Fisher score', style: Theme.of(context).textTheme.titleMedium?.copyWith(color: Colors.grey.shade400)),
            if (s.snapshotAt != null) Text(s.snapshotAt!, style: TextStyle(fontSize: 11, color: Colors.grey.shade500)),
            const SizedBox(height: 12),
            Row(
              children: [
                Text('Total', style: Theme.of(context).textTheme.titleSmall?.copyWith(color: Colors.grey.shade400)),
                const Spacer(),
                Text('${total.toStringAsFixed(1)} / 10', style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold, color: color)),
              ],
            ),
            const SizedBox(height: 12),
            ...(s.categoryScores.entries.map((e) {
              final v = e.value ?? 0.0;
              final barColor = v >= 7 ? Colors.green : v >= 5 ? Colors.orange : Colors.red;
              return Padding(
                padding: const EdgeInsets.only(bottom: 6),
                child: Row(
                  children: [
                    SizedBox(width: 100, child: Text(e.key, style: const TextStyle(fontSize: 12))),
                    Expanded(
                      child: LinearProgressIndicator(
                        value: (v / 10).clamp(0.0, 1.0),
                        backgroundColor: Colors.grey.shade800,
                        valueColor: AlwaysStoppedAnimation<Color>(barColor),
                      ),
                    ),
                    const SizedBox(width: 8),
                    Text(v.toStringAsFixed(1), style: TextStyle(fontSize: 12, color: barColor)),
                  ],
                ),
              );
            })),
          ],
        ),
      ),
    );
  }

  Widget _buildFisher15PointsCard(FisherSnapshot s) {
    return Card(
      color: const Color(0xFF161B22),
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Fisher 15 points', style: Theme.of(context).textTheme.titleMedium?.copyWith(color: Colors.grey.shade400)),
            const SizedBox(height: 8),
            ...(fisherPointsInOrder.map((meta) {
              final pointData = s.points[meta.id.toString()];
              final score = pointData?.score;
              final hasScore = score != null;
              final color = !hasScore ? Colors.grey : (score >= 7 ? Colors.green : score >= 5 ? Colors.orange : Colors.red);
              return Padding(
                padding: const EdgeInsets.only(bottom: 4),
                child: Row(
                  children: [
                    Expanded(
                      flex: 2,
                      child: Row(
                        children: [
                          Expanded(child: Text(meta.label, style: const TextStyle(fontSize: 12))),
                          Tooltip(
                            message: meta.tooltip,
                            child: Icon(Icons.info_outline, size: 14, color: Colors.grey.shade400),
                          ),
                        ],
                      ),
                    ),
                    SizedBox(
                      width: 28,
                      child: Text(
                        hasScore ? score.toStringAsFixed(1) : 'N/A',
                        style: TextStyle(fontSize: 12, color: color),
                      ),
                    ),
                  ],
                ),
              );
            })),
          ],
        ),
      ),
    );
  }

  Widget _buildValuationCard(ValuationResult result) {
    final composite = result.composite;
    final dcf = result.valuations['dcf'];
    final dcfValue = dcf?.value ?? composite.value;
    final divergencePct = dcfValue != null && dcfValue > 0
        ? ((result.currentPrice - dcfValue) / dcfValue * 100)
        : composite.divergencePct;

    // Use verdict that matches the DCF section when we show DCF value/divergence
    final displayVerdict = _verdictFromDivergence(divergencePct, dcfValue) ?? composite.verdict;

    return Card(
      color: const Color(0xFF161B22),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Valuation — buy price', style: Theme.of(context).textTheme.titleMedium?.copyWith(color: Colors.grey.shade400)),
            const SizedBox(height: 12),
            Row(
              children: [
                Expanded(
                  child: _metricChip('Current price', '\$${result.currentPrice.toStringAsFixed(2)}', Colors.blue),
                ),
                const SizedBox(width: 8),
                Expanded(
                  child: _metricChip('DCF / intrinsic', dcfValue != null ? '\$${dcfValue.toStringAsFixed(2)}' : 'N/A', Colors.purple),
                ),
              ],
            ),
            const SizedBox(height: 8),
            Row(
              children: [
                Expanded(
                  child: _metricChip('Buy at (DCF)', dcfValue != null ? '\$${dcfValue.toStringAsFixed(2)}' : 'N/A', Colors.green),
                ),
                const SizedBox(width: 8),
                Expanded(
                  child: _metricChip(
                    'Divergence',
                    divergencePct != null ? '${divergencePct >= 0 ? '+' : ''}${divergencePct.toStringAsFixed(1)}%' : 'N/A',
                    _divergenceColor(divergencePct),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 8),
            Text(displayVerdict, style: TextStyle(fontSize: 13, color: _verdictColor(displayVerdict))),
          ],
        ),
      ),
    );
  }

  /// Verdict from divergence % (matches backend: intrinsic_value_calculator.py thresholds).
  /// When card shows DCF value and DCF divergence, verdict should match that—not composite.
  String? _verdictFromDivergence(double? divergencePct, double? intrinsicValue) {
    if (divergencePct == null || intrinsicValue == null || intrinsicValue <= 0) return null;
    if (divergencePct < -30) return 'Significantly Undervalued';
    if (divergencePct < -20) return 'Undervalued';
    if (divergencePct < -10) return 'Slightly Undervalued';
    if (divergencePct <= 10) return 'Fair Value';
    if (divergencePct <= 20) return 'Slightly Overvalued';
    if (divergencePct <= 30) return 'Overvalued';
    return 'Significantly Overvalued';
  }

  Widget _metricChip(String label, String value, Color color) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: color.withOpacity(0.15),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(label, style: TextStyle(fontSize: 11, color: color.withOpacity(0.9))),
          const SizedBox(height: 2),
          Text(value, style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold, color: color)),
        ],
      ),
    );
  }

  Color _divergenceColor(double? d) {
    if (d == null) return Colors.grey;
    if (d < -10) return Colors.green;
    if (d <= 10) return Colors.orange;
    return Colors.red;
  }

  Color _verdictColor(String verdict) {
    if (verdict.toLowerCase().contains('undervalued')) return Colors.green;
    if (verdict.toLowerCase().contains('overvalued')) return Colors.red;
    return Colors.orange;
  }
}
