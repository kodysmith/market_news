import 'package:flutter/material.dart';
import 'package:market_news_app/services/probability_service.dart';
import 'package:market_news_app/widgets/asset_selector_widget.dart';
import 'package:market_news_app/widgets/asset_selection_provider.dart';

/// Probability Range screen — "From now to close" cone, HPH/HPL, close distribution.
/// Panel A: price line + probability cone (P10/P25/P50/P75/P90).
/// Panel B: HPH/HPL histograms + P50/P80 high and low.
/// Panel C: close distribution with mode and 50%/80% credible intervals.
class ProbabilityRangeScreen extends StatefulWidget {
  const ProbabilityRangeScreen({super.key});

  @override
  State<ProbabilityRangeScreen> createState() => _ProbabilityRangeScreenState();
}

class _ProbabilityRangeScreenState extends State<ProbabilityRangeScreen> {
  ProbabilityRangeResponse? _data;
  String? _error;
  bool _loading = false;

  String get _selectedTicker {
    try {
      return AssetSelectionProvider.of(context).selectedAsset;
    } catch (_) {
      return 'SPY';
    }
  }

  Future<void> _load() async {
    setState(() {
      _loading = true;
      _error = null;
    });
    try {
      final res = await ProbabilityService.getProbabilityRange(_selectedTicker);
      if (!mounted) return;
      setState(() {
        _data = res;
        _loading = false;
        if (res == null) _error = 'Failed to load probability range.';
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _loading = false;
        _error = e.toString();
      });
    }
  }

  @override
  void initState() {
    super.initState();
    _load();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF0D1117),
      appBar: AppBar(
        title: const Text('Range & Probabilities', style: TextStyle(fontFamily: 'JetBrains Mono')),
        backgroundColor: const Color(0xFF161B22),
        actions: [
          IconButton(icon: const Icon(Icons.refresh), onPressed: _load),
        ],
      ),
      body: Column(
        children: [
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
            child: Row(
              children: [
                Expanded(child: AssetSelectorWidget(compact: true, showQuickButtons: false, onAssetChanged: (_) => _load())),
                const SizedBox(width: 8),
                ElevatedButton(
                  onPressed: _loading ? null : _load,
                  child: _loading ? const SizedBox(width: 20, height: 20, child: CircularProgressIndicator(strokeWidth: 2)) : const Text('Refresh'),
                ),
              ],
            ),
          ),
          Expanded(
            child: _buildBody(),
          ),
        ],
      ),
    );
  }

  Widget _buildBody() {
    if (_loading && _data == null) {
      return const Center(child: CircularProgressIndicator(color: Color(0xFF58A6FF)));
    }
    if (_error != null) {
      return Center(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Text(_error!, style: const TextStyle(color: Colors.red), textAlign: TextAlign.center),
              const SizedBox(height: 16),
              ElevatedButton(onPressed: _load, child: const Text('Retry')),
            ],
          ),
        ),
      );
    }
    if (_data == null) return const SizedBox.shrink();
    return SingleChildScrollView(
      padding: const EdgeInsets.all(12),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          _buildPanelA(_data!),
          const SizedBox(height: 24),
          _buildPanelB(_data!),
          const SizedBox(height: 24),
          _buildPanelC(_data!),
        ],
      ),
    );
  }

  Widget _buildPanelA(ProbabilityRangeResponse d) {
    if (d.cone.isEmpty) return const SizedBox.shrink();
    final maxMin = d.cone.map((c) => c.minutesFromNow).reduce((a, b) => a > b ? a : b);
    final maxP90 = d.cone.map((c) => c.p90).reduce((a, b) => a > b ? a : b);
    final minP10 = d.cone.map((c) => c.p10).reduce((a, b) => a < b ? a : b);
    final priceRange = (maxP90 - minP10).clamp(0.01, double.infinity);
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFF161B22),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Colors.white12),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text('Panel A — Probability cone (from now to close)', style: TextStyle(fontSize: 14, fontWeight: FontWeight.bold, color: Color(0xFF8B949E), fontFamily: 'JetBrains Mono')),
          Text('Spot \$${d.spot.toStringAsFixed(2)} • ${d.minutesToClose} min to close • σ_eff ${d.sigmaEff.toStringAsFixed(4)}', style: const TextStyle(fontSize: 12, color: Colors.white54, fontFamily: 'JetBrains Mono')),
          const SizedBox(height: 16),
          SizedBox(
            height: 220,
            child: CustomPaint(
              painter: _ConePainter(
                cone: d.cone,
                spot: d.spot,
                maxMin: maxMin,
                minPrice: minP10,
                priceRange: priceRange,
              ),
              size: Size(MediaQuery.of(context).size.width - 56, 220),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildPanelB(ProbabilityRangeResponse d) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFF161B22),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Colors.white12),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text('Panel B — HPH / HPL (highest-probability high/low)', style: TextStyle(fontSize: 14, fontWeight: FontWeight.bold, color: Color(0xFF8B949E), fontFamily: 'JetBrains Mono')),
          const SizedBox(height: 8),
          _buildHistogramRow('High', d.highDistribution, d.hph, d.p50High, d.p80High, true),
          const SizedBox(height: 16),
          _buildHistogramRow('Low', d.lowDistribution, d.hpl, d.p50Low, d.p80Low, false),
        ],
      ),
    );
  }

  Widget _buildHistogramRow(String label, List<HistogramBucket> buckets, double mode, double p50, double p80, bool isHigh) {
    if (buckets.isEmpty) return const SizedBox.shrink();
    final maxCount = buckets.map((b) => b.count).reduce((a, b) => a > b ? a : b);
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text('$label: HPH/HPL \$${mode.toStringAsFixed(1)} • P50 \$${p50.toStringAsFixed(1)} • P80 \$${p80.toStringAsFixed(1)}', style: const TextStyle(fontSize: 12, color: Colors.white70, fontFamily: 'JetBrains Mono')),
        const SizedBox(height: 6),
        SizedBox(
          height: 80,
          child: Row(
            children: buckets.map((b) {
              final w = 4.0;
              final h = maxCount > 0 ? (b.count / maxCount) * 70 : 0.0;
              return Expanded(
                child: Tooltip(
                  message: '\$${b.bucket.toStringAsFixed(1)}: ${b.count}',
                  child: Align(
                    alignment: isHigh ? Alignment.bottomCenter : Alignment.topCenter,
                    child: Container(
                      width: w.clamp(2.0, 12.0),
                      height: h.clamp(2.0, 70.0),
                      margin: const EdgeInsets.symmetric(horizontal: 1),
                      decoration: BoxDecoration(
                        color: (b.bucket - mode).abs() < 1 ? const Color(0xFFFFA500) : const Color(0xFF58A6FF).withOpacity(0.6),
                        borderRadius: BorderRadius.circular(2),
                      ),
                    ),
                  ),
                ),
              );
            }).toList(),
          ),
        ),
      ],
    );
  }

  Widget _buildPanelC(ProbabilityRangeResponse d) {
    if (d.closeDistribution.isEmpty) return const SizedBox.shrink();
    final maxCount = d.closeDistribution.map((b) => b.count).reduce((a, b) => a > b ? a : b);
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFF161B22),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Colors.white12),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text('Panel C — Highest-probability close range', style: TextStyle(fontSize: 14, fontWeight: FontWeight.bold, color: Color(0xFF8B949E), fontFamily: 'JetBrains Mono')),
          Text('Mode \$${d.closeMode.toStringAsFixed(1)} • 50% band \$${d.closeP25.toStringAsFixed(1)}–\$${d.closeP75.toStringAsFixed(1)} • 80% band \$${d.closeP10.toStringAsFixed(1)}–\$${d.closeP90.toStringAsFixed(1)}', style: const TextStyle(fontSize: 12, color: Colors.white70, fontFamily: 'JetBrains Mono')),
          if (d.putWall != null || d.callWall != null)
            Text('Put wall \$${d.putWall?.toStringAsFixed(0) ?? '—'} • Call wall \$${d.callWall?.toStringAsFixed(0) ?? '—'}', style: const TextStyle(fontSize: 11, color: Colors.white54, fontFamily: 'JetBrains Mono')),
          const SizedBox(height: 12),
          SizedBox(
            height: 100,
            child: Row(
              children: d.closeDistribution.map((b) {
                final h = maxCount > 0 ? (b.count / maxCount) * 90 : 0.0;
                final isMode = (b.bucket - d.closeMode).abs() < 1;
                return Expanded(
                  child: Tooltip(
                    message: '\$${b.bucket.toStringAsFixed(1)}: ${b.count}',
                    child: Align(
                      alignment: Alignment.bottomCenter,
                      child: Container(
                        width: 6,
                        height: h.clamp(2.0, 90.0),
                        margin: const EdgeInsets.symmetric(horizontal: 1),
                        decoration: BoxDecoration(
                          color: isMode ? const Color(0xFF3FB950) : const Color(0xFF58A6FF).withOpacity(0.5),
                          borderRadius: BorderRadius.circular(2),
                        ),
                      ),
                    ),
                  ),
                );
              }).toList(),
            ),
          ),
        ],
      ),
    );
  }
}

class _ConePainter extends CustomPainter {
  final List<ConePoint> cone;
  final double spot;
  final double maxMin;
  final double minPrice;
  final double priceRange;

  _ConePainter({required this.cone, required this.spot, required this.maxMin, required this.minPrice, required this.priceRange});

  @override
  void paint(Canvas canvas, Size size) {
    if (cone.isEmpty) return;
    const padding = 40.0;
    final chartW = size.width - padding * 2;
    final chartH = size.height - padding * 2;
    final tRange = maxMin.clamp(0.01, double.infinity);
    double tx(double t) => padding + (t / tRange) * chartW;
    double py(double p) => padding + chartH - ((p - minPrice) / priceRange) * chartH;
    final pathP90 = Path()..moveTo(tx(cone[0].minutesFromNow), py(cone[0].p90));
    final pathP10 = Path()..moveTo(tx(cone[0].minutesFromNow), py(cone[0].p10));
    for (int i = 1; i < cone.length; i++) {
      final c = cone[i];
      pathP90.lineTo(tx(c.minutesFromNow), py(c.p90));
      pathP10.lineTo(tx(c.minutesFromNow), py(c.p10));
    }
    for (int i = cone.length - 1; i >= 0; i--) {
      pathP10.lineTo(tx(cone[i].minutesFromNow), py(cone[i].p10));
    }
    pathP10.close();
    canvas.drawPath(pathP10, Paint()..color = const Color(0xFF58A6FF).withOpacity(0.2)..style = PaintingStyle.fill);
    final pathP75 = Path()..moveTo(tx(cone[0].minutesFromNow), py(cone[0].p75));
    final pathP25 = Path()..moveTo(tx(cone[0].minutesFromNow), py(cone[0].p25));
    for (int i = 1; i < cone.length; i++) {
      final c = cone[i];
      pathP75.lineTo(tx(c.minutesFromNow), py(c.p75));
      pathP25.lineTo(tx(c.minutesFromNow), py(c.p25));
    }
    for (int i = cone.length - 1; i >= 0; i--) {
      pathP25.lineTo(tx(cone[i].minutesFromNow), py(cone[i].p25));
    }
    pathP25.close();
    canvas.drawPath(pathP25, Paint()..color = const Color(0xFF58A6FF).withOpacity(0.35)..style = PaintingStyle.fill);
    for (int i = 0; i < cone.length - 1; i++) {
      final c0 = cone[i];
      final c1 = cone[i + 1];
      canvas.drawLine(Offset(tx(c0.minutesFromNow), py(c0.p50)), Offset(tx(c1.minutesFromNow), py(c1.p50)), Paint()..color = const Color(0xFF3FB950)..strokeWidth = 2);
    }
    canvas.drawLine(Offset(padding, py(spot)), Offset(tx(cone[0].minutesFromNow), py(spot)), Paint()..color = Colors.white..strokeWidth = 2);
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => true;
}
