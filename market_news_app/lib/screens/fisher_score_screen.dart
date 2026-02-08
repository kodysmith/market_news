import 'package:flutter/material.dart';
import 'package:market_news_app/models/fisher_snapshot.dart';
import 'package:market_news_app/services/fisher_service.dart';
import 'package:market_news_app/widgets/asset_selector_widget.dart';
import 'package:market_news_app/widgets/asset_selection_provider.dart';

/// Fisher score screen: snapshot (total + category scores), delta, evidence panel.
class FisherScoreScreen extends StatefulWidget {
  const FisherScoreScreen({super.key});

  @override
  State<FisherScoreScreen> createState() => _FisherScoreScreenState();
}

class _FisherScoreScreenState extends State<FisherScoreScreen> {
  FisherSnapshot? _snapshot;
  FisherDelta? _delta;
  String? _error;
  bool _loading = false;

  String get _selectedTicker {
    try {
      return AssetSelectionProvider.of(context).selectedAsset;
    } catch (_) {
      return 'AAPL';
    }
  }

  Future<void> _load() async {
    setState(() {
      _loading = true;
      _error = null;
    });
    try {
      final snapshot = await FisherService.getSnapshot(_selectedTicker);
      final delta = await FisherService.getDelta(_selectedTicker);
      if (!mounted) return;
      setState(() {
        _snapshot = snapshot;
        _delta = delta;
        _loading = false;
        if (snapshot == null) _error = 'No Fisher snapshot for $_selectedTicker. Run pipeline first.';
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
        title: const Text('Fisher Score', style: TextStyle(fontFamily: 'JetBrains Mono')),
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
                Expanded(
                  child: AssetSelectorWidget(
                    compact: true,
                    showQuickButtons: false,
                    onAssetChanged: (_) => _load(),
                  ),
                ),
                const SizedBox(width: 8),
                ElevatedButton(
                  onPressed: _loading ? null : _load,
                  child: _loading
                      ? const SizedBox(width: 20, height: 20, child: CircularProgressIndicator(strokeWidth: 2))
                      : const Text('Refresh'),
                ),
              ],
            ),
          ),
          Expanded(child: _buildBody()),
        ],
      ),
    );
  }

  Widget _buildBody() {
    if (_loading && _snapshot == null) {
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
    if (_snapshot == null) return const SizedBox.shrink();
    return SingleChildScrollView(
      padding: const EdgeInsets.all(12),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          _buildSnapshotCard(_snapshot!),
          const SizedBox(height: 16),
          if (_delta != null) _buildDeltaCard(_delta!),
          if (_delta != null) const SizedBox(height: 16),
          _buildPointsCard(_snapshot!),
          const SizedBox(height: 16),
          _buildEvidenceSection(),
        ],
      ),
    );
  }

  Widget _buildSnapshotCard(FisherSnapshot s) {
    final total = s.totalScore ?? 0.0;
    final color = total >= 7 ? Colors.green : total >= 5 ? Colors.orange : Colors.red;
    return Card(
      color: const Color(0xFF161B22),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Latest snapshot', style: Theme.of(context).textTheme.titleMedium?.copyWith(color: Colors.grey.shade400)),
            if (s.snapshotAt != null) Text(s.snapshotAt!, style: TextStyle(fontSize: 12, color: Colors.grey.shade500)),
            const SizedBox(height: 12),
            Row(
              children: [
                Text('Total score', style: Theme.of(context).textTheme.titleSmall?.copyWith(color: Colors.grey.shade400)),
                const Spacer(),
                Text(total.toStringAsFixed(1), style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold, color: color)),
                Text(' / 10', style: TextStyle(fontSize: 16, color: Colors.grey.shade500)),
              ],
            ),
            const SizedBox(height: 16),
            Text('Category scores', style: Theme.of(context).textTheme.titleSmall?.copyWith(color: Colors.grey.shade400)),
            const SizedBox(height: 8),
            ...(s.categoryScores.entries.map((e) {
              final v = e.value ?? 0.0;
              final barColor = v >= 7 ? Colors.green : v >= 5 ? Colors.orange : Colors.red;
              return Padding(
                padding: const EdgeInsets.only(bottom: 8),
                child: Row(
                  children: [
                    SizedBox(width: 120, child: Text(e.key, style: const TextStyle(fontSize: 13))),
                    Expanded(
                      child: LinearProgressIndicator(
                        value: (v / 10).clamp(0.0, 1.0),
                        backgroundColor: Colors.grey.shade800,
                        valueColor: AlwaysStoppedAnimation<Color>(barColor),
                      ),
                    ),
                    const SizedBox(width: 8),
                    Text(v.toStringAsFixed(1), style: TextStyle(fontSize: 13, color: barColor)),
                  ],
                ),
              );
            })),
          ],
        ),
      ),
    );
  }

  Widget _buildDeltaCard(FisherDelta d) {
    if (d.pointDeltas.isEmpty && d.message != null) {
      return Card(
        color: const Color(0xFF161B22),
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text('What changed since last quarter', style: Theme.of(context).textTheme.titleMedium?.copyWith(color: Colors.grey.shade400)),
              const SizedBox(height: 8),
              Text(d.message!, style: TextStyle(color: Colors.grey.shade400)),
            ],
          ),
        ),
      );
    }
    return Card(
      color: const Color(0xFF161B22),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('What changed since last quarter', style: Theme.of(context).textTheme.titleMedium?.copyWith(color: Colors.grey.shade400)),
            if (d.currentSnapshotAt != null) Text('Current: ${d.currentSnapshotAt}', style: TextStyle(fontSize: 11, color: Colors.grey.shade500)),
            if (d.previousSnapshotAt != null) Text('Previous: ${d.previousSnapshotAt}', style: TextStyle(fontSize: 11, color: Colors.grey.shade500)),
            const SizedBox(height: 12),
            ...(d.pointDeltas.entries.map((e) {
              final v = e.value;
              final color = v > 0 ? Colors.green : v < 0 ? Colors.red : Colors.grey;
              return Padding(
                padding: const EdgeInsets.only(bottom: 4),
                child: Row(
                  children: [
                    Text('Point ${e.key}', style: const TextStyle(fontSize: 13)),
                    const Spacer(),
                    Text('${v >= 0 ? '+' : ''}${v.toStringAsFixed(2)}', style: TextStyle(fontSize: 13, color: color)),
                  ],
                ),
              );
            })),
          ],
        ),
      ),
    );
  }

  Widget _buildPointsCard(FisherSnapshot s) {
    final pointLabels = <String, String>{
      '1': 'Sales growth potential',
      '3': 'R&D effectiveness',
      '5': 'Profit margin',
      '6': 'Maintain/improve margins',
      '9': 'Depth of management',
      '13': 'Dilution risk',
      '14': 'Cash/debt management',
      '15': 'Moat (partial)',
    };
    return Card(
      color: const Color(0xFF161B22),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Point scores', style: Theme.of(context).textTheme.titleMedium?.copyWith(color: Colors.grey.shade400)),
            const SizedBox(height: 12),
            ...(s.points.entries.map((e) {
              final score = e.value.score ?? 0.0;
              final color = score >= 7 ? Colors.green : score >= 5 ? Colors.orange : Colors.red;
              final label = pointLabels[e.key] ?? 'Point ${e.key}';
              return Padding(
                padding: const EdgeInsets.only(bottom: 8),
                child: Row(
                  children: [
                    Expanded(
                      flex: 2,
                      child: Text(label, style: const TextStyle(fontSize: 13)),
                    ),
                    Expanded(
                      child: LinearProgressIndicator(
                        value: (score / 10).clamp(0.0, 1.0),
                        backgroundColor: Colors.grey.shade800,
                        valueColor: AlwaysStoppedAnimation<Color>(color),
                      ),
                    ),
                    const SizedBox(width: 8),
                    Text(score.toStringAsFixed(1), style: TextStyle(fontSize: 13, color: color)),
                  ],
                ),
              );
            })),
          ],
        ),
      ),
    );
  }

  Widget _buildEvidenceSection() {
    return Card(
      color: const Color(0xFF161B22),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Evidence by point', style: Theme.of(context).textTheme.titleMedium?.copyWith(color: Colors.grey.shade400)),
            const SizedBox(height: 8),
            Text('Tap a point to load evidence and feature values.', style: TextStyle(fontSize: 12, color: Colors.grey.shade500)),
            const SizedBox(height: 12),
            Wrap(
              spacing: 8,
              runSpacing: 8,
              children: ['1', '3', '5', '6', '9', '13', '14', '15'].map((pointId) {
                return OutlinedButton(
                  onPressed: () => _showEvidence(pointId),
                  child: Text('Point $pointId'),
                );
              }).toList(),
            ),
          ],
        ),
      ),
    );
  }

  Future<void> _showEvidence(String pointId) async {
    final evidence = await FisherService.getEvidence(_selectedTicker, pointId);
    if (!mounted) return;
    if (evidence == null) {
      ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('No evidence for this point.')));
      return;
    }
    showModalBottomSheet(
      context: context,
      backgroundColor: const Color(0xFF161B22),
      isScrollControlled: true,
      builder: (context) => DraggableScrollableSheet(
        initialChildSize: 0.5,
        minChildSize: 0.25,
        maxChildSize: 0.9,
        expand: false,
        builder: (context, scrollController) => SingleChildScrollView(
          controller: scrollController,
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text('Point $pointId — ${evidence.ticker}', style: Theme.of(context).textTheme.titleMedium),
              if (evidence.score != null) Text('Score: ${evidence.score!.toStringAsFixed(1)}', style: TextStyle(color: Colors.green.shade300)),
              if (evidence.confidence != null) Text('Confidence: ${evidence.confidence!.toStringAsFixed(2)}', style: TextStyle(color: Colors.grey.shade400)),
              const SizedBox(height: 12),
              const Text('Evidence', style: TextStyle(fontWeight: FontWeight.bold)),
              ...evidence.evidence.map<Widget>((e) => Padding(
                padding: const EdgeInsets.only(top: 4),
                child: Text(e.toString(), style: TextStyle(fontSize: 13, color: Colors.grey.shade300)),
              )),
              const SizedBox(height: 12),
              const Text('Feature values', style: TextStyle(fontWeight: FontWeight.bold)),
              ...evidence.featureValues.entries.map<Widget>((e) => Padding(
                padding: const EdgeInsets.only(top: 4),
                child: Text('${e.key}: ${e.value}', style: TextStyle(fontSize: 12, color: Colors.grey.shade400)),
              )),
            ],
          ),
        ),
      ),
    );
  }
}
