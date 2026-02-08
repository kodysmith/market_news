/// Models for Fisher score API: snapshot, delta, evidence, universe.

/// Per-point score (score, confidence, evidence, feature_values)
class FisherPointScore {
  final double? score;
  final double? confidence;
  final List<dynamic> evidence;
  final Map<String, dynamic> featureValues;

  FisherPointScore({
    this.score,
    this.confidence,
    this.evidence = const [],
    this.featureValues = const {},
  });

  factory FisherPointScore.fromJson(Map<String, dynamic> json) {
    final ev = json['evidence'];
    return FisherPointScore(
      score: _toDouble(json['score']),
      confidence: _toDouble(json['confidence']),
      evidence: ev is List ? ev : [],
      featureValues: json['feature_values'] is Map<String, dynamic>
          ? Map<String, dynamic>.from(json['feature_values'] as Map)
          : {},
    );
  }

  static double? _toDouble(dynamic v) {
    if (v == null) return null;
    if (v is num) return v.toDouble();
    if (v is String) return double.tryParse(v);
    return null;
  }
}

/// Latest Fisher score snapshot for a ticker
class FisherSnapshot {
  final String? companyId;
  final String? ticker;
  final String? snapshotAt;
  final double? totalScore;
  final int? version;
  final Map<String, FisherPointScore> points;
  final Map<String, double?> categoryScores;

  FisherSnapshot({
    this.companyId,
    this.ticker,
    this.snapshotAt,
    this.totalScore,
    this.version,
    this.points = const {},
    this.categoryScores = const {},
  });

  factory FisherSnapshot.fromJson(Map<String, dynamic> json) {
    final pts = json['points'] as Map<String, dynamic>? ?? {};
    final pointsMap = <String, FisherPointScore>{};
    for (final e in pts.entries) {
      if (e.value is Map) {
        pointsMap[e.key] = FisherPointScore.fromJson(Map<String, dynamic>.from(e.value as Map));
      }
    }
    final cat = json['category_scores'] as Map<String, dynamic>? ?? {};
    final catScores = <String, double?>{};
    for (final e in cat.entries) {
      catScores[e.key] = FisherPointScore._toDouble(e.value);
    }
    return FisherSnapshot(
      companyId: json['company_id']?.toString(),
      ticker: json['ticker']?.toString(),
      snapshotAt: json['snapshot_at']?.toString(),
      totalScore: FisherPointScore._toDouble(json['total_score']),
      version: json['version'] is int ? json['version'] as int : null,
      points: pointsMap,
      categoryScores: catScores,
    );
  }
}

/// Delta between current and previous snapshot
class FisherDelta {
  final String ticker;
  final String? currentSnapshotAt;
  final String? previousSnapshotAt;
  final Map<String, double> pointDeltas;
  final String? message;

  FisherDelta({
    required this.ticker,
    this.currentSnapshotAt,
    this.previousSnapshotAt,
    this.pointDeltas = const {},
    this.message,
  });

  factory FisherDelta.fromJson(Map<String, dynamic> json) {
    final deltas = json['point_deltas'] as Map<String, dynamic>? ?? {};
    final map = <String, double>{};
    for (final e in deltas.entries) {
      final v = e.value;
      if (v is num) map[e.key] = v.toDouble();
    }
    return FisherDelta(
      ticker: json['ticker']?.toString() ?? '',
      currentSnapshotAt: json['current_snapshot_at']?.toString(),
      previousSnapshotAt: json['previous_snapshot_at']?.toString(),
      pointDeltas: map,
      message: json['message']?.toString(),
    );
  }
}

/// Evidence for a single Fisher point
class FisherEvidence {
  final String ticker;
  final String pointId;
  final double? score;
  final double? confidence;
  final List<dynamic> evidence;
  final Map<String, dynamic> featureValues;

  FisherEvidence({
    required this.ticker,
    required this.pointId,
    this.score,
    this.confidence,
    this.evidence = const [],
    this.featureValues = const {},
  });

  factory FisherEvidence.fromJson(Map<String, dynamic> json) {
    final ev = json['evidence'];
    return FisherEvidence(
      ticker: json['ticker']?.toString() ?? '',
      pointId: json['point_id']?.toString() ?? '',
      score: FisherPointScore._toDouble(json['score']),
      confidence: FisherPointScore._toDouble(json['confidence']),
      evidence: ev is List ? ev : [],
      featureValues: json['feature_values'] is Map<String, dynamic>
          ? Map<String, dynamic>.from(json['feature_values'] as Map)
          : {},
    );
  }
}

/// High growth + profitable company (from GET /fisher/growth-profitable)
class FisherGrowthProfitableItem {
  final String ticker;
  final String name;
  final String sector;
  final double? growth;
  final double? financials;
  final double? totalScore;
  final String? snapshotAt;

  FisherGrowthProfitableItem({
    required this.ticker,
    required this.name,
    this.sector = '',
    this.growth,
    this.financials,
    this.totalScore,
    this.snapshotAt,
  });

  factory FisherGrowthProfitableItem.fromJson(Map<String, dynamic> json) {
    return FisherGrowthProfitableItem(
      ticker: json['ticker'] ?? '',
      name: json['name'] ?? json['ticker'] ?? '',
      sector: json['sector'] ?? '',
      growth: _toDouble(json['growth']),
      financials: _toDouble(json['financials']),
      totalScore: _toDouble(json['total_score']),
      snapshotAt: json['snapshot_at']?.toString(),
    );
  }

  static double? _toDouble(dynamic v) {
    if (v == null) return null;
    if (v is num) return v.toDouble();
    if (v is String) return double.tryParse(v);
    return null;
  }
}

/// Fisher universe (list of tickers)
class FisherUniverse {
  final List<String> tickers;

  FisherUniverse({this.tickers = const []});

  factory FisherUniverse.fromJson(Map<String, dynamic> json) {
    final list = json['tickers'] as List<dynamic>? ?? [];
    return FisherUniverse(
      tickers: list.map((e) => e.toString()).toList(),
    );
  }
}
