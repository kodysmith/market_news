/// Premarket scanner idea — one per symbol per run (gap + relvol + liquidity).
/// Matches API contract: GET /trade-ideas/scanner/today
class PremarketTradeIdea {
  final String symbol;
  final String asOf;
  final double pmLast;
  final double? pmChangePct;
  final double pmGapPct;
  final double? pmVolume;
  final double? pmRelVolume;
  final double? pmHigh;
  final double? pmLow;
  final double? pmRangePct;
  final double? prevDayDollarVol;
  final double? prevDayAtrPct;
  final double? spreadEstimate;
  final double? sharesOutstanding;
  final double? marketCap;
  final List<String> newsHeadlines;
  final bool? earningsFlag;
  final String? earningsTime;
  final double? callPutVolumeRatio;
  final double? oiChange;
  final double? ivChange;
  final int? unusualSweepsCount;
  final double? optionsHeatScore;
  final double gapScore;
  final double relvolScore;
  final double liquidityScore;
  final double optionsScore;
  final double haltRiskScore;
  final double totalScore;
  final List<String> reasons;
  final String? playbook;
  final Map<String, dynamic>? suggestedLevels;

  PremarketTradeIdea({
    required this.symbol,
    required this.asOf,
    required this.pmLast,
    this.pmChangePct,
    this.pmGapPct = 0,
    this.pmVolume,
    this.pmRelVolume,
    this.pmHigh,
    this.pmLow,
    this.pmRangePct,
    this.prevDayDollarVol,
    this.prevDayAtrPct,
    this.spreadEstimate,
    this.sharesOutstanding,
    this.marketCap,
    this.newsHeadlines = const [],
    this.earningsFlag,
    this.earningsTime,
    this.callPutVolumeRatio,
    this.oiChange,
    this.ivChange,
    this.unusualSweepsCount,
    this.optionsHeatScore,
    this.gapScore = 0,
    this.relvolScore = 0,
    this.liquidityScore = 0,
    this.optionsScore = 0,
    this.haltRiskScore = 0,
    this.totalScore = 0,
    this.reasons = const [],
    this.playbook,
    this.suggestedLevels,
  });

  factory PremarketTradeIdea.fromJson(Map<String, dynamic> json) {
    return PremarketTradeIdea(
      symbol: json['symbol'] as String? ?? '',
      asOf: json['asOf'] as String? ?? '',
      pmLast: _toDouble(json['pm_last']) ?? 0.0,
      pmChangePct: _toDoubleNull(json['pm_change_pct']),
      pmGapPct: _toDouble(json['pm_gap_pct']) ?? 0.0,
      pmVolume: _toDoubleNull(json['pm_volume']),
      pmRelVolume: _toDoubleNull(json['pm_rel_volume']),
      pmHigh: _toDoubleNull(json['pm_high']),
      pmLow: _toDoubleNull(json['pm_low']),
      pmRangePct: _toDoubleNull(json['pm_range_pct']),
      prevDayDollarVol: _toDoubleNull(json['prev_day_dollar_vol']),
      prevDayAtrPct: _toDoubleNull(json['prev_day_atr_pct']),
      spreadEstimate: _toDoubleNull(json['spread_estimate']),
      sharesOutstanding: _toDoubleNull(json['shares_outstanding']),
      marketCap: _toDoubleNull(json['market_cap']),
      newsHeadlines: (json['news_headlines'] as List<dynamic>?)?.map((e) => e.toString()).toList() ?? [],
      earningsFlag: json['earnings_flag'] as bool?,
      earningsTime: json['earnings_time'] as String?,
      callPutVolumeRatio: _toDoubleNull(json['call_put_volume_ratio']),
      oiChange: _toDoubleNull(json['oi_change']),
      ivChange: _toDoubleNull(json['iv_change']),
      unusualSweepsCount: json['unusual_sweeps_count'] as int?,
      optionsHeatScore: _toDoubleNull(json['options_heat_score']),
      gapScore: _toDouble(json['gap_score']) ?? 0.0,
      relvolScore: _toDouble(json['relvol_score']) ?? 0.0,
      liquidityScore: _toDouble(json['liquidity_score']) ?? 0.0,
      optionsScore: _toDouble(json['options_score']) ?? 0.0,
      haltRiskScore: _toDouble(json['halt_risk_score']) ?? 0.0,
      totalScore: _toDouble(json['total_score']) ?? 0.0,
      reasons: (json['reasons'] as List<dynamic>?)?.map((e) => e.toString()).toList() ?? [],
      playbook: json['playbook'] as String?,
      suggestedLevels: json['suggested_levels'] != null ? Map<String, dynamic>.from(json['suggested_levels'] as Map) : null,
    );
  }

  static double? _toDouble(dynamic v) {
    if (v == null) return null;
    if (v is num) return v.toDouble();
    if (v is String) return double.tryParse(v);
    return null;
  }

  static double? _toDoubleNull(dynamic v) {
    if (v == null) return null;
    if (v is num) return v.toDouble();
    if (v is String) return double.tryParse(v);
    return null;
  }
}
