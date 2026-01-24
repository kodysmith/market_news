/// GEX (Gamma Exposure) Data Models
/// Models for GEX calculation API responses

class GexCalculation {
  final String ticker;
  final double spotPrice;
  final GexMetrics metrics;
  final GexBreakdown breakdown;
  final List<GexByStrike> gexByStrike;
  final List<CumulativeGexPoint> cumulativeGex;
  final GammaSlope? gammaSlope;
  final ChartAnnotations chartAnnotations;
  final double timestamp;

  GexCalculation({
    required this.ticker,
    required this.spotPrice,
    required this.metrics,
    required this.breakdown,
    required this.gexByStrike,
    required this.cumulativeGex,
    this.gammaSlope,
    required this.chartAnnotations,
    required this.timestamp,
  });

  factory GexCalculation.fromJson(Map<String, dynamic> json) {
    return GexCalculation(
      ticker: json['ticker'] ?? '',
      spotPrice: (json['spot_price'] as num?)?.toDouble() ?? 0.0,
      metrics: GexMetrics.fromJson(json['metrics'] ?? {}),
      breakdown: GexBreakdown.fromJson(json['breakdown'] ?? {}),
      gexByStrike: (json['gex_by_strike'] as List?)
              ?.map((item) => GexByStrike.fromJson(item))
              .toList() ??
          [],
      cumulativeGex: (json['cumulative_gex'] as List?)
              ?.map((item) => CumulativeGexPoint.fromJson(item))
              .toList() ??
          [],
      gammaSlope: json['gamma_slope'] != null
          ? GammaSlope.fromJson(json['gamma_slope'])
          : null,
      chartAnnotations: ChartAnnotations.fromJson(json['chart_annotations'] ?? {}),
      timestamp: (json['timestamp'] as num?)?.toDouble() ?? 0.0,
    );
  }
}

class GexMetrics {
  final double totalGex;
  final double putWall;
  final double callWall;
  final double? flipLine;
  final double spotPrice;
  final String regime;

  GexMetrics({
    required this.totalGex,
    required this.putWall,
    required this.callWall,
    this.flipLine,
    required this.spotPrice,
    required this.regime,
  });

  factory GexMetrics.fromJson(Map<String, dynamic> json) {
    return GexMetrics(
      totalGex: (json['total_gex'] as num?)?.toDouble() ?? 0.0,
      putWall: (json['put_wall'] as num?)?.toDouble() ?? 0.0,
      callWall: (json['call_wall'] as num?)?.toDouble() ?? 0.0,
      flipLine: (json['flip_line'] as num?)?.toDouble(),
      spotPrice: (json['spot_price'] as num?)?.toDouble() ?? 0.0,
      regime: json['regime'] ?? 'unknown',
    );
  }
}

class GexBreakdown {
  final double callGex;
  final double putGex;
  final int totalContracts;
  final int skippedContracts;
  final int callContracts;
  final int putContracts;

  GexBreakdown({
    required this.callGex,
    required this.putGex,
    required this.totalContracts,
    required this.skippedContracts,
    required this.callContracts,
    required this.putContracts,
  });

  factory GexBreakdown.fromJson(Map<String, dynamic> json) {
    return GexBreakdown(
      callGex: (json['call_gex'] as num?)?.toDouble() ?? 0.0,
      putGex: (json['put_gex'] as num?)?.toDouble() ?? 0.0,
      totalContracts: json['total_contracts'] ?? 0,
      skippedContracts: json['skipped_contracts'] ?? 0,
      callContracts: json['call_contracts'] ?? 0,
      putContracts: json['put_contracts'] ?? 0,
    );
  }
}

class GexByStrike {
  final double strike;
  final double gex;
  final double cumulativeGex;

  GexByStrike({
    required this.strike,
    required this.gex,
    required this.cumulativeGex,
  });

  factory GexByStrike.fromJson(Map<String, dynamic> json) {
    return GexByStrike(
      strike: (json['strike'] as num?)?.toDouble() ?? 0.0,
      gex: (json['gex'] as num?)?.toDouble() ?? 0.0,
      cumulativeGex: (json['cumulative_gex'] as num?)?.toDouble() ?? 0.0,
    );
  }
}

class ChartAnnotations {
  final double spotPrice;
  final double? flipLine;
  final double putWall;
  final double callWall;

  ChartAnnotations({
    required this.spotPrice,
    this.flipLine,
    required this.putWall,
    required this.callWall,
  });

  factory ChartAnnotations.fromJson(Map<String, dynamic> json) {
    return ChartAnnotations(
      spotPrice: (json['spot_price'] as num?)?.toDouble() ?? 0.0,
      flipLine: (json['flip_line'] as num?)?.toDouble(),
      putWall: (json['put_wall'] as num?)?.toDouble() ?? 0.0,
      callWall: (json['call_wall'] as num?)?.toDouble() ?? 0.0,
    );
  }
}

class GexSummary {
  final List<GexTickerSummary> tickers;
  final List<Map<String, dynamic>>? errors;
  final double timestamp;

  GexSummary({
    required this.tickers,
    this.errors,
    required this.timestamp,
  });

  factory GexSummary.fromJson(Map<String, dynamic> json) {
    return GexSummary(
      tickers: (json['tickers'] as List?)
              ?.map((item) => GexTickerSummary.fromJson(item))
              .toList() ??
          [],
      errors: json['errors'] != null
          ? (json['errors'] as List).map((e) => Map<String, dynamic>.from(e)).toList()
          : null,
      timestamp: (json['timestamp'] as num?)?.toDouble() ?? 0.0,
    );
  }
}

class CumulativeGexPoint {
  final double strike;
  final double cumulativeGex;

  CumulativeGexPoint({
    required this.strike,
    required this.cumulativeGex,
  });

  factory CumulativeGexPoint.fromJson(Map<String, dynamic> json) {
    return CumulativeGexPoint(
      strike: (json['strike'] as num?)?.toDouble() ?? 0.0,
      cumulativeGex: (json['cumulative_gex'] as num?)?.toDouble() ?? 0.0,
    );
  }
}

class GammaSlope {
  final double slopeAtSpot;
  final double? slopeAtFlip;
  final String slopeBucket;
  final String interpretation;

  GammaSlope({
    required this.slopeAtSpot,
    this.slopeAtFlip,
    required this.slopeBucket,
    required this.interpretation,
  });

  factory GammaSlope.fromJson(Map<String, dynamic> json) {
    return GammaSlope(
      slopeAtSpot: (json['slope_at_spot'] as num?)?.toDouble() ?? 0.0,
      slopeAtFlip: (json['slope_at_flip'] as num?)?.toDouble(),
      slopeBucket: json['slope_bucket'] ?? 'NEUTRAL',
      interpretation: json['interpretation'] ?? 'Neutral',
    );
  }
}

class GexTickerSummary {
  final String ticker;
  final double spot;
  final double? flipLine;
  final double putWall;
  final double callWall;
  final double totalGex;
  final String regime;

  GexTickerSummary({
    required this.ticker,
    required this.spot,
    this.flipLine,
    required this.putWall,
    required this.callWall,
    required this.totalGex,
    required this.regime,
  });

  factory GexTickerSummary.fromJson(Map<String, dynamic> json) {
    return GexTickerSummary(
      ticker: json['ticker'] ?? '',
      spot: (json['spot'] as num?)?.toDouble() ?? 0.0,
      flipLine: (json['flip_line'] as num?)?.toDouble(),
      putWall: (json['put_wall'] as num?)?.toDouble() ?? 0.0,
      callWall: (json['call_wall'] as num?)?.toDouble() ?? 0.0,
      totalGex: (json['total_gex'] as num?)?.toDouble() ?? 0.0,
      regime: json['regime'] ?? 'unknown',
    );
  }
}
