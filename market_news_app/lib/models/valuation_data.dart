/// Models for intrinsic value calculations and divergence tracking

/// Individual valuation method result
class ValuationMethod {
  final double? value;
  final double confidence;
  final bool applicable;
  final String? reason;
  final Map<String, dynamic>? details;

  ValuationMethod({
    this.value,
    required this.confidence,
    required this.applicable,
    this.reason,
    this.details,
  });

  factory ValuationMethod.fromJson(Map<String, dynamic> json) {
    return ValuationMethod(
      value: json['value']?.toDouble(),
      confidence: (json['confidence'] ?? 0).toDouble(),
      applicable: json['applicable'] ?? false,
      reason: json['reason'],
      details: json['details'],
    );
  }

  Map<String, dynamic> toJson() => {
    'value': value,
    'confidence': confidence,
    'applicable': applicable,
    'reason': reason,
    'details': details,
  };
}

/// Composite valuation result (weighted average of all methods)
class CompositeValuation {
  final double? value;
  final int methodCount;
  final List<String> methodsUsed;
  final double? divergencePct;
  final String verdict;

  CompositeValuation({
    this.value,
    required this.methodCount,
    required this.methodsUsed,
    this.divergencePct,
    required this.verdict,
  });

  factory CompositeValuation.fromJson(Map<String, dynamic> json) {
    return CompositeValuation(
      value: json['value']?.toDouble(),
      methodCount: json['method_count'] ?? 0,
      methodsUsed: List<String>.from(json['methods_used'] ?? []),
      divergencePct: json['divergence_pct']?.toDouble(),
      verdict: json['verdict'] ?? 'Unknown',
    );
  }

  Map<String, dynamic> toJson() => {
    'value': value,
    'method_count': methodCount,
    'methods_used': methodsUsed,
    'divergence_pct': divergencePct,
    'verdict': verdict,
  };
  
  /// Get color based on verdict
  String get verdictColor {
    if (verdict.contains('Undervalued')) return 'green';
    if (verdict.contains('Overvalued')) return 'red';
    return 'yellow';
  }
}

/// Full valuation calculation result
class ValuationResult {
  final String ticker;
  final String? companyName;
  final String? sector;
  final String? industry;
  final double currentPrice;
  final Map<String, ValuationMethod> valuations;
  final CompositeValuation composite;
  final String timestamp;
  final List<ValuationAlert>? alerts;

  ValuationResult({
    required this.ticker,
    this.companyName,
    this.sector,
    this.industry,
    required this.currentPrice,
    required this.valuations,
    required this.composite,
    required this.timestamp,
    this.alerts,
  });

  factory ValuationResult.fromJson(Map<String, dynamic> json) {
    final valuationsJson = json['valuations'] as Map<String, dynamic>? ?? {};
    final valuations = <String, ValuationMethod>{};
    
    valuationsJson.forEach((key, value) {
      valuations[key] = ValuationMethod.fromJson(value as Map<String, dynamic>);
    });

    return ValuationResult(
      ticker: json['ticker'] ?? '',
      companyName: json['company_name'],
      sector: json['sector'],
      industry: json['industry'],
      currentPrice: (json['current_price'] ?? 0).toDouble(),
      valuations: valuations,
      composite: CompositeValuation.fromJson(json['composite'] ?? {}),
      timestamp: json['timestamp'] ?? '',
      alerts: (json['alerts'] as List<dynamic>?)
          ?.map((a) => ValuationAlert.fromJson(a))
          .toList(),
    );
  }

  Map<String, dynamic> toJson() => {
    'ticker': ticker,
    'company_name': companyName,
    'sector': sector,
    'industry': industry,
    'current_price': currentPrice,
    'valuations': valuations.map((k, v) => MapEntry(k, v.toJson())),
    'composite': composite.toJson(),
    'timestamp': timestamp,
    'alerts': alerts?.map((a) => a.toJson()).toList(),
  };

  /// Get list of applicable valuation methods
  List<MapEntry<String, ValuationMethod>> get applicableMethods {
    return valuations.entries
        .where((e) => e.value.applicable && e.value.value != null)
        .toList();
  }
}

/// Valuation alert
class ValuationAlert {
  final int? id;
  final String ticker;
  final String alertType;
  final double divergencePct;
  final double price;
  final double compositeValue;
  final String? message;
  final String? triggeredAt;
  final bool acknowledged;

  ValuationAlert({
    this.id,
    required this.ticker,
    required this.alertType,
    required this.divergencePct,
    required this.price,
    required this.compositeValue,
    this.message,
    this.triggeredAt,
    this.acknowledged = false,
  });

  factory ValuationAlert.fromJson(Map<String, dynamic> json) {
    return ValuationAlert(
      id: json['id'],
      ticker: json['ticker'] ?? '',
      alertType: json['alert_type'] ?? '',
      divergencePct: (json['divergence_pct'] ?? 0).toDouble(),
      price: (json['price'] ?? 0).toDouble(),
      compositeValue: (json['composite_value'] ?? 0).toDouble(),
      message: json['message'],
      triggeredAt: json['triggered_at'],
      acknowledged: json['acknowledged'] == 1 || json['acknowledged'] == true,
    );
  }

  Map<String, dynamic> toJson() => {
    'id': id,
    'ticker': ticker,
    'alert_type': alertType,
    'divergence_pct': divergencePct,
    'price': price,
    'composite_value': compositeValue,
    'message': message,
    'triggered_at': triggeredAt,
    'acknowledged': acknowledged,
  };
}

/// Historical valuation record
class ValuationHistory {
  final String ticker;
  final String date;
  final double price;
  final double? dcfValue;
  final double? grahamValue;
  final double? ddmValue;
  final double? epvValue;
  final double? compsValue;
  final double? navValue;
  final double? compositeValue;
  final double? divergencePct;
  final String? verdict;

  ValuationHistory({
    required this.ticker,
    required this.date,
    required this.price,
    this.dcfValue,
    this.grahamValue,
    this.ddmValue,
    this.epvValue,
    this.compsValue,
    this.navValue,
    this.compositeValue,
    this.divergencePct,
    this.verdict,
  });

  factory ValuationHistory.fromJson(Map<String, dynamic> json) {
    return ValuationHistory(
      ticker: json['ticker'] ?? '',
      date: json['date'] ?? '',
      price: (json['price'] ?? 0).toDouble(),
      dcfValue: json['dcf_value']?.toDouble(),
      grahamValue: json['graham_value']?.toDouble(),
      ddmValue: json['ddm_value']?.toDouble(),
      epvValue: json['epv_value']?.toDouble(),
      compsValue: json['comps_value']?.toDouble(),
      navValue: json['nav_value']?.toDouble(),
      compositeValue: json['composite_value']?.toDouble(),
      divergencePct: json['divergence_pct']?.toDouble(),
      verdict: json['verdict'],
    );
  }
}

/// Watchlist item with latest valuation
class WatchlistItem {
  final String ticker;
  final String? companyName;
  final String? sector;
  final String addedAt;
  final String? lastCalculated;
  final double alertThresholdPct;
  final double? price;
  final double? compositeValue;
  final double? divergencePct;
  final String? verdict;
  final String? date;

  WatchlistItem({
    required this.ticker,
    this.companyName,
    this.sector,
    required this.addedAt,
    this.lastCalculated,
    required this.alertThresholdPct,
    this.price,
    this.compositeValue,
    this.divergencePct,
    this.verdict,
    this.date,
  });

  factory WatchlistItem.fromJson(Map<String, dynamic> json) {
    return WatchlistItem(
      ticker: json['ticker'] ?? '',
      companyName: json['company_name'],
      sector: json['sector'],
      addedAt: json['added_at'] ?? '',
      lastCalculated: json['last_calculated'],
      alertThresholdPct: (json['alert_threshold_pct'] ?? 20).toDouble(),
      price: json['price']?.toDouble(),
      compositeValue: json['composite_value']?.toDouble(),
      divergencePct: json['divergence_pct']?.toDouble(),
      verdict: json['verdict'],
      date: json['date'],
    );
  }
}

/// Scan result for undervalued/overvalued stocks
class ValuationScanResult {
  final String ticker;
  final String? companyName;
  final double currentPrice;
  final double? compositeValue;
  final double divergencePct;
  final String verdict;

  ValuationScanResult({
    required this.ticker,
    this.companyName,
    required this.currentPrice,
    this.compositeValue,
    required this.divergencePct,
    required this.verdict,
  });

  factory ValuationScanResult.fromJson(Map<String, dynamic> json) {
    return ValuationScanResult(
      ticker: json['ticker'] ?? '',
      companyName: json['company_name'],
      currentPrice: (json['current_price'] ?? 0).toDouble(),
      compositeValue: json['composite_value']?.toDouble(),
      divergencePct: (json['divergence_pct'] ?? 0).toDouble(),
      verdict: json['verdict'] ?? 'Unknown',
    );
  }
}

/// Divergence breakdown by method
class DivergenceBreakdown {
  final String ticker;
  final String? companyName;
  final double currentPrice;
  final double? compositeIntrinsicValue;
  final double? compositeDivergencePct;
  final String? verdict;
  final Map<String, MethodDivergence> breakdown;
  final String timestamp;

  DivergenceBreakdown({
    required this.ticker,
    this.companyName,
    required this.currentPrice,
    this.compositeIntrinsicValue,
    this.compositeDivergencePct,
    this.verdict,
    required this.breakdown,
    required this.timestamp,
  });

  factory DivergenceBreakdown.fromJson(Map<String, dynamic> json) {
    final breakdownJson = json['breakdown'] as Map<String, dynamic>? ?? {};
    final breakdown = <String, MethodDivergence>{};
    
    breakdownJson.forEach((key, value) {
      breakdown[key] = MethodDivergence.fromJson(value as Map<String, dynamic>);
    });

    return DivergenceBreakdown(
      ticker: json['ticker'] ?? '',
      companyName: json['company_name'],
      currentPrice: (json['current_price'] ?? 0).toDouble(),
      compositeIntrinsicValue: json['composite_intrinsic_value']?.toDouble(),
      compositeDivergencePct: json['composite_divergence_pct']?.toDouble(),
      verdict: json['verdict'],
      breakdown: breakdown,
      timestamp: json['timestamp'] ?? '',
    );
  }
}

/// Per-method divergence
class MethodDivergence {
  final double intrinsicValue;
  final double divergencePct;
  final double confidence;

  MethodDivergence({
    required this.intrinsicValue,
    required this.divergencePct,
    required this.confidence,
  });

  factory MethodDivergence.fromJson(Map<String, dynamic> json) {
    return MethodDivergence(
      intrinsicValue: (json['intrinsic_value'] ?? 0).toDouble(),
      divergencePct: (json['divergence_pct'] ?? 0).toDouble(),
      confidence: (json['confidence'] ?? 0).toDouble(),
    );
  }
}
