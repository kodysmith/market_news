class BullPutSpread {
  final String symbol;
  final int daysToExpiration;
  final double shortStrike;
  final double longStrike;
  final double credit;
  final double width;
  final double yieldPercent;
  final double shortPutDelta;
  final double ivRank;
  final double probabilityOfProfit;
  final double liquidityScore;
  final double technicalScore;
  final double compositeScore;
  final double maxProfit;
  final double maxLoss;
  final double riskRewardRatio;
  final double expectedValue;
  final double expectedValueScore; // 0-100 score for EV quality
  final DateTime expirationDate;
  final int openInterestShort;
  final int openInterestLong;
  final double bidAskSpreadShort;
  final double bidAskSpreadLong;
  final bool hasEarnings;
  final bool hasMacroEvent;

  BullPutSpread({
    required this.symbol,
    required this.daysToExpiration,
    required this.shortStrike,
    required this.longStrike,
    required this.credit,
    required this.width,
    required this.yieldPercent,
    required this.shortPutDelta,
    required this.ivRank,
    required this.probabilityOfProfit,
    required this.liquidityScore,
    required this.technicalScore,
    required this.compositeScore,
    required this.maxProfit,
    required this.maxLoss,
    required this.riskRewardRatio,
    required this.expectedValue,
    required this.expectedValueScore,
    required this.expirationDate,
    required this.openInterestShort,
    required this.openInterestLong,
    required this.bidAskSpreadShort,
    required this.bidAskSpreadLong,
    required this.hasEarnings,
    required this.hasMacroEvent,
  });

  factory BullPutSpread.fromJson(Map<String, dynamic> json) {
    return BullPutSpread(
      symbol: json['symbol'] ?? '',
      daysToExpiration: json['days_to_expiration'] ?? 0,
      shortStrike: (json['short_strike'] as num?)?.toDouble() ?? 0.0,
      longStrike: (json['long_strike'] as num?)?.toDouble() ?? 0.0,
      credit: (json['credit'] as num?)?.toDouble() ?? 0.0,
      width: (json['width'] as num?)?.toDouble() ?? 0.0,
      yieldPercent: (json['yield_percent'] as num?)?.toDouble() ?? 0.0,
      shortPutDelta: (json['short_put_delta'] as num?)?.toDouble() ?? 0.0,
      ivRank: (json['iv_rank'] as num?)?.toDouble() ?? 0.0,
      probabilityOfProfit: (json['probability_of_profit'] as num?)?.toDouble() ?? 0.0,
      liquidityScore: (json['liquidity_score'] as num?)?.toDouble() ?? 0.0,
      technicalScore: (json['technical_score'] as num?)?.toDouble() ?? 0.0,
      compositeScore: (json['composite_score'] as num?)?.toDouble() ?? 0.0,
      maxProfit: (json['max_profit'] as num?)?.toDouble() ?? 0.0,
      maxLoss: (json['max_loss'] as num?)?.toDouble() ?? 0.0,
      riskRewardRatio: (json['risk_reward_ratio'] as num?)?.toDouble() ?? 0.0,
      expectedValue: (json['expected_value'] as num?)?.toDouble() ?? 0.0,
      expectedValueScore: (json['expected_value_score'] as num?)?.toDouble() ?? 0.0,
      expirationDate: DateTime.parse(json['expiration_date'] ?? DateTime.now().toIso8601String()),
      openInterestShort: json['open_interest_short'] ?? 0,
      openInterestLong: json['open_interest_long'] ?? 0,
      bidAskSpreadShort: (json['bid_ask_spread_short'] as num?)?.toDouble() ?? 0.0,
      bidAskSpreadLong: (json['bid_ask_spread_long'] as num?)?.toDouble() ?? 0.0,
      hasEarnings: json['has_earnings'] ?? false,
      hasMacroEvent: json['has_macro_event'] ?? false,
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'symbol': symbol,
      'days_to_expiration': daysToExpiration,
      'short_strike': shortStrike,
      'long_strike': longStrike,
      'credit': credit,
      'width': width,
      'yield_percent': yieldPercent,
      'short_put_delta': shortPutDelta,
      'iv_rank': ivRank,
      'probability_of_profit': probabilityOfProfit,
      'liquidity_score': liquidityScore,
      'technical_score': technicalScore,
      'composite_score': compositeScore,
      'max_profit': maxProfit,
      'max_loss': maxLoss,
      'risk_reward_ratio': riskRewardRatio,
      'expected_value': expectedValue,
      'expected_value_score': expectedValueScore,
      'expiration_date': expirationDate.toIso8601String(),
      'open_interest_short': openInterestShort,
      'open_interest_long': openInterestLong,
      'bid_ask_spread_short': bidAskSpreadShort,
      'bid_ask_spread_long': bidAskSpreadLong,
      'has_earnings': hasEarnings,
      'has_macro_event': hasMacroEvent,
    };
  }

  // Quality score based on key criteria
  String get qualityGrade {
    if (compositeScore >= 8.0) return 'A+';
    if (compositeScore >= 7.5) return 'A';
    if (compositeScore >= 7.0) return 'A-';
    if (compositeScore >= 6.5) return 'B+';
    if (compositeScore >= 6.0) return 'B';
    if (compositeScore >= 5.5) return 'B-';
    if (compositeScore >= 5.0) return 'C+';
    return 'C';
  }

  // Risk level assessment
  String get riskLevel {
    if (shortPutDelta < 0.20 && ivRank > 60 && yieldPercent > 0.35) return 'LOW';
    if (shortPutDelta < 0.25 && ivRank > 50 && yieldPercent > 0.30) return 'MODERATE';
    if (shortPutDelta < 0.30 && ivRank > 40 && yieldPercent > 0.25) return 'MODERATE-HIGH';
    return 'HIGH';
  }

  // Strategy summary
  String get strategySummary {
    return 'Sell \$${shortStrike.toStringAsFixed(0)} Put / Buy \$${longStrike.toStringAsFixed(0)} Put';
  }

  // Profit potential description
  String get profitDescription {
    final profitPercent = (maxProfit / (maxProfit + maxLoss) * 100);
    return '${profitPercent.toStringAsFixed(1)}% max return if $symbol stays above \$${shortStrike.toStringAsFixed(0)}';
  }
}
