class OpportunityV1 {
  final String ticker;
  final String sector;
  final String opportunityType;
  final double overallScore;
  final double currentPrice;
  final double? targetPrice;
  final double? stopLoss;
  final double? riskRewardRatio;
  final double? rsi;
  final String? trendDirection;
  final String timestamp;

  OpportunityV1({
    required this.ticker,
    required this.sector,
    required this.opportunityType,
    required this.overallScore,
    required this.currentPrice,
    this.targetPrice,
    this.stopLoss,
    this.riskRewardRatio,
    this.rsi,
    this.trendDirection,
    required this.timestamp,
  });

  factory OpportunityV1.fromJson(Map<String, dynamic> json) {
    return OpportunityV1(
      ticker: (json['ticker'] ?? '').toString(),
      sector: (json['sector'] ?? '').toString(),
      opportunityType: (json['opportunity_type'] ?? json['opportunityType'] ?? 'UNKNOWN').toString(),
      overallScore: (json['overall_score'] ?? json['overallScore'] ?? 0).toDouble(),
      currentPrice: (json['current_price'] ?? json['currentPrice'] ?? 0).toDouble(),
      targetPrice: (json['target_price'] ?? json['targetPrice'] as num?)?.toDouble(),
      stopLoss: (json['stop_loss'] ?? json['stopLoss'] as num?)?.toDouble(),
      riskRewardRatio: (json['risk_reward_ratio'] ?? json['riskRewardRatio'] as num?)?.toDouble(),
      rsi: (json['rsi'] as num?)?.toDouble(),
      trendDirection: (json['trend_direction'] ?? json['trendDirection'])?.toString(),
      timestamp: (json['timestamp'] ?? json['scan_date'] ?? '').toString(),
    );
  }

  Map<String, dynamic> toJson() => {
        'ticker': ticker,
        'sector': sector,
        'opportunity_type': opportunityType,
        'overall_score': overallScore,
        'current_price': currentPrice,
        'target_price': targetPrice,
        'stop_loss': stopLoss,
        'risk_reward_ratio': riskRewardRatio,
        'rsi': rsi,
        'trend_direction': trendDirection,
        'timestamp': timestamp,
      };
}

