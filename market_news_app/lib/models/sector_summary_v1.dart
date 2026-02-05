class SectorSummaryV1 {
  final String sector;
  final int totalOpportunities;
  final double avgScore;
  final double? maxScore;
  final double? minScore;
  final int strongBuyCount;
  final int buyCount;
  final int holdCount;
  final int sellCount;
  final int strongSellCount;
  final String timestamp;

  SectorSummaryV1({
    required this.sector,
    required this.totalOpportunities,
    required this.avgScore,
    this.maxScore,
    this.minScore,
    required this.strongBuyCount,
    required this.buyCount,
    required this.holdCount,
    required this.sellCount,
    required this.strongSellCount,
    required this.timestamp,
  });

  factory SectorSummaryV1.fromJson(Map<String, dynamic> json) {
    return SectorSummaryV1(
      sector: (json['sector'] ?? '').toString(),
      totalOpportunities: (json['total_opportunities'] ?? json['totalOpportunities'] ?? 0) as int,
      avgScore: (json['avg_score'] ?? json['avgScore'] ?? 0).toDouble(),
      maxScore: (json['max_score'] ?? json['maxScore'] as num?)?.toDouble(),
      minScore: (json['min_score'] ?? json['minScore'] as num?)?.toDouble(),
      strongBuyCount: (json['strong_buy_count'] ?? json['strongBuyCount'] ?? 0) as int,
      buyCount: (json['buy_count'] ?? json['buyCount'] ?? 0) as int,
      holdCount: (json['hold_count'] ?? json['holdCount'] ?? 0) as int,
      sellCount: (json['sell_count'] ?? json['sellCount'] ?? 0) as int,
      strongSellCount: (json['strong_sell_count'] ?? json['strongSellCount'] ?? 0) as int,
      timestamp: (json['timestamp'] ?? json['summary_timestamp'] ?? '').toString(),
    );
  }

  Map<String, dynamic> toJson() => {
        'sector': sector,
        'total_opportunities': totalOpportunities,
        'avg_score': avgScore,
        'max_score': maxScore,
        'min_score': minScore,
        'strong_buy_count': strongBuyCount,
        'buy_count': buyCount,
        'hold_count': holdCount,
        'sell_count': sellCount,
        'strong_sell_count': strongSellCount,
        'timestamp': timestamp,
      };
}

