class QuoteV1 {
  final String ticker;
  final double currentPrice;
  final double? change;
  final double? changePercent;
  final int? volume;
  final String timestamp;

  QuoteV1({
    required this.ticker,
    required this.currentPrice,
    this.change,
    this.changePercent,
    this.volume,
    required this.timestamp,
  });

  factory QuoteV1.fromJson(Map<String, dynamic> json) {
    return QuoteV1(
      ticker: (json['ticker'] ?? json['symbol'] ?? '').toString(),
      currentPrice: (json['current_price'] ?? json['currentPrice'] ?? json['price'] ?? 0).toDouble(),
      change: (json['change'] as num?)?.toDouble(),
      changePercent: (json['change_percent'] ?? json['changePercent'] as num?)?.toDouble(),
      volume: (json['volume'] as num?)?.toInt(),
      timestamp: (json['timestamp'] ?? json['as_of'] ?? '').toString(),
    );
  }

  Map<String, dynamic> toJson() => {
        'ticker': ticker,
        'current_price': currentPrice,
        'change': change,
        'change_percent': changePercent,
        'volume': volume,
        'timestamp': timestamp,
      };
}

