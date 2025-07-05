import 'package:flutter/foundation.dart';

class ReportData {
  final DateTime timestamp;
  final MarketSentiment marketSentiment;
  final List<TradeIdea> tradeIdeas;
  final List<SkippedTicker> skippedTickers;

  ReportData({
    required this.timestamp,
    required this.marketSentiment,
    required this.tradeIdeas,
    required this.skippedTickers,
  });

  factory ReportData.fromJson(Map<String, dynamic> json) {
    return ReportData(
      timestamp: DateTime.parse(json['timestamp']),
      marketSentiment: MarketSentiment.fromJson(json['market_sentiment']),
      tradeIdeas: (json['trade_ideas'] as List)
          .map((i) => TradeIdea.fromJson(i))
          .toList(),
      skippedTickers: (json['skipped_tickers'] as List)
          .map((i) => SkippedTicker.fromJson(i))
          .toList(),
    );
  }
}

class MarketSentiment {
  final List<Indicator> indicators;
  final String sentiment;

  MarketSentiment({
    required this.indicators,
    required this.sentiment,
  });

  factory MarketSentiment.fromJson(Map<String, dynamic> json) {
    return MarketSentiment(
      indicators: (json['indicators'] as List)
          .map((i) => Indicator.fromJson(i))
          .toList(),
      sentiment: json['sentiment'],
    );
  }
}

class Indicator {
  final String name;
  final String ticker;
  final String price;
  final String direction;

  Indicator({
    required this.name,
    required this.ticker,
    required this.price,
    required this.direction,
  });

  factory Indicator.fromJson(Map<String, dynamic> json) {
    return Indicator(
      name: json['name'],
      ticker: json['ticker'],
      price: json['price'],
      direction: json['direction'],
    );
  }
}

class TradeIdea {
  final String ticker;
  final String strategy;
  final String expiry;
  final String details;
  final double cost;
  final String metricName;
  final String metricValue;
  final double? maxProfit;
  final double? maxLoss;
  final double? riskRewardRatio;

  TradeIdea({
    required this.ticker,
    required this.strategy,
    required this.expiry,
    required this.details,
    required this.cost,
    required this.metricName,
    required this.metricValue,
    this.maxProfit,
    this.maxLoss,
    this.riskRewardRatio,
  });

  factory TradeIdea.fromJson(Map<String, dynamic> json) {
    return TradeIdea(
      ticker: json['ticker'],
      strategy: json['strategy'],
      expiry: json['expiry'],
      details: json['details'],
      cost: json['cost'].toDouble(),
      metricName: json['metric_name'],
      metricValue: json['metric_value'],
      maxProfit: json['max_profit']?.toDouble(),
      maxLoss: json['max_loss']?.toDouble(),
      riskRewardRatio: json['risk_reward_ratio']?.toDouble(),
    );
  }
}

class SkippedTicker {
  final String ticker;
  final String strategy;
  final String details;
  final double cost;
  final String metricName;
  final String metricValue;

  SkippedTicker({
    required this.ticker,
    required this.strategy,
    required this.details,
    required this.cost,
    required this.metricName,
    required this.metricValue,
  });

  factory SkippedTicker.fromJson(Map<String, dynamic> json) {
    return SkippedTicker(
      ticker: json['ticker'],
      strategy: json['strategy'],
      details: json['details'],
      cost: json['cost'].toDouble(),
      metricName: json['metric_name'],
      metricValue: json['metric_value'],
    );
  }
}