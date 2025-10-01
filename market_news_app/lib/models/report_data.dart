import 'package:market_news_app/models/vix_data.dart';

class ReportData {
  final DateTime timestamp;
  final MarketSentiment marketSentiment;
  final List<TradeIdea> tradeIdeas;
  final List<SkippedTicker> skippedTickers;
  final List<VixData> vixData;
  final GammaAnalysis? gammaAnalysis;
  final List<TopStrategy> topStrategies;
  final List<Map<String, dynamic>> economicCalendar;
  final List<Map<String, dynamic>> earningsCalendar;
  final List<Map<String, dynamic>> topGainers;
  final List<Map<String, dynamic>> topLosers;
  final List<Map<String, dynamic>> indices;

  ReportData({
    required this.timestamp,
    required this.marketSentiment,
    required this.tradeIdeas,
    required this.skippedTickers,
    required this.vixData,
    this.gammaAnalysis,
    required this.topStrategies,
    required this.economicCalendar,
    required this.earningsCalendar,
    required this.topGainers,
    required this.topLosers,
    required this.indices,
  });

  factory ReportData.fromJson(Map<String, dynamic> json) {
    return ReportData(
      timestamp: json['timestamp'] != null
          ? DateTime.parse(json['timestamp'])
          : (json['generated_at'] != null
              ? DateTime.parse(json['generated_at'])
              : DateTime.now()),
      marketSentiment: MarketSentiment.fromJson(json['market_sentiment']),
      tradeIdeas: (json['trade_ideas'] ?? []).map<TradeIdea>((i) => TradeIdea.fromJson(i)).toList(),
      skippedTickers: (json['skipped_tickers'] ?? []).map<SkippedTicker>((i) => SkippedTicker.fromJson(i)).toList(),
      vixData: (json['vix_data'] ?? []).map<VixData>((i) => VixData.fromJson(i)).toList(),
      gammaAnalysis: json['gamma_analysis'] != null ? GammaAnalysis.fromJson(json['gamma_analysis']) : null,
      topStrategies: (json['top_strategies'] ?? []).map<TopStrategy>((i) => TopStrategy.fromJson(i)).toList(),
      economicCalendar: (json['economic_calendar'] ?? []).map<Map<String, dynamic>>((i) => Map<String, dynamic>.from(i)).toList(),
      earningsCalendar: (json['earnings_calendar'] ?? []).map<Map<String, dynamic>>((i) => Map<String, dynamic>.from(i)).toList(),
      topGainers: (json['top_gainers'] ?? []).map<Map<String, dynamic>>((i) => Map<String, dynamic>.from(i)).toList(),
      topLosers: (json['top_losers'] ?? []).map<Map<String, dynamic>>((i) => Map<String, dynamic>.from(i)).toList(),
      indices: (json['indices'] ?? []).map<Map<String, dynamic>>((i) => Map<String, dynamic>.from(i)).toList(),
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
  final double? ivRank;
  final double? currentIv;
  final double? expectedMove;
  final double? expectedMovePct;
  final double? breakEvenPrice;

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
    this.ivRank,
    this.currentIv,
    this.expectedMove,
    this.expectedMovePct,
    this.breakEvenPrice,
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
      ivRank: json['iv_rank']?.toDouble(),
      currentIv: json['current_iv']?.toDouble(),
      expectedMove: json['expected_move']?.toDouble(),
      expectedMovePct: json['expected_move_pct']?.toDouble(),
      breakEvenPrice: json['break_even_price']?.toDouble(),
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

class GammaAnalysis {
  final String marketRecommendation;
  final double avgGammaScore;
  final int gammaScalpingCount;
  final int premiumSellingCount;
  final int totalAnalyzed;
  final List<TickerGammaAnalysis> individualAnalysis;

  GammaAnalysis({
    required this.marketRecommendation,
    required this.avgGammaScore,
    required this.gammaScalpingCount,
    required this.premiumSellingCount,
    required this.totalAnalyzed,
    required this.individualAnalysis,
  });

  factory GammaAnalysis.fromJson(Map<String, dynamic> json) {
    return GammaAnalysis(
      marketRecommendation: json['market_recommendation'],
      avgGammaScore: json['avg_gamma_score'].toDouble(),
      gammaScalpingCount: json['gamma_scalping_count'],
      premiumSellingCount: json['premium_selling_count'],
      totalAnalyzed: json['total_analyzed'],
      individualAnalysis: (json['individual_analysis'] as List)
          .map((i) => TickerGammaAnalysis.fromJson(i))
          .toList(),
    );
  }
}

class TickerGammaAnalysis {
  final String ticker;
  final String recommendation;
  final String strategy;
  final double gammaScore;
  final List<String> reasons;
  final GammaAnalysisDetails analysis;

  TickerGammaAnalysis({
    required this.ticker,
    required this.recommendation,
    required this.strategy,
    required this.gammaScore,
    required this.reasons,
    required this.analysis,
  });

  factory TickerGammaAnalysis.fromJson(Map<String, dynamic> json) {
    return TickerGammaAnalysis(
      ticker: json['ticker'],
      recommendation: json['recommendation'],
      strategy: json['strategy'],
      gammaScore: json['gamma_score'].toDouble(),
      reasons: List<String>.from(json['reasons']),
      analysis: GammaAnalysisDetails.fromJson(json['analysis']),
    );
  }
}

class GammaAnalysisDetails {
  final double ivRvRatio;
  final double vixPercentile;
  final double rvAcceleration;
  final double currentVix;
  final double realizedVol30d;
  final double impliedVolatility;

  GammaAnalysisDetails({
    required this.ivRvRatio,
    required this.vixPercentile,
    required this.rvAcceleration,
    required this.currentVix,
    required this.realizedVol30d,
    required this.impliedVolatility,
  });

  factory GammaAnalysisDetails.fromJson(Map<String, dynamic> json) {
    return GammaAnalysisDetails(
      ivRvRatio: json['iv_rv_ratio'].toDouble(),
      vixPercentile: json['vix_percentile'].toDouble(),
      rvAcceleration: json['rv_acceleration'].toDouble(),
      currentVix: json['current_vix'].toDouble(),
      realizedVol30d: json['realized_vol_30d'].toDouble(),
      impliedVolatility: json['implied_volatility'].toDouble(),
    );
  }
}

class TopStrategy {
  final String name;
  final double score;
  final String description;
  final List<StrategyTicker> topTickers;

  TopStrategy({
    required this.name,
    required this.score,
    required this.description,
    required this.topTickers,
  });

  factory TopStrategy.fromJson(Map<String, dynamic> json) {
    return TopStrategy(
      name: json['name'],
      score: (json['score'] as num).toDouble(),
      description: json['description'],
      topTickers: (json['top_tickers'] != null)
          ? (json['top_tickers'] as List).map((i) => StrategyTicker.fromJson(i)).toList()
          : [],
    );
  }
}

class StrategyTicker {
  final String ticker;
  final double score;
  final Map<String, dynamic> setup;
  final String reason;

  StrategyTicker({
    required this.ticker,
    required this.score,
    required this.setup,
    required this.reason,
  });

  factory StrategyTicker.fromJson(Map<String, dynamic> json) {
    return StrategyTicker(
      ticker: json['ticker'],
      score: (json['score'] as num).toDouble(),
      setup: json['setup'] as Map<String, dynamic>,
      reason: json['reason'],
    );
  }
}