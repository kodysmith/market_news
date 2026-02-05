import 'dart:math';

import '../models/opportunity_v1.dart';
import 'indicators.dart';

class OpportunityScorer {
  /// Produces a lightweight, stable score \(0-100\) and an `OpportunityV1` shape.
  ///
  /// This is intentionally simple to run on-device. It can be upgraded later
  /// without changing the wire contract.
  static OpportunityV1 score({
    required String ticker,
    required String sector,
    required List<double> closes,
    required DateTime asOf,
  }) {
    final currentPrice = closes.isNotEmpty ? closes.last : 0.0;

    final rsi = Indicators.rsi(closes);
    final sma20 = Indicators.sma(closes, 20);
    final sma50 = Indicators.sma(closes, 50);
    final trend = Indicators.trend(currentPrice: currentPrice, sma20: sma20, sma50: sma50);
    final (support, resistance) = Indicators.supportResistance(closes);

    // ---- score components ----
    double score = 50; // base

    // RSI: oversold bullish, overbought bearish
    if (rsi < 30) {
      score += 20;
    } else if (rsi > 70) {
      score -= 20;
    } else {
      score += 10; // neutral band
    }

    // Trend
    if (trend == 'uptrend') score += 20;
    if (trend == 'downtrend') score -= 20;

    // Position in range (near support bullish, near resistance bearish)
    if (support > 0 && resistance > support) {
      final pos = (currentPrice - support) / (resistance - support); // 0..1
      if (pos < 0.3) score += 10;
      if (pos > 0.7) score -= 10;
    }

    score = score.clamp(0, 100);

    final opportunityType = _opportunityType(score);

    // Targets/stops: simple percent bands; can be upgraded later.
    final targetPrice = _targetPrice(
      currentPrice: currentPrice,
      opportunityType: opportunityType,
      resistance: resistance,
      support: support,
    );
    final stopLoss = _stopLoss(currentPrice: currentPrice, opportunityType: opportunityType);
    final riskReward = _riskReward(currentPrice: currentPrice, targetPrice: targetPrice, stopLoss: stopLoss);

    return OpportunityV1(
      ticker: ticker,
      sector: sector,
      opportunityType: opportunityType,
      overallScore: score,
      currentPrice: currentPrice,
      targetPrice: targetPrice,
      stopLoss: stopLoss,
      riskRewardRatio: riskReward,
      rsi: rsi,
      trendDirection: trend,
      timestamp: asOf.toIso8601String(),
    );
  }

  static String _opportunityType(double score) {
    if (score >= 80) return 'STRONG_BUY';
    if (score >= 65) return 'BUY';
    if (score >= 35) return 'HOLD';
    if (score >= 20) return 'SELL';
    return 'STRONG_SELL';
  }

  static double? _targetPrice({
    required double currentPrice,
    required String opportunityType,
    required double resistance,
    required double support,
  }) {
    if (currentPrice <= 0) return null;
    if (opportunityType.contains('BUY')) {
      if (resistance > 0) return resistance;
      return currentPrice * 1.07;
    }
    if (opportunityType.contains('SELL')) {
      if (support > 0) return support;
      return currentPrice * 0.93;
    }
    return currentPrice;
  }

  static double? _stopLoss({required double currentPrice, required String opportunityType}) {
    if (currentPrice <= 0) return null;
    if (opportunityType.contains('BUY')) return currentPrice * 0.95;
    if (opportunityType.contains('SELL')) return currentPrice * 1.05;
    return currentPrice;
  }

  static double? _riskReward({
    required double currentPrice,
    required double? targetPrice,
    required double? stopLoss,
  }) {
    if (currentPrice <= 0 || targetPrice == null || stopLoss == null) return null;

    final reward = (targetPrice - currentPrice).abs();
    final risk = (currentPrice - stopLoss).abs();
    if (risk <= 0) return null;
    return max(0, reward / risk);
  }
}

