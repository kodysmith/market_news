import 'dart:convert';

/// Trade Idea Model - Regime-Aware Allowed-Only Trade Ideas
class TradeIdea {
  final String id;
  final String symbol;
  final String asOf;
  final String strategy;
  final String label;
  final bool allowed;
  final String mode; // "INTRADAY_ONLY" | "OVERNIGHT_ALLOWED" | "HEDGE_REQUIRED"
  final String confidence; // "HIGH" | "MEDIUM" | "LOW"
  final List<String> reasons;
  final String status; // "UNLOCKED" | "PREVIEW"
  final String? blockReason; // "TIME_LOCKED" | "REGIME_BLOCKED" | "REGIME_NEAR_UNLOCK" | "IMPLEMENTATION_LOCKED"
  final Map<String, dynamic>? nextWindow; // NextWindow info for preview ideas
  final bool preMarket; // True if market is closed
  final String? marketStatusNote; // Warning message for pre-market ideas
  final List<ReferenceExecution> executions;
  final BestMetrics? best;
  final ContextSnapshot? contextSnapshot;

  TradeIdea({
    required this.id,
    required this.symbol,
    required this.asOf,
    required this.strategy,
    required this.label,
    required this.allowed,
    required this.mode,
    required this.confidence,
    required this.reasons,
    this.status = 'UNLOCKED',
    this.blockReason,
    this.nextWindow,
    this.preMarket = false,
    this.marketStatusNote,
    required this.executions,
    this.best,
    this.contextSnapshot,
  });

  factory TradeIdea.fromJson(Map<String, dynamic> json) {
    return TradeIdea(
      id: json['id'] ?? '',
      symbol: json['symbol'] ?? '',
      asOf: json['asOf'] ?? '',
      strategy: json['strategy'] ?? '',
      label: json['label'] ?? '',
      allowed: json['allowed'] ?? true,
      mode: json['mode'] ?? 'OVERNIGHT_ALLOWED',
      confidence: json['confidence'] ?? 'MEDIUM',
      reasons: (json['reasons'] as List?)?.map((e) => e.toString()).toList() ?? [],
      status: json['status'] ?? 'UNLOCKED',
      blockReason: json['blockReason'],
      nextWindow: json['nextWindow'] != null ? Map<String, dynamic>.from(json['nextWindow']) : null,
      preMarket: json['preMarket'] ?? false,
      marketStatusNote: json['marketStatusNote'],
      executions: (json['executions'] as List?)
          ?.map((e) => ReferenceExecution.fromJson(e))
          .toList() ?? [],
      best: json['best'] != null ? BestMetrics.fromJson(json['best']) : null,
      contextSnapshot: json['contextSnapshot'] != null
          ? ContextSnapshot.fromJson(json['contextSnapshot'])
          : null,
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'symbol': symbol,
      'asOf': asOf,
      'strategy': strategy,
      'label': label,
      'allowed': allowed,
      'mode': mode,
      'confidence': confidence,
      'reasons': reasons,
      'status': status,
      'blockReason': blockReason,
      'nextWindow': nextWindow,
      'preMarket': preMarket,
      'marketStatusNote': marketStatusNote,
      'executions': executions.map((e) => e.toJson()).toList(),
      'best': best?.toJson(),
      'contextSnapshot': contextSnapshot?.toJson(),
    };
  }
}

/// Reference Execution - Individual spread structure
class ReferenceExecution {
  final String expiry;
  final int dte;
  final List<ExecutionLeg> legs;
  final ExecutionMetrics metrics;

  ReferenceExecution({
    required this.expiry,
    required this.dte,
    required this.legs,
    required this.metrics,
  });

  factory ReferenceExecution.fromJson(Map<String, dynamic> json) {
    return ReferenceExecution(
      expiry: json['expiry'] ?? '',
      dte: json['dte'] ?? 0,
      legs: (json['legs'] as List?)
          ?.map((e) => ExecutionLeg.fromJson(e))
          .toList() ?? [],
      metrics: ExecutionMetrics.fromJson(json['metrics'] ?? {}),
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'expiry': expiry,
      'dte': dte,
      'legs': legs.map((e) => e.toJson()).toList(),
      'metrics': metrics.toJson(),
    };
  }
}

/// Execution Leg - Single leg of a spread
class ExecutionLeg {
  final String action; // "SELL" | "BUY"
  final String type; // "PUT" | "CALL"
  final double strike;
  final double priceMid;
  final double bid;
  final double ask;
  final double? delta;
  final int oi;
  final int volume;

  ExecutionLeg({
    required this.action,
    required this.type,
    required this.strike,
    required this.priceMid,
    required this.bid,
    required this.ask,
    this.delta,
    required this.oi,
    required this.volume,
  });

  factory ExecutionLeg.fromJson(Map<String, dynamic> json) {
    return ExecutionLeg(
      action: json['action'] ?? '',
      type: json['type'] ?? '',
      strike: (json['strike'] ?? 0.0).toDouble(),
      priceMid: (json['priceMid'] ?? 0.0).toDouble(),
      bid: (json['bid'] ?? 0.0).toDouble(),
      ask: (json['ask'] ?? 0.0).toDouble(),
      delta: json['delta']?.toDouble(),
      oi: json['oi'] ?? 0,
      volume: json['volume'] ?? 0,
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'action': action,
      'type': type,
      'strike': strike,
      'priceMid': priceMid,
      'bid': bid,
      'ask': ask,
      'delta': delta,
      'oi': oi,
      'volume': volume,
    };
  }
}

/// Execution Metrics
class ExecutionMetrics {
  final double netCredit;
  final double width;
  final double maxProfit;
  final double maxLoss;
  final double roc;
  final double breakeven;
  final double pop;
  final double ev;
  final double wallBufferPct;
  final double liquidityScore;
  final double score;

  ExecutionMetrics({
    required this.netCredit,
    required this.width,
    required this.maxProfit,
    required this.maxLoss,
    required this.roc,
    required this.breakeven,
    required this.pop,
    required this.ev,
    required this.wallBufferPct,
    required this.liquidityScore,
    required this.score,
  });

  factory ExecutionMetrics.fromJson(Map<String, dynamic> json) {
    return ExecutionMetrics(
      netCredit: (json['netCredit'] ?? 0.0).toDouble(),
      width: (json['width'] ?? 0.0).toDouble(),
      maxProfit: (json['maxProfit'] ?? 0.0).toDouble(),
      maxLoss: (json['maxLoss'] ?? 0.0).toDouble(),
      roc: (json['roc'] ?? 0.0).toDouble(),
      breakeven: (json['breakeven'] ?? 0.0).toDouble(),
      pop: (json['pop'] ?? 0.0).toDouble(),
      ev: (json['ev'] ?? 0.0).toDouble(),
      wallBufferPct: (json['wallBufferPct'] ?? 0.0).toDouble(),
      liquidityScore: (json['liquidityScore'] ?? 0.0).toDouble(),
      score: (json['score'] ?? 0.0).toDouble(),
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'netCredit': netCredit,
      'width': width,
      'maxProfit': maxProfit,
      'maxLoss': maxLoss,
      'roc': roc,
      'breakeven': breakeven,
      'pop': pop,
      'ev': ev,
      'wallBufferPct': wallBufferPct,
      'liquidityScore': liquidityScore,
      'score': score,
    };
  }
}

/// Best Metrics - Summary for collapsed view
class BestMetrics {
  final String expiry;
  final int dte;
  final double netCredit;
  final double maxProfit;
  final double maxLoss;
  final double roc;
  final double pop;
  final double ev;
  final double wallBufferPct;
  final String liquidityGrade; // "A" | "B" | "C" | "D"
  final double score;

  BestMetrics({
    required this.expiry,
    required this.dte,
    required this.netCredit,
    required this.maxProfit,
    required this.maxLoss,
    required this.roc,
    required this.pop,
    required this.ev,
    required this.wallBufferPct,
    required this.liquidityGrade,
    required this.score,
  });

  factory BestMetrics.fromJson(Map<String, dynamic> json) {
    return BestMetrics(
      expiry: json['expiry'] ?? '',
      dte: json['dte'] ?? 0,
      netCredit: (json['netCredit'] ?? 0.0).toDouble(),
      maxProfit: (json['maxProfit'] ?? 0.0).toDouble(),
      maxLoss: (json['maxLoss'] ?? 0.0).toDouble(),
      roc: (json['roc'] ?? 0.0).toDouble(),
      pop: (json['pop'] ?? 0.0).toDouble(),
      ev: (json['ev'] ?? 0.0).toDouble(),
      wallBufferPct: (json['wallBufferPct'] ?? 0.0).toDouble(),
      liquidityGrade: json['liquidityGrade'] ?? 'C',
      score: (json['score'] ?? 0.0).toDouble(),
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'expiry': expiry,
      'dte': dte,
      'netCredit': netCredit,
      'maxProfit': maxProfit,
      'maxLoss': maxLoss,
      'roc': roc,
      'pop': pop,
      'ev': ev,
      'wallBufferPct': wallBufferPct,
      'liquidityGrade': liquidityGrade,
      'score': score,
    };
  }
}

/// Context Snapshot - Regime state at generation time
class ContextSnapshot {
  final double spot;
  final double? putWall;
  final double? callWall;
  final double? flipLine;
  final double? depinRisk;
  final String gexState;
  final String pinConfidence;
  final int earningsClusterScore;
  final bool macroEventNext24h;

  ContextSnapshot({
    required this.spot,
    this.putWall,
    this.callWall,
    this.flipLine,
    this.depinRisk,
    required this.gexState,
    required this.pinConfidence,
    required this.earningsClusterScore,
    required this.macroEventNext24h,
  });

  factory ContextSnapshot.fromJson(Map<String, dynamic> json) {
    return ContextSnapshot(
      spot: (json['spot'] ?? 0.0).toDouble(),
      putWall: json['putWall']?.toDouble(),
      callWall: json['callWall']?.toDouble(),
      flipLine: json['flipLine']?.toDouble(),
      depinRisk: json['depinRisk']?.toDouble(),
      gexState: json['gexState'] ?? 'NEUTRAL',
      pinConfidence: json['pinConfidence'] ?? 'LOW',
      earningsClusterScore: json['earningsClusterScore'] ?? 0,
      macroEventNext24h: json['macroEventNext24h'] ?? false,
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'spot': spot,
      'putWall': putWall,
      'callWall': callWall,
      'flipLine': flipLine,
      'depinRisk': depinRisk,
      'gexState': gexState,
      'pinConfidence': pinConfidence,
      'earningsClusterScore': earningsClusterScore,
      'macroEventNext24h': macroEventNext24h,
    };
  }
}
