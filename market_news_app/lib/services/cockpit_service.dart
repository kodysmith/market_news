import 'dart:convert';
import 'package:http/http.dart' as http;
import '../main.dart';
import 'compute_queue_service.dart';

/// Wrapper for cockpit state plus cache timestamp (for staleness checks).
class CockpitStateResult {
  final CockpitState? state;
  final DateTime? updatedAt;

  CockpitStateResult({this.state, this.updatedAt});
}

/// Service for Decision Cockpit: reads only from Supabase compute_result_cache (precomputed every 5 min). No API for state/tickers.
class CockpitService {
  /// Get the complete cockpit state for a ticker plus cache updated_at. Tries precompute cache first, then enqueues and waits if missing.
  /// Returns state and updatedAt for staleness; updatedAt is null when data came from enqueue-and-wait.
  static Future<CockpitStateResult> getCockpitState(String ticker) async {
    final t = ticker.toUpperCase();
    if (!ComputeQueueService.isAvailable) {
      print('[CockpitService] Supabase not configured (SUPABASE_URL, SUPABASE_ANON_KEY)');
      return CockpitStateResult(state: null, updatedAt: null);
    }
    final cached = await ComputeQueueService.getCachedResultFromTable(
      symbol: t,
      taskType: ComputeTaskType.cockpit,
    );
    if (cached != null) {
      try {
        final state = CockpitState.fromJson(cached.result);
        return CockpitStateResult(state: state, updatedAt: cached.updatedAt);
      } catch (e, st) {
        print('[CockpitService] Parse Supabase result: $e');
        print(st);
        return CockpitStateResult(state: null, updatedAt: cached.updatedAt);
      }
    }
    final jobResult = await ComputeQueueService.getCachedOrEnqueue(
      symbol: t,
      taskType: ComputeTaskType.cockpit,
    );
    if (jobResult.ok && jobResult.result != null) {
      try {
        final state = CockpitState.fromJson(jobResult.result!);
        return CockpitStateResult(state: state, updatedAt: null);
      } catch (e, st) {
        print('[CockpitService] Parse enqueue result: $e');
        print(st);
        return CockpitStateResult(state: null, updatedAt: null);
      }
    }
    return CockpitStateResult(state: null, updatedAt: null);
  }

  /// Get supported tickers (core symbols from Supabase precompute).
  static Future<List<String>> getSupportedTickers() async {
    return List.from(ComputeQueueService.coreSymbols);
  }
  
  /// Get upcoming market events for cockpit display
  static Future<CockpitEventsData?> getCockpitEvents({int daysAhead = 3, String? symbol}) async {
    var url = '$apiBaseUrl/cockpit/events?days=$daysAhead';
    if (symbol != null && symbol.isNotEmpty) {
      url += '&symbol=${Uri.encodeComponent(symbol)}';
    }
    print('[CockpitService] Fetching events: $url');
    
    try {
      final response = await http.get(Uri.parse(url));
      print('[CockpitService] Events response status: ${response.statusCode}');
      
      if (response.statusCode == 200) {
        final json = jsonDecode(response.body);
        return CockpitEventsData.fromJson(json);
      } else {
        print('[CockpitService] Events error: ${response.body}');
        return null;
      }
    } catch (e) {
      print('[CockpitService] Events exception: $e');
      return null;
    }
  }
}

/// Cockpit events data container
class CockpitEventsData {
  final List<EventBadge> badges;
  final List<CockpitEvent> events;
  final int daysAhead;
  final int count;
  
  CockpitEventsData({
    required this.badges,
    required this.events,
    required this.daysAhead,
    required this.count,
  });
  
  factory CockpitEventsData.fromJson(Map<String, dynamic> json) {
    return CockpitEventsData(
      badges: (json['badges'] as List?)
          ?.map((e) => EventBadge.fromJson(e))
          .toList() ?? [],
      events: (json['events'] as List?)
          ?.map((e) => CockpitEvent.fromJson(e))
          .toList() ?? [],
      daysAhead: json['days_ahead'] ?? 3,
      count: json['count'] ?? 0,
    );
  }
}

/// Compact event badge for header display
class EventBadge {
  final String text;
  final String type;  // 'fomc', 'cpi', 'ppi', 'nfp', 'earnings', 'opex', 'economic'
  final String dateLabel;  // 'Today', 'Tomorrow', 'Mon', etc.
  final String fullTitle;
  final String impact;
  
  EventBadge({
    required this.text,
    required this.type,
    required this.dateLabel,
    required this.fullTitle,
    required this.impact,
  });
  
  factory EventBadge.fromJson(Map<String, dynamic> json) {
    return EventBadge(
      text: json['text'] ?? '',
      type: json['type'] ?? 'economic',
      dateLabel: json['date_label'] ?? '',
      fullTitle: json['full_title'] ?? '',
      impact: json['impact'] ?? 'medium',
    );
  }
}

/// Full event data for expandable calendar
class CockpitEvent {
  final String type;  // 'fomc', 'cpi', 'ppi', 'nfp', 'earnings', 'opex', 'economic', 'gdp'
  final String title;
  final String date;
  final String time;
  final String impact;
  final String? ticker;  // For earnings
  final String source;
  final String? impactOnSymbol;  // 'high' | 'medium' | 'low' when symbol param was passed
  
  CockpitEvent({
    required this.type,
    required this.title,
    required this.date,
    required this.time,
    required this.impact,
    this.ticker,
    required this.source,
    this.impactOnSymbol,
  });
  
  factory CockpitEvent.fromJson(Map<String, dynamic> json) {
    return CockpitEvent(
      type: json['type'] ?? 'economic',
      title: json['title'] ?? '',
      date: json['date'] ?? '',
      time: json['time'] ?? '',
      impact: json['impact'] ?? 'medium',
      ticker: json['ticker'],
      source: json['source'] ?? '',
      impactOnSymbol: json['impact_on_symbol'],
    );
  }
  
  /// Get color based on event type
  int get colorValue {
    switch (type) {
      case 'fomc':
        return 0xFFE53935;  // Red
      case 'cpi':
      case 'ppi':
      case 'nfp':
        return 0xFFFF9800;  // Orange
      case 'earnings':
        return 0xFF2196F3;  // Blue
      case 'opex':
        return 0xFF9C27B0;  // Purple
      case 'gdp':
        return 0xFF4CAF50;  // Green
      default:
        return 0xFF757575;  // Gray
    }
  }
}

/// Driver chip for de-pin risk breakdown
class DriverChip {
  final String name;
  final double contribution;
  
  DriverChip({
    required this.name,
    required this.contribution,
  });
  
  factory DriverChip.fromJson(Map<String, dynamic> json) {
    return DriverChip(
      name: json['name'] ?? '',
      contribution: (json['contribution'] ?? 0).toDouble(),
    );
  }
}

/// De-pin risk data
class DepinRiskData {
  final int score;
  final String band;  // LOW, MID, HIGH
  final int? delta30m;
  final String? deltaDirection;  // up, down, stable
  final String guidance;
  final List<DriverChip> drivers;
  
  DepinRiskData({
    required this.score,
    required this.band,
    this.delta30m,
    this.deltaDirection,
    required this.guidance,
    required this.drivers,
  });
  
  factory DepinRiskData.fromJson(Map<String, dynamic> json) {
    return DepinRiskData(
      score: json['score'] ?? 0,
      band: json['band'] ?? 'LOW',
      delta30m: json['delta_30m'],
      deltaDirection: json['delta_direction'],
      guidance: json['guidance'] ?? '',
      drivers: (json['drivers'] as List?)
          ?.map((e) => DriverChip.fromJson(e))
          .toList() ?? [],
    );
  }
}

/// Quote for dashboard price strip (Block A)
class CockpitQuote {
  final double? current;
  final double? previousClose;
  final double? open;
  final double? change;
  final double? changePct;
  
  CockpitQuote({
    this.current,
    this.previousClose,
    this.open,
    this.change,
    this.changePct,
  });
  
  factory CockpitQuote.fromJson(Map<String, dynamic>? json) {
    if (json == null) return CockpitQuote();
    return CockpitQuote(
      current: (json['current'] as num?)?.toDouble(),
      previousClose: (json['previous_close'] as num?)?.toDouble(),
      open: (json['open'] as num?)?.toDouble(),
      change: (json['change'] as num?)?.toDouble(),
      changePct: (json['change_pct'] as num?)?.toDouble(),
    );
  }
}

/// Volatility strategy (Block C): headline + rationale + vol_allowed/forbidden
class VolatilityStrategy {
  final String headline;
  final String rationale;
  final List<String> volAllowed;
  final List<String> volForbidden;
  
  VolatilityStrategy({
    required this.headline,
    required this.rationale,
    this.volAllowed = const [],
    this.volForbidden = const [],
  });
  
  factory VolatilityStrategy.fromJson(Map<String, dynamic>? json) {
    if (json == null) return VolatilityStrategy(headline: '', rationale: '');
    return VolatilityStrategy(
      headline: json['headline'] ?? '',
      rationale: json['rationale'] ?? '',
      volAllowed: List<String>.from(json['vol_allowed'] ?? []),
      volForbidden: List<String>.from(json['vol_forbidden'] ?? []),
    );
  }
}

/// Max pain for nearest expiry (Block D)
class CockpitMaxPain {
  final double strike;
  final String expiration;
  
  CockpitMaxPain({required this.strike, required this.expiration});
  
  factory CockpitMaxPain.fromJson(Map<String, dynamic>? json) {
    if (json == null) return CockpitMaxPain(strike: 0, expiration: '');
    return CockpitMaxPain(
      strike: (json['strike'] as num?)?.toDouble() ?? 0,
      expiration: json['expiration'] ?? '',
    );
  }
}

/// Complete cockpit state model
class CockpitState {
  final String ticker;
  final String timestamp;
  final RegimeState regime;
  final VolatilityState volatility;
  final VolatilityStrategy? volatilityStrategy;
  final StructureState structure;
  final ActionFilter actionFilter;
  final String? opex;
  final int contractsProcessed;
  final int contractsTotal;
  final DepinRiskData? depinRisk;
  final CockpitQuote? quote;
  final CockpitMaxPain? maxPain;
  
  CockpitState({
    required this.ticker,
    required this.timestamp,
    required this.regime,
    required this.volatility,
    required this.structure,
    required this.actionFilter,
    this.opex,
    required this.contractsProcessed,
    required this.contractsTotal,
    this.depinRisk,
    this.volatilityStrategy,
    this.quote,
    this.maxPain,
  });
  
  factory CockpitState.fromJson(Map<String, dynamic> json) {
    return CockpitState(
      ticker: json['ticker'] ?? '',
      timestamp: json['timestamp'] ?? '',
      regime: RegimeState.fromJson(json['regime'] ?? {}),
      volatility: VolatilityState.fromJson(json['volatility'] ?? {}),
      volatilityStrategy: json['volatility_strategy'] != null
          ? VolatilityStrategy.fromJson(json['volatility_strategy'])
          : null,
      structure: StructureState.fromJson(json['structure'] ?? {}),
      actionFilter: ActionFilter.fromJson(json['action_filter'] ?? {}),
      opex: json['opex'],
      contractsProcessed: json['contracts_processed'] ?? 0,
      contractsTotal: json['contracts_total'] ?? 0,
      depinRisk: json['de_pin_risk'] != null 
          ? DepinRiskData.fromJson(json['de_pin_risk'])
          : null,
      quote: json['quote'] != null ? CockpitQuote.fromJson(json['quote']) : null,
      maxPain: json['max_pain'] != null ? CockpitMaxPain.fromJson(json['max_pain']) : null,
    );
  }
}

/// Regime state (GEX) - Enhanced with transition detection and GEX breakdown
class RegimeState {
  final String label;
  final double? spot;
  final double? flipLine;
  final String? flipLineReason;  // Explains why flip line might be null
  final String bias;
  final double distanceToFlip;
  final double distanceToFlipPct;
  final double? netGex;
  final double? callGex;    // Total call GEX
  final double? putGex;     // Total put GEX
  final double? gexRatio;   // Call/Put ratio
  final bool isPositive;
  final bool isNegative;
  final bool transition;
  final String? transitionReason;
  
  RegimeState({
    required this.label,
    this.spot,
    this.flipLine,
    this.flipLineReason,
    required this.bias,
    required this.distanceToFlip,
    required this.distanceToFlipPct,
    this.netGex,
    this.callGex,
    this.putGex,
    this.gexRatio,
    required this.isPositive,
    required this.isNegative,
    required this.transition,
    this.transitionReason,
  });
  
  factory RegimeState.fromJson(Map<String, dynamic> json) {
    final rawLabel = json['label']?.toString().trim();
    final isPositive = json['is_positive'] ?? false;
    final isNegative = json['is_negative'] ?? false;
    final transition = json['transition'] ?? false;
    // Derive a readable label when API sends none or "UNKNOWN"
    String label = rawLabel ?? '';
    if (label.isEmpty || label.toUpperCase() == 'UNKNOWN') {
      if (transition) {
        label = 'NEAR FLIP';
      } else if (isPositive) {
        label = 'POSITIVE GEX';
      } else if (isNegative) {
        label = 'NEGATIVE GEX';
      } else {
        label = 'MIXED / NEUTRAL';
      }
    }
    return RegimeState(
      label: label,
      spot: json['spot']?.toDouble(),
      flipLine: json['flip_line']?.toDouble(),
      flipLineReason: json['flip_line_reason'],
      bias: json['bias'] ?? '',
      distanceToFlip: (json['distance_to_flip'] ?? 0).toDouble(),
      distanceToFlipPct: (json['distance_to_flip_pct'] ?? 0).toDouble(),
      netGex: json['net_gex']?.toDouble(),
      callGex: json['call_gex']?.toDouble(),
      putGex: json['put_gex']?.toDouble(),
      gexRatio: json['gex_ratio']?.toDouble(),
      isPositive: isPositive,
      isNegative: isNegative,
      transition: transition,
      transitionReason: json['transition_reason'],
    );
  }
}

/// Volatility state
class VolatilityState {
  final double? frontIv;
  final double frontIvChange1h;
  final double frontIvChange1d;
  final String direction;
  final String directionSymbol;
  final String termStructure;
  final String state;
  final double? vix;
  final double? vixChange;
  
  VolatilityState({
    this.frontIv,
    required this.frontIvChange1h,
    required this.frontIvChange1d,
    required this.direction,
    required this.directionSymbol,
    required this.termStructure,
    required this.state,
    this.vix,
    this.vixChange,
  });
  
  factory VolatilityState.fromJson(Map<String, dynamic> json) {
    return VolatilityState(
      frontIv: json['front_iv']?.toDouble(),
      frontIvChange1h: (json['front_iv_change_1h'] ?? 0).toDouble(),
      frontIvChange1d: (json['front_iv_change_1d'] ?? 0).toDouble(),
      direction: json['direction'] ?? 'FLAT',
      directionSymbol: json['direction_symbol'] ?? '→',
      termStructure: json['term_structure'] ?? 'UNKNOWN',
      state: json['state'] ?? 'UNKNOWN',
      vix: json['vix']?.toDouble(),
      vixChange: json['vix_change']?.toDouble(),
    );
  }
}

/// Wall pair (put/call) for a single lens
class WallPair {
  final double? put;
  final double? call;
  
  WallPair({this.put, this.call});
  
  factory WallPair.fromJson(Map<String, dynamic>? json) {
    if (json == null) return WallPair();
    return WallPair(
      put: json['put']?.toDouble(),
      call: json['call']?.toDouble(),
    );
  }
}

/// Zone description for behavioral context
class ZoneDescription {
  final String type;
  final String summary;
  final String behavior;
  
  ZoneDescription({
    required this.type,
    required this.summary,
    required this.behavior,
  });
  
  factory ZoneDescription.fromJson(Map<String, dynamic>? json) {
    if (json == null) return ZoneDescription(type: '', summary: '', behavior: '');
    return ZoneDescription(
      type: json['type'] ?? '',
      summary: json['summary'] ?? '',
      behavior: json['behavior'] ?? '',
    );
  }
}

/// Level label for actionable context
class LevelLabel {
  final double? strike;
  final String type;
  final String source;
  final String label;
  final double? distance;
  
  LevelLabel({
    this.strike,
    required this.type,
    required this.source,
    required this.label,
    this.distance,
  });
  
  factory LevelLabel.fromJson(Map<String, dynamic> json) {
    return LevelLabel(
      strike: json['strike']?.toDouble(),
      type: json['type'] ?? '',
      source: json['source'] ?? '',
      label: json['label'] ?? '',
      distance: json['distance']?.toDouble(),
    );
  }
}

/// Structure state - Enhanced with GEX + OI walls and behavioral labels
class StructureState {
  // Primary walls (GEX-based for behavior)
  final WallPair primaryWalls;
  // OI walls (for context/validation)
  final WallPair oiPrimaryWalls;
  
  // Multi-lens walls (legacy format for compatibility)
  final WallPair wallsRegime;    // 0-60 DTE
  final WallPair wallsTactical;  // 0-14 DTE
  final WallPair wallsToday;     // 0-2 DTE
  final WallPair wallsOpex;      // Next monthly OPEX expiry only
  
  // Zone and distance metrics
  final List<double>? noTradeZone;
  final bool inNoTradeZone;
  final double? distanceToPutWall;
  final double? distanceToCallWall;
  final double? distanceToNearest;
  final String? nearestWall;
  final double? wallRange;
  final double? positionInRangePct;
  
  // Behavioral labels
  final List<LevelLabel> putLevels;
  final List<LevelLabel> callLevels;
  final ZoneDescription? zoneDescription;
  
  StructureState({
    required this.primaryWalls,
    required this.oiPrimaryWalls,
    required this.wallsRegime,
    required this.wallsTactical,
    required this.wallsToday,
    WallPair? wallsOpex,
    this.noTradeZone,
    required this.inNoTradeZone,
    this.distanceToPutWall,
    this.distanceToCallWall,
    this.distanceToNearest,
    this.nearestWall,
    this.wallRange,
    this.positionInRangePct,
    required this.putLevels,
    required this.callLevels,
    this.zoneDescription,
  }) : wallsOpex = wallsOpex ?? WallPair();
  
  // Convenience getters for primary (GEX) walls
  double? get putWall => primaryWalls.put;
  double? get callWall => primaryWalls.call;
  
  factory StructureState.fromJson(Map<String, dynamic> json) {
    List<double>? noTradeZone;
    if (json['no_trade_zone'] != null) {
      noTradeZone = List<double>.from(
        (json['no_trade_zone'] as List).map((e) => e.toDouble())
      );
    }
    
    // Parse level labels
    final levelLabels = json['level_labels'] ?? {};
    final putLevelsList = (levelLabels['put_levels'] as List?)
        ?.map((e) => LevelLabel.fromJson(e))
        .toList() ?? [];
    final callLevelsList = (levelLabels['call_levels'] as List?)
        ?.map((e) => LevelLabel.fromJson(e))
        .toList() ?? [];
    
    return StructureState(
      primaryWalls: WallPair.fromJson(json['primary_walls']),
      oiPrimaryWalls: WallPair.fromJson(json['oi_primary_walls']),
      wallsRegime: WallPair.fromJson(json['walls_regime']),
      wallsTactical: WallPair.fromJson(json['walls_tactical']),
      wallsToday: WallPair.fromJson(json['walls_today']),
      wallsOpex: WallPair.fromJson(json['walls_opex']),
      noTradeZone: noTradeZone,
      inNoTradeZone: json['in_no_trade_zone'] ?? false,
      distanceToPutWall: json['distance_to_put_wall']?.toDouble(),
      distanceToCallWall: json['distance_to_call_wall']?.toDouble(),
      distanceToNearest: json['distance_to_nearest']?.toDouble(),
      nearestWall: json['nearest_wall'],
      wallRange: json['wall_range']?.toDouble(),
      positionInRangePct: json['position_in_range_pct']?.toDouble(),
      putLevels: putLevelsList,
      callLevels: callLevelsList,
      zoneDescription: levelLabels['zone_description'] != null
          ? ZoneDescription.fromJson(levelLabels['zone_description'])
          : null,
    );
  }
}

/// Action filter (allowed/forbidden)
class ActionFilter {
  final List<String> allowed;
  final List<String> forbidden;
  
  ActionFilter({
    required this.allowed,
    required this.forbidden,
  });
  
  factory ActionFilter.fromJson(Map<String, dynamic> json) {
    return ActionFilter(
      allowed: List<String>.from(json['allowed'] ?? []),
      forbidden: List<String>.from(json['forbidden'] ?? []),
    );
  }
}
