import 'dart:convert';
import 'package:flutter_dotenv/flutter_dotenv.dart';
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
  
  /// Get upcoming market events for cockpit display.
  /// Tries Flask API first; if unreachable, fetches from FMP directly when FMP_API_KEY is set in .env.
  static Future<CockpitEventsData?> getCockpitEvents({int daysAhead = 14, String? symbol}) async {
    var url = '$apiBaseUrl/cockpit/events?days=$daysAhead';
    if (symbol != null && symbol.isNotEmpty) {
      url += '&symbol=${Uri.encodeComponent(symbol)}';
    }
    print('[CockpitService] Fetching events: $url');

    try {
      final response = await http.get(Uri.parse(url)).timeout(
        const Duration(seconds: 8),
        onTimeout: () => throw Exception('timeout'),
      );
      if (response.statusCode == 200) {
        final json = jsonDecode(response.body);
        if (json is Map && json.containsKey('badges')) {
          return CockpitEventsData.fromJson(Map<String, dynamic>.from(json));
        }
      }
    } catch (e) {
      print('[CockpitService] API events failed: $e');
    }

    // Fallback: fetch from FMP directly (earnings + economic calendar)
    final fmpKey = dotenv.env['FMP_API_KEY']?.trim();
    if (fmpKey != null && fmpKey.isNotEmpty) {
      try {
        final data = await _getCockpitEventsFromFmp(daysAhead: daysAhead, symbol: symbol, apiKey: fmpKey);
        if (data != null) {
          print('[CockpitService] Using FMP direct events (${data.events.length} events)');
          return data;
        }
      } catch (e) {
        print('[CockpitService] FMP direct events failed: $e');
      }
    }
    return null;
  }

  /// Get market regime (SPX vs SMA50, VIX zone, stance).
  static Future<MarketRegime?> getMarketRegime() async {
    try {
      final url = '$apiBaseUrl/bot/market-regime';
      final response = await http.get(
        Uri.parse(url),
        headers: {'x-api-key': apiSecretKey},
      ).timeout(const Duration(seconds: 10));
      if (response.statusCode == 200) {
        return MarketRegime.fromJson(jsonDecode(response.body));
      }
    } catch (e) {
      print('[CockpitService] getMarketRegime error: $e');
    }
    return null;
  }

  /// Get FMP stock news for a symbol (for cockpit dashboard).
  static Future<List<Map<String, dynamic>>> getCockpitNews(String symbol, {int limit = 15}) async {
    final t = symbol.toUpperCase();
    try {
      final url = '$apiBaseUrl/cockpit/news?symbol=${Uri.encodeComponent(t)}&limit=$limit';
      final response = await http.get(Uri.parse(url)).timeout(const Duration(seconds: 10));
      if (response.statusCode == 200) {
        final json = jsonDecode(response.body);
        if (json is Map && json['news'] is List) {
          return (json['news'] as List).map((e) => Map<String, dynamic>.from(e as Map)).toList();
        }
      }
    } catch (e) {
      print('[CockpitService] getCockpitNews failed: $e');
    }
    return [];
  }

  static const List<String> _spyTopHoldings = [
    'AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'GOOG', 'META', 'TSLA', 'BRK.B',
    'JPM', 'V', 'MA', 'UNH', 'HD', 'PG', 'JNJ', 'XOM', 'CVX', 'ABBV', 'MRK',
    'PFE', 'KO', 'PEP', 'COST', 'WMT', 'BAC', 'WFC', 'NFLX', 'ADBE', 'CRM',
    'ORCL', 'INTC', 'AMD', 'QCOM',
  ];

  static const List<String> _fomcDates = [
    '2026-01-28', '2026-01-29', '2026-03-17', '2026-03-18', '2026-05-05', '2026-05-06',
    '2026-06-16', '2026-06-17', '2026-07-28', '2026-07-29', '2026-09-15', '2026-09-16',
    '2026-11-03', '2026-11-04', '2026-12-15', '2026-12-16',
  ];

  static Future<CockpitEventsData?> _getCockpitEventsFromFmp({
    required int daysAhead,
    String? symbol,
    required String apiKey,
  }) async {
    final now = DateTime.now();
    final end = now.add(Duration(days: daysAhead));
    final fromStr = '${now.year}-${now.month.toString().padLeft(2, '0')}-${now.day.toString().padLeft(2, '0')}';
    final toStr = '${end.year}-${end.month.toString().padLeft(2, '0')}-${end.day.toString().padLeft(2, '0')}';
    final today = DateTime(now.year, now.month, now.day);

    final List<Map<String, dynamic>> events = [];

    // 1. FMP earnings calendar (SPY top holdings only)
    try {
      final earningsUrl = 'https://financialmodelingprep.com/api/v3/earning_calendar?from=$fromStr&to=$toStr&apikey=$apiKey';
      final earningsResp = await http.get(Uri.parse(earningsUrl)).timeout(const Duration(seconds: 10));
      if (earningsResp.statusCode == 200) {
        final list = jsonDecode(earningsResp.body);
        if (list is List) {
          for (final e in list) {
            final ticker = (e['symbol'] ?? '').toString().toUpperCase();
            if (!_spyTopHoldings.contains(ticker)) continue;
            final date = e['date']?.toString();
            if (date == null || date.isEmpty) continue;
            final time = (e['time'] ?? '').toString().toLowerCase();
            final timeLabel = time == 'bmo' ? 'BMO' : (time == 'amc' ? 'AH' : '');
            events.add({
              'type': 'earnings',
              'title': '$ticker Earnings',
              'date': date,
              'time': timeLabel,
              'impact': 'high',
              'ticker': ticker,
              'source': 'fmp',
            });
          }
        }
      }
    } catch (_) {}

    // 2. FMP economic calendar (US, medium/high impact)
    try {
      final econUrl = 'https://financialmodelingprep.com/api/v3/economic_calendar?from=$fromStr&to=$toStr&apikey=$apiKey';
      final econResp = await http.get(Uri.parse(econUrl)).timeout(const Duration(seconds: 10));
      if (econResp.statusCode == 200) {
        final list = jsonDecode(econResp.body);
        if (list is List) {
          for (final e in list) {
            if ((e['country'] ?? '').toString().toUpperCase() != 'US') continue;
            final impact = (e['impact'] ?? '').toString();
            if (impact.toUpperCase() != 'HIGH' && impact.toUpperCase() != 'MEDIUM') continue;
            final date = e['date']?.toString();
            if (date == null || date.isEmpty) continue;
            final eventName = (e['event'] ?? e['title'] ?? 'Event').toString();
            final type = eventName.toUpperCase().contains('FOMC') ? 'fomc'
                : eventName.toUpperCase().contains('CPI') ? 'cpi'
                : eventName.toUpperCase().contains('PPI') ? 'ppi'
                : eventName.toUpperCase().contains('NFP') || eventName.toUpperCase().contains('EMPLOYMENT') ? 'nfp'
                : eventName.toUpperCase().contains('GDP') ? 'gdp' : 'economic';
            events.add({
              'type': type,
              'title': eventName,
              'date': date.split(' ').first,
              'time': date.contains(' ') ? date.split(' ').last : '08:30',
              'impact': impact.toLowerCase(),
              'ticker': null,
              'source': 'fmp',
            });
          }
        }
      }
    } catch (_) {}

    // 3. OPEX (3rd Friday of month)
    int thirdFriday(int year, int month) {
      for (int day = 15; day <= 21; day++) {
        final dt = DateTime(year, month, day);
        if (dt.weekday == DateTime.friday) return day;
      }
      return 15;
    }
    for (int monthOffset = 0; monthOffset < 2; monthOffset++) {
      final d = DateTime(now.year, now.month + monthOffset, 1);
      final opexDay = thirdFriday(d.year, d.month);
      final opexDate = DateTime(d.year, d.month, opexDay);
      if (!opexDate.isBefore(today) && !opexDate.isAfter(end)) {
        events.add({
          'type': 'opex',
          'title': 'Monthly OPEX',
          'date': '${opexDate.year}-${opexDate.month.toString().padLeft(2, '0')}-${opexDate.day.toString().padLeft(2, '0')}',
          'time': 'Close',
          'impact': 'high',
          'ticker': null,
          'source': 'calculated',
        });
      }
    }

    // 4. FOMC (hardcoded decision days)
    const fomcDecisionSuffixes = ['29', '18', '06', '17', '16', '04'];
    for (final dateStr in _fomcDates) {
      if (fomcDecisionSuffixes.any((s) => dateStr.endsWith(s))) {
        try {
          final d = DateTime.parse(dateStr);
          if (!d.isBefore(today) && !d.isAfter(end)) {
            events.add({
              'type': 'fomc',
              'title': 'FOMC Decision',
              'date': dateStr,
              'time': '2:00 PM',
              'impact': 'high',
              'ticker': null,
              'source': 'scheduled',
            });
          }
        } catch (_) {}
      }
    }

    // Dedupe by date+type+title
    final seen = <String>{};
    final unique = <Map<String, dynamic>>[];
    for (final e in events) {
      final key = '${e['date']}_${e['type']}_${e['title']}';
      if (seen.add(key)) unique.add(e);
    }
    unique.sort((a, b) => (a['date'] as String).compareTo(b['date'] as String));

    // Build badges (top 3)
    final badges = <Map<String, dynamic>>[];
    for (final event in unique.take(3)) {
      final dateStr = event['date'] as String;
      final eventDate = DateTime.tryParse(dateStr) ?? today;
      final daysUntil = eventDate.difference(today).inDays;
      final dateLabel = daysUntil == 0 ? 'Today' : (daysUntil == 1 ? 'Tomorrow' : _dayLabel(eventDate));
      String badgeText;
      if (event['type'] == 'earnings') {
        badgeText = (event['ticker'] ?? '').toString();
      } else if (event['type'] == 'opex') {
        badgeText = 'OPEX';
      } else if (event['type'] == 'fomc') {
        badgeText = 'FOMC';
      } else {
        final title = (event['title'] ?? '').toString().toUpperCase();
        badgeText = title.contains('CPI') ? 'CPI' : title.contains('PPI') ? 'PPI' : title.contains('NFP') ? 'NFP' : title.contains('GDP') ? 'GDP' : (event['title'] ?? '').toString().length > 10 ? '${(event['title'] as String).substring(0, 10)}…' : (event['title'] ?? '').toString();
      }
      badges.add({
        'text': badgeText,
        'type': event['type'],
        'date_label': dateLabel,
        'full_title': event['title'],
        'impact': event['impact'],
      });
    }

    return CockpitEventsData(
      badges: badges.map((e) => EventBadge.fromJson(e)).toList(),
      events: unique.map((e) => CockpitEvent.fromJson(Map<String, dynamic>.from(e))).toList(),
      daysAhead: daysAhead,
      count: unique.length,
    );
  }

  static String _dayLabel(DateTime d) {
    const days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
    return days[d.weekday - 1];
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

/// Market regime: SPX vs SMA50, VIX zone, actionable guidance
class MarketRegime {
  final double spxPrice;
  final double sma50;
  final bool aboveSma50;
  final double distancePct;
  final double vix;
  final String vixZone;
  final String vixLabel;
  final double? rvRatio;
  final String stance;
  final String guidance;

  MarketRegime({
    required this.spxPrice,
    required this.sma50,
    required this.aboveSma50,
    required this.distancePct,
    required this.vix,
    required this.vixZone,
    required this.vixLabel,
    this.rvRatio,
    required this.stance,
    required this.guidance,
  });

  factory MarketRegime.fromJson(Map<String, dynamic> json) {
    return MarketRegime(
      spxPrice: (json['spx_price'] as num?)?.toDouble() ?? 0,
      sma50: (json['sma50'] as num?)?.toDouble() ?? 0,
      aboveSma50: json['above_sma50'] ?? false,
      distancePct: (json['distance_pct'] as num?)?.toDouble() ?? 0,
      vix: (json['vix'] as num?)?.toDouble() ?? 0,
      vixZone: json['vix_zone'] ?? '',
      vixLabel: json['vix_label'] ?? '',
      rvRatio: (json['rv_ratio'] as num?)?.toDouble(),
      stance: json['stance'] ?? '',
      guidance: json['guidance'] ?? '',
    );
  }

  bool get isBullish => stance == 'BULLISH';
  bool get isDefensive => stance == 'DEFENSIVE';
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
