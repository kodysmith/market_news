import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:supabase_flutter/supabase_flutter.dart';
import '../models/trade_idea.dart';
import '../main.dart' show apiBaseUrl, apiSecretKey;
import 'compute_queue_service.dart';

/// One row from trade_ideas_published (idea + valid date for display).
class PublishedTradeIdea {
  final String validDate;
  final String? asOfEt;
  final TradeIdea idea;

  PublishedTradeIdea({required this.validDate, this.asOfEt, required this.idea});
}

/// Service for fetching allowed trade ideas.
/// 1) Prefer Supabase trade_ideas_published (today/yesterday) when available — no API.
/// 2) Else compute_result_cache / compute_job_queue; 3) else API.
class TradeIdeasService {
  static const String _publishedTable = 'trade_ideas_published';

  /// Get today and yesterday in ET as YYYY-MM-DD for querying published ideas.
  static List<String> _todayAndYesterdayEt() {
    try {
      final now = DateTime.now().toUtc();
      // Approximate ET = UTC - 5 (or -4 DST). Use 4h offset for simplicity (ET ahead of UTC in winter).
      final et = now.subtract(const Duration(hours: 5));
      final today = '${et.year}-${et.month.toString().padLeft(2, '0')}-${et.day.toString().padLeft(2, '0')}';
      final yesterday = DateTime(et.year, et.month, et.day).subtract(const Duration(days: 1));
      final yesterdayStr = '${yesterday.year}-${yesterday.month.toString().padLeft(2, '0')}-${yesterday.day.toString().padLeft(2, '0')}';
      return [today, yesterdayStr];
    } catch (_) {
      final t = DateTime.now();
      final today = '${t.year}-${t.month.toString().padLeft(2, '0')}-${t.day.toString().padLeft(2, '0')}';
      return [today];
    }
  }

  /// Fetch published trade ideas from Supabase (valid for today or yesterday ET).
  /// Returns list of PublishedTradeIdea, or empty if table missing / no rows / Supabase unavailable.
  static Future<List<PublishedTradeIdea>> fetchPublishedByDate() async {
    if (!ComputeQueueService.isAvailable) return [];
    final dates = _todayAndYesterdayEt();
    try {
      final rows = await Supabase.instance.client
          .from(_publishedTable)
          .select('valid_date, as_of_et, symbol, payload')
          .inFilter('valid_date', dates)
          .order('valid_date', ascending: false)
          .order('as_of_et', ascending: false);
      if (rows.isEmpty) return [];
      final list = <PublishedTradeIdea>[];
      for (final row in rows) {
        final map = row is Map<String, dynamic> ? row : Map<String, dynamic>.from(row as Map);
        final payload = map['payload'];
        final payloadMap = payload is Map ? Map<String, dynamic>.from(payload as Map) : null;
        if (payloadMap == null) continue;
        try {
          list.add(PublishedTradeIdea(
            validDate: map['valid_date'] as String? ?? '',
            asOfEt: map['as_of_et'] as String?,
            idea: TradeIdea.fromJson(payloadMap),
          ));
        } catch (_) {
          // skip malformed row
        }
      }
      return list;
    } catch (_) {
      return [];
    }
  }
  /// Parse raw API/cache response into the map shape the screen expects.
  static Map<String, dynamic> _parseTradeIdeasResponse(dynamic decoded) {
    if (decoded is Map<String, dynamic>) {
      final result = <String, dynamic>{
        'diagnostics': decoded['diagnostics'] ?? <String, dynamic>{},
      };
      if (decoded.containsKey('unlocked')) {
        final unlockedData = decoded['unlocked'];
        if (unlockedData is List) {
          result['unlocked'] = unlockedData
              .map((json) => TradeIdea.fromJson(json as Map<String, dynamic>))
              .toList();
        } else if (unlockedData is Map) {
          final unlockedByTimeframe = <String, List<TradeIdea>>{};
          for (final entry in (unlockedData as Map<String, dynamic>).entries) {
            unlockedByTimeframe[entry.key] = (entry.value as List)
                .map((json) => TradeIdea.fromJson(json as Map<String, dynamic>))
                .toList();
          }
          result['unlockedByTimeframe'] = unlockedByTimeframe;
        }
      }
      if (decoded.containsKey('preview')) {
        result['preview'] = (decoded['preview'] as List)
            .map((json) => TradeIdea.fromJson(json as Map<String, dynamic>))
            .toList();
      }
      if (decoded.containsKey('nextWindow')) {
        result['nextWindow'] = decoded['nextWindow'] as Map<String, dynamic>;
      }
      if (decoded.containsKey('currentPhase')) {
        result['currentPhase'] = decoded['currentPhase'] as String;
      }
      if (decoded.containsKey('ideasByTimeframe')) {
        final ideasByTimeframe = decoded['ideasByTimeframe'] as Map<String, dynamic>;
        final ideasByTimeframeMap = <String, List<TradeIdea>>{};
        for (final entry in ideasByTimeframe.entries) {
          ideasByTimeframeMap[entry.key] = (entry.value as List)
              .map((json) => TradeIdea.fromJson(json as Map<String, dynamic>))
              .toList();
        }
        result['ideasByTimeframe'] = ideasByTimeframeMap;
      } else if (decoded.containsKey('ideas')) {
        result['ideas'] = (decoded['ideas'] as List)
            .map((json) => TradeIdea.fromJson(json as Map<String, dynamic>))
            .toList();
      }
      return result;
    }
    if (decoded is List) {
      return {
        'unlocked': decoded
            .map((json) => TradeIdea.fromJson(json as Map<String, dynamic>))
            .toList(),
        'preview': <TradeIdea>[],
        'diagnostics': <String, dynamic>{},
      };
    }
    throw Exception('Unexpected response format: ${decoded.runtimeType}');
  }

  /// Fetch allowed trade ideas for a ticker.
  /// Uses Supabase compute_result_cache / getCachedOrEnqueue when available; else calls API.
  /// Cached/enqueue path uses server defaults (max_ideas=3, timeframe=all).
  static Future<Map<String, dynamic>> fetchAllowedTradeIdeasWithDiagnostics(
    String ticker, {
    int maxIdeas = 3,
    String timeframe = 'all',  // 'all', 'thisWeek', 'thisMonth', 'thisYear'
    int? minDte,
    int? maxDte,
  }) async {
    final symbol = ticker.toUpperCase();

    // 1) Supabase: cache table then enqueue-and-wait (same pattern as cockpit)
    if (ComputeQueueService.isAvailable) {
      final cached = await ComputeQueueService.getCachedResultFromTable(
        symbol: symbol,
        taskType: ComputeTaskType.trade_ideas,
      );
      if (cached != null) {
        try {
          return _parseTradeIdeasResponse(cached.result);
        } catch (e) {
          // Fall through to API on parse error
        }
      }
      final jobResult = await ComputeQueueService.getCachedOrEnqueue(
        symbol: symbol,
        taskType: ComputeTaskType.trade_ideas,
      );
      if (jobResult.ok && jobResult.result != null) {
        try {
          return _parseTradeIdeasResponse(jobResult.result!);
        } catch (e) {
          // Fall through to API on parse error
        }
      }
    }

    // 2) Fallback: direct API (requires reachable apiBaseUrl)
    try {
      final params = <String, String>{
        'ticker': ticker,
        'max_ideas': maxIdeas.toString(),
        'timeframe': timeframe,
      };
      if (minDte != null) params['min_dte'] = minDte.toString();
      if (maxDte != null) params['max_dte'] = maxDte.toString();

      final uri = Uri.parse('$apiBaseUrl/trade-ideas/allowed')
          .replace(queryParameters: params);
      final response = await http.get(
        uri,
        headers: {
          'x-api-key': apiSecretKey,
          'Content-Type': 'application/json',
        },
      );

      if (response.statusCode == 200) {
        final dynamic decoded = json.decode(response.body);
        return _parseTradeIdeasResponse(decoded);
      }
      throw Exception('Failed to load trade ideas: ${response.statusCode} - ${response.body}');
    } catch (e) {
      throw Exception('Error fetching trade ideas: $e');
    }
  }
  
  /// Legacy method for backward compatibility
  static Future<List<TradeIdea>> fetchAllowedTradeIdeas(
    String ticker, {
    int maxIdeas = 3,
  }) async {
    final result = await fetchAllowedTradeIdeasWithDiagnostics(ticker, maxIdeas: maxIdeas);
    if (result.containsKey('ideas')) return result['ideas'] as List<TradeIdea>;
    if (result.containsKey('unlocked')) return result['unlocked'] as List<TradeIdea>;
    return <TradeIdea>[];
  }
}
