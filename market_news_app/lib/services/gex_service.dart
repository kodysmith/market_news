import '../models/gex_data.dart';
import 'compute_queue_service.dart';

/// Wrapper for GEX calculation plus cache timestamp (for staleness checks).
class GexResult {
  final GexCalculation? calculation;
  final DateTime? updatedAt;

  GexResult({this.calculation, this.updatedAt});
}

/// GEX service: reads only from Supabase compute_result_cache (precomputed every 5 min). No API.
class GexService {
  /// Calculate GEX for a ticker plus cache updated_at. Tries precompute cache first, then enqueues and waits if missing.
  /// Returns calculation and updatedAt for staleness; updatedAt is null when data came from enqueue-and-wait.
  static Future<GexResult> calculateGex(String ticker) async {
    final t = ticker.toUpperCase();
    if (!ComputeQueueService.isAvailable) return GexResult(calculation: null, updatedAt: null);
    final cached = await ComputeQueueService.getCachedResultFromTable(
      symbol: t,
      taskType: ComputeTaskType.gex,
    );
    if (cached != null) {
      try {
        final calculation = GexCalculation.fromJson(cached.result);
        return GexResult(calculation: calculation, updatedAt: cached.updatedAt);
      } catch (e) {
        print('❌ Error parsing GEX from Supabase: $e');
        return GexResult(calculation: null, updatedAt: cached.updatedAt);
      }
    }
    final jobResult = await ComputeQueueService.getCachedOrEnqueue(
      symbol: t,
      taskType: ComputeTaskType.gex,
    );
    if (jobResult.ok && jobResult.result != null) {
      try {
        final calculation = GexCalculation.fromJson(jobResult.result!);
        return GexResult(calculation: calculation, updatedAt: null);
      } catch (e) {
        print('❌ Error parsing GEX from enqueue: $e');
        return GexResult(calculation: null, updatedAt: null);
      }
    }
    return GexResult(calculation: null, updatedAt: null);
  }

  /// Get batch GEX summary from cache (all core symbols). No API.
  static Future<GexSummary?> getGexSummary({List<String>? tickers}) async {
    if (!ComputeQueueService.isAvailable) return null;
    final rows = await ComputeQueueService.getAllCachedResultsForTask(ComputeTaskType.gex);
    if (rows.isEmpty) return null;
    final tickerSummaries = <GexTickerSummary>[];
    for (final row in rows) {
      final symbol = row['symbol'] as String? ?? '';
      final result = row['result'] as Map<String, dynamic>?;
      if (result == null) continue;
      final metrics = result['metrics'] as Map<String, dynamic>? ?? {};
      tickerSummaries.add(GexTickerSummary(
        ticker: symbol,
        spot: (result['spot_price'] as num?)?.toDouble() ?? (metrics['spot_price'] as num?)?.toDouble() ?? 0.0,
        flipLine: (metrics['flip_line'] as num?)?.toDouble(),
        putWall: (metrics['put_wall'] as num?)?.toDouble() ?? 0.0,
        callWall: (metrics['call_wall'] as num?)?.toDouble() ?? 0.0,
        totalGex: (metrics['total_gex'] as num?)?.toDouble() ?? 0.0,
        regime: metrics['regime'] as String? ?? 'unknown',
      ));
    }
    if (tickerSummaries.isEmpty) return null;
    return GexSummary(
      tickers: tickerSummaries,
      errors: null,
      timestamp: DateTime.now().millisecondsSinceEpoch / 1000.0,
    );
  }

  /// Get list of supported GEX tickers (core symbols from Supabase precompute).
  static Future<List<String>> getGexTickers() async {
    if (ComputeQueueService.isAvailable) {
      return List.from(ComputeQueueService.coreSymbols);
    }
    return List.from(ComputeQueueService.coreSymbols);
  }

  /// Max pain: not available in Supabase-only mode (no API). Returns null.
  static Future<MaxPainResult?> getMaxPain(
    String ticker, {
    int? dte,
    String? expiration,
  }) async {
    return null;
  }

  /// Price comparison: not available in Supabase-only mode (no API). Returns null.
  static Future<Map<String, dynamic>?> getPriceComparison(String ticker) async {
    return null;
  }
}
