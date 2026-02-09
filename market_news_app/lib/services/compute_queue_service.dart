import 'package:supabase_flutter/supabase_flutter.dart';

/// Task types supported by the compute job queue (must match server schema).
enum ComputeTaskType {
  gex,
  valuation,
  cockpit,
  probability,
  trade_ideas,
}

/// Result of waiting for a compute job.
class ComputeJobResult {
  final bool ok;
  final Map<String, dynamic>? result;
  final String? error;

  ComputeJobResult({required this.ok, this.result, this.error});
}

/// Enqueue on-demand compute jobs and wait for results (table-based queue in Supabase).
/// Precomputed GEX/cockpit for core symbols live in compute_result_cache (filled every 5 min by cron).
class ComputeQueueService {
  static const String _table = 'compute_job_queue';
  static const String _cacheTable = 'compute_result_cache';
  static const Duration _pollInterval = Duration(seconds: 2);
  static const Duration _maxWait = Duration(minutes: 5);

  /// Core symbols for GEX/cockpit (SPX, XSP, SPY, NDX). Precompute script and app use this list.
  static const List<String> coreSymbols = ['SPX', 'XSP', 'SPY', 'NDX'];

  static String _taskTypeString(ComputeTaskType type) {
    switch (type) {
      case ComputeTaskType.gex:
        return 'gex';
      case ComputeTaskType.valuation:
        return 'valuation';
      case ComputeTaskType.cockpit:
        return 'cockpit';
      case ComputeTaskType.probability:
        return 'probability';
      case ComputeTaskType.trade_ideas:
        return 'trade_ideas';
    }
  }

  /// Returns true if Supabase is initialized and the queue can be used.
  static bool get isAvailable {
    try {
      Supabase.instance;
      return true;
    } catch (_) {
      return false;
    }
  }

  /// Enqueue a job and wait for the result (polling). Shows "give us a moment" by returning
  /// a future that completes when the server has processed the job.
  /// [symbol] e.g. 'AAPL', [taskType] e.g. ComputeTaskType.gex.
  /// Returns [ComputeJobResult] with result or error; [ok] is false if failed or Supabase unavailable.
  static Future<ComputeJobResult> enqueueAndWait({
    required String symbol,
    required ComputeTaskType taskType,
  }) async {
    if (!isAvailable) {
      return ComputeJobResult(ok: false, error: 'Supabase not configured (SUPABASE_URL, SUPABASE_ANON_KEY)');
    }
    final client = Supabase.instance.client;
    final taskTypeStr = _taskTypeString(taskType);
    try {
      final insert = await client.from(_table).insert({
        'symbol': symbol.toUpperCase(),
        'task_type': taskTypeStr,
        'status': 'pending',
      }).select('id').single();
      final jobId = insert['id'] as String;
      final deadline = DateTime.now().add(_maxWait);
      while (DateTime.now().isBefore(deadline)) {
        await Future<void>.delayed(_pollInterval);
        final row = await client.from(_table).select('status, result, error_text').eq('id', jobId).single();
        final status = row['status'] as String?;
        if (status == 'done') {
          final result = row['result'];
          return ComputeJobResult(
            ok: true,
            result: result is Map<String, dynamic> ? result : (result != null ? {'data': result} : null),
          );
        }
        if (status == 'failed') {
          return ComputeJobResult(ok: false, error: row['error_text'] as String? ?? 'Job failed');
        }
      }
      return ComputeJobResult(ok: false, error: 'Timed out waiting for result');
    } catch (e) {
      return ComputeJobResult(ok: false, error: e.toString());
    }
  }

  /// Enqueue a job only; returns job id. Caller can poll or subscribe for result.
  static Future<String?> enqueue({required String symbol, required ComputeTaskType taskType}) async {
    if (!isAvailable) return null;
    final client = Supabase.instance.client;
    final taskTypeStr = _taskTypeString(taskType);
    try {
      final insert = await client.from(_table).insert({
        'symbol': symbol.toUpperCase(),
        'task_type': taskTypeStr,
        'status': 'pending',
      }).select('id').single();
      return insert['id'] as String?;
    } catch (_) {
      return null;
    }
  }

  /// Poll for job result by id. Returns null if still pending or not found.
  static Future<ComputeJobResult?> getJobResult(String jobId) async {
    if (!isAvailable) return null;
    try {
      final row = await Supabase.instance.client.from(_table).select('status, result, error_text').eq('id', jobId).single();
      final status = row['status'] as String?;
      if (status == 'done') {
        final result = row['result'];
        return ComputeJobResult(
          ok: true,
          result: result is Map<String, dynamic> ? result : (result != null ? {'data': result} : null),
        );
      }
      if (status == 'failed') {
        return ComputeJobResult(ok: false, error: row['error_text'] as String? ?? 'Job failed');
      }
      return null;
    } catch (_) {
      return null;
    }
  }

  /// Get latest completed result from Supabase for (symbol, taskType). Returns null if none.
  static Future<Map<String, dynamic>?> getCachedResult({
    required String symbol,
    required ComputeTaskType taskType,
  }) async {
    if (!isAvailable) return null;
    try {
      final taskTypeStr = _taskTypeString(taskType);
      final rows = await Supabase.instance.client
          .from(_table)
          .select('result')
          .eq('symbol', symbol.toUpperCase())
          .eq('task_type', taskTypeStr)
          .eq('status', 'done')
          .order('completed_at', ascending: false)
          .limit(1);
      if (rows.isEmpty) return null;
      final result = rows.first['result'];
      if (result is Map<String, dynamic>) return result;
      if (result != null) return {'data': result};
      return null;
    } catch (_) {
      return null;
    }
  }

  /// Get result from Supabase cache or enqueue and wait. Use this so all data comes from Supabase.
  static Future<ComputeJobResult> getCachedOrEnqueue({
    required String symbol,
    required ComputeTaskType taskType,
  }) async {
    final cached = await getCachedResult(symbol: symbol, taskType: taskType);
    if (cached != null) return ComputeJobResult(ok: true, result: cached);
    return enqueueAndWait(symbol: symbol, taskType: taskType);
  }

  /// Get result from precompute cache table (compute_result_cache). One row per (symbol, task_type).
  /// Use for GEX/cockpit when app reads only from Supabase (no API, no enqueue).
  static Future<Map<String, dynamic>?> getCachedResultFromTable({
    required String symbol,
    required ComputeTaskType taskType,
  }) async {
    if (!isAvailable) return null;
    try {
      final taskTypeStr = _taskTypeString(taskType);
      final rows = await Supabase.instance.client
          .from(_cacheTable)
          .select('result')
          .eq('symbol', symbol.toUpperCase())
          .eq('task_type', taskTypeStr)
          .limit(1);
      if (rows.isEmpty) return null;
      final result = rows.first['result'];
      if (result == null) return null;
      if (result is Map<String, dynamic>) return result;
      if (result is Map) return Map<String, dynamic>.from(result);
      return null;
    } catch (_) {
      return null;
    }
  }

  /// Fetch all cache rows for a task type (e.g. gex). Returns list of (symbol, result) for core symbols.
  static Future<List<Map<String, dynamic>>> getAllCachedResultsForTask(ComputeTaskType taskType) async {
    if (!isAvailable) return [];
    try {
      final taskTypeStr = _taskTypeString(taskType);
      final rows = await Supabase.instance.client
          .from(_cacheTable)
          .select('symbol, result')
          .eq('task_type', taskTypeStr)
          .inFilter('symbol', coreSymbols);
      if (rows.isEmpty) return [];
      return rows.map((r) => r).toList();
    } catch (_) {
      return [];
    }
  }
}
