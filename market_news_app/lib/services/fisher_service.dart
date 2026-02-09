import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:supabase_flutter/supabase_flutter.dart';
import '../main.dart' show apiBaseUrl;
import '../models/fisher_snapshot.dart';

/// Service for Fisher score: reads from Supabase when configured, else localhost API.
class FisherService {
  static bool get _useSupabase {
    try {
      Supabase.instance;
      return true;
    } catch (_) {
      return false;
    }
  }

  /// Get latest Fisher score snapshot for a ticker (Supabase first, then API).
  static Future<FisherSnapshot?> getSnapshot(String ticker) async {
    final t = ticker.toUpperCase();
    if (_useSupabase) {
      try {
        final client = Supabase.instance.client;
        final company = await client.from('fisher_company').select('id, ticker').eq('ticker', t).maybeSingle();
        if (company == null) return null;
        final companyId = company['id'] as String?;
        final snapshot = await client.from('fisher_score_snapshot').select().eq('company_id', companyId!).order('snapshot_at', ascending: false).limit(1).maybeSingle();
        if (snapshot == null) return null;
        final snapshotAt = snapshot['snapshot_at'];
        final map = <String, dynamic>{
          'company_id': companyId,
          'ticker': company['ticker'],
          'snapshot_at': snapshotAt is DateTime ? snapshotAt.toIso8601String() : snapshotAt?.toString(),
          'total_score': snapshot['total_score'],
          'version': snapshot['version'],
          'points': snapshot['points'] ?? {},
          'category_scores': snapshot['category_scores'] ?? {},
        };
        return FisherSnapshot.fromJson(map);
      } catch (e) {
        print('[FisherService] getSnapshot Supabase: $e');
      }
    }
    try {
      final url = '$apiBaseUrl/fisher/snapshot?ticker=$t';
      final response = await http.get(Uri.parse(url));
      if (response.statusCode == 200) {
        final data = json.decode(response.body) as Map<String, dynamic>?;
        if (data == null || data['error'] != null) return null;
        return FisherSnapshot.fromJson(data);
      }
      return null;
    } catch (e) {
      print('[FisherService] getSnapshot: $e');
      return null;
    }
  }

  /// Get delta (what changed since last quarter) for a ticker.
  static Future<FisherDelta?> getDelta(String ticker) async {
    final t = ticker.toUpperCase();
    if (_useSupabase) {
      try {
        final client = Supabase.instance.client;
        final company = await client.from('fisher_company').select('id').eq('ticker', t).maybeSingle();
        if (company == null) return null;
        final companyId = company['id'] as String;
        final rows = await client.from('fisher_score_snapshot').select('snapshot_at, points').eq('company_id', companyId).order('snapshot_at', ascending: false).limit(2);
        if (rows.isEmpty) return FisherDelta(ticker: t, pointDeltas: {}, message: 'No snapshots');
        if (rows.length < 2) {
          final r = rows.first;
          final at = r['snapshot_at'];
          return FisherDelta(
            ticker: t,
            currentSnapshotAt: at is DateTime ? at.toIso8601String() : at?.toString(),
            previousSnapshotAt: null,
            pointDeltas: {},
            message: 'Only one snapshot; no delta available.',
          );
        }
        final curr = rows[0];
        final prev = rows[1];
        final currPts = curr['points'] as Map<String, dynamic>? ?? {};
        final prevPts = prev['points'] as Map<String, dynamic>? ?? {};
        final pointDeltas = <String, double>{};
        for (final pid in {...currPts.keys, ...prevPts.keys}) {
          final cScore = _pointScore(currPts[pid]);
          final pScore = _pointScore(prevPts[pid]);
          if (cScore != null && pScore != null) {
            pointDeltas[pid.toString()] = (cScore - pScore).roundToDouble();
          } else if (cScore != null) {
            pointDeltas[pid.toString()] = cScore;
          } else if (pScore != null) {
            pointDeltas[pid.toString()] = -pScore;
          }
        }
        final currAt = curr['snapshot_at'];
        final prevAt = prev['snapshot_at'];
        return FisherDelta(
          ticker: t,
          currentSnapshotAt: currAt is DateTime ? currAt.toIso8601String() : currAt?.toString(),
          previousSnapshotAt: prevAt is DateTime ? prevAt.toIso8601String() : prevAt?.toString(),
          pointDeltas: pointDeltas,
        );
      } catch (e) {
        print('[FisherService] getDelta Supabase: $e');
      }
    }
    try {
      final url = '$apiBaseUrl/fisher/delta?ticker=$t';
      final response = await http.get(Uri.parse(url));
      if (response.statusCode == 200) {
        final data = json.decode(response.body) as Map<String, dynamic>?;
        if (data == null || data['error'] != null) return null;
        return FisherDelta.fromJson(data);
      }
      return null;
    } catch (e) {
      print('[FisherService] getDelta: $e');
      return null;
    }
  }

  static double? _pointScore(dynamic pt) {
    if (pt == null || pt is! Map) return null;
    final v = pt['score'];
    if (v == null) return null;
    if (v is num) return v.toDouble();
    if (v is String) return double.tryParse(v);
    return null;
  }

  /// Get evidence for a given point.
  static Future<FisherEvidence?> getEvidence(String ticker, String pointId) async {
    final t = ticker.toUpperCase();
    if (_useSupabase) {
      try {
        final client = Supabase.instance.client;
        final company = await client.from('fisher_company').select('id').eq('ticker', t).maybeSingle();
        if (company == null) return null;
        final companyId = company['id'] as String;
        final row = await client.from('fisher_score_snapshot').select('points').eq('company_id', companyId).order('snapshot_at', ascending: false).limit(1).maybeSingle();
        if (row == null) return null;
        final points = row['points'] as Map<String, dynamic>? ?? {};
        final pt = points[pointId] ?? points[int.tryParse(pointId)];
        if (pt == null || pt is! Map) return null;
        final evidence = pt['evidence'];
        double? confidence;
        final c = pt['confidence'];
        if (c is num) confidence = c.toDouble();
        else if (c is String) confidence = double.tryParse(c);
        return FisherEvidence(
          ticker: t,
          pointId: pointId,
          score: _pointScore(pt),
          confidence: confidence,
          evidence: evidence is List ? evidence : [],
          featureValues: pt['feature_values'] is Map<String, dynamic> ? Map<String, dynamic>.from(pt['feature_values'] as Map) : {},
        );
      } catch (e) {
        print('[FisherService] getEvidence Supabase: $e');
      }
    }
    try {
      final url = '$apiBaseUrl/fisher/evidence?ticker=$t&point_id=$pointId';
      final response = await http.get(Uri.parse(url));
      if (response.statusCode == 200) {
        final data = json.decode(response.body) as Map<String, dynamic>?;
        if (data == null || data['error'] != null) return null;
        return FisherEvidence.fromJson(data);
      }
      return null;
    } catch (e) {
      print('[FisherService] getEvidence: $e');
      return null;
    }
  }

  /// Get Fisher universe (tickers from fisher_company when Supabase; else API).
  static Future<FisherUniverse?> getUniverse() async {
    if (_useSupabase) {
      try {
        final rows = await Supabase.instance.client.from('fisher_company').select('ticker').order('ticker');
        final tickers = (rows as List).map((r) => (r as Map)['ticker']?.toString() ?? '').where((s) => s.isNotEmpty).toList();
        return FisherUniverse(tickers: tickers);
      } catch (e) {
        print('[FisherService] getUniverse Supabase: $e');
      }
    }
    try {
      final url = '$apiBaseUrl/fisher/universe';
      final response = await http.get(Uri.parse(url));
      if (response.statusCode == 200) {
        final data = json.decode(response.body) as Map<String, dynamic>?;
        if (data == null || data['error'] != null) return null;
        return FisherUniverse.fromJson(data);
      }
      return null;
    } catch (e) {
      print('[FisherService] getUniverse: $e');
      return null;
    }
  }

  /// Get high growth + profitable companies (for Fisher & Valuation screen).
  /// Uses Supabase first (fisher_company + latest fisher_score_snapshot); falls back to API when Supabase empty or unavailable.
  static Future<({List<FisherGrowthProfitableItem> companies, String? error})> getGrowthProfitableWithError({
    double minGrowth = 6.0,
    double minFinancials = 6.0,
    int limit = 100,
  }) async {
    if (_useSupabase) {
      final fromSupabase = await _getGrowthProfitableFromSupabase(limit: limit);
      if (fromSupabase != null && fromSupabase.isNotEmpty) {
        return (companies: fromSupabase, error: null);
      }
    }
    try {
      final url = '$apiBaseUrl/fisher/growth-profitable?min_growth=$minGrowth&min_financials=$minFinancials&limit=$limit';
      final response = await http.get(Uri.parse(url));
      Map<String, dynamic>? data;
      try {
        data = json.decode(response.body) as Map<String, dynamic>?;
      } catch (_) {
        data = null;
      }
      final errorMsg = data?['error']?.toString();

      if (response.statusCode == 200 && errorMsg == null && data != null) {
        final list = data['companies'] as List<dynamic>? ?? [];
        final companies = list.map((e) => FisherGrowthProfitableItem.fromJson(Map<String, dynamic>.from(e as Map))).toList();
        return (companies: companies, error: null);
      }

      if (errorMsg != null && errorMsg.isNotEmpty) {
        return (companies: <FisherGrowthProfitableItem>[], error: errorMsg);
      }
      if (response.statusCode >= 400) {
        return (companies: <FisherGrowthProfitableItem>[], error: 'Fisher API error (${response.statusCode}). Is the server running?');
      }
      return (companies: <FisherGrowthProfitableItem>[], error: null);
    } catch (e) {
      print('[FisherService] getGrowthProfitable: $e');
      if (_useSupabase) {
        return (companies: <FisherGrowthProfitableItem>[], error: 'No Fisher companies in Supabase. Run Fisher pipeline and scan on the server, or connect to API.');
      }
      return (companies: <FisherGrowthProfitableItem>[], error: 'Can\'t reach Fisher API. Check API_BASE_URL and that the server is running.');
    }
  }

  /// Build growth list from Supabase: fisher_company + latest fisher_score_snapshot per company. Returns null on error.
  static Future<List<FisherGrowthProfitableItem>?> _getGrowthProfitableFromSupabase({int limit = 100}) async {
    try {
      final client = Supabase.instance.client;
      final companies = await client.from('fisher_company').select('id, ticker, name, sector').order('ticker');
      if (companies.isEmpty) return <FisherGrowthProfitableItem>[];
      final ids = companies.map<String>((c) => c['id'] as String).toList();
      final snapshots = await client.from('fisher_score_snapshot').select('company_id, total_score, snapshot_at').inFilter('company_id', ids).order('snapshot_at', ascending: false);
      final latestByCompany = <String, Map<String, dynamic>>{};
      for (final row in snapshots as List) {
        final map = Map<String, dynamic>.from(row as Map);
        final cid = map['company_id'] as String?;
        if (cid != null && !latestByCompany.containsKey(cid)) {
          latestByCompany[cid] = map;
        }
      }
      final list = <FisherGrowthProfitableItem>[];
      for (final c in companies as List) {
        final m = Map<String, dynamic>.from(c as Map);
        final id = m['id'] as String?;
        final ticker = m['ticker'] as String? ?? '';
        final name = m['name'] as String? ?? ticker;
        final sector = m['sector'] as String? ?? '';
        final snap = id != null ? latestByCompany[id] : null;
        final totalScore = snap != null ? _snapshotScore(snap['total_score']) : null;
        final snapshotAt = snap?['snapshot_at']?.toString();
        list.add(FisherGrowthProfitableItem(
          ticker: ticker,
          name: name,
          sector: sector,
          growth: null,
          financials: null,
          totalScore: totalScore,
          snapshotAt: snapshotAt,
        ));
      }
      list.sort((a, b) => (b.totalScore ?? 0).compareTo(a.totalScore ?? 0));
      return list.take(limit).toList();
    } catch (e) {
      print('[FisherService] _getGrowthProfitableFromSupabase: $e');
      return null;
    }
  }

  static double? _snapshotScore(dynamic v) {
    if (v == null) return null;
    if (v is num) return v.toDouble();
    if (v is String) return double.tryParse(v);
    return null;
  }

  /// Get high growth + profitable companies (returns empty list on any error; use getGrowthProfitableWithError for error message).
  static Future<List<FisherGrowthProfitableItem>> getGrowthProfitable({
    double minGrowth = 6.0,
    double minFinancials = 6.0,
    int limit = 100,
  }) async {
    final result = await getGrowthProfitableWithError(minGrowth: minGrowth, minFinancials: minFinancials, limit: limit);
    return result.companies;
  }
}
