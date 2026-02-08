import 'dart:convert';
import 'package:http/http.dart' as http;
import '../main.dart' show apiBaseUrl;
import '../models/fisher_snapshot.dart';

/// Service for Fisher score API: snapshot, delta, evidence, universe.
class FisherService {
  /// Get latest Fisher score snapshot for a ticker.
  static Future<FisherSnapshot?> getSnapshot(String ticker) async {
    try {
      final url = '$apiBaseUrl/fisher/snapshot?ticker=${ticker.toUpperCase()}';
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
    try {
      final url = '$apiBaseUrl/fisher/delta?ticker=${ticker.toUpperCase()}';
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

  /// Get evidence for a given point.
  static Future<FisherEvidence?> getEvidence(String ticker, String pointId) async {
    try {
      final url = '$apiBaseUrl/fisher/evidence?ticker=${ticker.toUpperCase()}&point_id=$pointId';
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

  /// Get Fisher universe (S&P 500 tickers).
  static Future<FisherUniverse?> getUniverse() async {
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
  /// On success returns (companies, null). On API/connection error returns ([], errorMessage).
  static Future<({List<FisherGrowthProfitableItem> companies, String? error})> getGrowthProfitableWithError({
    double minGrowth = 6.0,
    double minFinancials = 6.0,
    int limit = 100,
  }) async {
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

      // API returned error (e.g. 503 = DB not configured)
      if (errorMsg != null && errorMsg.isNotEmpty) {
        return (companies: <FisherGrowthProfitableItem>[], error: errorMsg);
      }
      if (response.statusCode >= 400) {
        return (companies: <FisherGrowthProfitableItem>[], error: 'Fisher API error (${response.statusCode}). Is the server running?');
      }
      return (companies: <FisherGrowthProfitableItem>[], error: null);
    } catch (e) {
      print('[FisherService] getGrowthProfitable: $e');
      return (companies: <FisherGrowthProfitableItem>[], error: 'Can\'t reach Fisher API. Check API_BASE_URL and that the server is running.');
    }
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
