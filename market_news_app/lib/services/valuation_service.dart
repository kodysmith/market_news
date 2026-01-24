import 'dart:convert';
import 'package:http/http.dart' as http;
import '../main.dart' show apiBaseUrl;
import '../models/valuation_data.dart';

/// Service for interacting with the Intrinsic Value / Valuation API endpoints
class ValuationService {
  /// Calculate intrinsic value using all 6 methods
  static Future<ValuationResult?> calculateIntrinsicValue(String ticker, {bool store = true}) async {
    try {
      final url = '$apiBaseUrl/valuation/calculate?ticker=${ticker.toUpperCase()}&store=$store';
      print('[ValuationService] Calling: $url');
      
      final response = await http.get(Uri.parse(url));
      print('[ValuationService] Response status: ${response.statusCode}');
      
      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        if (data['error'] != null && data['valuations'] == null) {
          print('[ValuationService] Error in response: ${data['error']}');
          return null;
        }
        return ValuationResult.fromJson(data);
      } else {
        print('[ValuationService] Error: ${response.body}');
        return null;
      }
    } catch (e) {
      print('[ValuationService] Exception: $e');
      return null;
    }
  }

  /// Get historical intrinsic value data
  static Future<List<ValuationHistory>> getHistory(String ticker, {int days = 365}) async {
    try {
      final url = '$apiBaseUrl/valuation/history?ticker=${ticker.toUpperCase()}&days=$days';
      print('[ValuationService] Calling: $url');
      
      final response = await http.get(Uri.parse(url));
      
      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        final historyList = data['history'] as List<dynamic>? ?? [];
        return historyList.map((h) => ValuationHistory.fromJson(h)).toList();
      }
      return [];
    } catch (e) {
      print('[ValuationService] Exception in getHistory: $e');
      return [];
    }
  }

  /// Get current divergence breakdown by method
  static Future<DivergenceBreakdown?> getDivergence(String ticker) async {
    try {
      final url = '$apiBaseUrl/valuation/divergence?ticker=${ticker.toUpperCase()}';
      print('[ValuationService] Calling: $url');
      
      final response = await http.get(Uri.parse(url));
      
      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        return DivergenceBreakdown.fromJson(data);
      }
      return null;
    } catch (e) {
      print('[ValuationService] Exception in getDivergence: $e');
      return null;
    }
  }

  /// Batch scan multiple tickers
  static Future<List<ValuationScanResult>> scanTickers(List<String> tickers, {double threshold = 20}) async {
    try {
      final tickersParam = tickers.map((t) => t.toUpperCase()).join(',');
      final url = '$apiBaseUrl/valuation/scan?tickers=$tickersParam&threshold=$threshold';
      print('[ValuationService] Calling: $url');
      
      final response = await http.get(Uri.parse(url));
      
      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        final results = data['results'] as List<dynamic>? ?? [];
        return results.map((r) => ValuationScanResult.fromJson(r)).toList();
      }
      return [];
    } catch (e) {
      print('[ValuationService] Exception in scanTickers: $e');
      return [];
    }
  }

  /// Get watchlist with latest valuations
  static Future<List<WatchlistItem>> getWatchlist() async {
    try {
      final url = '$apiBaseUrl/valuation/watchlist';
      print('[ValuationService] Calling: $url');
      
      final response = await http.get(Uri.parse(url));
      
      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        final watchlist = data['watchlist'] as List<dynamic>? ?? [];
        return watchlist.map((w) => WatchlistItem.fromJson(w)).toList();
      }
      return [];
    } catch (e) {
      print('[ValuationService] Exception in getWatchlist: $e');
      return [];
    }
  }

  /// Add ticker to watchlist
  static Future<bool> addToWatchlist(String ticker, {double alertThreshold = 20}) async {
    try {
      final url = '$apiBaseUrl/valuation/watchlist';
      print('[ValuationService] POST to: $url');
      
      final response = await http.post(
        Uri.parse(url),
        headers: {'Content-Type': 'application/json'},
        body: json.encode({
          'ticker': ticker.toUpperCase(),
          'alert_threshold': alertThreshold,
        }),
      );
      
      return response.statusCode == 200;
    } catch (e) {
      print('[ValuationService] Exception in addToWatchlist: $e');
      return false;
    }
  }

  /// Remove ticker from watchlist
  static Future<bool> removeFromWatchlist(String ticker) async {
    try {
      final url = '$apiBaseUrl/valuation/watchlist';
      print('[ValuationService] DELETE to: $url');
      
      final response = await http.delete(
        Uri.parse(url),
        headers: {'Content-Type': 'application/json'},
        body: json.encode({'ticker': ticker.toUpperCase()}),
      );
      
      return response.statusCode == 200;
    } catch (e) {
      print('[ValuationService] Exception in removeFromWatchlist: $e');
      return false;
    }
  }

  /// Get recent valuation alerts
  static Future<List<ValuationAlert>> getAlerts({int limit = 50}) async {
    try {
      final url = '$apiBaseUrl/valuation/alerts?limit=$limit';
      print('[ValuationService] Calling: $url');
      
      final response = await http.get(Uri.parse(url));
      
      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        final alerts = data['alerts'] as List<dynamic>? ?? [];
        return alerts.map((a) => ValuationAlert.fromJson(a)).toList();
      }
      return [];
    } catch (e) {
      print('[ValuationService] Exception in getAlerts: $e');
      return [];
    }
  }

  /// Acknowledge an alert
  static Future<bool> acknowledgeAlert(int alertId) async {
    try {
      final url = '$apiBaseUrl/valuation/alerts/$alertId/acknowledge';
      print('[ValuationService] POST to: $url');
      
      final response = await http.post(Uri.parse(url));
      return response.statusCode == 200;
    } catch (e) {
      print('[ValuationService] Exception in acknowledgeAlert: $e');
      return false;
    }
  }

  /// Get undervalued stocks
  static Future<List<ValuationScanResult>> getUndervalued({double threshold = 20}) async {
    try {
      final url = '$apiBaseUrl/valuation/undervalued?threshold=$threshold';
      print('[ValuationService] Calling: $url');
      
      final response = await http.get(Uri.parse(url));
      
      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        final stocks = data['stocks'] as List<dynamic>? ?? [];
        return stocks.map((s) => ValuationScanResult.fromJson(s)).toList();
      }
      return [];
    } catch (e) {
      print('[ValuationService] Exception in getUndervalued: $e');
      return [];
    }
  }
}
