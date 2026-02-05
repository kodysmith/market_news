import 'dart:convert';
import 'package:http/http.dart' as http;
import '../models/gex_data.dart';
import '../main.dart' show apiBaseUrl, apiSecretKey;

class GexService {
  /// Calculate GEX for a specific ticker
  static Future<GexCalculation?> calculateGex(String ticker) async {
    try {
      final uri = Uri.parse('$apiBaseUrl/gex/calculate').replace(
        queryParameters: {'ticker': ticker.toUpperCase()},
      );
      
      print('🌐 Calling GEX API: $uri');
      
      final response = await http.get(
        uri,
        headers: {
          'Content-Type': 'application/json',
          'x-api-key': apiSecretKey,
        },
      );
      
      print('📡 GEX API response status: ${response.statusCode}');

      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        print('✅ GEX calculation response received for $ticker');
        try {
          return GexCalculation.fromJson(data);
        } catch (parseError) {
          print('❌ Error parsing GEX response: $parseError');
          print('Response data: ${data.toString().substring(0, 200)}...');
          return null;
        }
      } else {
        print('❌ Error calculating GEX: ${response.statusCode} - ${response.body}');
        return null;
      }
    } catch (e, stackTrace) {
      print('❌ Error fetching GEX calculation: $e');
      print('Stack trace: $stackTrace');
      return null;
    }
  }

  /// Get batch GEX summary for all configured tickers or specific tickers
  static Future<GexSummary?> getGexSummary({List<String>? tickers}) async {
    try {
      final queryParams = <String, String>{};
      if (tickers != null && tickers.isNotEmpty) {
        queryParams['tickers'] = tickers.join(',');
      }

      final uri = Uri.parse('$apiBaseUrl/gex/summary').replace(
        queryParameters: queryParams,
      );

      final response = await http.get(
        uri,
        headers: {
          'Content-Type': 'application/json',
          'x-api-key': apiSecretKey,
        },
      );

      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        return GexSummary.fromJson(data);
      } else {
        print('Error getting GEX summary: ${response.statusCode} - ${response.body}');
        return null;
      }
    } catch (e) {
      print('Error fetching GEX summary: $e');
      return null;
    }
  }

  /// Get list of supported GEX tickers
  static Future<List<String>> getGexTickers() async {
    try {
      final uri = Uri.parse('$apiBaseUrl/gex/tickers');
      
      final response = await http.get(
        uri,
        headers: {
          'Content-Type': 'application/json',
          'x-api-key': apiSecretKey,
        },
      );

      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        final tickers = (data['tickers'] as List)
            .map((t) => t.toString().toUpperCase())
            .toList();
        return tickers;
      } else {
        print('Error getting GEX tickers: ${response.statusCode}');
        return ['SPY', 'SPX', 'QQQ']; // Default fallback
      }
    } catch (e) {
      print('Error fetching GEX tickers: $e');
      return ['SPY', 'SPX', 'QQQ']; // Default fallback
    }
  }

  /// Get max pain strike for a ticker and expiration (dte or specific expiration date).
  static Future<MaxPainResult?> getMaxPain(
    String ticker, {
    int? dte,
    String? expiration,
  }) async {
    try {
      final queryParams = <String, String>{'ticker': ticker.toUpperCase()};
      if (dte != null) queryParams['dte'] = dte.toString();
      if (expiration != null && expiration.isNotEmpty) queryParams['expiration'] = expiration;

      final uri = Uri.parse('$apiBaseUrl/gex/max-pain').replace(
        queryParameters: queryParams,
      );

      final response = await http.get(
        uri,
        headers: {
          'Content-Type': 'application/json',
          'x-api-key': apiSecretKey,
        },
      );

      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        return MaxPainResult.fromJson(data);
      } else {
        print('Error getting max pain: ${response.statusCode} - ${response.body}');
        return null;
      }
    } catch (e) {
      print('Error fetching max pain: $e');
      return null;
    }
  }

  /// Compare spot prices from different data sources
  static Future<Map<String, dynamic>?> getPriceComparison(String ticker) async {
    try {
      final uri = Uri.parse('$apiBaseUrl/gex/price-comparison').replace(
        queryParameters: {'ticker': ticker.toUpperCase()},
      );

      final response = await http.get(
        uri,
        headers: {
          'Content-Type': 'application/json',
          'x-api-key': apiSecretKey,
        },
      );

      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        return data;
      } else {
        print('Error getting price comparison: ${response.statusCode}');
        return null;
      }
    } catch (e) {
      print('Error fetching price comparison: $e');
      return null;
    }
  }
}
