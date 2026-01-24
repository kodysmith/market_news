import 'dart:convert';
import 'package:http/http.dart' as http;
import '../models/trade_idea.dart';
import '../main.dart' show apiBaseUrl, apiSecretKey;

/// Service for fetching allowed trade ideas
class TradeIdeasService {
  /// Fetch allowed trade ideas for a ticker
  /// Returns a map with 'ideas' (List of TradeIdea) or 'ideasByTimeframe' (Map) and 'diagnostics' (Map)
  static Future<Map<String, dynamic>> fetchAllowedTradeIdeasWithDiagnostics(
    String ticker, {
    int maxIdeas = 3,
    String timeframe = 'all',  // 'all', 'thisWeek', 'thisMonth', 'thisYear'
    int? minDte,
    int? maxDte,
  }) async {
    try {
      final params = <String, String>{
        'ticker': ticker,
        'max_ideas': maxIdeas.toString(),
        'timeframe': timeframe,
      };
      
      if (minDte != null) {
        params['min_dte'] = minDte.toString();
      }
      if (maxDte != null) {
        params['max_dte'] = maxDte.toString();
      }
      
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
        
        // Handle different response formats
        if (decoded is Map<String, dynamic>) {
          // New format (object with ideas/ideasByTimeframe + diagnostics)
          if (decoded.containsKey('ideasByTimeframe')) {
            // Multiple timeframes
            final ideasByTimeframe = decoded['ideasByTimeframe'] as Map<String, dynamic>;
            final ideasByTimeframeMap = <String, List<TradeIdea>>{};
            
            for (final entry in ideasByTimeframe.entries) {
              ideasByTimeframeMap[entry.key] = (entry.value as List)
                  .map((json) => TradeIdea.fromJson(json as Map<String, dynamic>))
                  .toList();
            }
            
            return {
              'ideasByTimeframe': ideasByTimeframeMap,
              'diagnostics': decoded['diagnostics'] ?? <String, dynamic>{},
            };
          } else if (decoded.containsKey('ideas')) {
            // Single timeframe
            return {
              'ideas': (decoded['ideas'] as List)
                  .map((json) => TradeIdea.fromJson(json as Map<String, dynamic>))
                  .toList(),
              'diagnostics': decoded['diagnostics'] ?? <String, dynamic>{},
            };
          } else {
            // Map but no 'ideas' key - treat as empty
            return {
              'ideas': <TradeIdea>[],
              'diagnostics': decoded,
            };
          }
        } else if (decoded is List) {
          // Old format (array) - for backward compatibility
          return {
            'ideas': decoded
                .map((json) => TradeIdea.fromJson(json as Map<String, dynamic>))
                .toList(),
            'diagnostics': <String, dynamic>{},
          };
        } else {
          throw Exception('Unexpected response format: ${decoded.runtimeType}');
        }
      } else {
        throw Exception('Failed to load trade ideas: ${response.statusCode} - ${response.body}');
      }
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
    return result['ideas'] as List<TradeIdea>;
  }
}
