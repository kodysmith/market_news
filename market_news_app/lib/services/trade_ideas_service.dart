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

      // #region agent log
      final logData = {
        'sessionId': 'debug-session',
        'runId': 'run1',
        'hypothesisId': 'D',
        'location': 'trade_ideas_service.dart:34',
        'message': 'HTTP GET request',
        'data': {'url': uri.toString(), 'ticker': ticker, 'timestamp': DateTime.now().millisecondsSinceEpoch},
        'timestamp': DateTime.now().millisecondsSinceEpoch
      };
      http.post(
        Uri.parse('http://127.0.0.1:7242/ingest/f2e5acba-93e2-4270-a633-fee604b9e81c'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode(logData),
      ).catchError((_) {});
      // #endregion

      final response = await http.get(
        uri,
        headers: {
          'x-api-key': apiSecretKey,
          'Content-Type': 'application/json',
        },
      );
      
      // #region agent log
      final logData2 = {
        'sessionId': 'debug-session',
        'runId': 'run1',
        'hypothesisId': 'D',
        'location': 'trade_ideas_service.dart:50',
        'message': 'HTTP response received',
        'data': {'statusCode': response.statusCode, 'bodyLength': response.body.length, 'timestamp': DateTime.now().millisecondsSinceEpoch},
        'timestamp': DateTime.now().millisecondsSinceEpoch
      };
      http.post(
        Uri.parse('http://127.0.0.1:7242/ingest/f2e5acba-93e2-4270-a633-fee604b9e81c'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode(logData2),
      ).catchError((_) {});
      // #endregion

      if (response.statusCode == 200) {
        final dynamic decoded = json.decode(response.body);
        
        // Handle different response formats
        if (decoded is Map<String, dynamic>) {
          // New format with unlocked/preview structure
          final result = <String, dynamic>{
            'diagnostics': decoded['diagnostics'] ?? <String, dynamic>{},
          };
          
          // Parse unlocked ideas (can be list or map by timeframe)
          if (decoded.containsKey('unlocked')) {
            final unlockedData = decoded['unlocked'];
            if (unlockedData is List) {
              result['unlocked'] = (unlockedData as List)
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
          
          // Parse preview ideas
          if (decoded.containsKey('preview')) {
            result['preview'] = (decoded['preview'] as List)
                .map((json) => TradeIdea.fromJson(json as Map<String, dynamic>))
                .toList();
          }
          
          // Parse next window
          if (decoded.containsKey('nextWindow')) {
            result['nextWindow'] = decoded['nextWindow'] as Map<String, dynamic>;
          }
          
          // Parse current phase
          if (decoded.containsKey('currentPhase')) {
            result['currentPhase'] = decoded['currentPhase'] as String;
          }
          
          // Backward compatibility: also parse old format if present
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
        } else if (decoded is List) {
          // Old format (array) - for backward compatibility
          return {
            'unlocked': decoded
                .map((json) => TradeIdea.fromJson(json as Map<String, dynamic>))
                .toList(),
            'preview': <TradeIdea>[],
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
