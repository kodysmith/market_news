import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:market_news_app/models/vix_data.dart';
import 'package:market_news_app/models/economic_event.dart';
import '../main.dart' show apiBaseUrl;
import '../utils/api_cache.dart';

class FmpApiService {
  final String _apiKey;
  
  // Initialize API cache with 10-second TTL
  final _cache = getCache(defaultTTL: 10);

  FmpApiService(this._apiKey);
  // static const String _baseUrl = 'https://financialmodelingprep.com/api/v3';

  Future<List<VixData>> fetchVixData({http.Client? client}) async {
    client ??= http.Client();
    final url = '${apiBaseUrl}/historical-price-full/VIX?apikey=$_apiKey';
    final params = {'apikey': _apiKey};
    
    // Generate cache key
    final cacheKey = _cache.generateKey(url, params, _apiKey);
    
    // Check cache first
    final cachedVixData = _cache.get<List<VixData>>(cacheKey);
    if (cachedVixData != null) {
      return cachedVixData;
    }
    
    final response = await client.get(Uri.parse(url));

    if (response.statusCode == 200) {
      if (response.body.trim() == '{}') {
        return []; // Return empty list for empty JSON object
      }

      final data = json.decode(response.body);
      List<VixData> vixDataList;

      if (data is Map<String, dynamic> && data.containsKey('historical') && data['historical'] is List) {
        final List<dynamic> historical = data['historical'];
        vixDataList = historical.map((json) => VixData.fromJson(json)).toList();
      } else if (data is List) {
        vixDataList = data.map((json) => VixData.fromJson(json)).toList();
      } else {
        print("Unexpected API response format: ${response.body}");
        throw Exception('VIX data not in expected format');
      }
      
      // Cache the result
      _cache.set(cacheKey, vixDataList, ttl: 10);
      return vixDataList;
    } else {
      throw Exception('Failed to load VIX data');
    }
  }

  Future<List<EconomicEvent>> fetchEconomicCalendar({http.Client? client}) async {
    client ??= http.Client();
    // Fetching events for the next 7 days
    final DateTime now = DateTime.now();
    final DateTime sevenDaysLater = now.add(const Duration(days: 7));
    final String fromDate = "${now.year}-${now.month.toString().padLeft(2, '0')}-${now.day.toString().padLeft(2, '0')}";
    final String toDate = "${sevenDaysLater.year}-${sevenDaysLater.month.toString().padLeft(2, '0')}-${sevenDaysLater.day.toString().padLeft(2, '0')}";

    final url = '${apiBaseUrl}/economic-calendar?from=$fromDate&to=$toDate&apikey=$_apiKey';
    final params = {
      'from': fromDate,
      'to': toDate,
      'apikey': _apiKey,
    };
    
    // Generate cache key
    final cacheKey = _cache.generateKey('${apiBaseUrl}/economic-calendar', params, _apiKey);
    
    // Check cache first
    final cachedEvents = _cache.get<List<EconomicEvent>>(cacheKey);
    if (cachedEvents != null) {
      return cachedEvents;
    }

    final response = await client.get(Uri.parse(url));

    if (response.statusCode == 200) {
      final List<dynamic> data = json.decode(response.body);
      final events = data.map((json) => EconomicEvent.fromJson(json)).toList();
      
      // Cache the result
      _cache.set(cacheKey, events, ttl: 10);
      return events;
    } else if (response.statusCode == 403) {
      print("API Response (403): ${response.body}");
      throw Exception('Failed to load economic calendar: Access Denied (403)');
    } else {
      throw Exception('Failed to load economic calendar');
    }
  }
}
