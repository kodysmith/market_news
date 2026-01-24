/// API Cache Utility for External Vendor APIs
/// 
/// Provides in-memory caching with TTL (Time To Live) for external API calls.
/// Reduces API rate limit issues and improves response times.
/// 
/// Usage:
///   final cache = APICache(defaultTTL: 10); // 10 second default TTL
///   
///   // Check cache first
///   final cachedData = cache.get<String>('cache_key');
///   if (cachedData != null) {
///     return cachedData;
///   }
///   
///   // Make API call
///   final data = await makeApiCall();
///   
///   // Cache the result
///   cache.set('cache_key', data, ttl: 10);
///   return data;

import 'dart:convert';
import 'dart:collection';

/// Cache entry storing data and expiration timestamp
class _CacheEntry<T> {
  final T data;
  final int expiration;

  _CacheEntry(this.data, this.expiration);

  bool get isExpired => DateTime.now().millisecondsSinceEpoch > expiration;
}

/// In-memory API cache with TTL support
class APICache {
  final int defaultTTL;
  final int cleanupInterval;
  
  final Map<String, _CacheEntry> _cache = {};
  int _lastCleanup = 0;

  APICache({this.defaultTTL = 10, this.cleanupInterval = 60});

  /// Generate a cache key from URL, parameters, and API key
  String generateKey(String url, Map<String, dynamic>? params, String? apiKey) {
    // Normalize URL (remove fragment)
    final uri = Uri.parse(url);
    final baseUrl = '${uri.scheme}://${uri.host}${uri.path}';
    
    // Merge URL query params with provided params
    final allParams = <String, String>{};
    
    // Add URL query params
    uri.queryParameters.forEach((key, value) {
      allParams[key] = value;
    });
    
    // Add provided params
    if (params != null) {
      params.forEach((key, value) {
        allParams[key] = value.toString();
      });
    }
    
    // Sort parameters for consistent keys
    final sortedKeys = allParams.keys.toList()..sort();
    final sortedParams = sortedKeys.map((key) => '$key=${allParams[key]}').join('&');
    
    // Create key components
    final keyParts = [baseUrl, sortedParams];
    
    // Hash API key if provided (simple hash for Dart)
    if (apiKey != null && apiKey.isNotEmpty) {
      final apiKeyHash = _hashString(apiKey).substring(0, 16);
      keyParts.add(apiKeyHash);
    }
    
    // Generate final key
    final keyString = keyParts.join('|');
    return _hashString(keyString);
  }

  /// Simple hash function for strings
  String _hashString(String input) {
    var bytes = utf8.encode(input);
    var hash = 0;
    for (var byte in bytes) {
      hash = ((hash << 5) - hash) + byte;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return hash.abs().toRadixString(16).padLeft(8, '0');
  }

  /// Get cached data if it exists and hasn't expired
  T? get<T>(String key) {
    // Periodic cleanup
    _cleanupIfNeeded();
    
    final entry = _cache[key];
    if (entry == null) {
      return null;
    }
    
    // Check if expired
    if (entry.isExpired) {
      _cache.remove(key);
      return null;
    }
    
    print('Cache HIT for key: ${key.substring(0, key.length > 16 ? 16 : key.length)}...');
    return entry.data as T;
  }

  /// Store data in cache with TTL
  void set<T>(String key, T value, {int? ttl}) {
    final ttlToUse = ttl ?? defaultTTL;
    final expiration = DateTime.now().millisecondsSinceEpoch + (ttlToUse * 1000);
    _cache[key] = _CacheEntry(value, expiration);
    print('Cache SET for key: ${key.substring(0, key.length > 16 ? 16 : key.length)}... (TTL: ${ttlToUse}s)');
  }

  /// Delete a cache entry
  void delete(String key) {
    if (_cache.containsKey(key)) {
      _cache.remove(key);
      print('Cache DELETE for key: ${key.substring(0, key.length > 16 ? 16 : key.length)}...');
    }
  }

  /// Clear all cache entries
  void clear() {
    _cache.clear();
    print('Cache cleared');
  }

  /// Remove expired entries if cleanup interval has passed
  void _cleanupIfNeeded() {
    final currentTime = DateTime.now().millisecondsSinceEpoch;
    if (currentTime - _lastCleanup < cleanupInterval * 1000) {
      return;
    }
    
    _lastCleanup = currentTime;
    final expiredKeys = <String>[];
    
    _cache.forEach((key, entry) {
      if (entry.isExpired) {
        expiredKeys.add(key);
      }
    });
    
    expiredKeys.forEach((key) => _cache.remove(key));
    
    if (expiredKeys.isNotEmpty) {
      print('Cleaned up ${expiredKeys.length} expired cache entries');
    }
  }

  /// Get cache statistics
  Map<String, dynamic> stats() {
    final currentTime = DateTime.now().millisecondsSinceEpoch;
    var activeCount = 0;
    var expiredCount = 0;
    
    _cache.forEach((key, entry) {
      if (entry.isExpired) {
        expiredCount++;
      } else {
        activeCount++;
      }
    });
    
    return {
      'totalEntries': _cache.length,
      'activeEntries': activeCount,
      'expiredEntries': expiredCount,
      'defaultTTL': defaultTTL,
    };
  }
}

/// Global cache instance (can be imported and used directly)
APICache? _globalCache;

APICache getCache({int defaultTTL = 10}) {
  _globalCache ??= APICache(defaultTTL: defaultTTL);
  return _globalCache!;
}
