import 'dart:convert';
import 'dart:io';
import 'package:flutter/services.dart';

class LocalAIService {
  static const MethodChannel _channel = MethodChannel('local_ai_service');
  
  /// Check if local AI is available on the device
  static Future<bool> isLocalAIAvailable() async {
    try {
      if (Platform.isAndroid) {
        final bool isAvailable = await _channel.invokeMethod('isLocalAIAvailable');
        return isAvailable;
      }
      return false;
    } catch (e) {
      print('Error checking local AI availability: $e');
      return false;
    }
  }
  
  /// Generate response using local AI
  static Future<String> generateResponse(String prompt) async {
    try {
      if (Platform.isAndroid) {
        final String response = await _channel.invokeMethod('generateResponse', {
          'prompt': prompt,
        });
        return response;
      }
      return 'Local AI not available on this platform';
    } catch (e) {
      print('Error generating local AI response: $e');
      return 'Error: $e';
    }
  }
  
  /// Analyze stock data using local AI
  static Future<Map<String, dynamic>> analyzeStockData(Map<String, dynamic> stockData) async {
    try {
      if (Platform.isAndroid) {
        final String response = await _channel.invokeMethod('analyzeStockData', {
          'stockData': stockData,
        });
        return json.decode(response);
      }
      return {'error': 'Local AI not available'};
    } catch (e) {
      print('Error analyzing stock data with local AI: $e');
      return {'error': e.toString()};
    }
  }
  
  /// Get device AI capabilities
  static Future<Map<String, dynamic>> getAICapabilities() async {
    try {
      if (Platform.isAndroid) {
        final String capabilities = await _channel.invokeMethod('getAICapabilities');
        return json.decode(capabilities);
      }
      return {'available': false, 'reason': 'Not Android device'};
    } catch (e) {
      print('Error getting AI capabilities: $e');
      return {'available': false, 'error': e.toString()};
    }
  }
}

