import 'package:flutter/material.dart';

class EconomicEvent {
  final String id;
  final String title;
  final String date;
  final String time;
  final String impact;
  final double? actual;
  final double? previous;
  final double? forecast;
  final String currency;
  final String source;

  EconomicEvent({
    required this.id,
    required this.title,
    required this.date,
    required this.time,
    required this.impact,
    this.actual,
    this.previous,
    this.forecast,
    required this.currency,
    required this.source,
  });

  factory EconomicEvent.fromJson(Map<String, dynamic> json) {
    return EconomicEvent(
      id: json['id'] as String,
      title: json['title'] as String,
      date: json['date'] as String,
      time: json['time'] as String? ?? '08:30',
      impact: json['impact'] as String,
      actual: json['actual'] as double?,
      previous: json['previous'] as double?,
      forecast: json['forecast'] as double?,
      currency: json['currency'] as String? ?? 'USD',
      source: json['source'] as String? ?? 'FRED',
    );
  }

  // Legacy compatibility - map to old field names
  String get event => title;
  String get country => 'US'; // Default to US for now

  // Computed properties
  double? get change => actual != null && previous != null ? actual! - previous! : null;

  String get impactEmoji {
    switch (impact.toLowerCase()) {
      case 'high':
        return '🔴';
      case 'medium':
        return '🟡';
      case 'low':
        return '🟢';
      default:
        return '⚪';
    }
  }

  Color get impactColor {
    switch (impact.toLowerCase()) {
      case 'high':
        return const Color(0xFFEF5350); // Red
      case 'medium':
        return const Color(0xFFFFA726); // Orange
      case 'low':
        return const Color(0xFF66BB6A); // Green
      default:
        return const Color(0xFF9E9E9E); // Grey
    }
  }
}
