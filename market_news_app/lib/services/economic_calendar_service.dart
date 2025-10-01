import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:market_news_app/models/economic_event.dart';

class EconomicCalendarService {
  // Use local API for development, can switch to production later
  static const String baseUrl = 'http://localhost:5000';

  static Future<List<EconomicEvent>> getEconomicCalendar({
    String? impact,
    int limit = 50,
  }) async {
    try {
      final queryParams = <String, String>{};
      if (impact != null) queryParams['impact'] = impact;
      queryParams['limit'] = limit.toString();

      final uri = Uri.parse('$baseUrl/economic-calendar').replace(queryParameters: queryParams);
      final response = await http.get(uri);

      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        final events = (data['events'] as List)
            .map((event) => EconomicEvent.fromJson(event))
            .toList();
        return events;
      } else {
        throw Exception('Failed to load economic calendar: ${response.statusCode}');
      }
    } catch (e) {
      print('Error fetching economic calendar: $e');
      return [];
    }
  }

  static Future<List<EconomicEvent>> getUpcomingEconomicEvents({
    int daysAhead = 7,
  }) async {
    try {
      final uri = Uri.parse('$baseUrl/economic-calendar/upcoming').replace(
        queryParameters: {'days': daysAhead.toString()},
      );
      final response = await http.get(uri);

      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        final events = (data['events'] as List)
            .map((event) => EconomicEvent.fromJson(event))
            .toList();
        return events;
      } else {
        throw Exception('Failed to load upcoming events: ${response.statusCode}');
      }
    } catch (e) {
      print('Error fetching upcoming economic events: $e');
      return [];
    }
  }

  static Future<List<EconomicEvent>> getHighImpactEvents() async {
    try {
      final response = await http.get(Uri.parse('$baseUrl/economic-calendar/high-impact'));

      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        final events = (data['events'] as List)
            .map((event) => EconomicEvent.fromJson(event))
            .toList();
        return events;
      } else {
        throw Exception('Failed to load high impact events: ${response.statusCode}');
      }
    } catch (e) {
      print('Error fetching high impact events: $e');
      return [];
    }
  }

  // Utility methods for filtering and sorting
  static List<EconomicEvent> filterByImpact(List<EconomicEvent> events, String impact) {
    return events.where((event) => event.impact.toLowerCase() == impact.toLowerCase()).toList();
  }

  static List<EconomicEvent> sortByDate(List<EconomicEvent> events, {bool ascending = false}) {
    return List.from(events)
      ..sort((a, b) {
        final dateA = DateTime.parse(a.date);
        final dateB = DateTime.parse(b.date);
        return ascending ? dateA.compareTo(dateB) : dateB.compareTo(dateA);
      });
  }

  static List<EconomicEvent> getTodaysEvents(List<EconomicEvent> events) {
    final today = DateTime.now().toIso8601String().split('T')[0];
    return events.where((event) => event.date == today).toList();
  }

  static List<EconomicEvent> getThisWeeksEvents(List<EconomicEvent> events) {
    final now = DateTime.now();
    final weekFromNow = now.add(const Duration(days: 7));
    return events.where((event) {
      final eventDate = DateTime.parse(event.date);
      return eventDate.isAfter(now) && eventDate.isBefore(weekFromNow);
    }).toList();
  }
}


