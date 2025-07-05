import 'package:flutter_test/flutter_test.dart';
import 'package:market_news_app/services/fmp_api_service.dart';
import 'package:market_news_app/models/vix_data.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'dart:convert';

void main() {
  group('FmpApiService', () {
    test('fetchVixData returns a list of VixData on successful response', () async {
      final apiService = FmpApiService('test_api_key');
      final mockClient = MockClient((request) async {
        final data = [
          {'date': '2023-01-01', 'close': 20.0},
          {'date': '2023-01-02', 'close': 21.0},
        ];
        return http.Response(json.encode(data), 200);
      });

      final result = await apiService.fetchVixData(client: mockClient);

      expect(result, isA<List<VixData>>());
      expect(result.length, 2);
    });

    test('fetchVixData throws an exception on failed response', () async {
      final apiService = FmpApiService('test_api_key');
      final mockClient = MockClient((request) async {
        return http.Response('Not Found', 404);
      });

      expect(apiService.fetchVixData(client: mockClient), throwsException);
    });
  });
}
