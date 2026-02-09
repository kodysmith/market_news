import 'dart:convert';
import 'package:http/http.dart' as http;
import '../main.dart' show apiBaseUrl, apiSecretKey;
import 'compute_queue_service.dart';

/// Response from GET /probability/range (Panel A/B/C).
class ProbabilityRangeResponse {
  final String ticker;
  final double spot;
  final double sigmaEff;
  final int minutesToClose;
  final List<ConePoint> cone;
  final List<HistogramBucket> highDistribution;
  final List<HistogramBucket> lowDistribution;
  final double hph;
  final double hpl;
  final double p50High;
  final double p80High;
  final double p50Low;
  final double p80Low;
  final List<HistogramBucket> closeDistribution;
  final double closeMode;
  final double closeP25;
  final double closeP75;
  final double closeP10;
  final double closeP90;
  final double? putWall;
  final double? callWall;

  ProbabilityRangeResponse({
    required this.ticker,
    required this.spot,
    required this.sigmaEff,
    required this.minutesToClose,
    required this.cone,
    required this.highDistribution,
    required this.lowDistribution,
    required this.hph,
    required this.hpl,
    required this.p50High,
    required this.p80High,
    required this.p50Low,
    required this.p80Low,
    required this.closeDistribution,
    required this.closeMode,
    required this.closeP25,
    required this.closeP75,
    required this.closeP10,
    required this.closeP90,
    this.putWall,
    this.callWall,
  });

  factory ProbabilityRangeResponse.fromJson(Map<String, dynamic> json) {
    final coneList = json['cone'] as List<dynamic>? ?? [];
    final highList = json['high_distribution'] as List<dynamic>? ?? [];
    final lowList = json['low_distribution'] as List<dynamic>? ?? [];
    final closeList = json['close_distribution'] as List<dynamic>? ?? [];
    return ProbabilityRangeResponse(
      ticker: json['ticker'] as String? ?? '',
      spot: (json['inputs']?['spot'] as num?)?.toDouble() ?? (json['spot'] as num?)?.toDouble() ?? 0,
      sigmaEff: (json['sigma_eff'] as num?)?.toDouble() ?? 0,
      minutesToClose: (json['inputs']?['minutes_to_close'] as int?) ?? (json['minutes_to_close'] as int?) ?? 0,
      cone: coneList.map((e) => ConePoint.fromJson(e as Map<String, dynamic>)).toList(),
      highDistribution: highList.map((e) => HistogramBucket.fromJson(e as Map<String, dynamic>)).toList(),
      lowDistribution: lowList.map((e) => HistogramBucket.fromJson(e as Map<String, dynamic>)).toList(),
      hph: (json['hph'] as num?)?.toDouble() ?? 0,
      hpl: (json['hpl'] as num?)?.toDouble() ?? 0,
      p50High: (json['p50_high'] as num?)?.toDouble() ?? 0,
      p80High: (json['p80_high'] as num?)?.toDouble() ?? 0,
      p50Low: (json['p50_low'] as num?)?.toDouble() ?? 0,
      p80Low: (json['p80_low'] as num?)?.toDouble() ?? 0,
      closeDistribution: closeList.map((e) => HistogramBucket.fromJson(e as Map<String, dynamic>)).toList(),
      closeMode: (json['close_mode'] as num?)?.toDouble() ?? 0,
      closeP25: (json['close_p25'] as num?)?.toDouble() ?? 0,
      closeP75: (json['close_p75'] as num?)?.toDouble() ?? 0,
      closeP10: (json['close_p10'] as num?)?.toDouble() ?? 0,
      closeP90: (json['close_p90'] as num?)?.toDouble() ?? 0,
      putWall: (json['put_wall'] as num?)?.toDouble(),
      callWall: (json['call_wall'] as num?)?.toDouble(),
    );
  }
}

class ConePoint {
  final double minutesFromNow;
  final double p10;
  final double p25;
  final double p50;
  final double p75;
  final double p90;

  ConePoint({
    required this.minutesFromNow,
    required this.p10,
    required this.p25,
    required this.p50,
    required this.p75,
    required this.p90,
  });

  factory ConePoint.fromJson(Map<String, dynamic> json) {
    return ConePoint(
      minutesFromNow: (json['minutes_from_now'] as num?)?.toDouble() ?? 0,
      p10: (json['p10'] as num?)?.toDouble() ?? 0,
      p25: (json['p25'] as num?)?.toDouble() ?? 0,
      p50: (json['p50'] as num?)?.toDouble() ?? 0,
      p75: (json['p75'] as num?)?.toDouble() ?? 0,
      p90: (json['p90'] as num?)?.toDouble() ?? 0,
    );
  }
}

class HistogramBucket {
  final double bucket;
  final int count;

  HistogramBucket({required this.bucket, required this.count});

  factory HistogramBucket.fromJson(Map<String, dynamic> json) {
    return HistogramBucket(
      bucket: (json['bucket'] as num?)?.toDouble() ?? 0,
      count: (json['count'] as int?) ?? 0,
    );
  }
}

class ProbabilityService {
  /// Get probability range (Supabase cache/queue first when configured, else API).
  static Future<ProbabilityRangeResponse?> getProbabilityRange(String ticker, {int? minutesToClose}) async {
    final t = ticker.toUpperCase();
    if (ComputeQueueService.isAvailable) {
      final result = await ComputeQueueService.getCachedOrEnqueue(
        symbol: t,
        taskType: ComputeTaskType.probability,
      );
      if (result.ok && result.result != null) {
        try {
          return ProbabilityRangeResponse.fromJson(result.result!);
        } catch (e) {
          print('Error parsing probability from Supabase: $e');
        }
      }
    }
    try {
      final params = <String, String>{'ticker': t};
      if (minutesToClose != null) params['minutes_to_close'] = minutesToClose.toString();
      final uri = Uri.parse('$apiBaseUrl/probability/range').replace(queryParameters: params);
      final response = await http.get(
        uri,
        headers: {'Content-Type': 'application/json', 'x-api-key': apiSecretKey},
      );
      if (response.statusCode == 200) {
        final data = json.decode(response.body) as Map<String, dynamic>;
        return ProbabilityRangeResponse.fromJson(data);
      }
      return null;
    } catch (e) {
      print('Error fetching probability range: $e');
      return null;
    }
  }
}
