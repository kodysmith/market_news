import '../models/opportunity_v1.dart';
import '../quant/opportunity_scorer.dart';
import 'yahoo_finance_service.dart';

class OpportunityScannerService {
  /// Minimal, on-device opportunity scan using Yahoo historical closes.
  ///
  /// `sectorByTicker` can be expanded later; for now, default to 'Unknown'.
  static Future<List<OpportunityV1>> scan({
    required List<String> tickers,
    Map<String, String>? sectorByTicker,
  }) async {
    final results = <OpportunityV1>[];
    final asOf = DateTime.now().toUtc();

    for (final t in tickers) {
      final ticker = t.trim().toUpperCase();
      if (ticker.isEmpty) continue;

      final closes = await YahooFinanceService.getHistoricalCloses(ticker, range: '3mo', interval: '1d');
      if (closes.length < 30) continue;

      final sector = sectorByTicker?[ticker] ?? 'Unknown';

      final opp = OpportunityScorer.score(
        ticker: ticker,
        sector: sector,
        closes: closes,
        asOf: asOf,
      );

      // Only keep meaningful signals to reduce noise.
      if (opp.overallScore <= 30 || opp.overallScore >= 70) {
        results.add(opp);
      }
    }

    // Sort: strongest first
    results.sort((a, b) => b.overallScore.compareTo(a.overallScore));
    return results;
  }
}

