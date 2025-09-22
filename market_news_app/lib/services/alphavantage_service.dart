import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:flutter_dotenv/flutter_dotenv.dart';
import '../models/bull_put_spread.dart';
import '../models/option_chain.dart';

class AlphaVantageService {
  static const String _baseUrl = 'https://www.alphavantage.co/query';
  static String get _apiKey => dotenv.env['ALPHAVANTAGE_API_KEY'] ?? '';

  // Get current stock price
  static Future<double?> getCurrentPrice(String symbol) async {
    try {
      final response = await http.get(
        Uri.parse('$_baseUrl?function=GLOBAL_QUOTE&symbol=$symbol&apikey=$_apiKey'),
      );

      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        final quote = data['Global Quote'];
        if (quote != null) {
          return double.tryParse(quote['05. price'] ?? '0');
        }
      }
    } catch (e) {
      print('Error fetching current price for $symbol: $e');
    }
    return null;
  }

  // Get options chain data
  static Future<OptionsChain?> getOptionsChain(String symbol) async {
    try {
      // Try Alpha Vantage REALTIME_OPTIONS first (requires premium)
      final realOptions = await _getRealTimeOptions(symbol);
      if (realOptions != null) {
        return realOptions;
      }

      // Fallback to mock data for free tier
      final currentPrice = await getCurrentPrice(symbol);
      if (currentPrice == null) return null;
      
      return _generateMockOptionsChain(symbol, currentPrice);
    } catch (e) {
      print('Error fetching options chain for $symbol: $e');
    }
    return null;
  }

  // Get real-time options data (requires Alpha Vantage premium)
  static Future<OptionsChain?> _getRealTimeOptions(String symbol) async {
    try {
      final response = await http.get(
        Uri.parse('https://www.alphavantage.co/query?function=REALTIME_OPTIONS&symbol=$symbol&apikey=$_apiKey'),
      );

      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        
        // Check if we have premium access
        if (data.containsKey('message') && data['message'].toString().contains('premium')) {
          print('⚠️  Alpha Vantage REALTIME_OPTIONS requires premium subscription');
          print('   Upgrade at: https://www.alphavantage.co/premium/');
          return null;
        }

        // Parse real options data
        if (data.containsKey('data') && data['data'] is List) {
          final optionsData = data['data'] as List;
          final puts = <OptionContract>[];
          final calls = <OptionContract>[];

          for (final option in optionsData) {
            final contract = OptionContract(
              contractSymbol: option['contractID'] ?? '',
              strike: double.tryParse(option['strike'] ?? '0') ?? 0.0,
              expirationDate: DateTime.tryParse(option['expiration'] ?? '') ?? DateTime.now(),
              type: option['type'] ?? 'put',
              lastPrice: double.tryParse(option['last'] ?? '0') ?? 0.0,
              bid: double.tryParse(option['bid'] ?? '0') ?? 0.0,
              ask: double.tryParse(option['ask'] ?? '0') ?? 0.0,
              change: 0.0, // Not provided in this endpoint
              percentChange: 0.0, // Not provided in this endpoint
              volume: int.tryParse(option['volume'] ?? '0') ?? 0,
              openInterest: int.tryParse(option['open_interest'] ?? '0') ?? 0,
              impliedVolatility: 0.0, // Not provided in this endpoint
              delta: 0.0, // Not provided in this endpoint
              gamma: 0.0,
              theta: 0.0,
              vega: 0.0,
              rho: 0.0,
              daysToExpiration: 0, // Calculate from expiration date
            );

            if (contract.type == 'put') {
              puts.add(contract);
            } else {
              calls.add(contract);
            }
          }

          return OptionsChain(
            symbol: symbol,
            underlyingPrice: await getCurrentPrice(symbol) ?? 0.0,
            lastUpdated: DateTime.now(),
            calls: calls,
            puts: puts,
          );
        }
      }

      return null;
    } catch (e) {
      print('Error getting real-time options for $symbol: $e');
      return null;
    }
  }

  // Generate realistic options chain for testing
  static OptionsChain _generateMockOptionsChain(String symbol, double currentPrice) {
    final List<OptionContract> puts = [];
    final expiration = DateTime.now().add(Duration(days: 35));
    
    // Generate put options at various strikes with more realistic pricing
    // Current market conditions: SPY ~$660, VIX ~15-20 (low volatility environment)
    for (int i = 5; i <= 20; i += 5) {
      final strikePercent = i / 100.0;
      final strike = (currentPrice * (1 - strikePercent)).roundToDouble();
      
      // More realistic Greeks and pricing for current low-vol environment
      final delta = (strikePercent * 0.4).clamp(0.05, 0.35); // More conservative delta
      final iv = 0.18 + (strikePercent * 0.15); // Lower IV in current market (15-20%)
      final intrinsicValue = (strike - currentPrice).clamp(0.0, double.infinity);
      
      // More realistic time value calculation for low-vol environment
      final timeValue = currentPrice * 0.008 * (strikePercent + 0.05) * (35 / 30.0);
      final totalValue = intrinsicValue + timeValue;
      
      // Ensure minimum bid-ask spread
      final bid = (totalValue * 0.92).clamp(0.01, double.infinity);
      final ask = (totalValue * 1.08).clamp(bid + 0.01, double.infinity);
      
      puts.add(OptionContract(
        contractSymbol: '${symbol}_${strike.toInt()}_P',
        strike: strike,
        expirationDate: expiration,
        type: 'put',
        lastPrice: (bid + ask) / 2,
        bid: bid,
        ask: ask,
        change: 0.0,
        percentChange: 0.0,
        volume: (1000 * (1 - strikePercent)).toInt(),
        openInterest: (5000 * (1 - strikePercent)).toInt(),
        impliedVolatility: iv,
        delta: -delta, // Puts have negative delta
        gamma: 0.03,
        theta: -0.015,
        vega: 0.08,
        rho: -0.01,
        daysToExpiration: 35,
      ));
    }
    
    return OptionsChain(
      symbol: symbol,
      underlyingPrice: currentPrice,
      lastUpdated: DateTime.now(),
      calls: [], // We only need puts for bull put spreads
      puts: puts,
    );
  }

  // Get implied volatility data
  static Future<double?> getImpliedVolatility(String symbol) async {
    try {
      // Alpha Vantage doesn't have direct IV endpoint, so we'll calculate from options data
      final optionsChain = await getOptionsChain(symbol);
      if (optionsChain != null && optionsChain.puts.isNotEmpty) {
        // Use ATM put IV as proxy for overall IV
        final atmPut = optionsChain.puts.firstWhere(
          (put) => (put.strike - optionsChain.underlyingPrice).abs() < 5,
          orElse: () => optionsChain.puts.first,
        );
        return atmPut.impliedVolatility;
      }
    } catch (e) {
      print('Error calculating implied volatility for $symbol: $e');
    }
    return null;
  }

  // Calculate IV Rank (requires historical IV data)
  static Future<double> calculateIVRank(String symbol) async {
    try {
      // For now, return a mock IV rank since Alpha Vantage doesn't provide historical IV directly
      // In production, you'd store historical IV data and calculate rank
      final currentIV = await getImpliedVolatility(symbol);
      if (currentIV != null) {
        // Mock calculation: assume IV ranges from 15-45% historically
        final mockLow = 15.0;
        final mockHigh = 45.0;
        final ivRank = ((currentIV - mockLow) / (mockHigh - mockLow) * 100).clamp(0.0, 100.0);
        return ivRank;
      }
    } catch (e) {
      print('Error calculating IV rank for $symbol: $e');
    }
    return 50.0; // Default to middle rank
  }

  // Generate Bull Put Spreads from real options data
  static Future<List<BullPutSpread>> generateBullPutSpreads({
    required String symbol,
    int maxDTE = 45,
    double minIVRank = 50.0,
    double minYield = 0.33,
    double minExpectedValue = 0.0, // Only show trades with positive EV
  }) async {
    try {
      final currentPrice = await getCurrentPrice(symbol);
      final optionsChain = await getOptionsChain(symbol);
      final ivRank = await calculateIVRank(symbol);

      if (currentPrice == null || optionsChain == null) {
        print('Failed to get required data for $symbol');
        return [];
      }

      // Filter options by DTE
      final validPuts = optionsChain.puts.where((put) => 
        put.daysToExpiration <= maxDTE && 
        put.daysToExpiration >= 10
      ).toList();

      if (validPuts.isEmpty) {
        print('No valid puts found for $symbol');
        return [];
      }

      List<BullPutSpread> spreads = [];

      // Generate spreads with different strike combinations
      for (int i = 0; i < validPuts.length - 1; i++) {
        final shortPut = validPuts[i];
        
        // Find corresponding long put (lower strike, same expiration)
        final longPut = validPuts.firstWhere(
          (put) => 
            put.expirationDate == shortPut.expirationDate &&
            put.strike < shortPut.strike &&
            (shortPut.strike - put.strike) >= 5 && // Minimum $5 width
            (shortPut.strike - put.strike) <= 20, // Maximum $20 width
          orElse: () => validPuts.last,
        );

        if (longPut.strike >= shortPut.strike) continue;

        // Calculate spread metrics
        final width = shortPut.strike - longPut.strike;
        final credit = shortPut.bid - longPut.ask; // Net credit received
        final yieldPercent = credit / width;
        
        // Calculate probability of profit (approximation)
        final pop = (1 - shortPut.delta) * 100;
        
        // Calculate max profit/loss
        final maxProfit = credit * 100; // Per contract
        final maxLoss = (width - credit) * 100;
        
        // Calculate Expected Value
        final winRate = pop / 100;
        final lossRate = 1 - winRate;
        final expectedValue = (winRate * maxProfit) - (lossRate * maxLoss);

        // Filter by criteria - EV must be positive for long-term profitability
        if (yieldPercent < minYield || ivRank < minIVRank || expectedValue < minExpectedValue) continue;
        if (credit <= 0) continue; // Must be a credit spread
        
        // Calculate composite score (now weighted by EV)
        final liquidityScore = _calculateLiquidityScore(shortPut, longPut);
        final technicalScore = _calculateTechnicalScore(currentPrice, shortPut.strike);
        
        // CRITICAL FILTER: Only include spreads with positive expected value
        if (expectedValue < minExpectedValue) {
          continue; // Skip this spread - negative EV
        }
        
        // Calculate EV Score (0-100) - higher is better
        final maxPossibleEV = maxProfit; // Best case scenario
        final minPossibleEV = -maxLoss; // Worst case scenario
        final evScore = ((expectedValue - minPossibleEV) / (maxPossibleEV - minPossibleEV) * 100).clamp(0.0, 100.0);
        
        // Weighted composite score prioritizing EV for long-term profitability
        final compositeScore = (evScore * 0.4) + (pop * 0.3) + (yieldPercent * 100 * 0.2) + (ivRank * 0.1);

        spreads.add(BullPutSpread(
          symbol: symbol,
          daysToExpiration: shortPut.daysToExpiration,
          shortStrike: shortPut.strike,
          longStrike: longPut.strike,
          credit: credit,
          width: width,
          yieldPercent: yieldPercent,
          shortPutDelta: shortPut.delta.abs(), // Make sure delta is positive for puts
          ivRank: ivRank,
          probabilityOfProfit: pop,
          liquidityScore: liquidityScore,
          technicalScore: technicalScore,
          compositeScore: compositeScore,
          maxProfit: maxProfit,
          maxLoss: maxLoss,
          riskRewardRatio: maxProfit / maxLoss,
          expectedValue: expectedValue,
          expectedValueScore: evScore,
          expirationDate: shortPut.expirationDate,
          openInterestShort: shortPut.openInterest,
          openInterestLong: longPut.openInterest,
          bidAskSpreadShort: shortPut.ask - shortPut.bid,
          bidAskSpreadLong: longPut.ask - longPut.bid,
          hasEarnings: false, // TODO: Add earnings calendar integration
          hasMacroEvent: false, // TODO: Add macro event calendar
        ));
      }

      // Sort by composite score (highest first)
      spreads.sort((a, b) => b.compositeScore.compareTo(a.compositeScore));
      
      // Add market analysis comment
      if (spreads.isEmpty) {
        print('⚠️  No positive EV spreads found in current market conditions');
        print('   This is common in low-volatility environments (VIX ~15-20)');
        print('   Consider waiting for higher IV or adjusting criteria');
      } else {
        print('✅ Found ${spreads.length} spreads with positive EV');
        print('   Best EV: \$${spreads.first.expectedValue.toStringAsFixed(2)}');
      }
      
      return spreads.take(10).toList(); // Return top 10 spreads
      
    } catch (e) {
      print('Error generating bull put spreads for $symbol: $e');
      return [];
    }
  }

  // Calculate liquidity score based on bid/ask spreads and volume
  static double _calculateLiquidityScore(OptionContract shortPut, OptionContract longPut) {
    final shortSpread = shortPut.ask - shortPut.bid;
    final longSpread = longPut.ask - longPut.bid;
    final avgSpread = (shortSpread + longSpread) / 2;
    
    // Score from 0-10 based on bid/ask spread (tighter = better)
    final spreadScore = (1 / (avgSpread + 0.01)).clamp(0.0, 10.0);
    
    // Factor in open interest (higher = better liquidity)
    final oiScore = ((shortPut.openInterest + longPut.openInterest) / 1000).clamp(0.0, 10.0);
    
    return (spreadScore + oiScore) / 2;
  }

  // Calculate technical score based on price levels
  static double _calculateTechnicalScore(double currentPrice, double shortStrike) {
    final distancePercent = ((currentPrice - shortStrike) / currentPrice) * 100;
    
    // Prefer strikes 5-15% out of the money
    if (distancePercent >= 5 && distancePercent <= 15) {
      return 9.0;
    } else if (distancePercent >= 3 && distancePercent <= 20) {
      return 7.0;
    } else if (distancePercent >= 1 && distancePercent <= 25) {
      return 5.0;
    } else {
      return 3.0;
    }
  }

  // Get market sentiment for filtering
  static Future<String> getMarketSentiment() async {
    try {
      // Use VIX or market indices to determine sentiment
      // For now, return neutral - in production, analyze price trends
      return 'NEUTRAL';
    } catch (e) {
      print('Error getting market sentiment: $e');
      return 'NEUTRAL';
    }
  }
}
