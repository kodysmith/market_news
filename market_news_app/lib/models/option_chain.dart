class OptionsChain {
  final String symbol;
  final double underlyingPrice;
  final DateTime lastUpdated;
  final List<OptionContract> calls;
  final List<OptionContract> puts;

  OptionsChain({
    required this.symbol,
    required this.underlyingPrice,
    required this.lastUpdated,
    required this.calls,
    required this.puts,
  });

  factory OptionsChain.fromJson(Map<String, dynamic> json) {
    // Alpha Vantage options data structure
    final data = json['data'] ?? [];
    final List<OptionContract> calls = [];
    final List<OptionContract> puts = [];

    for (var option in data) {
      final contract = OptionContract.fromJson(option);
      if (contract.type == 'call') {
        calls.add(contract);
      } else {
        puts.add(contract);
      }
    }

    return OptionsChain(
      symbol: json['symbol'] ?? '',
      underlyingPrice: double.tryParse(json['underlying_price']?.toString() ?? '0') ?? 0.0,
      lastUpdated: DateTime.tryParse(json['last_updated'] ?? '') ?? DateTime.now(),
      calls: calls,
      puts: puts,
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'symbol': symbol,
      'underlying_price': underlyingPrice,
      'last_updated': lastUpdated.toIso8601String(),
      'calls': calls.map((c) => c.toJson()).toList(),
      'puts': puts.map((p) => p.toJson()).toList(),
    };
  }
}

class OptionContract {
  final String contractSymbol;
  final double strike;
  final DateTime expirationDate;
  final String type; // 'call' or 'put'
  final double lastPrice;
  final double bid;
  final double ask;
  final double change;
  final double percentChange;
  final int volume;
  final int openInterest;
  final double impliedVolatility;
  final double delta;
  final double gamma;
  final double theta;
  final double vega;
  final double rho;
  final int daysToExpiration;

  OptionContract({
    required this.contractSymbol,
    required this.strike,
    required this.expirationDate,
    required this.type,
    required this.lastPrice,
    required this.bid,
    required this.ask,
    required this.change,
    required this.percentChange,
    required this.volume,
    required this.openInterest,
    required this.impliedVolatility,
    required this.delta,
    required this.gamma,
    required this.theta,
    required this.vega,
    required this.rho,
    required this.daysToExpiration,
  });

  factory OptionContract.fromJson(Map<String, dynamic> json) {
    return OptionContract(
      contractSymbol: json['contractSymbol'] ?? '',
      strike: double.tryParse(json['strike']?.toString() ?? '0') ?? 0.0,
      expirationDate: DateTime.tryParse(json['expiration'] ?? '') ?? DateTime.now(),
      type: json['type']?.toString().toLowerCase() ?? 'call',
      lastPrice: double.tryParse(json['lastPrice']?.toString() ?? '0') ?? 0.0,
      bid: double.tryParse(json['bid']?.toString() ?? '0') ?? 0.0,
      ask: double.tryParse(json['ask']?.toString() ?? '0') ?? 0.0,
      change: double.tryParse(json['change']?.toString() ?? '0') ?? 0.0,
      percentChange: double.tryParse(json['percentChange']?.toString() ?? '0') ?? 0.0,
      volume: int.tryParse(json['volume']?.toString() ?? '0') ?? 0,
      openInterest: int.tryParse(json['openInterest']?.toString() ?? '0') ?? 0,
      impliedVolatility: double.tryParse(json['impliedVolatility']?.toString() ?? '0') ?? 0.0,
      delta: double.tryParse(json['delta']?.toString() ?? '0') ?? 0.0,
      gamma: double.tryParse(json['gamma']?.toString() ?? '0') ?? 0.0,
      theta: double.tryParse(json['theta']?.toString() ?? '0') ?? 0.0,
      vega: double.tryParse(json['vega']?.toString() ?? '0') ?? 0.0,
      rho: double.tryParse(json['rho']?.toString() ?? '0') ?? 0.0,
      daysToExpiration: _calculateDTE(json['expiration'] ?? ''),
    );
  }

  static int _calculateDTE(String expirationString) {
    final expiration = DateTime.tryParse(expirationString);
    if (expiration != null) {
      final now = DateTime.now();
      return expiration.difference(now).inDays;
    }
    return 0;
  }

  Map<String, dynamic> toJson() {
    return {
      'contractSymbol': contractSymbol,
      'strike': strike,
      'expiration': expirationDate.toIso8601String(),
      'type': type,
      'lastPrice': lastPrice,
      'bid': bid,
      'ask': ask,
      'change': change,
      'percentChange': percentChange,
      'volume': volume,
      'openInterest': openInterest,
      'impliedVolatility': impliedVolatility,
      'delta': delta,
      'gamma': gamma,
      'theta': theta,
      'vega': vega,
      'rho': rho,
      'daysToExpiration': daysToExpiration,
    };
  }

  // Helper getters
  double get midPrice => (bid + ask) / 2;
  double get bidAskSpread => ask - bid;
  bool get isLiquid => bidAskSpread < 0.10 && openInterest > 50;
  bool get isITM => type == 'call' ? strike < lastPrice : strike > lastPrice;
  bool get isOTM => !isITM;
}

