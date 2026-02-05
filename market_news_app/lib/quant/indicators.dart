import 'dart:math';

class Indicators {
  static double sma(List<double> values, int period) {
    if (values.isEmpty) return 0;
    if (values.length < period) {
      final sum = values.fold<double>(0, (a, b) => a + b);
      return sum / values.length;
    }
    final window = values.sublist(values.length - period);
    final sum = window.fold<double>(0, (a, b) => a + b);
    return sum / period;
  }

  /// RSI using Wilder's smoothing approximation.
  static double rsi(List<double> closes, {int period = 14}) {
    if (closes.length < period + 1) return 50;

    double gain = 0;
    double loss = 0;

    for (var i = closes.length - period; i < closes.length; i++) {
      final delta = closes[i] - closes[i - 1];
      if (delta >= 0) {
        gain += delta;
      } else {
        loss -= delta;
      }
    }

    final avgGain = gain / period;
    final avgLoss = loss / period;
    if (avgLoss == 0) return 100;
    final rs = avgGain / avgLoss;
    return 100 - (100 / (1 + rs));
  }

  static (double support, double resistance) supportResistance(List<double> closes, {int lookback = 20}) {
    if (closes.isEmpty) return (0, 0);
    final window = closes.length <= lookback ? closes : closes.sublist(closes.length - lookback);
    final support = window.reduce(min);
    final resistance = window.reduce(max);
    return (support, resistance);
  }

  static String trend({
    required double currentPrice,
    required double sma20,
    required double sma50,
  }) {
    if (currentPrice > sma20 && sma20 > sma50) return 'uptrend';
    if (currentPrice < sma20 && sma20 < sma50) return 'downtrend';
    return 'sideways';
  }
}

