import 'package:flutter/material.dart';
import 'package:market_news_app/models/gex_data.dart';

/// Reusable GEX chart (bars + cumulative curve + spot/flip/max pain lines).
/// Tap or hover over a strike to see strike price and GEX value.
class GexChartWidget extends StatefulWidget {
  final List<GexByStrike> data;
  final ChartAnnotations annotations;
  final List<CumulativeGexPoint> cumulativeGex;
  final double height;
  /// Optional max pain strike to plot (e.g. from cockpit state).
  final double? maxPainStrike;

  const GexChartWidget({
    super.key,
    required this.data,
    required this.annotations,
    this.cumulativeGex = const [],
    this.height = 280,
    this.maxPainStrike,
  });

  @override
  State<GexChartWidget> createState() => _GexChartWidgetState();
}

class _GexChartWidgetState extends State<GexChartWidget> {
  static const double _padding = 40.0;

  /// Tooltip state: strike and GEX at current pointer, null when hidden.
  double? _tooltipStrike;
  double? _tooltipGex;
  Offset? _tooltipPosition;
  /// Tap keeps tooltip visible until next tap; hover clears on exit.
  bool _tooltipFromTap = false;

  void _updateTooltip(Offset localPosition, Size size) {
    if (widget.data.isEmpty) return;
    final chartWidth = size.width - (_padding * 2);
    if (chartWidth <= 0) return;
    final x = localPosition.dx;
    if (x < _padding || x > size.width - _padding) {
      if (_tooltipFromTap) {
        setState(() {
          _tooltipStrike = null;
          _tooltipGex = null;
          _tooltipPosition = null;
          _tooltipFromTap = false;
        });
      } else {
        _clearTooltip();
      }
      return;
    }
    final barIndex = ((x - _padding) / chartWidth * widget.data.length).floor().clamp(0, widget.data.length - 1);
    final item = widget.data[barIndex];
    setState(() {
      _tooltipStrike = item.strike;
      _tooltipGex = item.gex;
      _tooltipPosition = localPosition;
    });
  }

  void _clearTooltip() {
    if (_tooltipFromTap) return;
    if (_tooltipStrike != null || _tooltipPosition != null) {
      setState(() {
        _tooltipStrike = null;
        _tooltipGex = null;
        _tooltipPosition = null;
      });
    }
  }

  void _onTapDown(TapDownDetails details, Size size) {
    _tooltipFromTap = true;
    _updateTooltip(details.localPosition, size);
  }

  void _onTapUp(TapUpDetails details) {
    _tooltipFromTap = false;
  }

  void _onTapCancel() {
    _tooltipFromTap = false;
  }

  static String _formatGex(double gex) {
    final absGex = gex.abs();
    if (absGex >= 1e12) return '${(gex / 1e12).toStringAsFixed(1)}T';
    if (absGex >= 1e9) return '${(gex / 1e9).toStringAsFixed(1)}B';
    if (absGex >= 1e6) return '${(gex / 1e6).toStringAsFixed(1)}M';
    if (absGex >= 1e3) return '${(gex / 1e3).toStringAsFixed(1)}K';
    return gex.toStringAsFixed(0);
  }

  @override
  Widget build(BuildContext context) {
    if (widget.data.isEmpty) {
      return const SizedBox(
        height: 120,
        child: Center(child: Text('No chart data available')),
      );
    }
    return SizedBox(
      height: widget.height,
      child: LayoutBuilder(
        builder: (context, constraints) {
          final w = constraints.maxWidth;
          final h = constraints.maxHeight;
          if (w <= 0 || h <= 0) return const SizedBox.shrink();
          final size = Size(w, h);
          return MouseRegion(
            onExit: (_) => _clearTooltip(),
            child: Listener(
              onPointerMove: (event) {
                if (!_tooltipFromTap) _updateTooltip(event.localPosition, size);
              },
              child: GestureDetector(
                onTapDown: (details) => _onTapDown(details, size),
                onTapUp: (d) => _onTapUp(d),
                onTapCancel: _onTapCancel,
              behavior: HitTestBehavior.opaque,
              child: Stack(
                clipBehavior: Clip.none,
                children: [
                  CustomPaint(
                    painter: GexChartPainter(
                      widget.data,
                      widget.annotations,
                      widget.cumulativeGex,
                      maxPainStrike: widget.maxPainStrike,
                    ),
                    size: size,
                  ),
                  if (_tooltipStrike != null && _tooltipPosition != null)
                    Positioned(
                      left: (_tooltipPosition!.dx - 60).clamp(8.0, w - 128),
                      top: (_tooltipPosition!.dy - 36).clamp(8.0, h - 44),
                      child: Material(
                        elevation: 8,
                        borderRadius: BorderRadius.circular(8),
                        color: const Color(0xFF21262D),
                        child: Padding(
                          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                          child: Column(
                            mainAxisSize: MainAxisSize.min,
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text(
                                'Strike \$${_tooltipStrike!.toStringAsFixed(0)}',
                                style: const TextStyle(
                                  fontSize: 14,
                                  fontWeight: FontWeight.bold,
                                  color: Colors.white,
                                  fontFamily: 'JetBrains Mono',
                                ),
                              ),
                              const SizedBox(height: 2),
                              Text(
                                'GEX ${_tooltipGex! >= 0 ? '+' : ''}${_formatGex(_tooltipGex!)}',
                                style: TextStyle(
                                  fontSize: 12,
                                  color: _tooltipGex! >= 0 ? const Color(0xFF3FB950) : const Color(0xFFF85149),
                                  fontFamily: 'JetBrains Mono',
                                ),
                              ),
                            ],
                          ),
                        ),
                      ),
                    ),
                ],
              ),
            ),
            ),
          );
        },
      ),
    );
  }
}

/// Chart painter for GEX by strike (bars + cumulative curve + annotations).
class GexChartPainter extends CustomPainter {
  final List<GexByStrike> data;
  final ChartAnnotations annotations;
  final List<CumulativeGexPoint> cumulativeGex;
  final double? maxPainStrike;

  GexChartPainter(this.data, this.annotations, this.cumulativeGex, {this.maxPainStrike});

  static const double _labelPadding = 4.0;
  static const double _labelFontSize = 10.0;

  void _drawLabel(Canvas canvas, double x, String label, String value, Color color) {
    final text = '$label ${value}';
    final tp = TextPainter(
      text: TextSpan(
        text: text,
        style: TextStyle(
          color: Colors.white,
          fontSize: _labelFontSize,
          fontWeight: FontWeight.w600,
          fontFamily: 'JetBrains Mono',
        ),
      ),
      textDirection: TextDirection.ltr,
    )..layout(minWidth: 0, maxWidth: 120);

    final boxWidth = tp.width + _labelPadding * 2;
    final boxHeight = tp.height + _labelPadding;
    final left = x - boxWidth / 2;
    final top = _labelPadding;

    final boxR = RRect.fromRectAndRadius(
      Rect.fromLTWH(left, top, boxWidth, boxHeight),
      const Radius.circular(4),
    );
    canvas.drawRRect(boxR, Paint()..color = color);
    tp.paint(canvas, Offset(left + _labelPadding, top + _labelPadding / 2));
  }

  @override
  void paint(Canvas canvas, Size size) {
    if (data.isEmpty) return;

    final padding = 40.0;
    final chartWidth = size.width - (padding * 2);
    final chartHeight = size.height - (padding * 2);

    double minGex = data.map((d) => d.gex).reduce((a, b) => a < b ? a : b);
    double maxGex = data.map((d) => d.gex).reduce((a, b) => a > b ? a : b);
    double minStrike = data.map((d) => d.strike).reduce((a, b) => a < b ? a : b);
    double maxStrike = data.map((d) => d.strike).reduce((a, b) => a > b ? a : b);

    minGex = minGex < 0 ? minGex * 1.1 : 0;
    maxGex = maxGex > 0 ? maxGex * 1.1 : 0;

    final gexRange = maxGex - minGex;
    final strikeRange = maxStrike - minStrike;

    final zeroY = padding + chartHeight - ((0 - minGex) / gexRange * chartHeight);
    final zeroPaint = Paint()
      ..color = Colors.grey
      ..strokeWidth = 1;
    canvas.drawLine(
      Offset(padding, zeroY),
      Offset(size.width - padding, zeroY),
      zeroPaint,
    );

    double minCumGex = 0.0;
    double maxCumGex = 0.0;
    if (cumulativeGex.isNotEmpty) {
      minCumGex = cumulativeGex.map((d) => d.cumulativeGex).reduce((a, b) => a < b ? a : b);
      maxCumGex = cumulativeGex.map((d) => d.cumulativeGex).reduce((a, b) => a > b ? a : b);
    }
    final cumGexRange = maxCumGex - minCumGex;
    if (cumulativeGex.isNotEmpty && cumGexRange > 0) {
      final segments = <Path>[];
      final colors = <Color>[];
      for (int i = 0; i < cumulativeGex.length - 1; i++) {
        final point1 = cumulativeGex[i];
        final point2 = cumulativeGex[i + 1];
        if (point1.strike < minStrike || point2.strike > maxStrike) continue;
        final x1 = padding + ((point1.strike - minStrike) / strikeRange * chartWidth);
        final y1 = padding + chartHeight - ((point1.cumulativeGex - minCumGex) / cumGexRange * chartHeight);
        final x2 = padding + ((point2.strike - minStrike) / strikeRange * chartWidth);
        final y2 = padding + chartHeight - ((point2.cumulativeGex - minCumGex) / cumGexRange * chartHeight);
        final segmentPath = Path()..moveTo(x1, y1)..lineTo(x2, y2);
        segments.add(segmentPath);
        final avgCumGex = (point1.cumulativeGex + point2.cumulativeGex) / 2;
        colors.add(avgCumGex >= 0 ? Colors.green.shade700 : Colors.red.shade700);
      }
      for (int i = 0; i < segments.length; i++) {
        final segmentPaint = Paint()
          ..style = PaintingStyle.stroke
          ..strokeWidth = 3.0
          ..color = colors[i];
        canvas.drawPath(segments[i], segmentPaint);
      }
      if (minCumGex < 0 && maxCumGex > 0) {
        final zeroCumY = padding + chartHeight - ((0 - minCumGex) / cumGexRange * chartHeight);
        final zeroCumPaint = Paint()
          ..color = Colors.grey.withOpacity(0.5)
          ..strokeWidth = 1
          ..style = PaintingStyle.stroke;
        canvas.drawLine(
          Offset(padding, zeroCumY),
          Offset(size.width - padding, zeroCumY),
          zeroCumPaint,
        );
      }
    }

    final barWidth = chartWidth / data.length;
    for (int i = 0; i < data.length; i++) {
      final item = data[i];
      final x = padding + (i * barWidth);
      final barHeight = (item.gex.abs() / gexRange * chartHeight);
      final barY = item.gex >= 0 ? zeroY - barHeight : zeroY;
      final barPaint = Paint()
        ..color = item.gex >= 0 ? Colors.green.withOpacity(0.6) : Colors.red.withOpacity(0.6)
        ..style = PaintingStyle.fill;
      canvas.drawRect(
        Rect.fromLTWH(x, barY, barWidth * 0.8, barHeight),
        barPaint,
      );
    }

    final annotationPaint = Paint()
      ..strokeWidth = 2
      ..style = PaintingStyle.stroke;

    // Spot (green)
    if (annotations.spotPrice >= minStrike && annotations.spotPrice <= maxStrike) {
      final spotX = padding + ((annotations.spotPrice - minStrike) / strikeRange * chartWidth);
      annotationPaint.color = const Color(0xFF3FB950); // green
      annotationPaint.strokeWidth = 2;
      annotationPaint.style = PaintingStyle.stroke;
      canvas.drawLine(
        Offset(spotX, padding),
        Offset(spotX, size.height - padding),
        annotationPaint,
      );
      _drawLabel(canvas, spotX, 'SPOT', annotations.spotPrice.toStringAsFixed(1), const Color(0xFF3FB950));
    }

    // Flip line (orange)
    if (annotations.flipLine != null &&
        annotations.flipLine! >= minStrike &&
        annotations.flipLine! <= maxStrike) {
      final flipX = padding + ((annotations.flipLine! - minStrike) / strikeRange * chartWidth);
      annotationPaint.color = const Color(0xFFFFA500);
      canvas.drawLine(
        Offset(flipX, padding),
        Offset(flipX, size.height - padding),
        annotationPaint,
      );
      _drawLabel(canvas, flipX, 'FLIP', annotations.flipLine!.toStringAsFixed(1), const Color(0xFFFFA500));
    }

    // Put wall (red) - clamp to left edge if below chart range
    if (annotations.putWall > 0 && strikeRange > 0) {
      double putX;
      if (annotations.putWall <= minStrike) {
        putX = padding;
      } else if (annotations.putWall >= maxStrike) {
        putX = padding + chartWidth;
      } else {
        putX = padding + ((annotations.putWall - minStrike) / strikeRange * chartWidth);
      }
      annotationPaint.color = const Color(0xFFF85149);
      annotationPaint.strokeWidth = 2;
      annotationPaint.style = PaintingStyle.stroke;
      canvas.drawLine(
        Offset(putX, padding),
        Offset(putX, size.height - padding),
        annotationPaint,
      );
      _drawLabel(canvas, putX, 'PUT', annotations.putWall.toStringAsFixed(0), const Color(0xFFF85149));
    }

    // Call wall (green/teal) - clamp to right edge if above chart range
    if (annotations.callWall > 0 && strikeRange > 0) {
      double callX;
      if (annotations.callWall <= minStrike) {
        callX = padding;
      } else if (annotations.callWall >= maxStrike) {
        callX = padding + chartWidth;
      } else {
        callX = padding + ((annotations.callWall - minStrike) / strikeRange * chartWidth);
      }
      annotationPaint.color = const Color(0xFF58A6FF); // blue (call wall)
      annotationPaint.strokeWidth = 2;
      annotationPaint.style = PaintingStyle.stroke;
      canvas.drawLine(
        Offset(callX, padding),
        Offset(callX, size.height - padding),
        annotationPaint,
      );
      _drawLabel(canvas, callX, 'CALL', annotations.callWall.toStringAsFixed(0), const Color(0xFF58A6FF));
    }

    // Max pain (purple) - optional; clamp to edges if outside strike range
    if (maxPainStrike != null && maxPainStrike! > 0 && strikeRange > 0) {
      double maxPainX;
      if (maxPainStrike! <= minStrike) {
        maxPainX = padding;
      } else if (maxPainStrike! >= maxStrike) {
        maxPainX = padding + chartWidth;
      } else {
        maxPainX = padding + ((maxPainStrike! - minStrike) / strikeRange * chartWidth);
      }
      annotationPaint.color = const Color(0xFF9B59B6); // purple (max pain)
      annotationPaint.strokeWidth = 2;
      annotationPaint.style = PaintingStyle.stroke;
      canvas.drawLine(
        Offset(maxPainX, padding),
        Offset(maxPainX, size.height - padding),
        annotationPaint,
      );
      _drawLabel(canvas, maxPainX, 'MAX PAIN', maxPainStrike!.toStringAsFixed(0), const Color(0xFF9B59B6));
    }

    // Legend at bottom: color-coded labels
    final legendY = size.height - 14;
    final legendColors = <(Color, String)>[
      (const Color(0xFFF85149), 'Put'),
      (const Color(0xFFFFA500), 'Flip'),
      (const Color(0xFF3FB950), 'Spot'),
      (const Color(0xFF58A6FF), 'Call'),
      if (maxPainStrike != null && maxPainStrike! > 0) (const Color(0xFF9B59B6), 'Max Pain'),
    ];
    double legendX = padding;
    for (final entry in legendColors) {
      final color = entry.$1;
      final name = entry.$2;
      canvas.drawRect(
        Rect.fromLTWH(legendX, legendY - 4, 8, 8),
        Paint()..color = color,
      );
      final tp = TextPainter(
        text: TextSpan(
          text: name,
          style: TextStyle(color: color, fontSize: 9, fontFamily: 'JetBrains Mono', fontWeight: FontWeight.w500),
        ),
        textDirection: TextDirection.ltr,
      )..layout(minWidth: 0, maxWidth: 40);
      tp.paint(canvas, Offset(legendX + 10, legendY - 6));
      legendX += 14 + tp.width;
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => true;
}
