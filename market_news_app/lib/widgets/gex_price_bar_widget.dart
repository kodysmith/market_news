import 'package:flutter/material.dart';

/// Reusable horizontal price bar with put wall, flip line, spot, and call wall.
/// Used on GEX tab (and optionally Cockpit fallback).
class GexPriceBarWidget extends StatelessWidget {
  final double? spot;
  final double? flipLine;
  final double? putWall;
  final double? callWall;
  final double height;
  final bool darkBackground;

  const GexPriceBarWidget({
    super.key,
    this.spot,
    this.flipLine,
    this.putWall,
    this.callWall,
    this.height = 120,
    this.darkBackground = true,
  });

  @override
  Widget build(BuildContext context) {
    if (spot == null) {
      return const SizedBox.shrink();
    }

    final center = flipLine ?? spot!;
    const strikeIncrement = 1.0;
    const numStrikesEachSide = 20;
    final minStrike = center - (numStrikesEachSide * strikeIncrement);
    final maxStrike = center + (numStrikesEachSide * strikeIncrement);
    final strikes = <double>[];
    for (int i = 0; i <= numStrikesEachSide * 2; i++) {
      strikes.add(minStrike + (i * strikeIncrement));
    }
    final range = maxStrike - minStrike;
    double getPosition(double price) => ((price - minStrike) / range).clamp(0.0, 1.0);

    final putDistance = putWall != null ? spot! - putWall! : null;
    final callDistance = callWall != null ? callWall! - spot! : null;
    final flipDistance = flipLine != null ? (flipLine! - spot!) : null;

    final textColor = darkBackground ? Colors.white : Colors.black87;
    final mutedColor = darkBackground ? Colors.white54 : Colors.black54;
    final lineColor = darkBackground ? Colors.white.withOpacity(0.2) : Colors.black26;

    return Column(
      children: [
        const SizedBox(height: 8),
        SizedBox(
          height: height,
          child: LayoutBuilder(
            builder: (context, constraints) {
              final width = constraints.maxWidth;
              final h = constraints.maxHeight;
              final spotPos = getPosition(spot!);
              final putWallPos = putWall != null ? getPosition(putWall!) : null;
              final callWallPos = callWall != null ? getPosition(callWall!) : null;
              final flipPos = flipLine != null ? getPosition(flipLine!) : null;

              return Stack(
                children: [
                  ...strikes.map((strike) {
                    final pos = getPosition(strike);
                    final isKeyLevel = (strike == minStrike || strike == maxStrike ||
                        (flipLine != null && (strike - flipLine!).abs() < 0.5) ||
                        (putWall != null && (strike - putWall!).abs() < 0.5) ||
                        (callWall != null && (strike - callWall!).abs() < 0.5) ||
                        (strike - spot!).abs() < 0.5);
                    return Positioned(
                      left: pos * width,
                      child: Container(
                        width: 1,
                        height: isKeyLevel ? h : 8,
                        color: (darkBackground ? Colors.white : Colors.black).withOpacity(isKeyLevel ? 0.3 : 0.1),
                      ),
                    );
                  }),
                  Positioned(
                    top: h / 2 - 1,
                    child: Container(
                      width: width,
                      height: 2,
                      color: lineColor,
                    ),
                  ),
                  // Put Wall (show when present)
                  if (putWall != null && putWallPos != null)
                    Positioned(
                      left: putWallPos * width - 2,
                      top: 0,
                      child: Column(
                        children: [
                          Container(
                            padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 2),
                            decoration: BoxDecoration(
                              color: const Color(0xFFF85149).withOpacity(0.9),
                              borderRadius: BorderRadius.circular(3),
                            ),
                            child: Text(
                              putWall!.toStringAsFixed(0),
                              style: const TextStyle(fontSize: 11, fontWeight: FontWeight.bold, color: Colors.white, fontFamily: 'JetBrains Mono'),
                            ),
                          ),
                          const SizedBox(height: 2),
                          Container(
                            width: 4,
                            height: h - 30,
                            decoration: BoxDecoration(color: const Color(0xFFF85149), borderRadius: BorderRadius.circular(2)),
                          ),
                          const SizedBox(height: 4),
                          Container(
                            padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                            decoration: BoxDecoration(
                              color: const Color(0xFFF85149).withOpacity(0.2),
                              borderRadius: BorderRadius.circular(4),
                              border: Border.all(color: const Color(0xFFF85149)),
                            ),
                            child: Column(
                              children: [
                                const Text('PUT WALL', style: TextStyle(fontSize: 9, color: Color(0xFFF85149), fontFamily: 'JetBrains Mono')),
                                Text('(${(putDistance ?? 0).toStringAsFixed(1)} pts)', style: TextStyle(fontSize: 8, color: mutedColor, fontFamily: 'JetBrains Mono')),
                              ],
                            ),
                          ),
                        ],
                      ),
                    ),
                  // Flip line
                  if (flipPos != null)
                    Positioned(
                      left: flipPos * width - 2,
                      top: 0,
                      child: Column(
                        children: [
                          Container(
                            padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 2),
                            decoration: BoxDecoration(
                              color: const Color(0xFFFFA500).withOpacity(0.9),
                              borderRadius: BorderRadius.circular(3),
                            ),
                            child: Text(
                              flipLine!.toStringAsFixed(1),
                              style: const TextStyle(fontSize: 11, fontWeight: FontWeight.bold, color: Colors.white, fontFamily: 'JetBrains Mono'),
                            ),
                          ),
                          const SizedBox(height: 2),
                          Container(
                            width: 4,
                            height: h - 30,
                            decoration: BoxDecoration(color: const Color(0xFFFFA500), borderRadius: BorderRadius.circular(2)),
                          ),
                          const SizedBox(height: 4),
                          Container(
                            padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                            decoration: BoxDecoration(
                              color: const Color(0xFFFFA500).withOpacity(0.2),
                              borderRadius: BorderRadius.circular(4),
                              border: Border.all(color: const Color(0xFFFFA500)),
                            ),
                            child: Column(
                              children: [
                              const Text('FLIP', style: TextStyle(fontSize: 9, color: Color(0xFFFFA500), fontFamily: 'JetBrains Mono')),
                              Text('(${flipDistance!.toStringAsFixed(1)} pts)', style: TextStyle(fontSize: 8, color: mutedColor, fontFamily: 'JetBrains Mono')),
                            ],
                          ),
                        ),
                      ],
                    ),
                  ),
                  // Spot
                  Positioned(
                    left: spotPos * width - 15,
                    top: h / 2 - 15,
                    child: Column(
                      children: [
                        Container(
                          width: 30,
                          height: 30,
                          decoration: BoxDecoration(
                            color: Colors.white,
                            shape: BoxShape.circle,
                            border: Border.all(color: darkBackground ? const Color(0xFF0D1117) : Colors.grey.shade300, width: 3),
                            boxShadow: [BoxShadow(color: Colors.white.withOpacity(0.4), blurRadius: 6)],
                          ),
                          child: Center(
                            child: Text('●', style: TextStyle(fontSize: 12, color: darkBackground ? const Color(0xFF0D1117) : Colors.grey.shade700)),
                          ),
                        ),
                        const SizedBox(height: 4),
                        Container(
                          padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                          decoration: BoxDecoration(
                            color: (darkBackground ? Colors.white : Colors.black).withOpacity(0.2),
                            borderRadius: BorderRadius.circular(4),
                            border: Border.all(color: textColor),
                          ),
                          child: Column(
                            children: [
                              Text(spot!.toStringAsFixed(2), style: TextStyle(fontSize: 12, fontWeight: FontWeight.bold, color: textColor, fontFamily: 'JetBrains Mono')),
                              Text('SPOT', style: TextStyle(fontSize: 9, color: mutedColor, fontFamily: 'JetBrains Mono')),
                            ],
                          ),
                        ),
                      ],
                    ),
                  ),
                  // Call Wall (show when present)
                  if (callWall != null && callWallPos != null)
                    Positioned(
                      left: callWallPos * width - 2,
                      top: 0,
                      child: Column(
                        children: [
                          Container(
                            padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 2),
                            decoration: BoxDecoration(
                              color: const Color(0xFF3FB950).withOpacity(0.9),
                              borderRadius: BorderRadius.circular(3),
                            ),
                            child: Text(
                              callWall!.toStringAsFixed(0),
                              style: const TextStyle(fontSize: 11, fontWeight: FontWeight.bold, color: Colors.white, fontFamily: 'JetBrains Mono'),
                            ),
                          ),
                          const SizedBox(height: 2),
                          Container(
                            width: 4,
                            height: h - 30,
                            decoration: BoxDecoration(color: const Color(0xFF3FB950), borderRadius: BorderRadius.circular(2)),
                          ),
                          const SizedBox(height: 4),
                          Container(
                            padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                            decoration: BoxDecoration(
                              color: const Color(0xFF3FB950).withOpacity(0.2),
                              borderRadius: BorderRadius.circular(4),
                              border: Border.all(color: const Color(0xFF3FB950)),
                            ),
                            child: Column(
                              children: [
                                const Text('CALL WALL', style: TextStyle(fontSize: 9, color: Color(0xFF3FB950), fontFamily: 'JetBrains Mono')),
                                Text('(${(callDistance ?? 0).toStringAsFixed(1)} pts)', style: TextStyle(fontSize: 8, color: mutedColor, fontFamily: 'JetBrains Mono')),
                              ],
                            ),
                          ),
                        ],
                      ),
                    ),
                  Positioned(
                    left: 0,
                    top: h / 2 - 8,
                    child: Text(minStrike.toStringAsFixed(0), style: TextStyle(fontSize: 10, color: mutedColor, fontFamily: 'JetBrains Mono')),
                  ),
                  Positioned(
                    right: 0,
                    top: h / 2 - 8,
                    child: Text(maxStrike.toStringAsFixed(0), style: TextStyle(fontSize: 10, color: mutedColor, fontFamily: 'JetBrains Mono')),
                  ),
                ],
              );
            },
          ),
        ),
      ],
    );
  }
}
