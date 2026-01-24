import 'package:flutter/material.dart';
import '../services/cockpit_service.dart';

/// Decision Cockpit - Compact Single-Screen Trading State View
class DecisionCockpitScreen extends StatefulWidget {
  const DecisionCockpitScreen({super.key});

  @override
  State<DecisionCockpitScreen> createState() => _DecisionCockpitScreenState();
}

class _DecisionCockpitScreenState extends State<DecisionCockpitScreen> {
  CockpitState? _state;
  CockpitEventsData? _events;
  bool _isLoading = true;
  String? _error;
  String _selectedTicker = 'SPY';
  bool _eventsExpanded = false;
  final List<String> _tickers = ['SPY', 'QQQ', 'IWM'];

  @override
  void initState() {
    super.initState();
    _loadData();
  }

  Future<void> _loadData() async {
    setState(() {
      _isLoading = true;
      _error = null;
    });

    try {
      final results = await Future.wait([
        CockpitService.getCockpitState(_selectedTicker),
        CockpitService.getCockpitEvents(daysAhead: 14),
      ]);
      
      setState(() {
        _state = results[0] as CockpitState?;
        _events = results[1] as CockpitEventsData?;
        _isLoading = false;
        if (_state == null) {
          _error = 'Failed to load cockpit state';
        }
      });
    } catch (e) {
      setState(() {
        _isLoading = false;
        _error = e.toString();
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF0D1117),
      body: SafeArea(child: _buildBody()),
    );
  }

  Widget _buildBody() {
    if (_isLoading) {
      return const Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            CircularProgressIndicator(color: Color(0xFF58A6FF)),
            SizedBox(height: 16),
            Text('Loading cockpit...', style: TextStyle(color: Colors.white70, fontSize: 16)),
          ],
        ),
      );
    }

    if (_error != null || _state == null) {
      return Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Icon(Icons.error_outline, color: Colors.red, size: 48),
            const SizedBox(height: 16),
            Text(_error ?? 'Unknown error', style: const TextStyle(color: Colors.white70, fontSize: 16)),
            const SizedBox(height: 16),
            ElevatedButton(onPressed: _loadData, child: const Text('Retry')),
          ],
        ),
      );
    }

    return SingleChildScrollView(
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          children: [
            // Compact Header
            _buildCompactHeader(),
            const SizedBox(height: 12),
            // Regime Card with Price Bar
            _buildRegimeCard(),
            const SizedBox(height: 12),
            // Structure Card
            _buildStructureCard(),
            const SizedBox(height: 12),
            // Action Chips
            _buildActionChips(),
            const SizedBox(height: 12),
            // Events Section
            _buildEventsSection(),
          ],
        ),
      ),
    );
  }

  /// Compact Header: Ticker + Events + Vol + Refresh
  Widget _buildCompactHeader() {
    final vol = _state!.volatility;
    final volColor = vol.direction == 'RISING'
        ? const Color(0xFFF85149)
        : vol.direction == 'FALLING'
            ? const Color(0xFF3FB950)
            : Colors.white70;

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
      decoration: BoxDecoration(
        color: const Color(0xFF161B22),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Row(
        children: [
          // Ticker dropdown
          _buildTickerDropdown(),
          const SizedBox(width: 12),
          // Event badges
          Expanded(child: _buildEventBadges()),
          // Vol badge
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
            decoration: BoxDecoration(
              color: volColor.withOpacity(0.2),
              borderRadius: BorderRadius.circular(6),
            ),
            child: Text(
              'VOL${vol.directionSymbol}',
              style: TextStyle(fontSize: 14, fontWeight: FontWeight.bold, color: volColor, fontFamily: 'JetBrains Mono'),
            ),
          ),
          const SizedBox(width: 12),
          GestureDetector(
            onTap: _loadData,
            child: const Icon(Icons.refresh, color: Colors.white54, size: 24),
          ),
        ],
      ),
    );
  }

  Widget _buildTickerDropdown() {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
      decoration: BoxDecoration(
        color: const Color(0xFF21262D),
        borderRadius: BorderRadius.circular(6),
      ),
      child: DropdownButton<String>(
        value: _selectedTicker,
        dropdownColor: const Color(0xFF21262D),
        underline: const SizedBox(),
        isDense: true,
        style: const TextStyle(color: Colors.white, fontFamily: 'JetBrains Mono', fontWeight: FontWeight.bold, fontSize: 18),
        items: _tickers.map((t) => DropdownMenuItem(value: t, child: Text(t))).toList(),
        onChanged: (v) {
          if (v != null && v != _selectedTicker) {
            setState(() => _selectedTicker = v);
            _loadData();
          }
        },
      ),
    );
  }

  Widget _buildEventBadges() {
    final badges = _events?.badges ?? [];
    if (badges.isEmpty) {
      return const Text('No events', style: TextStyle(fontSize: 14, color: Colors.white38, fontFamily: 'JetBrains Mono'));
    }

    return SingleChildScrollView(
      scrollDirection: Axis.horizontal,
      child: Row(
        children: badges.take(3).map((badge) {
          final color = Color(CockpitEvent(
            type: badge.type, title: '', date: '', time: '', impact: '', source: '',
          ).colorValue);
          
          return Container(
            margin: const EdgeInsets.only(right: 8),
            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
            decoration: BoxDecoration(
              color: color.withOpacity(0.2),
              borderRadius: BorderRadius.circular(6),
              border: Border.all(color: color.withOpacity(0.4)),
            ),
            child: Text(
              '${badge.text} ${badge.dateLabel}',
              style: TextStyle(fontSize: 13, fontWeight: FontWeight.bold, color: color, fontFamily: 'JetBrains Mono'),
            ),
          );
        }).toList(),
      ),
    );
  }

  /// Regime Card: Compact with all key info
  Widget _buildRegimeCard() {
    final regime = _state!.regime;
    final structure = _state!.structure;
    
    Color accentColor;
    if (regime.transition) {
      accentColor = const Color(0xFFFFA500);
    } else if (regime.isPositive) {
      accentColor = const Color(0xFF3FB950);
    } else if (regime.isNegative) {
      accentColor = const Color(0xFFF85149);
    } else {
      accentColor = const Color(0xFF8B949E);
    }

    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: accentColor.withOpacity(0.1),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: accentColor.withOpacity(0.3), width: 2),
      ),
      child: Column(
        children: [
          // Row 1: Regime badge + metrics
          Row(
            children: [
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
                decoration: BoxDecoration(
                  color: accentColor.withOpacity(0.3),
                  borderRadius: BorderRadius.circular(6),
                  border: Border.all(color: accentColor, width: 2),
                ),
                child: Text(
                  regime.label,
                  style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold, color: accentColor, fontFamily: 'JetBrains Mono'),
                ),
              ),
              if (regime.transition) ...[
                const SizedBox(width: 8),
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                  decoration: BoxDecoration(color: Colors.amber.withOpacity(0.3), borderRadius: BorderRadius.circular(4)),
                  child: const Text('⚠ NEAR FLIP', style: TextStyle(fontSize: 12, fontWeight: FontWeight.bold, color: Colors.amber)),
                ),
              ],
              const Spacer(),
              // GEX metrics
              Column(
                crossAxisAlignment: CrossAxisAlignment.end,
                children: [
                  Row(
                    children: [
                      Text('C: ${_formatGex(regime.callGex)}', style: const TextStyle(fontSize: 14, color: Color(0xFF3FB950), fontFamily: 'JetBrains Mono')),
                      const SizedBox(width: 12),
                      Text('P: ${_formatGex(regime.putGex)}', style: const TextStyle(fontSize: 14, color: Color(0xFFF85149), fontFamily: 'JetBrains Mono')),
                    ],
                  ),
                  if (regime.gexRatio != null)
                    Text('C/P: ${regime.gexRatio!.toStringAsFixed(2)}x', style: const TextStyle(fontSize: 13, color: Colors.white54, fontFamily: 'JetBrains Mono')),
                ],
              ),
            ],
          ),
          const SizedBox(height: 16),
          // Price Bar
          _buildPriceBar(regime, structure),
          const SizedBox(height: 16),
          // De-Pin Risk Widget
          if (_state!.depinRisk != null) ...[
            _buildDepinRiskWidget(_state!.depinRisk!),
            const SizedBox(height: 16),
          ],
          // Hero Bias
          _buildHeroBias(regime),
        ],
      ),
    );
  }

  Widget _buildPriceBar(RegimeState regime, StructureState structure) {
    final putWall = structure.primaryWalls.put;
    final callWall = structure.primaryWalls.call;
    final spot = regime.spot;
    final flip = regime.flipLine;

    if (putWall == null || callWall == null || spot == null) {
      return const SizedBox.shrink();
    }

    // Determine center point (flip line if available, otherwise spot)
    final center = flip ?? spot;
    
    // Create evenly spaced strikes: 20 below center, center, 20 above center
    final strikeIncrement = 1.0; // $1 strikes for SPY
    final numStrikesEachSide = 20;
    final minStrike = center - (numStrikesEachSide * strikeIncrement);
    final maxStrike = center + (numStrikesEachSide * strikeIncrement);
    
    // Generate strike list
    final strikes = <double>[];
    for (int i = 0; i <= numStrikesEachSide * 2; i++) {
      strikes.add(minStrike + (i * strikeIncrement));
    }
    
    // Calculate positions
    final range = maxStrike - minStrike;
    double getPosition(double price) => ((price - minStrike) / range).clamp(0.0, 1.0);
    
    // Calculate distances
    final putDistance = spot - putWall;
    final callDistance = callWall - spot;
    final flipDistance = flip != null ? (flip - spot) : null;

    return Column(
      children: [
        const SizedBox(height: 8),
        // Price line graph
        SizedBox(
          height: 120,
          child: LayoutBuilder(
            builder: (context, constraints) {
              final width = constraints.maxWidth;
              final height = constraints.maxHeight;
              
              // Calculate positions
              final spotPos = getPosition(spot);
              final putWallPos = getPosition(putWall);
              final callWallPos = getPosition(callWall);
              final flipPos = flip != null ? getPosition(flip) : null;
              final minPos = 0.0;
              final maxPos = 1.0;
              
              return Stack(
                children: [
                  // Background grid lines (strike ticks)
                  ...strikes.map((strike) {
                    final pos = getPosition(strike);
                    // Only show tick marks, not labels for most strikes
                    final isKeyLevel = (strike == minStrike || strike == maxStrike || 
                                       (flip != null && (strike - flip).abs() < 0.5) ||
                                       (strike - putWall).abs() < 0.5 ||
                                       (strike - callWall).abs() < 0.5 ||
                                       (strike - spot).abs() < 0.5);
                    
                    return Positioned(
                      left: pos * width,
                      child: Container(
                        width: 1,
                        height: isKeyLevel ? height : 8,
                        color: Colors.white.withOpacity(isKeyLevel ? 0.3 : 0.1),
                      ),
                    );
                  }).toList(),
                  
                  // Horizontal price line
                  Positioned(
                    top: height / 2 - 1,
                    child: Container(
                      width: width,
                      height: 2,
                      color: Colors.white.withOpacity(0.2),
                    ),
                  ),
                  
                  // Put Wall marker
                  Positioned(
                    left: putWallPos * width - 2,
                    top: 0,
                    child: Column(
                      children: [
                        // Price label on the line
                        Container(
                          padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 2),
                          decoration: BoxDecoration(
                            color: const Color(0xFFF85149).withOpacity(0.9),
                            borderRadius: BorderRadius.circular(3),
                          ),
                          child: Text(
                            putWall.toStringAsFixed(0),
                            style: const TextStyle(fontSize: 11, fontWeight: FontWeight.bold, color: Colors.white, fontFamily: 'JetBrains Mono'),
                          ),
                        ),
                        const SizedBox(height: 2),
                        Container(
                          width: 4,
                          height: height - 30,
                          decoration: BoxDecoration(
                            color: const Color(0xFFF85149),
                            borderRadius: BorderRadius.circular(2),
                          ),
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
                              Text('(${putDistance.toStringAsFixed(1)} pts)', style: const TextStyle(fontSize: 8, color: Colors.white54, fontFamily: 'JetBrains Mono')),
                            ],
                          ),
                        ),
                      ],
                    ),
                  ),
                  
                  // Flip line marker
                  if (flipPos != null)
                    Positioned(
                      left: flipPos * width - 2,
                      top: 0,
                      child: Column(
                        children: [
                          // Price label on the line
                          Container(
                            padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 2),
                            decoration: BoxDecoration(
                              color: const Color(0xFFFFA500).withOpacity(0.9),
                              borderRadius: BorderRadius.circular(3),
                            ),
                            child: Text(
                              flip!.toStringAsFixed(1),
                              style: const TextStyle(fontSize: 11, fontWeight: FontWeight.bold, color: Colors.white, fontFamily: 'JetBrains Mono'),
                            ),
                          ),
                          const SizedBox(height: 2),
                          Container(
                            width: 4,
                            height: height - 30,
                            decoration: BoxDecoration(
                              color: const Color(0xFFFFA500),
                              borderRadius: BorderRadius.circular(2),
                            ),
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
                                Text('(${flipDistance!.toStringAsFixed(1)} pts)', style: const TextStyle(fontSize: 8, color: Colors.white54, fontFamily: 'JetBrains Mono')),
                              ],
                            ),
                          ),
                        ],
                      ),
                    ),
                  
                  // Spot marker
                  Positioned(
                    left: spotPos * width - 15,
                    top: height / 2 - 15,
                    child: Column(
                      children: [
                        Container(
                          width: 30,
                          height: 30,
                          decoration: BoxDecoration(
                            color: Colors.white,
                            shape: BoxShape.circle,
                            border: Border.all(color: const Color(0xFF0D1117), width: 3),
                            boxShadow: [BoxShadow(color: Colors.white.withOpacity(0.4), blurRadius: 6)],
                          ),
                          child: const Center(
                            child: Text('●', style: TextStyle(fontSize: 12, color: Color(0xFF0D1117))),
                          ),
                        ),
                        const SizedBox(height: 4),
                        Container(
                          padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                          decoration: BoxDecoration(
                            color: Colors.white.withOpacity(0.2),
                            borderRadius: BorderRadius.circular(4),
                            border: Border.all(color: Colors.white),
                          ),
                          child: Column(
                            children: [
                              Text(spot.toStringAsFixed(2), style: const TextStyle(fontSize: 12, fontWeight: FontWeight.bold, color: Colors.white, fontFamily: 'JetBrains Mono')),
                              const Text('SPOT', style: TextStyle(fontSize: 9, color: Colors.white70, fontFamily: 'JetBrains Mono')),
                            ],
                          ),
                        ),
                      ],
                    ),
                  ),
                  
                  // Call Wall marker
                  Positioned(
                    left: callWallPos * width - 2,
                    top: 0,
                    child: Column(
                      children: [
                        // Price label on the line
                        Container(
                          padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 2),
                          decoration: BoxDecoration(
                            color: const Color(0xFF3FB950).withOpacity(0.9),
                            borderRadius: BorderRadius.circular(3),
                          ),
                          child: Text(
                            callWall.toStringAsFixed(0),
                            style: const TextStyle(fontSize: 11, fontWeight: FontWeight.bold, color: Colors.white, fontFamily: 'JetBrains Mono'),
                          ),
                        ),
                        const SizedBox(height: 2),
                        Container(
                          width: 4,
                          height: height - 30,
                          decoration: BoxDecoration(
                            color: const Color(0xFF3FB950),
                            borderRadius: BorderRadius.circular(2),
                          ),
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
                              Text('(${callDistance.toStringAsFixed(1)} pts)', style: const TextStyle(fontSize: 8, color: Colors.white54, fontFamily: 'JetBrains Mono')),
                            ],
                          ),
                        ),
                      ],
                    ),
                  ),
                  
                  // Min/Max strike labels at ends
                  Positioned(
                    left: 0,
                    top: height / 2 - 8,
                    child: Text(
                      minStrike.toStringAsFixed(0),
                      style: TextStyle(fontSize: 10, color: Colors.white.withOpacity(0.5), fontFamily: 'JetBrains Mono'),
                    ),
                  ),
                  Positioned(
                    right: 0,
                    top: height / 2 - 8,
                    child: Text(
                      maxStrike.toStringAsFixed(0),
                      style: TextStyle(fontSize: 10, color: Colors.white.withOpacity(0.5), fontFamily: 'JetBrains Mono'),
                    ),
                  ),
                ],
              );
            },
          ),
        ),
      ],
    );
  }

  Widget _buildHeroBias(RegimeState regime) {
    String heroText;
    IconData heroIcon;
    
    if (regime.isNegative) {
      heroText = 'FOLLOW BREAKS • DON\'T FADE';
      heroIcon = Icons.trending_up;
    } else if (regime.isPositive) {
      heroText = 'SELL PREMIUM • FADE EXTREMES';
      heroIcon = Icons.swap_horiz;
    } else {
      heroText = 'WAIT FOR CLARITY';
      heroIcon = Icons.hourglass_empty;
    }

    return Container(
      padding: const EdgeInsets.symmetric(vertical: 10, horizontal: 16),
      decoration: BoxDecoration(
        color: Colors.black26,
        borderRadius: BorderRadius.circular(8),
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(heroIcon, color: Colors.white.withOpacity(0.7), size: 22),
          const SizedBox(width: 10),
          Text(
            heroText,
            style: const TextStyle(fontSize: 18, fontWeight: FontWeight.w900, color: Colors.white, fontFamily: 'JetBrains Mono', letterSpacing: 0.5),
          ),
        ],
      ),
    );
  }

  Widget _buildDepinRiskWidget(DepinRiskData depinRisk) {
    // Color based on band
    Color bandColor;
    if (depinRisk.band == 'LOW') {
      bandColor = const Color(0xFF3FB950);  // Green
    } else if (depinRisk.band == 'MID') {
      bandColor = const Color(0xFFFFA500);  // Orange
    } else {
      bandColor = const Color(0xFFF85149);  // Red
    }

    // Delta color and symbol
    Color deltaColor = Colors.white70;
    String deltaSymbol = '';
    String deltaText = '';
    if (depinRisk.delta30m != null && depinRisk.deltaDirection != null) {
      if (depinRisk.deltaDirection == 'up') {
        deltaColor = const Color(0xFFF85149);  // Red (bad)
        deltaSymbol = '↑';
      } else if (depinRisk.deltaDirection == 'down') {
        deltaColor = const Color(0xFF3FB950);  // Green (good)
        deltaSymbol = '↓';
      } else {
        deltaSymbol = '→';
      }
      deltaText = 'Δ: $deltaSymbol ${depinRisk.delta30m!.abs()} (30m)';
    }

    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: bandColor.withOpacity(0.1),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: bandColor.withOpacity(0.3), width: 2),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Header: DE-PIN RISK
          Row(
            children: [
              const Text('DE-PIN RISK', style: TextStyle(fontSize: 14, fontWeight: FontWeight.bold, color: Color(0xFF8B949E), fontFamily: 'JetBrains Mono')),
              const Spacer(),
              if (deltaText.isNotEmpty)
                Text(deltaText, style: TextStyle(fontSize: 12, color: deltaColor, fontFamily: 'JetBrains Mono')),
            ],
          ),
          const SizedBox(height: 12),
          // Thermometer/Gauge
          Row(
            children: [
              // Score and band
              Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    '${depinRisk.score}',
                    style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold, color: bandColor, fontFamily: 'JetBrains Mono'),
                  ),
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                    decoration: BoxDecoration(
                      color: bandColor,
                      borderRadius: BorderRadius.circular(4),
                    ),
                    child: Text(
                      depinRisk.band,
                      style: const TextStyle(fontSize: 12, fontWeight: FontWeight.bold, color: Colors.white, fontFamily: 'JetBrains Mono'),
                    ),
                  ),
                ],
              ),
              const SizedBox(width: 16),
              // Progress bar
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    ClipRRect(
                      borderRadius: BorderRadius.circular(4),
                      child: Container(
                        height: 20,
                        decoration: BoxDecoration(
                          color: Colors.white.withOpacity(0.1),
                          borderRadius: BorderRadius.circular(4),
                        ),
                        child: FractionallySizedBox(
                          widthFactor: depinRisk.score / 100,
                          alignment: Alignment.centerLeft,
                          child: Container(
                            decoration: BoxDecoration(
                              color: bandColor,
                              borderRadius: BorderRadius.circular(4),
                            ),
                          ),
                        ),
                      ),
                    ),
                    const SizedBox(height: 4),
                    const Text('0', style: TextStyle(fontSize: 10, color: Colors.white54, fontFamily: 'JetBrains Mono')),
                  ],
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          // Guidance text
          Text(
            depinRisk.guidance,
            style: TextStyle(fontSize: 14, color: bandColor, fontFamily: 'JetBrains Mono'),
          ),
          if (depinRisk.drivers.isNotEmpty) ...[
            const SizedBox(height: 12),
            // Driver chips
            Wrap(
              spacing: 8,
              runSpacing: 8,
              children: depinRisk.drivers.map((driver) {
                // Color based on contribution sign
                final driverColor = driver.contribution >= 0 
                    ? const Color(0xFFF85149)  // Red for destabilizing
                    : const Color(0xFF3FB950);  // Green for stabilizing
                final sign = driver.contribution >= 0 ? '+' : '';
                
                return Container(
                  padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
                  decoration: BoxDecoration(
                    color: driverColor.withOpacity(0.2),
                    borderRadius: BorderRadius.circular(6),
                    border: Border.all(color: driverColor.withOpacity(0.4)),
                  ),
                  child: Text(
                    '${driver.name}: $sign${driver.contribution.toStringAsFixed(2)}',
                    style: TextStyle(fontSize: 11, fontWeight: FontWeight.w500, color: driverColor, fontFamily: 'JetBrains Mono'),
                  ),
                );
              }).toList(),
            ),
          ],
        ],
      ),
    );
  }

  /// Structure Card
  Widget _buildStructureCard() {
    final structure = _state!.structure;
    final zoneColor = _getZoneColor(structure.zoneDescription?.type ?? '');

    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: const Color(0xFF161B22),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Column(
        children: [
          // Header
          Row(
            children: [
              const Text('STRUCTURE', style: TextStyle(fontSize: 14, fontWeight: FontWeight.bold, color: Color(0xFF8B949E), fontFamily: 'JetBrains Mono')),
              const Spacer(),
              if (structure.zoneDescription != null && structure.zoneDescription!.type.isNotEmpty)
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                  decoration: BoxDecoration(color: zoneColor.withOpacity(0.2), borderRadius: BorderRadius.circular(4), border: Border.all(color: zoneColor)),
                  child: Text(
                    structure.zoneDescription!.type.replaceAll('_', ' '),
                    style: TextStyle(fontSize: 12, fontWeight: FontWeight.bold, color: zoneColor, fontFamily: 'JetBrains Mono'),
                  ),
                ),
            ],
          ),
          const SizedBox(height: 12),
          // Walls
          Row(
            children: [
              _buildWallChip('PUT', structure.primaryWalls.put, structure.oiPrimaryWalls.put, const Color(0xFFF85149)),
              const Spacer(),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                decoration: BoxDecoration(color: Colors.white.withOpacity(0.1), borderRadius: BorderRadius.circular(4)),
                child: Text('Range: ${structure.wallRange?.toStringAsFixed(0) ?? '--'}pts', style: const TextStyle(fontSize: 14, color: Colors.white70, fontFamily: 'JetBrains Mono')),
              ),
              const Spacer(),
              _buildWallChip('CALL', structure.primaryWalls.call, structure.oiPrimaryWalls.call, const Color(0xFF3FB950)),
            ],
          ),
          if (structure.zoneDescription != null && structure.zoneDescription!.behavior.isNotEmpty) ...[
            const SizedBox(height: 10),
            Text(
              '→ ${structure.zoneDescription!.behavior}',
              style: const TextStyle(fontSize: 14, color: Colors.white70, fontFamily: 'JetBrains Mono'),
              textAlign: TextAlign.center,
            ),
          ],
        ],
      ),
    );
  }

  Widget _buildWallChip(String label, double? gexWall, double? oiWall, Color color) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
          decoration: BoxDecoration(color: color.withOpacity(0.2), borderRadius: BorderRadius.circular(6)),
          child: Text('$label: ${gexWall?.toStringAsFixed(0) ?? '--'}', style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold, color: color, fontFamily: 'JetBrains Mono')),
        ),
        const SizedBox(height: 2),
        Text('  OI: ${oiWall?.toStringAsFixed(0) ?? '--'}', style: const TextStyle(fontSize: 12, color: Colors.white38, fontFamily: 'JetBrains Mono')),
      ],
    );
  }

  Color _getZoneColor(String zoneType) {
    switch (zoneType) {
      case 'FLIP_IN_RANGE':
      case 'BELOW_FLIP':
        return Colors.orange;
      case 'ABOVE_FLIP':
        return const Color(0xFF3FB950);
      case 'TIGHT':
        return Colors.amber;
      default:
        return const Color(0xFF8B949E);
    }
  }

  /// Action Chips
  Widget _buildActionChips() {
    final filter = _state!.actionFilter;
    
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: const Color(0xFF161B22),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text('ACTION FILTER', style: TextStyle(fontSize: 14, fontWeight: FontWeight.bold, color: Color(0xFF8B949E), fontFamily: 'JetBrains Mono')),
          const SizedBox(height: 10),
          Wrap(
            spacing: 8,
            runSpacing: 8,
            children: [
              ...filter.allowed.take(4).map((a) => _buildChip(a, true)),
              ...filter.forbidden.take(4).map((f) => _buildChip(f, false)),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildChip(String text, bool isAllowed) {
    final color = isAllowed ? const Color(0xFF3FB950) : const Color(0xFFF85149);
    final prefix = isAllowed ? '✓' : '✗';
    final shortText = text.length > 16 ? '${text.substring(0, 14)}...' : text;
    
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: color.withOpacity(0.12),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: color.withOpacity(0.3)),
      ),
      child: Text('$prefix $shortText', style: TextStyle(fontSize: 13, fontWeight: FontWeight.w600, color: color, fontFamily: 'JetBrains Mono')),
    );
  }

  /// Events Section
  Widget _buildEventsSection() {
    final events = _events?.events ?? [];
    
    return Container(
      decoration: BoxDecoration(
        color: const Color(0xFF161B22),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Column(
        children: [
          // Header
          GestureDetector(
            onTap: () => setState(() => _eventsExpanded = !_eventsExpanded),
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
              child: Row(
                children: [
                  Icon(_eventsExpanded ? Icons.expand_less : Icons.expand_more, color: Colors.white54, size: 22),
                  const SizedBox(width: 8),
                  Text('UPCOMING EVENTS (${events.length})', style: const TextStyle(fontSize: 14, fontWeight: FontWeight.bold, color: Color(0xFF8B949E), fontFamily: 'JetBrains Mono')),
                  const Spacer(),
                  if (!_eventsExpanded && events.isNotEmpty)
                    Text(
                      events.take(2).map((e) => e.title.length > 10 ? '${e.title.substring(0, 8)}…' : e.title).join(' | '),
                      style: const TextStyle(fontSize: 12, color: Colors.white38, fontFamily: 'JetBrains Mono'),
                    ),
                ],
              ),
            ),
          ),
          // Expanded list
          if (_eventsExpanded)
            Container(
              constraints: const BoxConstraints(maxHeight: 200),
              child: ListView.builder(
                shrinkWrap: true,
                padding: const EdgeInsets.only(bottom: 8),
                itemCount: events.length,
                itemBuilder: (context, index) {
                  final event = events[index];
                  final color = Color(event.colorValue);
                  
                  return Container(
                    padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
                    child: Row(
                      children: [
                        Container(width: 4, height: 24, decoration: BoxDecoration(color: color, borderRadius: BorderRadius.circular(2))),
                        const SizedBox(width: 10),
                        SizedBox(width: 60, child: Text(_formatEventDate(event.date), style: const TextStyle(fontSize: 13, color: Colors.white54, fontFamily: 'JetBrains Mono'))),
                        Expanded(child: Text(event.title, style: const TextStyle(fontSize: 14, color: Colors.white70, fontFamily: 'JetBrains Mono'), overflow: TextOverflow.ellipsis)),
                        if (event.time.isNotEmpty)
                          Container(
                            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                            decoration: BoxDecoration(color: color.withOpacity(0.2), borderRadius: BorderRadius.circular(4)),
                            child: Text(event.time, style: TextStyle(fontSize: 12, fontWeight: FontWeight.bold, color: color, fontFamily: 'JetBrains Mono')),
                          ),
                      ],
                    ),
                  );
                },
              ),
            ),
        ],
      ),
    );
  }

  String _formatEventDate(String dateStr) {
    try {
      final date = DateTime.parse(dateStr);
      final now = DateTime.now();
      final diff = date.difference(DateTime(now.year, now.month, now.day)).inDays;
      
      if (diff == 0) return 'Today';
      if (diff == 1) return 'Tmrw';
      
      final weekdays = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
      return '${weekdays[date.weekday - 1]} ${date.day}';
    } catch (e) {
      return dateStr;
    }
  }

  String _formatGex(double? gex) {
    if (gex == null) return '--';
    final absGex = gex.abs();
    if (absGex >= 1e12) return '${(gex / 1e12).toStringAsFixed(1)}T';
    if (absGex >= 1e9) return '${(gex / 1e9).toStringAsFixed(1)}B';
    if (absGex >= 1e6) return '${(gex / 1e6).toStringAsFixed(1)}M';
    return gex.toStringAsFixed(0);
  }
}
