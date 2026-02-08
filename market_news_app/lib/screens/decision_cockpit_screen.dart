import 'package:flutter/material.dart';
import '../services/cockpit_service.dart';
import '../services/gex_service.dart';
import '../models/gex_data.dart';
import '../widgets/asset_selector_widget.dart';
import '../widgets/asset_selection_provider.dart';
import '../widgets/gex_chart_widget.dart';
import '../widgets/gex_price_bar_widget.dart';

/// Decision Cockpit - Compact Single-Screen Trading State View
class DecisionCockpitScreen extends StatefulWidget {
  const DecisionCockpitScreen({super.key});

  @override
  State<DecisionCockpitScreen> createState() => _DecisionCockpitScreenState();
}

class _DecisionCockpitScreenState extends State<DecisionCockpitScreen> {
  CockpitState? _state;
  CockpitEventsData? _events;
  GexCalculation? _gexChartData;
  bool _isLoading = true;
  String? _error;
  bool _eventsExpanded = false;

  @override
  void initState() {
    super.initState();
    _loadData();
  }
  
  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    // Set up listener after context is available
    final service = AssetSelectionProvider.of(context);
    service.addListener(_onAssetChanged);
  }
  
  @override
  void dispose() {
    try {
      final service = AssetSelectionProvider.of(context);
      service.removeListener(_onAssetChanged);
    } catch (e) {
      // Context might not be available during dispose
    }
    super.dispose();
  }
  
  void _onAssetChanged() {
    // Reload data when asset changes
    if (mounted) {
      _loadData();
    }
  }
  
  String get _selectedTicker {
    try {
      return AssetSelectionProvider.of(context).selectedAsset;
    } catch (e) {
      return 'SPY'; // Fallback
    }
  }

  Future<void> _loadData() async {
    setState(() {
      _isLoading = true;
      _error = null;
    });

    try {
      final results = await Future.wait([
        CockpitService.getCockpitState(_selectedTicker),
        CockpitService.getCockpitEvents(daysAhead: 14, symbol: _selectedTicker),
        GexService.calculateGex(_selectedTicker),
      ]);
      
      setState(() {
        _state = results[0] as CockpitState?;
        _events = results[1] as CockpitEventsData?;
        _gexChartData = results[2] as GexCalculation?;
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
            // Block A: Price strip
            _buildPriceStrip(),
            const SizedBox(height: 12),
            // Action filter — full width, above graph
            SizedBox(
              width: double.infinity,
              child: _buildActionChips(),
            ),
            const SizedBox(height: 12),
            // Regime Card: graph + strategy sections (De-Pin, Volatility, Hero Bias) just below
            _buildRegimeCard(),
            const SizedBox(height: 12),
            // Events (today & tomorrow with impact)
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
          // Ticker dropdown + symbol input — need enough width so dropdown isn't squished
          SizedBox(
            width: 280,
            child: _buildTickerDropdown(),
          ),
          const SizedBox(width: 12),
          // Event badges - flexible, scrollable
          Flexible(
            flex: 2,
            child: _buildEventBadges(),
          ),
          const SizedBox(width: 8),
          // Vol badge - fixed width, no shrink
          Flexible(
            flex: 0,
            child: Container(
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
          ),
          const SizedBox(width: 8),
          // Refresh icon - fixed width, no shrink
          Flexible(
            flex: 0,
            child: GestureDetector(
              onTap: _loadData,
              child: const Icon(Icons.refresh, color: Colors.white54, size: 24),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildTickerDropdown() {
    return AssetSelectorWidget(
      compact: true,
      showQuickButtons: false,
      onAssetChanged: (asset) {
        _loadData();
      },
    );
  }

  /// Block A: Price strip — current | prev close | today open
  Widget _buildPriceStrip() {
    final quote = _state!.quote;
    final spot = _state!.regime.spot;
    if (quote == null && spot == null) return const SizedBox.shrink();
    final current = quote?.current ?? spot;
    final prev = quote?.previousClose;
    final open = quote?.open;
    if (current == null) return const SizedBox.shrink();
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      decoration: BoxDecoration(
        color: const Color(0xFF161B22),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceAround,
        children: [
          Column(
            crossAxisAlignment: CrossAxisAlignment.center,
            children: [
              const Text('Current', style: TextStyle(fontSize: 11, color: Colors.white54, fontFamily: 'JetBrains Mono')),
              const SizedBox(height: 4),
              Text(
                current.toStringAsFixed(2),
                style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold, color: Colors.white, fontFamily: 'JetBrains Mono'),
              ),
              if (quote?.change != null && quote?.changePct != null) ...[
                const SizedBox(height: 2),
                Text(
                  '${quote!.change! >= 0 ? '+' : ''}${quote.change!.toStringAsFixed(2)} (${quote.changePct! >= 0 ? '+' : ''}${quote.changePct!.toStringAsFixed(2)}%)',
                  style: TextStyle(fontSize: 12, color: quote.change! >= 0 ? const Color(0xFF3FB950) : const Color(0xFFF85149), fontFamily: 'JetBrains Mono'),
                ),
              ],
            ],
          ),
          if (prev != null)
            Column(
              crossAxisAlignment: CrossAxisAlignment.center,
              children: [
                const Text('Prev close', style: TextStyle(fontSize: 11, color: Colors.white54, fontFamily: 'JetBrains Mono')),
                const SizedBox(height: 4),
                Text(prev.toStringAsFixed(2), style: const TextStyle(fontSize: 16, color: Colors.white70, fontFamily: 'JetBrains Mono')),
              ],
            ),
          if (open != null)
            Column(
              crossAxisAlignment: CrossAxisAlignment.center,
              children: [
                const Text('Open', style: TextStyle(fontSize: 11, color: Colors.white54, fontFamily: 'JetBrains Mono')),
                const SizedBox(height: 4),
                Text(open.toStringAsFixed(2), style: const TextStyle(fontSize: 16, color: Colors.white70, fontFamily: 'JetBrains Mono')),
              ],
            ),
        ],
      ),
    );
  }

  /// Block C: Volatility strategy card
  Widget _buildVolatilityStrategyCard() {
    final vol = _state!.volatility;
    final vs = _state!.volatilityStrategy;
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFF161B22),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Colors.white.withOpacity(0.08)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text('VOLATILITY STRATEGY', style: TextStyle(fontSize: 12, fontWeight: FontWeight.bold, color: Color(0xFF8B949E), fontFamily: 'JetBrains Mono')),
          const SizedBox(height: 12),
          Row(
            children: [
              if (vol.vix != null)
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
                  decoration: BoxDecoration(
                    color: const Color(0xFF58A6FF).withOpacity(0.2),
                    borderRadius: BorderRadius.circular(6),
                  ),
                  child: Text('VIX ${vol.vix!.toStringAsFixed(1)}${vol.vixChange != null ? (vol.vixChange! >= 0 ? ' ↑' : ' ↓') : ''}', style: const TextStyle(fontSize: 14, fontWeight: FontWeight.bold, color: Color(0xFF58A6FF), fontFamily: 'JetBrains Mono')),
                ),
              const SizedBox(width: 8),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                decoration: BoxDecoration(color: Colors.white.withOpacity(0.1), borderRadius: BorderRadius.circular(4)),
                child: Text(vol.termStructure, style: const TextStyle(fontSize: 12, color: Colors.white70, fontFamily: 'JetBrains Mono')),
              ),
              const SizedBox(width: 8),
              Text(vol.direction, style: const TextStyle(fontSize: 12, color: Colors.white54, fontFamily: 'JetBrains Mono')),
            ],
          ),
          if (vs != null && vs.headline.isNotEmpty) ...[
            const SizedBox(height: 12),
            Text(vs.headline, style: const TextStyle(fontSize: 16, fontWeight: FontWeight.bold, color: Colors.white, fontFamily: 'JetBrains Mono')),
            if (vs.rationale.isNotEmpty)
              Text(vs.rationale, style: const TextStyle(fontSize: 13, color: Colors.white70, fontFamily: 'JetBrains Mono')),
          ],
        ],
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
        mainAxisSize: MainAxisSize.min,
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
        color: accentColor.withOpacity(0.08),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: accentColor.withOpacity(0.25), width: 1.5),
      ),
      child: Column(
        children: [
          // Row 1: Regime badge + metrics
          Row(
            children: [
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                decoration: BoxDecoration(
                  color: accentColor.withOpacity(0.2),
                  borderRadius: BorderRadius.circular(6),
                  border: Border.all(color: accentColor, width: 1.5),
                ),
                child: Text(
                  regime.label,
                  style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold, color: accentColor, fontFamily: 'JetBrains Mono'),
                ),
              ),
              if (regime.transition) ...[
                const SizedBox(width: 8),
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 3),
                  decoration: BoxDecoration(
                    color: Colors.amber.withOpacity(0.15),
                    borderRadius: BorderRadius.circular(4),
                    border: Border.all(color: Colors.amber.withOpacity(0.3), width: 1),
                  ),
                  child: const Text(
                    '⚠ NEAR FLIP',
                    style: TextStyle(fontSize: 11, fontWeight: FontWeight.w600, color: Colors.amber, fontFamily: 'JetBrains Mono'),
                  ),
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
          // Chart title: ticker + "GEX by Strike"
          Text(
            '$_selectedTicker GEX by Strike',
            style: const TextStyle(
              fontSize: 13,
              fontWeight: FontWeight.w600,
              color: Colors.white54,
              fontFamily: 'JetBrains Mono',
            ),
          ),
          const SizedBox(height: 8),
          // GEX chart (from GEX tab) or fallback price bar
          if (_gexChartData != null && _gexChartData!.gexByStrike.isNotEmpty)
            GexChartWidget(
              data: _gexChartData!.gexByStrike,
              annotations: _gexChartData!.chartAnnotations,
              cumulativeGex: _gexChartData!.cumulativeGex,
              height: 280,
              maxPainStrike: _state!.maxPain != null && _state!.maxPain!.strike > 0 ? _state!.maxPain!.strike : null,
            )
          else
            GexPriceBarWidget(
              spot: regime.spot,
              flipLine: regime.flipLine,
              putWall: structure.primaryWalls.put,
              callWall: structure.primaryWalls.call,
              darkBackground: true,
            ),
          const SizedBox(height: 16),
          // De-Pin Risk Widget
          if (_state!.depinRisk != null) ...[
            _buildDepinRiskWidget(_state!.depinRisk!),
            const SizedBox(height: 16),
          ],
          // Volatility strategy (just under graph, next to Hero Bias)
          _buildVolatilityStrategyCard(),
          const SizedBox(height: 16),
          // Hero Bias
          _buildHeroBias(regime),
        ],
      ),
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
      padding: const EdgeInsets.symmetric(vertical: 8, horizontal: 14),
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.05),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: Colors.white.withOpacity(0.1)),
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(heroIcon, color: Colors.white.withOpacity(0.6), size: 18),
          const SizedBox(width: 8),
          Text(
            heroText,
            style: const TextStyle(fontSize: 14, fontWeight: FontWeight.w600, color: Colors.white70, fontFamily: 'JetBrains Mono', letterSpacing: 0.3),
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

  /// Action Chips
  Widget _buildActionChips() {
    final filter = _state!.actionFilter;
    
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFF161B22),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text('ACTION FILTER', style: TextStyle(fontSize: 14, fontWeight: FontWeight.bold, color: Color(0xFF8B949E), fontFamily: 'JetBrains Mono')),
          const SizedBox(height: 12),
          // Allowed filters
          if (filter.allowed.isNotEmpty) ...[
            Wrap(
              spacing: 8,
              runSpacing: 8,
              children: filter.allowed.map((a) => _buildChip(a, true)).toList(),
            ),
            if (filter.forbidden.isNotEmpty) const SizedBox(height: 12),
          ],
          // Forbidden filters
          if (filter.forbidden.isNotEmpty)
            Wrap(
              spacing: 8,
              runSpacing: 8,
              children: filter.forbidden.map((f) => _buildChip(f, false)).toList(),
            ),
        ],
      ),
    );
  }

  Widget _buildChip(String text, bool isAllowed) {
    final color = isAllowed ? const Color(0xFF3FB950) : const Color(0xFFF85149);
    final prefix = isAllowed ? '✓' : '✗';
    
    return Tooltip(
      message: text,
      child: Container(
        constraints: const BoxConstraints(maxWidth: 200),
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
        decoration: BoxDecoration(
          color: color.withOpacity(0.12),
          borderRadius: BorderRadius.circular(20),
          border: Border.all(color: color.withOpacity(0.3)),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Text(prefix, style: TextStyle(fontSize: 13, fontWeight: FontWeight.bold, color: color, fontFamily: 'JetBrains Mono')),
            const SizedBox(width: 6),
            Flexible(
              child: Text(
                text,
                style: TextStyle(fontSize: 13, fontWeight: FontWeight.w600, color: color, fontFamily: 'JetBrains Mono'),
                overflow: TextOverflow.ellipsis,
                maxLines: 1,
              ),
            ),
          ],
        ),
      ),
    );
  }

  /// Block E: Events — today & tomorrow with impact on symbol
  Widget _buildEventsSection() {
    final allEvents = _events?.events ?? [];
    final now = DateTime.now();
    final today = DateTime(now.year, now.month, now.day);
    final tomorrow = today.add(const Duration(days: 1));
    final events = allEvents.where((e) {
      try {
        final d = DateTime.parse(e.date);
        final eventDay = DateTime(d.year, d.month, d.day);
        return eventDay == today || eventDay == tomorrow;
      } catch (_) {
        return true;
      }
    }).toList();
    
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
                  Text('TODAY & TOMORROW (${events.length})', style: const TextStyle(fontSize: 14, fontWeight: FontWeight.bold, color: Color(0xFF8B949E), fontFamily: 'JetBrains Mono')),
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
                        if (event.impactOnSymbol != null && event.impactOnSymbol!.isNotEmpty)
                          Container(
                            margin: const EdgeInsets.only(right: 8),
                            padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                            decoration: BoxDecoration(
                              color: event.impactOnSymbol == 'high' ? const Color(0xFFF85149).withOpacity(0.2) : event.impactOnSymbol == 'medium' ? const Color(0xFFFFA500).withOpacity(0.2) : Colors.white.withOpacity(0.1),
                              borderRadius: BorderRadius.circular(4),
                            ),
                            child: Text(event.impactOnSymbol!, style: TextStyle(fontSize: 10, fontWeight: FontWeight.bold, color: event.impactOnSymbol == 'high' ? const Color(0xFFF85149) : event.impactOnSymbol == 'medium' ? const Color(0xFFFFA500) : Colors.white54, fontFamily: 'JetBrains Mono')),
                          ),
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
