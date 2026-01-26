import 'dart:async';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import '../models/trade_idea.dart';
import '../services/trade_ideas_service.dart';
import '../widgets/asset_selector_widget.dart';
import '../widgets/asset_selection_provider.dart';
import 'decision_cockpit_screen.dart';

/// Trade Ideas Screen - Regime-Aware Allowed-Only Trade Ideas
class TradeIdeasScreen extends StatefulWidget {
  const TradeIdeasScreen({super.key});

  @override
  State<TradeIdeasScreen> createState() => _TradeIdeasScreenState();
}

class _TradeIdeasScreenState extends State<TradeIdeasScreen> {
  List<TradeIdea> _ideas = [];
  Map<String, List<TradeIdea>>? _ideasByTimeframe;
  List<TradeIdea> _previewIdeas = [];
  Map<String, dynamic>? _nextWindow;
  String? _currentPhase;
  bool _isLoading = true;
  String? _error;
  String _selectedTimeframe = 'all';  // 'all', 'thisWeek', 'thisMonth', 'thisYear'
  final List<String> _timeframes = ['all', 'thisWeek', 'thisMonth', 'thisYear'];
  int? _customMinDte;
  int? _customMaxDte;
  bool _useCustomDte = false;
  final Map<String, bool> _expandedCards = {};
  Map<String, dynamic>? _diagnostics;

  @override
  void initState() {
    super.initState();
    // #region agent log
    final logData = {
      'sessionId': 'debug-session',
      'runId': 'run1',
      'hypothesisId': 'C',
      'location': 'trade_ideas_screen.dart:37',
      'message': 'initState called',
      'data': {'timestamp': DateTime.now().millisecondsSinceEpoch},
      'timestamp': DateTime.now().millisecondsSinceEpoch
    };
    http.post(
      Uri.parse('http://127.0.0.1:7242/ingest/f2e5acba-93e2-4270-a633-fee604b9e81c'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode(logData),
    ).catchError((_) {});
    // #endregion
    _loadTradeIdeas();
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
    // Reload trade ideas when asset changes
    if (mounted) {
      _loadTradeIdeas();
    }
  }
  
  String get _selectedTicker {
    try {
      return AssetSelectionProvider.of(context).selectedAsset;
    } catch (e) {
      return 'SPY'; // Fallback
    }
  }

  Future<void> _loadTradeIdeas() async {
    // #region agent log
    final logData1 = {
      'sessionId': 'debug-session',
      'runId': 'run1',
      'hypothesisId': 'C',
      'location': 'trade_ideas_screen.dart:74',
      'message': '_loadTradeIdeas called',
      'data': {'ticker': _selectedTicker, 'timeframe': _selectedTimeframe, 'timestamp': DateTime.now().millisecondsSinceEpoch},
      'timestamp': DateTime.now().millisecondsSinceEpoch
    };
    http.post(
      Uri.parse('http://127.0.0.1:7242/ingest/f2e5acba-93e2-4270-a633-fee604b9e81c'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode(logData1),
    ).catchError((_) {});
    // #endregion
    
    setState(() {
      _isLoading = true;
      _error = null;
      _diagnostics = null;
      _ideas = [];
      _ideasByTimeframe = null;
    });

    try {
      // #region agent log
      final logData2 = {
        'sessionId': 'debug-session',
        'runId': 'run1',
        'hypothesisId': 'D',
        'location': 'trade_ideas_screen.dart:91',
        'message': 'API call starting',
        'data': {'ticker': _selectedTicker, 'timestamp': DateTime.now().millisecondsSinceEpoch},
        'timestamp': DateTime.now().millisecondsSinceEpoch
      };
      http.post(
        Uri.parse('http://127.0.0.1:7242/ingest/f2e5acba-93e2-4270-a633-fee604b9e81c'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode(logData2),
      ).catchError((_) {});
      // #endregion
      
      final result = await TradeIdeasService.fetchAllowedTradeIdeasWithDiagnostics(
        _selectedTicker,
        maxIdeas: 3,
        timeframe: _selectedTimeframe,
        minDte: _useCustomDte ? _customMinDte : null,
        maxDte: _useCustomDte ? _customMaxDte : null,
      );
      
      // #region agent log
      final logData3 = {
        'sessionId': 'debug-session',
        'runId': 'run1',
        'hypothesisId': 'D',
        'location': 'trade_ideas_screen.dart:105',
        'message': 'API call completed',
        'data': {
          'hasUnlocked': result.containsKey('unlocked'),
          'hasUnlockedByTimeframe': result.containsKey('unlockedByTimeframe'),
          'ideasCount': result.containsKey('unlocked') ? (result['unlocked'] as List).length : 0,
          'timestamp': DateTime.now().millisecondsSinceEpoch
        },
        'timestamp': DateTime.now().millisecondsSinceEpoch
      };
      http.post(
        Uri.parse('http://127.0.0.1:7242/ingest/f2e5acba-93e2-4270-a633-fee604b9e81c'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode(logData3),
      ).catchError((_) {});
      // #endregion
      
      // #region agent log
      final logData4 = {
        'sessionId': 'debug-session',
        'runId': 'run1',
        'hypothesisId': 'C',
        'location': 'trade_ideas_screen.dart:170',
        'message': 'setState called with result',
        'data': {
          'hasUnlockedByTimeframe': result.containsKey('unlockedByTimeframe'),
          'hasUnlocked': result.containsKey('unlocked'),
          'hasIdeasByTimeframe': result.containsKey('ideasByTimeframe'),
          'timestamp': DateTime.now().millisecondsSinceEpoch
        },
        'timestamp': DateTime.now().millisecondsSinceEpoch
      };
      http.post(
        Uri.parse('http://127.0.0.1:7242/ingest/f2e5acba-93e2-4270-a633-fee604b9e81c'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode(logData4),
      ).catchError((_) {});
      // #endregion
      
      setState(() {
        // Parse unlocked ideas
        if (result.containsKey('unlockedByTimeframe')) {
          _ideasByTimeframe = result['unlockedByTimeframe'] as Map<String, List<TradeIdea>>;
          _ideas = [];
        } else if (result.containsKey('unlocked')) {
          _ideas = result['unlocked'] as List<TradeIdea>;
          _ideasByTimeframe = null;
        } else if (result.containsKey('ideasByTimeframe')) {
          // Backward compatibility
          _ideasByTimeframe = result['ideasByTimeframe'] as Map<String, List<TradeIdea>>;
          _ideas = [];
        } else if (result.containsKey('ideas')) {
          // Backward compatibility
          _ideas = result['ideas'] as List<TradeIdea>;
          _ideasByTimeframe = null;
        } else {
          _ideas = [];
          _ideasByTimeframe = null;
        }
        
        // Parse preview ideas
        _previewIdeas = result['preview'] as List<TradeIdea>? ?? [];
        
        // Parse next window and current phase
        _nextWindow = result['nextWindow'] as Map<String, dynamic>?;
        _currentPhase = result['currentPhase'] as String?;
        
        _diagnostics = result['diagnostics'] as Map<String, dynamic>?;
        _isLoading = false;
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
    // #region agent log
    final logData = {
      'sessionId': 'debug-session',
      'runId': 'run1',
      'hypothesisId': 'C',
      'location': 'trade_ideas_screen.dart:132',
      'message': 'build method called',
      'data': {
        'isLoading': _isLoading,
        'hasError': _error != null,
        'ideasCount': _ideas.length,
        'hasIdeasByTimeframe': _ideasByTimeframe != null,
        'timestamp': DateTime.now().millisecondsSinceEpoch
      },
      'timestamp': DateTime.now().millisecondsSinceEpoch
    };
    http.post(
      Uri.parse('http://127.0.0.1:7242/ingest/f2e5acba-93e2-4270-a633-fee604b9e81c'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode(logData),
    ).catchError((_) {});
    // #endregion
    
    return Scaffold(
      backgroundColor: const Color(0xFF0D1117),
      appBar: AppBar(
        title: const Text('Trade Ideas', style: TextStyle(color: Colors.white, fontSize: 18, fontFamily: 'JetBrains Mono')),
        backgroundColor: const Color(0xFF161B22),
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh, color: Colors.white),
            onPressed: _loadTradeIdeas,
            tooltip: 'Refresh',
          ),
        ],
      ),
      body: _isLoading
          ? const Center(child: CircularProgressIndicator())
          : _error != null
              ? Center(
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Text('Error: $_error', style: const TextStyle(color: Colors.red)),
                      const SizedBox(height: 16),
                      ElevatedButton(
                        onPressed: _loadTradeIdeas,
                        child: const Text('Retry'),
                      ),
                    ],
                  ),
                )
              : Column(
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    children: [
                      // Controls bar
                      Container(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
            color: const Color(0xFF161B22),
            child: Column(
              children: [
                // First row: Ticker and Timeframe
                Row(
                  children: [
                    Expanded(
                      child: AssetSelectorWidget(
                        onAssetChanged: (asset) {
                          _loadTradeIdeas();
                        },
                      ),
                    ),
                    const SizedBox(width: 16),
                    const Text(
                      'Timeframe: ',
                      style: TextStyle(color: Colors.white70, fontSize: 14, fontFamily: 'JetBrains Mono'),
                    ),
                    const SizedBox(width: 8),
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                      decoration: BoxDecoration(
                        color: const Color(0xFF0D1117),
                        borderRadius: BorderRadius.circular(6),
                        border: Border.all(color: Colors.white24),
                      ),
                      child: DropdownButton<String>(
                        value: _selectedTimeframe,
                        dropdownColor: const Color(0xFF161B22),
                        underline: const SizedBox(),
                        style: const TextStyle(color: Colors.white, fontSize: 14, fontFamily: 'JetBrains Mono'),
                        items: [
                          const DropdownMenuItem(value: 'all', child: Text('All Timeframes')),
                          const DropdownMenuItem(value: 'thisWeek', child: Text('This Week (0-5 DTE)')),
                          const DropdownMenuItem(value: 'thisMonth', child: Text('This Month (7-30 DTE)')),
                          const DropdownMenuItem(value: 'thisYear', child: Text('This Year (30-365 DTE)')),
                        ],
                        onChanged: (value) {
                          if (value != null) {
                            setState(() {
                              _selectedTimeframe = value;
                              _useCustomDte = false;
                            });
                            _loadTradeIdeas();
                          }
                        },
                      ),
                    ),
                    const Spacer(),
                    if (_ideas.isNotEmpty || (_ideasByTimeframe != null && _ideasByTimeframe!.values.any((list) => list.isNotEmpty)))
                      Text(
                        _getTotalIdeasCount().toString() + ' idea${_getTotalIdeasCount() == 1 ? '' : 's'}',
                        style: const TextStyle(color: Colors.white54, fontSize: 12, fontFamily: 'JetBrains Mono'),
                      ),
                  ],
                ),
                const SizedBox(height: 8),
                // Second row: Custom DTE selector
                Row(
                  children: [
                    Checkbox(
                      value: _useCustomDte,
                      onChanged: (value) {
                        setState(() {
                          _useCustomDte = value ?? false;
                          if (_useCustomDte) {
                            _selectedTimeframe = 'all';  // Reset timeframe when using custom
                          }
                        });
                      },
                      checkColor: Colors.white,
                      fillColor: MaterialStateProperty.all(const Color(0xFF58A6FF)),
                    ),
                    const Text(
                      'Custom DTE: ',
                      style: TextStyle(color: Colors.white70, fontSize: 12, fontFamily: 'JetBrains Mono'),
                    ),
                    const SizedBox(width: 8),
                    SizedBox(
                      width: 80,
                      child: TextField(
                        enabled: _useCustomDte,
                        style: const TextStyle(color: Colors.white, fontSize: 12, fontFamily: 'JetBrains Mono'),
                        decoration: InputDecoration(
                          hintText: 'Min',
                          hintStyle: const TextStyle(color: Colors.white38, fontSize: 12),
                          filled: true,
                          fillColor: _useCustomDte ? const Color(0xFF0D1117) : Colors.grey.withOpacity(0.1),
                          border: OutlineInputBorder(
                            borderRadius: BorderRadius.circular(4),
                            borderSide: BorderSide(color: Colors.white24),
                          ),
                          contentPadding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                        ),
                        keyboardType: TextInputType.number,
                        onChanged: (value) {
                          _customMinDte = int.tryParse(value);
                          if (_useCustomDte && _customMinDte != null) {
                            _loadTradeIdeas();
                          }
                        },
                      ),
                    ),
                    const Text(
                      ' - ',
                      style: TextStyle(color: Colors.white70, fontSize: 12),
                    ),
                    SizedBox(
                      width: 80,
                      child: TextField(
                        enabled: _useCustomDte,
                        style: const TextStyle(color: Colors.white, fontSize: 12, fontFamily: 'JetBrains Mono'),
                        decoration: InputDecoration(
                          hintText: 'Max',
                          hintStyle: const TextStyle(color: Colors.white38, fontSize: 12),
                          filled: true,
                          fillColor: _useCustomDte ? const Color(0xFF0D1117) : Colors.grey.withOpacity(0.1),
                          border: OutlineInputBorder(
                            borderRadius: BorderRadius.circular(4),
                            borderSide: BorderSide(color: Colors.white24),
                          ),
                          contentPadding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                        ),
                        keyboardType: TextInputType.number,
                        onChanged: (value) {
                          _customMaxDte = int.tryParse(value);
                          if (_useCustomDte && _customMaxDte != null) {
                            _loadTradeIdeas();
                          }
                        },
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ),
          Expanded(
            child: SingleChildScrollView(
              child: _buildBody(),
            ),
          ),
        ],
                  ),
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
            Text('Loading trade ideas...', style: TextStyle(color: Colors.white70, fontSize: 16, fontFamily: 'JetBrains Mono')),
          ],
        ),
      );
    }

    if (_error != null) {
      return Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Icon(Icons.error_outline, color: Colors.red, size: 48),
            const SizedBox(height: 16),
            Text(_error!, style: const TextStyle(color: Colors.white70, fontSize: 16, fontFamily: 'JetBrains Mono')),
            const SizedBox(height: 16),
            ElevatedButton(
              onPressed: _loadTradeIdeas,
              child: const Text('Retry', style: TextStyle(fontFamily: 'JetBrains Mono')),
            ),
          ],
        ),
      );
    }

    // Check if we have unlocked ideas (either single list or by timeframe)
    final hasUnlockedIdeas = _ideas.isNotEmpty || (_ideasByTimeframe != null && _ideasByTimeframe!.values.any((list) => list.isNotEmpty));
    
    return RefreshIndicator(
      onRefresh: _loadTradeIdeas,
      child: SingleChildScrollView(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Section A: Unlocked Now
            _buildUnlockedSection(hasUnlockedIdeas),
            
            // Section B: Preview Next Window
            _buildPreviewSection(),
          ],
        ),
      ),
    );
  }
  
  int _getTotalIdeasCount() {
    if (_ideasByTimeframe != null) {
      return _ideasByTimeframe!.values.fold(0, (sum, list) => sum + list.length);
    }
    return _ideas.length;
  }
  
  Widget _buildTimeframeView() {
    final timeframes = ['thisWeek', 'thisMonth', 'thisYear'];
    final labels = {
      'thisWeek': 'This Week (0-5 DTE)',
      'thisMonth': 'This Month (7-30 DTE)',
      'thisYear': 'This Year (30-365 DTE)',
    };
    
    return ListView.builder(
      padding: const EdgeInsets.all(12),
      itemCount: timeframes.length,
      itemBuilder: (context, index) {
        final tf = timeframes[index];
        final ideas = _ideasByTimeframe![tf] ?? [];
        
        if (ideas.isEmpty) {
          return const SizedBox.shrink();  // Skip empty timeframes
        }
        
        return Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Padding(
              padding: EdgeInsets.only(bottom: 8, top: index > 0 ? 16.0 : 0.0),
              child: Text(
                labels[tf] ?? tf,
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 16,
                  fontWeight: FontWeight.bold,
                  fontFamily: 'JetBrains Mono',
                ),
              ),
            ),
            ...ideas.map((idea) => _buildTradeIdeaCard(idea)),
          ],
        );
      },
    );
  }

  Widget _buildUnlockedSection(bool hasUnlockedIdeas) {
    return Container(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          // Header
          const Text(
            'Unlocked Now',
            style: TextStyle(
              color: Colors.white,
              fontSize: 20,
              fontWeight: FontWeight.bold,
              fontFamily: 'JetBrains Mono',
            ),
          ),
          const SizedBox(height: 4),
          const Text(
            'These setups are permitted under current regime and time window.',
            style: TextStyle(
              color: Colors.white70,
              fontSize: 12,
              fontFamily: 'JetBrains Mono',
            ),
          ),
          const SizedBox(height: 16),
          
          // Content
          if (!hasUnlockedIdeas)
            _buildEmptyState()
          else if (_ideasByTimeframe != null)
            _buildTimeframeView()
          else
            ..._ideas.map((idea) => _buildTradeIdeaCard(idea)),
        ],
      ),
    );
  }
  
  Widget _buildPreviewSection() {
    if (_previewIdeas.isEmpty) {
      return const SizedBox.shrink();
    }
    
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFF161B22),
        border: Border(
          top: BorderSide(color: Colors.white.withOpacity(0.1)),
        ),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          // Header
          const Text(
            'Preview Next Window',
            style: TextStyle(
              color: Colors.white70,
              fontSize: 20,
              fontWeight: FontWeight.bold,
              fontFamily: 'JetBrains Mono',
            ),
          ),
          const SizedBox(height: 4),
          Text(
            'Planning view — candidates for the next valid trading window. Not actionable yet.${_nextWindow != null ? '\nNext window: ${_nextWindow!['label'] ?? 'Unknown'}' : ''}',
            style: const TextStyle(
              color: Colors.white54,
              fontSize: 12,
              fontFamily: 'JetBrains Mono',
            ),
          ),
          const SizedBox(height: 16),
          
          // Preview ideas
          ..._previewIdeas.map((idea) => _buildPreviewCard(idea)),
        ],
      ),
    );
  }
  
  Widget _buildPreviewCard(TradeIdea idea) {
    final isExpanded = _expandedCards[idea.id] ?? false;
    
    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      decoration: BoxDecoration(
        color: const Color(0xFF0D1117).withOpacity(0.5),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: Colors.white.withOpacity(0.1)),
      ),
      child: InkWell(
        onTap: () {
          setState(() {
            _expandedCards[idea.id] = !isExpanded;
          });
        },
        child: Padding(
          padding: const EdgeInsets.all(12),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // Header with PREVIEW badge
              Row(
                children: [
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                    decoration: BoxDecoration(
                      color: Colors.white.withOpacity(0.1),
                      borderRadius: BorderRadius.circular(4),
                    ),
                    child: const Text(
                      'PREVIEW',
                      style: TextStyle(
                        color: Colors.white54,
                        fontSize: 10,
                        fontWeight: FontWeight.bold,
                        fontFamily: 'JetBrains Mono',
                      ),
                    ),
                  ),
                  const SizedBox(width: 8),
                  if (idea.blockReason != null)
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                      decoration: BoxDecoration(
                        color: Colors.orange.withOpacity(0.2),
                        borderRadius: BorderRadius.circular(4),
                      ),
                      child: Text(
                        idea.blockReason!.replaceAll('_', ' '),
                        style: const TextStyle(
                          color: Colors.orange,
                          fontSize: 10,
                          fontFamily: 'JetBrains Mono',
                        ),
                      ),
                    ),
                  const Spacer(),
                  Icon(
                    isExpanded ? Icons.expand_less : Icons.expand_more,
                    color: Colors.white54,
                    size: 20,
                  ),
                ],
              ),
              const SizedBox(height: 8),
              
              // Strategy label
              Text(
                idea.label,
                style: const TextStyle(
                  color: Colors.white70,
                  fontSize: 14,
                  fontFamily: 'JetBrains Mono',
                ),
              ),
              
              // Next window info
              if (idea.nextWindow != null) ...[
                const SizedBox(height: 4),
                Text(
                  'Expected entry: ${idea.nextWindow!['label'] ?? 'Unknown'}',
                  style: const TextStyle(
                    color: Colors.white54,
                    fontSize: 11,
                    fontFamily: 'JetBrains Mono',
                  ),
                ),
              ],
              
              // Unlock conditions
              if (idea.blockReason == 'REGIME_NEAR_UNLOCK' && idea.contextSnapshot != null) ...[
                const SizedBox(height: 8),
                Container(
                  padding: const EdgeInsets.all(8),
                  decoration: BoxDecoration(
                    color: Colors.blue.withOpacity(0.1),
                    borderRadius: BorderRadius.circular(4),
                  ),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Text(
                        'Watch Trigger:',
                        style: TextStyle(
                          color: Colors.blue,
                          fontSize: 11,
                          fontWeight: FontWeight.bold,
                          fontFamily: 'JetBrains Mono',
                        ),
                      ),
                      const SizedBox(height: 4),
                      if (idea.contextSnapshot!.flipLine != null)
                        Text(
                          'Unlock if spot > ${idea.contextSnapshot!.flipLine!.toStringAsFixed(2)} and GEX turns Neutral',
                          style: const TextStyle(
                            color: Colors.white70,
                            fontSize: 11,
                            fontFamily: 'JetBrains Mono',
                          ),
                        ),
                    ],
                  ),
                ),
              ],
              
              // Expanded view
              if (isExpanded) ...[
                const SizedBox(height: 12),
                const Divider(color: Colors.white24),
                const SizedBox(height: 8),
                Text(
                  'What must happen to unlock:',
                  style: const TextStyle(
                    color: Colors.white70,
                    fontSize: 12,
                    fontWeight: FontWeight.bold,
                    fontFamily: 'JetBrains Mono',
                  ),
                ),
                const SizedBox(height: 8),
                ...idea.reasons.map((reason) => Padding(
                  padding: const EdgeInsets.only(bottom: 4),
                  child: Row(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Text('• ', style: TextStyle(color: Colors.white54, fontSize: 12)),
                      Expanded(
                        child: Text(
                          reason,
                          style: const TextStyle(
                            color: Colors.white54,
                            fontSize: 11,
                            fontFamily: 'JetBrains Mono',
                          ),
                        ),
                      ),
                    ],
                  ),
                )),
                
                // Show strikes but hide prices (or show with stale label)
                if (idea.executions.isNotEmpty) ...[
                  const SizedBox(height: 12),
                  const Divider(color: Colors.white24),
                  const SizedBox(height: 8),
                  Text(
                    'Potential structures (indicative — refresh at open):',
                    style: const TextStyle(
                      color: Colors.white54,
                      fontSize: 11,
                      fontStyle: FontStyle.italic,
                      fontFamily: 'JetBrains Mono',
                    ),
                  ),
                  const SizedBox(height: 8),
                  ...idea.executions.take(2).map((exec) => _buildPreviewExecution(exec)),
                ],
              ],
            ],
          ),
        ),
      ),
    );
  }
  
  Widget _buildPreviewExecution(ReferenceExecution exec) {
    return Container(
      margin: const EdgeInsets.only(bottom: 8),
      padding: const EdgeInsets.all(8),
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.05),
        borderRadius: BorderRadius.circular(4),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            '${exec.dte} DTE — ${exec.expiry}',
            style: const TextStyle(
              color: Colors.white54,
              fontSize: 10,
              fontFamily: 'JetBrains Mono',
            ),
          ),
          const SizedBox(height: 4),
          ...exec.legs.map((leg) => Padding(
            padding: const EdgeInsets.only(bottom: 2),
            child: Text(
              '${leg.action} ${leg.type} ${leg.strike.toStringAsFixed(0)} (Δ${leg.delta?.toStringAsFixed(2) ?? 'N/A'})',
              style: const TextStyle(
                color: Colors.white54,
                fontSize: 10,
                fontFamily: 'JetBrains Mono',
              ),
            ),
          )),
        ],
      ),
    );
  }

  Widget _buildEmptyState() {
    final diagnostics = _diagnostics;
    final isBlocked = diagnostics?['blocked'] == true;
    final blockReason = diagnostics?['blockReason'] as String?;
    final contextData = diagnostics?['context'] as Map<String, dynamic>?;
    final additionalReasons = diagnostics?['additionalReasons'] as List<dynamic>?;
    
    // New diagnostics fields
    final flipLineAnalysis = diagnostics?['flipLineAnalysis'] as Map<String, dynamic>?;
    final depinRiskAnalysis = diagnostics?['depinRiskAnalysis'] as Map<String, dynamic>?;
    final strategyFiltering = diagnostics?['strategyFiltering'] as Map<String, dynamic>?;
    final spotAboveFlip = flipLineAnalysis?['spotAboveFlip'];
    final flipLine = flipLineAnalysis?['flipLine'];
    final spot = flipLineAnalysis?['spot'];
    final depinRisk = depinRiskAnalysis?['depinRisk'];
    final depinRiskElevated = depinRiskAnalysis?['depinRiskElevated'] == true;
    final depinRiskConfirmed = depinRiskAnalysis?['depinRiskConfirmed'] == true;
    final allowedStrategies = strategyFiltering?['allowedStrategies'] as List<dynamic>?;
    final strategyFilterReason = strategyFiltering?['strategyFilterReason'] as String?;
    
    return SingleChildScrollView(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 24),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          // Standby Mode Header
          _buildStandbyHeader(),
          const SizedBox(height: 20),
          
          // Discipline Win (if regime-blocked, not technical error)
          if (!isBlocked || (isBlocked && blockReason != null && !blockReason!.toLowerCase().contains('unavailable')))
            _buildDisciplineWin(),
          if (!isBlocked || (isBlocked && blockReason != null && !blockReason!.toLowerCase().contains('unavailable')))
            const SizedBox(height: 16),
            
            // Blocking reason
            if (isBlocked && blockReason != null) ...[
              Container(
                padding: const EdgeInsets.all(16),
                margin: const EdgeInsets.only(bottom: 16),
                decoration: BoxDecoration(
                  color: Colors.red.withOpacity(0.1),
                  borderRadius: BorderRadius.circular(8),
                  border: Border.all(color: Colors.red.withOpacity(0.3)),
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Row(
                      children: [
                        Icon(Icons.warning, color: Colors.red, size: 20),
                        SizedBox(width: 8),
                        Text(
                          'Blocked by Safety Rules',
                          style: TextStyle(
                            color: Colors.red,
                            fontSize: 14,
                            fontWeight: FontWeight.bold,
                            fontFamily: 'JetBrains Mono',
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 8),
                    Text(
                      blockReason,
                      style: const TextStyle(
                        color: Colors.white70,
                        fontSize: 12,
                        fontFamily: 'JetBrains Mono',
                      ),
                    ),
                  ],
                ),
              ),
            ],
            
            // Additional reasons (only show if not regime-blocked)
            if (additionalReasons != null && additionalReasons.isNotEmpty && 
                (allowedStrategies == null || allowedStrategies!.isNotEmpty)) ...[
              Container(
                padding: const EdgeInsets.all(16),
                margin: const EdgeInsets.only(bottom: 16),
                decoration: BoxDecoration(
                  color: Colors.orange.withOpacity(0.1),
                  borderRadius: BorderRadius.circular(8),
                  border: Border.all(color: Colors.orange.withOpacity(0.3)),
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Row(
                      children: [
                        Icon(Icons.info_outline, color: Colors.orange, size: 20),
                        SizedBox(width: 8),
                        Text(
                          'Additional Filters',
                          style: TextStyle(
                            color: Colors.orange,
                            fontSize: 14,
                            fontWeight: FontWeight.bold,
                            fontFamily: 'JetBrains Mono',
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 8),
                    ...additionalReasons.map((reason) => Padding(
                      padding: const EdgeInsets.only(bottom: 4),
                      child: Text(
                        '• $reason',
                        style: const TextStyle(
                          color: Colors.white70,
                          fontSize: 12,
                          fontFamily: 'JetBrains Mono',
                        ),
                      ),
                    )),
                  ],
                ),
              ),
            ],
            
            // What We're Waiting For
            _buildWaitingConditions(
              contextData: contextData,
              spotAboveFlip: spotAboveFlip,
              flipLine: flipLine,
              depinRisk: depinRisk,
              depinRiskElevated: depinRiskElevated,
              depinRiskConfirmed: depinRiskConfirmed,
              allowedStrategies: allowedStrategies,
            ),
            
            // Current market context
            if (contextData != null) ...[
              Container(
                padding: const EdgeInsets.all(16),
                margin: const EdgeInsets.only(bottom: 16),
                decoration: BoxDecoration(
                  color: Colors.white.withOpacity(0.05),
                  borderRadius: BorderRadius.circular(8),
                  border: Border.all(color: Colors.white12),
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text(
                      'Current Market State',
                      style: TextStyle(
                        color: Colors.white,
                        fontSize: 14,
                        fontWeight: FontWeight.bold,
                        fontFamily: 'JetBrains Mono',
                      ),
                    ),
                    const SizedBox(height: 8),
                    _buildContextRow('GEX State', _formatValue(contextData['gexState'])),
                    // Clarify REAL_OK vs Regime
                    if (contextData['actionFilter'] == 'REAL_OK') ...[
                      _buildContextRow('Execution Status', 'Allowed ✓'),
                      _buildContextRow(
                        'Strategy Permission',
                        (allowedStrategies != null && allowedStrategies!.isEmpty)
                            ? 'Blocked (Regime)'
                            : 'Allowed',
                      ),
                    ] else
                      _buildContextRow('Action Filter', _formatValue(contextData['actionFilter'])),
                    _buildContextRow('De-Pin Risk', _formatValue(contextData['depinRisk'])),
                    _buildContextRow('Pin Confidence', _formatValue(contextData['pinConfidence'])),
                    if (flipLine != null && spot != null) ...[
                      _buildContextRow('Spot Price', _formatValue(spot, isPrice: true)),
                      _buildContextRow('Flip Line', _formatValue(flipLine, isPrice: true)),
                      if (spotAboveFlip != null)
                        _buildContextRow(
                          'Position',
                          spotAboveFlip == true
                              ? 'Above Flip ✓'
                              : spotAboveFlip == false
                                  ? 'Below Flip ⚠️'
                                  : 'At Flip',
                        ),
                    ] else ...[
                      if (contextData['spot'] != null)
                        _buildContextRow('Spot Price', _formatValue(contextData['spot'], isPrice: true)),
                      if (contextData['flipLine'] != null)
                        _buildContextRow('Flip Line', _formatValue(contextData['flipLine'], isPrice: true)),
                    ],
                    if (depinRisk != null) ...[
                      _buildContextRow(
                        'DePin Status',
                        depinRiskConfirmed
                            ? 'Confirmed High ⚠️'
                            : depinRiskElevated
                                ? 'Elevated'
                                : 'Normal',
                      ),
                    ],
                    if (contextData['putWall'] != null)
                      _buildContextRow('Put Wall', _formatValue(contextData['putWall'], isPrice: true)),
                    if (contextData['callWall'] != null)
                      _buildContextRow('Call Wall', _formatValue(contextData['callWall'], isPrice: true)),
                  ],
                ),
              ),
            ],
            
            // Locked Strategies Preview
            if (allowedStrategies != null && allowedStrategies.isEmpty) ...[
              Container(
                padding: const EdgeInsets.all(16),
                margin: const EdgeInsets.only(bottom: 16),
                decoration: BoxDecoration(
                  color: Colors.white.withOpacity(0.05),
                  borderRadius: BorderRadius.circular(8),
                  border: Border.all(color: Colors.white12),
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Row(
                      children: [
                        Icon(Icons.lock_outline, color: Colors.white70, size: 20),
                        SizedBox(width: 8),
                        Text(
                          'Locked Strategies Preview',
                          style: TextStyle(
                            color: Colors.white,
                            fontSize: 14,
                            fontWeight: FontWeight.bold,
                            fontFamily: 'JetBrains Mono',
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 12),
                    const Text(
                      'These strategies are locked due to current regime conditions:',
                      style: TextStyle(
                        color: Colors.white60,
                        fontSize: 11,
                        fontFamily: 'JetBrains Mono',
                      ),
                    ),
                    const SizedBox(height: 12),
                    // Show locked strategies from config
                    _buildLockedStrategyCard(
                      strategy: 'PUT_CREDIT_SPREAD',
                      contextData: contextData,
                      spotAboveFlip: spotAboveFlip,
                      depinRiskConfirmed: depinRiskConfirmed,
                    ),
                    _buildLockedStrategyCard(
                      strategy: 'CALL_CREDIT_SPREAD',
                      contextData: contextData,
                      spotAboveFlip: spotAboveFlip,
                      depinRiskConfirmed: depinRiskConfirmed,
                    ),
                    if (spotAboveFlip == false) ...[
                      _buildLockedStrategyCard(
                        strategy: 'IRON_CONDOR',
                        contextData: contextData,
                        spotAboveFlip: spotAboveFlip,
                        depinRiskConfirmed: depinRiskConfirmed,
                      ),
                      _buildLockedStrategyCard(
                        strategy: 'BUTTERFLY',
                        contextData: contextData,
                        spotAboveFlip: spotAboveFlip,
                        depinRiskConfirmed: depinRiskConfirmed,
                      ),
                    ],
                  ],
                ),
              ),
            ],
            
            // Rules explanation
            Container(
              padding: const EdgeInsets.all(16),
              margin: const EdgeInsets.only(bottom: 16),
              decoration: BoxDecoration(
                color: Colors.blue.withOpacity(0.1),
                borderRadius: BorderRadius.circular(8),
                border: Border.all(color: Colors.blue.withOpacity(0.3)),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Row(
                    children: [
                      Icon(Icons.rule, color: Colors.blue, size: 20),
                      SizedBox(width: 8),
                      Text(
                        'Safety Rules',
                        style: TextStyle(
                          color: Colors.blue,
                          fontSize: 14,
                          fontWeight: FontWeight.bold,
                          fontFamily: 'JetBrains Mono',
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 8),
                  const Text(
                    '• Ideas blocked if Action Filter is BLOCKED or WAIT_FOR_CLARITY',
                    style: TextStyle(color: Colors.white70, fontSize: 11, fontFamily: 'JetBrains Mono'),
                  ),
                  const Text(
                    '• Short premium blocked when GEX is NEGATIVE',
                    style: TextStyle(color: Colors.white70, fontSize: 11, fontFamily: 'JetBrains Mono'),
                  ),
                  const Text(
                    '• Requires valid de-pin risk data',
                    style: TextStyle(color: Colors.white70, fontSize: 11, fontFamily: 'JetBrains Mono'),
                  ),
                  const Text(
                    '• Strikes must be beyond wall buffers',
                    style: TextStyle(color: Colors.white70, fontSize: 11, fontFamily: 'JetBrains Mono'),
                  ),
                  const Text(
                    '• Minimum liquidity (OI, volume, spread) required',
                    style: TextStyle(color: Colors.white70, fontSize: 11, fontFamily: 'JetBrains Mono'),
                  ),
                ],
              ),
            ),
            
            const SizedBox(height: 24),
            ElevatedButton.icon(
              onPressed: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(builder: (context) => const DecisionCockpitScreen()),
                );
              },
              icon: const Icon(Icons.dashboard),
              label: const Text('View Full Market Conditions', style: TextStyle(fontFamily: 'JetBrains Mono')),
              style: ElevatedButton.styleFrom(
                backgroundColor: const Color(0xFF58A6FF),
                foregroundColor: Colors.white,
                padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
              ),
            ),
            const SizedBox(height: 16),
          ],
        ),
      );
  }
  
  Widget _buildContextRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 4),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(
            label,
            style: const TextStyle(
              color: Colors.white54,
              fontSize: 11,
              fontFamily: 'JetBrains Mono',
            ),
          ),
          Text(
            value,
            style: const TextStyle(
              color: Colors.white,
              fontSize: 11,
              fontWeight: FontWeight.bold,
              fontFamily: 'JetBrains Mono',
            ),
          ),
        ],
      ),
    );
  }
  
  String _formatValue(dynamic value, {bool isPrice = false}) {
    if (value == null) return 'N/A';
    if (value is num) {
      if (isPrice) {
        return '\$${value.toStringAsFixed(2)}';
      }
      return value.toStringAsFixed(0);
    }
    return value.toString();
  }

  // ===============================
  // Standby Mode Components
  // ===============================

  Widget _buildStandbyHeader() {
    return Column(
      children: [
        Icon(
          Icons.shield,
          color: const Color(0xFF58A6FF),
          size: 64,
        ),
        const SizedBox(height: 16),
        const Text(
          'Standby Mode — Capital Protected',
          style: TextStyle(
            color: Colors.white,
            fontSize: 22,
            fontWeight: FontWeight.bold,
            fontFamily: 'JetBrains Mono',
          ),
          textAlign: TextAlign.center,
        ),
        const SizedBox(height: 8),
        Text(
          'Current conditions favor patience over exposure',
          style: TextStyle(
            color: Colors.white70,
            fontSize: 14,
            fontFamily: 'JetBrains Mono',
          ),
          textAlign: TextAlign.center,
        ),
      ],
    );
  }

  Widget _buildDisciplineWin() {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      margin: const EdgeInsets.symmetric(horizontal: 0),
      decoration: BoxDecoration(
        color: const Color(0xFF58A6FF).withOpacity(0.15),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: const Color(0xFF58A6FF).withOpacity(0.3)),
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.center,
        mainAxisSize: MainAxisSize.min,
        children: [
          const Icon(Icons.shield, color: Color(0xFF58A6FF), size: 20),
          const SizedBox(width: 8),
          const Flexible(
            child: Text(
              'Discipline Win: No forced trades in unstable regime',
              style: TextStyle(
                color: Color(0xFF58A6FF),
                fontSize: 13,
                fontWeight: FontWeight.bold,
                fontFamily: 'JetBrains Mono',
              ),
              textAlign: TextAlign.center,
            ),
          ),
        ],
      ),
    );
  }

  List<String> _generateWaitingConditions({
    required Map<String, dynamic>? contextData,
    required dynamic spotAboveFlip,
    required double? flipLine,
    required double? depinRisk,
    required bool depinRiskElevated,
    required bool depinRiskConfirmed,
    required List<dynamic>? allowedStrategies,
  }) {
    final conditions = <String>[];
    final gexState = contextData?['gexState'] as String?;
    final macroEventNext24h = contextData?['macroEventNext24h'] == true;
    final megaCapEarningsNext24h = contextData?['megaCapEarningsNext24h'] == true;
    
    // GEX condition
    if (gexState == 'NEGATIVE' && spotAboveFlip != false) {
      conditions.add('GEX returns to neutral or positive');
    }
    
    // Flip line condition
    if (spotAboveFlip == false && flipLine != null) {
      conditions.add('Spot reclaims flip line (~\$${flipLine.toStringAsFixed(2)})');
    }
    
    // DePin risk condition
    if (depinRiskConfirmed) {
      conditions.add('DePin risk decreases below 70 (currently ${depinRisk?.toStringAsFixed(0) ?? 'N/A'})');
    } else if (depinRiskElevated) {
      conditions.add('DePin risk decreases below 50 or confirms direction (currently ${depinRisk?.toStringAsFixed(0) ?? 'N/A'})');
    }
    
    // Event risk condition
    if (macroEventNext24h || megaCapEarningsNext24h) {
      conditions.add('Event risk clears (earnings/macro events)');
    }
    
    // Hedged structures condition
    if (spotAboveFlip == false) {
      // Check if hedged strategies are available (they're not implemented yet)
      final allowedStrategiesList = allowedStrategies;
      final hasHedgedStrategies = allowedStrategiesList?.any((s) => 
        s.toString().contains('IRON_CONDOR') || s.toString().contains('BUTTERFLY')
      ) ?? false;
      if (!hasHedgedStrategies) {
        conditions.add('Hedged structures implemented (Iron Condors, Butterflies)');
      }
    }
    
    return conditions;
  }

  Map<String, String> _getStrategyLockReason({
    required String strategy,
    required Map<String, dynamic>? contextData,
    required dynamic spotAboveFlip,
    required bool depinRiskConfirmed,
  }) {
    final gexState = contextData?['gexState'] as String?;
    final requirements = <String>[];
    String reason = '';
    
    if (strategy == 'PUT_CREDIT_SPREAD' || strategy == 'CALL_CREDIT_SPREAD') {
      if (gexState == 'NEGATIVE' && spotAboveFlip != false) {
        reason = 'GEX negative - time-decay strategies blocked';
        requirements.add('GEX ≥ Neutral');
      } else if (spotAboveFlip == false) {
        reason = 'Spot below flip line - blind short premium blocked';
        requirements.add('Spot above flip');
      } else if (depinRiskConfirmed) {
        reason = 'DePin risk confirmed - all strategies blocked';
        requirements.add('DePin risk < 70');
      } else {
        reason = 'Regime conditions not met';
      }
      
      if (spotAboveFlip != false) {
        requirements.add('Spot above flip');
      }
      if (!depinRiskConfirmed) {
        requirements.add('DePin risk < 70');
      }
    } else if (strategy == 'IRON_CONDOR' || strategy == 'BUTTERFLY') {
      reason = 'Hedged structures not yet implemented';
      requirements.add('Implementation pending');
      if (depinRiskConfirmed) {
        requirements.add('DePin risk < 70');
      }
    }
    
    return {
      'reason': reason,
      'requirements': requirements.join(', '),
    };
  }

  Widget _buildWaitingConditions({
    required Map<String, dynamic>? contextData,
    required dynamic spotAboveFlip,
    required double? flipLine,
    required double? depinRisk,
    required bool depinRiskElevated,
    required bool depinRiskConfirmed,
    required List<dynamic>? allowedStrategies,
  }) {
    final conditions = _generateWaitingConditions(
      contextData: contextData,
      spotAboveFlip: spotAboveFlip,
      flipLine: flipLine,
      depinRisk: depinRisk,
      depinRiskElevated: depinRiskElevated,
      depinRiskConfirmed: depinRiskConfirmed,
      allowedStrategies: allowedStrategies,
    );
    
    if (conditions.isEmpty) {
      return const SizedBox.shrink();
    }
    
    return Container(
      padding: const EdgeInsets.all(16),
      margin: const EdgeInsets.only(bottom: 16),
      decoration: BoxDecoration(
        color: const Color(0xFF58A6FF).withOpacity(0.1),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: const Color(0xFF58A6FF).withOpacity(0.3)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Row(
            children: [
              Icon(Icons.schedule, color: Color(0xFF58A6FF), size: 20),
              SizedBox(width: 8),
              Text(
                'What We\'re Waiting For',
                style: TextStyle(
                  color: Color(0xFF58A6FF),
                  fontSize: 14,
                  fontWeight: FontWeight.bold,
                  fontFamily: 'JetBrains Mono',
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          ...conditions.map((condition) => Padding(
            padding: const EdgeInsets.only(bottom: 8),
            child: Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  '• ',
                  style: TextStyle(
                    color: Color(0xFF58A6FF),
                    fontSize: 14,
                    fontFamily: 'JetBrains Mono',
                  ),
                ),
                Expanded(
                  child: Text(
                    condition,
                    style: const TextStyle(
                      color: Colors.white70,
                      fontSize: 12,
                      fontFamily: 'JetBrains Mono',
                    ),
                  ),
                ),
              ],
            ),
          )),
        ],
      ),
    );
  }

  Widget _buildLockedStrategyCard({
    required String strategy,
    required Map<String, dynamic>? contextData,
    required dynamic spotAboveFlip,
    required bool depinRiskConfirmed,
  }) {
    final lockInfo = _getStrategyLockReason(
      strategy: strategy,
      contextData: contextData,
      spotAboveFlip: spotAboveFlip,
      depinRiskConfirmed: depinRiskConfirmed,
    );
    final strategyLabel = strategy.replaceAll('_', ' ');
    
    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.05),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: Colors.white24),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Header row with lock icon
          Row(
            children: [
              Expanded(
                child: Opacity(
                  opacity: 0.3,
                  child: Text(
                    strategyLabel,
                    style: const TextStyle(
                      color: Colors.white,
                      fontSize: 14,
                      fontWeight: FontWeight.bold,
                      fontFamily: 'JetBrains Mono',
                    ),
                  ),
                ),
              ),
              Container(
                padding: const EdgeInsets.all(4),
                decoration: BoxDecoration(
                  color: Colors.orange.withOpacity(0.2),
                  borderRadius: BorderRadius.circular(4),
                ),
                child: const Icon(
                  Icons.lock,
                  color: Colors.orange,
                  size: 16,
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          // Ghosted metrics
          Opacity(
            opacity: 0.3,
            child: Container(
              padding: const EdgeInsets.all(8),
              decoration: BoxDecoration(
                color: Colors.white.withOpacity(0.05),
                borderRadius: BorderRadius.circular(4),
              ),
              child: const Row(
                children: [
                  Text(
                    'Credit: \$--',
                    style: TextStyle(
                      color: Colors.white70,
                      fontSize: 11,
                      fontFamily: 'JetBrains Mono',
                    ),
                  ),
                  SizedBox(width: 16),
                  Text(
                    'ROC: --%',
                    style: TextStyle(
                      color: Colors.white70,
                      fontSize: 11,
                      fontFamily: 'JetBrains Mono',
                    ),
                  ),
                ],
              ),
            ),
          ),
          const SizedBox(height: 8),
          // Lock reason (separate section, not overlapping)
          Container(
            padding: const EdgeInsets.all(8),
            decoration: BoxDecoration(
              color: Colors.orange.withOpacity(0.15),
              borderRadius: BorderRadius.circular(4),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  lockInfo['reason'] ?? 'Locked',
                  style: const TextStyle(
                    color: Colors.orange,
                    fontSize: 11,
                    fontWeight: FontWeight.bold,
                    fontFamily: 'JetBrains Mono',
                  ),
                ),
                if (lockInfo['requirements'] != null && lockInfo['requirements']!.isNotEmpty)
                  Padding(
                    padding: const EdgeInsets.only(top: 4),
                    child: Text(
                      'Requires: ${lockInfo['requirements']}',
                      style: const TextStyle(
                        color: Colors.white60,
                        fontSize: 10,
                        fontFamily: 'JetBrains Mono',
                      ),
                    ),
                  ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildTradeIdeaCard(TradeIdea idea) {
    final isExpanded = _expandedCards[idea.id] ?? false;
    final best = idea.best;

    return Card(
      margin: const EdgeInsets.only(bottom: 16),
      color: const Color(0xFF161B22),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Pre-market warning banner
          if (idea.preMarket && idea.marketStatusNote != null)
            Container(
              width: double.infinity,
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
              decoration: BoxDecoration(
                color: Colors.orange.withOpacity(0.2),
                border: Border(
                  bottom: BorderSide(color: Colors.orange.withOpacity(0.5), width: 1),
                ),
              ),
              child: Row(
                children: [
                  const Icon(Icons.schedule, color: Colors.orange, size: 16),
                  const SizedBox(width: 8),
                  Expanded(
                    child: Text(
                      idea.marketStatusNote!,
                      style: const TextStyle(
                        color: Colors.orange,
                        fontSize: 12,
                        fontWeight: FontWeight.w500,
                        fontFamily: 'JetBrains Mono',
                      ),
                    ),
                  ),
                ],
              ),
            ),
          // Rest of the card content
          InkWell(
            onTap: () {
              setState(() {
                _expandedCards[idea.id] = !isExpanded;
              });
            },
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  // Header: Title + Badges
                  Row(
                    children: [
                      Expanded(
                        child: Text(
                          idea.label,
                          style: const TextStyle(
                            color: Colors.white,
                            fontSize: 16,
                            fontWeight: FontWeight.bold,
                            fontFamily: 'JetBrains Mono',
                          ),
                          softWrap: true,
                          overflow: TextOverflow.ellipsis,
                          maxLines: 2,
                        ),
                      ),
                      // Mode badge
                      Container(
                        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                        decoration: BoxDecoration(
                          color: idea.mode == 'INTRADAY_ONLY'
                              ? Colors.orange.withOpacity(0.2)
                              : idea.mode == 'HEDGE_REQUIRED'
                                  ? Colors.yellow.withOpacity(0.2)
                                  : Colors.green.withOpacity(0.2),
                          borderRadius: BorderRadius.circular(4),
                          border: Border.all(
                            color: idea.mode == 'INTRADAY_ONLY'
                                ? Colors.orange
                                : idea.mode == 'HEDGE_REQUIRED'
                                    ? Colors.yellow
                                    : Colors.green,
                            width: 1,
                          ),
                        ),
                        child: Text(
                          idea.mode.replaceAll('_', ' '),
                          style: TextStyle(
                            color: idea.mode == 'INTRADAY_ONLY'
                                ? Colors.orange
                                : idea.mode == 'HEDGE_REQUIRED'
                                    ? Colors.yellow
                                    : Colors.green,
                            fontSize: 11,
                            fontWeight: FontWeight.bold,
                            fontFamily: 'JetBrains Mono',
                          ),
                        ),
                      ),
                      const SizedBox(width: 8),
                      // Confidence badge
                      Container(
                        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                        decoration: BoxDecoration(
                          color: idea.confidence == 'HIGH'
                              ? Colors.green.withOpacity(0.2)
                              : idea.confidence == 'MEDIUM'
                                  ? Colors.orange.withOpacity(0.2)
                                  : Colors.red.withOpacity(0.2),
                          borderRadius: BorderRadius.circular(4),
                        ),
                        child: Text(
                          idea.confidence,
                          style: TextStyle(
                            color: idea.confidence == 'HIGH'
                                ? Colors.green
                                : idea.confidence == 'MEDIUM'
                                    ? Colors.orange
                                    : Colors.red,
                            fontSize: 11,
                            fontWeight: FontWeight.bold,
                            fontFamily: 'JetBrains Mono',
                          ),
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 12),
              
              // Summary metrics (collapsed view)
              if (best != null) ...[
                Container(
                  padding: const EdgeInsets.all(12),
                  decoration: BoxDecoration(
                    color: Colors.white.withOpacity(0.05),
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: Column(
                    children: [
                      Row(
                        children: [
                          Expanded(
                            child: _buildMetric('Credit', '\$${best.netCredit.toStringAsFixed(2)}', Colors.green),
                          ),
                          Expanded(
                            child: _buildMetric('Max Loss', '\$${best.maxLoss.toStringAsFixed(0)}', Colors.red),
                          ),
                          Expanded(
                            child: _buildMetric('ROC', '${(best.roc * 100).toStringAsFixed(1)}%', Colors.blue),
                          ),
                          Expanded(
                            child: _buildMetric('POP', '${(best.pop * 100).toStringAsFixed(0)}%', Colors.amber),
                          ),
                        ],
                      ),
                      const SizedBox(height: 8),
                      Row(
                        children: [
                          Expanded(
                            child: _buildMetric('EV', '\$${best.ev.toStringAsFixed(0)}', best.ev >= 0 ? Colors.green : Colors.red),
                          ),
                          Expanded(
                            child: _buildMetric('Liq', best.liquidityGrade, _getLiquidityColor(best.liquidityGrade)),
                          ),
                          Expanded(
                            child: _buildMetric('DTE', '${best.dte}', Colors.white70),
                          ),
                          Expanded(
                            child: _buildMetric('Exp', '${best.expiry.split('-')[1]}/${best.expiry.split('-')[2]}', Colors.white54),
                          ),
                        ],
                      ),
                    ],
                  ),
                ),
                const SizedBox(height: 8),
                // Wall buffer indicator
                if (best.wallBufferPct > 0)
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                    decoration: BoxDecoration(
                      color: Colors.blue.withOpacity(0.1),
                      borderRadius: BorderRadius.circular(4),
                    ),
                    child: Text(
                      'Wall buffer: ${(best.wallBufferPct * 100).toStringAsFixed(1)}%',
                      style: const TextStyle(
                        color: Colors.blue,
                        fontSize: 11,
                        fontFamily: 'JetBrains Mono',
                      ),
                    ),
                  ),
                const SizedBox(height: 4),
                // Timestamp
                Text(
                  'As of: ${_formatTimestamp(idea.asOf)}',
                  style: const TextStyle(
                    color: Colors.white38,
                    fontSize: 10,
                    fontFamily: 'JetBrains Mono',
                  ),
                ),
              ],
              
              // Expanded content
              if (isExpanded) ...[
                const Divider(color: Colors.white24, height: 24),
                // Why Allowed
                if (idea.reasons.isNotEmpty) ...[
                  const Text(
                    'Why Allowed:',
                    style: TextStyle(
                      color: Colors.white,
                      fontSize: 14,
                      fontWeight: FontWeight.bold,
                      fontFamily: 'JetBrains Mono',
                    ),
                  ),
                  const SizedBox(height: 8),
                  ...idea.reasons.map((reason) => Padding(
                    padding: const EdgeInsets.only(left: 8, bottom: 4),
                    child: Row(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Text('• ', style: TextStyle(color: Colors.white70, fontSize: 12)),
                        Expanded(
                          child: Text(
                            reason,
                            style: const TextStyle(
                              color: Colors.white70,
                              fontSize: 12,
                              fontFamily: 'JetBrains Mono',
                            ),
                            softWrap: true,
                          ),
                        ),
                      ],
                    ),
                  )),
                  const SizedBox(height: 16),
                ],
                
                // Reference Executions
                const Text(
                  'Reference Executions:',
                  style: TextStyle(
                    color: Colors.white,
                    fontSize: 14,
                    fontWeight: FontWeight.bold,
                    fontFamily: 'JetBrains Mono',
                  ),
                ),
                const SizedBox(height: 8),
                ...idea.executions.take(3).map((exec) => _buildExecutionCard(exec, idea)),
                
                // Risk notes
                if (idea.mode == 'INTRADAY_ONLY') ...[
                  const SizedBox(height: 16),
                  Container(
                    padding: const EdgeInsets.all(12),
                    decoration: BoxDecoration(
                      color: Colors.orange.withOpacity(0.1),
                      borderRadius: BorderRadius.circular(8),
                      border: Border.all(color: Colors.orange.withOpacity(0.3)),
                    ),
                    child: const Row(
                      children: [
                        Icon(Icons.warning, color: Colors.orange, size: 20),
                        SizedBox(width: 8),
                        Expanded(
                          child: Text(
                            'Not advised to hold through overnight events',
                            style: TextStyle(
                              color: Colors.orange,
                              fontSize: 12,
                              fontFamily: 'JetBrains Mono',
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
                if (idea.mode == 'HEDGE_REQUIRED') ...[
                  const SizedBox(height: 16),
                  Container(
                    padding: const EdgeInsets.all(12),
                    decoration: BoxDecoration(
                      color: Colors.yellow.withOpacity(0.1),
                      borderRadius: BorderRadius.circular(8),
                      border: Border.all(color: Colors.yellow.withOpacity(0.3)),
                    ),
                    child: const Row(
                      children: [
                        Icon(Icons.shield, color: Colors.yellow, size: 20),
                        SizedBox(width: 8),
                        Expanded(
                          child: Text(
                            'Hedge required for overnight exposure - consider protective positions',
                            style: TextStyle(
                              color: Colors.yellow,
                              fontSize: 12,
                              fontFamily: 'JetBrains Mono',
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              ],
              
              // Expand/collapse indicator
              Align(
                alignment: Alignment.centerRight,
                child: Icon(
                  isExpanded ? Icons.expand_less : Icons.expand_more,
                  color: Colors.white54,
                ),
              ),
            ],
          ),
        ),
          ),
        ],
      ),
    );
  }

  Widget _buildMetric(String label, String value, Color color) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.center,
      children: [
        Text(
          label,
          style: const TextStyle(
            color: Colors.white54,
            fontSize: 10,
            fontFamily: 'JetBrains Mono',
          ),
        ),
        const SizedBox(height: 4),
        Text(
          value,
          style: TextStyle(
            color: color,
            fontSize: 13,
            fontWeight: FontWeight.bold,
            fontFamily: 'JetBrains Mono',
          ),
          textAlign: TextAlign.center,
          overflow: TextOverflow.ellipsis,
        ),
      ],
    );
  }

  Color _getLiquidityColor(String grade) {
    switch (grade) {
      case 'A':
        return Colors.green;
      case 'B':
        return Colors.blue;
      case 'C':
        return Colors.orange;
      case 'D':
        return Colors.red;
      default:
        return Colors.white70;
    }
  }

  Widget _buildExecutionCard(ReferenceExecution exec, TradeIdea idea) {
    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: const Color(0xFF0D1117),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: Colors.white12),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Expiry info
          Row(
            children: [
              Text(
                'Expiry: ${exec.expiry} (${exec.dte} DTE)',
                style: const TextStyle(
                  color: Colors.white70,
                  fontSize: 12,
                  fontWeight: FontWeight.bold,
                  fontFamily: 'JetBrains Mono',
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          // Legs
          const Text(
            'Legs:',
            style: TextStyle(
              color: Colors.white54,
              fontSize: 11,
              fontFamily: 'JetBrains Mono',
            ),
          ),
          const SizedBox(height: 8),
          // Use a more mobile-friendly layout instead of Table
          // Strategy-specific info
          if (idea.strategy == 'IRON_CONDOR' || idea.strategy == 'BUTTERFLY')
            Container(
              margin: const EdgeInsets.only(bottom: 12),
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: Colors.blue.withOpacity(0.1),
                borderRadius: BorderRadius.circular(6),
                border: Border.all(
                  color: Colors.blue.withOpacity(0.3),
                  width: 1,
                ),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Icon(
                        idea.strategy == 'IRON_CONDOR' ? Icons.auto_awesome : Icons.center_focus_strong,
                        color: Colors.blue,
                        size: 16,
                      ),
                      const SizedBox(width: 8),
                      Text(
                        idea.strategy == 'IRON_CONDOR' ? 'Iron Condor' : 'Butterfly',
                        style: const TextStyle(
                          color: Colors.blue,
                          fontSize: 12,
                          fontWeight: FontWeight.bold,
                          fontFamily: 'JetBrains Mono',
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 8),
                  if (idea.strategy == 'IRON_CONDOR')
                    Text(
                      'Profit Zone: \$${exec.metrics.breakeven.toStringAsFixed(2)} - \$${(exec.metrics.breakeven + exec.metrics.width).toStringAsFixed(2)}',
                      style: const TextStyle(
                        color: Colors.white70,
                        fontSize: 11,
                        fontFamily: 'JetBrains Mono',
                      ),
                    )
                  else if (idea.strategy == 'BUTTERFLY' && exec.legs.length >= 2)
                    Text(
                      'Max Profit at: \$${exec.legs[1].strike.toStringAsFixed(2)}',
                      style: const TextStyle(
                        color: Colors.white70,
                        fontSize: 11,
                        fontFamily: 'JetBrains Mono',
                      ),
                    ),
                ],
              ),
            ),
          ...exec.legs.map((leg) => Container(
            margin: const EdgeInsets.only(bottom: 8),
            padding: const EdgeInsets.all(8),
            decoration: BoxDecoration(
              color: Colors.white.withOpacity(0.03),
              borderRadius: BorderRadius.circular(6),
              border: Border.all(
                color: leg.action == 'SELL' ? Colors.red.withOpacity(0.3) : Colors.green.withOpacity(0.3),
                width: 1,
              ),
            ),
            child: Row(
              children: [
                // Action badge
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                  decoration: BoxDecoration(
                    color: leg.action == 'SELL' ? Colors.red.withOpacity(0.2) : Colors.green.withOpacity(0.2),
                    borderRadius: BorderRadius.circular(4),
                  ),
                  child: Text(
                    leg.action,
                    style: TextStyle(
                      color: leg.action == 'SELL' ? Colors.red : Colors.green,
                      fontSize: 10,
                      fontWeight: FontWeight.bold,
                      fontFamily: 'JetBrains Mono',
                    ),
                  ),
                ),
                const SizedBox(width: 8),
                // Type
                Text(
                  leg.type,
                  style: const TextStyle(color: Colors.white70, fontSize: 11, fontFamily: 'JetBrains Mono'),
                ),
                const SizedBox(width: 12),
                // Strike
                Text(
                  'Strike: ${leg.strike.toStringAsFixed(0)}',
                  style: const TextStyle(color: Colors.white, fontSize: 11, fontWeight: FontWeight.bold, fontFamily: 'JetBrains Mono'),
                ),
                const Spacer(),
                // Mid price
                Column(
                  crossAxisAlignment: CrossAxisAlignment.end,
                  children: [
                    Text(
                      '\$${leg.priceMid.toStringAsFixed(2)}',
                      style: const TextStyle(color: Colors.white, fontSize: 12, fontWeight: FontWeight.bold, fontFamily: 'JetBrains Mono'),
                    ),
                    if (leg.delta != null)
                      Text(
                        'Δ ${leg.delta!.toStringAsFixed(3)}',
                        style: const TextStyle(color: Colors.white54, fontSize: 9, fontFamily: 'JetBrains Mono'),
                      ),
                  ],
                ),
              ],
            ),
          )),
          const SizedBox(height: 12),
          // Metrics block
          Container(
            padding: const EdgeInsets.all(8),
            decoration: BoxDecoration(
              color: Colors.white.withOpacity(0.05),
              borderRadius: BorderRadius.circular(6),
            ),
            child: Column(
              children: [
                Row(
                  children: [
                    Expanded(child: _buildMetric('Net Credit', '\$${exec.metrics.netCredit.toStringAsFixed(2)}', Colors.green)),
                    Expanded(child: _buildMetric('Max Loss', '\$${exec.metrics.maxLoss.toStringAsFixed(0)}', Colors.red)),
                  ],
                ),
                const SizedBox(height: 8),
                Row(
                  children: [
                    Expanded(child: _buildMetric('ROC', '${(exec.metrics.roc * 100).toStringAsFixed(1)}%', Colors.blue)),
                    Expanded(child: _buildMetric('POP', '${(exec.metrics.pop * 100).toStringAsFixed(0)}%', Colors.amber)),
                  ],
                ),
                const SizedBox(height: 8),
                Text(
                  'Breakeven: \$${exec.metrics.breakeven.toStringAsFixed(2)}',
                  style: const TextStyle(
                    color: Colors.white70,
                    fontSize: 11,
                    fontFamily: 'JetBrains Mono',
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  String _formatTimestamp(String isoString) {
    try {
      final dt = DateTime.parse(isoString);
      return '${dt.hour.toString().padLeft(2, '0')}:${dt.minute.toString().padLeft(2, '0')}';
    } catch (e) {
      return isoString;
    }
  }
}
