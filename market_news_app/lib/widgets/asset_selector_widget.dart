/// Asset Selector Widget
/// 
/// Unified asset selection widget with dropdown, text input, and quick-select buttons.
/// Provides consistent UI/UX across all screens.
/// 
/// Usage:
///   AssetSelectorWidget(
///     onAssetChanged: (asset) => print('Selected: $asset'),
///   )

import 'dart:async';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import '../services/asset_selection_service.dart';

class AssetSelectorWidget extends StatefulWidget {
  final ValueChanged<String>? onAssetChanged;
  final bool showQuickButtons;
  final bool compact;
  
  const AssetSelectorWidget({
    super.key,
    this.onAssetChanged,
    this.showQuickButtons = true,
    this.compact = false,
  });

  @override
  State<AssetSelectorWidget> createState() => _AssetSelectorWidgetState();
}

/// Debounce delay before submitting symbol from text input (ms).
const int _kInputDebounceMs = 400;

class _AssetSelectorWidgetState extends State<AssetSelectorWidget> {
  late AssetSelectionService _service;
  late TextEditingController _textController;
  Timer? _debounceTimer;

  @override
  void initState() {
    super.initState();
    _service = AssetSelectionService();
    _textController = TextEditingController(text: _service.selectedAsset);
    _service.addListener(_onAssetChanged);
  }

  @override
  void dispose() {
    _service.removeListener(_onAssetChanged);
    _textController.dispose();
    _debounceTimer?.cancel();
    super.dispose();
  }

  void _onAssetChanged() {
    if (_textController.text != _service.selectedAsset) {
      _textController.text = _service.selectedAsset;
    }
    if (widget.onAssetChanged != null) {
      widget.onAssetChanged!(_service.selectedAsset);
    }
  }

  /// Only update service and notify after user stops typing (debounce).
  /// Do NOT call _service.setAsset on every keystroke — other listeners (e.g. Cockpit)
  /// would reload the whole page and make typing impossible.
  void _handleTextChange(String value) {
    final ticker = value.toUpperCase().trim();
    _debounceTimer?.cancel();

    if (ticker.isEmpty) return;

    // Debounce: update service only after user stops typing for 400ms.
    // This prevents the shared AssetSelectionService from notifying the Cockpit
    // on every keystroke (which would trigger _loadData() and a full reload).
    _debounceTimer = Timer(const Duration(milliseconds: _kInputDebounceMs), () {
      final currentTicker = _textController.text.toUpperCase().trim();
      if (currentTicker.isEmpty) return;
      _service.setAsset(currentTicker);
      // Parent callback is fired by _onAssetChanged when service notifies
    });
  }

  void _handleTextSubmitted(String value) {
    final ticker = value.toUpperCase().trim();
    if (ticker.isNotEmpty) {
      _debounceTimer?.cancel();
      _service.setAsset(ticker);
      if (widget.onAssetChanged != null) {
        widget.onAssetChanged!(ticker);
      }
    }
  }

  void _handleQuickSelect(String ticker) {
    _debounceTimer?.cancel();
    _textController.text = ticker;
    _service.setAsset(ticker);
    if (widget.onAssetChanged != null) {
      widget.onAssetChanged!(ticker);
    }
  }
  
  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, constraints) {
        return ListenableBuilder(
          listenable: _service,
          builder: (context, _) {
            // Ensure selected asset is always in the list
            final availableAssets = _service.availableAssets.contains(_service.selectedAsset)
                ? _service.availableAssets
                : [..._service.availableAssets, _service.selectedAsset];
            
            if (widget.compact) {
              return _buildCompactSelector(availableAssets);
            } else {
              return _buildFullSelector(availableAssets, constraints);
            }
          },
        );
      },
    );
  }
  
  Widget _buildCompactSelector(List<String> availableAssets) {
    return LayoutBuilder(
      builder: (context, constraints) {
        final maxWidth = constraints.maxWidth != double.infinity
            ? constraints.maxWidth
            : MediaQuery.of(context).size.width;
        // Reserve fixed space: input 90, gap 12, dropdown 168+ so dropdown is never squished
        const inputWidth = 90.0;
        const gapWidth = 12.0;
        const dropdownMinWidth = 168.0;

        return SizedBox(
          width: maxWidth,
          child: Row(
            children: [
              // Text field: type any symbol
              SizedBox(
                width: inputWidth,
                child: TextField(
                  controller: _textController,
                  style: const TextStyle(
                    color: Colors.white,
                    fontSize: 14,
                    fontWeight: FontWeight.w600,
                    fontFamily: 'JetBrains Mono',
                  ),
                  textAlign: TextAlign.center,
                  textCapitalization: TextCapitalization.characters,
                  decoration: InputDecoration(
                    hintText: 'Symbol',
                    hintStyle: TextStyle(color: Colors.white.withOpacity(0.4), fontSize: 12),
                    border: OutlineInputBorder(borderRadius: BorderRadius.circular(6)),
                    filled: true,
                    fillColor: const Color(0xFF0D1117),
                    contentPadding: const EdgeInsets.symmetric(horizontal: 6, vertical: 8),
                    isDense: true,
                  ),
                  inputFormatters: [
                    FilteringTextInputFormatter.allow(RegExp(r'[A-Za-z0-9.]')),
                    LengthLimitingTextInputFormatter(12),
                  ],
                  onChanged: _handleTextChange,
                  onSubmitted: _handleTextSubmitted,
                ),
              ),
              const SizedBox(width: gapWidth),
              // Dropdown: fixed minimum width so it's never collapsed; take remaining space
              Expanded(
                child: ConstrainedBox(
                  constraints: const BoxConstraints(minWidth: dropdownMinWidth),
                  child: DropdownButtonFormField<String>(
                    value: _service.selectedAsset,
                    decoration: InputDecoration(
                      labelText: 'Or pick',
                      labelStyle: TextStyle(color: Colors.white.withOpacity(0.7), fontSize: 11),
                      border: OutlineInputBorder(borderRadius: BorderRadius.circular(6)),
                      contentPadding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
                    ),
                    dropdownColor: const Color(0xFF161B22),
                    isExpanded: true,
                    items: availableAssets
                        .map((asset) => DropdownMenuItem(
                              value: asset,
                              child: Text(asset, style: const TextStyle(fontFamily: 'JetBrains Mono', fontSize: 13)),
                            ))
                        .toList(),
                    onChanged: (value) {
                      if (value != null) {
                        _debounceTimer?.cancel();
                        _textController.text = value;
                        _service.setAsset(value);
                        if (widget.onAssetChanged != null) {
                          widget.onAssetChanged!(value);
                        }
                      }
                    },
                  ),
                ),
              ),
            ],
          ),
        );
      },
    );
  }
  
  Widget _buildFullSelector(List<String> availableAssets, BoxConstraints constraints) {
    // Ensure we have bounded width constraints
    final maxWidth = constraints.maxWidth != double.infinity 
        ? constraints.maxWidth 
        : MediaQuery.of(context).size.width;
    
    return ConstrainedBox(
      constraints: BoxConstraints(
        maxWidth: maxWidth,
        minWidth: 0,
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        mainAxisSize: MainAxisSize.min,
        children: [
          // Row with bounded width to prevent unconstrained layout errors
          Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              const Text(
                'Symbol: ',
                style: TextStyle(color: Colors.white70, fontSize: 14, fontFamily: 'JetBrains Mono'),
              ),
              const SizedBox(width: 8),
              Container(
                width: 120,
                padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
                decoration: BoxDecoration(
                  color: const Color(0xFF0D1117),
                  borderRadius: BorderRadius.circular(6),
                  border: Border.all(color: Colors.white24),
                ),
                child: TextField(
                  controller: _textController,
                  style: const TextStyle(
                    color: Colors.white,
                    fontSize: 16,
                    fontWeight: FontWeight.bold,
                    fontFamily: 'JetBrains Mono',
                  ),
                  textAlign: TextAlign.center,
                  textCapitalization: TextCapitalization.characters,
                  decoration: const InputDecoration(
                    hintText: 'Enter symbol',
                    hintStyle: TextStyle(color: Colors.white38, fontSize: 14),
                    border: InputBorder.none,
                    contentPadding: EdgeInsets.zero,
                    isDense: true,
                  ),
                  inputFormatters: [
                    // Allow a-z and A-Z so keystrokes are accepted; textCapitalization shows uppercase
                    FilteringTextInputFormatter.allow(RegExp(r'[A-Za-z0-9]')),
                    LengthLimitingTextInputFormatter(5),
                  ],
                  onChanged: _handleTextChange,
                  onSubmitted: _handleTextSubmitted,
                ),
              ),
              const SizedBox(width: 8),
              Flexible(
                fit: FlexFit.loose,
                child: Builder(
                  builder: (context) {
                    return DropdownButtonFormField<String>(
                      value: _service.selectedAsset,
                      decoration: const InputDecoration(
                        labelText: 'Select Asset',
                        border: OutlineInputBorder(),
                        contentPadding: EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                      ),
                      items: availableAssets
                          .map((asset) => DropdownMenuItem(
                                value: asset,
                                child: Text(asset),
                              ))
                          .toList(),
                      onChanged: (value) {
                        if (value != null) {
                          _textController.text = value;
                          _service.setAsset(value);
                          if (widget.onAssetChanged != null) {
                            widget.onAssetChanged!(value);
                          }
                        }
                      },
                    );
                  },
                ),
              ),
            ],
          ),
          if (widget.showQuickButtons) ...[
              const SizedBox(height: 8),
              Wrap(
                spacing: 4,
                runSpacing: 4,
                children: _service.suggestedAssets.map((ticker) {
                  final isSelected = _service.selectedAsset == ticker;
                  return InkWell(
                    onTap: () => _handleQuickSelect(ticker),
                    child: Container(
                      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                      decoration: BoxDecoration(
                        color: isSelected
                            ? const Color(0xFF58A6FF).withOpacity(0.3)
                            : Colors.white.withOpacity(0.05),
                        borderRadius: BorderRadius.circular(4),
                        border: Border.all(
                          color: isSelected
                              ? const Color(0xFF58A6FF)
                              : Colors.white24,
                          width: isSelected ? 1.5 : 1,
                        ),
                      ),
                      child: Text(
                        ticker,
                        style: TextStyle(
                          color: isSelected ? Colors.white : Colors.white70,
                          fontSize: 11,
                          fontWeight: isSelected ? FontWeight.bold : FontWeight.normal,
                          fontFamily: 'JetBrains Mono',
                        ),
                      ),
                    ),
                  );
                }).toList(),
              ),
            ],
          ],
        ),
      );
  }
}
