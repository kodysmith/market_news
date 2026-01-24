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
import 'asset_selection_provider.dart';

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
  
  void _handleTextChange(String value) {
    final ticker = value.toUpperCase().trim();
    if (ticker != _service.selectedAsset && ticker.isNotEmpty && ticker.length >= 1) {
      // Cancel previous timer
      _debounceTimer?.cancel();
      
      // Set new asset immediately for UI
      _service.setAsset(ticker);
      
      // Debounce the callback (wait 500ms after user stops typing)
      _debounceTimer = Timer(const Duration(milliseconds: 500), () {
        final currentTicker = _textController.text.toUpperCase().trim();
        if (currentTicker == ticker && currentTicker.isNotEmpty) {
          if (widget.onAssetChanged != null) {
            widget.onAssetChanged!(currentTicker);
          }
        }
      });
    }
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
        
        return SizedBox(
          width: maxWidth,
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              Flexible(
                fit: FlexFit.loose,
                child: DropdownButtonFormField<String>(
                  value: _service.selectedAsset,
                  decoration: const InputDecoration(
                    labelText: 'Asset',
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
                    FilteringTextInputFormatter.allow(RegExp(r'[A-Z0-9]')),
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
