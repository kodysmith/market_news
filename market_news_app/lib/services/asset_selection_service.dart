/// Asset Selection Service
/// 
/// Singleton service that manages the currently selected asset (stock/index)
/// across all screens. Uses ChangeNotifier for reactive state management.
/// 
/// Usage:
///   final service = AssetSelectionService();
///   service.setAsset('SPY');
///   final currentAsset = service.selectedAsset;
///   service.addListener(() => print('Asset changed!'));

import 'package:flutter/foundation.dart';

class AssetSelectionService extends ChangeNotifier {
  static final AssetSelectionService _instance = AssetSelectionService._internal();
  factory AssetSelectionService() => _instance;
  AssetSelectionService._internal();
  
  String _selectedAsset = 'SPY';
  final List<String> _availableAssets = [
    // Major indices
    'SPY', 'QQQ', 'IWM', 'SPX', 'NDX', 'XSP',
    // Popular stocks
    'AAPL', 'MSFT', 'NVDA', 'TSLA', 'META', 'GOOGL', 'AMZN',
    'NFLX', 'AMD', 'INTC', 'JPM', 'BAC', 'V', 'MA',
  ];
  
  /// Get the currently selected asset
  String get selectedAsset => _selectedAsset;
  
  /// Get the list of available assets (read-only)
  List<String> get availableAssets => List.unmodifiable(_availableAssets);
  
  /// Get suggested assets for quick selection (top 6)
  List<String> get suggestedAssets => _availableAssets.take(6).toList();
  
  /// Set the selected asset
  /// Automatically adds to available assets if not already present
  void setAsset(String asset) {
    final normalizedAsset = asset.toUpperCase().trim();
    
    if (normalizedAsset.isEmpty) {
      return;
    }
    
    if (_selectedAsset != normalizedAsset) {
      _selectedAsset = normalizedAsset;
      
      // Add to available assets if not already present
      if (!_availableAssets.contains(normalizedAsset)) {
        _availableAssets.add(normalizedAsset);
        // Keep list sorted for consistency
        _availableAssets.sort();
      }
      
      notifyListeners();
    }
  }
  
  /// Add an asset to the available list without selecting it
  void addAsset(String asset) {
    final normalizedAsset = asset.toUpperCase().trim();
    if (normalizedAsset.isNotEmpty && !_availableAssets.contains(normalizedAsset)) {
      _availableAssets.add(normalizedAsset);
      _availableAssets.sort();
      notifyListeners();
    }
  }
  
  /// Check if an asset is in the available list
  bool hasAsset(String asset) {
    return _availableAssets.contains(asset.toUpperCase().trim());
  }
}
