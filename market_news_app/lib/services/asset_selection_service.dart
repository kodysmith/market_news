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
  static const List<String> _availableAssets = [
    'XSP', 'SPX', 'SPY', 'NDX', 'AMZN', 'GOOGL', 'UNH', 'GLD',
  ];
  
  /// Get the currently selected asset
  String get selectedAsset => _selectedAsset;
  
  /// Get the list of available assets (read-only)
  List<String> get availableAssets => List.unmodifiable(_availableAssets);

  /// Get suggested assets for quick selection (first 6 from list)
  List<String> get suggestedAssets => _availableAssets.take(6).toList();
  
  /// Set the selected asset
  void setAsset(String asset) {
    final normalizedAsset = asset.toUpperCase().trim();
    
    if (normalizedAsset.isEmpty) {
      return;
    }
    
    if (_selectedAsset != normalizedAsset) {
      _selectedAsset = normalizedAsset;
      notifyListeners();
    }
  }
  
  /// Check if an asset is in the available list
  bool hasAsset(String asset) {
    return _availableAssets.contains(asset.toUpperCase().trim());
  }

  /// No-op: dropdown list is fixed; kept for API compatibility.
  void addAsset(String asset) {}
}
