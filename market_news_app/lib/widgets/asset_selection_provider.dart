/// Asset Selection Provider
/// 
/// InheritedWidget that provides AssetSelectionService to the widget tree.
/// Allows any widget to access the current asset selection service.
/// 
/// Usage:
///   final service = AssetSelectionProvider.of(context);
///   final currentAsset = service.selectedAsset;

import 'package:flutter/material.dart';
import '../services/asset_selection_service.dart';

class AssetSelectionProvider extends InheritedWidget {
  final AssetSelectionService service;
  
  const AssetSelectionProvider({
    super.key,
    required this.service,
    required super.child,
  });
  
  static AssetSelectionService of(BuildContext context) {
    final provider = context.dependOnInheritedWidgetOfExactType<AssetSelectionProvider>();
    assert(provider != null, 'AssetSelectionProvider not found in widget tree');
    return provider!.service;
  }
  
  @override
  bool updateShouldNotify(AssetSelectionProvider oldWidget) {
    return service != oldWidget.service;
  }
}
