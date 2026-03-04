import 'dart:async';
import 'package:flutter/services.dart';
import 'package:local_auth/local_auth.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:supabase_flutter/supabase_flutter.dart';

class AuthService {
  static SupabaseClient get _supabase => Supabase.instance.client;
  static final _localAuth = LocalAuthentication();

  static const _kBiometricEnabled = 'biometric_lock_enabled';
  static const _redirectUrl = 'marketnews://login-callback';

  // ---------------------------------------------------------------------------
  // Auth state
  // ---------------------------------------------------------------------------

  static User? get currentUser => _supabase.auth.currentUser;
  static Session? get currentSession => _supabase.auth.currentSession;
  static Stream<AuthState> get onAuthStateChange =>
      _supabase.auth.onAuthStateChange;

  /// Headers to attach to API calls so the backend can verify the user.
  static Map<String, String> get authHeaders {
    final session = _supabase.auth.currentSession;
    return {
      if (session != null) 'Authorization': 'Bearer ${session.accessToken}',
    };
  }

  // ---------------------------------------------------------------------------
  // Google OAuth via Supabase (opens browser, redirects back via deep link)
  // ---------------------------------------------------------------------------

  static Future<bool> signInWithGoogle() async {
    return await _supabase.auth.signInWithOAuth(
      OAuthProvider.google,
      redirectTo: _redirectUrl,
      authScreenLaunchMode: LaunchMode.externalApplication,
    );
  }

  // ---------------------------------------------------------------------------
  // Biometric authentication
  // ---------------------------------------------------------------------------

  static Future<bool> canUseBiometrics() async {
    try {
      final canCheck = await _localAuth.canCheckBiometrics;
      final isSupported = await _localAuth.isDeviceSupported();
      return canCheck || isSupported;
    } catch (_) {
      return false;
    }
  }

  static Future<List<BiometricType>> availableBiometrics() async {
    try {
      return await _localAuth.getAvailableBiometrics();
    } catch (_) {
      return [];
    }
  }

  static Future<bool> authenticateWithBiometrics() async {
    try {
      return await _localAuth.authenticate(
        localizedReason: 'Authenticate to access Market News',
        options: const AuthenticationOptions(
          stickyAuth: true,
          biometricOnly: false,
        ),
      );
    } on PlatformException {
      return false;
    }
  }

  // ---------------------------------------------------------------------------
  // Biometric preference (persisted)
  // ---------------------------------------------------------------------------

  static Future<bool> isBiometricEnabled() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getBool(_kBiometricEnabled) ?? false;
  }

  static Future<void> setBiometricEnabled(bool enabled) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool(_kBiometricEnabled, enabled);
  }

  // ---------------------------------------------------------------------------
  // Sign out
  // ---------------------------------------------------------------------------

  static Future<void> signOut() async {
    await _supabase.auth.signOut();
  }
}
