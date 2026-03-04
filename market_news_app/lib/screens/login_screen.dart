import 'package:flutter/material.dart';
import 'package:market_news_app/services/auth_service.dart';

class LoginScreen extends StatefulWidget {
  final VoidCallback? onBiometricUnlock;

  const LoginScreen({super.key, this.onBiometricUnlock});

  @override
  State<LoginScreen> createState() => _LoginScreenState();
}

class _LoginScreenState extends State<LoginScreen>
    with SingleTickerProviderStateMixin {
  bool _isLoading = false;
  String? _error;
  bool _canBiometric = false;
  late AnimationController _fadeController;
  late Animation<double> _fadeAnimation;

  @override
  void initState() {
    super.initState();
    _fadeController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 800),
    );
    _fadeAnimation = CurvedAnimation(
      parent: _fadeController,
      curve: Curves.easeOut,
    );
    _fadeController.forward();
    _checkBiometric();
  }

  Future<void> _checkBiometric() async {
    final canBio = await AuthService.canUseBiometrics();
    final bioEnabled = await AuthService.isBiometricEnabled();
    final hasSession = AuthService.currentSession != null;
    if (mounted) {
      setState(() {
        _canBiometric = canBio && bioEnabled && hasSession;
      });
    }
  }

  @override
  void dispose() {
    _fadeController.dispose();
    super.dispose();
  }

  Future<void> _signInWithGoogle() async {
    setState(() {
      _isLoading = true;
      _error = null;
    });

    try {
      final launched = await AuthService.signInWithGoogle();
      if (!launched && mounted) {
        setState(() {
          _error = 'Could not open sign-in page. Please try again.';
          _isLoading = false;
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _error = 'Sign-in failed: $e';
          _isLoading = false;
        });
      }
    }
  }

  Future<void> _unlockWithBiometric() async {
    final passed = await AuthService.authenticateWithBiometrics();
    if (passed) {
      widget.onBiometricUnlock?.call();
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: FadeTransition(
        opacity: _fadeAnimation,
        child: SafeArea(
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 32),
            child: Column(
              children: [
                const Spacer(flex: 3),
                _buildLogo(),
                const SizedBox(height: 28),
                const Text(
                  'Market News',
                  style: TextStyle(
                    fontSize: 34,
                    fontWeight: FontWeight.w700,
                    letterSpacing: -0.5,
                  ),
                ),
                const SizedBox(height: 8),
                Text(
                  'AI-powered trading intelligence',
                  style: TextStyle(
                    fontSize: 16,
                    color: Colors.grey[500],
                    fontWeight: FontWeight.w400,
                  ),
                ),
                const Spacer(flex: 3),
                if (_error != null) ...[
                  _buildError(),
                  const SizedBox(height: 16),
                ],
                if (_canBiometric) ...[
                  _buildBiometricButton(),
                  const SizedBox(height: 12),
                  Text(
                    'or',
                    style: TextStyle(color: Colors.grey[600], fontSize: 14),
                  ),
                  const SizedBox(height: 12),
                ],
                _buildGoogleButton(),
                const Spacer(flex: 2),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildLogo() {
    return Container(
      width: 88,
      height: 88,
      decoration: BoxDecoration(
        gradient: LinearGradient(
          colors: [
            Colors.blue.shade700,
            Colors.blue.shade400,
          ],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
        borderRadius: BorderRadius.circular(22),
        boxShadow: [
          BoxShadow(
            color: Colors.blue.withOpacity(0.3),
            blurRadius: 20,
            offset: const Offset(0, 8),
          ),
        ],
      ),
      child: const Icon(
        Icons.show_chart_rounded,
        size: 48,
        color: Colors.white,
      ),
    );
  }

  Widget _buildError() {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.red.withOpacity(0.1),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Colors.red.withOpacity(0.3)),
      ),
      child: Row(
        children: [
          const Icon(Icons.error_outline, color: Colors.red, size: 20),
          const SizedBox(width: 8),
          Expanded(
            child: Text(
              _error!,
              style: const TextStyle(color: Colors.red, fontSize: 13),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildBiometricButton() {
    return SizedBox(
      width: double.infinity,
      height: 52,
      child: FilledButton.icon(
        onPressed: _unlockWithBiometric,
        icon: const Icon(Icons.fingerprint, size: 24),
        label: const Text(
          'Unlock with Biometrics',
          style: TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
        ),
        style: FilledButton.styleFrom(
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(14),
          ),
        ),
      ),
    );
  }

  Widget _buildGoogleButton() {
    return SizedBox(
      width: double.infinity,
      height: 52,
      child: OutlinedButton(
        onPressed: _isLoading ? null : _signInWithGoogle,
        style: OutlinedButton.styleFrom(
          side: BorderSide(color: Colors.grey.shade700),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(14),
          ),
        ),
        child: _isLoading
            ? const SizedBox(
                width: 22,
                height: 22,
                child: CircularProgressIndicator(strokeWidth: 2),
              )
            : Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  _buildGoogleIcon(),
                  const SizedBox(width: 12),
                  const Text(
                    'Continue with Google',
                    style: TextStyle(
                      fontSize: 16,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                ],
              ),
      ),
    );
  }

  Widget _buildGoogleIcon() {
    return SizedBox(
      width: 20,
      height: 20,
      child: CustomPaint(painter: _GoogleLogoPainter()),
    );
  }
}

class _GoogleLogoPainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final double w = size.width;
    final double h = size.height;
    final center = Offset(w / 2, h / 2);
    final radius = w / 2;

    final bgPaint = Paint()
      ..color = Colors.white
      ..style = PaintingStyle.fill;
    canvas.drawCircle(center, radius, bgPaint);

    final textPainter = TextPainter(
      text: TextSpan(
        text: 'G',
        style: TextStyle(
          fontSize: w * 0.7,
          fontWeight: FontWeight.w700,
          color: const Color(0xFF4285F4),
        ),
      ),
      textDirection: TextDirection.ltr,
    );
    textPainter.layout();
    textPainter.paint(
      canvas,
      Offset(
        (w - textPainter.width) / 2,
        (h - textPainter.height) / 2,
      ),
    );
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}
