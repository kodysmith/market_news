import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'package:market_news_app/models/report_data.dart';
import 'package:market_news_app/screens/market_insights_screen.dart';
import 'package:market_news_app/screens/market_sentiment_detail_screen.dart';
import 'package:market_news_app/screens/strategy_detail_screen.dart';
import 'package:market_news_app/screens/economic_calendar_screen.dart';
import 'package:market_news_app/screens/settings_screen.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';
import 'screens/news_screen.dart';
import 'dart:math' as math;
import 'package:market_news_app/models/vix_data.dart';
import 'dart:ui';
import 'package:firebase_core/firebase_core.dart';
import 'firebase_options.dart';
import 'package:firebase_messaging/firebase_messaging.dart';
import 'package:flutter/foundation.dart' show kIsWeb;

// Set the API base URL here for easy switching between local, staging, and production
const String apiBaseUrl = 'https://api-hvi4gdtdka-uc.a.run.app'; // TODO: Update this when moving to a custom domain
const String apiSecretKey = 'b7e2f8c4e1a94e2b8c9d4e7f2a1b3c4d';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  
  // Initialize Firebase
  await Firebase.initializeApp(
    options: DefaultFirebaseOptions.currentPlatform,
  );
  
  await dotenv.load(fileName: ".env");

  // Initialize Firebase Messaging (Android)
  await _initFirebaseMessaging();

  runApp(const MyApp());
}

Future<void> _initFirebaseMessaging() async {
  // Skip Firebase messaging initialization for web to avoid service worker errors
  if (kIsWeb) {
    print('Skipping Firebase messaging for web development');
    return;
  }
  
  FirebaseMessaging messaging = FirebaseMessaging.instance;

  // Request permission (not strictly needed for Android, but good practice)
  await messaging.requestPermission();

  // Get the token
  String? token = await messaging.getToken();
  print('FCM Token: ' + (token ?? 'null'));

  // Optionally: handle foreground messages
  FirebaseMessaging.onMessage.listen((message) {
    print('Received a foreground message: \n');
    print('Title: \${message.notification?.title}');
    print('Body: \${message.notification?.body}');
  });
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Market News',
      theme: ThemeData.dark(),
      home: const MainNavigation(),
    );
  }
}

class MainNavigation extends StatefulWidget {
  const MainNavigation({super.key});

  @override
  State<MainNavigation> createState() => _MainNavigationState();
}

class _MainNavigationState extends State<MainNavigation> {
  int _selectedIndex = 0;

  ReportData? _reportData;
  String? _error;
  bool _isLoading = false;

  @override
  void initState() {
    super.initState();
    _loadReportData();
  }

  Future<void> _loadReportData() async {
    if (!mounted) return;
    setState(() {
      _isLoading = true;
      _error = null;
    });
    try {
      final response = await http.get(
        Uri.parse('$apiBaseUrl/report.json'),
        headers: {'x-api-key': apiSecretKey},
      );
      if (!mounted) return;
      if (response.statusCode == 200) {
        final jsonString = response.body;
        final jsonData = json.decode(jsonString);
        if (jsonData == null || jsonData is! Map<String, dynamic>) {
          if (!mounted) return;
          setState(() {
            _error = 'Received invalid data from backend.';
          });
        } else {
          if (!mounted) return;
          setState(() {
            _reportData = ReportData.fromJson(jsonData);
          });
        }
      } else {
        if (!mounted) return;
        setState(() {
          _error = 'Failed to load report.json from backend (status: ${response.statusCode})';
        });
      }
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _error = 'Unable to connect to backend. Please check your connection and try again.\n\nDetails: $e';
      });
    } finally {
      if (!mounted) return;
      setState(() {
        _isLoading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final List<Widget> widgetOptions = <Widget>[
      DashboardScreen(),
      _reportData != null
          ? MarketInsightsScreen(reportData: _reportData!)
          : const Center(child: CircularProgressIndicator()),
      NewsScreen(),
      _reportData != null
          ? EconomicCalendarScreen(economicCalendar: _reportData!.economicCalendar)
          : const Center(child: CircularProgressIndicator()),
      SettingsScreen(),
    ];
    return Scaffold(
      body: _error != null
          ? Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Text(_error!, textAlign: TextAlign.center, style: TextStyle(color: Colors.red)),
                  SizedBox(height: 16),
                  ElevatedButton(
                    onPressed: _loadReportData,
                    child: Text('Retry'),
                  ),
                ],
              ),
            )
          : widgetOptions[_selectedIndex],
      bottomNavigationBar: BottomNavigationBar(
        type: BottomNavigationBarType.fixed,
        items: const <BottomNavigationBarItem>[
          BottomNavigationBarItem(
            icon: Icon(Icons.dashboard),
            label: 'Dashboard',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.insights),
            label: 'Insights',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.article),
            label: 'News',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.calendar_today),
            label: 'Calendar',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.settings),
            label: 'Settings',
          ),
        ],
        currentIndex: _selectedIndex,
        onTap: (index) {
          setState(() {
            _selectedIndex = index;
          });
        },
      ),
    );
  }
}

class DashboardScreen extends StatelessWidget {
  const DashboardScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return SimplifiedDashboard();
  }
}

class SimplifiedDashboard extends StatefulWidget {
  const SimplifiedDashboard({super.key});

  @override
  State<SimplifiedDashboard> createState() => _SimplifiedDashboardState();
}

class _SimplifiedDashboardState extends State<SimplifiedDashboard> {
  ReportData? _reportData;
  String? _error;
  bool _isLoading = false;

  @override
  void initState() {
    super.initState();
    _loadReportData();
  }

  Future<void> _loadReportData() async {
    if (!mounted) return;
    setState(() {
      _isLoading = true;
      _error = null;
    });
    try {
      final response = await http.get(
        Uri.parse('$apiBaseUrl/report.json'),
        headers: {'x-api-key': apiSecretKey},
      );
      if (!mounted) return;
      if (response.statusCode == 200) {
        final jsonString = response.body;
        final jsonData = json.decode(jsonString);
        if (jsonData == null || jsonData is! Map<String, dynamic>) {
          if (!mounted) return;
          setState(() {
            _error = 'Received invalid data from backend.';
          });
        } else {
          if (!mounted) return;
          setState(() {
            _reportData = ReportData.fromJson(jsonData);
          });
        }
      } else {
        if (!mounted) return;
        setState(() {
          _error = 'Failed to load report.json from backend (status: ${response.statusCode})';
        });
      }
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _error = 'Unable to connect to backend. Please check your connection and try again.\n\nDetails: $e';
      });
    } finally {
      if (!mounted) return;
      setState(() {
        _isLoading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    if (_error != null) {
      return Scaffold(
        body: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Text(_error!, textAlign: TextAlign.center, style: TextStyle(color: Colors.red)),
              SizedBox(height: 16),
              ElevatedButton(
                onPressed: _loadReportData,
                child: Text('Retry'),
              ),
            ],
          ),
        ),
      );
    }
    if (_reportData == null) {
      return const Scaffold(
        body: Center(
          child: CircularProgressIndicator(),
        ),
      );
    }
    final vixData = _reportData!.vixData;
    final indices = _reportData!.indices;
    final sentiment = _reportData!.marketSentiment;
    final topStrategies = _reportData!.topStrategies;
    return Scaffold(
      appBar: AppBar(
        title: const Text('Market News'),
      ),
      body: RefreshIndicator(
        onRefresh: _loadReportData,
        child: SingleChildScrollView(
          physics: const AlwaysScrollableScrollPhysics(),
          padding: const EdgeInsets.all(16.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              _buildMarketDirectionCard(sentiment),
              const SizedBox(height: 20),
              _buildVolatilityTrendCard(vixData),
              const SizedBox(height: 20),
              _buildFuturesCard(indices),
              const SizedBox(height: 20),
              _buildTopStrategyTypesCard(topStrategies),
              const SizedBox(height: 20),
              _buildIndicatorsCard(context),
              const SizedBox(height: 20),
              SizedBox(
                width: double.infinity,
                child: ElevatedButton(
                  onPressed: () {
                    Navigator.push(
                      context,
                      MaterialPageRoute(builder: (context) => MarketInsightsScreen(reportData: _reportData!)),
                    );
                  },
                  child: const Text('View Market Insights'),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildMarketDirectionCard(MarketSentiment sentiment) {
    // For now, use today's sentiment for all periods. In future, use historical data.
    String day = sentiment.sentiment;
    String week = sentiment.sentiment;
    String month = sentiment.sentiment;
    Color getColor(String s) {
      if (s.toLowerCase().contains('bull')) return Colors.green;
      if (s.toLowerCase().contains('bear')) return Colors.red;
      return Colors.grey;
    }
    IconData getIcon(String s) {
      if (s.toLowerCase().contains('bull')) return Icons.trending_up;
      if (s.toLowerCase().contains('bear')) return Icons.trending_down;
      return Icons.horizontal_rule;
    }
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Market Direction', style: Theme.of(context).textTheme.headlineSmall),
            const SizedBox(height: 10),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceAround,
              children: [
                _buildDirectionColumn('Day', day, getColor(day), getIcon(day)),
                _buildDirectionColumn('Week', week, getColor(week), getIcon(week)),
                _buildDirectionColumn('Month', month, getColor(month), getIcon(month)),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildDirectionColumn(String label, String sentiment, Color color, IconData icon) {
    return Column(
      children: [
        Text(label, style: const TextStyle(fontWeight: FontWeight.bold)),
        Icon(icon, color: color, size: 32),
        Text(sentiment, style: TextStyle(color: color, fontWeight: FontWeight.bold)),
      ],
    );
  }

  Widget _buildVolatilityTrendCard(List<VixData> vixData) {
    if (vixData.isEmpty) {
      return Card(
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Text('No volatility data available.'),
        ),
      );
    }
    final last7 = vixData.take(7).toList();
    final values = last7.map((v) => v.close).toList();
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Volatility Trend (VIX, 7d)', style: Theme.of(context).textTheme.headlineSmall),
            const SizedBox(height: 10),
            SizedBox(
              height: 48,
              child: CustomPaint(
                painter: _SparklinePainter(values),
                child: Container(),
              ),
            ),
            const SizedBox(height: 6),
            if (values.isNotEmpty)
              Text('Latest: ${values.first.toStringAsFixed(2)}', style: const TextStyle(fontWeight: FontWeight.bold)),
          ],
        ),
      ),
    );
  }

  Widget _buildFuturesCard(List<Map<String, dynamic>> indices) {
    // Show S&P 500, Nasdaq, Dow
    List<String> symbols = ['^GSPC', '^IXIC', '^DJI'];
    Map<String, String> names = {
      '^GSPC': 'S&P 500',
      '^IXIC': 'Nasdaq',
      '^DJI': 'Dow',
    };
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Market Futures', style: Theme.of(context).textTheme.headlineSmall),
            const SizedBox(height: 10),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceAround,
              children: symbols.map((s) {
                final idx = indices.firstWhere((i) => i['symbol'] == s, orElse: () => {});
                final price = idx['price']?.toStringAsFixed(2) ?? '--';
                final change = idx['change'] ?? 0.0;
                final color = change > 0 ? Colors.green : (change < 0 ? Colors.red : Colors.grey);
                final icon = change > 0 ? Icons.arrow_upward : (change < 0 ? Icons.arrow_downward : Icons.horizontal_rule);
                return Column(
                  children: [
                    Text(names[s] ?? s, style: const TextStyle(fontWeight: FontWeight.bold)),
                    Icon(icon, color: color, size: 28),
                    Text(' 24$price', style: TextStyle(color: color, fontWeight: FontWeight.bold)),
                    Text('Chg: $change', style: TextStyle(color: color)),
                  ],
                );
              }).toList(),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildTopStrategyTypesCard(List<TopStrategy> strategies) {
    if (strategies.isEmpty) return const SizedBox();
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text(
          'Top Strategy Types for Today',
          style: TextStyle(fontSize: 22, fontWeight: FontWeight.bold),
        ),
        const SizedBox(height: 4),
        Text(
          'Most likely to succeed based on current market conditions.',
          style: TextStyle(fontSize: 14, color: Colors.blueGrey),
        ),
        const SizedBox(height: 16),
        ...strategies.take(3).map((strategy) => _buildStrategyTypeCard(strategy)),
      ],
    );
  }

  Widget _buildStrategyTypeCard(TopStrategy strategy) {
    Color color = _getStrategyColor(strategy.name);
    IconData icon = _getStrategyIcon(strategy.name);
    return Card(
      elevation: 4,
      margin: const EdgeInsets.only(bottom: 12),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Row(
          children: [
            Icon(icon, color: color, size: 32),
            const SizedBox(width: 14),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    strategy.name,
                    style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold, color: color),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    strategy.description,
                    style: TextStyle(fontSize: 13, color: Colors.grey.shade700),
                  ),
                ],
              ),
            ),
            const SizedBox(width: 10),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
              decoration: BoxDecoration(
                color: color.withOpacity(0.13),
                borderRadius: BorderRadius.circular(8),
              ),
              child: Text(
                'Score: ${strategy.score.toStringAsFixed(1)}',
                style: TextStyle(fontSize: 14, fontWeight: FontWeight.w700, color: color),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildIndicatorsCard(BuildContext context) {
    try {
      return Card(
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                'Key Indicators',
                style: Theme.of(context).textTheme.headlineSmall,
              ),
              const SizedBox(height: 10),
              ..._reportData!.marketSentiment.indicators.map((indicator) => ListTile(
                    title: Text(indicator.name),
                    trailing: Text(indicator.price),
                  ))
            ],
          ),
        ),
      );
    } catch (e) {
      return Text("Error in Indicators Card: $e");
    }
  }

  Color _getStrategyColor(String name) {
    if (name.toLowerCase().contains('scalp')) return Colors.blue.shade700;
    if (name.toLowerCase().contains('covered')) return Colors.orange.shade700;
    if (name.toLowerCase().contains('bull')) return Colors.green.shade700;
    if (name.toLowerCase().contains('bear')) return Colors.red.shade700;
    return Colors.purple.shade700;
  }

  IconData _getStrategyIcon(String name) {
    if (name.toLowerCase().contains('scalp')) return Icons.show_chart;
    if (name.toLowerCase().contains('covered')) return Icons.call;
    if (name.toLowerCase().contains('bull')) return Icons.trending_up;
    if (name.toLowerCase().contains('bear')) return Icons.trending_down;
    return Icons.auto_graph;
  }
}

class _SparklinePainter extends CustomPainter {
  final List<double> values;
  _SparklinePainter(this.values);
  @override
  void paint(Canvas canvas, Size size) {
    if (values.isEmpty) return;
    final paint = Paint()
      ..color = Colors.orange
      ..strokeWidth = 3
      ..style = PaintingStyle.stroke;
    final minV = values.reduce(math.min);
    final maxV = values.reduce(math.max);
    final range = (maxV - minV).abs() < 1e-3 ? 1.0 : (maxV - minV);
    final points = <Offset>[];
    for (int i = 0; i < values.length; i++) {
      final x = i * size.width / (values.length - 1);
      final y = size.height - ((values[i] - minV) / range * size.height);
      points.add(Offset(x, y));
    }
    canvas.drawPoints(PointMode.lines, points, paint);
  }
  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => true;
}
