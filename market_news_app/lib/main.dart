import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'dart:convert';
import 'package:market_news_app/models/report_data.dart';
import 'package:market_news_app/screens/market_insights_screen.dart';
import 'package:market_news_app/screens/gamma_scalping_screen.dart';
import 'package:market_news_app/screens/market_sentiment_detail_screen.dart';
import 'package:market_news_app/screens/strategy_detail_screen.dart';
import 'package:market_news_app/screens/economic_calendar_screen.dart';
import 'package:market_news_app/screens/settings_screen.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await dotenv.load(fileName: ".env");
  runApp(const MyApp());
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

  final List<Widget> _screens = [
    const DashboardScreen(),
    MarketInsightsScreenPlaceholder(),
    EconomicCalendarScreen(),
    SettingsScreen(),
  ];

  void _onItemTapped(int index) {
    setState(() {
      _selectedIndex = index;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: _screens[_selectedIndex],
      bottomNavigationBar: BottomNavigationBar(
        currentIndex: _selectedIndex,
        onTap: _onItemTapped,
        type: BottomNavigationBarType.fixed,
        items: const [
          BottomNavigationBarItem(
            icon: Icon(Icons.dashboard),
            label: 'Dashboard',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.insights),
            label: 'Insights',
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

  @override
  void initState() {
    super.initState();
    _loadReportData();
  }

  Future<void> _loadReportData() async {
    try {
      final jsonString = await rootBundle.loadString('assets/results.json');
      final data = json.decode(jsonString);
      setState(() {
        _reportData = ReportData.fromJson(data);
      });
    } catch (e) {
      setState(() {
        _error = "Error loading or parsing data: $e";
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    if (_error != null) {
      return Scaffold(
        body: Center(
          child: Text(_error!),
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
    try {
      return Scaffold(
        appBar: AppBar(
          title: const Text('Market News'),
        ),
        body: SingleChildScrollView(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              _buildSentimentCard(context),
              const SizedBox(height: 20),
              _buildTopStrategies(context),
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
              const SizedBox(height: 20),
              _buildTradeIdeasList(context),
            ],
          ),
        ),
      );
    } catch (e) {
      return Scaffold(
        body: Center(
          child: Text('Error: $e'),
        ),
      );
    }
  }

  Widget _buildSentimentCard(BuildContext context) {
    try {
      return Card(
        child: InkWell(
          onTap: () {
            Navigator.push(
              context,
              MaterialPageRoute(
                builder: (context) => MarketSentimentDetailScreen(
                  marketSentiment: _reportData!.marketSentiment,
                ),
              ),
            );
          },
          child: Padding(
            padding: const EdgeInsets.all(16.0),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Expanded(
                      child: Text(
                        'Market Sentiment',
                        style: Theme.of(context).textTheme.headlineSmall,
                      ),
                    ),
                    Icon(
                      Icons.info_outline,
                      color: Colors.grey.shade600,
                      size: 20,
                    ),
                  ],
                ),
                const SizedBox(height: 10),
                Text(
                  _reportData!.marketSentiment.sentiment,
                  style: Theme.of(context).textTheme.titleLarge,
                ),
                const SizedBox(height: 8),
                Text(
                  'Tap for detailed analysis',
                  style: TextStyle(
                    fontSize: 12,
                    color: Colors.grey.shade600,
                    fontStyle: FontStyle.italic,
                  ),
                ),
              ],
            ),
          ),
        ),
      );
    } catch (e) {
      return Text("Error in Sentiment Card: $e");
    }
  }

  Widget _buildTopStrategies(BuildContext context) {
    final strategies = _reportData!.topStrategies;
    if (strategies.isEmpty) return const SizedBox();
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text(
          'Top Strategies for Today',
          style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
        ),
        const SizedBox(height: 12),
        ...strategies.take(3).map((strategy) => _buildStrategyCard(context, strategy)),
      ],
    );
  }

  Widget _buildStrategyCard(BuildContext context, TopStrategy strategy) {
    Color color = _getStrategyColor(strategy.name);
    IconData icon = _getStrategyIcon(strategy.name);
    return Card(
      elevation: 3,
      margin: const EdgeInsets.only(bottom: 12),
      child: InkWell(
        onTap: () {
          Navigator.push(
            context,
            MaterialPageRoute(
              builder: (context) => StrategyDetailScreen(strategy: strategy),
            ),
          );
        },
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Row(
            children: [
              Icon(icon, color: color, size: 32),
              const SizedBox(width: 16),
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
              const SizedBox(width: 12),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
                decoration: BoxDecoration(
                  color: color.withOpacity(0.12),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Text(
                  'Score: ${strategy.score.toStringAsFixed(1)}',
                  style: TextStyle(fontSize: 14, fontWeight: FontWeight.w600, color: color),
                ),
              ),
            ],
          ),
        ),
      ),
    );
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

  Widget _buildTradeIdeasList(BuildContext context) {
    try {
      return Card(
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                'Trade Ideas',
                style: Theme.of(context).textTheme.headlineSmall,
              ),
              const SizedBox(height: 10),
              ..._reportData!.tradeIdeas.map((trade) => ListTile(
                    title: Text('${trade.ticker} - ${trade.strategy}'),
                    subtitle: Text(trade.details),
                    trailing: Text(trade.cost > 0 ? 'Credit' : 'Debit'),
                  ))
            ],
          ),
        ),
      );
    } catch (e) {
      return Text("Error in Trade Ideas List: $e");
    }
  }
}
