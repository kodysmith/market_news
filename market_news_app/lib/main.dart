import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'dart:convert';
import 'package:market_news_app/models/report_data.dart';
import 'package:market_news_app/screens/market_insights_screen.dart';
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
      home: const SimplifiedDashboard(),
    );
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
              _buildIndicatorsCard(context),
              const SizedBox(height: 20),
              SizedBox(
                width: double.infinity,
                child: ElevatedButton(
                  onPressed: () {
                    Navigator.push(
                      context,
                      MaterialPageRoute(builder: (context) => const MarketInsightsScreen()),
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
          child: Text("Error building UI: $e"),
        ),
      );
    }
  }

  Widget _buildSentimentCard(BuildContext context) {
    try {
      return Card(
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                'Market Sentiment',
                style: Theme.of(context).textTheme.headlineSmall,
              ),
              const SizedBox(height: 10),
              Text(
                _reportData!.marketSentiment.sentiment,
                style: Theme.of(context).textTheme.titleLarge,
              ),
            ],
          ),
        ),
      );
    } catch (e) {
      return Text("Error in Sentiment Card: $e");
    }
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
