import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'package:url_launcher/url_launcher.dart';
import 'package:supabase_flutter/supabase_flutter.dart';
import '../models/news_item.dart';
import 'package:flutter/foundation.dart' show kIsWeb;
import '../main.dart' show apiBaseUrl, apiSecretKey;

class NewsScreen extends StatefulWidget {
  const NewsScreen({super.key});

  @override
  State<NewsScreen> createState() => _NewsScreenState();
}

class _NewsScreenState extends State<NewsScreen> {
  late Future<List<NewsItem>> _futureNews;
  bool _refreshInProgress = false;

  @override
  void initState() {
    super.initState();
    _futureNews = fetchNews();
  }

  /// Trigger backend news refresh (congressional symbols → FMP → Supabase), then reload list.
  Future<void> _triggerRefreshNews() async {
    if (_refreshInProgress) return;
    setState(() => _refreshInProgress = true);
    try {
      final response = await http.post(
        Uri.parse('$apiBaseUrl/news/refresh'),
        headers: {'x-api-key': apiSecretKey},
      );
      final body = json.decode(response.body) as Map<String, dynamic>? ?? {};
      if (response.statusCode == 200 && body['success'] == true) {
        setState(() {
          _futureNews = fetchNews();
        });
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(content: Text('News refreshed (${body['articles'] ?? 0} articles)')),
          );
        }
      } else {
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(content: Text(body['error']?.toString() ?? 'Refresh failed')),
          );
        }
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Refresh failed: $e')),
        );
      }
    } finally {
      if (mounted) setState(() => _refreshInProgress = false);
    }
  }

  /// Try Supabase news_feed first (when configured); else API news.json.
  Future<List<NewsItem>> fetchNews() async {
    if (_supabaseAvailable()) {
      final fromDb = await _fetchNewsFromSupabase();
      if (fromDb != null) return fromDb;
    }
    return _fetchNewsFromApi();
  }

  bool _supabaseAvailable() {
    try {
      Supabase.instance;
      return true;
    } catch (_) {
      return false;
    }
  }

  Future<List<NewsItem>?> _fetchNewsFromSupabase() async {
    try {
      final rows = await Supabase.instance.client
          .from('news_feed')
          .select('headline, source, url, summary')
          .order('created_at', ascending: false)
          .limit(100);
      if (rows.isEmpty) return null;
      return rows.map((r) {
        final m = Map<String, dynamic>.from(r as Map);
        return NewsItem(
          headline: m['headline'] as String? ?? '',
          source: m['source'] as String? ?? '',
          url: m['url'] as String? ?? '',
          summary: m['summary'] as String? ?? '',
        );
      }).toList();
    } catch (_) {
      return null;
    }
  }

  Future<List<NewsItem>> _fetchNewsFromApi() async {
    try {
      final response = await http.get(
        Uri.parse('$apiBaseUrl/news.json'),
        headers: {'x-api-key': apiSecretKey},
      );
      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        if (data == null || data is! List) {
          throw Exception('Received invalid data from backend.');
        }
        return data.map<NewsItem>((item) => NewsItem.fromJson(Map<String, dynamic>.from(item as Map))).toList();
      } else {
        throw Exception('Failed to load news from backend (status: ${response.statusCode})');
      }
    } catch (e) {
      throw Exception('Unable to connect to backend. Please check your connection and try again.\n\nDetails: $e');
    }
  }

  Future<void> _launchUrl(String url) async {
    final uri = Uri.parse(url);
    if (kIsWeb) {
      await launchUrl(uri, webOnlyWindowName: '_blank');
      return;
    }
    if (await canLaunchUrl(uri)) {
      await launchUrl(uri, mode: LaunchMode.externalApplication);
    } else {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Could not launch URL')),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Market News'),
        actions: [
          IconButton(
            icon: _refreshInProgress
                ? const SizedBox(
                    width: 20,
                    height: 20,
                    child: CircularProgressIndicator(strokeWidth: 2),
                  )
                : const Icon(Icons.refresh),
            onPressed: _refreshInProgress ? null : _triggerRefreshNews,
            tooltip: 'Refresh news (fetches from server)',
          ),
        ],
      ),
      body: RefreshIndicator(
        onRefresh: () async {
          setState(() {
            _futureNews = fetchNews();
          });
          await _futureNews;
        },
        child: FutureBuilder<List<NewsItem>>(
          future: _futureNews,
          builder: (context, snapshot) {
            if (snapshot.connectionState == ConnectionState.waiting) {
              return const Center(child: CircularProgressIndicator());
            } else if (snapshot.hasError) {
              return Center(
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Text('Error: ${snapshot.error}', textAlign: TextAlign.center, style: TextStyle(color: Colors.red)),
                    SizedBox(height: 16),
                    ElevatedButton(
                      onPressed: () {
                        setState(() {
                          _futureNews = fetchNews();
                        });
                      },
                      child: Text('Retry'),
                    ),
                  ],
                ),
              );
            } else if (!snapshot.hasData || snapshot.data!.isEmpty) {
              return const Center(child: Text('No news available.'));
            }
            final newsList = snapshot.data!;
            return ListView.separated(
              physics: const AlwaysScrollableScrollPhysics(),
              itemCount: newsList.length,
              separatorBuilder: (context, i) => const Divider(),
              itemBuilder: (context, i) {
                final news = newsList[i];
                return InkWell(
                  onTap: () => _launchUrl(news.url),
                  child: Card(
                    margin: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                    child: ListTile(
                      title: Text(
                        news.headline,
                        style: const TextStyle(
                          fontWeight: FontWeight.bold,
                          color: Colors.blue,
                          decoration: TextDecoration.underline,
                        ),
                      ),
                      subtitle: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          const SizedBox(height: 6),
                          Text(news.summary),
                          const SizedBox(height: 8),
                          Text(
                            news.source,
                            style: const TextStyle(fontSize: 12, color: Colors.grey),
                          ),
                        ],
                      ),
                    ),
                  ),
                );
              },
            );
          },
        ),
      ),
    );
  }
} 