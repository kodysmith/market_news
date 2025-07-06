import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'package:url_launcher/url_launcher.dart';
import '../models/news_item.dart';
import 'package:flutter/foundation.dart' show kIsWeb;
import 'dart:html' as html; // Only for web

class NewsScreen extends StatefulWidget {
  const NewsScreen({super.key});

  @override
  State<NewsScreen> createState() => _NewsScreenState();
}

class _NewsScreenState extends State<NewsScreen> {
  late Future<List<NewsItem>> _futureNews;

  @override
  void initState() {
    super.initState();
    _futureNews = fetchNews();
  }

  Future<List<NewsItem>> fetchNews() async {
    final response = await http.get(Uri.parse('http://localhost:5000/news.json'));
    if (response.statusCode == 200) {
      final List<dynamic> data = json.decode(response.body);
      return data.map((item) => NewsItem.fromJson(item)).toList();
    } else {
      throw Exception('Failed to load news');
    }
  }

  Future<void> _launchUrl(String url) async {
    if (kIsWeb) {
      html.window.open(url, '_blank');
      return;
    }
    final uri = Uri.parse(url);
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
      appBar: AppBar(title: const Text('Market News')),
      body: FutureBuilder<List<NewsItem>>(
        future: _futureNews,
        builder: (context, snapshot) {
          if (snapshot.connectionState == ConnectionState.waiting) {
            return const Center(child: CircularProgressIndicator());
          } else if (snapshot.hasError) {
            return Center(child: Text('Error: \\${snapshot.error}'));
          } else if (!snapshot.hasData || snapshot.data!.isEmpty) {
            return const Center(child: Text('No news available.'));
          }
          final newsList = snapshot.data!;
          return ListView.separated(
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
    );
  }
} 