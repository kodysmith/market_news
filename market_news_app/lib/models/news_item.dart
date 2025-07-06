class NewsItem {
  final String headline;
  final String source;
  final String url;
  final String summary;

  NewsItem({
    required this.headline,
    required this.source,
    required this.url,
    required this.summary,
  });

  factory NewsItem.fromJson(Map<String, dynamic> json) {
    return NewsItem(
      headline: json['headline'] ?? '',
      source: json['source'] ?? '',
      url: json['url'] ?? '',
      summary: json['summary'] ?? '',
    );
  }
} 