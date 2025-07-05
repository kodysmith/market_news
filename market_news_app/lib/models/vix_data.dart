class VixData {
  final String date;
  final double close;

  VixData({
    required this.date,
    required this.close,
  });

  factory VixData.fromJson(Map<String, dynamic> json) {
    return VixData(
      date: json['date'] as String,
      close: (json['close'] as num).toDouble(),
    );
  }
}
