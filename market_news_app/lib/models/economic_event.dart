class EconomicEvent {
  final String date;
  final String time;
  final String event;
  final String country;
  final String impact;

  EconomicEvent({
    required this.date,
    required this.time,
    required this.event,
    required this.country,
    required this.impact,
  });

  factory EconomicEvent.fromJson(Map<String, dynamic> json) {
    return EconomicEvent(
      date: json['date'] as String,
      time: json['time'] as String,
      event: json['event'] as String,
      country: json['country'] as String,
      impact: json['impact'] as String,
    );
  }
}
