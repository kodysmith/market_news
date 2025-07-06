import 'package:flutter/material.dart';

class EconomicCalendarScreen extends StatelessWidget {
  final List<Map<String, dynamic>> economicCalendar;
  const EconomicCalendarScreen({super.key, required this.economicCalendar});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Economic Calendar')),
      body: economicCalendar.isEmpty
          ? const Center(child: Text('No economic events available.', style: TextStyle(fontSize: 18)))
          : ListView.separated(
              padding: const EdgeInsets.all(16),
              itemCount: economicCalendar.length,
              separatorBuilder: (context, i) => const Divider(),
              itemBuilder: (context, i) {
                final event = economicCalendar[i];
                return ListTile(
                  title: Text(event['event'] ?? ''),
                  subtitle: Text('${event['date'] ?? ''} | ${event['country'] ?? ''}'),
                  trailing: Text(event['impact'] ?? '', style: TextStyle(
                    color: event['impact'] == 'High' ? Colors.red : event['impact'] == 'Medium' ? Colors.orange : Colors.grey,
                  )),
                  onTap: () {
                    showDialog(
                      context: context,
                      builder: (context) {
                        return AlertDialog(
                          title: Text(event['event'] ?? ''),
                          content: SingleChildScrollView(
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Text('Date: ${event['date'] ?? ''}'),
                                Text('Country: ${event['country'] ?? ''}'),
                                Text('Currency: ${event['currency'] ?? ''}'),
                                Text('Impact: ${event['impact'] ?? ''}'),
                                if (event['previous'] != null) Text('Previous: ${event['previous']}'),
                                if (event['estimate'] != null) Text('Estimate: ${event['estimate']}'),
                                if (event['actual'] != null) Text('Actual: ${event['actual']}'),
                                if (event['change'] != null) Text('Change: ${event['change']}'),
                                if (event['changePercentage'] != null) Text('Change %: ${event['changePercentage']}'),
                                if (event['unit'] != null) Text('Unit: ${event['unit']}'),
                              ],
                            ),
                          ),
                          actions: [
                            TextButton(
                              onPressed: () => Navigator.of(context).pop(),
                              child: const Text('Close'),
                            ),
                          ],
                        );
                      },
                    );
                  },
                );
              },
            ),
    );
  }
} 