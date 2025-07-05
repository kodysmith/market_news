import 'package:flutter/material.dart';

class DailyStrategyGuide extends StatelessWidget {
  const DailyStrategyGuide({super.key});

  @override
  Widget build(BuildContext context) {
    final today = DateTime.now().weekday;
    final days = ['S', 'M', 'T', 'W', 'T', 'F', 'S'];
    final strategies = {
      1: 'Premium',
      2: 'Premium',
      3: 'Mixed',
      4: 'Scalp',
      5: 'Scalp',
      6: 'Analysis',
      7: 'Analysis',
    };

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceAround,
              children: List.generate(7, (index) {
                final dayIndex = index + 1;
                final isToday = dayIndex == today;
                return Column(
                  children: [
                    Text(
                      days[index],
                      style: TextStyle(
                        fontWeight: isToday ? FontWeight.bold : FontWeight.normal,
                        color: isToday ? Colors.blue : Colors.white,
                      ),
                    ),
                    const SizedBox(height: 4),
                    Text(
                      strategies[dayIndex]!,
                      style: TextStyle(
                        fontSize: 12,
                        color: isToday ? Colors.blue : Colors.white70,
                      ),
                    ),
                  ],
                );
              }),
            ),
          ],
        ),
      ),
    );
  }
}
