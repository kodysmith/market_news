import 'package:flutter/material.dart';

class EconomicCalendarScreen extends StatelessWidget {
  const EconomicCalendarScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Economic Calendar')),
      body: const Center(
        child: Text('Economic calendar coming soon!', style: TextStyle(fontSize: 18)),
      ),
    );
  }
} 