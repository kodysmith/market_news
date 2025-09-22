import 'package:flutter/material.dart';
import 'package:market_news_app/models/economic_event.dart';
import 'package:market_news_app/services/economic_calendar_service.dart';

class EconomicCalendarScreen extends StatefulWidget {
  const EconomicCalendarScreen({super.key});

  @override
  State<EconomicCalendarScreen> createState() => _EconomicCalendarScreenState();
}

class _EconomicCalendarScreenState extends State<EconomicCalendarScreen>
    with TickerProviderStateMixin {
  late TabController _tabController;
  List<EconomicEvent> _allEvents = [];
  List<EconomicEvent> _filteredEvents = [];
  bool _isLoading = true;
  String? _error;
  String _selectedImpact = 'all'; // 'all', 'high', 'medium', 'low'

  @override
  void initState() {
    super.initState();
    _tabController = TabController(length: 4, vsync: this);
    _loadEconomicCalendar();
  }

  @override
  void dispose() {
    _tabController.dispose();
    super.dispose();
  }

  Future<void> _loadEconomicCalendar() async {
    if (!mounted) return;

    setState(() {
      _isLoading = true;
      _error = null;
    });

    try {
      final events = await EconomicCalendarService.getEconomicCalendar(limit: 100);
      if (!mounted) return;

      setState(() {
        _allEvents = events;
        _filteredEvents = _filterEvents(events, _selectedImpact);
        _isLoading = false;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _error = 'Failed to load economic calendar: $e';
        _isLoading = false;
      });
    }
  }

  List<EconomicEvent> _filterEvents(List<EconomicEvent> events, String impact) {
    if (impact == 'all') return events;
    return EconomicCalendarService.filterByImpact(events, impact);
  }

  void _onImpactFilterChanged(String impact) {
    setState(() {
      _selectedImpact = impact;
      _filteredEvents = _filterEvents(_allEvents, impact);
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Economic Calendar'),
        bottom: TabBar(
          controller: _tabController,
          tabs: const [
            Tab(text: 'All'),
            Tab(text: 'High'),
            Tab(text: 'Medium'),
            Tab(text: 'Low'),
          ],
          onTap: (index) {
            final impacts = ['all', 'high', 'medium', 'low'];
            _onImpactFilterChanged(impacts[index]);
          },
        ),
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh),
            onPressed: _loadEconomicCalendar,
            tooltip: 'Refresh Calendar',
          ),
        ],
      ),
      body: _buildBody(),
    );
  }

  Widget _buildBody() {
    if (_error != null) {
      return Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Text(_error!, textAlign: TextAlign.center, style: const TextStyle(color: Colors.red)),
            const SizedBox(height: 16),
            ElevatedButton(
              onPressed: _loadEconomicCalendar,
              child: const Text('Retry'),
            ),
          ],
        ),
      );
    }

    if (_isLoading) {
      return const Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            CircularProgressIndicator(),
            SizedBox(height: 16),
            Text('Loading economic calendar...'),
          ],
        ),
      );
    }

    if (_filteredEvents.isEmpty) {
      return Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(Icons.calendar_today, size: 64, color: Colors.grey),
            const SizedBox(height: 16),
            Text(
              'No ${_selectedImpact == 'all' ? '' : _selectedImpact + ' impact '}economic events available.',
              style: const TextStyle(fontSize: 18, color: Colors.grey),
              textAlign: TextAlign.center,
            ),
          ],
        ),
      );
    }

    return RefreshIndicator(
      onRefresh: _loadEconomicCalendar,
      child: ListView.separated(
        padding: const EdgeInsets.all(16),
        itemCount: _filteredEvents.length,
        separatorBuilder: (context, i) => const Divider(height: 1),
        itemBuilder: (context, i) {
          final event = _filteredEvents[i];
          return _buildEventCard(event);
        },
      ),
    );
  }

  Widget _buildEventCard(EconomicEvent event) {
    return Card(
      elevation: 2,
      margin: const EdgeInsets.symmetric(vertical: 4),
      child: InkWell(
        onTap: () => _showEventDetails(event),
        borderRadius: BorderRadius.circular(8),
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                    decoration: BoxDecoration(
                      color: event.impactColor.withOpacity(0.1),
                      borderRadius: BorderRadius.circular(12),
                      border: Border.all(color: event.impactColor.withOpacity(0.3)),
                    ),
                    child: Text(
                      '${event.impactEmoji} ${event.impact.toUpperCase()}',
                      style: TextStyle(
                        color: event.impactColor,
                        fontSize: 12,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ),
                  const Spacer(),
                  Text(
                    event.time,
                    style: const TextStyle(color: Colors.grey, fontSize: 12),
                  ),
                ],
              ),
              const SizedBox(height: 8),
              Text(
                event.title,
                style: const TextStyle(
                  fontSize: 16,
                  fontWeight: FontWeight.w600,
                ),
              ),
              const SizedBox(height: 4),
              Row(
                children: [
                  Text(
                    event.date,
                    style: const TextStyle(color: Colors.grey, fontSize: 14),
                  ),
                  const SizedBox(width: 8),
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                    decoration: BoxDecoration(
                      color: Colors.blue.shade50,
                      borderRadius: BorderRadius.circular(8),
                    ),
                    child: Text(
                      event.currency,
                      style: TextStyle(
                        color: Colors.blue.shade700,
                        fontSize: 12,
                        fontWeight: FontWeight.w500,
                      ),
                    ),
                  ),
                ],
              ),
              if (event.actual != null) ...[
                const SizedBox(height: 8),
                Row(
                  children: [
                    const Text(
                      'Actual: ',
                      style: TextStyle(fontWeight: FontWeight.w500),
                    ),
                    Text(
                      event.actual!.toStringAsFixed(2),
                      style: const TextStyle(fontWeight: FontWeight.w600),
                    ),
                    if (event.change != null) ...[
                      const SizedBox(width: 12),
                      Icon(
                        event.change! > 0 ? Icons.arrow_upward : Icons.arrow_downward,
                        size: 16,
                        color: event.change! > 0 ? Colors.green : Colors.red,
                      ),
                      Text(
                        event.change!.abs().toStringAsFixed(2),
                        style: TextStyle(
                          color: event.change! > 0 ? Colors.green : Colors.red,
                          fontWeight: FontWeight.w500,
                        ),
                      ),
                    ],
                  ],
                ),
              ],
            ],
          ),
        ),
      ),
    );
  }

  void _showEventDetails(EconomicEvent event) {
    showDialog(
      context: context,
      builder: (context) {
        return AlertDialog(
          title: Row(
            children: [
              Text(event.impactEmoji),
              const SizedBox(width: 8),
              Expanded(child: Text(event.title)),
            ],
          ),
          content: SingleChildScrollView(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              mainAxisSize: MainAxisSize.min,
              children: [
                _buildDetailRow('Date', event.date),
                _buildDetailRow('Time', event.time),
                _buildDetailRow('Currency', event.currency),
                _buildDetailRow('Impact', event.impact.toUpperCase()),
                _buildDetailRow('Source', event.source),
                if (event.actual != null)
                  _buildDetailRow('Actual', event.actual!.toStringAsFixed(2)),
                if (event.previous != null)
                  _buildDetailRow('Previous', event.previous!.toStringAsFixed(2)),
                if (event.forecast != null)
                  _buildDetailRow('Forecast', event.forecast!.toStringAsFixed(2)),
                if (event.change != null)
                  _buildDetailRow(
                    'Change',
                    '${event.change! > 0 ? '+' : ''}${event.change!.toStringAsFixed(2)}',
                    color: event.change! > 0 ? Colors.green : Colors.red,
                  ),
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
  }

  Widget _buildDetailRow(String label, String value, {Color? color}) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(
            width: 80,
            child: Text(
              '$label:',
              style: const TextStyle(fontWeight: FontWeight.w500),
            ),
          ),
          Expanded(
            child: Text(
              value,
              style: TextStyle(
                color: color,
                fontWeight: color != null ? FontWeight.w600 : null,
              ),
            ),
          ),
        ],
      ),
    );
  }
} 