/// Fisher 15 points (Common Stocks and Uncommon Profits): labels and tooltips for how to get info when N/A.
class FisherPointMeta {
  final int id;
  final String label;
  final String tooltip;

  const FisherPointMeta({
    required this.id,
    required this.label,
    required this.tooltip,
  });
}

/// All 15 Fisher points in display order (1..15).
const List<FisherPointMeta> fisherPointsInOrder = [
  FisherPointMeta(id: 1, label: 'Sales growth potential', tooltip: 'Use revenue YoY from 10-K/10-Q; we derive from SEC facts when available.'),
  FisherPointMeta(id: 2, label: 'New products/processes', tooltip: 'Read 10-K Item 1 (Business), R&D and product pipeline; earnings call Q&A.'),
  FisherPointMeta(id: 3, label: 'R&D effectiveness', tooltip: 'R&D % of revenue from 10-K; we use SEC facts when available.'),
  FisherPointMeta(id: 4, label: 'Sales organization', tooltip: '10-K description of sales force, channels, geographic mix; industry reports.'),
  FisherPointMeta(id: 5, label: 'Profit margin', tooltip: 'Gross/operating margin from 10-K; we use SEC facts when available.'),
  FisherPointMeta(id: 6, label: 'Maintain/improve margins', tooltip: 'Compare margins YoY; we use prior vs current filing when available.'),
  FisherPointMeta(id: 7, label: 'Labor/personnel relations', tooltip: '10-K risk factors, employee counts, turnover; Glassdoor/union news.'),
  FisherPointMeta(id: 8, label: 'Executive relations', tooltip: 'Proxy (DEF 14A): compensation, succession, board; governance ratings.'),
  FisherPointMeta(id: 9, label: 'Depth of management', tooltip: '10-K Item 10 (directors/officers), proxy bios, succession disclosure.'),
  FisherPointMeta(id: 10, label: 'Cost analysis & controls', tooltip: '10-K MD&A on cost structure, internal controls (Item 9A), audit opinion.'),
  FisherPointMeta(id: 11, label: 'Competitive position', tooltip: '10-K Item 1 (competition), market share data, industry reports.'),
  FisherPointMeta(id: 12, label: 'Short vs long-range profits', tooltip: 'MD&A outlook, earnings guidance, capital allocation (buybacks, capex).'),
  FisherPointMeta(id: 13, label: 'Dilution risk', tooltip: 'Share count trend, SBC, option grants; we use shares/buybacks when available.'),
  FisherPointMeta(id: 14, label: 'Management communication', tooltip: 'Earnings call tone, 8-K frequency, guidance changes; read transcripts.'),
  FisherPointMeta(id: 15, label: 'Management integrity', tooltip: 'Proxy, related-party transactions, restatements, enforcement actions.'),
];

/// Lookup by point id (1..15).
FisherPointMeta fisherPointMeta(int id) {
  if (id >= 1 && id <= 15) return fisherPointsInOrder[id - 1];
  return fisherPointsInOrder[0]; // fallback
}
