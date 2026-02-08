# Fisher Score API Contract

Base path: `/fisher`. All responses are JSON.

## GET /fisher/snapshot

Returns the latest Fisher score snapshot for a ticker.

**Query parameters**

| Parameter | Required | Description |
|----------|----------|-------------|
| ticker   | Yes      | Stock ticker symbol (e.g. AAPL) |

**Response (200)**

```json
{
  "company_id": "uuid",
  "ticker": "AAPL",
  "snapshot_at": "2025-02-05T12:00:00",
  "total_score": 6.5,
  "version": 1,
  "points": {
    "1": { "score": 7.0, "confidence": 0.9, "evidence": ["SEC revenue YoY"], "feature_values": { "revenue_growth_yoy_pct": 12.5 } },
    "3": { "score": 6.0, "confidence": 0.85, "evidence": ["SEC R&D expense"], "feature_values": { "rd_pct_revenue": 5.2 } }
  },
  "category_scores": {
    "growth": 6.5,
    "financials": 6.2,
    "capital_allocation": 6.0,
    "moat": 6.5
  }
}
```

**Errors**

- `400`: Missing `ticker` parameter.
- `404`: Ticker not in Fisher universe or no snapshot for ticker.
- `500` / `503`: Server or database error.

---

## GET /fisher/delta

Returns what changed since the previous quarter (point score deltas between latest and previous snapshot).

**Query parameters**

| Parameter | Required | Description |
|----------|----------|-------------|
| ticker   | Yes      | Stock ticker symbol |

**Response (200)**

```json
{
  "ticker": "AAPL",
  "current_snapshot_at": "2025-02-05T12:00:00",
  "previous_snapshot_at": "2024-11-01T12:00:00",
  "point_deltas": {
    "1": 0.5,
    "3": -0.2,
    "5": 0.1
  }
}
```

If only one or zero snapshots exist:

```json
{
  "ticker": "AAPL",
  "current_snapshot_at": "2025-02-05T12:00:00",
  "previous_snapshot_at": null,
  "point_deltas": {},
  "message": "Only one or zero snapshots; no delta available."
}
```

**Errors**

- `400`: Missing `ticker` parameter.
- `404`: Ticker not in Fisher universe.
- `500` / `503`: Server or database error.

---

## GET /fisher/evidence

Returns evidence and feature values for a single Fisher point.

**Query parameters**

| Parameter | Required | Description |
|----------|----------|-------------|
| ticker   | Yes      | Stock ticker symbol |
| point_id | Yes      | Fisher point id (1–15, e.g. 13) |

**Response (200)**

```json
{
  "ticker": "AAPL",
  "point_id": "13",
  "score": 6.5,
  "confidence": 0.9,
  "evidence": ["SEC shares, SBC, buybacks"],
  "feature_values": {
    "shares_yoy_pct": -1.2,
    "sbc_pct_revenue": 2.1,
    "buybacks": -5000000000
  }
}
```

**Errors**

- `400`: Missing `ticker` or `point_id` parameter.
- `404`: Ticker not in universe, no snapshot, or point_id not in snapshot.
- `500` / `503`: Server or database error.

---

## GET /fisher/universe

Returns the list of tickers in the Fisher universe (S&P 500 or config).

**Query parameters**

None.

**Response (200)**

```json
{
  "tickers": ["AAPL", "MSFT", "GOOGL", ...]
}
```

**Errors**

- `500`: Server error (e.g. config not found).

---

## Request/response conventions

- All endpoints use GET; no request body.
- Optional `x-api-key` header may be used if auth is enabled for Fisher routes.
- Numeric scores are 0–10; confidence 0–1. Dates/timestamps are ISO 8601.
