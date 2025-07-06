# Market News App — System Architecture

## Overview
A modular, cross-platform market analytics system with a Python backend (data, logic, API) and a Flutter frontend (UI, UX). Designed for future migration of the backend to a cloud platform.

---

## Current Architecture

### 1. **Backend (Python)**
- **Responsibilities:**
  - Fetches and analyzes market data (Yahoo Finance, VIX, Treasuries, etc.)
  - Runs strategy scoring (gamma scalping, premium selling, etc.)
  - Generates daily reports (JSON/HTML) with:
    - Market sentiment
    - Top strategies (with scores, tickers, setups)
    - Trade ideas and risk metrics
  - Serves data via Flask API (for live use)
- **Key Modules:**
  - `generate_report.py`: Orchestrates data collection and report generation
  - `GammaScalpingAnalyzer.py`: Volatility regime and strategy scoring
  - `Scanner1.py`: Options scanner for spreads, straddles, etc.
  - `MarketDashboard.py`: Sentiment analysis
  - `config.json`: Central config for thresholds, tickers, etc.

### 2. **Frontend (Flutter)**
- **Responsibilities:**
  - Loads and parses report data (from assets or API)
  - Presents dashboard, strategy cards, and drill-downs
  - Handles navigation and user interaction
  - Responsive UI for mobile, tablet, and web
- **Key Modules:**
  - `main.dart`: App entry, dashboard, navigation
  - `models/`: Data models for report, strategies, tickers
  - `screens/`: Dashboard, sentiment detail, strategy detail
  - `widgets/`: Reusable UI components

---

## Data Flow

```mermaid
flowchart TD
    subgraph Backend
      A[Market Data Fetch] --> B[Analysis & Scoring]
      B --> C[Report Generation (JSON/HTML)]
      C --> D[Flask API]
    end
    subgraph Frontend
      E[Flutter App]
    end
    D --> E
    C --> E
```

- **Backend** fetches data, analyzes, scores, and outputs a report
- **Frontend** loads report (from API or assets) and renders UI

---

## Modularity & Extensibility
- **Backend**: Each strategy/scanner is a separate module; new strategies can be added easily
- **Frontend**: New screens/cards can be added with minimal changes to models and navigation
- **Config-driven**: Tickers, thresholds, and strategy parameters are all configurable

---

## Future Cloud Migration
- **Backend** will move to a cloud platform (e.g., AWS Lambda, GCP Cloud Run, Azure Functions)
  - Stateless, scheduled, or event-driven report generation
  - API endpoints for live data
  - Secure, scalable, and maintainable
- **Frontend** remains Flutter (mobile/web)
  - Will fetch data from new cloud API endpoints

---

## Summary
- **Separation of concerns**: Data/logic (Python) vs. UI/UX (Flutter)
- **Designed for change**: Easy to add new strategies, migrate backend, or enhance UI
- **Cloud-ready**: Backend logic is portable to serverless/cloud environments 