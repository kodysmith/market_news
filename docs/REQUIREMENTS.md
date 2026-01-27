# Requirements files overview

This repo has **multiple Python sub-projects** with their own dependencies. We are intentionally **not consolidating** them right now; this document explains what each requirements file is for and when to use it.

## Root requirements

- **`/mnt/4tb/stock_scanner/market_news/requirements.txt`**
  - **Use for**: Lightweight/common scripting (basic data pulls and helpers).
  - **Includes**: `requests`, `pandas`, `numpy`.

- **`/mnt/4tb/stock_scanner/market_news/requirements_news.txt`**
  - **Use for**: News bot / LLM summarization workflows.
  - **Includes**: `requests`, `python-dotenv`, `feedparser`, `anthropic`, `pytz`.

- **`/mnt/4tb/stock_scanner/market_news/requirements_options.txt`**
  - **Use for**: Options strategy backtesting utilities (plots + data).
  - **Includes**: `backtesting`, `scipy`, `matplotlib`, `seaborn`, `yfinance`, `pandas`, `numpy`.

## QuantEngine

- **`/mnt/4tb/stock_scanner/market_news/QuantEngine/requirements.txt`**
  - **Use for**: Full QuantEngine environment (research, backtesting, ML, reporting).
  - **Highlights**: `polars`, `duckdb`, `pyarrow`, `backtrader`, `ta-lib`, `scikit-learn`, `optuna`, `mlflow`, `wandb`.

- **`/mnt/4tb/stock_scanner/market_news/QuantEngine/requirements_enhanced.txt`**
  - **Use for**: GPU/LLM-enabled “enhanced scanner” workflows.
  - **Highlights**: `torch*`, `transformers`, `accelerate`, `bitsandbytes`, `ollama`, plus dashboard libs (`streamlit`, `plotly`, `dash`).

## HedgeFund

- **`/mnt/4tb/stock_scanner/market_news/HedgeFund/requirements.txt`**
  - **Use for**: HedgeFund system dependencies (brokers, pricing, database, API).
  - **Highlights**: `alpaca-py`, `ib_insync`, `py_vollib`, `riskfolio-lib`, `fastapi`, `sqlalchemy`, `redis`, monitoring/logging tools.

## Options scanner

- **`/mnt/4tb/stock_scanner/market_news/options_scanner/requirements.txt`**
  - **Use for**: The options scanner worker + publishing utilities.
  - **Highlights**: `firebase-admin`, `scipy`, `python-dotenv`, `pytz`.

## Buffett screener

- **`/mnt/4tb/stock_scanner/market_news/buffett_screener/requirements.txt`**
  - **Use for**: Buffett/value screener.
  - **Includes**: `yfinance`, `requests`, `pandas`, `numpy`.

## CryptoArbitrage

- **`/mnt/4tb/stock_scanner/market_news/CryptoArbitrage/requirements.txt`**
  - **Use for**: Crypto arbitrage daemon + dashboard.
  - **Highlights**: `ccxt`, `ccxt.pro`, `aiohttp`, `websockets`, `aiosqlite`, `flask`, `pydantic`.

## Flutter app backtesting_v2 (Python tooling)

- **`/mnt/4tb/stock_scanner/market_news/market_news_app/backtesting_v2/requirements.txt`**
  - **Use for**: The Python backtesting tools that live under the Flutter app folder.
  - **Highlights**: `vectorbt`, `duckdb`, `awswrangler`, `pyarrow`, `mlflow`, `prefect`.

## Suggested usage pattern

- **Keep environments isolated**: create a venv per sub-project (e.g., `QuantEngine/.venv`, `HedgeFund/.venv`, etc.).
- **Install only what you need** for the subsystem you’re running (these sets are large and partially overlapping).

