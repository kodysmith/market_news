#!/bin/bash
# Switch to SQLite (no PostgreSQL needed)

set -e

cd "$(dirname "$0")"

echo "================================================"
echo "🗄️  SWITCHING TO SQLITE"
echo "================================================"
echo ""
echo "SQLite requires no installation - it's built into Python!"
echo ""

# Update .env file
ENV_FILE=".env"

if [ ! -f "$ENV_FILE" ]; then
    echo "Creating .env file..."
    cat > "$ENV_FILE" << 'ENVEOF'
# Hedge Fund Environment Configuration

# Trading Mode
ENV=paper

# Alpaca API (Paper Trading)
ALPACA_API_KEY=your_paper_key_here
ALPACA_SECRET_KEY=your_paper_secret_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# Database Configuration (SQLite)
USE_SQLITE=true
SQLITE_DB_PATH=./data/hedgefund.db

# Dashboard
DASHBOARD_PORT=5000
ENVEOF
else
    # Add/update SQLite config
    if ! grep -q "USE_SQLITE" "$ENV_FILE"; then
        echo "" >> "$ENV_FILE"
        echo "# Database Configuration (SQLite)" >> "$ENV_FILE"
        echo "USE_SQLITE=true" >> "$ENV_FILE"
        echo "SQLITE_DB_PATH=./data/hedgefund.db" >> "$ENV_FILE"
    fi
fi

echo "✅ .env configured for SQLite"
echo ""

# Create data directory
mkdir -p data
echo "✅ Created data directory"
echo ""

# Create SQLite database with tables
echo "Creating SQLite database..."
python3 << 'PYEOF'
import sqlite3
import os

db_path = './data/hedgefund.db'

# Create database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Create tables
cursor.executescript('''
-- Trades table
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME NOT NULL,
    asset TEXT NOT NULL,
    action TEXT NOT NULL,
    quantity INTEGER,
    strike REAL,
    expiration DATE,
    fill_price REAL,
    status TEXT,
    pnl REAL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_trades_asset ON trades(asset, timestamp);

-- Positions table
CREATE TABLE IF NOT EXISTS positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    cost_basis REAL,
    current_price REAL,
    unrealized_pnl REAL,
    position_type TEXT,
    status TEXT DEFAULT 'open',
    date_opened DATE,
    date_closed DATE,
    strike REAL,
    expiration DATE,
    delta REAL,
    gamma REAL,
    theta REAL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol);
CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status);

-- NAV History table
CREATE TABLE IF NOT EXISTS nav_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME NOT NULL,
    nav REAL NOT NULL,
    cash REAL NOT NULL,
    positions_value REAL,
    daily_return REAL,
    ytd_return REAL,
    sharpe_ratio REAL,
    max_drawdown REAL,
    num_short_puts INTEGER,
    num_covered_calls INTEGER,
    num_shares INTEGER,
    num_hedges INTEGER,
    capital_deployed REAL,
    capital_deployed_pct REAL,
    portfolio_delta REAL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_nav_timestamp ON nav_history(timestamp DESC);

-- Audit Log table
CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME NOT NULL,
    event_type TEXT NOT NULL,
    severity TEXT,
    component TEXT,
    message TEXT,
    data TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_audit_event ON audit_log(event_type);
''')

conn.commit()
conn.close()

print(f"✅ SQLite database created: {os.path.abspath(db_path)}")
PYEOF

echo ""
echo "================================================"
echo "✅ SQLITE SETUP COMPLETE!"
echo "================================================"
echo ""
echo "Database location: ./data/hedgefund.db"
echo ""
echo "Next steps:"
echo "  1. Start daemon:    ./start_daemon.sh supervisor"
echo "  2. Start dashboard: ./start_dashboard.sh"
echo "  3. Open browser:    http://localhost:5000"
echo ""
echo "Note: The system will automatically use SQLite"
echo "      You can install PostgreSQL later if needed"
echo ""
echo "================================================"


