#!/bin/bash
# Quick database setup for hedge fund system

set -e

echo "================================================"
echo "🗄️  Hedge Fund Database Setup"
echo "================================================"
echo ""

# Check if PostgreSQL is installed
if ! command -v psql &> /dev/null; then
    echo "❌ PostgreSQL is not installed"
    echo ""
    echo "Install with:"
    echo "  sudo apt-get update"
    echo "  sudo apt-get install postgresql postgresql-contrib"
    exit 1
fi

echo "✅ PostgreSQL is installed"
echo ""

# Check if PostgreSQL is running
if ! sudo systemctl is-active --quiet postgresql; then
    echo "Starting PostgreSQL..."
    sudo systemctl start postgresql
    sleep 2
fi

echo "✅ PostgreSQL is running"
echo ""

# Database details
DB_NAME="hedgefund_db"
DB_USER="hedgefund"
DB_PASSWORD="${HEDGEFUND_DB_PASSWORD:-hedgefund_secure_pass_$(date +%s)}"

echo "Creating database and user..."
echo ""

# Create user and database
sudo -u postgres psql << EOF
-- Create user if not exists
DO \$\$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_user WHERE usename = '$DB_USER') THEN
        CREATE USER $DB_USER WITH PASSWORD '$DB_PASSWORD';
    END IF;
END
\$\$;

-- Create database if not exists
SELECT 'CREATE DATABASE $DB_NAME'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = '$DB_NAME')\gexec

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE $DB_NAME TO $DB_USER;
EOF

echo "✅ Database and user created"
echo ""

# Create tables
echo "Creating tables..."
sudo -u postgres psql -d $DB_NAME << 'EOF'
-- Trades table
CREATE TABLE IF NOT EXISTS trades (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    asset TEXT NOT NULL,
    action TEXT NOT NULL,
    quantity INTEGER,
    strike DECIMAL(10,2),
    expiration DATE,
    fill_price DECIMAL(10,4),
    status TEXT,
    pnl DECIMAL(12,2),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_trades_asset ON trades(asset, timestamp);

-- Positions table
CREATE TABLE IF NOT EXISTS positions (
    id SERIAL PRIMARY KEY,
    symbol TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    cost_basis DECIMAL(12,2),
    current_price DECIMAL(12,2),
    unrealized_pnl DECIMAL(12,2),
    position_type TEXT,
    status TEXT DEFAULT 'open',
    date_opened DATE,
    date_closed DATE,
    strike DECIMAL(10,2),
    expiration DATE,
    delta DECIMAL(8,4),
    gamma DECIMAL(8,4),
    theta DECIMAL(8,4),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol);
CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status);

-- NAV History table
CREATE TABLE IF NOT EXISTS nav_history (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    nav DECIMAL(15,2) NOT NULL,
    cash DECIMAL(15,2) NOT NULL,
    positions_value DECIMAL(15,2),
    daily_return DECIMAL(10,6),
    ytd_return DECIMAL(10,6),
    sharpe_ratio DECIMAL(8,4),
    max_drawdown DECIMAL(8,6),
    num_short_puts INTEGER,
    num_covered_calls INTEGER,
    num_shares INTEGER,
    num_hedges INTEGER,
    capital_deployed DECIMAL(15,2),
    capital_deployed_pct DECIMAL(6,4),
    portfolio_delta DECIMAL(12,4),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_nav_timestamp ON nav_history(timestamp DESC);

-- Audit Log table
CREATE TABLE IF NOT EXISTS audit_log (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    event_type TEXT NOT NULL,
    severity TEXT,
    component TEXT,
    message TEXT,
    data JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_audit_event ON audit_log(event_type);

-- Grant permissions to hedgefund user
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO hedgefund;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO hedgefund;
EOF

echo "✅ Tables created"
echo ""

# Update .env file
ENV_FILE="$(dirname "$0")/.env"

if [ ! -f "$ENV_FILE" ]; then
    echo "Creating .env file..."
    cat > "$ENV_FILE" << ENVEOF
# Hedge Fund Environment Configuration

# Trading Mode
ENV=paper

# Alpaca API (Paper Trading)
ALPACA_API_KEY=your_paper_key_here
ALPACA_SECRET_KEY=your_paper_secret_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# Database Configuration
DATABASE_HOST=localhost
DATABASE_PORT=5432
DATABASE_USER=$DB_USER
DATABASE_PASSWORD=$DB_PASSWORD
DATABASE_NAME=$DB_NAME

# Dashboard
DASHBOARD_PORT=5000
ENVEOF
    echo "✅ .env file created"
else
    echo "✅ .env file already exists"
    
    # Update database credentials if needed
    if ! grep -q "DATABASE_PASSWORD=" "$ENV_FILE"; then
        echo "" >> "$ENV_FILE"
        echo "# Database Configuration" >> "$ENV_FILE"
        echo "DATABASE_HOST=localhost" >> "$ENV_FILE"
        echo "DATABASE_PORT=5432" >> "$ENV_FILE"
        echo "DATABASE_USER=$DB_USER" >> "$ENV_FILE"
        echo "DATABASE_PASSWORD=$DB_PASSWORD" >> "$ENV_FILE"
        echo "DATABASE_NAME=$DB_NAME" >> "$ENV_FILE"
        echo "✅ Database config added to .env"
    fi
fi

echo ""
echo "================================================"
echo "✅ Database Setup Complete!"
echo "================================================"
echo ""
echo "Database Details:"
echo "  Name:     $DB_NAME"
echo "  User:     $DB_USER"
echo "  Password: $DB_PASSWORD"
echo "  Host:     localhost"
echo "  Port:     5432"
echo ""
echo "Connection String:"
echo "  postgresql://$DB_USER:$DB_PASSWORD@localhost:5432/$DB_NAME"
echo ""
echo "Test connection:"
echo "  psql -U $DB_USER -d $DB_NAME -c 'SELECT COUNT(*) FROM trades;'"
echo ""
echo "Next Steps:"
echo "  1. Restart the daemon:  ./stop_daemon.sh && ./start_daemon.sh supervisor"
echo "  2. Refresh dashboard:   http://localhost:5000"
echo "  3. Database should now show 'Connected'"
echo ""
echo "================================================"

