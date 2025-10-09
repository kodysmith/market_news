#!/bin/bash
# One-command fix for common issues

set -e

cd "$(dirname "$0")"

echo "================================================"
echo "🔧 FIXING COMMON ISSUES"
echo "================================================"
echo ""

# 1. Stop any running processes
echo "1️⃣ Stopping old processes..."
./stop_daemon.sh 2>/dev/null || true
pkill -f dashboard_server 2>/dev/null || true
echo "   ✅ Cleaned up old processes"
echo ""

# 2. Check PostgreSQL
echo "2️⃣ Checking PostgreSQL..."
if ! systemctl is-active --quiet postgresql 2>/dev/null; then
    echo "   PostgreSQL is not running"
    echo "   Run: sudo systemctl start postgresql"
    echo ""
    read -p "   Start PostgreSQL now? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        sudo systemctl start postgresql
        sudo systemctl enable postgresql
        echo "   ✅ PostgreSQL started and enabled"
    else
        echo "   ⚠️  Skipping PostgreSQL start"
        echo "   Warning: Database features won't work!"
    fi
else
    echo "   ✅ PostgreSQL is running"
fi
echo ""

# 3. Setup database if needed
echo "3️⃣ Checking database setup..."
if ! psql -U hedgefund -d hedgefund_db -c "SELECT 1" &>/dev/null; then
    echo "   Database not set up"
    read -p "   Run database setup? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        ./setup_database.sh
    else
        echo "   ⚠️  Skipping database setup"
    fi
else
    echo "   ✅ Database is configured"
fi
echo ""

# 4. Start daemon
echo "4️⃣ Starting daemon in supervisor mode..."
./start_daemon.sh supervisor
sleep 2
echo ""

# 5. Check daemon
echo "5️⃣ Checking daemon status..."
./check_daemon.sh
echo ""

echo "================================================"
echo "✅ SETUP COMPLETE!"
echo "================================================"
echo ""
echo "Next steps:"
echo "  1. Start dashboard: ./start_dashboard.sh"
echo "  2. Open browser: http://localhost:5000"
echo ""
echo "Check status anytime:"
echo "  ./check_daemon.sh"
echo ""
echo "View logs:"
echo "  tail -f logs/daemon_stdout.log"
echo ""
echo "================================================"

