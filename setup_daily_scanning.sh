#!/bin/bash
# Setup Daily QuantBot Scanning
# This script sets up cron jobs for daily market scanning and publishing

echo "🚀 Setting up Daily QuantBot Scanning..."
echo "========================================"

# Get the current directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QUANT_ENGINE_DIR="$SCRIPT_DIR/QuantEngine"

echo "📁 QuantEngine Directory: $QUANT_ENGINE_DIR"

# Check if QuantEngine directory exists
if [ ! -d "$QUANT_ENGINE_DIR" ]; then
    echo "❌ QuantEngine directory not found at $QUANT_ENGINE_DIR"
    exit 1
fi

# Create the cron job entry
CRON_ENTRY="0 9 * * 1-5 cd $QUANT_ENGINE_DIR && python3 schedule_opportunity_scanner.py >> scanner_scheduler.log 2>&1"

echo "📅 Cron Job Entry:"
echo "   $CRON_ENTRY"
echo ""

# Check if cron job already exists
if crontab -l 2>/dev/null | grep -q "schedule_opportunity_scanner.py"; then
    echo "⚠️  Cron job already exists for opportunity scanner"
    echo "   Current crontab entries:"
    crontab -l 2>/dev/null | grep "schedule_opportunity_scanner.py"
    echo ""
    read -p "Do you want to replace it? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "❌ Setup cancelled"
        exit 0
    fi
fi

# Add the cron job
echo "➕ Adding cron job..."
(crontab -l 2>/dev/null; echo "$CRON_ENTRY") | crontab -

if [ $? -eq 0 ]; then
    echo "✅ Cron job added successfully!"
    echo ""
    echo "📋 Current crontab entries:"
    crontab -l 2>/dev/null
    echo ""
    echo "🎯 Daily scanning will run:"
    echo "   - Monday to Friday at 9:00 AM"
    echo "   - Logs will be saved to: $QUANT_ENGINE_DIR/scanner_scheduler.log"
    echo ""
    echo "🔍 To monitor the scanning:"
    echo "   tail -f $QUANT_ENGINE_DIR/scanner_scheduler.log"
    echo ""
    echo "🛑 To remove the cron job:"
    echo "   crontab -e"
    echo "   (then delete the line with schedule_opportunity_scanner.py)"
else
    echo "❌ Failed to add cron job"
    exit 1
fi

echo ""
echo "🎉 Daily QuantBot scanning setup complete!"
echo "   The system will now scan markets daily and publish results to the mobile app."



