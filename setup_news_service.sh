#!/bin/bash
# Setup script for News Intelligence Bot systemd service

set -e

echo "====================================="
echo "News Intelligence Bot Service Setup"
echo "====================================="

# Get the absolute path to the project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_EXEC="$(which python3)"
USER="$(whoami)"

echo ""
echo "Project directory: $PROJECT_DIR"
echo "Python executable: $PYTHON_EXEC"
echo "Running as user: $USER"
echo ""

# Check if .env exists
if [ ! -f "$PROJECT_DIR/.env" ]; then
    echo "WARNING: .env file not found!"
    echo "Please create .env from env_news_template.txt"
    echo ""
    read -p "Do you want to continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "This script must be run with sudo"
    exit 1
fi

# Create service file
SERVICE_FILE="/etc/systemd/system/market-news-bot.service"

echo "Creating systemd service file at $SERVICE_FILE..."

cat > "$SERVICE_FILE" << EOF
[Unit]
Description=Market News Intelligence Bot
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$PROJECT_DIR
Environment="PYTHONUNBUFFERED=1"
EnvironmentFile=$PROJECT_DIR/.env
ExecStart=$PYTHON_EXEC -m news_bot.news_service
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

# Restart policy
StartLimitInterval=0
StartLimitBurst=5

# Security settings
NoNewPrivileges=true
PrivateTmp=true

[Install]
WantedBy=multi-user.target
EOF

echo "Service file created successfully!"
echo ""

# Reload systemd
echo "Reloading systemd daemon..."
systemctl daemon-reload

echo ""
echo "====================================="
echo "Setup Complete!"
echo "====================================="
echo ""
echo "Service installed as: market-news-bot"
echo ""
echo "Next steps:"
echo ""
echo "1. Start the service:"
echo "   sudo systemctl start market-news-bot"
echo ""
echo "2. Enable auto-start on boot:"
echo "   sudo systemctl enable market-news-bot"
echo ""
echo "3. Check status:"
echo "   sudo systemctl status market-news-bot"
echo ""
echo "4. View logs:"
echo "   sudo journalctl -u market-news-bot -f"
echo ""
echo "5. Stop the service:"
echo "   sudo systemctl stop market-news-bot"
echo ""
echo "====================================="


