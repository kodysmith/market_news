#!/bin/bash
# Automated Service Setup Script
# Creates and enables systemd service for 24/7 paper trading

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Crypto Arbitrage Service Setup${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Get current directory
INSTALL_DIR="$(cd "$(dirname "$0")" && pwd)"
echo "Installation directory: $INSTALL_DIR"

# Get current user
CURRENT_USER=$(whoami)
echo "Running as user: $CURRENT_USER"
echo ""

# Check if .env exists
if [ ! -f "$INSTALL_DIR/.env" ]; then
    echo -e "${RED}❌ Error: .env file not found!${NC}"
    echo "Please create .env with your API keys first:"
    echo "  cp env_template.txt .env"
    echo "  nano .env"
    exit 1
fi

# Check if venv exists
if [ ! -d "$INSTALL_DIR/venv" ]; then
    echo -e "${RED}❌ Error: Virtual environment not found!${NC}"
    echo "Please create virtual environment first:"
    echo "  python3 -m venv venv"
    echo "  source venv/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# Create directories
echo -e "${GREEN}Creating directories...${NC}"
mkdir -p "$INSTALL_DIR/data"
mkdir -p "$INSTALL_DIR/data/backups"
mkdir -p "$INSTALL_DIR/logs"
echo -e "${GREEN}✓${NC} Directories created"
echo ""

# Create service file
SERVICE_FILE="/etc/systemd/system/crypto-arbitrage-paper.service"

echo -e "${GREEN}Creating systemd service file...${NC}"
echo "Service file: $SERVICE_FILE"
echo ""

# Generate service file content
cat > /tmp/crypto-arbitrage-paper.service <<EOF
[Unit]
Description=Crypto Arbitrage Paper Trading Service
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$CURRENT_USER
WorkingDirectory=$INSTALL_DIR
Environment="CRYPTO_ENV=simulation"
Environment="PATH=$INSTALL_DIR/venv/bin:/usr/bin:/bin"

# Run paper trading script
ExecStart=$INSTALL_DIR/venv/bin/python3 $INSTALL_DIR/scripts/paper_trading.py

# Auto-restart on failure
Restart=always
RestartSec=10

# Logging
StandardOutput=append:$INSTALL_DIR/logs/service.log
StandardError=append:$INSTALL_DIR/logs/service_error.log

# Resource limits
MemoryLimit=1G
CPUQuota=50%

[Install]
WantedBy=multi-user.target
EOF

# Copy service file (requires sudo)
echo "This requires sudo permission to install the service..."
sudo cp /tmp/crypto-arbitrage-paper.service $SERVICE_FILE
sudo chmod 644 $SERVICE_FILE

# Reload systemd
echo -e "${GREEN}Reloading systemd...${NC}"
sudo systemctl daemon-reload

# Enable service
echo -e "${GREEN}Enabling service...${NC}"
sudo systemctl enable crypto-arbitrage-paper.service

echo -e "${GREEN}✓${NC} Service installed and enabled"
echo ""

# Ask to start now
echo -e "${YELLOW}Start the service now? (y/n)${NC}"
read -r START_NOW

if [[ "$START_NOW" =~ ^[Yy]$ ]]; then
    echo -e "${GREEN}Starting service...${NC}"
    sudo systemctl start crypto-arbitrage-paper.service
    
    sleep 2
    
    echo ""
    echo -e "${GREEN}Service status:${NC}"
    sudo systemctl status crypto-arbitrage-paper.service --no-pager
    echo ""
    
    if sudo systemctl is-active crypto-arbitrage-paper.service > /dev/null; then
        echo -e "${GREEN}✓ Service is running!${NC}"
    else
        echo -e "${RED}❌ Service failed to start. Check logs:${NC}"
        echo "  sudo journalctl -u crypto-arbitrage-paper.service -n 50"
    fi
else
    echo "Service not started. To start later, run:"
    echo "  sudo systemctl start crypto-arbitrage-paper.service"
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Setup Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Useful commands:"
echo "  sudo systemctl status crypto-arbitrage-paper.service  # Check status"
echo "  sudo systemctl start crypto-arbitrage-paper.service   # Start service"
echo "  sudo systemctl stop crypto-arbitrage-paper.service    # Stop service"
echo "  sudo systemctl restart crypto-arbitrage-paper.service # Restart service"
echo "  tail -f logs/paper_trading.log                        # View logs"
echo "  ./check_health.sh                                     # Health check"
echo "  ./analyze_opportunities.sh                            # Analyze results"
echo ""


