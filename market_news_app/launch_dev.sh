#!/bin/bash
# Launch Flutter web app in development mode on port 8080
# This auto-reloads on code changes

cd "$(dirname "$0")"

# Get local IP address
LOCAL_IP=$(hostname -I | awk '{print $1}')

echo "=========================================="
echo "Flutter Web App (Development Mode)"
echo "=========================================="
echo "Local URL:  http://localhost:8080"
echo "Network URL: http://$LOCAL_IP:8080"
echo ""
echo "Access from other devices on your network:"
echo "  - Open browser on phone/tablet"
echo "  - Go to: http://$LOCAL_IP:8080"
echo ""
echo "Note: Hot reload enabled - changes auto-refresh"
echo "Press Ctrl+C to stop"
echo "=========================================="
echo ""

# Launch Flutter web on port 8080, binding to all interfaces
flutter run -d web-server --web-port=8080 --web-hostname=0.0.0.0
