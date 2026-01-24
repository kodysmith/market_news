#!/bin/bash
# Serve Flutter web app on port 8081, accessible on local network

cd "$(dirname "$0")"

# Build the web app if needed
if [ ! -d "build/web" ]; then
    echo "Building Flutter web app..."
    flutter build web
fi

# Get local IP address
LOCAL_IP=$(hostname -I | awk '{print $1}')

echo "=========================================="
echo "Flutter Web App Server"
echo "=========================================="
echo "Local URL:  http://localhost:8081"
echo "Network URL: http://$LOCAL_IP:8081"
echo ""
echo "Access from other devices on your network:"
echo "  - Open browser on phone/tablet"
echo "  - Go to: http://$LOCAL_IP:8081"
echo ""
echo "Press Ctrl+C to stop"
echo "=========================================="
echo ""

# Serve on port 8081, binding to all interfaces (0.0.0.0)
cd build/web
python3 -m http.server 8081 --bind 0.0.0.0
