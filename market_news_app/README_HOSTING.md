# Hosting Flutter Web App on Local Network (Port 8080)

## Quick Start

### Option 1: Production Build (Recommended for sharing)
```bash
cd /mnt/4tb/stock_scanner/market_news/market_news_app
./serve_web.sh
```

This will:
- Build the web app (if not already built)
- Serve it on port 8080
- Make it accessible on your local network at `http://YOUR_IP:8080`

### Option 2: Development Mode (Hot Reload)
```bash
cd /mnt/4tb/stock_scanner/market_news/market_news_app
./launch_dev.sh
```

This will:
- Launch Flutter in development mode
- Enable hot reload (auto-refresh on code changes)
- Serve on port 8080, accessible on local network

## Access from Other Devices

1. **Find your server IP:**
   ```bash
   hostname -I | awk '{print $1}'
   ```
   Example: `192.168.1.31`

2. **On your phone/tablet/other computer:**
   - Connect to the same WiFi network
   - Open browser
   - Go to: `http://192.168.1.31:8080`

## Manual Commands

### Build only:
```bash
flutter build web
```

### Serve with Python (after build):
```bash
cd build/web
python3 -m http.server 8080 --bind 0.0.0.0
```

### Flutter dev server (development):
```bash
flutter run -d chrome --web-port=8080 --web-hostname=0.0.0.0
```

## Important Notes

1. **Backend API**: Make sure your Flask API is running on port 5000:
   ```bash
   cd /mnt/4tb/stock_scanner/market_news/apis
   python3 api.py
   ```

2. **Firewall**: If devices can't connect, check your firewall:
   ```bash
   # Ubuntu/Debian
   sudo ufw allow 8080
   ```

3. **API URL**: The app is configured to use `http://192.168.1.31:5000` for the API.
   - Update `lib/main.dart` if your IP changes
   - Or set it dynamically based on the hostname

## Troubleshooting

- **Can't access from phone**: Check firewall, ensure same WiFi network
- **API not working**: Verify Flask API is running on port 5000
- **Port already in use**: Change port in the scripts (8080 → 8081, etc.)
