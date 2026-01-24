# 🖥️ Dashboard Observability Guide

## Overview

Your hedge fund now has a **real-time web dashboard** for complete system observability:

✅ Live NAV and performance metrics  
✅ System health monitoring  
✅ Recent trades and positions  
✅ Interactive charts  
✅ Auto-refreshes every 30 seconds  
✅ Beautiful dark-themed UI  

---

## 🚀 Quick Start

### Start the Dashboard

```bash
cd /mnt/4tb/stock_scanner/market_news/HedgeFund

# Start on default port (5000)
./start_dashboard.sh

# Or specify custom port
./start_dashboard.sh 8080
```

### Access the Dashboard

Open your browser and navigate to:

```
http://localhost:5000
```

Or from another computer on your network:

```
http://YOUR_SERVER_IP:5000
```

---

## 📊 Dashboard Features

### 1. **Key Metrics Cards**

Top section shows 6 key metrics that update in real-time:

| Metric | Description |
|--------|-------------|
| **Net Asset Value** | Current portfolio value |
| **Daily Return** | Today's P&L percentage |
| **YTD Return** | Year-to-date performance |
| **Sharpe Ratio** | Risk-adjusted returns |
| **Open Positions** | Number of active positions + breakdown |
| **Today's Trades** | Trades executed today + last trade info |

---

### 2. **System Health Panel**

Real-time system monitoring:

- **CPU Usage**: With visual bar (green/yellow/red)
- **Memory Usage**: RAM consumption tracking
- **Disk Usage**: Storage space monitoring
- **Daemon Status**: Running/Stopped indicator
- **Database Status**: Connection health

**Health Indicator**: 
- 🟢 Green pulsing = System healthy
- 🔴 Red pulsing = Issues detected

---

### 3. **NAV History Chart**

Interactive 30-day NAV chart showing:
- Portfolio value over time
- Visual trends
- Hover for exact values

---

### 4. **Recent Trades Table**

Last 20 trades with:
- Timestamp
- Asset (SPY, QQQ, etc.)
- Action (SELL_PUT, BUY_CALL, etc.)
- Quantity & Strike
- Fill Price
- Status (filled/pending/rejected)
- P&L (profit/loss)

Color coded:
- 🟢 Green = Profit
- 🔴 Red = Loss

---

### 5. **Current Positions Table**

All open positions showing:
- Symbol
- Position Type
- Quantity
- Cost Basis & Current Price
- Unrealized P&L
- Greeks (Delta, Gamma, Theta)
- Date Opened

---

## 🔄 Auto-Refresh

The dashboard automatically refreshes every **30 seconds**.

Manual refresh: Click the **🔄 Refresh** button in the header.

---

## 🛠️ Running with Daemon

### Run Both Services

**Option 1: Separate Terminals**

Terminal 1 (Daemon):
```bash
cd /mnt/4tb/stock_scanner/market_news/HedgeFund
./start_daemon.sh supervisor
```

Terminal 2 (Dashboard):
```bash
cd /mnt/4tb/stock_scanner/market_news/HedgeFund
./start_dashboard.sh
```

**Option 2: Background Dashboard**

```bash
# Start daemon
./start_daemon.sh supervisor

# Start dashboard in background
nohup ./start_dashboard.sh > logs/dashboard.out 2>&1 &

# Check it's running
ps aux | grep dashboard_server
```

---

## 🔒 Security

### Local Access Only (Default)

By default, the dashboard binds to `0.0.0.0` which allows network access.

### Restrict to Localhost Only

Edit `src/dashboard/dashboard_server.py`, change last line:

```python
# Before (allows network access)
app.run(host='0.0.0.0', port=port, debug=False)

# After (localhost only)
app.run(host='127.0.0.1', port=port, debug=False)
```

### Add Authentication (Production)

For production, add authentication:

```bash
pip install flask-httpauth
```

Then add to `dashboard_server.py`:

```python
from flask_httpauth import HTTPBasicAuth
auth = HTTPBasicAuth()

users = {
    "admin": "your_secure_password_here"
}

@auth.verify_password
def verify_password(username, password):
    if username in users and users[username] == password:
        return username

# Then add @auth.login_required to routes
@app.route('/')
@auth.login_required
def index():
    return render_template('dashboard.html')
```

---

## 🔌 API Endpoints

The dashboard exposes REST API endpoints you can use:

### Health Check
```bash
curl http://localhost:5000/api/health
```

Returns:
```json
{
  "status": "healthy",
  "daemon_running": true,
  "database_connected": true,
  "system": {
    "cpu_percent": 15.2,
    "memory_percent": 45.8,
    "memory_available_gb": 8.5,
    "disk_percent": 62.3,
    "disk_free_gb": 150.2
  }
}
```

### Trading Status
```bash
curl http://localhost:5000/api/status
```

### Performance Metrics
```bash
curl http://localhost:5000/api/performance?days=30
```

### Recent Trades
```bash
curl http://localhost:5000/api/trades/recent?limit=20
```

### Current Positions
```bash
curl http://localhost:5000/api/positions
```

### NAV History
```bash
curl http://localhost:5000/api/nav/history?days=30
```

### Recent Logs
```bash
curl http://localhost:5000/api/logs?limit=50&level=ERROR
```

---

## 📱 Mobile Access

The dashboard is responsive and works on mobile devices!

Access from your phone:
1. Find your server's IP: `hostname -I`
2. Open browser on phone
3. Navigate to `http://YOUR_SERVER_IP:5000`

---

## 🐛 Troubleshooting

### Dashboard Won't Start

**Error: "Address already in use"**
```bash
# Check what's using port 5000
lsof -i :5000

# Kill the process
kill <PID>

# Or use different port
./start_dashboard.sh 8080
```

**Error: "Database connection failed"**
```bash
# Verify PostgreSQL is running
sudo systemctl status postgresql

# Check database credentials in .env
cat .env | grep DATABASE
```

**Error: "Module not found"**
```bash
# Reinstall dependencies
cd /mnt/4tb/stock_scanner/market_news
source venv/bin/activate
pip install flask flask-cors psutil asyncpg
```

---

### Dashboard Shows No Data

1. **Check if daemon is running**:
   ```bash
   ./check_daemon.sh
   ```

2. **Verify database has data**:
   ```bash
   psql -U hedgefund -d hedgefund_db -c "SELECT COUNT(*) FROM nav_history;"
   ```

3. **Check dashboard logs**:
   ```bash
   # If running in foreground, check terminal output
   # If running in background:
   tail -50 logs/dashboard.out
   ```

---

### Dashboard Shows Errors

Check browser console:
1. Open browser developer tools (F12)
2. Look at Console tab
3. Check for JavaScript errors

Check server logs:
```bash
# Watch live logs
./start_dashboard.sh  # Run in foreground to see errors
```

---

## 🎨 Customization

### Change Refresh Interval

Edit `templates/dashboard.html`, line with `setInterval`:

```javascript
// Change from 30 seconds to 10 seconds
setInterval(refreshAll, 10000);  // 10000 ms = 10 sec
```

### Change Theme Colors

Edit the `<style>` section in `templates/dashboard.html`:

```css
/* Main background */
background: #0f1419;  /* Dark blue-black */

/* Card background */
background: #161b22;  /* Slightly lighter */

/* Accent color (positive) */
color: #3fb950;  /* Green */

/* Accent color (negative) */
color: #f85149;  /* Red */
```

---

### Add Custom Metrics

Add new API endpoint in `dashboard_server.py`:

```python
@app.route('/api/custom/metric')
def api_custom_metric():
    # Your custom logic
    return jsonify({
        'value': 123.45
    })
```

Add to dashboard HTML:
```html
<div class="card">
    <div class="card-title">My Custom Metric</div>
    <div class="card-value" id="customMetric">-</div>
</div>

<script>
async function updateCustom() {
    const response = await fetch('/api/custom/metric');
    const data = await response.json();
    document.getElementById('customMetric').textContent = data.value;
}
</script>
```

---

## 🚀 Production Deployment

### Using Systemd

Create dashboard service:

```bash
sudo nano /etc/systemd/system/hedgefund-dashboard.service
```

```ini
[Unit]
Description=Hedge Fund Dashboard
After=network.target hedgefund-daemon.service

[Service]
Type=simple
User=kody
WorkingDirectory=/mnt/4tb/stock_scanner/market_news/HedgeFund
ExecStart=/mnt/4tb/stock_scanner/market_news/venv/bin/python src/dashboard/dashboard_server.py
Restart=always
Environment="DASHBOARD_PORT=5000"

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable hedgefund-dashboard
sudo systemctl start hedgefund-dashboard
sudo systemctl status hedgefund-dashboard
```

---

### Using Nginx Reverse Proxy

For production, put dashboard behind Nginx:

```nginx
server {
    listen 80;
    server_name hedgefund.yourdomain.com;

    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        
        # WebSocket support (if added later)
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

Then add SSL with Let's Encrypt:
```bash
sudo certbot --nginx -d hedgefund.yourdomain.com
```

---

## 📊 Monitoring the Dashboard

### Check Dashboard Status

```bash
# Is it running?
ps aux | grep dashboard_server

# Check port
lsof -i :5000

# View logs
tail -f logs/dashboard.out
```

### Resource Usage

```bash
# Memory usage
ps aux | grep dashboard_server | awk '{print $4, $6}'

# Full process info
top -p $(pgrep -f dashboard_server)
```

---

## 🎯 Next Steps

1. **Test the Dashboard**:
   ```bash
   ./start_dashboard.sh
   # Open http://localhost:5000
   ```

2. **Run with Daemon**:
   ```bash
   # Terminal 1
   ./start_daemon.sh supervisor
   
   # Terminal 2
   ./start_dashboard.sh
   ```

3. **Monitor Live**:
   - Watch trades execute at 9:30 AM
   - See NAV update in real-time
   - Monitor system health

4. **Customize**:
   - Add your own metrics
   - Change colors/theme
   - Adjust refresh rate

---

## 💡 Pro Tips

1. **Bookmark it**: Add `http://localhost:5000` to browser bookmarks
2. **Full screen**: Press F11 for immersive monitoring
3. **Multiple tabs**: Open trades, positions, and health in separate tabs
4. **Mobile widget**: Add to home screen on mobile for quick access
5. **API integration**: Use API endpoints to build custom alerts/bots

---

## 🎉 You Now Have

✅ 24/7 paper trading daemon  
✅ Real-time web dashboard  
✅ Complete observability  
✅ System health monitoring  
✅ Performance tracking  
✅ Auto-restart on crash  
✅ Survives reboots  

**Your hedge fund system is production-ready!** 🚀


