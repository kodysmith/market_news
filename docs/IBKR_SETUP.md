# IBKR API setup (TWS / IB Gateway)

**Connection refused** means nothing is listening on the port. The IBKR API does **not** use an API key. Instead, a **desktop app** from Interactive Brokers must be running and logged in; your Python script then connects to that app on your machine.

## What are TWS and IB Gateway?

- **TWS (Trader Workstation)** — IBKR’s full trading desktop app. You log in with your IBKR username/password; the app keeps the session and exposes an API on a local port.
- **IB Gateway** — A lighter-weight app from IBKR that does the same thing (login + API) with a minimal window. Often used for automation/API-only use.

Your script talks to whichever one is running. It does **not** log you in; it only connects to an already-logged-in app.

## Steps to fix “Connection refused”

### 1. Install one of the apps

- **TWS (recommended to start):**  
  https://www.interactivebrokers.com/en/trading/tws.php  
  Download and install “Trader Workstation” for your OS.

- **IB Gateway (lighter):**  
  https://www.interactivebrokers.com/en/trading/ibgateway-stable.php  
  Download and install “IB Gateway”.

### 2. Run it and log in

- Open TWS (or IB Gateway).
- Log in with your **IBKR account** (username + password).
  - For paper trading, use **Paper Trading** login (separate credentials if you have a paper account).
  - Default ports: **7497** = TWS paper, **7496** = TWS live; **4001** = Gateway paper, **4002** = Gateway live.
- Leave the app running (minimized is fine). Don’t log out.

### 3. Enable the API and set the port

**In TWS:**

1. Menu: **Edit → Global Configuration** (or **File → Global Configuration** on Mac).
2. Left sidebar: **API → Settings**.
3. Enable **“Enable ActiveX and Socket Clients”** (or “Enable Socket Clients”).
4. Note the **Socket port** (e.g. **7497** for paper, **7496** for live).
5. Optionally add **127.0.0.1** under “Trusted IPs” if you only connect from this machine.
6. Click **Apply** / **OK**.

**In IB Gateway:**  
Same idea: open configuration, find **API** / **Settings**, enable socket clients, and note the port (often 4001 for paper, 4002 for live).

### 4. Run the test script again

```bash
# TWS paper (default)
python scripts/test_ibkr_connection.py

# Or if you use a different port (e.g. Gateway paper)
IBKR_PORT=4001 python scripts/test_ibkr_connection.py
```

If the app is running, logged in, and the API is enabled on that port, the script should connect.

## Summary

| You want to…           | Do this |
|------------------------|--------|
| Use the API at all     | Install and run **TWS** or **IB Gateway**, log in, enable API. |
| Avoid “Connection refused” | Make sure the app is **running** and **logged in** before running the script. |
| Use paper trading      | Log into **Paper Trading** in TWS/Gateway; typical port 7497 (TWS) or 4001 (Gateway). |

No API key is stored in the repo; the “auth” is your normal IBKR login inside TWS/Gateway.
