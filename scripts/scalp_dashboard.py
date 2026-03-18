#!/usr/bin/env python3
"""
Live dashboard for SPX Scalp Bot — runs locally on port 5050.

Usage:
    source venv/bin/activate
    python scripts/scalp_dashboard.py [--port 5050] [--log-dir logs/scalp_bot]
"""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from flask import Flask, Response

ROOT = Path(__file__).resolve().parent.parent
ET = ZoneInfo("America/New_York")

app = Flask(__name__)
LOG_DIR = ROOT / "logs" / "scalp_bot"


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def get_today_str():
    return datetime.now(ET).strftime("%Y%m%d")


def compute_stats(trades: list[dict]) -> dict:
    if not trades:
        return {"total": 0, "wins": 0, "losses": 0, "win_rate": 0,
                "total_pnl": 0, "avg_win": 0, "avg_loss": 0,
                "profit_factor": 0, "max_dd": 0, "best": 0, "worst": 0,
                "equity_curve": [], "by_regime": {}, "by_direction": {}}

    pnls = [float(t.get("pnl_points", 0)) for t in trades]
    gross_pnls = [float(t.get("gross_pnl_points", t.get("pnl_points", 0))) for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    # Equity curve
    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    equity = []
    for i, p in enumerate(pnls):
        cum += p
        peak = max(peak, cum)
        max_dd = min(max_dd, cum - peak)
        equity.append({"trade": i + 1, "pnl": round(cum, 2)})

    # By regime
    by_regime = {}
    for t in trades:
        r = t.get("gex_regime", "UNKNOWN")
        if r not in by_regime:
            by_regime[r] = {"trades": 0, "wins": 0, "pnl": 0.0}
        by_regime[r]["trades"] += 1
        p = float(t["pnl_points"])
        by_regime[r]["pnl"] = round(by_regime[r]["pnl"] + p, 2)
        if p > 0:
            by_regime[r]["wins"] += 1

    # By direction
    by_dir = {}
    for t in trades:
        d = t["direction"]
        if d not in by_dir:
            by_dir[d] = {"trades": 0, "wins": 0, "pnl": 0.0}
        by_dir[d]["trades"] += 1
        p = float(t["pnl_points"])
        by_dir[d]["pnl"] = round(by_dir[d]["pnl"] + p, 2)
        if p > 0:
            by_dir[d]["wins"] += 1

    return {
        "total": len(pnls),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / len(pnls) * 100, 1) if pnls else 0,
        "total_pnl": round(sum(pnls), 2),
        "gross_pnl": round(sum(gross_pnls), 2),
        "total_slippage": round(sum(float(t.get("slippage_cost", 0)) for t in trades), 4),
        "total_commission": round(sum(float(t.get("commission_cost", 0)) for t in trades), 4),
        "avg_win": round(sum(wins) / len(wins), 2) if wins else 0,
        "avg_loss": round(sum(losses) / len(losses), 2) if losses else 0,
        "profit_factor": round(abs(sum(wins) / sum(losses)), 2) if losses and sum(losses) != 0 else 999,
        "max_dd": round(max_dd, 2),
        "best": round(max(pnls), 2),
        "worst": round(min(pnls), 2),
        "equity_curve": equity,
        "by_regime": by_regime,
        "by_direction": by_dir,
    }


@app.route("/api/data")
def api_data():
    today = get_today_str()
    trades = read_csv(LOG_DIR / f"trades_{today}.csv")
    signals = read_csv(LOG_DIR / f"signals_{today}.csv")
    stats = compute_stats(trades)
    last_signal = signals[-1] if signals else {}
    return Response(
        json.dumps({"stats": stats, "trades": trades[-50:], "last_signal": last_signal,
                     "signal_count": len(signals), "updated": datetime.now(ET).strftime("%H:%M:%S")}),
        content_type="application/json",
    )


@app.route("/")
def index():
    return DASHBOARD_HTML


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Scalp Bot Dashboard</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #0a0e17; color: #e0e0e0; font-family: 'SF Mono', 'Fira Code', monospace; font-size: 13px; padding: 16px; }
  h1 { font-size: 18px; color: #00d4ff; margin-bottom: 12px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 10px; margin-bottom: 16px; }
  .card { background: #141b2d; border: 1px solid #1e2a45; border-radius: 8px; padding: 14px; }
  .card .label { font-size: 10px; color: #6b7280; text-transform: uppercase; letter-spacing: 1px; }
  .card .value { font-size: 24px; font-weight: 700; margin-top: 4px; }
  .positive { color: #00e676; }
  .negative { color: #ff5252; }
  .neutral { color: #ffd740; }
  .section { margin-bottom: 16px; }
  .section-title { font-size: 13px; color: #6b7280; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px; border-bottom: 1px solid #1e2a45; padding-bottom: 4px; }
  table { width: 100%; border-collapse: collapse; }
  th { text-align: left; color: #6b7280; font-size: 10px; text-transform: uppercase; letter-spacing: 1px; padding: 6px 8px; border-bottom: 1px solid #1e2a45; }
  td { padding: 6px 8px; border-bottom: 1px solid #0d1525; font-size: 12px; }
  tr:hover { background: #1a2340; }
  .tag { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 10px; font-weight: 600; }
  .tag-long { background: #00e67622; color: #00e676; }
  .tag-short { background: #ff525222; color: #ff5252; }
  .tag-pos { background: #00d4ff22; color: #00d4ff; }
  .tag-neg { background: #ff914d22; color: #ff914d; }
  .chart-container { background: #141b2d; border: 1px solid #1e2a45; border-radius: 8px; padding: 16px; height: 200px; position: relative; }
  canvas { width: 100% !important; height: 100% !important; }
  .live-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #00e676; margin-right: 6px; animation: pulse 1.5s infinite; }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.3; } }
  .status-bar { display: flex; align-items: center; gap: 16px; margin-bottom: 12px; font-size: 11px; color: #6b7280; }
  .breakdown { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
  .breakdown .card { padding: 10px; }
  .breakdown .row { display: flex; justify-content: space-between; padding: 3px 0; font-size: 12px; }
</style>
</head>
<body>

<div style="display:flex; align-items:center; justify-content:space-between;">
  <h1>SCALP BOT — LIVE DASHBOARD</h1>
  <div class="status-bar">
    <span><span class="live-dot"></span>LIVE</span>
    <span id="clock"></span>
    <span id="signals-count"></span>
    <span id="updated"></span>
  </div>
</div>

<!-- Stats Cards -->
<div class="grid" id="stats-grid"></div>

<!-- Equity Curve -->
<div class="section">
  <div class="section-title">Equity Curve (Cumulative PnL)</div>
  <div class="chart-container"><canvas id="equityChart"></canvas></div>
</div>

<!-- Breakdowns -->
<div class="breakdown">
  <div class="card">
    <div class="section-title" style="border:none;padding:0;margin-bottom:6px;">By GEX Regime</div>
    <div id="regime-breakdown"></div>
  </div>
  <div class="card">
    <div class="section-title" style="border:none;padding:0;margin-bottom:6px;">By Direction</div>
    <div id="dir-breakdown"></div>
  </div>
</div>

<!-- Current Signal -->
<div class="section" style="margin-top:16px;">
  <div class="section-title">Latest Signal</div>
  <div class="card" id="last-signal" style="font-size:12px;"></div>
</div>

<!-- Trade Log -->
<div class="section" style="margin-top:16px;">
  <div class="section-title">Trade Log</div>
  <div style="max-height:300px;overflow-y:auto;">
    <table id="trade-table">
      <thead><tr>
        <th>Time</th><th>Ticker</th><th>Dir</th><th>Entry</th><th>Exit</th><th>Net PnL</th><th>Gross</th><th>Slip</th><th>Comm</th><th>Duration</th><th>Regime</th><th>Exit Reason</th>
      </tr></thead>
      <tbody></tbody>
    </table>
  </div>
</div>

<script>
// Minimal chart drawing on canvas
function drawEquity(canvas, data) {
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.parentElement.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);
  const W = rect.width, H = rect.height;
  ctx.clearRect(0, 0, W, H);

  if (!data || data.length === 0) {
    ctx.fillStyle = '#6b7280';
    ctx.font = '12px monospace';
    ctx.fillText('No trades yet...', W/2-50, H/2);
    return;
  }

  const vals = data.map(d => d.pnl);
  const minV = Math.min(0, ...vals);
  const maxV = Math.max(0, ...vals);
  const range = maxV - minV || 1;
  const pad = 30;

  // Grid lines
  ctx.strokeStyle = '#1e2a45';
  ctx.lineWidth = 0.5;
  for (let i = 0; i <= 4; i++) {
    const y = pad + (H - 2*pad) * i / 4;
    ctx.beginPath(); ctx.moveTo(pad, y); ctx.lineTo(W-10, y); ctx.stroke();
    const label = (maxV - range * i / 4).toFixed(1);
    ctx.fillStyle = '#6b7280'; ctx.font = '9px monospace';
    ctx.fillText(label, 2, y + 3);
  }

  // Zero line
  const zeroY = pad + (H - 2*pad) * (maxV / range);
  ctx.strokeStyle = '#ffffff33';
  ctx.lineWidth = 1;
  ctx.setLineDash([4,4]);
  ctx.beginPath(); ctx.moveTo(pad, zeroY); ctx.lineTo(W-10, zeroY); ctx.stroke();
  ctx.setLineDash([]);

  // Line
  const step = (W - pad - 10) / Math.max(data.length - 1, 1);
  ctx.beginPath();
  data.forEach((d, i) => {
    const x = pad + i * step;
    const y = pad + (H - 2*pad) * ((maxV - d.pnl) / range);
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  const lastPnl = vals[vals.length-1];
  ctx.strokeStyle = lastPnl >= 0 ? '#00e676' : '#ff5252';
  ctx.lineWidth = 2;
  ctx.stroke();

  // Fill gradient
  const lastX = pad + (data.length-1) * step;
  const lastY = pad + (H - 2*pad) * ((maxV - lastPnl) / range);
  ctx.lineTo(lastX, zeroY);
  ctx.lineTo(pad, zeroY);
  ctx.closePath();
  const grad = ctx.createLinearGradient(0, 0, 0, H);
  const color = lastPnl >= 0 ? '0,230,118' : '255,82,82';
  grad.addColorStop(0, `rgba(${color},0.15)`);
  grad.addColorStop(1, `rgba(${color},0)`);
  ctx.fillStyle = grad;
  ctx.fill();

  // Dot on last point
  ctx.beginPath();
  ctx.arc(lastX, lastY, 4, 0, Math.PI*2);
  ctx.fillStyle = lastPnl >= 0 ? '#00e676' : '#ff5252';
  ctx.fill();
}

function pnlClass(v) { return parseFloat(v) >= 0 ? 'positive' : 'negative'; }
function fmtDur(s) { s = parseInt(s); return s >= 60 ? Math.floor(s/60)+'m '+s%60+'s' : s+'s'; }

function renderStats(stats) {
  const cards = [
    { label: 'Total Trades', value: stats.total, cls: 'neutral' },
    { label: 'Win Rate', value: stats.win_rate+'%', cls: stats.win_rate >= 50 ? 'positive' : 'negative' },
    { label: 'Net PnL', value: (stats.total_pnl >= 0 ? '+' : '')+stats.total_pnl+' pts', cls: pnlClass(stats.total_pnl) },
    { label: 'Gross PnL', value: (stats.gross_pnl >= 0 ? '+' : '')+(stats.gross_pnl||0)+' pts', cls: pnlClass(stats.gross_pnl||0) },
    { label: 'Costs (Slip+Comm)', value: ((stats.total_slippage||0)+(stats.total_commission||0)).toFixed(2)+' pts', cls: 'negative' },
    { label: 'Profit Factor', value: stats.profit_factor, cls: stats.profit_factor >= 1 ? 'positive' : 'negative' },
    { label: 'Avg Win', value: '+'+stats.avg_win, cls: 'positive' },
    { label: 'Avg Loss', value: stats.avg_loss, cls: 'negative' },
    { label: 'Max Drawdown', value: stats.max_dd+' pts', cls: 'negative' },
    { label: 'Best / Worst', value: '+'+stats.best+' / '+stats.worst, cls: 'neutral' },
  ];
  document.getElementById('stats-grid').innerHTML = cards.map(c =>
    `<div class="card"><div class="label">${c.label}</div><div class="value ${c.cls}">${c.value}</div></div>`
  ).join('');
}

function renderBreakdowns(stats) {
  let html = '';
  for (const [k,v] of Object.entries(stats.by_regime || {})) {
    const wr = v.trades ? Math.round(v.wins/v.trades*100) : 0;
    const cls = k.includes('POS') ? 'tag-pos' : 'tag-neg';
    html += `<div class="row"><span class="tag ${cls}">${k}</span><span>${v.trades} trades | ${wr}% WR | <span class="${pnlClass(v.pnl)}">${v.pnl>0?'+':''}${v.pnl}</span></span></div>`;
  }
  document.getElementById('regime-breakdown').innerHTML = html || '<span style="color:#6b7280">No data</span>';

  html = '';
  for (const [k,v] of Object.entries(stats.by_direction || {})) {
    const wr = v.trades ? Math.round(v.wins/v.trades*100) : 0;
    const cls = k === 'LONG' ? 'tag-long' : 'tag-short';
    html += `<div class="row"><span class="tag ${cls}">${k}</span><span>${v.trades} trades | ${wr}% WR | <span class="${pnlClass(v.pnl)}">${v.pnl>0?'+':''}${v.pnl}</span></span></div>`;
  }
  document.getElementById('dir-breakdown').innerHTML = html || '<span style="color:#6b7280">No data</span>';
}

function renderSignal(sig) {
  if (!sig || !sig.timestamp) {
    document.getElementById('last-signal').innerHTML = '<span style="color:#6b7280">Waiting for signal...</span>';
    return;
  }
  const sigCls = sig.signal === 'LONG' ? 'tag-long' : sig.signal === 'SHORT' ? 'tag-short' : '';
  document.getElementById('last-signal').innerHTML = `
    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:8px;">
      <div><span style="color:#6b7280">Time:</span> ${sig.timestamp}</div>
      <div><span style="color:#6b7280">Price:</span> $${parseFloat(sig.spx_price).toFixed(2)}</div>
      <div><span style="color:#6b7280">VIX:</span> ${sig.vix}</div>
      <div><span style="color:#6b7280">GEX:</span> <span class="tag ${sig.gex_regime?.includes('POS')?'tag-pos':'tag-neg'}">${sig.gex_regime}</span></div>
      <div><span style="color:#6b7280">Flip:</span> ${sig.flip_line}</div>
      <div><span style="color:#6b7280">RSI:</span> ${parseFloat(sig.rsi14).toFixed(0)}</div>
      <div><span style="color:#6b7280">EMA 9/21:</span> ${parseFloat(sig.ema9).toFixed(2)} / ${parseFloat(sig.ema21).toFixed(2)}</div>
      <div><span style="color:#6b7280">Signal:</span> <span class="tag ${sigCls}">${sig.signal}</span> (${sig.confidence})</div>
    </div>
    <div style="margin-top:6px;color:#6b7280;">Reason: ${sig.reason}</div>
  `;
}

function renderTrades(trades) {
  const tbody = document.querySelector('#trade-table tbody');
  const reversed = [...trades].reverse();
  tbody.innerHTML = reversed.map(t => {
    const dirCls = t.direction === 'LONG' ? 'tag-long' : 'tag-short';
    const pnl = parseFloat(t.pnl_points);
    const pnlStr = (pnl >= 0 ? '+' : '') + pnl.toFixed(2);
    const gross = parseFloat(t.gross_pnl_points||t.pnl_points);
    const slip = parseFloat(t.slippage_cost||0).toFixed(3);
    const comm = parseFloat(t.commission_cost||0).toFixed(2);
    return `<tr>
      <td>${t.entry_time?.split(' ')[1] || ''}</td>
      <td>${t.ticker||''}</td>
      <td><span class="tag ${dirCls}">${t.direction}</span></td>
      <td>$${parseFloat(t.entry_price).toFixed(2)}</td>
      <td>$${parseFloat(t.exit_price).toFixed(2)}</td>
      <td class="${pnlClass(pnl)}">${pnlStr}</td>
      <td class="${pnlClass(gross)}">${(gross>=0?'+':'')+gross.toFixed(2)}</td>
      <td>${slip}</td>
      <td>${comm}</td>
      <td>${fmtDur(t.duration_sec)}</td>
      <td><span class="tag ${t.gex_regime?.includes('POS')?'tag-pos':'tag-neg'}">${t.gex_regime||''}</span></td>
      <td>${t.exit_reason}</td>
    </tr>`;
  }).join('');
}

async function refresh() {
  try {
    const res = await fetch('/api/data');
    const d = await res.json();
    renderStats(d.stats);
    renderBreakdowns(d.stats);
    renderSignal(d.last_signal);
    renderTrades(d.trades);
    drawEquity(document.getElementById('equityChart'), d.stats.equity_curve);
    document.getElementById('signals-count').textContent = `${d.signal_count} signals`;
    document.getElementById('updated').textContent = `Updated: ${d.updated}`;
  } catch(e) { console.error(e); }
}

function updateClock() {
  const now = new Date();
  document.getElementById('clock').textContent = now.toLocaleTimeString('en-US', {timeZone:'America/New_York', hour12:false}) + ' ET';
}

setInterval(refresh, 3000);
setInterval(updateClock, 1000);
refresh();
updateClock();
</script>
</body>
</html>"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scalp Bot Live Dashboard")
    parser.add_argument("--port", type=int, default=5050)
    parser.add_argument("--log-dir", type=str, default="logs/scalp_bot")
    args = parser.parse_args()
    LOG_DIR = ROOT / args.log_dir
    print(f"Dashboard: http://localhost:{args.port}")
    app.run(host="0.0.0.0", port=args.port, debug=False)
