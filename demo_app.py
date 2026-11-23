"""
One-command interactive web demo for Commodity MLOps Engine (Forecasting & Freight Anomaly Detection).
Run: python demo_app.py
"""
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
import numpy as np
import pandas as pd
from src.anomaly import FreightAnomalyDetector
from src.forecast import CommodityDemandForecaster

PORT = int(os.environ.get("PORT", "8003"))


def run_ml_demo():
    np.random.seed(42)
    # 1. Freight Anomaly Detection (IsolationForest)
    n = 60
    rates = np.random.normal(182.0, 11.0, n)
    rates[14] = 278.0
    rates[39] = 294.5
    rates[52] = 96.0
    df = pd.DataFrame({
        "origin": ["Port North"] * 30 + ["Port South"] * 30,
        "destination": ["Foundry A"] * 30 + ["Foundry B"] * 30,
        "date": pd.date_range("2026-09-01", periods=n, freq="D"),
        "distance_km": [350.0] * 30 + [540.0] * 30,
        "cargo_tonnage": np.random.uniform(32.0, 44.0, n),
        "diesel_price_ref": np.random.uniform(5.7, 6.1, n),
        "rate_per_ton": rates,
    })
    detector = FreightAnomalyDetector(contamination=0.06, random_state=42)
    detector.fit(df)
    scored = detector.predict(df)
    anomalies = scored[scored["anomaly_label"] == "ANOMALY"].head(8)

    # 2. Statistical Time-Series Forecasting (Pure Damped Holt-Winters)
    dates = pd.date_range("2025-06-01", periods=140, freq="D")
    t = np.arange(140)
    vals = 120.0 + 0.25 * t + 12.0 * np.sin(2 * np.pi * t / 7) + np.random.normal(0, 2.2, 140)
    series = pd.Series(vals, index=dates)
    train, test = series.iloc[:-14], series.iloc[-14:]
    forecaster = CommodityDemandForecaster(seasonal_periods=7, holdout_days=14)
    forecaster.fit(train)
    preds = forecaster.predict_future(steps=14)
    wape = float(np.sum(np.abs(test.values - preds)) / np.sum(np.abs(test.values)) * 100.0)

    return {
        "total_shipments": int(len(scored)),
        "anomaly_count": int((scored["anomaly_label"] == "ANOMALY").sum()),
        "wape_pct": round(wape, 2),
        "forecast_horizon": 14,
        "anomalies": [
            {
                "date": str(r["date"])[:10],
                "route": f"{r['origin']} -> {r['destination']}",
                "tonnage": round(float(r["cargo_tonnage"]), 1),
                "rate_per_ton": round(float(r["rate_per_ton"]), 2),
                "score": round(float(r["anomaly_score"]), 4),
            }
            for _, r in anomalies.iterrows()
        ],
        "forecast_comparison": [
            {"day": str(test.index[i])[:10], "actual": round(float(test.values[i]), 1), "predicted": round(float(preds[i]), 1)}
            for i in range(len(test))
        ],
    }


HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>Commodity MLOps Engine — Live Demo</title>
  <style>
    :root { --bg:#090e1a; --card:#121b2e; --surf:#0c1322; --bdr:#22314f; --txt:#f1f5f9; --mut:#94a3b8; --acc:#a855f7; --blue:#38bdf8; --grn:#10b981; --red:#ef4444; }
    * { box-sizing:border-box; font-family:-apple-system,BlinkMacSystemFont,"Inter","Segoe UI",sans-serif; }
    body { margin:0; background:var(--bg); color:var(--txt); padding:24px; line-height:1.45; }
    .wrap { max-width:1280px; margin:0 auto; }
    header { display:flex; justify-content:space-between; align-items:center; padding-bottom:16px; margin-bottom:20px; border-bottom:1px solid var(--bdr); }
    h1 { margin:0; font-size:1.35rem; }
    .sub { margin:3px 0 0; font-size:0.84rem; color:var(--mut); }
    .pill { padding:5px 11px; border-radius:999px; font-size:0.75rem; font-weight:600; background:rgba(168,85,247,0.14); color:#c084fc; border:1px solid rgba(168,85,247,0.3); }
    .kpis { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-bottom:18px; }
    .kpi { background:var(--card); border:1px solid var(--bdr); border-radius:10px; padding:14px 16px; }
    .kpi span { font-size:0.74rem; text-transform:uppercase; color:var(--mut); }
    .kpi strong { display:block; font-size:1.35rem; margin-top:5px; }
    .grid { display:grid; grid-template-columns:1fr 1fr; gap:18px; }
    .card { background:var(--card); border:1px solid var(--bdr); border-radius:10px; padding:18px; margin-bottom:18px; }
    .card h2 { margin:0 0 12px; font-size:0.88rem; text-transform:uppercase; letter-spacing:0.06em; color:var(--mut); border-bottom:1px solid var(--bdr); padding-bottom:8px; }
    table { width:100%; border-collapse:collapse; font-size:0.84rem; font-variant-numeric:tabular-nums; }
    th, td { text-align:left; padding:8px 9px; border-bottom:1px solid var(--bdr); }
    th { font-size:0.73rem; text-transform:uppercase; color:var(--mut); }
    .badge { display:inline-block; padding:3px 8px; border-radius:4px; font-size:0.72rem; font-weight:700; background:rgba(239,68,68,0.16); color:var(--red); }
    svg { width:100%; height:190px; background:var(--surf); border-radius:8px; border:1px solid var(--bdr); }
  </style>
</head>
<body>
  <div class="wrap">
    <header>
      <div>
        <h1>Commodity &amp; Logistics MLOps Engine</h1>
        <p class="sub">Pure NumPy/SciPy Damped Holt-Winters Demand Forecasting &amp; IsolationForest Spot Freight Anomaly Detection</p>
      </div>
      <span class="pill">Python · MLflow Ready · Synthetic Telemetry</span>
    </header>
    <div class="kpis" id="kpis"></div>
    <div class="grid">
      <div class="card">
        <h2>Holdout Demand Forecast vs Actual (14-Day Horizon)</h2>
        <svg id="chart" viewBox="0 0 560 190"></svg>
        <div style="font-size:0.76rem;color:var(--mut);margin-top:8px;">
          <span style="color:var(--blue)">&#9644; Actual Demand (MT)</span> &nbsp;&nbsp;
          <span style="color:#c084fc">&#9644; Damped Holt-Winters Forecast</span>
        </div>
      </div>
      <div class="card">
        <h2>Flagged Freight Rate Anomalies (IsolationForest)</h2>
        <table>
          <thead><tr><th>Date</th><th>Route</th><th>Tonnage</th><th>Rate ($/t)</th><th>Status</th></tr></thead>
          <tbody id="anomBody"></tbody>
        </table>
      </div>
    </div>
  </div>
  <script>
    fetch("/api/data").then(r => r.json()).then(d => {
      document.getElementById("kpis").innerHTML = `
        <div class="kpi"><span>Holdout WAPE</span><strong style="color:var(--grn)">${d.wape_pct}%</strong></div>
        <div class="kpi"><span>Forecast Horizon</span><strong>${d.forecast_horizon} Days</strong></div>
        <div class="kpi"><span>Shipments Scored</span><strong>${d.total_shipments}</strong></div>
        <div class="kpi"><span>Anomalies Flagged</span><strong style="color:var(--red)">${d.anomaly_count}</strong></div>
      `;
      document.getElementById("anomBody").innerHTML = d.anomalies.map(a => `
        <tr>
          <td>${a.date}</td><td>${a.route}</td><td>${a.tonnage} t</td>
          <td><strong>$${a.rate_per_ton.toFixed(2)}</strong></td>
          <td><span class="badge">ANOMALY</span></td>
        </tr>`).join("");
      const pts = d.forecast_comparison;
      const allV = pts.flatMap(p => [p.actual, p.predicted]);
      const minV = Math.min(...allV) - 5, maxV = Math.max(...allV) + 5;
      const sx = i => 28 + (i / (pts.length - 1)) * 510;
      const sy = v => 165 - ((v - minV) / (maxV - minV)) * 140;
      const pAct = pts.map((p,i) => `${sx(i)},${sy(p.actual)}`).join(" ");
      const pPre = pts.map((p,i) => `${sx(i)},${sy(p.predicted)}`).join(" ");
      document.getElementById("chart").innerHTML = `
        <polyline fill="none" stroke="#38bdf8" stroke-width="2.5" points="${pAct}" />
        <polyline fill="none" stroke="#c084fc" stroke-width="2.5" stroke-dasharray="5,4" points="${pPre}" />
      `;
    });
  </script>
</body>
</html>"""


class DemoHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML.encode("utf-8"))
        elif self.path == "/api/data":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(run_ml_demo()).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()


if __name__ == "__main__":
    print(f"Commodity MLOps Engine Demo running at http://127.0.0.1:{PORT}")
    HTTPServer(("127.0.0.1", PORT), DemoHandler).serve_forever()
