from http.server import BaseHTTPRequestHandler
import json, numpy as np
from urllib.parse import urlparse, parse_qs
import urllib.request, urllib.error
from datetime import datetime, timedelta
import time

TICKER_MAP = {
    "TQQQ": "TQQQ", "SOXL": "SOXL", "SQQQ": "SQQQ",
    "WTI": "USO", "NG": "UNG", "BOIL": "BOIL",
}

def fetch_yahoo(sym, days=70):
    """Yahoo Finance v8 API 직접 호출 — yfinance 라이브러리 없이"""
    end   = int(time.time())
    start = int((datetime.utcnow() - timedelta(days=days)).timestamp())
    url   = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
        f"?interval=1d&period1={start}&period2={end}"
    )
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
    })
    with urllib.request.urlopen(req, timeout=8) as resp:
        import json as _json
        data = _json.loads(resp.read())

    result = data["chart"]["result"][0]
    closes = result["indicators"]["quote"][0]["close"]
    timestamps = result["timestamp"]

    # None 제거
    pairs = [(t, c) for t, c in zip(timestamps, closes) if c is not None]
    if len(pairs) < 2:
        raise ValueError(f"Not enough data: {len(pairs)} rows")

    prices = [c for _, c in pairs]
    return prices

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        qs    = parse_qs(urlparse(self.path).query)
        asset = qs.get("asset", ["TQQQ"])[0].upper()
        sym   = TICKER_MAP.get(asset, asset)

        try:
            prices  = fetch_yahoo(sym, days=70)
            current = float(prices[-1])
            prev    = float(prices[-2])
            change  = current - prev
            chg_pct = change / prev * 100

            # 수익률
            returns = [
                (prices[i] - prices[i-1]) / prices[i-1]
                for i in range(1, len(prices))
            ]

            body = json.dumps({
                "asset":   asset,
                "price":   round(current, 2),
                "change":  round(change, 2),
                "chg_pct": round(chg_pct, 2),
                "up":      chg_pct >= 0,
                "returns": [round(r, 6) for r in returns],
            })
        except Exception as e:
            body = json.dumps({"error": str(e), "asset": asset})

        self._respond(body)

    def _respond(self, body):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body.encode())

    def log_message(self, *args):
        pass
