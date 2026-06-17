from http.server import BaseHTTPRequestHandler
import json, numpy as np
from scipy.stats import norm
from urllib.parse import urlparse, parse_qs
import urllib.request
from datetime import datetime, timedelta

TICKER_MAP = {
    "TQQQ": "TQQQ", "SOXL": "SOXL", "SQQQ": "SQQQ",
    "WTI":  "USO",  "NG":   "UNG",  "BOIL": "BOIL",
}

CASES = {
    "TQQQ": [
        {"name": "COVID Crash",     "train_end": "2020-02-01", "crash_end": "2020-03-23"},
        {"name": "Rate Shock 2022", "train_end": "2022-01-03", "crash_end": "2022-06-16"},
        {"name": "Yen Carry 2024",  "train_end": "2024-08-01", "crash_end": "2024-08-05"},
    ],
    "SOXL": [
        {"name": "COVID Crash",     "train_end": "2020-02-01", "crash_end": "2020-03-23"},
        {"name": "Rate Shock 2022", "train_end": "2022-01-03", "crash_end": "2022-06-16"},
        {"name": "Yen Carry 2024",  "train_end": "2024-08-01", "crash_end": "2024-08-05"},
    ],
    "SQQQ": [
        {"name": "COVID Recovery",  "train_end": "2020-03-23", "crash_end": "2020-06-08"},
        {"name": "2023 AI Rally",   "train_end": "2023-01-02", "crash_end": "2023-07-19"},
        {"name": "Nov 2024 Rally",  "train_end": "2024-11-04", "crash_end": "2024-11-29"},
    ],
    "USO": [
        {"name": "COVID Oil Crash", "train_end": "2020-02-01", "crash_end": "2020-04-20"},
        {"name": "2022 Peak&Crash", "train_end": "2022-06-01", "crash_end": "2022-12-09"},
        {"name": "2023 OPEC Shock", "train_end": "2023-09-01", "crash_end": "2023-10-06"},
    ],
    "UNG": [
        {"name": "2021 Winter",      "train_end": "2021-02-01", "crash_end": "2021-03-01"},
        {"name": "2022 EU Crisis",   "train_end": "2022-08-01", "crash_end": "2022-12-01"},
        {"name": "2024 Supply Glut", "train_end": "2024-02-01", "crash_end": "2024-04-15"},
    ],
    "BOIL": [
        {"name": "2021 Winter",      "train_end": "2021-02-01", "crash_end": "2021-03-01"},
        {"name": "2022 EU Crisis",   "train_end": "2022-08-01", "crash_end": "2022-12-01"},
        {"name": "2024 Supply Glut", "train_end": "2024-02-01", "crash_end": "2024-04-15"},
    ],
}

def date_to_ts(date_str):
    return int(datetime.strptime(date_str, "%Y-%m-%d").timestamp())

def fetch_range(sym, start_str, end_str):
    start = date_to_ts(start_str)
    end   = date_to_ts(end_str) + 86400
    url   = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
        f"?interval=1d&period1={start}&period2={end}"
    )
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
    })
    with urllib.request.urlopen(req, timeout=8) as resp:
        data = json.loads(resp.read())
    result = data["chart"]["result"][0]
    closes = result["indicators"]["quote"][0]["close"]
    prices = [c for c in closes if c is not None]
    return prices

def make_rolling_20d(prices):
    """
    20일 rolling 누적 수익률 계산
    r_20[t] = P[t+20] / P[t] - 1   (최솟값 아닌 종가 기준)
    → BS와 Historical CVaR 모두 동일한 분포에서 계산
    """
    rets = []
    for t in range(len(prices) - 20):
        r = prices[t + 20] / prices[t] - 1
        rets.append(r)
    return np.array(rets)

def bs_cvar_20d(rolling_rets, alpha=0.05):
    """
    BS 20-day CVaR: 정규분포 가정
    20일 수익률이 N(μ, σ²)를 따른다고 가정

    CVaR_α = μ - σ · φ(z_α) / α
    """
    mu    = float(np.mean(rolling_rets))
    sigma = float(np.std(rolling_rets))
    if sigma < 1e-9:
        return 0.0
    z_alpha = norm.ppf(alpha)
    cvar    = mu - sigma * norm.pdf(z_alpha) / alpha
    return float(cvar * 100)

def historical_cvar_20d(rolling_rets, alpha=0.05):
    """
    Historical 20-day CVaR: 분포 가정 없음
    실제 20일 수익률 분포의 하위 α% 평균

    Fat Tail이 있으면 BS CVaR보다 더 음수(보수적)
    """
    var  = np.percentile(rolling_rets, alpha * 100)
    tail = rolling_rets[rolling_rets <= var]
    cvar = float(tail.mean()) if len(tail) > 0 else float(var)
    return float(cvar * 100)

def actual_20d_drawdown(crash_prices):
    """
    실제 위기 시작 후 20일간 최대 손실
    (BS/Historical CVaR와 동일한 20일 horizon)
    """
    window = crash_prices[:21]   # 시작가 + 최대 20일
    if len(window) < 2:
        return None
    dd = (min(window[1:]) / window[0] - 1) * 100
    return float(dd)

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        qs    = parse_qs(urlparse(self.path).query)
        asset = qs.get("asset", ["TQQQ"])[0].upper()
        sym   = TICKER_MAP.get(asset, asset)
        cases = CASES.get(sym, CASES.get(asset, CASES["TQQQ"]))

        results = []
        for case in cases:
            try:
                train_end = case["train_end"]
                train_dt  = datetime.strptime(train_end, "%Y-%m-%d")
                # 훈련 데이터: 위기 직전 200일
                # (20일 rolling 최소 30개 샘플 확보)
                train_start = (train_dt - timedelta(days=200)).strftime("%Y-%m-%d")

                # ── 훈련 데이터 (위기 직전)
                train_prices = fetch_range(sym, train_start, train_end)
                if len(train_prices) < 25:
                    raise ValueError(f"Not enough train data: {len(train_prices)}")

                # ── 20일 rolling 수익률 (훈련 데이터 기반)
                rolling = make_rolling_20d(train_prices)
                if len(rolling) < 10:
                    raise ValueError(f"Not enough rolling samples: {len(rolling)}")

                # ── BS 20-day CVaR (정규분포 가정)
                bs_cvar = bs_cvar_20d(rolling)

                # ── Historical 20-day CVaR (실제 분포)
                hist_cvar = historical_cvar_20d(rolling)

                # ── 실제 위기 데이터
                crash_prices = fetch_range(sym, train_end, case["crash_end"])
                if len(crash_prices) < 2:
                    raise ValueError("Not enough crash data")

                # ── 실제 20일 손실 (동일 horizon)
                actual = actual_20d_drawdown(crash_prices)
                if actual is None:
                    raise ValueError("Cannot compute actual drawdown")

                # ── 오차: 실제 손실과 예측의 차이
                # 셋 다 20일 기준 → 직접 비교 가능
                bs_err   = round(abs(actual) - abs(bs_cvar),   1)
                hist_err = round(abs(actual) - abs(hist_cvar), 1)

                # historical이 bs보다 실제에 얼마나 가까운가
                bs_miss   = abs(actual - bs_cvar)
                hist_miss = abs(actual - hist_cvar)
                captured  = round((bs_miss - hist_miss) / bs_miss * 100, 1) if bs_miss > 0 else 0.0

                results.append({
                    "name":       case["name"],
                    "period":     f"{train_end} → {case['crash_end']}",
                    "horizon":    "20-day",
                    # 세 값 모두 20일 기준 → 직접 비교 가능
                    "bs_cvar":     round(bs_cvar,   1),
                    "regime_cvar": round(hist_cvar, 1),
                    "actual":      round(actual,    1),
                    # 과소평가 정도
                    "bs_underestimate":     max(bs_err,   0.0),
                    "regime_underestimate": max(hist_err, 0.0),
                    # 추가 포착률
                    "additional_captured": captured,
                    "note": (
                        "All values: 20-day horizon. "
                        "GBM CVaR = normal distribution assumption on 20-day returns. "
                        "Historical CVaR = historical simulation on 20-day returns (no dist. assumption). "
                        "Actual = realized 20-day max drawdown from crash start."
                    )
                })

            except Exception as e:
                results.append({"name": case["name"], "error": str(e)})

        # ── 요약
        valid = [r for r in results if "additional_captured" in r]
        if valid:
            avg_bs   = round(np.mean([r["bs_underestimate"]     for r in valid]), 1)
            avg_reg  = round(np.mean([r["regime_underestimate"] for r in valid]), 1)
            avg_cap  = round(np.mean([r["additional_captured"]  for r in valid]), 1)
        else:
            avg_bs = avg_reg = avg_cap = None

        body = json.dumps({
            "asset":   asset,
            "cases":   results,
            "summary": {
                "avg_bs_underestimate":     avg_bs,
                "avg_regime_underestimate": avg_reg,
                "avg_additional_captured":  avg_cap,
                "method": (
                    "20-day horizon unified. "
                    "GBM CVaR uses normal distribution. "
                    "Historical CVaR uses historical simulation (fat tail aware). "
                    "Actual = realized 20-day drawdown. "
                    "All three on identical scale → directly comparable."
                )
            }
        })

        self._respond(body)

    def _respond(self, body):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body.encode())

    def log_message(self, *args):
        pass
