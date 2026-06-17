from http.server import BaseHTTPRequestHandler
import json, numpy as np
from scipy.stats import norm
from urllib.parse import urlparse, parse_qs
import urllib.request, time
from datetime import datetime, timedelta

TICKER_MAP = {
    "TQQQ": "TQQQ", "SOXL": "SOXL", "SQQQ": "SQQQ",
    "WTI": "USO",   "NG": "UNG",    "BOIL": "BOIL",
}
LEVERAGES = [0.5, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.5, 2.8, 3.0]

def fetch_yahoo(sym, days=120):
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
        data = json.loads(resp.read())
    result = data["chart"]["result"][0]
    closes = result["indicators"]["quote"][0]["close"]
    prices = [c for c in closes if c is not None]
    if len(prices) < 10:
        raise ValueError(f"Not enough data: {len(prices)}")
    return prices

def hist_cvar(returns, alpha=0.05):
    """Historical CVaR: 실제 분포 하위 α% 평균"""
    var  = np.percentile(returns, alpha * 100)
    tail = returns[returns <= var]
    return float(tail.mean()) if len(tail) > 0 else float(var)

def gbm_cvar(returns, alpha=0.05):
    """
    GBM CVaR: 정규분포 가정
    CVaR_α = μ - σ·φ(z_α)/α
    항상 실제보다 낙관적 (꼬리가 얇음)
    """
    mu    = float(np.mean(returns))
    sigma = float(np.std(returns))
    z     = norm.ppf(alpha)
    return float(mu - sigma * norm.pdf(z) / alpha)

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        qs    = parse_qs(urlparse(self.path).query)
        asset = qs.get("asset", ["TQQQ"])[0].upper()
        sym   = TICKER_MAP.get(asset, asset)

        try:
            prices  = fetch_yahoo(sym, days=120)
            returns = np.array([
                (prices[i]-prices[i-1])/prices[i-1]
                for i in range(1, len(prices))
            ])
            n = len(returns)

            # ── GBM 파라미터
            sigma_ann = float(np.std(returns) * np.sqrt(252))  # 연율화 변동성
            mu_ann    = float(np.mean(returns) * 252)
            S         = float(prices[-1])
            K, T, r   = S, 0.5, 0.05

            d1 = (np.log(S/K) + (r + sigma_ann**2/2)*T) / (sigma_ann*np.sqrt(T))
            d2 = d1 - sigma_ann*np.sqrt(T)

            prob_profit     = float(norm.cdf(d2))
            bs_expected_ret = float((np.exp(r*T)-1)*100)
            monthly_sigma   = sigma_ann / np.sqrt(12)
            # GBM 예측 꼬리 확률 (-15% 기준, -30%는 상승장에서 안 나옴)
            bs_tail_prob_15 = float(norm.cdf(-0.15 / monthly_sigma) * 100)
            bs_tail_prob_30 = float(norm.cdf(-0.30 / monthly_sigma) * 100)

            # ── Fat Tail Ratio (-15% 기준)
            monthly_rets = [
                float(np.prod(1+returns[i:i+21])-1)
                for i in range(0, n-21, 3)
            ]
            crash_15 = sum(1 for r_ in monthly_rets if r_ < -0.15)
            actual_15 = crash_15 / max(len(monthly_rets),1) * 100
            fat_tail_ratio = round(actual_15 / bs_tail_prob_15, 1) if bs_tail_prob_15 > 0 else 1.0

            # ── Volatility Decay (3× 레버리지)
            lev3 = float(np.prod(1+returns*3)-1)*100
            lev1 = float(np.prod(1+returns)-1)*100
            vol_decay = round(lev3 - lev1*3, 1)

            # ── GBM CVaR (정규분포 가정 — 과소평가)
            # CVaR_α = μ - σ·φ(z_α)/α
            # 정규분포는 꼬리가 얇아서 실제보다 낙관적
            mu_d    = float(np.mean(returns))
            sigma_d = float(np.std(returns))
            z_alpha = norm.ppf(0.05)
            gbm_cvar_1x = float((mu_d - sigma_d * norm.pdf(z_alpha) / 0.05) * 100)

            # ── Historical CVaR (실제 분포 — 항상 GBM보다 보수적)
            # 실제 수익률 하위 5% 평균
            # Fat Tail이 있으면 정의상 GBM CVaR보다 더 음수
            var5         = np.percentile(returns, 5)
            tail5        = returns[returns <= var5]
            hist_cvar_1x = float(tail5.mean() * 100) if len(tail5) > 0 else float(var5 * 100)

            # ── 레버리지별 Historical CVaR
            cvar_by_lev = []
            mean_by_lev = []
            for lev in LEVERAGES:
                lr = returns * lev
                cvar_by_lev.append(round(hist_cvar(lr)*100, 2))
                mean_by_lev.append(float(np.mean(lr)))

            # ── 최적 레버리지: 기대수익 양수 중 CVaR 절댓값 가장 작은 것
            feasible = [(i, cvar_by_lev[i]) for i in range(len(LEVERAGES)) if mean_by_lev[i] > 0]
            opt_idx  = min(feasible, key=lambda x: abs(x[1]))[0] if feasible else 2

            max_abs  = max(abs(v) for v in cvar_by_lev) or 1
            energies = [round(abs(c)/max_abs, 4) for c in cvar_by_lev]

            body = json.dumps({
                "sigma":             round(sigma_ann*100, 1),
                "bs_prob_profit":    round(prob_profit*100, 1),
                "bs_expected_ret":   round(bs_expected_ret, 1),
                "bs_tail_prob":      round(bs_tail_prob_15, 3),
                "bs_tail_prob_30":   round(bs_tail_prob_30, 3),
                "fat_tail_ratio":    fat_tail_ratio,
                "vol_decay":         vol_decay,
                # 핵심 비교: GBM(정규분포) vs Historical(실제 분포)
                # Historical이 항상 더 음수 (Fat Tail 반영)
                "gbm_cvar":          round(gbm_cvar_1x, 2),
                "regime_cvar":       round(hist_cvar_1x, 2),  # Historical CVaR
                "cvar_5pct":         round(hist_cvar_1x, 2),
                "cvar_by_leverage":  cvar_by_lev,
                "optimal_leverage":  LEVERAGES[opt_idx],
                "optimal_idx":       opt_idx,
                "energies":          energies,
                "returns":           [round(float(r),6) for r in returns],
            })
        except Exception as e:
            body = json.dumps({"error": str(e)})

        self._respond(body)

    def _respond(self, body):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body.encode())

    def log_message(self, *args):
        pass
