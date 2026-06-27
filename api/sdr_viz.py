"""
SDR Visualization API — 시각화 전용 경량 버전
탭③ Hilbert Space 산점도에만 사용. 실제 SDR 계산(/api/sdr)과 분리.

차이점:
- 시장변수 3개만 사용 (VIX, SPY, TLT) — 가장 핵심적인 변수
- 기간 60일 (시각화에 충분한 샘플)
- points 생성만 목적 — 레짐 확률/CVaR 계산 없음
- Vercel 무료 10초 제한 안에서 동작
"""
from http.server import BaseHTTPRequestHandler
import json, numpy as np
from scipy.linalg import eigh
from urllib.parse import urlparse, parse_qs
import urllib.request, time
from datetime import datetime, timedelta

TICKER_MAP = {
    "TQQQ":"TQQQ","SOXL":"SOXL","WTI":"USO","NG":"UNG"
}
# 시각화 전용: 핵심 3개 변수만 (fetch 시간 단축)
VIZ_MARKET = ["^VIX", "SPY", "TLT"]

from concurrent.futures import ThreadPoolExecutor, as_completed

def fetch_yahoo(sym, days=60):
    end   = int(time.time())
    start = int((datetime.utcnow() - timedelta(days=days)).timestamp())
    url   = (f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
             f"?interval=1d&period1={start}&period2={end}")
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=6) as resp:
        data = json.loads(resp.read())
    closes = data["chart"]["result"][0]["indicators"]["quote"][0]["close"]
    return [c for c in closes if c is not None]

def fetch_parallel(syms, days=60):
    """여러 심볼을 병렬로 fetch — 총 소요시간 ≈ 가장 느린 1개"""
    results = {}
    with ThreadPoolExecutor(max_workers=len(syms)) as ex:
        futures = {ex.submit(fetch_yahoo, s, days): s for s in syms}
        for f in as_completed(futures, timeout=7):
            s = futures[f]
            try:
                results[s] = f.result()
            except Exception:
                pass
    return results

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        qs    = parse_qs(urlparse(self.path).query)
        asset = qs.get("asset", ["TQQQ"])[0].upper()
        sym   = TICKER_MAP.get(asset, asset)

        try:
            # 자산 + 시장변수 3개 병렬 fetch (총 소요시간 ≈ 가장 느린 1개)
            all_syms = [sym] + VIZ_MARKET
            fetched = fetch_parallel(all_syms, days=60)

            asset_prices = fetched.get(sym)
            if not asset_prices or len(asset_prices) < 30:
                raise ValueError("Not enough data")

            horizon = 15   # 시각화용: 20→15일로 단축
            Y = np.array([
                min(asset_prices[t+1:t+horizon+1]) / asset_prices[t] - 1
                for t in range(len(asset_prices) - horizon)
            ])

            mkt_cols = [fetched[s] for s in VIZ_MARKET
                        if fetched.get(s) and len(fetched.get(s)) > 20]

            if len(mkt_cols) < 2:
                raise ValueError("Market data unavailable")

            min_len  = min(len(Y), min(len(c) for c in mkt_cols))
            mkt_cols = [c[-min_len:] for c in mkt_cols]
            Y_cut    = Y[-min_len:]

            # 수익률로 변환
            X = np.column_stack([
                np.diff(np.log(np.array(c)+1e-9))
                for c in mkt_cols
            ])
            n_align = min(len(X), len(Y_cut))
            X = X[-n_align:]; Y_cut = Y_cut[-n_align:]

            # SIR
            Sigma = np.cov(X.T) + 1e-6*np.eye(X.shape[1])
            h = 6
            quantiles = np.percentile(Y_cut, np.linspace(0,100,h+1))
            sm, sw = [], []
            for j in range(h):
                lo, hi = quantiles[j], quantiles[j+1]
                mask = (Y_cut>=lo)&(Y_cut<=hi) if j==h-1 else (Y_cut>=lo)&(Y_cut<hi)
                if mask.sum() >= 2:
                    sm.append(X[mask].mean(axis=0)); sw.append(mask.sum())
            if len(sm) < 2:
                raise ValueError("SIR failed")
            sm = np.array(sm); sw=np.array(sw,dtype=float); sw/=sw.sum()
            gm = (sw[:,None]*sm).sum(axis=0)
            M  = sum(w*np.outer(m-gm,m-gm) for w,m in zip(sw,sm))
            eigvals, eigvecs = eigh(np.linalg.inv(Sigma)@M)
            order = np.argsort(eigvals)[::-1]
            beta1 = eigvecs[:, order[0]]
            beta2 = eigvecs[:, order[1]] if X.shape[1] > 1 else beta1

            # 부호 정렬
            proj1 = X @ beta1
            if np.corrcoef(proj1, Y_cut)[0,1] < 0:
                beta1 = -beta1; proj1 = -proj1

            projected = X @ np.column_stack([beta1, beta2])

            # Y 기준 색깔 (사후 시각화)
            y_q33 = float(np.percentile(Y_cut, 33))
            y_q66 = float(np.percentile(Y_cut, 66))
            points = []
            for i, (px, py) in enumerate(projected[:-1]):
                y_val = Y_cut[i]
                col = "#F03860" if y_val<=y_q33 else ("#F0A800" if y_val<=y_q66 else "#00D878")
                points.append({"x":round(float(px)*60,4),"y":round(float(py)*60,4),"col":col})

            # 현재 상태
            proj_q33 = float(np.percentile(proj1, 33))
            proj_q66 = float(np.percentile(proj1, 66))
            recent   = float(proj1[-3:].mean())
            regime   = "CRASH ZONE" if recent<=proj_q33 else ("ELEVATED" if recent<=proj_q66 else "SAFE ZONE")

            body = json.dumps({
                "ok": True,
                "points": points,
                "regime": regime,
                "viz": {
                    "beta1": [round(float(v),4) for v in beta1[:2]],
                    "beta2": [round(float(v),4) for v in beta2[:2]],
                    "current_proj_x": round(float(projected[-1,0])*60,4),
                    "current_proj_y": round(float(projected[-1,1])*60,4),
                    "crash_boundary_x": round(float(proj_q66)*60,4),
                },
                "note": "visualization-only (3 vars, 60d) — regime/CVaR from /api/sdr"
            })
        except Exception as e:
            body = json.dumps({"ok": False, "error": str(e), "points": []})

        self.send_response(200)
        self.send_header("Content-Type","application/json")
        self.send_header("Access-Control-Allow-Origin","*")
        self.end_headers()
        self.wfile.write(body.encode())

    def log_message(self, *args): pass
