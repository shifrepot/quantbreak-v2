from http.server import BaseHTTPRequestHandler
import json, numpy as np
from scipy.linalg import eigh, cholesky
from urllib.parse import urlparse, parse_qs
import urllib.request, time
from datetime import datetime, timedelta

TICKER_MAP = {
    "TQQQ": "TQQQ", "SOXL": "SOXL", "SQQQ": "SQQQ",
    "WTI": "USO",   "NG": "UNG",    "BOIL": "BOIL",
}
MARKET_TICKERS = ["^VIX", "SPY", "TLT", "GLD", "^TNX"]
MARKET_NAMES   = ["VIX", "SPY", "TLT(Bond)", "Gold", "10Y Yield"]

def fetch_yahoo(sym, days=140):
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
    with urllib.request.urlopen(req, timeout=6) as resp:
        data = json.loads(resp.read())
    result = data["chart"]["result"][0]
    closes = result["indicators"]["quote"][0]["close"]
    return [c for c in closes if c is not None]

def compute_sir(X, Y, h=8):
    """
    Sliced Inverse Regression in Rᵖ(Σ) Hilbert Space

    내적: ⟨u,v⟩_Σ = uᵀΣv
    SDR 방향: β = eigenvec(Σ⁻¹M)

    M = Σₕ wₕ(m̄ₕ - m̄)(m̄ₕ - m̄)ᵀ
    """
    n, p  = X.shape
    Sigma = np.cov(X.T) + 1e-6 * np.eye(p)

    # Y 슬라이스
    quantiles = np.percentile(Y, np.linspace(0, 100, h+1))
    slice_means, slice_weights = [], []
    for j in range(h):
        lo, hi = quantiles[j], quantiles[j+1]
        mask   = (Y >= lo) & (Y <= hi) if j==h-1 else (Y >= lo) & (Y < hi)
        if mask.sum() >= 2:
            slice_means.append(X[mask].mean(axis=0))
            slice_weights.append(mask.sum())

    if len(slice_means) < 2:
        # PCA fallback
        eigvals, eigvecs = eigh(Sigma)
        idx = np.argsort(eigvals)[::-1]
        return eigvecs[:, idx[0]], eigvecs[:, idx[1]], Sigma, "PCA-fallback"

    sm = np.array(slice_means)
    sw = np.array(slice_weights, dtype=float); sw /= sw.sum()
    gm = (sw[:, None] * sm).sum(axis=0)
    M  = sum(w * np.outer(m-gm, m-gm) for w, m in zip(sw, sm))

    # Σ⁻¹M 고유벡터 → SDR 방향
    Sigma_inv = np.linalg.inv(Sigma)
    eigvals, eigvecs = eigh(Sigma_inv @ M)
    idx = np.argsort(eigvals)[::-1]
    b1, b2 = eigvecs[:, idx[0]], eigvecs[:, idx[1]]

    return b1, b2, Sigma, "SIR"

def project_hilbert(X, beta1, beta2, Sigma):
    """
    Rᵖ(Σ) Hilbert Space에서 2D 투영

    ⟨u,v⟩_Σ = uᵀΣv 내적 사용
    Σ-whitening: L = cholesky(Σ), X_w = X @ inv(L.T)
    투영: projected = X_w @ [Σ^{1/2}β₁, Σ^{1/2}β₂]

    결과: 유클리드 X@beta가 아닌 Σ-내적 기반 투영
    """
    try:
        L      = cholesky(Sigma, lower=True)
        L_inv  = np.linalg.inv(L)
        X_w    = X @ L_inv.T              # Σ-whitened X

        # β를 Σ 공간으로 변환
        Sb1    = Sigma @ beta1
        Sb2    = Sigma @ beta2

        # 정규화
        Sb1   /= (np.sqrt(beta1 @ Sigma @ beta1) + 1e-9)
        Sb2   /= (np.sqrt(beta2 @ Sigma @ beta2) + 1e-9)

        proj   = X_w @ np.column_stack([L_inv @ Sb1, L_inv @ Sb2])
        return proj, True
    except Exception:
        # fallback: 일반 투영
        proj = X @ np.column_stack([beta1, beta2])
        return proj, False

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        qs    = parse_qs(urlparse(self.path).query)
        asset = qs.get("asset", ["TQQQ"])[0].upper()
        sym   = TICKER_MAP.get(asset, asset)

        try:
            # ── 자산 가격
            asset_prices = fetch_yahoo(sym, days=140)
            if len(asset_prices) < 40:
                raise ValueError(f"Not enough asset data: {len(asset_prices)}")

            # ── Y = 미래 20일 forward maximum drawdown
            horizon = 20
            Y = np.array([
                min(asset_prices[t+1:t+horizon+1]) / asset_prices[t] - 1
                for t in range(len(asset_prices) - horizon)
            ])

            # ── 시장 변수 X
            mkt_cols = []
            for mkt_sym in MARKET_TICKERS:
                try:
                    mp = fetch_yahoo(mkt_sym, days=140)
                    mkt_cols.append(mp)
                except Exception:
                    continue

            if len(mkt_cols) < 2:
                raise ValueError("Not enough market data")

            min_len  = min(len(Y), min(len(c) for c in mkt_cols))
            mkt_cols = [c[-min_len:] for c in mkt_cols]
            Y_cut    = Y[-min_len:]

            # 수익률
            X = np.column_stack([
                np.array([(c[i]-c[i-1])/c[i-1] for i in range(1, len(c))])
                for c in mkt_cols
            ])
            Y_cut = Y_cut[1:]

            if len(Y_cut) < 15:
                raise ValueError(f"Not enough aligned: {len(Y_cut)}")

            # ── SIR in Rᵖ(Σ)
            beta1, beta2, Sigma, method = compute_sir(X, Y_cut, h=8)

            # ── Rᵖ(Σ) Hilbert Space 투영
            projected, hilbert_used = project_hilbert(X, beta1, beta2, Sigma)

            # ── β₁ 부호를 Y와 같은 방향으로 정렬
            # (고유벡터는 부호가 임의이므로, projection이 클수록 위험이 커지는
            #  방향이 되도록 정렬해야 이후 레짐 분류가 의미를 가짐)
            corr_sign = float(np.corrcoef(projected[:, 0], Y_cut)[0, 1])
            if np.isnan(corr_sign):
                # SIR projection의 분산이 거의 0이면 corrcoef가 NaN을 낼 수 있음
                # (NaN은 표준 JSON이 아니라 JS 쪽에서 깨진 값으로 보임 — 0으로 안전하게 처리)
                corr_sign = 0.0
            elif corr_sign < 0:
                beta1 = -beta1
                projected[:, 0] = -projected[:, 0]
                corr_sign = -corr_sign

            # ── 레짐 분류: SIR 투영값(현재 관측 가능한 시장변수의 투영) 기준
            # Y_cut은 미래 20일 결과이므로 "지금" 레짐 판단에 직접 쓰면 룩어헤드가 됨.
            # SIR이 학습한 방향 beta1을 통해 X(현재 시점에 알 수 있는 시장변수)의
            # 투영값으로 분류해야 SIR이 실제로 사용되는 것이 됨.
            proj1 = projected[:, 0]
            proj_q33 = float(np.percentile(proj1, 33))
            proj_q66 = float(np.percentile(proj1, 66))
            y_q33 = float(np.percentile(Y_cut, 33))   # 산점도 색칠 + 참고용으로 유지
            y_q66 = float(np.percentile(Y_cut, 66))

            # 레짐별 포인트 — 산점도는 "사후적으로 실제 어떤 레짐이었는지" 보여주는
            # 교육용 시각화이므로 실제 Y(미래 결과) 기준 색칠을 유지함.
            # (이 색은 레짐 확률 계산에는 쓰이지 않음 — 그건 아래에서 projection 기준으로 별도 계산)
            points = []
            regime_counts = {"crash":0, "elev":0, "safe":0}
            for i, (px, py) in enumerate(projected[:-1]):
                y_val = Y_cut[i]
                if y_val <= y_q33:
                    col = "#F03860"; regime_counts["crash"] += 1
                elif y_val <= y_q66:
                    col = "#F0A800"; regime_counts["elev"]  += 1
                else:
                    col = "#00D878"; regime_counts["safe"]  += 1
                points.append({
                    "x": round(float(px)*60, 4),
                    "y": round(float(py)*60, 4),
                    "col": col
                })

            # ── 현재 상태
            cx = round(float(projected[-1, 0])*60, 4)
            cy = round(float(projected[-1, 1])*60, 4)

            # ── 현재 레짐: SIR projection 기준 (룩어헤드 없음)
            recent_proj = float(proj1[-5:].mean())
            if recent_proj <= proj_q33:   regime = "CRASH ZONE"
            elif recent_proj <= proj_q66: regime = "ELEVATED"
            else:                          regime = "SAFE ZONE"

            # ── 레짐 확률: SIR projection 기준으로 분류한 비율
            # (산점도 색깔의 regime_counts와는 다른 기준 — 의도적으로 분리)
            regime_labels = np.where(proj1<=proj_q33, 0, np.where(proj1<=proj_q66, 1, 2))
            total_proj = len(regime_labels)
            p_crash = round(float(np.sum(regime_labels==0)) / total_proj, 4)
            p_elev  = round(float(np.sum(regime_labels==1)) / total_proj, 4)
            p_safe  = round(float(np.sum(regime_labels==2)) / total_proj, 4)

            # ── Density Matrix
            # 대각항: 실제 레짐 확률 (SIR projection 기준)
            # 비대각항: 실제 레짐 전환 빈도 (coherence)
            # crash(0) ↔ safe(2) 전환 빈도
            transitions_cs = sum(
                1 for i in range(len(regime_labels)-1)
                if (regime_labels[i]==0 and regime_labels[i+1]==2) or
                   (regime_labels[i]==2 and regime_labels[i+1]==0)
            )
            coh = round(transitions_cs / max(len(regime_labels)-1, 1), 4)
            density_matrix = {
                "p_crash": p_crash,
                "p_elev":  p_elev,
                "p_safe":  p_safe,
                "coherence": coh,
                # 2×2 표시용 (crash vs safe 대표)
                "dm00": str(round(p_crash, 2)),
                "dm01": str(round(coh, 2))+"i",
                "dm10": "−"+str(round(coh, 2))+"i",
                "dm11": str(round(p_safe, 2)),
            }

            # ── 레짐별 실제 Historical CVaR (자산 수익률 기준)
            # H_risk = Σᵢ cᵢ|i⟩⟨i|  (cᵢ = 레짐별 historical CVaR)
            asset_returns = np.array([
                (asset_prices[i]-asset_prices[i-1])/asset_prices[i-1]
                for i in range(1, len(asset_prices))
            ])
            # regime_labels는 Y_cut(20일 forward dd) 기준으로 분류됨
            # asset_returns와 길이를 맞춰 정렬 (뒤쪽 정렬, Y_cut과 동일 구간)
            ar_aligned = asset_returns[-len(regime_labels):] if len(asset_returns) >= len(regime_labels) else asset_returns

            def regime_hist_cvar(returns_slice, alpha=0.05):
                if len(returns_slice) < 2:
                    return 0.0
                v = np.percentile(returns_slice, alpha*100)
                t = returns_slice[returns_slice <= v]
                return float(t.mean()) if len(t) > 0 else float(v)

            n_align = min(len(ar_aligned), len(regime_labels))
            ar_cut  = ar_aligned[-n_align:]
            rl_cut  = regime_labels[-n_align:]

            r_crash = ar_cut[rl_cut == 0]
            r_elev  = ar_cut[rl_cut == 1]
            r_safe  = ar_cut[rl_cut == 2]

            cvar_crash = regime_hist_cvar(r_crash) if len(r_crash) >= 2 else regime_hist_cvar(ar_cut)
            cvar_elev  = regime_hist_cvar(r_elev)  if len(r_elev)  >= 2 else regime_hist_cvar(ar_cut)
            cvar_safe  = regime_hist_cvar(r_safe)  if len(r_safe)  >= 2 else regime_hist_cvar(ar_cut)

            # ── Tr(ρ·H_risk) = Σᵢ pᵢ·cᵢ = 현재 시장의 기대 꼬리위험
            expected_tail_risk = p_crash*cvar_crash + p_elev*cvar_elev + p_safe*cvar_safe

            # ── β₁ 설명력 (부호 정렬 후 상관계수 재사용, 제곱이라 부호 무관)
            var_explained = round(corr_sign**2 * 100, 1)

            # ── 주요 리스크 팩터
            top_idx    = int(np.argmax(np.abs(beta1)))
            top_factor = MARKET_NAMES[top_idx] if top_idx < len(MARKET_NAMES) else "Market"

            body = json.dumps({
                "points":           points,
                "current":          {"x": cx, "y": cy},
                "regime":           regime,
                "variance_explained": var_explained,
                "top_risk_factor":  top_factor,
                "method":           method,
                "hilbert_used":     hilbert_used,
                # Density Matrix (실제 레짐 확률 기반)
                "density_matrix":   density_matrix,
                # Born Rule 결과 = 레짐 확률 (이미 계산됨)
                "born_rule": {
                    "p_crash": p_crash,
                    "p_elev":  p_elev,
                    "p_safe":  p_safe,
                    "formula": "P(regimeᵢ) = Tr(Πᵢ·ρ) = pᵢ",
                    "note":    "For diagonal ρ, Born Rule reduces to reading diagonal elements"
                },
                # Risk(ρ) 연결용 — 실제 계산된 값
                "risk_hamiltonian": {
                    "formula":     "Tr(ρ·H_risk) = Σᵢ pᵢ·CVaRᵢ",
                    "weights":     [p_crash, p_elev, p_safe],
                    "regime_labels": ["crash", "elevated", "safe"],
                    "cvar_crash":  round(cvar_crash*100, 2),
                    "cvar_elev":   round(cvar_elev*100, 2),
                    "cvar_safe":   round(cvar_safe*100, 2),
                    "expected_tail_risk": round(expected_tail_risk*100, 2),
                },
                "sir_meta": {
                    "Y_definition":  "20-day forward maximum drawdown",
                    "Y_mean":        round(float(Y_cut.mean())*100, 2),
                    "Y_cvar5":       round(float(np.percentile(Y_cut, 5))*100, 2),
                    "corr_beta1_Y":  round(corr_sign, 3),
                    "n_samples":     len(Y_cut),
                    "inner_product": "⟨u,v⟩_Σ = uᵀΣv (Rᵖ(Σ) Hilbert Space)",
                },
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
