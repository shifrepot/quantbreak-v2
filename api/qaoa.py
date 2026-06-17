# ══════════════════════════════════════════════════════════
# DEPRECATED — kept for reference only, not called by index.html
#
# This was the single-asset "pick one leverage" QAOA. It was replaced
# by api/portfolio.py (Multi-Asset Risk Allocation QUBO) because
# CVaR(L) = L·CVaR(1) is linear in leverage, so any risk-only cost
# function always picks the lowest leverage — making QAOA mathematically
# unnecessary for that formulation. portfolio.py instead optimizes
# exposure across 4 assets × 3 levels (12 qubits, 81 combinations),
# which is a genuine combinatorial problem.
#
# The /api/qaoa route still exists in vercel.json but the frontend
# no longer calls it. Safe to delete; left here for the record.
# ══════════════════════════════════════════════════════════

from http.server import BaseHTTPRequestHandler
import json, numpy as np
from urllib.parse import urlparse, parse_qs
import urllib.request, time
from datetime import datetime, timedelta

TICKER_MAP = {
    "TQQQ": "TQQQ", "SOXL": "SOXL", "SQQQ": "SQQQ",
    "WTI": "USO",   "NG": "UNG",    "BOIL": "BOIL",
}
MARKET_TICKERS = ["^VIX", "SPY", "TLT", "GLD", "^TNX"]

def fetch_yahoo(sym, days=130):
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
    if len(prices) < 20:
        raise ValueError(f"Not enough data: {len(prices)}")
    return prices

def compute_regime_probs_and_risk(asset_sym, days=130):
    """
    SDR → Density Matrix → Born Rule → Risk Hamiltonian
    레짐 확률과 기대 꼬리위험(Tr(ρH_risk))을 계산.
    (sdr.py와 동일한 로직을 QAOA가 직접 사용하기 위해 재계산)
    """
    from scipy.linalg import eigh

    asset_prices = fetch_yahoo(asset_sym, days=days)
    horizon = 20
    Y = np.array([
        min(asset_prices[t+1:t+horizon+1]) / asset_prices[t] - 1
        for t in range(len(asset_prices) - horizon)
    ])

    mkt_cols = []
    for mkt_sym in MARKET_TICKERS:
        try:
            mkt_cols.append(fetch_yahoo(mkt_sym, days=days))
        except Exception:
            continue
    if len(mkt_cols) < 2:
        raise ValueError("Not enough market data")

    min_len  = min(len(Y), min(len(c) for c in mkt_cols))
    mkt_cols = [c[-min_len:] for c in mkt_cols]
    Y_cut    = Y[-min_len:]
    X = np.column_stack([
        np.array([(c[i]-c[i-1])/c[i-1] for i in range(1, len(c))])
        for c in mkt_cols
    ])
    Y_cut = Y_cut[1:]
    if len(Y_cut) < 15:
        raise ValueError(f"Not enough aligned: {len(Y_cut)}")

    # SIR
    Sigma = np.cov(X.T) + 1e-6*np.eye(X.shape[1])
    h = 8
    quantiles = np.percentile(Y_cut, np.linspace(0,100,h+1))
    sm, sw = [], []
    for j in range(h):
        lo, hi = quantiles[j], quantiles[j+1]
        mask = (Y_cut>=lo)&(Y_cut<=hi) if j==h-1 else (Y_cut>=lo)&(Y_cut<hi)
        if mask.sum() >= 2:
            sm.append(X[mask].mean(axis=0)); sw.append(mask.sum())
    if len(sm) >= 2:
        sm = np.array(sm); sw = np.array(sw,dtype=float); sw/=sw.sum()
        gm = (sw[:,None]*sm).sum(axis=0)
        M  = sum(w*np.outer(m-gm,m-gm) for w,m in zip(sw,sm))
        eigvals, eigvecs = eigh(np.linalg.inv(Sigma)@M)
        idx = np.argsort(eigvals)[::-1]
        beta1 = eigvecs[:, idx[0]]
    else:
        eigvals, eigvecs = eigh(Sigma)
        idx = np.argsort(eigvals)[::-1]
        beta1 = eigvecs[:, idx[0]]

    proj1 = X @ beta1

    y_q33 = float(np.percentile(Y_cut, 33))
    y_q66 = float(np.percentile(Y_cut, 66))
    regime_labels = np.where(Y_cut<=y_q33, 0, np.where(Y_cut<=y_q66, 1, 2))

    total = len(regime_labels)
    p_crash = float(np.sum(regime_labels==0)) / total
    p_elev  = float(np.sum(regime_labels==1)) / total
    p_safe  = float(np.sum(regime_labels==2)) / total

    # 레짐별 historical CVaR (자산 수익률 기준)
    asset_returns = np.array([
        (asset_prices[i]-asset_prices[i-1])/asset_prices[i-1]
        for i in range(1, len(asset_prices))
    ])
    n_align = min(len(asset_returns), len(regime_labels))
    ar_cut  = asset_returns[-n_align:]
    rl_cut  = regime_labels[-n_align:]

    def hcvar(r, alpha=0.05):
        if len(r) < 2: return 0.0
        v = np.percentile(r, alpha*100)
        t = r[r<=v]
        return float(t.mean()) if len(t)>0 else float(v)

    r_crash, r_elev, r_safe = ar_cut[rl_cut==0], ar_cut[rl_cut==1], ar_cut[rl_cut==2]
    cvar_crash = hcvar(r_crash) if len(r_crash)>=2 else hcvar(ar_cut)
    cvar_elev  = hcvar(r_elev)  if len(r_elev)>=2  else hcvar(ar_cut)
    cvar_safe  = hcvar(r_safe)  if len(r_safe)>=2  else hcvar(ar_cut)

    # Tr(ρ·H_risk) = Σᵢ pᵢ·CVaRᵢ
    expected_tail_risk = p_crash*cvar_crash + p_elev*cvar_elev + p_safe*cvar_safe

    recent_dd = float(Y_cut[-5:].mean())
    if recent_dd <= y_q33: regime = "CRASH"
    elif recent_dd <= y_q66: regime = "ELEVATED"
    else: regime = "SAFE"

    return {
        "p_crash": p_crash, "p_elev": p_elev, "p_safe": p_safe,
        "cvar_crash": cvar_crash, "cvar_elev": cvar_elev, "cvar_safe": cvar_safe,
        "expected_tail_risk": expected_tail_risk,
        "regime": regime,
        "asset_mean_return": float(np.mean(ar_cut)),
    }

def rx(theta):
    c, s = np.cos(theta/2), np.sin(theta/2)
    return np.array([[c,-1j*s],[-1j*s,c]], dtype=complex)

def run_qaoa_risk_budget(costs, p=2):
    """
    QAOA — Risk Budget Allocation
    4개 행동(Full/Moderate/Reduced/Defensive) 중 QUBO 최적화

    H_cost = Σᵢ hᵢσᵢᶻ + A·(Σxᵢ-1)²
    각 행동은 독립적인 비용을 가짐 (연속 스케일 아님)
    → 레버리지 선택과 달리 단조성 문제 없음
    """
    n = len(costs)
    dim = 2**n

    h = np.array(costs, dtype=float)
    h = (h-h.min())/(h.max()-h.min()+1e-9)

    A = 3.0
    cost_diag = np.zeros(dim)
    for x in range(dim):
        bits = [(x>>(n-1-i))&1 for i in range(n)]
        n_ones = sum(bits)
        c_cost = sum(h[i]*bits[i] for i in range(n))
        penalty = A*(n_ones-1)**2
        cost_diag[x] = c_cost + penalty

    best_energy = float('inf')
    best_probs  = None
    best_gamma  = 0.4
    best_beta   = 0.5

    for g in [0.2,0.4,0.6,0.8]:
        for b in [0.3,0.5,0.7,0.9]:
            psi = np.ones(dim, dtype=complex)/np.sqrt(dim)
            for layer in range(p):
                psi *= np.exp(-1j*g*cost_diag)
                for i in range(n):
                    gate = rx(2*b)
                    I2 = np.eye(2, dtype=complex)
                    ops = [I2]*n; ops[i]=gate
                    full = ops[0]
                    for m in ops[1:]: full = np.kron(full, m)
                    psi = full @ psi
            probs = np.abs(psi)**2
            energy = float(np.dot(probs, cost_diag))
            if energy < best_energy:
                best_energy = energy
                best_probs  = probs.copy()
                best_gamma  = g
                best_beta   = b

    cp = np.zeros(n)
    for state in range(dim):
        bits = [(state>>(n-1-i))&1 for i in range(n)]
        if sum(bits) == 1:
            cp[bits.index(1)] += best_probs[state]
    total = cp.sum()
    if total > 0:
        cp /= total
    else:
        cp[np.argmin(h)] = 1.0

    return cp, best_gamma, best_beta

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        qs    = parse_qs(urlparse(self.path).query)
        asset = qs.get("asset", ["TQQQ"])[0].upper()
        sym   = TICKER_MAP.get(asset, asset)

        try:
            # ── SDR → Density Matrix → Born Rule → Risk Hamiltonian
            r = compute_regime_probs_and_risk(sym, days=130)
            expected_tail_risk = r["expected_tail_risk"]  # Tr(ρH_risk), 음수
            mu = r["asset_mean_return"]

            # ── Risk Budget 옵션 4개 (연속 레버리지가 아닌 독립 정책)
            budgets = [
                ("Full Exposure",     1.00),
                ("Moderate Exposure", 0.70),
                ("Reduced Exposure",  0.40),
                ("Defensive",         0.15),
            ]

            # 비용함수:
            # cost = |Tr(ρH_risk) × budget| + opportunity_cost
            # opportunity_cost = (1-budget) × 기회비용(정상시장 기대수익 상실)
            costs = []
            risk_costs = []
            opp_costs  = []
            for name, budget in budgets:
                risk_cost = abs(expected_tail_risk * budget)
                opp_cost  = (1-budget) * abs(mu) * 5  # 기회비용 가중치
                cost = risk_cost + opp_cost
                costs.append(cost)
                risk_costs.append(round(risk_cost*100, 3))
                opp_costs.append(round(opp_cost*100, 3))

            # QAOA — Risk Budget 최적화
            probs, best_g, best_b = run_qaoa_risk_budget(costs, p=2)
            opt_idx = int(np.argmax(probs))
            opt_name, opt_budget = budgets[opt_idx]

            # ── Action 결정
            if opt_budget >= 0.85:
                action = "Maintain Full Exposure"
            elif opt_budget >= 0.55:
                action = "Slightly Reduce Exposure"
            elif opt_budget >= 0.30:
                action = "Reduce Exposure"
            else:
                action = "Defensive Positioning"

            body = json.dumps({
                # Risk Budget 결과 (레버리지 대체)
                "current_regime":        r["regime"],
                "crash_probability":     round(r["p_crash"]*100, 1),
                "elevated_probability":  round(r["p_elev"]*100, 1),
                "safe_probability":      round(r["p_safe"]*100, 1),
                "expected_tail_risk":    round(expected_tail_risk*100, 2),
                "recommended_budget":    round(opt_budget*100, 0),
                "recommended_action":    action,
                "budget_idx":            opt_idx,
                "budget_labels":         [b[0] for b in budgets],
                "budget_pcts":           [round(b[1]*100,0) for b in budgets],
                "probabilities":         [round(float(p_),4) for p_ in probs],
                "costs":                 [round(c,4) for c in costs],
                "risk_costs":            risk_costs,
                "opportunity_costs":     opp_costs,
                "cvar_by_regime": {
                    "crash":   round(r["cvar_crash"]*100, 2),
                    "elevated": round(r["cvar_elev"]*100, 2),
                    "safe":    round(r["cvar_safe"]*100, 2),
                },
                "n_qubits": len(budgets),
                "p_layers": 2,
                "best_gamma": round(best_g,3),
                "best_beta":  round(best_b,3),
                "circuit_info": {
                    "ansatz":  "|ψ(γ,β)⟩ = ∏ e^{-iβH_M} e^{-iγH_C} |s⟩",
                    "H_cost":  "Σᵢ cost(budgetᵢ)·σᵢᶻ + A·(Σxᵢ-1)²",
                    "objective": "min |Tr(ρH_risk)×budget| + opportunity_cost(1-budget)",
                    "pipeline": "SDR → Density Matrix → Born Rule → Risk Hamiltonian → QAOA Risk Budget",
                }
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
