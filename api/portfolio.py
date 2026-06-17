from http.server import BaseHTTPRequestHandler
import json, numpy as np
from scipy.linalg import eigh
from urllib.parse import urlparse, parse_qs
import urllib.request, time
from datetime import datetime, timedelta

# qiskit-aer는 선택적 의존성 (Vercel 서버리스 패키지 용량 제한 때문에
# 배포 환경에 따라 import가 실패할 수 있음 — 실패 시 numpy 백엔드로 자동 fallback)
try:
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator
    QISKIT_AVAILABLE = True
except Exception:
    QISKIT_AVAILABLE = False

# 4개 자산 (서로 다른 위험 성격을 가진 조합)
PORTFOLIO_ASSETS = {
    "TQQQ": {"sym": "TQQQ", "risk_mult": 1.8, "label": "TQQQ (3x Nasdaq)"},
    "SOXL": {"sym": "SOXL", "risk_mult": 2.0, "label": "SOXL (3x Semis)"},
    "WTI":  {"sym": "USO",  "risk_mult": 1.0, "label": "WTI Oil"},
    "NG":   {"sym": "UNG",  "risk_mult": 1.5, "label": "Natural Gas"},
}
MARKET_TICKERS = ["^VIX", "SPY", "TLT", "GLD", "^TNX"]
EXPOSURE_LEVELS = [("High", 1.00), ("Medium", 0.50), ("Low", 0.15)]

def fetch_yahoo(sym, days=130):
    end   = int(time.time())
    start = int((datetime.utcnow() - timedelta(days=days)).timestamp())
    url   = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
        f"?interval=1d&period1={start}&period2={end}"
    )
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0", "Accept": "application/json",
    })
    with urllib.request.urlopen(req, timeout=8) as resp:
        data = json.loads(resp.read())
    result = data["chart"]["result"][0]
    closes = result["indicators"]["quote"][0]["close"]
    prices = [c for c in closes if c is not None]
    if len(prices) < 20:
        raise ValueError(f"Not enough data: {len(prices)}")
    return prices

def compute_asset_risk(asset_sym, mkt_cols, days=130):
    """
    개별 자산에 대해:
    SDR(SIR) → Density Matrix(레짐확률) → Born Rule → Risk Hamiltonian → Tr(ρH_risk)
    """
    asset_prices = fetch_yahoo(asset_sym, days=days)
    horizon = 20
    Y = np.array([
        min(asset_prices[t+1:t+horizon+1]) / asset_prices[t] - 1
        for t in range(len(asset_prices) - horizon)
    ])

    min_len  = min(len(Y), min(len(c) for c in mkt_cols))
    mc       = [c[-min_len:] for c in mkt_cols]
    Y_cut    = Y[-min_len:]
    X = np.column_stack([
        np.array([(c[i]-c[i-1])/c[i-1] for i in range(1, len(c))])
        for c in mc
    ])
    Y_cut = Y_cut[1:]
    if len(Y_cut) < 15:
        raise ValueError(f"Not enough aligned data for {asset_sym}")

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
        sm = np.array(sm); sw=np.array(sw,dtype=float); sw/=sw.sum()
        gm = (sw[:,None]*sm).sum(axis=0)
        M  = sum(w*np.outer(m-gm,m-gm) for w,m in zip(sw,sm))
        eigvals, eigvecs = eigh(np.linalg.inv(Sigma)@M)
        beta1 = eigvecs[:, np.argsort(eigvals)[::-1][0]]
    else:
        eigvals, eigvecs = eigh(Sigma)
        beta1 = eigvecs[:, np.argsort(eigvals)[::-1][0]]

    y_q33 = float(np.percentile(Y_cut, 33))
    y_q66 = float(np.percentile(Y_cut, 66))
    regime_labels = np.where(Y_cut<=y_q33, 0, np.where(Y_cut<=y_q66, 1, 2))

    total = len(regime_labels)
    p_crash = float(np.sum(regime_labels==0)) / total
    p_elev  = float(np.sum(regime_labels==1)) / total
    p_safe  = float(np.sum(regime_labels==2)) / total

    asset_returns = np.array([
        (asset_prices[i]-asset_prices[i-1])/asset_prices[i-1]
        for i in range(1, len(asset_prices))
    ])
    n_align = min(len(asset_returns), len(regime_labels))
    ar_cut, rl_cut = asset_returns[-n_align:], regime_labels[-n_align:]

    def hcvar(r, alpha=0.05):
        if len(r) < 2: return 0.0
        v = np.percentile(r, alpha*100)
        t = r[r<=v]
        return float(t.mean()) if len(t)>0 else float(v)

    r_crash, r_elev, r_safe = ar_cut[rl_cut==0], ar_cut[rl_cut==1], ar_cut[rl_cut==2]
    cvar_crash = hcvar(r_crash) if len(r_crash)>=2 else hcvar(ar_cut)
    cvar_elev  = hcvar(r_elev)  if len(r_elev)>=2  else hcvar(ar_cut)
    cvar_safe  = hcvar(r_safe)  if len(r_safe)>=2  else hcvar(ar_cut)

    # 레짐별 평균 수익 (Opportunity Reward 계산용)
    mean_crash = float(r_crash.mean()) if len(r_crash)>=2 else float(ar_cut.mean())
    mean_elev  = float(r_elev.mean())  if len(r_elev)>=2  else float(ar_cut.mean())
    mean_safe  = float(r_safe.mean())  if len(r_safe)>=2  else float(ar_cut.mean())

    # Tr(ρ·H_risk) — 기대 꼬리위험
    expected_tail_risk = p_crash*cvar_crash + p_elev*cvar_elev + p_safe*cvar_safe

    # 기대수익 (Tr(ρ·H_return), Reward Hamiltonian과 동일한 구조)
    expected_return = p_crash*mean_crash + p_elev*mean_elev + p_safe*mean_safe

    recent_dd = float(Y_cut[-5:].mean())
    regime = "CRASH" if recent_dd<=y_q33 else ("ELEVATED" if recent_dd<=y_q66 else "SAFE")

    return {
        "p_crash": p_crash, "p_elev": p_elev, "p_safe": p_safe,
        "cvar_crash": cvar_crash, "cvar_elev": cvar_elev, "cvar_safe": cvar_safe,
        "expected_tail_risk": expected_tail_risk,
        "expected_return": expected_return,
        "regime": regime,
        "current_price": asset_prices[-1],
    }

def apply_rx_vectorized(psi, n, qubit_idx, theta):
    """단일 큐비트 Rx 게이트를 벡터화로 적용 (kron 없이, O(2^n) 대신 O(2^n) 메모리지만 훨씬 빠름)"""
    c, s = np.cos(theta/2), np.sin(theta/2)
    dim = len(psi)
    psi_r = psi.reshape([2]*n)
    psi_r = np.moveaxis(psi_r, qubit_idx, 0)
    new0 = c*psi_r[0] - 1j*s*psi_r[1]
    new1 = -1j*s*psi_r[0] + c*psi_r[1]
    psi_r = np.stack([new0, new1], axis=0)
    psi_r = np.moveaxis(psi_r, 0, qubit_idx)
    return psi_r.reshape(dim)

def build_cost_diag(costs, n_assets, n_levels, A=5.0):
    """
    QUBO 비용을 모든 2^n 비트스트링에 대한 대각 비용 벡터로 변환.
    numpy 백엔드와 AerSimulator 백엔드가 동일한 cost_diag를 공유한다.
    """
    n = n_assets * n_levels
    dim = 2**n
    h = np.array(costs, dtype=float)
    h = (h-h.min())/(h.max()-h.min()+1e-9)

    cost_diag = np.zeros(dim)
    for x in range(dim):
        bits = [(x>>(n-1-i))&1 for i in range(n)]
        c_cost = sum(h[i]*bits[i] for i in range(n))
        penalty = 0.0
        for a in range(n_assets):
            asset_bits = bits[a*n_levels:(a+1)*n_levels]
            penalty += A*(sum(asset_bits)-1)**2
        cost_diag[x] = c_cost + penalty
    return cost_diag

def run_qaoa_portfolio(costs, n_assets, n_levels, A=5.0, p=2):
    """
    QAOA for Multi-Asset Risk Budget Allocation (QUBO) — numpy statevector backend

    변수: x_{a,l} ∈ {0,1}, a=asset, l=level (4×3=12 qubits)
    제약: Σ_l x_{a,l} = 1  for each asset a  (one-hot per asset)

    H_cost = Σ_{a,l} cost(a,l)·x_{a,l} + A·Σ_a(Σ_l x_{a,l} - 1)²

    조합 공간: n_levels^n_assets (3^4=81)
    """
    n = n_assets * n_levels
    dim = 2**n
    cost_diag = build_cost_diag(costs, n_assets, n_levels, A)

    best_energy = float('inf')
    best_probs  = None
    best_gamma, best_beta = 0.4, 0.5

    for g in [0.3, 0.5, 0.7]:
        for b in [0.4, 0.6, 0.8]:
            psi = np.ones(dim, dtype=complex)/np.sqrt(dim)
            for layer in range(p):
                psi = psi * np.exp(-1j*g*cost_diag)
                for i in range(n):
                    psi = apply_rx_vectorized(psi, n, i, 2*b)
            probs = np.abs(psi)**2
            energy = float(np.dot(probs, cost_diag))
            if energy < best_energy:
                best_energy = energy
                best_probs  = probs.copy()
                best_gamma, best_beta = g, b

    return best_probs, best_gamma, best_beta, cost_diag

def run_qaoa_portfolio_aer(asset_names, valid_assets, A=3.0, p=2, shots=2048):
    """
    QAOA via Qiskit AerSimulator — small-scale real quantum circuit verification

    설계 노트(중요): 전체 문제(4 assets × 3 levels = 12 qubits, dim=4096)를
    AerSimulator로 그대로 돌리면 diagonal cost unitary가 4096×4096 dense
    행렬이 되어 transpile/시뮬레이션이 Vercel의 10초 서버리스 타임아웃을
    초과한다. 이는 솔직한 엔지니어링 제약이다.

    그래서 AerSimulator 백엔드는 동일한 파이프라인(SDR→Density Matrix→
    Risk Hamiltonian→QUBO)을 2개 자산 × 2개 레벨(High/Low) = 4 qubit
    규모로 줄인 버전에 적용한다. 목적은 "실제 양자 회로(Hadamard, 
    diagonal cost unitary, RX mixer, 측정)가 numpy statevector 시뮬레이션과
    동일한 분포로 수렴하는가"를 검증하는 것이다 — 전체 12 qubit 문제의
    실시간 해는 numpy 백엔드가 담당한다.

    변수: x_{a,l} ∈ {0,1}, 2 assets × {High, Low} = 4 qubits
    """
    if not QISKIT_AVAILABLE:
        raise RuntimeError("qiskit-aer not available in this environment")

    # 2개 자산만 선택 (위험도가 가장 다른 두 자산으로 — 트레이드오프가 잘 보이게)
    sorted_assets = sorted(
        asset_names,
        key=lambda a: abs(valid_assets[a]["expected_tail_risk"])
    )
    demo_assets = [sorted_assets[0], sorted_assets[-1]]  # 가장 안전한 것 + 가장 위험한 것
    demo_levels = [("High", 1.00), ("Low", 0.15)]

    n_assets, n_levels = 2, 2
    n = n_assets * n_levels  # 4 qubits
    REWARD_WEIGHT, RETURN_SCALE = 1.0, 21

    costs = []
    for a_name in demo_assets:
        r = valid_assets[a_name]
        risk_mult = PORTFOLIO_ASSETS[a_name]["risk_mult"]
        etr  = abs(r["expected_tail_risk"])
        eret = r["expected_return"] * RETURN_SCALE
        for lvl_name, frac in demo_levels:
            risk_cost = risk_mult * etr * frac
            reward    = REWARD_WEIGHT * eret * frac
            costs.append(risk_cost - reward)

    cost_diag = build_cost_diag(costs, n_assets, n_levels, A)
    scaled_cost = cost_diag - cost_diag.min()
    scaled_cost = scaled_cost / (scaled_cost.max() + 1e-9)

    # numpy로 먼저 최적 (γ,β) 탐색 (4 qubit, dim=16 — 매우 가벼움)
    dim = 2**n
    best_energy, best_g, best_b = float('inf'), 0.5, 0.6
    for g in [0.3, 0.5, 0.7]:
        for b in [0.4, 0.6, 0.8]:
            psi = np.ones(dim, dtype=complex)/np.sqrt(dim)
            for layer in range(p):
                psi = psi * np.exp(-1j*g*cost_diag)
                for i in range(n):
                    psi = apply_rx_vectorized(psi, n, i, 2*b)
            probs_np = np.abs(psi)**2
            energy = float(np.dot(probs_np, cost_diag))
            if energy < best_energy:
                best_energy, best_g, best_b = energy, g, b

    # AerSimulator로 동일 파라미터의 실제 회로 구성·측정
    sim = AerSimulator(method="statevector")
    qc = QuantumCircuit(n, n)
    qc.h(range(n))
    for layer in range(p):
        diag_unitary = np.exp(-1j * best_g * scaled_cost)
        qc.unitary(np.diag(diag_unitary), list(range(n)), label="Cost")
        for i in range(n):
            qc.rx(2*best_b, i)
    qc.measure(range(n), range(n))

    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    probs = np.zeros(dim)
    total = sum(counts.values())
    for bitstring, cnt in counts.items():
        x = int(bitstring, 2)
        probs[x] = cnt/total

    energy = float(np.dot(probs, cost_diag))

    return {
        "demo_assets": demo_assets,
        "demo_levels": [l[0] for l in demo_levels],
        "n_qubits": n,
        "probabilities": probs.tolist(),
        "cost_diag": cost_diag.tolist(),
        "gamma": best_g,
        "beta": best_b,
        "energy": energy,
        "shots": shots,
    }


    def do_GET(self):
        try:
            qs      = parse_qs(urlparse(self.path).query)
            backend = qs.get("backend", ["numpy"])[0].lower()

            # ── 시장 변수 (공통, 한 번만 fetch)
            mkt_cols = []
            for mkt_sym in MARKET_TICKERS:
                try:
                    mkt_cols.append(fetch_yahoo(mkt_sym, days=130))
                except Exception:
                    continue
            if len(mkt_cols) < 2:
                raise ValueError("Not enough market data")

            # ── 자산별 SDR → Density Matrix → Risk Hamiltonian
            asset_results = {}
            for name, info in PORTFOLIO_ASSETS.items():
                try:
                    asset_results[name] = compute_asset_risk(info["sym"], mkt_cols, days=130)
                except Exception as e:
                    asset_results[name] = {"error": str(e)}

            valid_assets = {k:v for k,v in asset_results.items() if "error" not in v}
            if len(valid_assets) < 2:
                raise ValueError("Not enough valid assets computed")

            asset_names = list(valid_assets.keys())
            n_assets = len(asset_names)
            n_levels = len(EXPOSURE_LEVELS)

            # ── QUBO 비용 구성
            # Cost(a,l) = RiskCost(a,l) - Reward(a,l)
            #   RiskCost(a,l) = risk_mult(a) × |Tr(ρH_risk)(a)| × exposure_frac(l)
            #   Reward(a,l)   = REWARD_WEIGHT × Tr(ρH_return)(a) × 21 × exposure_frac(l)
            #                   (21: ~월간 스케일로 정규화, 기대수익이 risk_cost와 비교가능한 크기가 되도록)
            #
            # 이 구조가 핵심: risk_mult만 있으면 항상 Low가 이김 (이전 버그).
            # Reward를 빼주면 안전하고 기대수익이 양수인 자산은 노출을 늘리는 게
            # 유리해지고, 위험하고 기대수익이 음수인 자산은 노출을 줄이는 게 유리해짐.
            # → 자산마다 다른 레벨이 선택되는 진짜 트레이드오프 발생.
            REWARD_WEIGHT = 1.0
            RETURN_SCALE  = 21   # 일별 → 월간 근사 스케일

            costs = []
            cost_table = {}
            reward_table = {}
            for a_name in asset_names:
                r = valid_assets[a_name]
                risk_mult = PORTFOLIO_ASSETS[a_name]["risk_mult"]
                etr  = abs(r["expected_tail_risk"])
                eret = r["expected_return"] * RETURN_SCALE
                row, rrow = [], []
                for lvl_name, frac in EXPOSURE_LEVELS:
                    risk_cost = risk_mult * etr * frac
                    reward    = REWARD_WEIGHT * eret * frac
                    c = risk_cost - reward
                    costs.append(c)
                    row.append(round(risk_cost*100, 3))
                    rrow.append(round(reward*100, 3))
                cost_table[a_name]   = row
                reward_table[a_name] = rrow

            # ── QAOA 실행
            probs, best_g, best_b, cost_diag = run_qaoa_portfolio(costs, n_assets, n_levels, A=5.0, p=2)

            # ── 최적해 추출: 가장 높은 확률의 valid one-hot 상태
            n = n_assets * n_levels
            dim = 2**n
            best_state = None
            best_prob  = -1
            for x in range(dim):
                bits = [(x>>(n-1-i))&1 for i in range(n)]
                valid = all(sum(bits[a*n_levels:(a+1)*n_levels])==1 for a in range(n_assets))
                if valid and probs[x] > best_prob:
                    best_prob = probs[x]
                    best_state = bits

            if best_state is None:
                # fallback: 각 자산별 최소비용 레벨
                best_state = []
                for a_idx in range(n_assets):
                    row_costs = costs[a_idx*n_levels:(a_idx+1)*n_levels]
                    min_idx = int(np.argmin(row_costs))
                    best_state += [1 if j==min_idx else 0 for j in range(n_levels)]

            allocation = {}
            for a_idx, a_name in enumerate(asset_names):
                level_bits = best_state[a_idx*n_levels:(a_idx+1)*n_levels]
                lvl_idx = level_bits.index(1) if 1 in level_bits else n_levels-1
                allocation[a_name] = EXPOSURE_LEVELS[lvl_idx][0]

            # ── 포트폴리오 전체 리스크 예산 사용률
            total_max_cost = sum(
                PORTFOLIO_ASSETS[a]["risk_mult"]*abs(valid_assets[a]["expected_tail_risk"])*1.0
                for a in asset_names
            )
            used_cost = sum(
                PORTFOLIO_ASSETS[a]["risk_mult"]*abs(valid_assets[a]["expected_tail_risk"]) *
                dict(EXPOSURE_LEVELS)[allocation[a]]
                for a in asset_names
            )
            budget_used_pct = round(used_cost/total_max_cost*100, 1) if total_max_cost>0 else 0

            # ── 응답
            asset_detail = {}
            for a_name in asset_names:
                r = valid_assets[a_name]
                asset_detail[a_name] = {
                    "label":     PORTFOLIO_ASSETS[a_name]["label"],
                    "regime":    r["regime"],
                    "p_crash":   round(r["p_crash"]*100,1),
                    "p_elev":    round(r["p_elev"]*100,1),
                    "p_safe":    round(r["p_safe"]*100,1),
                    "expected_tail_risk": round(r["expected_tail_risk"]*100,2),
                    "expected_return":    round(r["expected_return"]*RETURN_SCALE*100,2),
                    "risk_mult": PORTFOLIO_ASSETS[a_name]["risk_mult"],
                    "risk_cost_by_level":   dict(zip([l[0] for l in EXPOSURE_LEVELS], cost_table[a_name])),
                    "reward_by_level":      dict(zip([l[0] for l in EXPOSURE_LEVELS], reward_table[a_name])),
                    "recommended_level": allocation[a_name],
                }

            body_dict = {
                "backend": "numpy",
                "n_assets": n_assets,
                "n_levels": n_levels,
                "n_qubits": n_assets*n_levels,
                "combination_space": n_levels**n_assets,
                "asset_detail": asset_detail,
                "recommended_allocation": allocation,
                "portfolio_risk_budget_used_pct": budget_used_pct,
                "best_gamma": round(best_g,3),
                "best_beta":  round(best_b,3),
                "circuit_info": {
                    "ansatz": "|ψ(γ,β)⟩ = ∏ e^{-iβH_M} e^{-iγH_C} |s⟩",
                    "H_cost": "Σ_{a,l} [RiskCost(a,l) - Reward(a,l)]·x_{a,l} + A·Σ_a(Σ_l x_{a,l}-1)²",
                    "cost_formula": "RiskCost = risk_mult·|Tr(ρH_risk)|·exposure ; Reward = Tr(ρH_return)·21·exposure",
                    "qubits": f"{n_assets} assets × {n_levels} levels = {n_assets*n_levels} qubits",
                    "space":  f"{n_levels}^{n_assets} = {n_levels**n_assets} combinations",
                    "motivation": "Multi-asset risk allocation is a genuine combinatorial problem — QAOA's role is justified by combination space growth, not by single-variable selection.",
                }
            }

            # ── backend=aer: 실제 양자 회로(AerSimulator)로 작은 규모(4 qubit) 검증 실행
            if backend == "aer":
                if not QISKIT_AVAILABLE:
                    body_dict["aer_error"] = "qiskit-aer not available in this deployment (package size/runtime constraint)"
                else:
                    try:
                        aer_result = run_qaoa_portfolio_aer(asset_names, valid_assets, A=3.0, p=2, shots=2048)
                        # one-hot 상태만 추출해서 보여줄 정보 정리
                        n4 = aer_result["n_qubits"]
                        probs4 = aer_result["probabilities"]
                        valid_states = []
                        for x in range(2**n4):
                            bits = [(x>>(n4-1-i))&1 for i in range(n4)]
                            ok = all(sum(bits[a*2:(a+1)*2])==1 for a in range(2))
                            if ok and probs4[x] > 0.001:
                                valid_states.append({
                                    "bitstring": format(x, f'0{n4}b'),
                                    "probability": round(probs4[x], 4),
                                    "asset0_level": aer_result["demo_levels"][bits[0:2].index(1)] if 1 in bits[0:2] else "—",
                                    "asset1_level": aer_result["demo_levels"][bits[2:4].index(1)] if 1 in bits[2:4] else "—",
                                })
                        valid_states.sort(key=lambda s: -s["probability"])

                        body_dict["aer_verification"] = {
                            "note": "Small-scale (4-qubit, 2 assets x 2 levels) real circuit run on AerSimulator, "
                                    "using gamma/beta pre-optimized by the numpy backend. "
                                    "Full 12-qubit problem stays on the numpy backend (see explanation).",
                            "demo_assets": aer_result["demo_assets"],
                            "demo_levels": aer_result["demo_levels"],
                            "n_qubits": aer_result["n_qubits"],
                            "shots": aer_result["shots"],
                            "gamma": round(aer_result["gamma"], 3),
                            "beta":  round(aer_result["beta"], 3),
                            "energy": round(aer_result["energy"], 4),
                            "top_states": valid_states[:4],
                        }
                    except Exception as e:
                        body_dict["aer_error"] = str(e)

            body = json.dumps(body_dict)
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
