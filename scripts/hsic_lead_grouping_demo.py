"""
HSIC 기반 12리드 그룹핑 재현 (GS-SHAP arxiv 2601.06114, Appendix D 사양)
=====================================================================
목적: GS-SHAP의 HSIC 그룹핑 단계만 떼어내 실제 12-lead ECG(PTB-XL 서브셋)에
적용했을 때, 데이터 기반 그룹이 알려진 해부학적 리드 그룹(하벽/측벽/전중격)을
복원하는지 관찰한다. — 카드 D 실증(계획 §6.5).

★이건 논문 재현이 아니다(전체 파이프라인·베이스라인·ΔAUC 비교 없음).
  HSIC 그룹핑이라는 *한 단계*만, 논문 부록 D의 하이퍼파라미터로 재구현한다.

수식(1차소스 = GS-SHAP_fulltext.txt):
  H = I_n - (1/n) 11^T                         (centering matrix)
  HSIC(Xd,Xd') = 1/(n-1)^2 * tr(H K H L)        (K=RBF(Xd), L=RBF(Xd'))
  구현 최적화: H@K@H는 행/열 평균 차감(더블센터링)으로 O(n^2)에 계산되고,
  tr(대칭 A @ 대칭 B) = sum(A*B) (elementwise, Frobenius inner product) —
  O(n^3) 행렬곱을 피한다. 두 방식이 일치하는지 자가검증으로 확인한다.

하이퍼파라미터(부록 D, 논문 그대로):
  N_HSIC=3000(가용시 3000, 부족하면 가용량 전부) · median heuristic 대역폭
  · eigengap으로 k 선택(k<=6) · affinity=precomputed spectral clustering.
"""
import glob
import os

import numpy as np
import wfdb
from sklearn.cluster import SpectralClustering

SEED = 42
N_HSIC_TARGET = 3000

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "data", "raw", "ptbxl")

ANATOMICAL_GROUPS = {
    "하벽(inferior)": ["II", "III", "AVF"],
    "측벽(lateral)": ["I", "AVL", "V5", "V6"],
    "전중격(anteroseptal)": ["V1", "V2", "V3", "V4"],
    "aVR(단독)": ["AVR"],
}


def load_signals():
    with open(os.path.join(DATA_DIR, "SUBSET_RECORDS.txt")) as f:
        records = [line.strip() for line in f if line.strip()]
    all_signals = []
    lead_names = None
    for rec in records:
        rec_path = os.path.join(DATA_DIR, rec.replace(".hea", "").replace(".dat", ""))
        r = wfdb.rdrecord(rec_path)
        if lead_names is None:
            lead_names = [s.upper() for s in r.sig_name]
        all_signals.append(r.p_signal.astype(np.float64))  # (5000, 12)
    X_all = np.concatenate(all_signals, axis=0)  # (n_records*5000, 12)
    return X_all, lead_names, len(records)


def median_heuristic_bandwidth(x):
    """x: (n,) 1D. 제곱거리 median heuristic, 0(자기자신) 제외."""
    diff = x[:, None] - x[None, :]
    sq = diff ** 2
    nz = sq[sq > 1e-12]
    med = np.median(nz) if nz.size else 1.0
    return max(med, 1e-6)


def rbf_kernel_1d(x, bandwidth_sq):
    diff = x[:, None] - x[None, :]
    return np.exp(-(diff ** 2) / (2.0 * bandwidth_sq))


def double_center(K):
    """H@K@H — 행/열 평균 차감으로 O(n^2). 표준 커널 이중센터링 공식."""
    row_mean = K.mean(axis=1, keepdims=True)
    col_mean = K.mean(axis=0, keepdims=True)
    grand_mean = K.mean()
    return K - row_mean - col_mean + grand_mean


def hsic_fast(Kc_d, K_dprime, n):
    """tr(HKHL) = sum(Kc_d * K_dprime) — 대칭행렬 프로베니우스 내적으로 O(n^2)."""
    return np.sum(Kc_d * K_dprime) / ((n - 1) ** 2)


def hsic_naive_matmul_check(H, K, L, n):
    """자가검증용 — O(n^3) 정공법 tr(HKHL)로 fast 버전과 대조."""
    return np.trace(H @ K @ H @ L) / ((n - 1) ** 2)


def main():
    print("[1/4] 신호 로드")
    X_all, lead_names, n_records = load_signals()
    print(f"  레코드 {n_records}건 · 리드 순서: {lead_names}")
    print(f"  풀링된 전체 시점: {X_all.shape[0]}")

    n = min(N_HSIC_TARGET, X_all.shape[0])
    rng = np.random.default_rng(SEED)
    idx = rng.choice(X_all.shape[0], size=n, replace=False)
    X = X_all[idx]  # (n, 12)
    print(f"  HSIC 표본: n={n} (seed={SEED})")

    D = X.shape[1]
    print(f"\n[2/4] 리드별 RBF 커널 + 이중센터링 (D={D})")
    K_list, Kc_list, bw_list = [], [], []
    for d in range(D):
        bw = median_heuristic_bandwidth(X[:, d])
        K = rbf_kernel_1d(X[:, d], bw)
        K_list.append(K)
        Kc_list.append(double_center(K))
        bw_list.append(bw)
    print(f"  bandwidth^2 (median heuristic): {[round(b,4) for b in bw_list]}")

    print("\n[3/4] HSIC 12x12 affinity 행렬")
    W = np.zeros((D, D))
    for i in range(D):
        for j in range(D):
            W[i, j] = hsic_fast(Kc_list[i], K_list[j], n)

    # 자가검증: fast(O(n^2)) vs naive matmul(O(n^3)) 대조 — 표본 시점(작은 n')으로.
    n_check = min(300, n)
    Hc = np.eye(n_check) - np.ones((n_check, n_check)) / n_check
    Xc = X[:n_check]
    d0, d1 = 0, 1
    Kc0 = rbf_kernel_1d(Xc[:, d0], bw_list[d0])
    K1c = rbf_kernel_1d(Xc[:, d1], bw_list[d1])
    fast_val = hsic_fast(double_center(Kc0), K1c, n_check)
    naive_val = hsic_naive_matmul_check(Hc, Kc0, K1c, n_check)
    err = abs(fast_val - naive_val)
    print(f"  ★자가검증(O(n^2) vs O(n^3) 정공법, n'={n_check}): fast={fast_val:.6e} naive={naive_val:.6e} 오차={err:.2e}")
    assert err < 1e-8, "HSIC fast/naive 불일치 — 이중센터링 구현 오류"
    print("  ✓ 두 계산법 일치 (10^-8 이내)")

    sym_err = np.abs(W - W.T).max()
    print(f"  W 대칭성 오차: {sym_err:.2e} (수치오차 수준이어야 함)")
    W = (W + W.T) / 2  # 부동소수 비대칭 제거

    neg = (W < 0).sum()
    if neg:
        print(f"  ⚠️ 음수 HSIC 추정값 {neg}개(유한표본 편향, 알려진 현상) — 0으로 클리핑")
        W = np.clip(W, 0, None)

    print("\n  HSIC affinity 행렬 (리드 순서 =", lead_names, "):")
    with np.printoptions(precision=3, suppress=True, linewidth=160):
        print(W)

    print("\n[4/4] eigengap → k 선택 → spectral clustering")
    deg = W.sum(axis=1)
    deg_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(deg, 1e-12)))
    L_norm = np.eye(D) - deg_inv_sqrt @ W @ deg_inv_sqrt
    eigvals = np.sort(np.linalg.eigvalsh(L_norm))
    print(f"  정규화 라플라시안 고유값(오름차순): {[round(e,4) for e in eigvals]}")

    k_max = min(6, D - 1)
    gaps = np.diff(eigvals[: k_max + 1])
    k = int(np.argmax(gaps)) + 1
    k = max(2, min(k, k_max))
    print(f"  eigengap: {[round(g,4) for g in gaps]} → k={k}")

    sc = SpectralClustering(n_clusters=k, affinity="precomputed", random_state=SEED)
    labels = sc.fit_predict(W)

    print(f"\n  === 결과: k={k}개 그룹 ===")
    clusters = {}
    for lead, lab in zip(lead_names, labels):
        clusters.setdefault(int(lab), []).append(lead)
    for cid, leads in sorted(clusters.items()):
        print(f"  그룹 {cid}: {leads}")

    print("\n  === 해부학적 그룹(참조, 논문에 없는 임상 사전지식) ===")
    for name, leads in ANATOMICAL_GROUPS.items():
        print(f"  {name}: {leads}")

    return clusters


if __name__ == "__main__":
    main()
