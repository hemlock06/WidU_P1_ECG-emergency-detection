"""
MMD 시간분할(Step 2) — 커널을 바꾸면 절단점이 달라지는가? (가우시안 vs 라플라시안)
=================================================================================
`hsic_kernel_comparison.py`의 짝. 그쪽은 Step 1(변수 그룹핑)이 커널에 강건함을 보였다.
GS-SHAP은 **Step 2(MMD 시간분할)에도 같은 RBF 커널**을 쓴다 — 거기도 강건한가?

1차 소스 (GS-SHAP_fulltext.txt):
  §3.3 Eq 6 (unbiased MMD^2):
      MMD^2 = 1/(n(n-1)) Σ_{i≠i'} k(x_i,x_i')
            + 1/(m(m-1)) Σ_{j≠j'} k(y_j,y_j')
            - 2/(nm)     Σ_i Σ_j  k(x_i,y_j)
  §3.3: "If the MMD value at a split exceeds a threshold τ, we accept t as a change point
         and recursively apply the same search to the two subintervals"
  §3.3: "applies the same shift-detection procedure **independently to each feature group**"
  부록 D: "scan t ∈ [s+L_min, e−L_min] and select the split maximizing the unbiased MMD
          statistic (mmd2_unbiased; RBF with median bandwidth)"
  부록 D: "accept a change point only when the maximal statistic exceeds the
          permutation-calibrated threshold, **reused throughout recursion**"
          ↑ ★τ를 재귀마다 다시 뽑지 않는다. 논문 사양 그대로 따른다.
  부록 D: "cap recursion depth at 5"
  L_min(PTB-XL) = 100  (논문 Table — 프라이머 [보충 6])

★논문에 없어서 내가 정한 값 (발표에서 반드시 "제 선택"이라 밝힐 것):
  - α = 0.05            (부록 D는 "at level α"라고만 쓰고 값이 없다)
  - 순열 반복 = 200      (논문에 반복 횟수가 없다)
  - T = 1000            (500Hz 원본을 5배 다운샘플 → 10초를 1000시점. 논문 T=1000에 맞춤)
  - 레코드 5건          (속도. 80건 전수는 안 돌렸다)

★양성 대조가 **필수**인 이유
  Step 1은 07-23 재현 결과가 대조군이었지만 Step 2는 대조군이 없다. 구현이 고장나
  아무것도 못 찾아도 "두 커널이 똑같이 0개"로 보이고, 그걸 "커널 무관"으로 읽으면
  거짓 결론이 된다. → 평균이 확 바뀌는 **인공 신호를 심어** 두 커널이 그 지점을
  잡는지 먼저 확인한다. 못 잡으면 즉시 중단.

사용:
    .venv-hsic-demo/Scripts/python.exe scripts/mmd_kernel_comparison.py
"""
import io
import os
import sys

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import hsic_lead_grouping_demo as m  # noqa: E402

SEED = 42
ALPHA = 0.05          # ★제 선택 (논문 미명시)
N_PERM = 200          # ★제 선택 (논문 미명시)
L_MIN = 100           # 논문 PTB-XL 값
MAX_DEPTH = 5         # 부록 D
T_TARGET = 1000       # ★제 선택 (다운샘플로 논문 T=1000에 맞춤)
N_RECORDS = 5         # ★제 선택 (속도)

# Step 1 결과 그대로 — 커널을 바꿔도 동일했음이 hsic_kernel_comparison.py에서 확인됨
GROUPS = {
    "사지(limb)": ["I", "II", "III", "AVR", "AVL", "AVF"],
    "흉부(chest)": ["V1", "V2", "V3", "V4", "V5", "V6"],
}


# ────────────────────────────────────────────────────────────── 커널 (벡터 표본)
def pdist_sq(X):
    """X:(T,d) → (T,T) 제곱거리. 그룹 표본은 벡터라 노름이 실제로 필요하다."""
    sq = (X ** 2).sum(axis=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
    return np.maximum(d2, 0.0)


def kernel_gaussian(X):
    d2 = pdist_sq(X)
    nz = d2[d2 > 1e-12]
    bw_sq = max(float(np.median(nz)) if nz.size else 1.0, 1e-12)   # median heuristic (제곱거리)
    return np.exp(-d2 / (2.0 * bw_sq))


def kernel_laplacian(X):
    """sigma = 2*sqrt(median(d^2)) — 가우시안과 같은 기준거리에서 같은 값(0.607).
    (hsic_kernel_comparison.py의 'matched'와 동일한 맞춤. 자 눈금이 아니라 꼬리만 비교)"""
    d2 = pdist_sq(X)
    nz = d2[d2 > 1e-12]
    bw_sq = max(float(np.median(nz)) if nz.size else 1.0, 1e-12)
    sigma = max(2.0 * np.sqrt(bw_sq), 1e-12)
    return np.exp(-np.sqrt(d2) / sigma)


KERNELS = {"gaussian(논문)": kernel_gaussian, "laplacian": kernel_laplacian}


# ───────────────────────────────────────── prefix sum으로 모든 분할점을 O(1)에
def prefix2d(K):
    """P[i,j] = K[:i,:j] 합. 블록합을 O(1)에 뽑기 위한 적분영상."""
    P = np.zeros((K.shape[0] + 1, K.shape[1] + 1))
    P[1:, 1:] = K.cumsum(axis=0).cumsum(axis=1)
    return P


def block(P, r0, r1, c0, c1):
    return P[r1, c1] - P[r0, c1] - P[r1, c0] + P[r0, c0]


def mmd2_curve(K, s, e, l_min):
    """구간 [s,e)의 모든 후보 분할점 t에 대한 unbiased MMD^2. Eq 6 그대로."""
    P = prefix2d(K)
    ts, vals = [], []
    for t in range(s + l_min, e - l_min + 1):
        n, mm = t - s, e - t
        if n < 2 or mm < 2:
            continue
        sxx = block(P, s, t, s, t) - n          # 대각(=1) 제외 → unbiased
        syy = block(P, t, e, t, e) - mm
        sxy = block(P, s, t, t, e)
        vals.append(sxx / (n * (n - 1)) + syy / (mm * (mm - 1)) - 2.0 * sxy / (n * mm))
        ts.append(t)
    if not ts:
        return None, None
    return np.array(ts), np.array(vals)


def perm_threshold(K, s, e, l_min, rng, n_perm=N_PERM, alpha=ALPHA):
    """순열로 귀무분포를 만들고 상위 alpha 분위를 tau로. 부록 D 사양(재귀 내 재사용)."""
    stats = []
    idx = np.arange(s, e)
    for _ in range(n_perm):
        p = rng.permutation(idx)
        Kp = K[np.ix_(p, p)]                     # ★행·열을 **같은** 순열로 — 짝 유지
        _, v = mmd2_curve(Kp, 0, e - s, l_min)
        if v is not None and v.size:
            stats.append(v.max())
    if not stats:
        return np.inf
    return float(np.quantile(stats, 1.0 - alpha))


def segment(K, s, e, tau, l_min, depth, out):
    if depth >= MAX_DEPTH:
        return
    ts, vals = mmd2_curve(K, s, e, l_min)
    if ts is None or not vals.size:
        return
    i = int(np.argmax(vals))
    if vals[i] <= tau:
        return
    t = int(ts[i])
    out.append(t)
    segment(K, s, t, tau, l_min, depth + 1, out)
    segment(K, t, e, tau, l_min, depth + 1, out)


def run_one(X, kernel_fn, rng):
    """X:(T,d) 한 그룹의 시계열 → 절단점 리스트."""
    K = kernel_fn(X)
    T = X.shape[0]
    tau = perm_threshold(K, 0, T, L_MIN, rng)
    cuts = []
    segment(K, 0, T, tau, L_MIN, 0, cuts)
    return sorted(cuts), tau


# ─────────────────────────────────────────────────────────────────── 양성 대조
def positive_control():
    print("[양성 대조] 평균이 확 바뀌는 인공 신호에 심은 변화점을 잡는가?")
    rng = np.random.default_rng(SEED)
    T, d, true_cut = T_TARGET, 3, 500
    X = rng.normal(0, 1, size=(T, d))
    X[true_cut:] += 3.0                       # 500번째부터 평균이 +3 — 명백한 변화
    ok_all = True
    for kname, kfn in KERNELS.items():
        cuts, tau = run_one(X, kfn, np.random.default_rng(SEED))
        near = [c for c in cuts if abs(c - true_cut) <= 20]
        ok = len(near) > 0
        ok_all &= ok
        print(f"  {kname:<16} tau={tau:.3e} 절단점={cuts} → 심은 500 근처 검출: "
              f"{'✅' if ok else '❌'}")
    if not ok_all:
        print("  ❌ 구현이 명백한 변화점도 못 잡는다 — 실제 ECG 결과를 믿을 수 없으므로 중단.")
        sys.exit(1)

    print("  [음성 대조] 변화점이 없는 신호에서는 안 잡아야 한다")
    Xn = np.random.default_rng(7).normal(0, 1, size=(T, d))
    for kname, kfn in KERNELS.items():
        cuts, _ = run_one(Xn, kfn, np.random.default_rng(SEED))
        print(f"  {kname:<16} 절단점={cuts} → {'✅ 없음' if not cuts else '⚠️ 거짓 검출'}")
    print()


def main():
    print("=" * 92)
    print("MMD 시간분할(Step 2) — 커널 비교")
    print("=" * 92)
    print(f"설정: T={T_TARGET}(5배 다운샘플) · L_min={L_MIN} · 깊이≤{MAX_DEPTH} · "
          f"α={ALPHA}(제 선택) · 순열 {N_PERM}회(제 선택) · 레코드 {N_RECORDS}건\n")

    positive_control()

    X_all_raw, lead_names, _ = m.load_signals()
    per_rec = X_all_raw.shape[0] // 80
    lead_idx = {ln: i for i, ln in enumerate(lead_names)}

    agree, total = 0, 0
    for r in range(N_RECORDS):
        rec = X_all_raw[r * per_rec:(r + 1) * per_rec]
        rec = rec[::5][:T_TARGET]                     # 500Hz → 100Hz, T=1000
        print(f"── 레코드 #{r} (T={rec.shape[0]})")
        for gname, leads in GROUPS.items():
            cols = [lead_idx[a] for a in leads]
            Xg = rec[:, cols]
            res = {}
            for kname, kfn in KERNELS.items():
                cuts, tau = run_one(Xg, kfn, np.random.default_rng(SEED))
                res[kname] = cuts
                print(f"   {gname:<12} {kname:<16} J={len(cuts)+1:>2} 절단점={cuts}")
            a, b = list(res.values())
            same = (a == b)
            close = (len(a) == len(b) and all(abs(x - y) <= 10 for x, y in zip(a, b)))
            total += 1
            agree += int(same)
            print(f"   {'':<12} → 동일={'O' if same else 'X'} · 10시점이내={'O' if close else 'X'}")
        print()

    print("=" * 92)
    print(f"요약 — (레코드×그룹) {total}건 중 절단점 완전 동일: {agree}/{total}")
    print("=" * 92)
    print("⚠️ 해석은 사람이 한다. 이 스크립트는 '달라지는가/같은가'의 재료만 낸다.")
    print("⚠️ α·순열횟수·T·레코드수는 논문에 없어 내가 정했다 — 발표에서 반드시 밝힐 것.")


if __name__ == "__main__":
    main()
