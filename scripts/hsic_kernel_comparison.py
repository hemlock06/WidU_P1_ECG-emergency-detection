"""
HSIC 리드그룹핑 — 커널을 바꾸면 그룹이 달라지는가? (가우시안 vs 라플라시안)
==========================================================================
왜: GS-SHAP(arxiv 2601.06114)은 RBF(가우시안) 커널만 쓰고 **선택 이유도 민감도
실험도 쓰지 않는다**. RBF가 커널 방법의 사실상 기본값이라 굳이 안 쓴 것으로 보이지만,
ECG는 대부분 기저선 근처에 있다가 R파에서만 크게 튀는 신호라 **커널의 꼬리 두께**가
그룹핑에 영향을 줄 수 있다. 그걸 실제로 재본다.

  가우시안  exp(-d^2 / (2 s^2))   ← 꼬리가 얇다. 먼 값은 사실상 0으로 죽는다
  라플라시안 exp(-d   /  s   )    ← 꼬리가 두껍다. 먼 값도 덜 죽는다
  (sigma=1 기준 거리 5에서 라플라시안이 가우시안의 약 1,808배 — 계산 확인)

★설계 원칙
 1) `hsic_lead_grouping_demo.py`를 **고치지 않는다**. 그 파일은 2026-07-23 재현의
    검증된 산출물이고, 손대면 그 결과의 재현성이 흔들린다. 여기서 **import해 재사용**한다.
 2) ★**양성 대조 필수** — 이 스크립트의 가우시안 경로가 07-23 결과(k=2 · 흉부/사지)를
    그대로 재현해야 라플라시안 결과를 믿을 수 있다. 재현 실패면 즉시 중단한다.
    (「비교 실험은 한 번의 실행 안에서 조건만 전환하라」 — 조작변인이 커널 하나뿐임을 보장)
 3) ★**sigma 선택이 결론을 만들지 않게** 라플라시안을 두 가지 대역폭으로 돌린다.
    두 설정에서 결론이 같아야 "커널 때문"이라 말할 수 있다.
      - med   : sigma = median(|d|)                  (소박한 유사물)
      - match : sigma = 2 * sqrt(median(d^2))        (가우시안과 **같은 기준거리에서
                                                      같은 커널값 0.607**이 되도록 맞춤)
    이게 없으면 "가우시안엔 좋은 자, 라플라시안엔 나쁜 자"를 준 비교가 된다.

사용:
    .venv-hsic-demo/Scripts/python.exe scripts/hsic_kernel_comparison.py
"""
import io
import os
import sys

import numpy as np
from sklearn.cluster import SpectralClustering

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import hsic_lead_grouping_demo as m  # noqa: E402  (경로 삽입 후에 import해야 한다)

SEEDS = [42, 7, 123, 2024, 31337]      # hsic_stability_check.py와 동일 — 비교 가능하게
N_TARGET = m.N_HSIC_TARGET

# 07-23 재현 결과(FINDINGS.md). 양성 대조의 기대값이다.
EXPECTED_GAUSSIAN_K = 2
EXPECTED_CHEST = {"V1", "V2", "V3", "V4", "V5", "V6"}
EXPECTED_LIMB = {"I", "II", "III", "AVR", "AVL", "AVF"}


# ───────────────────────────────────────────────────────────── 커널 3종
def kernel_gaussian(x):
    """논문 사양 그대로 — median heuristic on squared pairwise distances."""
    bw_sq = m.median_heuristic_bandwidth(x)          # = median(d^2)
    diff = x[:, None] - x[None, :]
    return np.exp(-(diff ** 2) / (2.0 * bw_sq))


def kernel_laplacian_med(x):
    """sigma = median(|d|) — 절댓값 거리의 중앙값. 소박한 유사물."""
    diff = np.abs(x[:, None] - x[None, :])
    nz = diff[diff > 1e-12]
    sigma = max(float(np.median(nz)) if nz.size else 1.0, 1e-6)
    return np.exp(-diff / sigma)


def kernel_laplacian_matched(x):
    """sigma = 2*sqrt(median(d^2)) — 가우시안과 **같은 기준거리에서 같은 값**(0.607).

    가우시안은 d_ref = sqrt(median(d^2))에서 exp(-1/2)=0.6065를 준다.
    라플라시안이 같은 d_ref에서 0.6065가 되려면 exp(-d_ref/sigma)=exp(-0.5),
    즉 sigma = 2*d_ref. 이렇게 맞춰야 '자의 눈금'이 아니라 '꼬리 모양'만 비교된다.
    """
    bw_sq = m.median_heuristic_bandwidth(x)
    sigma = max(2.0 * np.sqrt(bw_sq), 1e-6)
    diff = np.abs(x[:, None] - x[None, :])
    return np.exp(-diff / sigma)


KERNELS = {
    "gaussian(논문)": kernel_gaussian,
    "laplacian-med": kernel_laplacian_med,
    "laplacian-match": kernel_laplacian_matched,
}


# ───────────────────────────────────────────────────────── 파이프라인 (커널만 갈아끼움)
def build_W(X_all, seed, kernel_fn):
    n = min(N_TARGET, X_all.shape[0])
    rng = np.random.default_rng(seed)
    idx = rng.choice(X_all.shape[0], size=n, replace=False)
    X = X_all[idx]
    D = X.shape[1]

    K_list, Kc_list = [], []
    for d in range(D):
        K = kernel_fn(X[:, d])
        K_list.append(K)
        Kc_list.append(m.double_center(K))

    W = np.zeros((D, D))
    for i in range(D):
        for j in range(D):
            W[i, j] = m.hsic_fast(Kc_list[i], K_list[j], n)
    W = (W + W.T) / 2
    W = np.clip(W, 0, None)
    return W


def pick_k(W):
    D = W.shape[0]
    deg = W.sum(axis=1)
    dis = np.diag(1.0 / np.sqrt(np.maximum(deg, 1e-12)))
    L = np.eye(D) - dis @ W @ dis
    ev = np.sort(np.linalg.eigvalsh(L))
    k_max = min(6, D - 1)
    gaps = np.diff(ev[: k_max + 1])
    k = int(np.argmax(gaps)) + 1
    return max(2, min(k, k_max)), ev


def cluster(W, k, seed):
    sc = SpectralClustering(n_clusters=k, affinity="precomputed", random_state=seed)
    return sc.fit_predict(W)


def groups_of(labels, lead_names):
    g = {}
    for lead, lab in zip(lead_names, labels):
        g.setdefault(int(lab), []).append(lead)
    return {cid: sorted(v) for cid, v in sorted(g.items())}


def same_group(labels, lead_names, a, b):
    idx = {ln: i for i, ln in enumerate(lead_names)}
    return labels[idx[a]] == labels[idx[b]]


def main():
    print("=" * 90)
    print("HSIC 리드그룹핑 — 커널 비교 (가우시안 vs 라플라시안)")
    print("=" * 90)
    X_all, lead_names, n_records = m.load_signals()
    print(f"레코드 {n_records}건 · 리드 {lead_names}")
    print(f"풀링 시점 {X_all.shape[0]} · HSIC 표본 n={min(N_TARGET, X_all.shape[0])} · seed {SEEDS}\n")

    # ── 양성 대조: 가우시안이 07-23 결과를 재현하는가 (실패면 중단)
    print("[양성 대조] 내 스크립트의 가우시안 경로가 2026-07-23 결과를 재현하나?")
    W0 = build_W(X_all, SEEDS[0], kernel_gaussian)
    k0, _ = pick_k(W0)
    g0 = groups_of(cluster(W0, k0, SEEDS[0]), lead_names)
    sets0 = [set(v) for v in g0.values()]
    ok = (k0 == EXPECTED_GAUSSIAN_K and EXPECTED_CHEST in sets0 and EXPECTED_LIMB in sets0)
    print(f"  eigengap k={k0} (기대 {EXPECTED_GAUSSIAN_K}) · 그룹 {g0}")
    if not ok:
        print("  ❌ 재현 실패 — 내 구현이 기존과 다르다. 라플라시안 결과를 믿을 수 없으므로 중단.")
        sys.exit(1)
    print("  ✅ 재현됨 (k=2 · 흉부/사지) — 조작변인은 커널 하나뿐임이 확보됐다\n")

    # ── 커널 3종 × seed 5개
    rows = []
    for kname, kfn in KERNELS.items():
        print(f"── {kname} " + "─" * (70 - len(kname)))
        for seed in SEEDS:
            W = build_W(X_all, seed, kfn)
            k, ev = pick_k(W)
            labels = cluster(W, k, seed)
            g = groups_of(labels, lead_names)
            chest_split = set(EXPECTED_CHEST) in [set(v) for v in g.values()]
            iii_avl_k3 = same_group(cluster(W, 3, seed), lead_names, "III", "AVL")
            iii_avl_k4 = same_group(cluster(W, 4, seed), lead_names, "III", "AVL")
            rows.append(dict(kernel=kname, seed=seed, k=k, groups=g,
                             chest_limb=chest_split, iii_avl_k3=iii_avl_k3,
                             iii_avl_k4=iii_avl_k4, ev=ev))
            print(f"  seed={seed:>6} k={k} 흉부/사지={'O' if chest_split else 'X'} "
                  f"III-aVL@k3={'O' if iii_avl_k3 else 'X'} @k4={'O' if iii_avl_k4 else 'X'}  {g}")
        print()

    # ── 요약
    print("=" * 90)
    print("요약 — 커널별 일관성 (seed 5개)")
    print("=" * 90)
    print(f"{'커널':<18}{'eigengap k':<14}{'흉부/사지 분할':<16}{'III-aVL@k3':<14}{'III-aVL@k4'}")
    for kname in KERNELS:
        r = [x for x in rows if x["kernel"] == kname]
        ks = sorted({x["k"] for x in r})
        cl = sum(x["chest_limb"] for x in r)
        a3 = sum(x["iii_avl_k3"] for x in r)
        a4 = sum(x["iii_avl_k4"] for x in r)
        print(f"{kname:<18}{str(ks):<14}{f'{cl}/5':<16}{f'{a3}/5':<14}{f'{a4}/5'}")

    print()
    gk = {x["k"] for x in rows if x["kernel"] == "gaussian(논문)"}
    lk = {x["k"] for x in rows if x["kernel"].startswith("laplacian")}
    gc = sum(x["chest_limb"] for x in rows if x["kernel"] == "gaussian(논문)")
    lc = sum(x["chest_limb"] for x in rows if x["kernel"].startswith("laplacian"))
    print(f"판정 재료: eigengap k — 가우시안 {sorted(gk)} vs 라플라시안 {sorted(lk)}")
    print(f"           흉부/사지 분할 — 가우시안 {gc}/5 · 라플라시안 {lc}/10 (두 대역폭 합)")
    print("\n⚠️ 해석은 사람이 한다. 이 스크립트는 '달라지는가/같은가'의 재료만 낸다.")
    print("⚠️ 라플라시안 두 대역폭에서 결론이 갈리면 그건 '커널 차이'가 아니라 'sigma 차이'다.")


if __name__ == "__main__":
    main()
