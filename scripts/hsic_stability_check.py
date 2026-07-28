"""
HSIC 리드그룹핑 안정성 검증 — seed를 바꿔도 같은 결론이 나오나?
=================================================================
왜 필요한가: hsic_lead_grouping_demo.py는 seed=42 단 1회 결과다. 그 결과를
발표에서 주장(특히 "III-aVL이 계속 같이 묶인다")으로 쓰려면, 그게 **표본 추출
운**이 아니라 데이터의 성질인지 확인해야 한다.
준거: CLAUDE.md 「내가 만든 증거에도 소스 적합성 체크」 — 내가 잰 것과
내가 말하려는 것이 같은 방향인지.

검증 대상 주장 3개:
  C1. eigengap이 고르는 k가 2로 일관되는가
  C2. k=2 분할이 '흉부(V1-V6) vs 사지(I,II,III,aVR,aVL,aVF)'로 일관되는가
  C3. k를 3~6으로 강제했을 때 III-aVL이 같은 그룹에 계속 묶이는가
"""
import os
import sys

import numpy as np
from sklearn.cluster import SpectralClustering

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hsic_lead_grouping_demo as m  # noqa: E402

SEEDS = [42, 7, 123, 2024, 31337]


def build_W(X_all, seed, n_target=m.N_HSIC_TARGET):
    n = min(n_target, X_all.shape[0])
    rng = np.random.default_rng(seed)
    idx = rng.choice(X_all.shape[0], size=n, replace=False)
    X = X_all[idx]
    D = X.shape[1]
    K_list, Kc_list = [], []
    for d in range(D):
        bw = m.median_heuristic_bandwidth(X[:, d])
        K = m.rbf_kernel_1d(X[:, d], bw)
        K_list.append(K)
        Kc_list.append(m.double_center(K))
    W = np.zeros((D, D))
    for i in range(D):
        for j in range(D):
            W[i, j] = m.hsic_fast(Kc_list[i], K_list[j], n)
    W = (W + W.T) / 2
    return np.clip(W, 0, None)


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
    out = {}
    for lead, lab in zip(lead_names, labels):
        out.setdefault(int(lab), []).append(lead)
    return {cid: sorted(v) for cid, v in sorted(out.items())}


def same_group(labels, lead_names, a, b):
    idx = {ln: i for i, ln in enumerate(lead_names)}
    return labels[idx[a]] == labels[idx[b]]


def main():
    print("신호 로드...")
    X_all, lead_names, n_records = m.load_signals()
    print(f"  레코드 {n_records}건 · 전체 시점 {X_all.shape[0]}\n")

    CHEST = {"V1", "V2", "V3", "V4", "V5", "V6"}
    LIMB = {"I", "II", "III", "AVR", "AVL", "AVF"}

    c1_ks, c2_hits, c3 = [], [], {3: [], 4: [], 5: [], 6: []}

    for seed in SEEDS:
        W = build_W(X_all, seed)
        k, ev = pick_k(W)
        c1_ks.append(k)

        labels = cluster(W, k, seed)
        g = groups_of(labels, lead_names)
        # C2: k=2일 때만 판정 가능
        if k == 2:
            sets = [set(v) for v in g.values()]
            c2 = (sets[0] == CHEST and sets[1] == LIMB) or (sets[0] == LIMB and sets[1] == CHEST)
        else:
            c2 = None
        c2_hits.append(c2)

        print(f"[seed={seed}] eigengap k={k}")
        print(f"  분할: {g}")
        print(f"  C2(흉부/사지 정확분할): {c2}")

        for kf in [3, 4, 5, 6]:
            lab_f = cluster(W, kf, seed)
            together = same_group(lab_f, lead_names, "III", "AVL")
            c3[kf].append(together)
        print(f"  C3(III·AVL 동일그룹) k=3..6: {[c3[kf][-1] for kf in [3,4,5,6]]}\n")

    print("=" * 60)
    print("판정")
    print("=" * 60)
    print(f"C1 eigengap k 값들: {c1_ks} → {'★일관' if len(set(c1_ks)) == 1 else '⚠️ 불안정'}")
    measurable = [c for c in c2_hits if c is not None]
    if measurable:
        print(f"C2 흉부/사지 정확분할: {sum(measurable)}/{len(measurable)} "
              f"(k!=2라 측정불가 {len(c2_hits)-len(measurable)}건 제외)")
    else:
        print("C2: 측정 가능한 표본 0건(모든 seed에서 k!=2)")
    for kf in [3, 4, 5, 6]:
        hits = sum(c3[kf])
        mark = "★일관" if hits == len(SEEDS) else ("⚠️ 부분" if hits else "✗ 기각")
        print(f"C3 III·AVL 동일그룹 @k={kf}: {hits}/{len(SEEDS)} {mark}")


if __name__ == "__main__":
    main()
