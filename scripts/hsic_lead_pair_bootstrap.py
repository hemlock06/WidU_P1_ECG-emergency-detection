#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""W 부트스트랩 — 「증폭유도가 자기 원본과 묶인다」가 **데이터 표집**에도 강건한가. (2026-08-02)

    .venv-hsic-demo/Scripts/python.exe scripts/hsic_lead_pair_bootstrap.py

## 왜 이게 필요한가 (앞 실험의 구멍)
`hsic_lead_k_sweep.py` 는 aVF-II 20/20 · aVL-III 19/20 을 냈지만, 그 분모 20은
**클러스터링 시드만** 흔든 것이다. W(HSIC 행렬)는 80개 레코드를 한 번 풀링해 만든 **하나**였다.
즉 그 결과는 "이 W에서는" 이라는 조건 아래에서만 참이고,
**데이터 표집이 바뀌어도 그런지는 재본 적이 없다.**
(준거: 「내가 만든 증거에도 소스 적합성 체크」 — 내가 잰 것과 내가 말하려는 것이 다르면 안 된다)

## 설계 — 조작변인은 표집 하나
- 80개 레코드를 **서로 겹치지 않는 B개 조각**으로 나눈다(부트스트랩 재표집이 아니라 분할 —
  겹치면 조각끼리 독립이 아니라 분모가 부풀려진다).
- 조각마다 **독립적으로 W를 새로 만든다**(같은 절차·같은 n).
- 클러스터링 시드는 **고정 대조**로 2개만 쓴다(42, 0). 시드 요인은 앞 실험이 이미 쟀다.
- 그래야 조각 간 차이가 **표집 하나에서만** 온다.

## 무엇을 보나
1. 쌍 동거율 — 분모 = 조각 B × k(3~6) × 시드 2
2. 조각마다 eigengap이 고르는 k — k=2(흉부/사지)가 표집에 강건한가
3. ★양성/음성 대조 — 항등식 있는 쌍 vs 없는 쌍이 갈리는가

## 해석 주의
동거율이 높아도 **인과가 아니다.** 항등식이 원인인지 보려면 계수를 바꿔 심고
그룹이 예측 방향으로 따라오는지 봐야 한다(다음 실험).
"""
import os
import sys

import numpy as np
import wfdb
from sklearn.cluster import SpectralClustering

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hsic_lead_grouping_demo as demo  # noqa: E402

N_SPLIT = 8
CLUSTER_SEEDS = [42, 0]
K_RANGE = range(3, 7)
SPLIT_SEED = 20260802

# 항등식이 있는 쌍(양성 대조) / 없는 쌍(음성 대조)
POS = [("AVF", "II"), ("AVL", "III"), ("AVR", "I")]      # 증폭유도 ~ 정의에 든 원본
NEG = [("II", "III"), ("I", "II"), ("V1", "V6"), ("I", "V3"), ("AVR", "V1")]


def load_per_record():
    with open(os.path.join(demo.DATA_DIR, "SUBSET_RECORDS.txt")) as f:
        recs = [l.strip() for l in f if l.strip()]
    sigs, leads = [], None
    for rec in recs:
        p = os.path.join(demo.DATA_DIR, rec.replace(".hea", "").replace(".dat", ""))
        r = wfdb.rdrecord(p)
        if leads is None:
            leads = [s.upper() for s in r.sig_name]
        sigs.append(r.p_signal.astype(np.float64))
    return sigs, leads


def build_W(X, n_target, seed):
    rng = np.random.default_rng(seed)
    n = min(n_target, X.shape[0])
    X = X[rng.choice(X.shape[0], size=n, replace=False)]
    D = X.shape[1]
    Kc = []
    for d in range(D):
        K = demo.rbf_kernel_1d(X[:, d], demo.median_heuristic_bandwidth(X[:, d]))
        Kc.append((K, demo.double_center(K)))
    W = np.zeros((D, D))
    for a in range(D):
        for b in range(D):
            W[a, b] = demo.hsic_fast(Kc[a][1], Kc[b][0], n)
    return (W + W.T) / 2.0


def eigengap_k(W, k_max=6):
    d = W.sum(1)
    L = np.eye(len(W)) - (W / np.sqrt(np.outer(d, d)))
    ev = np.sort(np.linalg.eigvalsh(L))[:k_max + 1]
    k = int(np.argmax(np.diff(ev))) + 1
    return max(2, min(k, k_max)), ev


def groups_of(labels, leads):
    out = {}
    for lab, nm in zip(labels, leads):
        out.setdefault(int(lab), []).append(nm)
    return [sorted(v) for _, v in sorted(out.items())]


def main():
    print("=" * 96)
    print("W 부트스트랩 — 표집을 바꿔도 「증폭유도-원본유도」 쌍이 유지되는가")
    print("=" * 96)
    sigs, leads = load_per_record()
    print(f"레코드 {len(sigs)}건 · 리드 {leads}")

    rng = np.random.default_rng(SPLIT_SEED)
    order = rng.permutation(len(sigs))
    chunks = np.array_split(order, N_SPLIT)
    print(f"→ 겹치지 않는 {N_SPLIT}조각 (조각당 레코드 {[len(c) for c in chunks]})\n")

    hits = {p: 0 for p in POS + NEG}
    den = 0
    k2_hits = 0

    for ci, ch in enumerate(chunks):
        X = np.concatenate([sigs[i] for i in ch], axis=0)
        n_t = min(demo.N_HSIC_TARGET, X.shape[0])
        W = build_W(X, n_t, seed=42)
        k_pick, ev = eigengap_k(W)
        grp2 = groups_of(SpectralClustering(n_clusters=2, affinity="precomputed",
                                            random_state=42).fit_predict(W), leads)
        limb = {"I", "II", "III", "AVR", "AVL", "AVF"}
        clean2 = any(set(g) == limb for g in grp2)
        k2_hits += clean2
        print(f"── 조각 {ci+1}/{N_SPLIT}  (레코드 {len(ch)}건 · n={n_t}) "
              f"· eigengap k={k_pick} · k=2에서 사지/흉부 정확분할: {'O' if clean2 else 'X'}")

        for k in K_RANGE:
            for cs in CLUSTER_SEEDS:
                gs = groups_of(SpectralClustering(n_clusters=k, affinity="precomputed",
                                                  random_state=cs).fit_predict(W), leads)
                den += 1
                for pair in POS + NEG:
                    a, b = pair
                    if any(a in g and b in g for g in gs):
                        hits[pair] += 1
        ex = groups_of(SpectralClustering(n_clusters=4, affinity="precomputed",
                                          random_state=42).fit_predict(W), leads)
        print(f"      (k=4 예시) {ex}")

    print("\n" + "=" * 96)
    print(f"쌍 동거율 — 분모 = 조각 {N_SPLIT} × k(3~6) × 시드 {len(CLUSTER_SEEDS)} = {den}")
    print("-" * 96)
    print("  [양성 대조] 항등식에 그 원본이 들어가는 쌍")
    for p in POS:
        h = hits[p]
        print(f"     {p[0]:>4} - {p[1]:<4} : {h:>3}/{den}  ({100*h/den:5.1f}%)  "
              + "■" * int(round(20 * h / den)))
    print("  [음성 대조] 그런 관계가 없는 쌍")
    for p in NEG:
        h = hits[p]
        print(f"     {p[0]:>4} - {p[1]:<4} : {h:>3}/{den}  ({100*h/den:5.1f}%)  "
              + "■" * int(round(20 * h / den)))
    print("=" * 96)
    print(f"k=2 에서 사지/흉부 정확분할: {k2_hits}/{N_SPLIT} 조각")

    pos_min = min(hits[p] for p in POS) / den
    neg_max = max(hits[p] for p in NEG) / den
    print("-" * 96)
    if pos_min > neg_max:
        print(f"판정: ✅ 양성 최저({100*pos_min:.1f}%) > 음성 최고({100*neg_max:.1f}%) — "
              "표집을 바꿔도 두 부류가 갈린다.")
    else:
        print(f"판정: ❌ 양성 최저({100*pos_min:.1f}%) ≤ 음성 최고({100*neg_max:.1f}%) — "
              "표집을 바꾸면 구분이 무너진다. 45쪽 문장을 다시 봐라.")
    print("★ 동거율이 높아도 인과는 아니다 — 계수를 바꿔 심는 실험이 남아 있다.")


if __name__ == "__main__":
    main()
