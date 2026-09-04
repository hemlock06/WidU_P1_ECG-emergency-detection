#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""k 스윕 — 「k가 커질수록 Lead I이 독자 그룹으로 분리된다」는 주장을 실측한다. (2026-08-02)

    .venv-hsic-demo/Scripts/python.exe scripts/hsic_lead_k_sweep.py

## 왜 이 스크립트가 있나
발표덱 **부록 A3(45쪽)** 에 다음 문장이 있다:

    "그리고 k가 커질수록 Lead I이 독자 그룹으로 분리되는 것도 항등식과 관련 있어 보인다 (제 판단)"

그런데 `hsic_lead_grouping_demo.py` 는 **eigengap이 고른 k=2 하나만** 돌린다.
즉 위 문장은 **재본 적이 없는 주장**이었다. 교신저자 본인 앞 발표(2026-08-03)에
근거 없는 관찰을 올릴 수 없으므로 여기서 k=2..6 을 전수로 돌려 확인한다.

## 무엇을 재는가
- W(HSIC affinity)는 `hsic_lead_grouping_demo.py` 와 **비트 단위로 같은 절차**로 만든다
  (같은 SUBSET_RECORDS · N_HSIC=3000 · seed=42 · median heuristic).
  → 그래야 45쪽의 k=2 결과(흉부/사지 분할)와 같은 계보의 실험이 된다.
- 그 W를 고정한 채 **k만** 2→6으로 바꿔 spectral clustering을 돌린다.
- 각 k에서 **Lead I이 혼자 있는 그룹인지**(singleton) 를 판정한다.

## ★ 시드 강건성도 같이 잰다
spectral clustering은 `random_state` 에 따라 결과가 흔들릴 수 있다.
시드 하나에서만 나오는 현상을 "관찰됐다"고 적으면 그건 관찰이 아니라 우연이다.
→ 5개 시드에서 돌려 **몇 개 시드에서 그런지**를 분모와 함께 보고한다.
   (준거: 「검증 부재 ≠ 검증 통과」의 분모 버전 — 비율은 분모를 밝혀야 뜻이 있다)

## 해석 주의
이 스크립트는 **"갈리는가/안 갈리는가"의 재료만** 낸다.
Einthoven 항등식(II = I + III)과의 인과는 이 실험이 답하지 못한다 — 상관 관찰일 뿐이다.
"""
import os
import sys

import numpy as np
import wfdb
from sklearn.cluster import SpectralClustering

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hsic_lead_grouping_demo as demo  # noqa: E402  — W 생성 절차를 그대로 재사용한다

SEED_W = 42          # W(표본 추출)용 — demo 와 동일해야 같은 W가 나온다
K_RANGE = range(2, 7)
CLUSTER_SEEDS = [42, 0, 1, 7, 2026]


def build_W():
    """demo 와 동일 절차로 HSIC affinity 행렬을 만든다."""
    X_all, lead_names, n_rec = demo.load_signals()
    rng = np.random.default_rng(SEED_W)
    n_avail = X_all.shape[0]
    n = min(demo.N_HSIC_TARGET, n_avail)
    idx = rng.choice(n_avail, size=n, replace=False)
    X = X_all[idx]
    D = X.shape[1]

    Kc = []
    for d in range(D):
        bw_sq = demo.median_heuristic_bandwidth(X[:, d])
        K = demo.rbf_kernel_1d(X[:, d], bw_sq)
        Kc.append((K, demo.double_center(K)))

    W = np.zeros((D, D))
    for a in range(D):
        for b in range(D):
            W[a, b] = demo.hsic_fast(Kc[a][1], Kc[b][0], n)
    W = (W + W.T) / 2.0
    return W, lead_names, n_rec, n


def groups_of(labels, lead_names):
    out = {}
    for lab, name in zip(labels, lead_names):
        out.setdefault(int(lab), []).append(name)
    return [sorted(v) for _, v in sorted(out.items())]


def main():
    print("=" * 92)
    print("k 스윕 — 「k가 커질수록 Lead I이 독자 그룹으로 분리」 실측")
    print("=" * 92)
    W, leads, n_rec, n = build_W()
    print(f"레코드 {n_rec}건 · HSIC 표본 n={n} · W seed={SEED_W} · 리드순서 {leads}\n")

    lead1 = leads.index("I")
    summary = {}

    # ★쌍 안정성도 같이 센다 — 「어떤 리드끼리 늘 붙어 있나」가 항등식과 잇는 진짜 근거다.
    #   (초판은 첫 시드의 그룹만 출력해서, 다른 시드에서도 그런지 알 수 없었다. 실제로 그
    #    출력만 보고 "전 시드 유지"라고 잘못 말했다 — 그래서 전 시드를 다 찍는다.)
    PAIRS = [("AVL", "III"), ("AVR", "I"), ("AVF", "II"), ("II", "III"), ("I", "II")]
    pair_hits = {p: 0 for p in PAIRS}
    pair_den = 0

    for k in K_RANGE:
        print(f"── k = {k} " + "─" * 74)
        singleton_hits = 0
        for cs in CLUSTER_SEEDS:
            sc = SpectralClustering(n_clusters=k, affinity="precomputed", random_state=cs)
            labels = sc.fit_predict(W)
            gs = groups_of(labels, leads)
            my = [g for g in gs if "I" in g][0]
            solo = (len(my) == 1)
            singleton_hits += solo
            print(f"      seed {cs:>4} : {gs}{'   ★Lead I 단독' if solo else ''}")
            if k >= 3:                       # k=2 는 사지/흉부 2분할이라 쌍 판정이 무의미
                pair_den += 1
                for a, b in PAIRS:
                    if any(a in g and b in g for g in gs):
                        pair_hits[(a, b)] += 1
        summary[k] = singleton_hits
        print(f"   ⇒ Lead I 단독 그룹: {singleton_hits}/{len(CLUSTER_SEEDS)} 시드\n")

    print("─" * 92)
    print(f"쌍 안정성 — 두 리드가 같은 그룹에 있었던 횟수 (k=3~6 x 시드 {len(CLUSTER_SEEDS)}개 = 분모 {pair_den})")
    for (a, b), h in sorted(pair_hits.items(), key=lambda kv: -kv[1]):
        print(f"   {a:>4} - {b:<4} : {h:>2}/{pair_den}")
    print()

    print("=" * 92)
    print("요약 — k별 「Lead I 단독」 시드 수 (분모 = %d)" % len(CLUSTER_SEEDS))
    for k, hits in summary.items():
        bar = "■" * hits + "□" * (len(CLUSTER_SEEDS) - hits)
        print(f"   k={k} : {hits}/{len(CLUSTER_SEEDS)}  {bar}")
    print("=" * 92)

    mono = all(summary[k] <= summary[k + 1] for k in list(K_RANGE)[:-1])
    any_solo = any(v > 0 for v in summary.values())
    if not any_solo:
        print("판정: ❌ 어느 k에서도 Lead I이 단독으로 갈리지 않았다.")
        print("      → 45쪽의 「k가 커질수록 Lead I이 독자 그룹으로 분리」는 이 실험이 지지하지 않는다. 문장을 내려라.")
    elif mono:
        print("판정: ✅ k가 커질수록 단독 분리가 늘어난다 (단조). 45쪽 문장이 지지된다.")
    else:
        print("판정: ⚠️ 단독 분리가 나타나긴 하나 k에 대해 단조가 아니다.")
        print("      → 「k가 커질수록」이라는 표현은 과장이다. 어느 k에서 그런지를 명시해 적어라.")
    print("★ 이 실험은 Einthoven 항등식과의 **인과**를 보이지 않는다 — 동시발생 관찰일 뿐이다.")


if __name__ == "__main__":
    main()
