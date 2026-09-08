"""Validate candidate stability summaries and render the bounded interpretation."""

import csv
import json
from pathlib import Path

import numpy as np
from summarize_exhaustive_lead_subsets import (
    electrode_count,
    indices,
    select_candidates,
    summary_bytes,
    validate,
)
from verify_lead_subset_assets import sha256


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    root = Path("results")
    raw_path = root / "exhaustive_lead_subsets_1to4.csv"
    rows = read_csv(raw_path)
    validate(rows)
    assert (
        summary_bytes(rows)
        == (root / "exhaustive_lead_subsets_summary.csv").read_bytes()
    )
    candidates = select_candidates(rows)
    baseline = next(
        r
        for r in read_csv(root / "exhaustive_lead_subsets_controls.csv")
        if r["model"] == "P1_a07" and r["n_leads"] == "12"
    )
    ci = read_csv(root / "exhaustive_lead_subsets_paired_ci.csv")
    repeated = read_csv(root / "exhaustive_lead_candidates_stability_raw.csv")
    stable = read_csv(root / "exhaustive_lead_candidates_stability_summary.csv")
    metadata = json.loads(
        (root / "exhaustive_lead_candidates_stability.json").read_text()
    )
    independent = json.loads(
        (root / "exhaustive_lead_subsets_verification.json").read_text()
    )
    assert (
        metadata["run_fingerprint"]
        == independent["run_fingerprint"]
        == rows[0]["run_fingerprint"]
    )
    assert (
        metadata["raw_search_sha256"] == independent["raw_sha256"] == sha256(raw_path)
    )
    assert metadata["failures"] == metadata["skips"] == 0
    assert metadata["script_sha256"] == sha256(
        Path("scripts/evaluate_lead_candidate_stability.py")
    )
    assert len(ci) == 70 and all(
        int(r["valid"]) == 2000 and int(r["skipped"]) == 0 for r in ci
    )
    assert len(stable) == 90
    seeds = list(range(20000, 20010))
    assert metadata["seeds_predeclared"] == seeds
    choices = {r["lead_names"] for r in candidates.values()} | {baseline["lead_names"]}
    lookup = {(r["leads"], int(r["seed"])): r for r in repeated}
    assert (
        len(lookup)
        == len(repeated)
        == len(choices) * len(seeds)
        == metadata["realizations"]
    )
    assert set(lookup) == {(lead, seed) for lead in choices for seed in seeds}
    largest_error = 0.0
    for row in stable:
        assert row["leads"] == candidates[row["category"]]["lead_names"]
        values = np.array(
            [float(lookup[(row["leads"], seed)][row["metric"]]) for seed in seeds]
        )
        reference = np.array(
            [
                float(lookup[(baseline["lead_names"], seed)][row["metric"]])
                for seed in seeds
            ]
        )
        delta = values - reference
        expected = {
            "mean": values.mean(),
            "std": values.std(ddof=1),
            "min": values.min(),
            "max": values.max(),
            "paired_mean_delta_vs12": delta.mean(),
            "paired_min_delta_vs12": delta.min(),
            "paired_max_delta_vs12": delta.max(),
            "fraction_strictly_better_vs12": (delta > 0).mean(),
        }
        for key, value in expected.items():
            error = abs(float(row[key]) - value)
            assert np.isfinite(error) and error < 1e-12
            largest_error = max(largest_error, error)
    lines = [
        "# ECG 리드 전수평가 — 결과 해석",
        "",
        "동결 P1 모델의 동일 CPSC 5-class test 936레코드에서 1–4리드 793조합을 평가했다. 참고 이진 모델은 별도의 역사적 binary test 474레코드에서 793조합을 평가했다. 모델 간 수치는 합치지 않는다.",
        "",
        "## 후보 비교",
        "",
        "아래 후보는 전체 탐색의 고정 seed 42 Macro-F1을 기준으로 각 제약 안에서 선정했다. Macro-F1과 macro sensitivity는 NSR 포함 5개 class 평균이다.",
        "",
        "| 후보 유형 | 리드 | Macro-F1 | Macro sensitivity | 전극 하한 |",
        "|---|---|---:|---:|---:|",
        f"| 기준 | 12리드 | {float(baseline['macro_f1']):.6f} | {float(baseline['macro_sensitivity']):.6f} | 9 |",
    ]
    for category, row in candidates.items():
        lines.append(
            f"| {category} | {row['lead_names']} | {float(row['macro_f1']):.6f} | {float(row['macro_sensitivity']):.6f} | {electrode_count(indices(row))} |"
        )
    lines += [
        "",
        "전극 하한은 접지/DRL을 제외한다. aVR에는 RA·LA·LL이 필요하므로 aVR 단독을 물리 전극 2개로 구현할 수 있다는 뜻이 아니다. 서로 다른 후보 유형이 같은 조합을 선택할 수 있다.",
        "",
        "## 질환별 성능",
        "",
        "| 후보 | 질환 | AUROC | F1 | Argmax sensitivity | Sens@95Sp |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for name, row in [("12리드", baseline)] + [
        (r["lead_names"], r)
        for r in {r["lead_names"]: r for r in candidates.values()}.values()
    ]:
        for disease in ("af", "ischemia", "conduction", "ectopy"):
            lines.append(
                f"| {name} | {disease} | {float(row['auroc_' + disease]):.6f} | {float(row['f1_' + disease]):.6f} | {float(row['sensitivity_' + disease]):.6f} | {float(row['sens95_' + disease]):.6f} |"
            )
    lines += [
        "",
        "## 12리드 대비 차이와 난수 안정성",
        "",
        "| 후보 유형 | Macro-F1 차이, seed42 | Paired bootstrap 95% CI | 10 seeds 평균 차이 | 12리드보다 높은 횟수 |",
        "|---|---:|---|---:|---:|",
    ]
    for category in candidates:
        interval = next(
            r for r in ci if r["category"] == category and r["metric"] == "macro_f1"
        )
        repeat = next(
            r for r in stable if r["category"] == category and r["metric"] == "macro_f1"
        )
        wins = round(float(repeat["fraction_strictly_better_vs12"]) * 10)
        lines.append(
            f"| {category} | {float(interval['difference_vs_12']):+.6f} | [{float(interval['ci_low']):+.6f}, {float(interval['ci_high']):+.6f}] | {float(repeat['paired_mean_delta_vs12']):+.6f} | {wins}/10 |"
        )
    lines += [
        "",
        "CI는 같은 레코드를 함께 재표집한 2,000회 percentile 구간이며 mask seed42에 조건부다. 10개 seed는 별도의 feature-mask 변동 진단이다. 둘 다 후보 선택에 사용한 test를 재사용하므로 외부 검증·환자군 독립 검증이나 다중비교 보정 결과가 아니다.",
        "",
        "## 판정 범위",
        "",
        "단일 최고점만으로 최종 하드웨어를 확정하지 않는다. 전수평가 상위 후보와 질환별 recall/Sens@95Sp 간 절충을 제시한다. 특히 이소성 recall을 전체 Macro-F1로 가리지 않는다.",
        "",
        "기존 단일 seed 역사값 대조는 실패 상태를 보존한다. 원 코드 8조건 일치·자산 해시·홈/랩 소규모 forward를 확인하고, 별도 사전 고정 39 seeds × 5조건의 17지표 stochastic reproduction을 통과한 후 전수평가했다. 정확한 과거 난수 실현의 복원과 동등한 의미가 아니다.",
        "",
        "현재 10초 ECG의 분류 탐색이며 조기예측, 의류형 전극 신호품질, 독립 임상 성능, 모델 학습법의 최적성을 입증하지 않는다. 원자료와 이전 사전 대조 실패 이력은 보존한다.",
        "",
        "전체 분포·질환별 top10·Pareto·3→4리드 차이는 exhaustive_lead_subsets_report.md, 모든 paired 차이는 exhaustive_lead_subsets_paired_ci.csv에 있다.",
    ]
    (root / "exhaustive_lead_subsets_interpretation.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    evidence = {
        "status": "passed",
        "stability_summary_rows": len(stable),
        "stability_realizations": len(repeated),
        "maximum_summary_recalculation_error": largest_error,
        "paired_ci_rows": len(ci),
        "bootstrap_per_category": 2000,
        "bootstrap_skips": 0,
        "summary_reproduced_byte_identically": True,
        "source_sha256": sha256(raw_path),
        "script_sha256": sha256(Path(__file__)),
    }
    (root / "exhaustive_lead_subsets_analysis_verification.json").write_text(
        json.dumps(evidence, indent=2), encoding="utf-8"
    )
    print(json.dumps(evidence))


if __name__ == "__main__":
    main()
