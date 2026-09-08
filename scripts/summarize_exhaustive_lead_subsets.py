"""Regenerate summaries and paired, record-level candidate intervals from predictions."""

import argparse
import csv
import io
import json
from pathlib import Path

import numpy as np
from ablation_exhaustive_lead_subsets import (
    CLASSES,
    LEAD_NAMES,
    MODELS,
    metrics,
    subsets,
)
from verify_lead_subset_assets import sha256


def electrode_count(indices):
    electrodes = set()
    limb = [{"RA", "LA"}, {"RA", "LL"}, {"LA", "LL"}]
    for i in indices:
        electrodes.update(limb[i] if i < 3 else {"RA", "LA", "LL"})
        if i >= 6:
            electrodes.add(LEAD_NAMES[i])
    return len(electrodes)


def indices(row):
    return tuple(map(int, row["lead_indices"].split(";")))


def validate(rows):
    expected = {(model, combo) for model in MODELS for combo in subsets()}
    actual = [(r["model"], indices(r)) for r in rows]
    if len(set(actual)) != len(actual) or set(actual) != expected:
        raise ValueError(
            "Incomplete/duplicate/invalid combinations: need 793 per model"
        )
    if len({r["run_fingerprint"] for r in rows}) != 1:
        raise ValueError("Mixed run identities")
    for row in rows:
        combo = indices(row)
        if int(row["n_leads"]) != len(combo):
            raise ValueError("Wrong lead count")
        if row["lead_names"] != ";".join(LEAD_NAMES[i] for i in combo):
            raise ValueError("Lead names and indices disagree")
        if int(row["n_samples"]) != MODELS[row["model"]][2]:
            raise ValueError("Wrong sample count")
        numeric = [k for k in row if k.startswith(("auroc_", "f1_", "sens"))]
        numeric += ["binary_auroc", "macro_f1", "macro_sensitivity"]
        for key in numeric:
            if row[key] and not np.isfinite(float(row[key])):
                raise ValueError(f"Nonfinite {key}")


def summary_bytes(rows):
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(
        [
            "model",
            "n_leads",
            "metric",
            "count",
            "best",
            "median",
            "worst",
            "mean",
            "variance",
            "top_leads",
        ]
    )
    columns = ["macro_f1", "macro_sensitivity", "binary_auroc", "sens_at_95sp"]
    columns += [
        f"{metric}_{c}"
        for c in CLASSES
        for metric in ("auroc", "f1", "sensitivity", "sens95")
    ]
    for model in MODELS:
        for n in range(1, 5):
            group = [r for r in rows if r["model"] == model and int(r["n_leads"]) == n]
            for key in columns:
                available = [r for r in group if r[key] != ""]
                if not available:
                    continue
                values = np.array([float(r[key]) for r in available])
                best = max(available, key=lambda r: float(r[key]))
                writer.writerow(
                    [
                        model,
                        n,
                        key,
                        len(values),
                        values.max(),
                        np.median(values),
                        values.min(),
                        values.mean(),
                        values.var(),
                        best["lead_names"],
                    ]
                )
    return output.getvalue().encode("utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default="results/exhaustive_lead_subsets_1to4.csv")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.bootstrap < 1000:
        raise ValueError("At least 1000 bootstrap replicates required")
    source = Path(args.csv)
    with source.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    validate(rows)
    root = source.parent
    summary_path = root / "exhaustive_lead_subsets_summary.csv"
    payload = summary_bytes(rows)
    summary_path.write_bytes(payload)
    # Fresh parse of the persisted raw file must reproduce the same summary bytes.
    with source.open(newline="", encoding="utf-8") as stream:
        if summary_bytes(list(csv.DictReader(stream))) != summary_path.read_bytes():
            raise RuntimeError("Summary regeneration mismatch")
    p1 = [r for r in rows if r["model"] == "P1_a07"]
    best = lambda group: max(group, key=lambda r: float(r["macro_f1"]))
    candidates = {
        "performance": best(p1),
        "minimum_electrodes": min(
            p1, key=lambda r: (electrode_count(indices(r)), -float(r["macro_f1"]))
        ),
        "chest_included": best([r for r in p1 if any(i >= 6 for i in indices(r))]),
        "limb_only": best([r for r in p1 if all(i < 6 for i in indices(r))]),
        "two_channels": best([r for r in p1 if int(r["n_leads"]) == 2]),
    }
    fingerprint = p1[0]["run_fingerprint"]
    pred_dir = Path("outputs/exhaustive_lead_subsets") / fingerprint
    for row in rows:
        saved = pred_dir / f"{row['model']}_{'-'.join(map(str, indices(row)))}.npz"
        if sha256(saved) != row["prediction_sha256"]:
            raise ValueError(f"Prediction hash mismatch: {saved}")

    def predictions(row):
        path = pred_dir / f"P1_a07_{'-'.join(map(str, indices(row)))}.npz"
        return np.load(path, allow_pickle=False)

    baseline = np.load(
        pred_dir / "P1_a07_0-1-2-3-4-5-6-7-8-9-10-11.npz", allow_pickle=False
    )
    if len(baseline["record_ids"]) != 936 or len(set(baseline["record_ids"])) != 936:
        raise ValueError("Baseline must contain exactly 936 unique records")
    baseline_metrics = metrics(
        baseline["labels_bin"],
        baseline["binary_probs"],
        baseline["labels_mc"],
        baseline["multiclass_probs"],
    )
    interval_keys = ["macro_f1", "macro_sensitivity"] + [
        f"{m}_{c}" for c in CLASSES[1:] for m in ("auroc", "sensitivity", "sens95")
    ]
    intervals = []
    for category, row in candidates.items():
        pred = predictions(row)
        for key in ("record_ids", "labels_bin", "labels_mc"):
            if not np.array_equal(pred[key], baseline[key]):
                raise ValueError(f"Unpaired records or labels: {category}")
        observed = metrics(
            pred["labels_bin"],
            pred["binary_probs"],
            pred["labels_mc"],
            pred["multiclass_probs"],
        )
        for key in interval_keys:
            if abs(observed[key] - float(row[key])) > 1e-7:
                raise ValueError(
                    f"Raw metric and predictions disagree: {category}/{key}"
                )
        rng = np.random.default_rng(args.seed)
        differences = {key: [] for key in interval_keys}
        skipped = 0
        for _ in range(args.bootstrap):
            sample = rng.integers(0, len(pred["record_ids"]), len(pred["record_ids"]))
            if len(np.unique(pred["labels_mc"][sample])) != 5:
                skipped += 1
                continue
            results = []
            for data in (pred, baseline):
                results.append(
                    metrics(
                        data["labels_bin"][sample],
                        data["binary_probs"][sample],
                        data["labels_mc"][sample],
                        data["multiclass_probs"][sample],
                    )
                )
            for key in interval_keys:
                differences[key].append(results[0][key] - results[1][key])
        if skipped == args.bootstrap:
            raise ValueError("All bootstrap replicates invalid")
        for key, values in differences.items():
            lo, hi = np.percentile(values, [2.5, 97.5])
            intervals.append(
                {
                    "category": category,
                    "leads": row["lead_names"],
                    "metric": key,
                    "difference_vs_12": float(row[key]) - baseline_metrics[key],
                    "ci_low": lo,
                    "ci_high": hi,
                    "valid": len(values),
                    "skipped": skipped,
                }
            )
    ci_path = root / "exhaustive_lead_subsets_paired_ci.csv"
    with ci_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=intervals[0].keys())
        writer.writeheader()
        writer.writerows(intervals)
    failure_logs = list((root / "exhaustive_lead_subsets_failures").glob("*.json"))
    run_status = json.loads(
        (root / "exhaustive_lead_subsets_run_status.json").read_text(encoding="utf-8")
    )
    if run_status["run_fingerprint"] != fingerprint:
        raise ValueError("Run status does not match the CSV")
    lines = [
        "# ECG 1–4리드 전수평가",
        "",
        "상태: 미검증 — 이전 중단 로그의 해소 여부 확인 필요."
        if failure_logs
        else "상태: 조합 수·유효값·요약 재생성 검증 완료.",
        f"실행 중단 로그 {len(failure_logs)}건; 완주 실행 실패 {run_status['failures']}건; 재개 스킵 {run_status['resumed_skips']}건; CSV NaN/Inf 0건.",
        "793개/모델. 응급 이진 참고모델과 P1 멀티태스크 결과를 구분한다.",
        f"원시 CSV SHA-256: `{sha256(source)}`",
        "",
        "현재 10초 ECG의 탐색적 분류 평가다. 조기예측·직물 전극 품질·독립 검증 성능이 아니다.",
        "후보 선택과 CI가 같은 test에 기반하므로 선택 편향과 다중비교가 존재한다.",
        "",
        "## 후보",
        "",
        "| 유형 | 리드 | Macro-F1 | 측정 전극 수 하한 |",
        "|---|---|---:|---:|",
    ]
    for category, row in candidates.items():
        lines.append(
            f"| {category} | {row['lead_names']} | {float(row['macro_f1']):.6f} | {electrode_count(indices(row))} |"
        )
    lines += [
        "",
        "전극 수는 표준 유도 계산에 필요한 전극의 하한이다. 접지/DRL·장치 제약은 별도다.",
        "사지 파생유도를 독립 측정 채널로 중복 계산하지 않는다. 최소 전극 후보는 성능 동등성을 뜻하지 않는다.",
    ]
    for key in ["macro_f1", "macro_sensitivity"] + [f"sens95_{c}" for c in CLASSES[1:]]:
        top = sorted(p1, key=lambda r: float(r[key]), reverse=True)[:10]
        lines += ["", f"## {key} top 10", "", "| 리드 | 값 |", "|---|---:|"]
        lines.extend(f"| {r['lead_names']} | {float(r[key]):.6f} |" for r in top)
    for key in interval_keys:
        best3 = max(float(r[key]) for r in p1 if int(r["n_leads"]) == 3)
        best4 = max(float(r[key]) for r in p1 if int(r["n_leads"]) == 4)
        lines.append(
            f"\n3→4리드 {key}의 각 N별 최댓값 차이: {best4 - best3:+.6f} (동일 후보에 리드를 추가한 효과와 구분)."
        )
    objective = np.array(
        [
            [float(r[f"sens95_{c}"]) for c in CLASSES[1:]] + [-int(r["n_leads"])]
            for r in p1
        ]
    )
    pareto = [
        i
        for i, point in enumerate(objective)
        if not np.any(
            np.all(objective >= point, axis=1) & np.any(objective > point, axis=1)
        )
    ]
    lines += [
        "",
        "## 질환별 Sens@95Sp·리드 수 Pareto 후보",
        "",
        ", ".join(p1[i]["lead_names"] for i in pareto),
        "",
        "N별 최고·중앙·최저·평균·분산: exhaustive_lead_subsets_summary.csv.",
        "최종 후보와 12리드 paired record bootstrap: exhaustive_lead_subsets_paired_ci.csv.",
        "완주한 CSV에는 실패 행·NaN/Inf 없음. 실행 중단 기록은 exhaustive_lead_subsets_failures/에서 별도 감사한다.",
    ]
    (root / "exhaustive_lead_subsets_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "rows": len(rows),
                "summary_sha256": sha256(summary_path),
                "paired_ci_rows": len(intervals),
            }
        )
    )


if __name__ == "__main__":
    main()
