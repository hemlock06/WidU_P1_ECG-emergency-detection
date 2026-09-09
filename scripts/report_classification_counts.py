"""Report actual correct/missed/false-positive classifications from frozen predictions."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/classification_counts_20260910"
CLASSES = ["NSR", "AF", "ST_change_ischemia_surrogate", "conduction", "ectopy"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(exist_ok=True)
    labels_path = ROOT / "data/processed/cpsc2018_mc/test/labels.npy"
    ids_path = ROOT / "data/processed/cpsc2018_mc/test/record_ids.npy"
    labels = np.load(labels_path)
    ids = np.load(ids_path, allow_pickle=True).astype(str)
    cohort, disease, sources = [], [], []
    for stage, filename, expected in (
        ("seed42", "raw.csv", 4095),
        ("repeat", "stability_raw.csv", 80),
    ):
        path = ROOT / "results/electrode_coverage_v1" / filename
        sources.append({"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path)})
        rows = list(csv.DictReader(path.open(encoding="utf-8")))
        if len(rows) != expected:
            raise ValueError("Unexpected number of saved runs")
        for row in rows:
            prediction = ROOT / row["prediction_path"]
            if sha(prediction) != row["prediction_sha256"]:
                raise ValueError("Saved prediction hash mismatch")
            with np.load(prediction, allow_pickle=False) as data:
                if not np.array_equal(data["labels_mc"], labels) or not np.array_equal(
                    data["record_ids"].astype(str), ids
                ):
                    raise ValueError("Cohort identity mismatch")
                probabilities = data["multiclass_probs"]
                predicted = probabilities.argmax(axis=1)
            cm = np.bincount(5 * labels + predicted, minlength=25).reshape(5, 5)
            if not np.array_equal(
                cm, confusion_matrix(labels, predicted, labels=range(5))
            ):
                raise ValueError("Independent confusion calculations disagree")
            common = {
                "stage": stage,
                "seed": row["seed"],
                "lead_names": row["lead_names"],
                "lead_indices": row["lead_indices"],
                "prediction_sha256": row["prediction_sha256"],
            }
            correct = int(np.trace(cm))
            cohort.append(
                {
                    **common,
                    "n": len(labels),
                    "correct": correct,
                    "incorrect": len(labels) - correct,
                    "accuracy": correct / len(labels),
                }
            )
            f1s = []
            for k, name in enumerate(CLASSES):
                tp = int(cm[k, k])
                fn = int(cm[k].sum()) - tp
                fp = int(cm[:, k].sum()) - tp
                tn = len(labels) - tp - fn - fp
                precision = tp / (tp + fp) if tp + fp else 0.0
                recall = tp / (tp + fn)
                f1 = 2 * tp / (2 * tp + fp + fn)
                f1s.append(f1)
                disease.append(
                    {
                        **common,
                        "class": name,
                        "actual_positive": tp + fn,
                        "actual_negative": tn + fp,
                        "TP_found": tp,
                        "FN_missed": fn,
                        "FP_wrong_classification": fp,
                        "TN": tn,
                        "precision": precision,
                        "sensitivity": recall,
                        "specificity": tn / (tn + fp),
                        "f1": f1,
                    }
                )
            if not np.isclose(np.mean(f1s), float(row["macro_f1"]), atol=1e-12, rtol=0):
                raise ValueError("Counts do not reproduce frozen Macro-F1")
    write_csv(OUT / "cohort_accuracy.csv", cohort)
    write_csv(OUT / "disease_counts.csv", disease)

    durations, header_manifest = {}, []
    for path in sorted((ROOT / "data/raw/cpsc2018").glob("*.hea")):
        first = path.read_text(encoding="utf-8").splitlines()[0].split()
        durations[path.stem] = int(first[3]) / float(first[2].split("/")[0])
        header_manifest.append(
            {
                "record_id": path.stem,
                "seconds": durations[path.stem],
                "header_sha256": sha(path),
            }
        )
    if len(durations) != 6877 or any(rid not in durations for rid in ids):
        raise ValueError("Raw header coverage mismatch")
    write_csv(OUT / "raw_record_durations.csv", header_manifest)
    signals = np.load(
        ROOT / "data/processed/cpsc2018_mc/test/signals.npy", mmap_mode="r"
    )
    if signals.shape != (936, 12, 5000):
        raise ValueError("Unexpected model input duration")
    durations_test = [durations[rid] for rid in ids]
    summary = {
        "status": "passed",
        "prediction_files_checked": len(cohort),
        "cohort_rows": len(cohort),
        "disease_rows": len(disease),
        "classification_rule": "argmax of five class probabilities, no threshold optimization",
        "false_positive_unit": "10-second record classification, not clinical alert or episode",
        "test_records": len(ids),
        "test_shape": list(signals.shape),
        "sample_rate_hz": 500,
        "model_window_seconds": 10,
        "test_input_seconds_including_padding": len(ids) * 10,
        "test_actual_signal_seconds_within_model_windows": sum(
            min(x, 10) for x in durations_test
        ),
        "raw_all_records": len(durations),
        "raw_all_seconds": sum(durations.values()),
        "raw_min_seconds": min(durations.values()),
        "raw_max_seconds": max(durations.values()),
        "raw_test_full_record_seconds": sum(durations_test),
        "seven_days_seconds_per_person": 7 * 24 * 3600,
        "seven_days_nonoverlapping_10_second_windows_per_person": 7 * 24 * 360,
        "unique_people_verified": False,
        "seven_day_continuity_available_in_this_cohort": False,
        "sources": sources,
        "labels_sha256": sha(labels_path),
        "ids_sha256": sha(ids_path),
        "script_sha256": sha(Path(__file__)),
        "new_gpu_inferences": 0,
        "failures": 0,
        "skips": 0,
        "limitations": "Same exploratory cohort; known published SSL exposure and unresolved "
        "warm lineage remain. Counts are against the historical single-label reference, "
        "not patient diagnoses, event detection, false alerts per day or seven-day performance.",
    }
    (OUT / "verification.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
