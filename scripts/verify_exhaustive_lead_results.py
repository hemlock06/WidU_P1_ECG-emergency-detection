"""Independently audit persisted predictions against source labels and raw metrics."""

import csv
import hashlib
import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import auc, confusion_matrix, roc_curve

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
CLASSES = ["nsr", "af", "ischemia", "conduction", "ectopy"]


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def curve_values(truth, score):
    fpr, tpr, _ = roc_curve(truth, score, drop_intermediate=False)
    return float(auc(fpr, tpr)), float(tpr[(1 - fpr) >= 0.95].max())


def confusion_values(truth, predicted, labels):
    matrix = confusion_matrix(truth, predicted, labels=labels)
    tp = np.diag(matrix).astype(float)
    f1_denominator = matrix.sum(0) + matrix.sum(1)
    recall_denominator = matrix.sum(1)
    f1 = np.divide(
        2 * tp, f1_denominator, out=np.zeros_like(tp), where=f1_denominator > 0
    )
    recall = np.divide(
        tp, recall_denominator, out=np.zeros_like(tp), where=recall_denominator > 0
    )
    return f1, recall


def main():
    root = Path("results")
    source = root / "exhaustive_lead_subsets_1to4.csv"
    with source.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    status = json.loads((root / "exhaustive_lead_subsets_run_status.json").read_text())
    controls = json.loads((root / "exhaustive_lead_subsets_controls.json").read_text())
    fingerprint = status["run_fingerprint"]
    assert status["rows"] == len(rows) == 1586 and status["failures"] == 0
    assert controls["fingerprint"] == fingerprint
    # Verify every concrete file in the original identity; fairseq aggregate is
    # verified separately by the runner identity check in candidate stability.
    file_hashes = controls["identity"]["sha256"]
    checked_files = 0
    for name, expected in file_hashes.items():
        if name == "fairseq_signals_python_source":
            continue
        assert digest(name) == expected, name
        checked_files += 1
    sources = {}
    for model, folder in (("P1_a07", "cpsc2018_mc"), ("reference_iii", "cpsc2018")):
        base = Path("data/processed") / folder / "test"
        # Locally restored historical IDs are object arrays, already hash-verified.
        ids = np.load(base / "record_ids.npy", allow_pickle=True).astype(str)
        labels = np.load(base / "labels.npy", allow_pickle=False)
        binary = np.load(base / "labels_bin.npy") if model == "P1_a07" else labels
        sources[model] = (ids, labels, binary)
    expected_combos = {
        c for n in range(1, 5) for c in itertools.combinations(range(12), n)
    }
    seen = set()
    max_error = 0.0
    metric_checks = 0
    for row in rows:
        model = row["model"]
        combo = tuple(map(int, row["lead_indices"].split(";")))
        assert combo in expected_combos and (model, combo) not in seen
        seen.add((model, combo))
        assert row["run_fingerprint"] == fingerprint
        assert row["lead_names"] == ";".join(LEADS[i] for i in combo)
        assert int(row["n_leads"]) == len(combo)
        pred_path = (
            Path("outputs/exhaustive_lead_subsets")
            / fingerprint
            / f"{model}_{'-'.join(map(str, combo))}.npz"
        )
        assert digest(pred_path) == row["prediction_sha256"]
        ids, labels, binary = sources[model]
        assert int(row["n_samples"]) == len(ids)
        with np.load(pred_path, allow_pickle=False) as pred:
            np.testing.assert_array_equal(pred["record_ids"], ids)
            np.testing.assert_array_equal(pred["labels_bin"], binary)
            pb = pred["binary_probs"]
            assert np.isfinite(pb).all() and ((pb >= 0) & (pb <= 1)).all()
            auroc, sens = curve_values(binary, pb)
            f1, _ = confusion_values(binary, pb >= 0.5, [0, 1])
            values = {"binary_auroc": auroc, "sens_at_95sp": sens, "f1_at_05": f1[1]}
            if model == "P1_a07":
                np.testing.assert_array_equal(pred["labels_mc"], labels)
                pm = pred["multiclass_probs"]
                assert np.isfinite(pm).all() and ((pm >= 0) & (pm <= 1)).all()
                np.testing.assert_allclose(pm.sum(1), 1, atol=1e-6)
                f1, recalls = confusion_values(labels, pm.argmax(1), list(range(5)))
                values.update(macro_f1=f1.mean(), macro_sensitivity=recalls.mean())
                for index, name in enumerate(CLASSES):
                    auroc, sens = curve_values(labels == index, pm[:, index])
                    values.update(
                        {
                            f"auroc_{name}": auroc,
                            f"sens95_{name}": sens,
                            f"f1_{name}": f1[index],
                            f"sensitivity_{name}": recalls[index],
                        }
                    )
                assert row["measurable"] == "binary;multiclass"
            else:
                assert row["measurable"] == "binary"
                assert row["macro_f1"] == row["macro_sensitivity"] == ""
                assert all(
                    row[f"{metric}_{name}"] == ""
                    for name in CLASSES
                    for metric in ("auroc", "sens95", "f1", "sensitivity")
                )
            for key, value in values.items():
                error = abs(float(row[key]) - value)
                assert np.isfinite(error) and error < 1e-7, (model, combo, key)
                max_error = max(max_error, error)
                metric_checks += 1
    assert seen == {(model, c) for model in sources for c in expected_combos}
    evidence = {
        "status": "passed",
        "run_fingerprint": fingerprint,
        "raw_sha256": digest(source),
        "script_sha256": digest(__file__),
        "rows": len(rows),
        "model_counts": dict(Counter(r["model"] for r in rows)),
        "n_counts_per_model": {
            model: dict(Counter(len(c) for m, c in seen if m == model))
            for model in sources
        },
        "identity_files_rehashed": checked_files,
        "prediction_files_rehashed_and_source_labels_matched": len(rows),
        "independent_metric_checks": metric_checks,
        "maximum_absolute_metric_error": max_error,
        "failures": 0,
        "resumed_skips": status["resumed_skips"],
        "scope": "Independent confusion-matrix and full-ROC recalculation; not independent cohort validation",
    }
    (root / "exhaustive_lead_subsets_verification.json").write_text(
        json.dumps(evidence, indent=2), encoding="utf-8"
    )
    print(json.dumps(evidence))


if __name__ == "__main__":
    main()
