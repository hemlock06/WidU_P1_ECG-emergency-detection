"""Cross-check the independent numerical path on ties and missing predictions."""

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, recall_score, roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from verify_electrode_analysis_final import counts, roc, scores


def test_tied_roc_with_bootstrap_duplicates():
    rng = np.random.default_rng(19)
    truth = np.tile([0, 1], 100)
    score = rng.integers(0, 8, len(truth)) / 8
    for _ in range(20):
        idx = rng.integers(0, len(truth), len(truth))
        area, sensitivity = roc(truth[idx], score[idx])
        fpr, tpr, _ = roc_curve(truth[idx], score[idx], drop_intermediate=False)
        np.testing.assert_allclose(
            area, roc_auc_score(truth[idx], score[idx]), atol=1e-14
        )
        assert sensitivity == tpr[(1 - fpr) >= 0.95].max()


def test_confusion_when_a_class_is_never_predicted():
    truth = np.repeat(np.arange(5), [5, 3, 9, 2, 4])
    prediction = np.zeros(len(truth), dtype=int)
    f1, recall = counts(truth, prediction, 5)
    np.testing.assert_array_equal(
        f1, f1_score(truth, prediction, average=None, zero_division=0)
    )
    np.testing.assert_array_equal(
        recall, recall_score(truth, prediction, average=None, zero_division=0)
    )


def test_saved_full_cohort_control_metrics():
    import analyze_electrode_coverage as original

    selected = original.read_csv(original.ROOT / "raw.csv")
    controls = [
        r
        for r in selected
        if r["lead_indices"] in {"1", "0;1", "0;1;7;10", ";".join(map(str, range(12)))}
    ]
    assert len(controls) == 4
    for row in controls:
        with np.load(row["prediction_path"], allow_pickle=False) as z:
            result = scores(
                z["labels_bin"],
                z["labels_mc"],
                z["binary_probs"],
                z["multiclass_probs"],
            )
        assert len(result) == 25
        for key, value in result.items():
            np.testing.assert_allclose(value, float(row[key]), atol=1e-14, rtol=0)
