"""Fail-closed and metric checks for the frozen-model evaluation runner."""

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import ablation_exhaustive_lead_subsets as experiment
import summarize_exhaustive_lead_subsets as summary


def test_complete_subset_space_and_mask():
    combinations = experiment.subsets()
    assert len(set(combinations)) == 793
    assert Counter(map(len, combinations)) == {1: 12, 2: 66, 3: 220, 4: 495}
    x = torch.ones(2, 12, 7)
    masked = experiment.apply_lead_mask(x, [0, 1, 7, 10])
    assert torch.equal(masked[:, [0, 1, 7, 10]], torch.ones(2, 4, 7))
    assert masked.sum() == 2 * 4 * 7
    assert x.sum() == 2 * 12 * 7


def test_perfect_five_class_and_unmeasurable_binary_model():
    ym = np.tile(np.arange(5), 4)
    yb = np.isin(ym, [1, 2]).astype(int)
    pm = np.eye(5)[ym] * 0.9 + 0.02
    pb = yb * 0.8 + 0.1
    result = experiment.metrics(yb, pb, ym, pm)
    assert result["macro_f1"] == result["macro_sensitivity"] == 1
    for c in experiment.CLASSES:
        for metric in ("auroc", "f1", "sensitivity", "sens95"):
            assert result[f"{metric}_{c}"] == 1
    binary = experiment.metrics(yb, pb)
    assert binary["macro_f1"] == ""
    assert binary["auroc_af"] == ""
    assert binary["measurable"] == "binary"


def test_missing_assets_block_before_model_load(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="Required assets missing"):
        experiment.asset_identity(42, 32)


def test_wrong_historical_model_is_not_a_passing_control():
    observed = {"binary_auroc": 0.9142, "sens_at_95sp": 0.7072}
    check = experiment.historical_checks("reference_iii", tuple(range(12)), observed)
    assert check["status"] == "failed"
    assert (
        experiment.historical_checks("P1_a07", (1,), observed)["status"]
        == "not_available"
    )


def test_summary_rejects_missing_duplicate_and_nonfinite_rows():
    base = experiment.metrics(np.array([0, 1, 0, 1]), np.array([0.1, 0.9, 0.2, 0.8]))
    rows = []
    for model in experiment.MODELS:
        for combo in experiment.subsets():
            rows.append(
                {
                    **{k: str(v) for k, v in base.items()},
                    "model": model,
                    "lead_indices": ";".join(map(str, combo)),
                    "lead_names": ";".join(experiment.LEAD_NAMES[i] for i in combo),
                    "n_leads": str(len(combo)),
                    "n_samples": str(experiment.MODELS[model][2]),
                    "run_fingerprint": "unit-test-only",
                }
            )
    summary.validate(rows)
    assert summary.summary_bytes(rows) == summary.summary_bytes(list(rows))
    with pytest.raises(ValueError, match="combinations"):
        summary.validate(rows[:-1])
    with pytest.raises(ValueError, match="combinations"):
        summary.validate(rows + rows[:1])
    rows[0]["binary_auroc"] = "nan"
    with pytest.raises(ValueError, match="Nonfinite"):
        summary.validate(rows)


def test_derived_limb_leads_do_not_inflate_electrode_count():
    assert summary.electrode_count((0,)) == 2
    assert summary.electrode_count((0, 1, 2, 3)) == 3
    assert summary.electrode_count((0, 1, 7, 10)) == 5


def test_prediction_resets_feature_mask_rng_for_resume(monkeypatch):
    class RandomFeatures(torch.nn.Module):
        def forward(self, source, **kwargs):
            return {
                "x": torch.tensor(
                    np.random.normal(size=(len(source), 2, 768)), dtype=torch.float32
                )
            }

    monkeypatch.setattr(torch.Tensor, "cuda", lambda self: self)
    model = (RandomFeatures(), experiment.BinaryHead(), experiment.MCHead())
    x = np.zeros((3, 12, 5), dtype=np.float32)
    first = experiment.predict(model, x, (1,), 2, seed=42)
    np.random.random(100)
    repeated = experiment.predict(model, x, (1,), 2, seed=42)
    changed = experiment.predict(model, x, (1,), 2, seed=43)
    for a, b in zip(first, repeated):
        np.testing.assert_array_equal(a, b)
    assert not np.array_equal(first[1], changed[1])
