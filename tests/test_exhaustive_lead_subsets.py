"""Fail-closed and metric checks for the frozen-model evaluation runner."""

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import ablation_exhaustive_lead_subsets as experiment
import calibrate_lead_subset_controls as calibration
import summarize_exhaustive_lead_subsets as summary
import verify_exhaustive_lead_results as independent


def test_independent_audit_confusion_and_tied_roc_fixture():
    truth = np.array([0, 0, 1, 1])
    predicted = np.array([0, 1, 1, 1])
    f1, recall = independent.confusion_values(truth, predicted, [0, 1])
    np.testing.assert_allclose(f1, [2 / 3, 0.8])
    np.testing.assert_allclose(recall, [0.5, 1.0])
    auroc, sensitivity = independent.curve_values(
        truth, np.array([0.1, 0.4, 0.4, 0.9])
    )
    assert auroc == 0.875
    assert sensitivity == 0.5


def test_complete_subset_space_and_mask():
    combinations = experiment.subsets()
    assert len(set(combinations)) == 793
    assert Counter(map(len, combinations)) == {1: 12, 2: 66, 3: 220, 4: 495}
    x = torch.ones(2, 12, 7)
    masked = experiment.apply_lead_mask(x, [0, 1, 7, 10])
    assert torch.equal(masked[:, [0, 1, 7, 10]], torch.ones(2, 4, 7))
    assert masked.sum() == 2 * 4 * 7
    assert x.sum() == 2 * 12 * 7


def test_stochastic_gate_requires_all_draws_and_both_historical_and_fixed_seed():
    values = np.linspace(0.6, 0.7, 39)
    assert calibration.interval_check(0.65, values, 0.66)["passed"]
    assert not calibration.interval_check(0.71, values, 0.66)["passed"]
    assert not calibration.interval_check(0.65, values, 0.71)["passed"]
    with pytest.raises(ValueError, match="39 finite"):
        calibration.interval_check(0.65, values[:-1], 0.66)
    values[0] = np.nan
    with pytest.raises(ValueError, match="39 finite"):
        calibration.interval_check(0.65, values, 0.66)


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
    ym = np.tile(np.arange(5), 2)
    yb = np.isin(ym, [1, 2]).astype(int)
    multiclass = experiment.metrics(yb, yb * 0.8 + 0.1, ym, np.eye(5)[ym] * 0.9 + 0.02)
    rows = []
    for model in experiment.MODELS:
        for combo in experiment.subsets():
            metric_base = base
            if model == "P1_a07":
                metric_base = multiclass
            rows.append(
                {
                    **{k: str(v) for k, v in metric_base.items()},
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
    saved_macro = rows[0]["macro_f1"]
    rows[0]["macro_f1"] = ""
    with pytest.raises(ValueError, match="Missing measurable"):
        summary.validate(rows)
    rows[0]["macro_f1"] = saved_macro
    rows[0]["binary_auroc"] = "nan"
    with pytest.raises(ValueError, match="Nonfinite"):
        summary.validate(rows)


def test_derived_limb_leads_do_not_inflate_electrode_count():
    assert summary.electrode_count((0,)) == 2
    assert summary.electrode_count((0, 1, 2, 3)) == 3
    assert summary.electrode_count((0, 1, 7, 10)) == 5


def test_candidate_categories_apply_distinct_constraints():
    examples = [
        ((0,), 0.3),
        ((1,), 0.6),
        ((0, 1), 0.7),
        ((0, 7), 0.8),
        ((0, 1, 7, 10), 0.9),
    ]
    rows = [
        {
            "model": "P1_a07",
            "lead_indices": ";".join(map(str, c)),
            "n_leads": len(c),
            "macro_f1": score,
        }
        for c, score in examples
    ]
    selected = summary.select_candidates(rows)
    assert summary.indices(selected["performance"]) == (0, 1, 7, 10)
    assert summary.indices(selected["minimum_electrodes"]) == (1,)
    assert summary.indices(selected["limb_only"]) == (0, 1)
    assert summary.indices(selected["two_channels"]) == (0, 7)
    assert summary.indices(selected["chest_included"]) == (0, 1, 7, 10)


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


def test_adapter_only_load_preserves_frozen_weights_and_rejects_missing_adapter():
    from ablation_nlead_curve import LoRALinear

    model = LoRALinear(torch.nn.Linear(3, 2), rank=1, alpha=2)
    frozen = model.original.weight.detach().clone()
    adapters = {
        k: torch.ones_like(v) for k, v in model.state_dict().items() if "lora_" in k
    }
    experiment.load_backbone_state(model, adapters, adapter_only=True)
    torch.testing.assert_close(model.original.weight, frozen)
    assert torch.all(model.lora_B.weight == 1)
    with pytest.raises(ValueError, match="every LoRA"):
        experiment.load_backbone_state(
            model, {"lora_A.weight": adapters["lora_A.weight"]}, adapter_only=True
        )
    with pytest.raises(RuntimeError):
        experiment.load_backbone_state(model, adapters, adapter_only=False)
