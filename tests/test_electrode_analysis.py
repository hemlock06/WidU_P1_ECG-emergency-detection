"""Check independent audit failure cases and multiobjective selection rules."""

import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import analyze_electrode_coverage as analysis


def make_row(indices, scores=(0.6, 0.6, 0.6, 0.6), macro=0.6):
    points = analysis.contact_set(indices)
    return {
        "lead_indices": ";".join(map(str, indices)),
        "lead_names": ";".join(analysis.LEADS[i] for i in indices),
        "measurement_contacts": str(len(points)),
        "contacts_with_extra_drl": str(len(points) + 1),
        "contact_names": ";".join(sorted(points)),
        "n_leads": str(len(indices)),
        "stage1_run_id": "test",
        "model": "P1_a07",
        "seed": "42",
        "n_samples": "936",
        "measurable": "binary;multiclass",
        "origin": "prior_exhaustive"
        if len(indices) <= 4
        else ("prior_control" if len(indices) == 12 else "new_inference"),
        **{metric: str(macro) for metric in analysis.METRICS},
        **{f"sens95_{d}": str(s) for d, s in zip(analysis.DISEASES, scores)},
    }


def test_pareto_preserves_disease_tradeoffs_and_exact_ties():
    rows = [
        make_row((0,), (0.7, 0.7, 0.7, 0.7)),
        make_row((1,), (0.7, 0.7, 0.7, 0.7)),  # Exact objective tie survives.
        make_row((0, 1), (0.7, 0.7, 0.7, 0.7)),  # Extra contact with no gain.
        make_row((3,), (0.9, 0.9, 0.9, 0.1)),  # Weak ectopy, still nondominated.
        make_row((2,), (0.6, 0.6, 0.6, 0.6)),
    ]
    assert analysis.pareto_indices(rows) == [0, 1, 3]


def test_worst_disease_selection_follows_all_tie_breaks():
    rows = analysis.enrich(
        [
            make_row((0, 1), (0.5, 0.5, 0.5, 0.5), 0.99),
            make_row((0, 2), (0.5, 0.6, 0.6, 0.6), 0.70),
            make_row((3,), (0.5, 0.6, 0.6, 0.6), 0.71),
            make_row((4,), (0.5, 0.6, 0.6, 0.6), 0.71),
            make_row(tuple(range(12))),
        ]
    )
    winners, shortlist = analysis.selections(rows)
    selected = {
        r["criterion"]: r["lead_indices"]
        for r in winners
        if r["measurement_contacts"] == "3"
    }
    assert selected == {"macro_f1": "0;1", "worst_disease_sens95": "3"}
    assert len(shortlist) == 3


def test_complete_space_products_and_corrupt_metadata():
    raw = [
        make_row(c) for n in range(1, 13) for c in itertools.combinations(range(12), n)
    ]
    analysis.validate_rows(raw, "test")
    result = analysis.products(raw)
    assert len(result["summary.csv"]) == 4095
    assert len(result["coverage_grid.csv"]) == 4095 * 5
    assert len(result["contact_winners.csv"]) == 16
    assert len(result["complete_physical_configurations.csv"]) == 67
    assert {r["lead_indices"] for r in result["pareto.csv"]} == {"0", "1", "2"}
    assert {
        float(r["sensitivity_threshold"]) for r in result["coverage_grid.csv"]
    } == set(analysis.GRID)
    for row in result["coverage_grid.csv"]:
        assert row["disease_groups_met"] == (
            4 if row["sensitivity_threshold"] <= 0.6 else 0
        )
    with pytest.raises(ValueError, match="4095"):
        analysis.validate_rows(raw[:-1] + [raw[0]], "test")
    raw[0]["contacts_with_extra_drl"] = "2"
    with pytest.raises(ValueError, match="metadata"):
        analysis.validate_rows(raw, "test")


def test_incomplete_stage_cannot_produce_audit(tmp_path, monkeypatch):
    from contextlib import nullcontext

    import evaluate_electrode_coverage as runner

    (tmp_path / "status.json").write_text('{"status":"running"}')
    monkeypatch.setattr(analysis, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "exclusive_run", nullcontext)
    with pytest.raises(ValueError, match="finish"):
        analysis.audit()
    assert not (tmp_path / "independent_audit.json").exists()


def test_independent_prediction_checks_reject_wrong_labels(tmp_path):
    ids = np.array(["a", "b", "c", "d", "e"])
    ym = np.arange(5)
    yb = np.array([0, 1, 1, 0, 0])
    path = tmp_path / "example.npz"
    np.savez(
        path,
        record_ids=ids,
        labels_mc=ym,
        labels_bin=yb,
        binary_probs=np.full(5, 0.5),
        multiclass_probs=np.full((5, 5), 0.2),
    )
    row = {"prediction_path": str(path), "prediction_sha256": analysis.digest(path)}
    analysis.check_prediction(row, ids, yb, ym)
    with pytest.raises(AssertionError):
        analysis.check_prediction(row, ids[::-1], yb, ym)
    with pytest.raises(ValueError, match="hash"):
        analysis.check_prediction({**row, "prediction_sha256": "0" * 64}, ids, yb, ym)


def test_independent_metrics_against_four_saved_parent_controls(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(root)
    base = root / "data/processed/cpsc2018_mc/test"
    if not base.exists():
        pytest.skip("Restored local research assets are not in Git")
    ids = np.load(base / "record_ids.npy", allow_pickle=True).astype(str)
    ym = np.load(base / "labels.npy")
    yb = np.load(base / "labels_bin.npy")
    rows = [
        r
        for r in analysis.read_csv(
            root / "results/exhaustive_lead_subsets_controls.csv"
        )
        if r["model"] == "P1_a07"
    ]
    assert len(rows) == 4
    for row in rows:
        path = (
            root
            / "outputs/exhaustive_lead_subsets"
            / row["run_fingerprint"]
            / ("P1_a07_" + "-".join(map(str, analysis.combo(row))) + ".npz")
        )
        pb, pm = analysis.check_prediction(
            {**row, "prediction_path": str(path)}, ids, yb, ym
        )
        expected = analysis.independent_metrics(yb, pb, ym, pm)
        assert len(expected) == 25
        for metric, value in expected.items():
            assert value == pytest.approx(float(row[metric]), abs=1e-12)


def test_paired_bootstrap_retains_pairing_and_counts_skips_without_reroll():
    import repeat_electrode_candidates as repeated

    ym = np.arange(5)
    yb = np.array([0, 1, 1, 0, 0])
    probabilities = (yb * 0.8 + 0.1, np.eye(5) * 0.8 + 0.04)
    result = repeated.paired_intervals(
        yb, ym, probabilities, probabilities, draws=200, seed=31415
    )
    rng = np.random.default_rng(31415)
    valid = sum(len(set(rng.integers(0, 5, 5))) == 5 for _ in range(200))
    assert 0 < valid < 200
    assert len(result) == 18
    for row in result:
        assert row["valid"] == valid and row["skipped"] == 200 - valid
        assert row["difference_vs12"] == row["ci_low"] == row["ci_high"] == 0
    assert repeated.SEEDS == tuple(range(30000, 30010))


def test_bootstrap_never_rerolls_when_no_valid_draw():
    import repeat_electrode_candidates as repeated

    ym = np.arange(5)
    yb = np.array([0, 1, 1, 0, 0])
    probabilities = (yb * 0.8 + 0.1, np.eye(5) * 0.8 + 0.04)
    with pytest.raises(ValueError, match="No valid bootstrap"):
        repeated.paired_intervals(
            yb, ym, probabilities, probabilities, draws=1, seed=31415
        )
