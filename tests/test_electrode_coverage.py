"""Check the physical search space and retained prediction integrity."""

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import evaluate_electrode_coverage as coverage


def test_all_nonempty_subsets_and_physical_contact_counts():
    combinations = coverage.all_subsets()
    assert len(combinations) == len(set(combinations)) == 4095
    assert Counter(len(coverage.contacts(c)) for c in combinations) == {
        2: 3,
        3: 60,
        4: 384,
        5: 960,
        6: 1280,
        7: 960,
        8: 384,
        9: 64,
    }
    old = set(coverage.e.subsets()) | {tuple(range(12))}
    assert len(set(combinations) - old) == 3301
    assert coverage.contacts((0, 3)) == ("LA", "LL", "RA")
    assert coverage.contacts(tuple(range(6))) == ("LA", "LL", "RA")
    assert len(coverage.contacts(tuple(range(12)))) == 9
    assert [len(coverage.contacts(c)) for c in combinations] == sorted(
        len(coverage.contacts(c)) for c in combinations
    )


def test_saved_prediction_rejects_modified_labels_and_never_overwrites(tmp_path):
    ids = np.array([f"r{i}" for i in range(936)])
    ym = np.arange(936) % 5
    yb = np.isin(ym, [1, 2]).astype(int)
    pb = np.full(936, 0.5)
    pm = np.full((936, 5), 0.2)
    path = tmp_path / "pred.npz"
    coverage.save_prediction(path, ids, yb, ym, pb, pm)
    row = {"prediction_path": str(path), "prediction_sha256": coverage.e.sha256(path)}
    coverage.checked_prediction(row, ids, yb, ym)
    with pytest.raises(AssertionError):
        coverage.checked_prediction(row, ids[::-1], yb, ym)
    with pytest.raises(FileExistsError):
        coverage.save_prediction(path, ids, yb, ym, pb, pm)
    with pytest.raises(ValueError, match="hash mismatch"):
        coverage.checked_prediction({**row, "prediction_sha256": "0" * 64}, ids, yb, ym)
