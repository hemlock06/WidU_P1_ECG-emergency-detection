"""Adversarial checks for split ID validation and overlap accounting."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "stage2", Path(__file__).resolve().parents[1] / "scripts/audit_stage2_lineage.py"
)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


@pytest.mark.parametrize("ids", [["A0001", "A0001"], ["A0001", "B0002"], [1, 2]])
def test_reject_invalid_or_duplicate_ids(ids):
    with pytest.raises(ValueError):
        AUDIT.check_ids(np.array(ids, dtype=object))


def test_overlap_handles_order_and_disjoint_splits():
    left = AUDIT.check_ids(np.array(["A0003", "A0001", "A0002"], dtype=object))
    assert AUDIT.intersection(left, np.array(["A0002", "A0003"])) == ["A0002", "A0003"]
    assert AUDIT.intersection(left, np.array(["A0004"])) == []
