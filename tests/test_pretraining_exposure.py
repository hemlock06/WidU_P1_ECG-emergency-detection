"""Reject ambiguous upstream paths and unknown split labels."""

import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "exposure",
    Path(__file__).resolve().parents[1] / "scripts/audit_pretraining_exposure.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_namespaces_prevent_cross_dataset_id_collision():
    row = {
        "source_path": "files/challenge-2021/1.0.3/training/cpsc_2018/g1/A0001",
        "split": "train",
    }
    assert MODULE.parse_source(row) == (("cpsc_2018", "A0001"), "train")
    row["source_path"] = row["source_path"].replace("cpsc_2018", "georgia")
    assert MODULE.parse_source(row)[0] == ("georgia", "A0001")


@pytest.mark.parametrize(
    "path,split",
    [
        ("cpsc_2018/A0001", "train"),
        ("files/challenge-2021/1.0.3/training/cpsc_2018/g1/A0001", "validation"),
    ],
)
def test_reject_unrecognized_upstream_schema(path, split):
    with pytest.raises(ValueError):
        MODULE.parse_source({"source_path": path, "split": split})
