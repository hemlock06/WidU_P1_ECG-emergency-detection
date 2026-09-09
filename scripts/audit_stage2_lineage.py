"""Audit restored historical split IDs without modifying data or model weights."""

import csv
import hashlib
import itertools
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/stage2_lineage"
CHECKPOINT = "outputs/lora_multitask_snr_a07/lora_multitask_snr_best.pt"
CHECKPOINT_SHA = "287148bfd01ac67b5192268c6c69cc4cc230c004bdfdf332285f38a3b43b08dd"


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def intersection(left, right):
    """Cross-check two independent intersection implementations."""
    result = sorted(set(left.tolist()) & set(right.tolist()))
    if result != np.intersect1d(left, right).tolist():
        raise ValueError("Intersection implementations disagree")
    return result


def check_ids(values):
    if values.ndim != 1 or not all(
        isinstance(v, str) and re.fullmatch(r"A\d+", v) for v in values
    ):
        raise ValueError("Unexpected CPSC record ID schema")
    values = values.astype(str)
    if len(np.unique(values)) != len(values):
        raise ValueError("Duplicate record IDs within a split")
    return values


def main():
    transfer = json.loads((OUT / "transfer.json").read_text(encoding="utf-8"))
    sources = []
    for item in transfer["files"]:
        path = ROOT / item["destination"]
        if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise ValueError(f"Restored file differs from home manifest: {path}")

    def read_array(path, ids=False):
        sources.append(
            {
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
        # Historical ID arrays use NumPy object storage; restored files have
        # already been checked against the source manifest. Labels never pickle.
        array = np.load(path, allow_pickle=ids)
        return check_ids(array) if ids else array

    arrays, counts, mc_labels = {}, {}, {}
    expected = {"cpsc2018_mc": [4357, 933, 936], "cpsc2018": [2217, 474, 474]}
    for dataset, sizes in expected.items():
        for split, size in zip(("train", "val", "test"), sizes):
            prefix = (
                "data/processed"
                if split == "test"
                else ("work/stage2_original_assets/data/processed")
            )
            directory = ROOT / prefix / dataset / split
            key = f"original/{dataset}/{split}"
            ids = read_array(directory / "record_ids.npy", ids=True)
            if len(ids) != size:
                raise ValueError(f"Unexpected split length: {key}")
            arrays[key] = ids
            counts[key] = {"n": len(ids), "unique": len(np.unique(ids))}
            if dataset == "cpsc2018_mc":
                labels = read_array(directory / "labels.npy")
                binary = read_array(directory / "labels_bin.npy")
                if labels.shape != ids.shape or binary.shape != ids.shape:
                    raise ValueError("ID/label shape mismatch")
                if not np.isin(labels, np.arange(5)).all():
                    raise ValueError("Invalid five-class label")
                if not np.array_equal(binary, np.isin(labels, [1, 2])):
                    raise ValueError("Binary label is not AF-or-ischemia")
                counts[key]["class_counts"] = np.bincount(labels, minlength=5).tolist()
                counts[key]["binary_counts"] = np.bincount(binary, minlength=2).tolist()
                mc_labels.update(zip(ids.tolist(), labels.tolist()))

    missing_optional = []
    for dataset in ("cpsc2018_mc_ml", "cpsc2018_regenerated_20260908"):
        for split in ("train", "val", "test"):
            path = ROOT / "data/processed" / dataset / split / "record_ids.npy"
            key = f"regenerated/{dataset}/{split}"
            if not path.exists():
                missing_optional.append(key)
                continue
            arrays[key] = read_array(path, ids=True)
            counts[key] = {"n": len(arrays[key]), "unique": len(np.unique(arrays[key]))}

    matrix = []
    for left, right in itertools.combinations(sorted(arrays), 2):
        shared = intersection(arrays[left], arrays[right])
        matrix.append({"left": left, "right": right, "overlap": len(shared)})
    for dataset in expected:
        for a, b in itertools.combinations(("train", "val", "test"), 2):
            if intersection(
                arrays[f"original/{dataset}/{a}"], arrays[f"original/{dataset}/{b}"]
            ):
                raise ValueError(f"Historical within-dataset split overlap: {dataset}")

    test = arrays["original/cpsc2018_mc/test"]
    cross_rows = []
    for split in ("train", "val", "test"):
        for record_id in intersection(test, arrays[f"original/cpsc2018/{split}"]):
            cross_rows.append(
                {
                    "record_id": record_id,
                    "mc_test_label": mc_labels[record_id],
                    "historical_binary_split": split,
                }
            )

    checkpoint = ROOT / CHECKPOINT
    if sha256(checkpoint) != CHECKPOINT_SHA:
        raise ValueError("Final checkpoint identity differs from frozen experiment")
    import torch

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    metadata_keys = [
        "epoch",
        "val_bin_auroc",
        "val_macro_f1",
        "val_composite",
        "alpha",
        "lora_rank",
        "lora_alpha",
        "n_classes",
        "class_names",
        "emergency_classes",
    ]
    metadata = {key: state[key] for key in metadata_keys if key in state}
    result = {
        "status": "split_integrity_passed_training_lineage_unresolved",
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Record IDs only; no patient-level or signal-duplicate independence claim",
        "source_transfer_sha256": sha256(OUT / "transfer.json"),
        "audit_script_sha256": sha256(Path(__file__)),
        "counts": counts,
        "source_files": sources,
        "original_within_dataset_cross_split_overlap": 0,
        "mc_test_overlap_binary_splits": {
            split: sum(r["historical_binary_split"] == split for r in cross_rows)
            for split in ("train", "val", "test")
        },
        "missing_optional_splits": missing_optional,
        "checkpoint": {
            "path": CHECKPOINT,
            "sha256": CHECKPOINT_SHA,
            "keys": sorted(state),
            "metadata": metadata,
        },
        "lineage_verdict": "Final checkpoint contains no split hash or warm-start identity. "
        "Historical defaults/logs describe a 5d warm start, but the requested file is missing. "
        "Binary train/test overlap is not itself proof that the final model saw those records.",
        "missing_requested_assets": transfer["missing"],
        "inference_runs": 0,
        "training_runs": 0,
        "external_evaluation_gate": "Require upstream training exposure and label protocol audit",
    }
    for name, rows in [
        ("overlap_matrix.csv", matrix),
        ("mc_test_binary_overlap_ids.csv", cross_rows),
    ]:
        with (OUT / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    result["output_hashes"] = {
        name: sha256(OUT / name)
        for name in ("overlap_matrix.csv", "mc_test_binary_overlap_ids.csv")
    }
    (OUT / "split_audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "status",
                    "mc_test_overlap_binary_splits",
                    "missing_optional_splits",
                )
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
