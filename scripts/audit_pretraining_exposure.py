"""Join historical P1 IDs to the pinned, published ECG-FM pretraining split."""

import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "work/exposure_sources_20260910/meta_split_physionet.csv"
SOURCE_SHA = "a2d17b0485a8186d227e82f2200ceb8b2ba7423df23a4fd306275f2197b4d88d"
REVISION = "9f926f1911bb9f24789b5c6407677d58ad753054"
OUT = ROOT / "results/pretraining_exposure_20260910"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_source(row):
    parts = PurePosixPath(row["source_path"]).parts
    if len(parts) != 7 or parts[:4] != ("files", "challenge-2021", "1.0.3", "training"):
        raise ValueError("Unexpected upstream path schema")
    if row["split"] not in {"train", "valid", "test"}:
        raise ValueError("Unexpected upstream split")
    return (parts[4], parts[-1]), row["split"]


def main():
    if sha(SOURCE) != SOURCE_SHA:
        raise ValueError("Published source fingerprint changed")
    mapping, totals = {}, defaultdict(Counter)
    with SOURCE.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key, split = parse_source(row)
            if key in mapping:
                raise ValueError("Duplicate upstream record")
            mapping[key] = split
            totals[key[0]][split] += 1

    # Verify the exact original ID files previously audited, not regenerated splits.
    prior_path = ROOT / "results/stage2_lineage/split_audit.json"
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    fingerprints = {item["path"]: item["sha256"] for item in prior["source_files"]}
    summaries, rows = {}, []
    for dataset in ("cpsc2018_mc", "cpsc2018"):
        for split in ("train", "val", "test"):
            prefix = (
                "data/processed"
                if split == "test"
                else ("work/stage2_original_assets/data/processed")
            )
            relative = f"{prefix}/{dataset}/{split}/record_ids.npy"
            path = ROOT / relative
            if sha(path) != fingerprints[relative]:
                raise ValueError(f"Historical ID fingerprint changed: {relative}")
            ids = np.load(path, allow_pickle=True).astype(str)
            if len(np.unique(ids)) != len(ids):
                raise ValueError("Historical IDs are not unique")
            membership = [mapping.get(("cpsc_2018", rid), "unlisted") for rid in ids]
            counts = dict(Counter(membership))
            for upstream_split in ("train", "valid", "test"):
                upstream_ids = {
                    k[1]
                    for k, v in mapping.items()
                    if k[0] == "cpsc_2018" and v == upstream_split
                }
                crosscheck = np.intersect1d(ids, list(upstream_ids))
                if len(crosscheck) != counts.get(upstream_split, 0):
                    raise ValueError("Independent intersection count mismatch")
            summaries[f"{dataset}/{split}"] = {"n": len(ids), "membership": counts}
            for rid, member in zip(ids, membership):
                rows.append(
                    {
                        "p1_dataset": dataset,
                        "p1_split": split,
                        "record_id": rid,
                        "published_pretraining_split": member,
                    }
                )

    OUT.mkdir(exist_ok=True)
    table = OUT / "record_membership.csv"
    with table.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "status": "published_split_membership_verified",
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "upstream_revision": REVISION,
        "upstream_url": f"https://raw.githubusercontent.com/bowang-lab/ecg-fm/{REVISION}/splits/meta_split_physionet.csv",
        "upstream_sha256": SOURCE_SHA,
        "upstream_rows": len(mapping),
        "upstream_dataset_counts": dict(totals),
        "original_p1_splits": summaries,
        "prior_split_audit_sha256": sha(prior_path),
        "script_sha256": sha(Path(__file__)),
        "record_table_rows": len(rows),
        "record_table_sha256": sha(table),
        "unlisted_is_independent": False,
        "interpretation": "Published pretraining split membership, not runtime training logs. "
        "SSL train exposure differs from supervised label leakage. Published test membership "
        "alone does not establish patient-level or full warm-chain independence.",
        "new_inference_runs": 0,
        "new_training_runs": 0,
    }
    (OUT / "audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"datasets": dict(totals), "p1": summaries}, indent=2))


if __name__ == "__main__":
    main()
