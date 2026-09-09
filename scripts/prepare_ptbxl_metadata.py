"""Metadata feasibility only; numeric cross-version IDs remain provisional."""

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "work/exposure_sources_20260910"
OUT = ROOT / "results/pretraining_exposure_20260910"


def main():
    manifest = json.loads((SOURCE / "manifest.json").read_text(encoding="utf-8"))
    for item in manifest:
        if (
            hashlib.sha256((ROOT / item["path"]).read_bytes()).hexdigest()
            != item["sha256"]
        ):
            raise ValueError("Source metadata hash changed")
    records = pd.read_csv(SOURCE / "ptbxl_database.csv")
    upstream = pd.read_csv(SOURCE / "meta_split_physionet.csv")
    upstream = upstream[upstream.source_path.str.contains("/ptb-xl/")].copy()
    upstream["ecg_id"] = upstream.source_path.str.extract(r"/HR(\d+)$").astype(int)
    records = records.merge(
        upstream[["ecg_id", "split"]], on="ecg_id", how="left", validate="one_to_one"
    )
    if len(records) != 21799 or records.patient_id.nunique() != 18869:
        raise ValueError("Unexpected PTB-XL release counts")
    records["current_version_all_patient_records_published_test"] = records.groupby(
        "patient_id"
    )["split"].transform(lambda values: values.eq("test").all())
    subset = records[records.current_version_all_patient_records_published_test]
    columns = [
        "ecg_id",
        "patient_id",
        "strat_fold",
        "split",
        "current_version_all_patient_records_published_test",
    ]
    OUT.mkdir(exist_ok=True)
    output = OUT / "ptbxl_provisional_membership.csv"
    records[columns].to_csv(output, index=False)
    result = {
        "status": "metadata_only_not_an_eligible_external_cohort",
        "sources": manifest,
        "records": len(records),
        "patients": records.patient_id.nunique(),
        "numeric_id_mapping_assumption": "Challenge HRnnnnn corresponds to PTB-XL ecg_id nnnnn",
        "cross_version_signal_identity_verified": False,
        "all_historical_patient_records_checked": False,
        "membership": records["split"].fillna("unlisted").value_counts().to_dict(),
        "provisional_current_version_patient_filter": {
            "records": len(subset),
            "patients": subset.patient_id.nunique(),
        },
        "warnings": [
            (
                "The old Challenge release contains records removed from PTB-XL 1.0.3; "
                "their patient associations must also be checked."
            ),
            "Upstream adapted-checkpoint exposure is unresolved.",
            "No waveform, model prediction, threshold, or external performance was inspected.",
        ],
        "table_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (OUT / "ptbxl_metadata_readiness.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "status",
                    "membership",
                    "provisional_current_version_patient_filter",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
