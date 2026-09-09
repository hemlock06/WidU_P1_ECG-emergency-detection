"""Acquire official VFDB files and audit format/annotations, without model inference."""
import csv
import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import wfdb

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://physionet.org/files/vfdb/1.0.0/"
RAW = ROOT / "data/raw/vfdb_1.0.0"
OUT = ROOT / "results/vfdb_readiness_20260910"


def acquire(name):
    path = RAW / name
    reused = path.exists()
    if not reused:
        with urlopen(BASE + name, timeout=60) as response:
            payload = response.read()
        with path.open("xb") as handle:
            handle.write(payload)
    payload = path.read_bytes()
    return {"file": name, "url": BASE + name, "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(), "reused": reused}


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = [acquire("RECORDS")]
    records = (RAW / "RECORDS").read_text().split()
    assert len(records) == len(set(records)) == 22
    assert all(name.isdigit() for name in records)
    names = [name + suffix for name in records for suffix in (".hea", ".atr", ".dat")]
    failures = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [(name, pool.submit(acquire, name)) for name in names]
        for name, future in futures:
            try:
                manifest.append(future.result())
            except (OSError, ValueError) as exc:
                failures.append({"file": name, "error": repr(exc)})
    (OUT / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if failures:
        (OUT / "failures.json").write_text(json.dumps(failures, indent=2) + "\n")
        raise RuntimeError(f"{len(failures)} acquisition failures; retained in failures.json")
    inventory, annotations = [], []
    for name in records:
        path = str(RAW / name)
        header = wfdb.rdheader(path)
        signal = wfdb.rdrecord(path, physical=False)
        ann = wfdb.rdann(path, "atr")
        assert signal.d_signal.shape == (header.sig_len, header.n_sig)
        assert np.isfinite(signal.d_signal).all()
        assert np.all(np.diff(ann.sample) >= 0)
        assert np.all((ann.sample >= 0) & (ann.sample <= header.sig_len))
        checksums = [int(signal.d_signal[:, i].astype(np.int64).sum() % 65536)
                     for i in range(header.n_sig)]
        assert checksums == [int(v) % 65536 for v in header.checksum]
        inventory.append({"record_id": name, "channels": header.n_sig, "fs": header.fs,
                          "samples": header.sig_len, "seconds": header.sig_len / header.fs,
                          "channel_names": ";".join(header.sig_name),
                          "annotations": len(ann.sample), "wfdb_checksum_passed": True})
        for sample, symbol, aux in zip(ann.sample, ann.symbol, ann.aux_note):
            annotations.append({"record_id": name, "sample": int(sample),
                                "seconds": float(sample / header.fs), "symbol": symbol,
                                "aux_note": aux.rstrip("\x00")})
    counts = Counter(row["aux_note"] for row in annotations)
    write_csv(OUT / "record_inventory.csv", inventory)
    write_csv(OUT / "raw_annotations.csv", annotations)
    write_csv(OUT / "annotation_counts.csv", [{"aux_note": key, "count": value}
                                              for key, value in sorted(counts.items())])
    result = {
        "status": "format_and_annotation_readiness_verified_not_model_validation",
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "records": len(inventory), "files_hashed": len(manifest),
        "waveform_checksums_passed": len(inventory),
        "total_record_seconds": sum(row["seconds"] for row in inventory),
        "channels": sorted({row["channels"] for row in inventory}),
        "sample_rates": sorted({row["fs"] for row in inventory}),
        "channel_names": sorted({row["channel_names"] for row in inventory}),
        "annotations": len(annotations), "raw_aux_counts": dict(sorted(counts.items())),
        "failures": 0, "skips": 0, "new_training_runs": 0, "model_inferences": 0,
        "limitations": [
            "Raw change-marker counts are not adjudicated independent clinical events.",
            "Noise markers do not terminate the underlying rhythm; no episode labels generated here.",
            "Generic ECG channel names do not establish standard lead or electrode geometry.",
            "Patient-level independence, full pretraining/adaptation exposure and pulse status unresolved.",
            "Selected abnormal recordings do not validate seven-day false-alert rates.",
            "Page describes half-hour records; use header-derived lengths for actual duration.",
        ],
    }
    (OUT / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
