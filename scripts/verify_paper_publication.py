"""Read back manuscript tables and release membership against original evidence."""
import csv
import hashlib
import json
import re
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/paper_package_20260910"


def rows(path):
    with (ROOT / path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def main():
    table = rows("results/paper_package_20260910/candidate_table.csv")
    raw = {r["lead_indices"]: r for r in rows("results/electrode_coverage_v1/raw.csv")}
    counts = {r["lead_indices"]: r for r in rows("results/classification_counts_20260910/cohort_accuracy.csv")
              if r["stage"] == "seed42"}
    for row in table:
        key = row["lead_indices"]
        assert abs(float(row["macro_f1"]) - float(raw[key]["macro_f1"])) < 1e-14
        for field in ("correct", "incorrect", "n", "accuracy"):
            assert abs(float(row[field])-float(counts[key][field])) < 1e-14
    disease = rows("results/paper_package_20260910/candidate_disease_counts.csv")
    for row in disease:
        tp, fn, fp, tn = [int(row[k]) for k in ("TP_found", "FN_missed", "FP_wrong_classification", "TN")]
        assert tp + fn + fp + tn == 936
        assert tp + fn == int(row["actual_positive"])
        assert abs(tp / (tp+fn) - float(row["sensitivity"])) < 1e-14
    manifest_rows = rows("results/paper_package_20260910/prediction_manifest.csv")
    manifest = {r["path"]: r for r in manifest_rows}
    assert len(manifest) == len(manifest_rows) == 4986
    for source in ("results/exhaustive_lead_subsets_1to4.csv", "results/electrode_coverage_v1/raw.csv",
                   "results/electrode_coverage_v1/stability_raw.csv"):
        for row in rows(source):
            path = row.get("prediction_path")
            if path is None:
                path = ("outputs/exhaustive_lead_subsets/" + row["run_fingerprint"] + "/"
                        + row["model"] + "_" + row["lead_indices"].replace(";", "-") + ".npz")
            assert manifest[path]["sha256"] == row["prediction_sha256"]
    archive = ROOT / "work/paper_release_20260910/p1-predictions-20260910.zip"
    with zipfile.ZipFile(archive) as handle:
        assert len(handle.namelist()) == 4987
        assert set(handle.namelist()) == set(manifest) | {"prediction_manifest.csv"}
        for path, row in manifest.items():
            assert digest(handle.read(path)) == row["sha256"]
    vf_manifest = json.loads((ROOT / "results/vfdb_readiness_20260910/source_manifest.json").read_text())
    total = 0
    for entry in vf_manifest:
        path = ROOT / "data/raw/vfdb_1.0.0" / entry["file"]
        assert digest(path.read_bytes()) == entry["sha256"]
        if path.suffix == ".hea":
            parts = path.read_text().splitlines()[0].split()
            assert parts[1:4] == ["2", "250", "525000"]
            total += int(parts[3]) / int(parts[2])
    assert total == 46200
    changed = subprocess.check_output(["git", "diff", "--name-only", "origin/main...HEAD"], cwd=ROOT).decode().splitlines()
    pattern = re.compile(rb"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{30,}|sk-(?:proj-)?[A-Za-z0-9_-]{35,}")
    hits = []
    for path in changed:
        content = (ROOT / path).read_bytes()
        if pattern.search(content):
            hits.append(path)
    assert not hits, hits
    output = {"status": "passed", "candidate_rows": len(table), "disease_rows": len(disease),
              "archive_members_hashed": len(manifest), "all_primary_csv_prediction_links_present": True,
              "vfdb_source_hashes": len(vf_manifest), "vfdb_raw_header_seconds": total,
              "outgoing_preexisting_paths_secret_pattern_checked": len(changed),
              "secret_pattern_hits": hits, "secret_scan_limit": "High-specificity key patterns; not universal privacy certification.",
              "initial_default_tests": "37 passed, 2 legacy weights-only setup errors",
              "scoped_numpy_safe_globals_tests": "39 passed in 4.01s; original checkpoint SHA verified",
              "initial_packaging_failure": "Unsupported manifest wrapper; schema handling fixed before successful packaging",
              "initial_publication_verifier_failure": "Historical CSV has fingerprint/model/subset-derived path; explicit reconstruction added",
              "visual_checks": "Both PNG figures opened; labels, units and captions legible",
              "new_training_runs": 0, "clinical_or_external_validation": False}
    (OUT / "publication_validation.json").write_text(json.dumps(output, indent=2) + "\n")
    paths = [ROOT / "records/paper_synthesis_20260910.md",
             *[ROOT / "scripts" / name for name in ("audit_vfdb_readiness.py", "build_paper_package.py", "verify_paper_publication.py")]]
    paths += [p for folder in (OUT, ROOT / "results/vfdb_readiness_20260910")
              for p in folder.iterdir() if p.is_file() and p.name != "artifact_manifest.json"]
    artifacts = {p.relative_to(ROOT).as_posix(): {"bytes": p.stat().st_size,
                 "sha256": digest(p.read_bytes())} for p in sorted(paths)}
    (OUT / "artifact_manifest.json").write_text(json.dumps(artifacts, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
