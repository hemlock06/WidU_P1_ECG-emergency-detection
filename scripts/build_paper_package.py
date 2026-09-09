"""Build traceable manuscript tables/figures and a raw-prediction release archive."""
import csv
import hashlib
import json
import zipfile
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/paper_package_20260910"
ARCHIVE = ROOT / "work/paper_release_20260910"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return pd.read_csv(ROOT / path, dtype={"lead_indices": str})


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    selected = read("results/electrode_coverage_v1/shortlist.csv")
    counts = read("results/classification_counts_20260910/cohort_accuracy.csv")
    disease = read("results/classification_counts_20260910/disease_counts.csv")
    summary = read("results/electrode_coverage_v1/stability_summary.csv")
    intervals = read("results/electrode_coverage_v1/paired_intervals.csv")
    table = selected[["lead_indices", "lead_names", "measurement_contacts",
                      "contacts_with_extra_drl", "macro_f1", "checkpoint_sha256",
                      "data_fingerprint", "prediction_sha256"]].merge(
        counts[counts.stage == "seed42"][["lead_indices", "n", "correct", "incorrect", "accuracy"]],
        on="lead_indices", validate="one_to_one")
    mean = summary[summary.metric == "macro_f1"][["lead_indices", "mean", "std"]].rename(
        columns={"mean": "repeat_mean_macro_f1", "std": "repeat_sd_macro_f1"})
    ci = intervals[intervals.metric == "macro_f1"][["lead_indices", "difference_vs12", "ci_low", "ci_high"]]
    table = table.merge(mean, on="lead_indices", validate="one_to_one").merge(
        ci, on="lead_indices", validate="one_to_one").sort_values(["measurement_contacts", "lead_names"])
    assert len(table) == 8 and len(counts) == 4175 and len(disease) == 20875
    assert np.allclose(table.correct / table.n, table.accuracy, atol=1e-14)
    table.to_csv(OUT / "candidate_table.csv", index=False, lineterminator="\n")
    dc = disease[(disease.stage == "seed42") & disease.lead_indices.isin(table.lead_indices)]
    assert len(dc) == 40
    dc.to_csv(OUT / "candidate_disease_counts.csv", index=False, lineterminator="\n")
    labels = [f"{int(r.measurement_contacts)} contacts | " +
              ("12 leads" if r.measurement_contacts == 9 else r.lead_names.replace(";", "+"))
              for r in table.itertuples()]
    plt.rcParams.update({"font.size": 10, "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(10, 5.5))
    y = np.arange(8)
    ax.errorbar(table.difference_vs12, y,
                xerr=np.stack([table.difference_vs12-table.ci_low,
                               table.ci_high-table.difference_vs12]),
                fmt="o", capsize=4, color="#176B87")
    ax.axvline(0, color="gray", linestyle="--")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Macro-F1 difference vs 12 leads (seed 42)")
    ax.set_title("Paired record bootstrap: exploratory 95% intervals")
    ax.grid(axis="x", alpha=.2)
    fig.text(.5, .01, "Same selection cohort; unadjusted intervals. Includes normal in five-class Macro-F1.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .035, 1, 1))
    for ext in ("svg", "png"):
        fig.savefig(OUT / f"macro_f1_intervals.{ext}", dpi=180)
    plt.close(fig)
    metrics = ["sens95_af", "sens95_ischemia", "sens95_conduction", "sens95_ectopy"]
    heat = summary.pivot(index="lead_indices", columns="metric", values="mean").loc[
        table.lead_indices, metrics].to_numpy()
    assert heat.shape == (8, 4) and np.isfinite(heat).all()
    fig, ax = plt.subplots(figsize=(10, 5.8))
    im = ax.imshow(heat, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
    ax.set_yticks(y, labels)
    ax.set_xticks(range(4), ["AF", "ST-change surrogate", "Conduction", "Ectopy"])
    for i in range(8):
        for j in range(4):
            ax.text(j, i, f"{heat[i,j]:.3f}", ha="center", va="center",
                    color="white" if heat[i,j] > .7 else "black")
    ax.set_title("Ten feature-mask seeds: mean empirical Sens@95Sp")
    fig.colorbar(im, ax=ax, label="Sensitivity", shrink=.8)
    fig.text(.5, .01, "Operating points derived on the same test; no wearable or independent-cohort claim.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .035, 1, 1))
    for ext in ("svg", "png"):
        fig.savefig(OUT / f"disease_sensitivity.{ext}", dpi=180)
    plt.close(fig)
    files = sorted([*ROOT.glob("outputs/exhaustive_lead_subsets/**/*.npz"),
                    *ROOT.glob("outputs/electrode_coverage_v1/**/*.npz")])
    assert len(files) >= 4986
    rows = []
    allowed = {"record_ids", "labels_bin", "labels_mc", "binary_probs", "multiclass_probs"}
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            assert set(data.files) <= allowed, (path, data.files)
            for key in data.files:
                value = data[key]
                if key == "record_ids":
                    assert all(str(v).startswith("A") and str(v)[1:].isdigit() for v in value)
                else:
                    assert value.dtype.kind in "fiu" and np.isfinite(value).all()
        rows.append({"path": path.relative_to(ROOT).as_posix(),
                     "bytes": path.stat().st_size, "sha256": sha(path)})
    with (OUT / "prediction_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    archive = ARCHIVE / "p1-predictions-20260910.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as handle:
        for row in rows:
            handle.write(ROOT / row["path"], row["path"])
        handle.write(OUT / "prediction_manifest.csv", "prediction_manifest.csv")
    with zipfile.ZipFile(archive) as handle:
        assert handle.testzip() is None
        for row in rows:
            assert hashlib.sha256(handle.read(row["path"])).hexdigest() == row["sha256"]
    evidence = {}
    for directory in ("electrode_coverage_v1", "stage2_lineage", "pretraining_exposure_20260910",
                      "classification_counts_20260910"):
        manifest_path = ROOT / "results" / directory / "artifact_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if "files" in manifest:
            manifest = {entry["path"]: entry for entry in manifest["files"]}
        for path, item in manifest.items():
            assert sha(ROOT / path) == item["sha256"], path
        evidence[directory] = len(manifest)
    result = {"status": "passed", "candidate_rows": 8, "disease_rows": 40,
              "prior_manifest_files_checked": evidence, "raw_prediction_files": len(rows),
              "raw_prediction_bytes": sum(r["bytes"] for r in rows),
              "archive": archive.relative_to(ROOT).as_posix(), "archive_bytes": archive.stat().st_size,
              "archive_sha256": sha(archive), "archive_member_hash_checks": len(rows),
              "new_model_inferences": 0, "new_training_runs": 0,
              "scope": "Publication packaging and source-byte verification; prior numerical audits retained."}
    (OUT / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
