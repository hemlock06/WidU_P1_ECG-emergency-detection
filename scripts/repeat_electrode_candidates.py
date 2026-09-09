"""Preserve every frozen-shortlist feature-mask realization and paired record interval."""

import argparse
import hashlib
import json
import os
import time
import traceback
from pathlib import Path

import analyze_electrode_coverage as a
import numpy as np

ROOT = a.ROOT
SEEDS = tuple(range(30000, 30010))
BOOTSTRAP_SEED = 31415
BOOTSTRAP_DRAWS = 2000


def event(kind, **details):
    import evaluate_electrode_coverage as runner

    with (ROOT / "candidate_events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"utc": runner.now(), "event": kind, **details}) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def inputs():
    import evaluate_electrode_coverage as runner

    identity, run_id = runner.identity()
    metadata = json.loads((ROOT / "analysis.json").read_text())
    if (
        metadata["status"] != "selection_complete_pending_stability_and_bootstrap"
        or metadata["run_id"] != run_id
        or metadata["raw_sha256"] != a.digest(ROOT / "raw.csv")
        or metadata["script_sha256"] != a.digest(Path(a.__file__))
        or metadata["audit_sha256"] != a.digest(ROOT / "independent_audit.json")
    ):
        raise ValueError("Matching audited candidate selection required")
    for name, info in metadata["files"].items():
        if a.digest(ROOT / name) != info["sha256"]:
            raise ValueError(f"Changed selection artifact: {name}")
    rows = a.read_csv(ROOT / "shortlist.csv")
    source = {
        "expanded_run_id": run_id,
        "analysis_sha256": a.digest(ROOT / "analysis.json"),
        "script_sha256": a.digest(__file__),
        "seeds": SEEDS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "parent_identity": identity,
    }
    repeat_id = hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()
    return rows, source, repeat_id


def source_labels():
    base = Path("data/processed/cpsc2018_mc/test")
    return (
        np.load(base / "record_ids.npy", allow_pickle=True).astype(str),
        np.load(base / "labels_bin.npy"),
        np.load(base / "labels.npy"),
    )


def repeats():
    import csv

    import evaluate_electrode_coverage as runner

    with runner.exclusive_run():
        selected, source, repeat_id = inputs()
        x, ids, yb, ym = runner.e.data_for("P1_a07")
        path = ROOT / "stability_raw.csv"
        rows = a.read_csv(path) if path.exists() else []
        expected = {(r["lead_indices"], seed) for r in selected for seed in SEEDS}
        done = {(r["lead_indices"], int(r["seed"])) for r in rows}
        if len(done) != len(rows) or not done <= expected:
            raise ValueError("Duplicate or unknown repeated configurations")
        for row in rows:
            if row["repeat_run_id"] != repeat_id:
                raise ValueError("Changed repeat identity; preserve existing results")
            pb, pm = a.check_prediction(row, ids, yb, ym)
            verify_scores(row, yb, pb, ym, pm)
        event(
            "repeat_started",
            repeat_run_id=repeat_id,
            resumed=len(rows),
            expected=len(expected),
        )
        model = runner.e.model_for("P1_a07")
        pred_dir = (
            Path("outputs/electrode_coverage_v1")
            / source["expanded_run_id"]
            / "stability"
            / repeat_id
        )
        pred_dir.mkdir(parents=True, exist_ok=True)
        for seed in SEEDS:
            for candidate in selected:
                if (candidate["lead_indices"], seed) in done:
                    continue
                start = time.perf_counter()
                indices = a.combo(candidate)
                pb, pm = runner.e.predict(model, x, indices, 32, seed)
                scores = runner.e.metrics(yb, pb, ym, pm)
                verify_scores(scores, yb, pb, ym, pm)
                saved = pred_dir / f"seed{seed}_{'-'.join(map(str, indices))}.npz"
                runner.save_prediction(saved, ids, yb, ym, pb, pm)
                row = {
                    "lead_indices": candidate["lead_indices"],
                    "lead_names": candidate["lead_names"],
                    "measurement_contacts": candidate["measurement_contacts"],
                    "seed": seed,
                    **scores,
                    "repeat_run_id": repeat_id,
                    "expanded_run_id": source["expanded_run_id"],
                    "prediction_path": saved.as_posix(),
                    "prediction_sha256": a.digest(saved),
                    "runtime_sec": time.perf_counter() - start,
                }
                with path.open("a", encoding="utf-8", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(row))
                    if not rows:
                        writer.writeheader()
                    writer.writerow(row)
                    stream.flush()
                    os.fsync(stream.fileno())
                rows.append(row)
                done.add((candidate["lead_indices"], seed))
                runner.atomic_json(
                    ROOT / "stability_status.json",
                    {
                        "status": "running",
                        "repeat_run_id": repeat_id,
                        "rows": len(rows),
                        "expected": len(expected),
                        "utc": runner.now(),
                    },
                )
                print("stability", len(rows), "/", len(expected), flush=True)
        if done != expected:
            raise ValueError("Incomplete repeated experiment")
        receipt = {
            "status": "inference_complete_pending_summary_audit",
            "repeat_run_id": repeat_id,
            "identity": source,
            "rows": len(rows),
            "expected": len(expected),
            "raw_sha256": a.digest(path),
            "utc": runner.now(),
        }
        runner.atomic_json(ROOT / "stability_status.json", receipt)
        event("repeat_inference_complete", **receipt)
        return receipt


def verify_scores(row, yb, pb, ym, pm):
    for metric, value in a.independent_metrics(yb, pb, ym, pm).items():
        difference = abs(float(row[metric]) - value)
        if not np.isfinite(difference) or difference > 1e-7:
            raise ValueError(f"Repeated metric mismatch: {metric}")


def paired_intervals(
    yb, ym, candidate, baseline, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED
):
    """Record bootstrap; all candidates use the same draws; absent-class draws are skipped."""
    rng = np.random.default_rng(seed)
    original = a.independent_metrics(yb, candidate[0], ym, candidate[1])
    reference = a.independent_metrics(yb, baseline[0], ym, baseline[1])
    differences = []
    skipped = 0
    for _ in range(draws):
        index = rng.integers(0, len(ym), len(ym))
        if len(np.unique(ym[index])) != 5 or len(np.unique(yb[index])) != 2:
            skipped += 1
            continue
        c = a.independent_metrics(
            yb[index], candidate[0][index], ym[index], candidate[1][index]
        )
        b = a.independent_metrics(
            yb[index], baseline[0][index], ym[index], baseline[1][index]
        )
        differences.append([c[m] - b[m] for m in a.METRICS])
    if not differences:
        raise ValueError("No valid bootstrap draws; do not reroll")
    values = np.asarray(differences)
    return [
        {
            "metric": metric,
            "difference_vs12": original[metric] - reference[metric],
            "ci_low": float(np.quantile(values[:, i], 0.025)),
            "ci_high": float(np.quantile(values[:, i], 0.975)),
            "valid": len(differences),
            "skipped": skipped,
            "requested": draws,
            "seed": seed,
        }
        for i, metric in enumerate(a.METRICS)
    ]


def finish():
    import evaluate_electrode_coverage as runner

    with runner.exclusive_run():
        selected, source, repeat_id = inputs()
        ids, yb, ym = source_labels()
        status = json.loads((ROOT / "stability_status.json").read_text())
        if (
            status["repeat_run_id"] != repeat_id
            or status["status"] != "inference_complete_pending_summary_audit"
            or status["raw_sha256"] != a.digest(ROOT / "stability_raw.csv")
        ):
            raise ValueError("Verified complete repeats required")
        repeated = a.read_csv(ROOT / "stability_raw.csv")
        expected = {(r["lead_indices"], s) for r in selected for s in SEEDS}
        lookup = {(r["lead_indices"], int(r["seed"])): r for r in repeated}
        if len(lookup) != len(repeated) or set(lookup) != expected:
            raise ValueError("Repeated table incomplete or duplicated")
        for row in repeated:
            if row["repeat_run_id"] != repeat_id:
                raise ValueError("Repeat identity mismatch")
            pb, pm = a.check_prediction(row, ids, yb, ym)
            verify_scores(row, yb, pb, ym, pm)
        baseline = next(r for r in selected if len(a.combo(r)) == 12)
        baseline_prediction = a.check_prediction(baseline, ids, yb, ym)
        summary, intervals = [], []
        for candidate in selected:
            key = candidate["lead_indices"]
            for metric in a.METRICS:
                values = np.array([float(lookup[key, seed][metric]) for seed in SEEDS])
                reference = np.array(
                    [
                        float(lookup[baseline["lead_indices"], seed][metric])
                        for seed in SEEDS
                    ]
                )
                delta = values - reference
                summary.append(
                    {
                        "lead_indices": key,
                        "lead_names": candidate["lead_names"],
                        "metric": metric,
                        "n": len(SEEDS),
                        "mean": float(values.mean()),
                        "std": float(values.std(ddof=1)),
                        "min": float(values.min()),
                        "max": float(values.max()),
                        "paired_mean_delta_vs12": float(delta.mean()),
                        "fraction_strictly_better_vs12": float((delta > 0).mean()),
                    }
                )
            probabilities = a.check_prediction(candidate, ids, yb, ym)
            for result in paired_intervals(yb, ym, probabilities, baseline_prediction):
                intervals.append(
                    {
                        "lead_indices": key,
                        "lead_names": candidate["lead_names"],
                        **result,
                    }
                )
            print("bootstrap", candidate["lead_names"], "complete", flush=True)
        (ROOT / "stability_summary.csv").write_bytes(a.csv_bytes(summary))
        (ROOT / "paired_intervals.csv").write_bytes(a.csv_bytes(intervals))
        receipt = {
            "status": "computed_pending_final_analysis_review",
            "repeat_run_id": repeat_id,
            "expanded_run_id": source["expanded_run_id"],
            "script_sha256": a.digest(__file__),
            "prediction_files_independently_checked": len(repeated),
            "summary_rows": len(summary),
            "interval_rows": len(intervals),
            "skips_by_candidate": {r["lead_indices"]: r["skipped"] for r in intervals},
            "files": {
                name: a.digest(ROOT / name)
                for name in (
                    "stability_raw.csv",
                    "stability_summary.csv",
                    "paired_intervals.csv",
                )
            },
            "utc": runner.now(),
            "scope": "Exploratory record and feature-mask uncertainty; no independent-cohort claim",
        }
        a.write_json(ROOT / "candidate_analysis.json", receipt)
        return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["repeat", "finish"], required=True)
    args = parser.parse_args()
    try:
        print(json.dumps((repeats if args.stage == "repeat" else finish)()))
    except Exception:
        event("failure", stage=args.stage, traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
