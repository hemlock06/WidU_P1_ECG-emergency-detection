"""Audit complete electrode inference and render the frozen exploratory selection."""

import argparse
import csv
import io
import itertools
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from verify_exhaustive_lead_results import (
    CLASSES,
    LEADS,
    confusion_values,
    curve_values,
    digest,
)

ROOT = Path("results/electrode_coverage_v1")
DISEASES = CLASSES[1:]
GRID = (0.5, 0.6, 0.7, 0.8, 0.9)
METRICS = ["macro_f1", "macro_sensitivity"] + [
    f"{metric}_{name}"
    for name in DISEASES
    for metric in ("auroc", "f1", "sensitivity", "sens95")
]
INCIDENCE = [
    {"RA", "LA"},
    {"RA", "LL"},
    {"LA", "LL"},
    *[{"RA", "LA", "LL"} for _ in range(3)],
    *[{"RA", "LA", "LL", f"V{i}"} for i in range(1, 7)],
]


def combo(row):
    return tuple(map(int, row["lead_indices"].split(";")))


def contact_set(indices):
    return set().union(*(INCIDENCE[i] for i in indices))


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def csv_bytes(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def independent_metrics(yb, pb, ym, pm):
    """Use full ROC and confusion counts, not the inference runner's metric function."""
    auroc, sensitivity = curve_values(yb, pb)
    binary_f1, _ = confusion_values(yb, pb >= 0.5, [0, 1])
    f1, recall = confusion_values(ym, pm.argmax(1), list(range(5)))
    values = {
        "binary_auroc": auroc,
        "sens_at_95sp": sensitivity,
        "f1_at_05": binary_f1[1],
        "macro_f1": f1.mean(),
        "macro_sensitivity": recall.mean(),
    }
    for index, name in enumerate(CLASSES):
        auroc, sensitivity = curve_values(ym == index, pm[:, index])
        values.update(
            {
                f"auroc_{name}": auroc,
                f"sens95_{name}": sensitivity,
                f"f1_{name}": f1[index],
                f"sensitivity_{name}": recall[index],
            }
        )
    return values


def check_prediction(row, ids, yb, ym):
    path = Path(row["prediction_path"])
    if digest(path) != row["prediction_sha256"]:
        raise ValueError(f"Prediction hash mismatch: {path}")
    with np.load(path, allow_pickle=False) as saved:
        for name, expected in (
            ("record_ids", ids),
            ("labels_bin", yb),
            ("labels_mc", ym),
        ):
            np.testing.assert_array_equal(saved[name], expected)
        pb, pm = saved["binary_probs"], saved["multiclass_probs"]
    if pb.shape != (len(ids),) or pm.shape != (len(ids), 5):
        raise ValueError("Invalid prediction dimensions")
    for probabilities in (pb, pm):
        if (
            not np.isfinite(probabilities).all()
            or np.any(probabilities < 0)
            or np.any(probabilities > 1)
        ):
            raise ValueError("Invalid probabilities")
    np.testing.assert_allclose(pm.sum(1), 1, atol=1e-6, rtol=0)
    return pb, pm


def validate_rows(rows, run_id):
    expected = {c for n in range(1, 13) for c in itertools.combinations(range(12), n)}
    observed = [combo(r) for r in rows]
    if len(rows) != 4095 or len(set(observed)) != 4095 or set(observed) != expected:
        raise ValueError("Expected exactly 4095 unique nonempty subsets")
    for row, indices in zip(rows, observed):
        points = contact_set(indices)
        expected_origin = (
            "prior_exhaustive"
            if len(indices) <= 4
            else ("prior_control" if len(indices) == 12 else "new_inference")
        )
        if (
            row["stage1_run_id"] != run_id
            or row["model"] != "P1_a07"
            or row["origin"] != expected_origin
            or int(row["seed"]) != 42
            or row["measurable"] != "binary;multiclass"
            or int(row["n_samples"]) != 936
            or int(row["n_leads"]) != len(indices)
            or row["lead_names"] != ";".join(LEADS[i] for i in indices)
            or row["contact_names"] != ";".join(sorted(points))
            or int(row["measurement_contacts"]) != len(points)
            or int(row["contacts_with_extra_drl"]) != len(points) + 1
        ):
            raise ValueError(f"Invalid row metadata: {indices}")
        values = np.array([float(row[m]) for m in METRICS])
        if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
            raise ValueError("Invalid primary metric")


def audit():
    # The runner's OS lock prevents auditing a concurrently appended result as complete.
    import evaluate_electrode_coverage as runner

    with runner.exclusive_run():
        status = json.loads((ROOT / "status.json").read_text())
        if status["status"] != "inference_complete_pending_independent_audit":
            raise ValueError("Inference must finish before the full audit")
        identity, run_id = runner.identity()
        gate = json.loads((ROOT / "preflight.json").read_text())
        if (
            gate["run_id"] != run_id
            or gate["identity"] != identity
            or status["run_id"] != run_id
            or gate["status"] != "passed"
            or len(gate["checks"]) != 6
        ):
            raise ValueError("Preflight identity mismatch")
        source = ROOT / "raw.csv"
        raw_hash = digest(source)
        if raw_hash != status["raw_sha256"] or status["rows"] != 4095:
            raise ValueError("Completed raw digest/count mismatch")
        rows = read_csv(source)
        validate_rows(rows, run_id)
        base = Path("data/processed/cpsc2018_mc/test")
        ids = np.load(base / "record_ids.npy", allow_pickle=True).astype(str)
        ym = np.load(base / "labels.npy", allow_pickle=False)
        yb = np.load(base / "labels_bin.npy", allow_pickle=False)
        if len(set(ids)) != 936 or np.bincount(ym, minlength=5).tolist() != [
            130,
            180,
            165,
            379,
            82,
        ]:
            raise ValueError("Historical cohort mismatch")
        parent_rows = runner.imported_rows(ids, yb, ym, run_id)
        parent = {r["lead_indices"]: r for r in parent_rows}
        maximum, count = 0.0, 0
        source_identity = identity["parent_identity"]["sha256"]
        checkpoint = source_identity[
            "outputs/lora_multitask_snr_a07/lora_multitask_snr_best.pt"
        ]
        for row in rows:
            if (
                row["checkpoint_sha256"] != checkpoint
                or row["run_fingerprint"] != runner.PARENT
                or row["data_fingerprint"] != parent_rows[0]["data_fingerprint"]
            ):
                raise ValueError("Row model/data identity mismatch")
            if row["origin"] != "new_inference":
                if {k: str(v) for k, v in parent[row["lead_indices"]].items()} != row:
                    raise ValueError("Imported row changed from its parent")
            else:
                expected_path = (
                    Path("outputs/electrode_coverage_v1")
                    / run_id
                    / "predictions"
                    / ("-".join(map(str, combo(row))) + ".npz")
                )
                if Path(row["prediction_path"]) != expected_path:
                    raise ValueError("Unexpected new prediction path")
            pb, pm = check_prediction(row, ids, yb, ym)
            for metric, value in independent_metrics(yb, pb, ym, pm).items():
                error = abs(float(row[metric]) - value)
                if not np.isfinite(error) or error > 1e-7:
                    raise ValueError(
                        f"Independent metric mismatch: {row['lead_indices']} {metric}"
                    )
                maximum = max(maximum, error)
                count += 1
        for control in gate["checks"]:
            if control["status"] != "passed":
                raise ValueError("Failed preflight control")
            pb, pm = check_prediction(control, ids, yb, ym)
            for metric, value in independent_metrics(yb, pb, ym, pm).items():
                if abs(float(control["metrics"][metric]) - value) > 1e-7:
                    raise ValueError("Preflight metric mismatch")
        expected_controls = {runner.key(c) for c in runner.e.CONTROLS}
        expected_controls.update({"0;1;2;3;4;5", "0;1;2;3;4;5;7"})
        if {c["leads"] for c in gate["checks"]} != expected_controls:
            raise ValueError("Missing required preflight configuration")
        events = [
            json.loads(line)
            for line in (ROOT / "events.jsonl").read_text().splitlines()
        ]
        failures = [event for event in events if event["event"] == "failure"]
        if failures or status["failures_this_invocation"]:
            raise ValueError(
                "Failure events need explicit investigation before progression"
            )
        interruptions = [
            event
            for event in events
            if event["event"] == "external_interruption_observed"
        ]
        for interruption in interruptions:
            evidence_path = Path(interruption["resume_audit_path"])
            if digest(evidence_path) != interruption["resume_audit_sha256"]:
                raise ValueError("Resume audit evidence changed")
            evidence = json.loads(evidence_path.read_text())
            if (
                evidence["status"] != "passed_for_resume"
                or evidence["run_id"] != run_id
            ):
                raise ValueError(
                    "Interruption lacks a matching successful resume audit"
                )
        if interruptions:
            parity = json.loads((ROOT / "restart_parity.json").read_text())
            if (
                parity["status"] != "passed"
                or parity["run_id"] != run_id
                or parity["raw_sha256"] != raw_hash
                or parity["script_sha256"]
                != digest(Path("scripts/verify_electrode_restart_parity.py"))
            ):
                raise ValueError(
                    "Restart-boundary replay must pass before the final audit"
                )
            expected_replay = {
                rows[evidence["saved_rows"] - 1]["lead_indices"],
                rows[evidence["saved_rows"]]["lead_indices"],
                ";".join(map(str, range(12))),
            }
            if (
                len(parity["checks"]) != 3
                or {c["leads"] for c in parity["checks"]} != expected_replay
            ):
                raise ValueError("Missing restart-boundary replay condition")
            by_combo = {r["lead_indices"]: r for r in rows}
            for check in parity["checks"]:
                original = by_combo[check["leads"]]
                if (
                    check["status"] != "passed"
                    or check["source_prediction_sha256"]
                    != original["prediction_sha256"]
                ):
                    raise ValueError("Restart replay source mismatch")
                pb, pm = check_prediction(check, ids, yb, ym)
                old_pb, old_pm = check_prediction(original, ids, yb, ym)
                np.testing.assert_allclose(pb, old_pb, atol=1e-6, rtol=1e-6)
                np.testing.assert_allclose(pm, old_pm, atol=1e-6, rtol=1e-6)
                np.testing.assert_array_equal(pm.argmax(1), old_pm.argmax(1))
        starts = [event for event in events if event["event"] == "run_started"]
        ends = [event for event in events if event["event"] == "inference_complete"]
        if not starts or len(ends) != 1 or ends[0]["raw_sha256"] != raw_hash:
            raise ValueError("Missing or ambiguous completion provenance")
        if status["reused_parent_rows"] != 794 or (
            status["resumed_new_rows"] + status["new_rows_this_invocation"] != 3301
        ):
            raise ValueError("Incorrect import/resume/new accounting")
        if digest(source) != raw_hash:
            raise ValueError("Source changed during audit")
        receipt = {
            "status": "passed",
            "run_id": run_id,
            "raw_sha256": raw_hash,
            "preflight_sha256": digest(ROOT / "preflight.json"),
            "auditor_sha256": digest(__file__),
            "metric_helper_sha256": digest(
                Path("scripts/verify_exhaustive_lead_results.py")
            ),
            "utc": datetime.now(timezone.utc).isoformat(),
            "rows": len(rows),
            "origins": dict(Counter(r["origin"] for r in rows)),
            "contact_counts": dict(
                Counter(int(r["measurement_contacts"]) for r in rows)
            ),
            "metrics_checked": count,
            "maximum_absolute_error": maximum,
            "prediction_files_checked": len(rows),
            "preflight_files_checked": 6,
            "failure_events": 0,
            "externally_observed_interruptions": len(interruptions),
            "run_invocations": len(starts),
            "resumed_new_rows_final_invocation": status["resumed_new_rows"],
            "scope": "Independent metric/data audit; not independent cohort validation",
        }
        write_json(ROOT / "independent_audit.json", receipt)
        return receipt


def enrich(rows):
    result = []
    for row in sorted(rows, key=combo):
        vector = [float(row[f"sens95_{d}"]) for d in DISEASES]
        available = contact_set(combo(row))
        closure = tuple(
            i for i, requirement in enumerate(INCIDENCE) if requirement <= available
        )
        result.append(
            {
                **row,
                "worst_disease_sens95": min(vector),
                "mean_disease_sens95": sum(vector) / 4,
                "complete_available_leads": int(closure == combo(row)),
            }
        )
    return result


def pareto_indices(rows):
    # A point dominates only if every objective is no worse and at least one is better.
    objectives = np.array(
        [
            [-int(r["measurement_contacts"])]
            + [float(r[f"sens95_{d}"]) for d in DISEASES]
            for r in rows
        ]
    )
    return [
        i
        for i, value in enumerate(objectives)
        if not np.any(
            np.all(objectives >= value, axis=1) & np.any(objectives > value, axis=1)
        )
    ]


def selections(rows):
    winners = []
    for contacts in sorted({int(r["measurement_contacts"]) for r in rows}):
        group = [r for r in rows if int(r["measurement_contacts"]) == contacts]
        # Macro-F1 exact ties use lexicographic indices; worst-disease ties follow protocol.
        macro = min(group, key=lambda r: (-float(r["macro_f1"]), combo(r)))
        worst = min(
            group,
            key=lambda r: (
                -float(r["worst_disease_sens95"]),
                -float(r["mean_disease_sens95"]),
                -float(r["macro_f1"]),
                combo(r),
            ),
        )
        winners.extend(
            [
                {**macro, "criterion": "macro_f1"},
                {**worst, "criterion": "worst_disease_sens95"},
            ]
        )
    by_key = {}
    for row in winners:
        if int(row["measurement_contacts"]) <= 5:
            by_key.setdefault(row["lead_indices"], []).append(row["criterion"])
    baseline = next(r for r in rows if len(combo(r)) == 12)
    by_key.setdefault(baseline["lead_indices"], []).append("12_lead_baseline")
    shortlist = [
        {**r, "selection_reasons": ";".join(by_key[r["lead_indices"]])}
        for r in rows
        if r["lead_indices"] in by_key
    ]
    return winners, shortlist


def products(raw):
    rows = enrich(raw)
    winners, shortlist = selections(rows)
    front = set(pareto_indices(rows))
    all_rows = [{**r, "pareto": int(i in front)} for i, r in enumerate(rows)]
    grid = []
    for row in rows:
        for threshold in GRID:
            flags = {
                f"meets_{d}": int(float(row[f"sens95_{d}"]) >= threshold)
                for d in DISEASES
            }
            grid.append(
                {
                    "lead_indices": row["lead_indices"],
                    "lead_names": row["lead_names"],
                    "measurement_contacts": row["measurement_contacts"],
                    "sensitivity_threshold": threshold,
                    **flags,
                    "disease_groups_met": sum(flags.values()),
                }
            )
    return {
        "summary.csv": all_rows,
        "pareto.csv": [r for r in all_rows if r["pareto"]],
        "contact_winners.csv": winners,
        "shortlist.csv": shortlist,
        "complete_physical_configurations.csv": [
            r for r in rows if r["complete_available_leads"]
        ],
        "coverage_grid.csv": grid,
    }


def analyze():
    raw_hash = digest(ROOT / "raw.csv")
    receipt = json.loads((ROOT / "independent_audit.json").read_text())
    if (
        receipt["status"] != "passed"
        or receipt["raw_sha256"] != raw_hash
        or receipt["auditor_sha256"] != digest(__file__)
        or receipt["metric_helper_sha256"]
        != digest(Path("scripts/verify_exhaustive_lead_results.py"))
        or receipt["preflight_sha256"] != digest(ROOT / "preflight.json")
    ):
        raise ValueError("Matching independent audit required")
    raw = read_csv(ROOT / "raw.csv")
    validate_rows(raw, receipt["run_id"])
    generated = products(raw)
    for name, values in generated.items():
        (ROOT / name).write_bytes(csv_bytes(values))
    # Verify deterministic regeneration from the retained input, then independent selection rules.
    for name, values in products(read_csv(ROOT / "raw.csv")).items():
        if (ROOT / name).read_bytes() != csv_bytes(values):
            raise ValueError(f"Summary regeneration mismatch: {name}")
    for winner in generated["contact_winners.csv"]:
        group = [
            r
            for r in raw
            if int(r["measurement_contacts"]) == int(winner["measurement_contacts"])
        ]
        if winner["criterion"] == "macro_f1":
            assert float(winner["macro_f1"]) == max(float(r["macro_f1"]) for r in group)
        else:
            assert float(winner["worst_disease_sens95"]) == max(
                min(float(r[f"sens95_{d}"]) for d in DISEASES) for r in group
            )
    result = {
        "status": "selection_complete_pending_stability_and_bootstrap",
        "run_id": receipt["run_id"],
        "raw_sha256": raw_hash,
        "script_sha256": digest(__file__),
        "audit_sha256": digest(ROOT / "independent_audit.json"),
        "files": {
            name: {"sha256": digest(ROOT / name), "rows": len(rows)}
            for name, rows in generated.items()
        },
        "grid": GRID,
        "shortlist_count": len(generated["shortlist.csv"]),
        "macro_exact_tie_rule": "lexicographic lead indices",
        "scope": "Exploratory selection; test-selected ROC thresholds; not clinical coverage",
    }
    write_json(ROOT / "analysis.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["audit", "analyze"], required=True)
    args = parser.parse_args()
    print(json.dumps((audit if args.stage == "audit" else analyze)()))


if __name__ == "__main__":
    main()
