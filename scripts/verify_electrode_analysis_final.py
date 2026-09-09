"""Recompute candidate summaries and intervals using a separate NumPy metric path."""

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path("results/electrode_coverage_v1")
NAMES = ("nsr", "af", "ischemia", "conduction", "ectopy")
METRICS = ["macro_f1", "macro_sensitivity"] + [
    f"{m}_{c}" for c in NAMES[1:] for m in ("auroc", "f1", "sensitivity", "sens95")
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def table(name):
    with (ROOT / name).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def roc(truth, score):
    """Integrate tied-score cumulative counts without sklearn's ROC implementation."""
    order = np.argsort(-score, kind="stable")
    ranked = np.asarray(truth, dtype=np.int64)[order]
    ends = np.r_[np.flatnonzero(np.diff(score[order]) != 0), len(order) - 1]
    tp = np.r_[0, np.cumsum(ranked)[ends]]
    fp = np.r_[0, ends + 1 - tp[1:]]
    tpr = tp / ranked.sum()
    fpr = fp / (len(ranked) - ranked.sum())
    area = np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2)
    return float(area), float(tpr[(1 - fpr) >= 0.95].max())


def counts(truth, predicted, n):
    matrix = np.bincount(n * truth + predicted, minlength=n * n).reshape(n, n)
    tp = matrix.diagonal()
    den = matrix.sum(0) + matrix.sum(1)
    f1 = np.divide(2 * tp, den, out=np.zeros(n), where=den != 0)
    den = matrix.sum(1)
    recall = np.divide(tp, den, out=np.zeros(n), where=den != 0)
    return f1, recall


def scores(yb, ym, pb, pm):
    f1, recall = counts(ym, pm.argmax(1), 5)
    auc, sens = roc(yb, pb)
    binary_f1, _ = counts(yb, (pb >= 0.5).astype(int), 2)
    result = {
        "binary_auroc": auc,
        "sens_at_95sp": sens,
        "f1_at_05": binary_f1[1],
        "macro_f1": f1.mean(),
        "macro_sensitivity": recall.mean(),
    }
    for i, name in enumerate(NAMES):
        auc, sens = roc(ym == i, pm[:, i])
        result.update(
            {
                f"auroc_{name}": auc,
                f"sens95_{name}": sens,
                f"f1_{name}": f1[i],
                f"sensitivity_{name}": recall[i],
            }
        )
    return result


def close(actual, expected):
    np.testing.assert_allclose(float(actual), expected, atol=1e-10, rtol=0)


def verify_selection():
    raw = table("raw.csv")
    vectors = np.array([[float(r[f"sens95_{c}"]) for c in NAMES[1:]] for r in raw])
    contacts = np.array([int(r["measurement_contacts"]) for r in raw])
    frontier = set()
    for i, row in enumerate(raw):
        eligible = contacts <= contacts[i]
        weak = (vectors >= vectors[i]).all(axis=1)
        strict = (contacts < contacts[i]) | (vectors > vectors[i]).any(axis=1)
        if not (eligible & weak & strict).any():
            frontier.add(row["lead_indices"])
    assert {r["lead_indices"] for r in table("pareto.csv")} == frontier
    for r in table("summary.csv"):
        assert int(r["pareto"]) == int(r["lead_indices"] in frontier)
    winners = {}
    for n in range(2, 10):
        group = [r for r in raw if int(r["measurement_contacts"]) == n]

        def indices(r):
            return tuple(map(int, r["lead_indices"].split(";")))

        winners[n, "macro_f1"] = min(
            group, key=lambda r: (-float(r["macro_f1"]), indices(r))
        )["lead_indices"]

        def rank(r):
            values = [float(r[f"sens95_{c}"]) for c in NAMES[1:]]
            return (-min(values), -sum(values) / 4, -float(r["macro_f1"]), indices(r))

        winners[n, "worst_disease_sens95"] = min(group, key=rank)["lead_indices"]
    assert {
        (int(r["measurement_contacts"]), r["criterion"]): r["lead_indices"]
        for r in table("contact_winners.csv")
    } == winners
    expected = {v for (n, _), v in winners.items() if n <= 5}
    expected.add(";".join(map(str, range(12))))
    assert {r["lead_indices"] for r in table("shortlist.csv")} == expected
    by_key = {r["lead_indices"]: r for r in raw}
    grid = table("coverage_grid.csv")
    assert len(grid) == 4095 * 5
    assert {(r["lead_indices"], float(r["sensitivity_threshold"])) for r in grid} == {
        (k, t) for k in by_key for t in (0.5, 0.6, 0.7, 0.8, 0.9)
    }
    for r in grid:
        source = by_key[r["lead_indices"]]
        flags = [
            int(float(source[f"sens95_{c}"]) >= float(r["sensitivity_threshold"]))
            for c in NAMES[1:]
        ]
        assert [int(r[f"meets_{c}"]) for c in NAMES[1:]] == flags
        assert int(r["disease_groups_met"]) == sum(flags)
    return len(frontier)


def main():
    receipt = json.loads((ROOT / "candidate_analysis.json").read_text())
    assert receipt["status"] == "computed_pending_final_analysis_review"
    analysis = json.loads((ROOT / "analysis.json").read_text())
    audit = json.loads((ROOT / "independent_audit.json").read_text())
    assert audit["status"] == "passed" and audit["rows"] == 4095
    assert analysis["audit_sha256"] == sha(ROOT / "independent_audit.json")
    assert analysis["raw_sha256"] == audit["raw_sha256"] == sha(ROOT / "raw.csv")
    for name, info in analysis["files"].items():
        assert sha(ROOT / name) == info["sha256"]
    for name, digest in receipt["files"].items():
        assert sha(ROOT / name) == digest
    frontier_count = verify_selection()
    selected = table("shortlist.csv")
    repeated = table("stability_raw.csv")
    summary = table("stability_summary.csv")
    intervals = table("paired_intervals.csv")
    seeds = tuple(range(30000, 30010))
    keys = {r["lead_indices"] for r in selected}
    lookup = {(r["lead_indices"], int(r["seed"])): r for r in repeated}
    assert len(lookup) == len(repeated) == len(keys) * 10
    assert set(lookup) == {(k, s) for k in keys for s in seeds}
    base = Path("data/processed/cpsc2018_mc/test")
    ids = np.load(base / "record_ids.npy", allow_pickle=True).astype(str)
    ym, yb = np.load(base / "labels.npy"), np.load(base / "labels_bin.npy")

    def prediction(row):
        assert sha(row["prediction_path"]) == row["prediction_sha256"]
        with np.load(row["prediction_path"], allow_pickle=False) as z:
            for name, original in (
                ("record_ids", ids),
                ("labels_mc", ym),
                ("labels_bin", yb),
            ):
                np.testing.assert_array_equal(z[name], original)
            pb, pm = z["binary_probs"], z["multiclass_probs"]
        assert pb.shape == (936,) and pm.shape == (936, 5)
        assert all(
            np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all() for p in (pb, pm)
        )
        np.testing.assert_allclose(pm.sum(1), 1, atol=1e-6, rtol=0)
        return pb, pm

    recalculated = {}
    for key, row in lookup.items():
        assert row["repeat_run_id"] == receipt["repeat_run_id"]
        pb, pm = prediction(row)
        values = scores(yb, ym, pb, pm)
        for m, value in values.items():
            close(row[m], value)
        recalculated[key] = values
    expanded = receipt["expanded_run_id"]
    new_dir = Path("outputs/electrode_coverage_v1") / expanded / "predictions"
    assert {p.as_posix() for p in new_dir.glob("*.npz")} == {
        r["prediction_path"] for r in table("raw.csv") if r["origin"] == "new_inference"
    }
    repeat_dir = (
        Path("outputs/electrode_coverage_v1")
        / expanded
        / "stability"
        / receipt["repeat_run_id"]
    )
    assert {p.as_posix() for p in repeat_dir.glob("*.npz")} == {
        r["prediction_path"] for r in repeated
    }
    baseline = next(r for r in selected if int(r["n_leads"]) == 12)
    bk = baseline["lead_indices"]
    assert len(summary) == len(intervals) == len(keys) * len(METRICS)
    assert {(r["lead_indices"], r["metric"]) for r in summary} == {
        (k, m) for k in keys for m in METRICS
    }
    assert {(r["lead_indices"], r["metric"]) for r in intervals} == {
        (k, m) for k in keys for m in METRICS
    }
    for row in summary:
        k, m = row["lead_indices"], row["metric"]
        values = np.array([recalculated[k, s][m] for s in seeds])
        delta = values - [recalculated[bk, s][m] for s in seeds]
        expected = {
            "n": 10,
            "mean": values.mean(),
            "std": values.std(ddof=1),
            "min": values.min(),
            "max": values.max(),
            "paired_mean_delta_vs12": delta.mean(),
            "fraction_strictly_better_vs12": (delta > 0).mean(),
        }
        for m, value in expected.items():
            close(row[m], value)
    predictions = {r["lead_indices"]: prediction(r) for r in selected}
    points = {k: scores(yb, ym, *p) for k, p in predictions.items()}
    # Common record draws: evaluate each baseline realization only once.
    rng = np.random.default_rng(31415)
    differences = {k: [] for k in keys}
    skipped = 0
    for _ in range(2000):
        idx = rng.integers(0, len(ym), len(ym))
        if len(np.unique(ym[idx])) != 5 or len(np.unique(yb[idx])) != 2:
            skipped += 1
            continue
        b = scores(yb[idx], ym[idx], *(p[idx] for p in predictions[bk]))
        for k, p in predictions.items():
            c = b if k == bk else scores(yb[idx], ym[idx], *(v[idx] for v in p))
            differences[k].append([c[m] - b[m] for m in METRICS])
    interval_map = {(r["lead_indices"], r["metric"]): r for r in intervals}
    for k, data in differences.items():
        bounds = np.quantile(np.asarray(data), [0.025, 0.975], axis=0)
        for j, m in enumerate(METRICS):
            r = interval_map[k, m]
            for field, value in {
                "difference_vs12": points[k][m] - points[bk][m],
                "ci_low": bounds[0, j],
                "ci_high": bounds[1, j],
                "valid": 2000 - skipped,
                "skipped": skipped,
                "requested": 2000,
                "seed": 31415,
            }.items():
                close(r[field], value)
    events = [
        json.loads(line)
        for line in (ROOT / "candidate_events.jsonl").read_text().splitlines()
    ]
    assert not any(e["event"] == "failure" for e in events)
    output = {
        "status": "passed",
        "utc": datetime.now(timezone.utc).isoformat(),
        "run_id": receipt["expanded_run_id"],
        "repeat_run_id": receipt["repeat_run_id"],
        "verifier_sha256": sha(__file__),
        "candidate_receipt_sha256": sha(ROOT / "candidate_analysis.json"),
        "repeated_predictions": len(repeated),
        "repeated_metrics": len(repeated) * 25,
        "summary_rows": len(summary),
        "interval_rows": len(intervals),
        "pareto_rows_independently_checked": frontier_count,
        "coverage_grid_rows_checked": 20475,
        "orphan_or_missing_expanded_and_repeat_predictions": 0,
        "bootstrap_draws": 2000,
        "bootstrap_seed": 31415,
        "skipped": skipped,
        "method": "Separate NumPy tied-rank ROC and bincount confusion path; shared paired draws",
        "scope": "Numerical verification only; exploratory selection bias remains",
    }
    (ROOT / "final_analysis_verification.json").write_text(
        json.dumps(output, indent=2) + "\n"
    )
    print(json.dumps(output))


if __name__ == "__main__":
    main()
