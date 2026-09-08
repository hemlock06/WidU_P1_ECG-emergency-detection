"""Assess shortlisted P1 leads across 10 predeclared feature-mask seeds."""

import csv
import json
import time
from pathlib import Path

import ablation_exhaustive_lead_subsets as e
import numpy as np
import torch
from summarize_exhaustive_lead_subsets import indices, select_candidates, validate

# Fixed before exhaustive results/candidate selection, independent of controls and seed 42.
SEEDS = tuple(range(20000, 20010))


def main():
    root = Path("results")
    raw_path = root / "exhaustive_lead_subsets_1to4.csv"
    with raw_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    validate(rows)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    _, fingerprint = e.asset_identity(42, 32)
    if rows[0]["run_fingerprint"] != fingerprint:
        raise ValueError("Candidate input identity mismatch")
    candidates = select_candidates(rows)
    choices = sorted({indices(row) for row in candidates.values()} | {tuple(range(12))})
    x, _, yb, ym = e.data_for("P1_a07")
    model = e.model_for("P1_a07")
    realized = {}
    start = time.perf_counter()
    raw_out = root / "exhaustive_lead_candidates_stability_raw.csv"
    with raw_out.open("w", newline="", encoding="utf-8") as stream:
        writer = None
        for seed in SEEDS:
            for leads in choices:
                pb, pm = e.predict(model, x, leads, 32, seed)
                scores = e.metrics(yb, pb, ym, pm)
                realized[(leads, seed)] = scores
                row = {
                    "leads": ";".join(e.LEAD_NAMES[i] for i in leads),
                    "seed": seed,
                    **scores,
                }
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=row.keys())
                    writer.writeheader()
                writer.writerow(row)
                stream.flush()
                print(row["leads"], seed, scores["macro_f1"], flush=True)
    summary = []
    keys = ["macro_f1", "macro_sensitivity"] + [
        f"{m}_{c}"
        for c in e.CLASSES[1:]
        for m in ("auroc", "f1", "sensitivity", "sens95")
    ]
    for category, chosen in candidates.items():
        leads = indices(chosen)
        for key in keys:
            values = np.array([realized[(leads, s)][key] for s in SEEDS])
            baseline = np.array([realized[(tuple(range(12)), s)][key] for s in SEEDS])
            diff = values - baseline
            summary.append(
                {
                    "category": category,
                    "leads": chosen["lead_names"],
                    "metric": key,
                    "seeds": len(SEEDS),
                    "mean": values.mean(),
                    "std": values.std(ddof=1),
                    "min": values.min(),
                    "max": values.max(),
                    "paired_mean_delta_vs12": diff.mean(),
                    "paired_min_delta_vs12": diff.min(),
                    "paired_max_delta_vs12": diff.max(),
                    "fraction_strictly_better_vs12": float(np.mean(diff > 0)),
                }
            )
    with (root / "exhaustive_lead_candidates_stability_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)
    metadata = {
        "seeds_predeclared": SEEDS,
        "run_fingerprint": fingerprint,
        "raw_search_sha256": e.sha256(raw_path),
        "script_sha256": e.sha256(Path(__file__)),
        "unique_lead_configurations_including12": len(choices),
        "realizations": len(realized),
        "runtime_sec": time.perf_counter() - start,
        "failures": 0,
        "skips": 0,
        "scope": "post-selection feature-mask stability, not independent cohort validation",
    }
    (root / "exhaustive_lead_candidates_stability.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
