"""Predeclared stochastic reproduction gate; retains the failed single-seed gate."""

import csv
import json
import time
from pathlib import Path

import ablation_exhaustive_lead_subsets as e
import numpy as np
import torch

SEEDS = tuple(range(10000, 10039))


def interval_check(expected, values, current):
    if len(values) != len(SEEDS) or not np.isfinite(values).all():
        raise ValueError("Need all 39 finite realizations")
    lo, hi = float(min(values)), float(max(values))
    return {
        "historical": expected,
        "current_seed42": current,
        "min": lo,
        "max": hi,
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)),
        "passed": bool(
            lo - 0.00005 <= expected <= hi + 0.00005 and lo <= current <= hi
        ),
    }


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    identity, fingerprint = e.asset_identity(42, 32)
    root = Path("results")
    single = json.loads(
        (root / "exhaustive_lead_subsets_controls.json").read_text(encoding="utf-8")
    )
    if single["fingerprint"] != fingerprint or len(single["rows"]) != 8:
        raise ValueError("Matching complete single-seed controls are required")
    if not all(r["original_code_check"] == "passed" for r in single["rows"]):
        raise ValueError("Original implementation comparison failed")
    historical_path = Path("outputs/lora_multitask_snr_a07/test_results.npz")
    historical = np.load(historical_path, allow_pickle=False)
    _, _, yb, ym = e.data_for("P1_a07")
    if not np.array_equal(historical["bin_labels"], yb) or not np.array_equal(
        historical["mc_labels"], ym
    ):
        raise ValueError("Historical labels disagree")
    past_metrics = e.metrics(yb, historical["bin_probs"], ym, historical["mc_probs"])
    if e.historical_checks("P1_a07", e.CONTROLS[0], past_metrics)["status"] != "passed":
        raise ValueError("Historical prediction audit failed")
    with (root / "exhaustive_lead_subsets_controls.csv").open(
        newline="", encoding="utf-8"
    ) as f:
        current = {(r["model"], r["lead_indices"]): r for r in csv.DictReader(f)}
    gate_path = root / "exhaustive_lead_subsets_stochastic_controls.json"
    gate = {
        "status": "running",
        "fingerprint": fingerprint,
        "identity": identity,
        "seeds": SEEDS,
        "checks": [],
        "completed_realizations": 0,
        "historical_prediction_sha256": e.sha256(historical_path),
        "single_seed_status": single["status"],
        "original_code_checks_passed": 8,
    }
    gate_path.write_text(json.dumps(gate, indent=2), encoding="utf-8")
    start = time.perf_counter()
    with (root / "exhaustive_lead_subsets_stochastic_controls.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = None
        for name in e.MODELS:
            x, _, yb, ym = e.data_for(name)
            model = e.model_for(name)
            for leads in [e.CONTROLS[0]] if name == "P1_a07" else e.CONTROLS:
                key = ";".join(map(str, leads))
                observed = []
                for seed in SEEDS:
                    pb, pm = e.predict(model, x, leads, 32, seed)
                    scores = e.metrics(yb, pb, ym, pm)
                    observed.append(scores)
                    row = {"model": name, "leads": key, "seed": seed, **scores}
                    if writer is None:
                        writer = csv.DictWriter(stream, fieldnames=row.keys())
                        writer.writeheader()
                    writer.writerow(row)
                    stream.flush()
                    gate["completed_realizations"] += 1
                    print(
                        name,
                        key,
                        seed,
                        scores["binary_auroc"],
                        scores["macro_f1"],
                        flush=True,
                    )
                targets = e.historical_checks(name, leads, observed[0])["checks"]
                checks = {
                    metric: interval_check(
                        target["expected"],
                        [r[metric] for r in observed],
                        float(current[(name, key)][metric]),
                    )
                    for metric, target in targets.items()
                }
                gate["checks"].append({"model": name, "leads": key, "metrics": checks})
                gate_path.write_text(json.dumps(gate, indent=2), encoding="utf-8")
            del model
            torch.cuda.empty_cache()
    gate["runtime_sec"] = time.perf_counter() - start
    passed = len(gate["checks"]) == 5 and all(
        c["passed"] for group in gate["checks"] for c in group["metrics"].values()
    )
    gate["status"] = "passed_stochastic_reproduction" if passed else "failed"
    gate_path.write_text(json.dumps(gate, indent=2), encoding="utf-8")
    print(gate["status"], flush=True)
    if not passed:
        raise RuntimeError("Stochastic reproduction failed; exhaustive run blocked")


if __name__ == "__main__":
    main()
