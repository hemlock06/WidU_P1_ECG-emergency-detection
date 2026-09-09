"""Render a manuscript-oriented exploratory report only after numerical verification."""

import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path

ROOT = Path("results/electrode_coverage_v1")
DISEASES = ("af", "ischemia", "conduction", "ectopy")


def table(name):
    with (ROOT / name).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    verification = json.loads((ROOT / "final_analysis_verification.json").read_text())
    assert verification["status"] == "passed"
    assert verification["candidate_receipt_sha256"] == sha(
        ROOT / "candidate_analysis.json"
    )
    assert verification["verifier_sha256"] == sha(
        "scripts/verify_electrode_analysis_final.py"
    )
    selected = table("shortlist.csv")
    summary = table("stability_summary.csv")
    intervals = table("paired_intervals.csv")
    raw = table("raw.csv")
    events = [
        json.loads(line) for line in (ROOT / "events.jsonl").read_text().splitlines()
    ]
    starts = [e for e in events if e["event"] == "run_started"]
    end = next(e for e in events if e["event"] == "inference_complete")
    wall = (
        datetime.fromisoformat(end["utc"]) - datetime.fromisoformat(starts[0]["utc"])
    ).total_seconds()
    new_seconds = sum(
        float(r["runtime_sec"]) for r in raw if r["origin"] == "new_inference"
    )
    lines = [
        "# Electrode-based P1 coverage exploration — verified results",
        "",
        "## Scope and primary result",
        "",
        "This study evaluates every nonempty subset of the 12 standard ECG leads (4095) using a frozen P1 multitask checkpoint. It reports a tradeoff between physical measurement contacts and four disease groups, not one clinically established optimal wearable configuration.",
        "The historical CPSC test has 936 records: NSR130, AF180, ischemia165, conduction379 and ectopy82. The model was not retrained. All results below reuse this exploration test.",
        "",
        "## Acquisition assumptions",
        "",
        "I uses RA/LA, II RA/LL, III LA/LL; augmented limb leads require RA/LA/LL. Standard chest leads additionally require their chest site and RA/LA/LL for the reference. Count the union of measurement contacts; add one separately if the circuit needs DRL/ground. V1+V2 therefore represents five measurement contacts, not two electrodes. These are standard acquisition assumptions, not a measured wearable circuit or arbitrary torso placement equivalence.",
        "",
        "## Frozen seed42 shortlist",
        "",
        "Sens@95Sp is the maximum empirical sensitivity at specificity >=95% on this same test ROC. Thresholds have not been independently fixed for deployment. Macro-F1 and macro recall include NSR and all four disease groups.",
        "",
        "|Input leads|Measurement contacts|With extra DRL|Macro-F1|AF Sens95|Ischemia Sens95|Conduction Sens95|Ectopy Sens95|",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    ordered = sorted(
        selected, key=lambda r: (int(r["measurement_contacts"]), r["lead_indices"])
    )
    for r in ordered:
        name = "12-lead baseline" if int(r["n_leads"]) == 12 else r["lead_names"]
        values = [float(r["macro_f1"])] + [float(r[f"sens95_{c}"]) for c in DISEASES]
        lines.append(
            f"|{name}|{r['measurement_contacts']}|{r['contacts_with_extra_drl']}|"
            + "|".join(f"{v:.6f}" for v in values)
            + "|"
        )
    lines += [
        "",
        "## Repeatability and paired record uncertainty",
        "",
        "Eight shortlisted configurations, including the baseline, were evaluated with seeds30000–30009. Every realization retains record IDs, labels and probabilities. Feature-mask variation is not training-run or external-cohort variation.",
        "Paired record bootstrap uses 2000 draws, seed31415, on the fixed seed42 predictions. The same sampled records are used for each candidate and baseline. Draws missing any of five classes or either binary label are skipped without replacement draws. Intervals are unadjusted exploratory percentile intervals after test-based selection; they do not establish superiority or noninferiority.",
        "",
        "|Input leads|10-seed mean Macro-F1|Seed SD|Mean paired delta vs12|Seed42 delta vs12 [95% CI]|",
        "|---|---:|---:|---:|---|",
    ]
    for r in ordered:
        key = r["lead_indices"]
        s = next(
            s for s in summary if s["lead_indices"] == key and s["metric"] == "macro_f1"
        )
        ci = next(
            s
            for s in intervals
            if s["lead_indices"] == key and s["metric"] == "macro_f1"
        )
        name = "12-lead baseline" if int(r["n_leads"]) == 12 else r["lead_names"]
        lines.append(
            f"|{name}|{float(s['mean']):.6f}|{float(s['std']):.6f}|{float(s['paired_mean_delta_vs12']):+.6f}|{float(ci['difference_vs12']):+.6f} [{float(ci['ci_low']):+.6f}, {float(ci['ci_high']):+.6f}]|"
        )
    lines += [
        "",
        "The complete 18 primary metric intervals and seed summaries are in paired_intervals.csv and stability_summary.csv. Disease-specific weaknesses must be considered alongside the aggregate.",
        "For V1+V2 (five contacts), mean repeated Sens95 is AF0.907222, ischemia0.550303, conduction0.848813 and ectopy0.470732 (ectopy range0.426829–0.512195). Thus the fixed-seed result meeting all four0.50 sensitivity floors is not stable across feature-mask realizations. Its seed42 ectopy delta vs12 is +0.170732 with an unadjusted exploratory95% interval[+0.030920,+0.283951], while ischemia delta is -0.042424. These test-selected findings motivate independent validation, not a superiority claim.",
        "For two contacts, II is the minimal-acquisition candidate. At three contacts, I+aVR maximizes Macro-F1 while I+III+aVR+aVF maximizes worst-disease Sens95 under the frozen criterion. At five contacts, II+aVR+V1+V2 maximizes Macro-F1; V1+V2 improves the weakest disease sensitivity at an ischemia tradeoff. All non-baseline Macro-F1 difference intervals include zero. Increasing electrode count does not automatically improve every disease score in this frozen model.",
        "",
        "## Coverage and Pareto frontier",
        "",
        "The frontier contains 44 nondominated input subsets under measurement-contact count and the four Sens@95Sp objectives. Full available-lead configurations for all 67 distinct physical contact sets are retained separately. Selecting fewer derived input leads with the same contacts changes model input, not physical acquisition cost.",
        "Coverage_grid.csv retains all4095 x5 predefined thresholds (0.50,0.60,0.70,0.80,0.90). These thresholds are sensitivity analyses, not clinical acceptance criteria.",
        "",
        "|Sensitivity floor at specificity >=95%|Fewest contacts with all four groups meeting the floor|",
        "|---|---|",
    ]
    for t in (0.5, 0.6, 0.7, 0.8, 0.9):
        qualifying = [
            int(r["measurement_contacts"])
            for r in raw
            if all(float(r[f"sens95_{c}"]) >= t for c in DISEASES)
        ]
        lines.append(
            f"|{t:.2f}|{min(qualifying) if qualifying else 'No configuration'}|"
        )
    lines += [
        "",
        "## Integrity, timing and provenance",
        "",
        f"Run ID: `{verification['run_id']}`. Repeat ID: `{verification['repeat_run_id']}`.",
        "Checkpoint SHA256: `287148bfd01ac67b5192268c6c69cc4cc230c004bdfdf332285f38a3b43b08dd`.",
        f"Raw CSV SHA256: `{sha(ROOT / 'raw.csv')}`.",
        "RTX5070Ti, fp32, batch32, TF32 off; frozen legacy feature-mask behavior retained. Parent source/data/runtime identity was checked before inference and again during audit/repeats.",
        f"Expanded inference wall span, including interruption: {wall:.2f} seconds ({wall / 3600:.3f} hours). Sum of3301 new combination runtimes: {new_seconds:.2f} seconds. Imported794 results were not rerun as part of this sum; controls, replay, repeats and analysis are additional work.",
        "Original inference completed 2026-09-09 22:19:24 KST. One external process interruption occurred; cause unconfirmed. Saved1610 rows were audited before identical-condition resume. Boundary replay (last before interruption, first after, and12-lead) matched probabilities exactly (maximum error0). Original files were preserved.",
        "Full audit checked4095 NPZ,102375 metrics,6 preflight files, unique subsets, original IDs/labels/hashes, with maximum metric error3.33e-16. Inference failure events0; observed external interruptions1. Repeated80 NPZ and2000 metrics,144 summary rows and144 intervals were independently rechecked via separate NumPy ROC/confusion calculations.",
        f"Bootstrap skipped draws per candidate: {verification['skipped']}. No reroll.",
        "",
        "## Interpretation limits and next stage",
        "",
        "A low-contact candidate is a research shortlist, not established clinical reliability. Ectopy and ischemia sensitivity tradeoffs remain visible. The same internal test selected subsets and ROC operating points; bootstrap and feature-mask repeats do not remove selection bias, establish calibration, or prove independent generalization. These are classifications of current ECGs, not early prediction.",
        "The original single-seed historical reproduction failure remains recorded; the separate predeclared39-seed stochastic gate passed. Neither fact should be substituted for the other.",
        "Next, recover original training/validation IDs and warm-start lineage, audit external training exposure and label compatibility, and freeze the external evaluation protocol before viewing external test results. New overlapping CPSC splits are not independent validation. Actual wearable validity requires synchronized reference recordings and confirmed electrode/circuit geometry.",
        "Existing reduced-lead and exhaustive reconstruction/electrode-cost publications are documented in records/paper_related_work_20260909.md. This study does not claim the first exhaustive search or absence of prior work.",
        "",
        "## Retained evidence and backup",
        "",
        "Protocols, source, CSVs and small verification manifests are locally versioned. Each raw row points to its retained prediction NPZ and SHA. Large predictions, checkpoints and waveforms remain outside Git. Review copies on the same disk are not an independent backup. Off-device backup of newly generated predictions has not yet been verified. Push is held for user review.",
        "",
    ]
    target = ROOT / "report.md"
    target.write_text("\n".join(lines), encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
