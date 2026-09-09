"""Build retrospective evidence pointers and candidate tables without model inference."""

import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BRAIN = Path("C:/brain")
RECORDS = ROOT / "records"
HISTORY = Path(
    "C:/Users/hemlo/.codex/sessions/2026/09/08/"
    "rollout-2026-09-08T15-30-14-01a07fb5-c7d5-7d73-9800-a662f5dba63b.jsonl"
)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(name, rows):
    with (RECORDS / name).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    raw_path = ROOT / "results/exhaustive_lead_subsets_1to4.csv"
    raw = read_csv(raw_path)
    audit = json.loads((ROOT / "results/exhaustive_lead_subsets_verification.json").read_text())
    assert digest(raw_path) == audit["raw_sha256"]
    assert Counter(r["model"] for r in raw) == {"P1_a07": 793, "reference_iii": 793}
    assert len({(r["model"], r["lead_indices"]) for r in raw}) == 1586
    candidates = [
        ("II", 2), ("I;aVR", 3), ("II;aVR;V1;V2", 5),
    ]
    controls = read_csv(ROOT / "results/exhaustive_lead_subsets_controls.csv")
    baseline = next(r for r in controls if r["model"] == "P1_a07" and r["n_leads"] == "12")
    metrics = ["macro_f1", "macro_sensitivity"] + [
        f"{metric}_{group}" for group in ("af", "ischemia", "conduction", "ectopy")
        for metric in ("auroc", "f1", "sensitivity", "sens95")
    ]
    table = []
    for row, contacts in [(baseline, 9)] + [
        (next(r for r in raw if r["model"] == "P1_a07" and r["lead_names"] == name), contacts)
        for name, contacts in candidates
    ]:
        table.append({
            "lead_names": row["lead_names"], "measurement_contacts": contacts,
            "extra_drl_contacts_scenario": contacts + 1,
            "n_samples": row["n_samples"], **{m: row[m] for m in metrics},
            "checkpoint_sha256": row["checkpoint_sha256"],
            "data_fingerprint": row["data_fingerprint"],
            "run_fingerprint": row["run_fingerprint"],
            "prediction_sha256": row["prediction_sha256"],
        })
    write_csv("paper_candidate_table_20260909.csv", table)

    # Only selected research requests and timing evidence, never entire raw conversations.
    actions = {
        9: "Initial 1-4 lead exhaustive experiment request; stop before push",
        1482: "Request to search historical source assets on home PC",
        1960: "Continue after control-failure investigation",
        1998: "Write stochastic reproduction protocol and calibration implementation",
        2037: "Launch stochastic calibration",
        2112: "Commit stochastic protocol while calibration is running",
        2129: "Consider possible effects of historical home-PC resource limits",
        3372: "Optimize actual electrode count, disease coverage and reliability",
        3395: "Check readiness for additional validation",
        3454: "Authorize sequential further validation",
        3468: "Preserve research records for manuscript use",
        3592: "Request full retrospective and permission to consult prior work",
    }
    events = []
    with HISTORY.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if line_no not in actions:
                continue
            item = json.loads(line)
            assert item["type"] == "response_item"
            events.append({
                "source_path": str(HISTORY), "source_line": line_no,
                "timestamp_utc": item["timestamp"],
                "source_line_sha256": hashlib.sha256(line.encode()).hexdigest(),
                "kind": item["payload"]["type"], "research_relevance": actions[line_no],
                "evidence_scope": "selected contemporaneous event; not full transcript export",
            })
    assert len(events) == len(actions)
    assert next(e for e in events if e["source_line"] == 1998)["timestamp_utc"] < next(
        e for e in events if e["source_line"] == 2037
    )["timestamp_utc"] < next(e for e in events if e["source_line"] == 2112)["timestamp_utc"]
    write_csv("paper_history_events_20260909.csv", events)

    sources = []
    for pattern in (
        "results/lead_subset_*", "results/exhaustive_lead_subsets_*.json",
        "results/exhaustive_lead_subsets_*.csv", "results/exhaustive_lead_subsets_*.md",
        "results/exhaustive_lead_candidates_stability*", "results/control_attempts/*.json",
        "results/exhaustive_lead_subsets_failures/*.json",
    ):
        sources.extend(p for p in ROOT.glob(pattern) if p.is_file())
    sources += [ROOT / p for p in (
        "scripts/ablation_nlead_curve.py", "results/nlead_curve.csv",
        "scripts/ablation_exhaustive_lead_subsets.py", "scripts/calibrate_lead_subset_controls.py",
        "scripts/verify_exhaustive_lead_results.py", "scripts/interpret_lead_subset_results.py",
        "scripts/evaluate_electrode_coverage.py", "scripts/build_paper_evidence.py",
        "records/03_eval_results.md", "records/01_design_decisions.md",
        "records/04_run_history.md", "docs/REPRODUCIBILITY.md",
        "records/electrode_coverage_protocol_v1.md", "results/electrode_coverage_v1/preflight.json",
        "records/paper_retrospective_20260909.md", "records/paper_related_work_20260909.md",
        "records/paper_manuscript_scaffold_20260909.md",
        "records/paper_candidate_table_20260909.csv", "records/paper_history_events_20260909.csv",
    )]
    sources += [BRAIN / p for p in (
        "exchange/_archive/2026-09-08T1525_mac_p1-exhaustive-lead-subsets-lab-pc.md",
        "records/session-closeouts/2026-09-08/lead-configuration-experiment-history.md",
        "records/session-closeouts/2026-09-08/2142-p1-exhaustive-lead-subsets.md",
        "records/session-closeouts/2026-09-08/2201-electrode-validation-readiness.md",
        "wiki/reduced-lead-ecg.md",
    )]
    manifest = []
    for path in sorted(set(sources)):
        assert path.is_file(), path
        manifest.append({
            "source_path": str(path),
            "sha256": digest(path), "bytes": path.stat().st_size,
            "role": "context record" if path.is_relative_to(BRAIN) else "experiment evidence",
            "scope": "snapshot; historical blocked reports retain their original time-specific status",
        })
    write_csv("paper_evidence_manifest_20260909.csv", manifest)
    # Exact original labels provide an independent check for cohort counts in the prose.
    import numpy as np
    labels = np.load(ROOT / "data/processed/cpsc2018_mc/test/labels.npy")
    assert np.bincount(labels, minlength=5).tolist() == [130, 180, 165, 379, 82]
    assert len(np.load(ROOT / "data/processed/cpsc2018/test/labels.npy")) == 474
    receipt = {
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed", "source_files_hashed": len(manifest),
        "selected_history_events": len(events), "candidate_rows_copied_exactly": len(table),
        "raw_rows": len(raw), "raw_sha256_matches_previous_independent_audit": True,
        "mc_class_counts": np.bincount(labels, minlength=5).tolist(),
        "historical_protocol_timing": "written before calibration; committed during calibration",
        "scope": "dossier provenance, selected figures and source readback; not a new full prediction audit",
        "active_experiment_modified": False,
    }
    (RECORDS / "paper_evidence_verification_20260909.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()
