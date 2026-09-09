# Research evidence index

Purpose: preserve an auditable path from manuscript claims and tables to the exact
experiment, code, model, input records and predictions. This index describes retained
evidence and its limits; file presence is not proof of clinical validity or off-device backup.

## Completed experiment: reduced-lead search, 2026-09-08

- Scope: frozen P1 793 subsets, separate binary reference793; 1586 rows total.
- Code/results commit: 5fd1408f24202edc2a38fa953ab28bf7c3977712; no push at completion.
- Protocol: `results/lead_subset_stochastic_protocol.md` and parent control manifests.
- Raw: `results/exhaustive_lead_subsets_1to4.csv`; row-level prediction SHA and model/data identities.
- Predictions: `outputs/exhaustive_lead_subsets/34060e20ca6ca6de13d114a5b1b14cf4e1fd9ffccc3f7ec4453bb00d8b5646f1/`.
- Metrics/interpretation: `results/exhaustive_lead_subsets_{summary,paired_ci}.csv`,
  `results/exhaustive_lead_subsets_{report,interpretation}.md`.
- Verification: `results/exhaustive_lead_subsets_{verification,analysis_verification}.json`.
- Environment/timing/tests: `results/lead_subset_completion_validation.md` and recovery audits.
- Negative results: `results/exhaustive_lead_subsets_failures/`, controls.json,
  `results/control_attempts/`. Single historical realization failed tolerance; a
  separately defined stochastic gate passed. The protocol was written before calibration,
  then committed while calibration was running; this was not public preregistration.
- The ten-seed candidate stability experiment retained per-realization metrics and seed
  values, but did not retain each realization's record-level probabilities. This is an
  explicit evidence limit; reconstructing those probabilities would require a new logged rerun.

## Completed internal experiment: electrode coverage v1, 2026-09-09

- Protocol frozen before new outcomes: `records/electrode_coverage_protocol_v1.md`.
- Runner: `scripts/evaluate_electrode_coverage.py`.
- Scope: P1 all4095 nonempty lead subsets, indexed by standard physical measurement
  contacts; 794 parent predictions reused after checks,3301 new predictions planned.
- Manifest/current controls: `results/electrode_coverage_v1/preflight.json`.
- State/events/raw: `results/electrode_coverage_v1/{status.json,events.jsonl,raw.csv}`.
- New raw probabilities: `outputs/electrode_coverage_v1/<run_id>/` (all retained).
- Reused prediction paths remain explicitly recorded in each row. Do not conflate
  imported rows, new inference, resume skips, preflight evaluations and failed attempts.
- Progression to external validation requires independent metric and data-integrity audit.
- Later repeatability studies must save record-level probabilities as well as metrics.

## Manuscript traceability and limitations

Each table must identify its run and analysis source, checkpoint SHA, data/split identity,
metric definition, operating-point selection, aggregation unit and uncertainty method.
Data use and disease definitions must remain explicit. Checkpoint lineage and original
split recovery precede assertions that a cohort is unseen. Multi-label pathologies must
not be equated with mutually exclusive five-class predictions without a stated mapping.

All current P1 search results reuse an exploration test. Bootstrap and feature-mask
repeats do not remove selection bias or establish independent-cohort generalization.
Favorable and negative class results, aborted controls, processing skips and protocol
changes are retained. New protocols do not retroactively validate old comparisons.

## Storage status

Code, protocols, metric CSVs, reports and small manifests are Git-tracked and locally
committed at milestones. Checkpoints, input waveforms, raw prediction NPZ and operational
logs are retained locally outside Git. Git status alone cannot confirm their backup.
Review copies are also local, on the same disk; they are not independent off-device backups.
The restored source assets have external originals as documented in recovery audits,
but newly generated predictions have no independently verified off-device backup yet.
Do not delete local data on the assumption that a commit or review ZIP contains them.
Push remains held for final review.

## Retrospective manuscript dossier, 2026-09-09

- `records/paper_retrospective_20260909.md`: motivation, recovery, original failures,
  protocol amendments, completed results, electrode extension and remaining limitations.
- `records/paper_related_work_20260909.md`: targeted primary-paper review, including
  existing exhaustive/electrode-cost work; no absence-of-literature or first-ever claim.
- `records/paper_history_events_20260909.csv`: selected original conversation timestamps,
  source lines and hashes; no full private transcript copied into the repository.
- `records/paper_candidate_table_20260909.csv`: exact source-row candidate metrics.
- `records/paper_evidence_manifest_20260909.csv` and verification JSON: source fingerprints
  and checked scope. `scripts/build_paper_evidence.py` regenerates these retrospective tables.
- Original protocol timing was checked against raw history: written19:05, launched19:07,
  committed19:12 KST on September8. A prior description of a pre-calibration commit was
  inaccurate; it was a pre-exhaustive-run commit. Original artifacts remain unchanged.
- The new dossier is retrospective and does not turn completed exploratory work into a
  prospective study or make the ongoing4095 experiment complete.

## Interruption and postprocessing preparation, 2026-09-09

- `records/electrode_postprocessing_readiness_20260909.md`: one externally observed
  process interruption, independent saved-result audit, identical-condition restart,
  new postprocessing scripts and tested versus pending execution scope.
- `results/electrode_resume_audit_20260909_1748.json`:1610 saved prediction files,
  40250 independently recalculated metrics, unchanged parent identity, no orphan files.
- The first1610 raw rows remained byte-identical after restart. Cause of process loss
  remains unknown; a stale running flag is not evidence of a live process.
- At the preparation milestone, restart replay and full4095 audit/selection/repeats/intervals
  were pending. They subsequently passed the checks recorded below.

## Expanded results and final numerical review, 2026-09-09

- Expanded inference finished22:19:24 KST:4095 subsets,794 imported and3301 new,
  failure events0, external interruption1. The interruption cause remains unknown.
- `results/electrode_coverage_v1/restart_parity.json`: both restart-boundary cases
  and12-lead full936 predictions replayed with maximum probability error0.
- `independent_audit.json`:4095 NPZ and102375 metrics checked; maximum error3.33e-16.
- `analysis.json`, six selection CSVs:44 Pareto rows,67 complete physical configurations,
  20475 coverage-grid rows and8 shortlisted configurations including12-lead.
- `stability_raw.csv` and `stability_status.json`:80 seed30000–30009 realizations,
  all record-level NPZ preserved. `candidate_analysis.json`:144 summaries/144 paired
  intervals (2000 record draws, seed31415, skips0 for each candidate).
- `final_analysis_verification.json`: separate NumPy tied-rank ROC/bincount computation
  reproduced2000 repeated metrics, all summaries and intervals; separately checked
  Pareto, coverage and selection. No orphan/missing expanded or repeat predictions.
- `report.md`: manuscript-oriented methods/results/tradeoffs/limitations and timing.
  `artifact_manifest.json`:28 result/source/test files with SHA256 and byte counts.
- New final verifier tests plus existing electrode tests:13 passed,2.55 seconds;
  Ruff passed. Source scripts are additive; frozen inference/protocol unchanged.
- V1+V2 uses five measurement contacts and reaches all four0.50 sensitivity floors
  at seed42, but ectopy Sens95 averages0.470732 over ten feature-mask seeds. This is
  not stable four-disease coverage or measured wearable validity. All selected
  non-baseline Macro-F1 difference intervals include zero. External validation remains pending.
