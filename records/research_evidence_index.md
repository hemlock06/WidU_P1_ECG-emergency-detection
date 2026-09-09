# Research evidence index

## Current synthesis and publication package — 2026-09-10

- `records/paper_synthesis_20260910.md` joins decisions, rejected alternatives,
  completed results and manuscript implications across the full research sequence.
- `results/paper_package_20260910/` contains exact-source tables, exportable figures,
  all4986 available prediction-file hashes and publication validation records.
- `results/vfdb_readiness_20260910/` verifies67 official files,22 waveforms and raw
  rhythm annotations. This is expansion readiness, not a trained VT/VF detector.
- The user authorized publication after verification on September10. Historical
  below-mentioned push holds record the state at those earlier milestones.
- Follow the publication receipt for actual remote commit/asset verification;
  no independent external performance or seven-day wearable results exist here.


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

## Historical split recovery and lineage audit, 2026-09-09

- `results/stage2_lineage/transfer.json`: eight original train/validation arrays
  copied with source/destination hashes verified; requested 5d warm checkpoint missing.
- `split_audit.json`, `overlap_matrix.csv`, `mc_test_binary_overlap_ids.csv`:
  original mc4357/933/936 and binary2217/474/474 have no within-dataset split overlap.
  Original mc test overlaps binary train343/validation65/test67. Inheritance of
  binary-trained weights into the final model is unresolved; overlap alone does not
  establish leakage. Regenerated mc_ml overlap672/125/139 reconfirmed.
- `report.md`, `home_repository_inventory.json`: search scope, checkpoint key list,
  missing warm identity and next historical-directory recovery step. Record-level
  integrity is not patient-level or upstream-training independence.
- `scripts/audit_stage2_lineage.py` and four passing tests preserve the checks.
  No new training/inference; all completed electrode artifacts remain unchanged.
- `search_update.md` and `legacy_directory_inventory.json` supersede the pending
  historical-directory status: home host reported that directory absent23:57:25 KST.
  Further preparation must audit external-dataset exposure during backbone pretraining.

## Published pretraining exposure audit, 2026-09-10

- `results/pretraining_exposure_20260910/report.md`, `audit.json` and
  `record_membership.csv`: original five-class test936 maps to published ECG-FM
  train763/validation94/test79; binary test474 maps to380/51/42/unlisted1.
  Self-supervised exposure is distinct from supervised label leakage. Previous
  results are internal test-selected comparisons, not full-lifecycle unseen ECGs.
- Official paper and the pinned split list include PTB-XL but exclude INCART/PTB.
  The public Challenge's broader composition alone is insufficient evidence.
- PTB-XL metadata preparation is explicitly provisional: current-version patient
  filtering yields1685 records/1676 patients under numeric ID assumptions; deleted
  old-version records and actual recording identity are not yet reconciled.
  This is not an eligible independent cohort and no model outcomes were inspected.
- New audit/preparation scripts, three passing schema tests, source fingerprints
  and complete membership tables preserve the finding. Frozen results unchanged.

## Interpretable classification counts and duration, 2026-09-10

- `results/classification_counts_20260910/`:4175 accuracy rows,20875 class-count
  rows,6877 raw-header durations/hashes, verification and report. All stored NPZ
  hashes/IDs checked; NumPy and sklearn confusion counts agree and reproduce F1.
- Five-contact II+aVR+V1+V2 seed42:723/936 correct (77.24%); ten-repeat mean76.98%.
  Ectopy30/82 found,52 missed,36 wrongly assigned, exposing the aggregate's limit.
  Counts use argmax, not the test-selected Sens@95Sp operating point.
- Test input nominal duration2h36m across936 records; all local raw CPSC totals
  about30h28m across6877 short recordings. Neither is a seven-day person trajectory.
  Classification FP is not a clinical alert or a false-alert-per-day estimate.
- Continuous seven-day validation and episode/alert/usable-wear-time metrics remain
  part of the external/wearable preparation. No new GPU inference or training.
