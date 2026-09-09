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
  `results/control_attempts/`. Single historical realization failed tolerance; separate
  preregistered-within-repository stochastic gate passed. This was not public preregistration.
- The ten-seed candidate stability experiment retained per-realization metrics and seed
  values, but did not retain each realization's record-level probabilities. This is an
  explicit evidence limit; reconstructing those probabilities would require a new logged rerun.

## Active experiment: electrode coverage v1, 2026-09-09

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
