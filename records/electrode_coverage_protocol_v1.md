# Electrode coverage protocol v1 — fixed before expanded evaluation

Date: 2026-09-09. Stage 1 is exploratory frozen-model analysis, not clinical validation.

## Objective and sequence

Minimize physical measurement contacts while maximizing disease-specific detection and
reliability. Execute sequentially: (1) electrode-based internal exploration; (2) restore
historical split and warm-start lineage, define independent external label/threshold
protocol and evaluate; (3) specify and assess simultaneous reference/wearable recordings
when actual hardware and labeled recordings are available. No automatic retraining or
claim of clinical suitability. Do not publish or push before review.

## Frozen stage 1

- Model: P1 epoch18 alpha0.7, checkpoint SHA256
  287148bfd01ac67b5192268c6c69cc4cc230c004bdfdf332285f38a3b43b08dd.
- Historical CPSC single-label test936, labels [NSR130,AF180,ischemia165,conduction379,ectopy82].
- Full parent identity: 34060e20ca6ca6de13d114a5b1b14cf4e1fd9ffccc3f7ec4453bb00d8b5646f1.
- float32, batch32, seed42 reset before each configuration, TF32 off, deterministic
  cuDNN, unchanged legacy stochastic feature mask. No threshold/model tuning in this stage.
- Evaluate every nonempty subset of 12 standard leads: 4095. Reuse 793 previously
  validated P1 subset predictions and one 12-lead control, only after SHA/source-label
  validation and four current forward controls. Compute the remaining 3301 configurations.
- Lower-contact subsets run first. Controls additionally check six limb leads and six
  limb leads+V2 against the original evaluation implementation before expanded inference.
- Reused predictions remain in their original immutable paths; every new configuration
  saves record IDs, both labels, binary probabilities and five-class probabilities.

## Physical model and restrictions

Standard electrode incidence only: I requires RA/LA; II RA/LL; III LA/LL; augmented
limb leads RA/LA/LL; each chest lead RA/LA/LL plus its chest site. Contacts for a
configuration are the union of these sets. DRL/ground is excluded from the measurement
count and an additional-contact scenario is shown separately. This is not a tested
wearable circuit. RA/LA/LL standard potentials must not be equated with arbitrary torso
placements. Derived leads do not create independent electrode information.

## Metrics and selection, fixed before new outcomes

- Primary four disease groups: AF, ischemia, conduction, ectopy. Keep AUROC, per-class
  F1, argmax recall, and Sens@95Sp, plus NSR-inclusive macro-F1/macro recall.
- Sens@95Sp is an empirical ROC operating point chosen in this test; it is not an
  independently fixed deployable threshold. Report this selection limitation.
- Multiobjective Pareto analysis uses measurement-contact count (minimize) and all
  four Sens@95Sp values (maximize). No mean score may hide a weak disease class.
- Coverage is reported over the entire predeclared sensitivity grid 0.50/0.60/0.70/
  0.80/0.90 at specificity>=0.95, as disease-specific flags and number of groups met.
  These are sensitivity analyses, NOT clinical acceptance criteria. Do not pick one
  favorable threshold after seeing results or count unsupported diagnoses as covered.
- For each exact contact count retain the macro-F1 maximum, the maximum worst-disease
  Sens@95Sp (tie break: four-disease mean, then macro-F1, then lexicographic lead indices),
  and the complete-available-lead physical configurations. Preserve all rows and frontier.
- Shortlist the winners from both criteria for contact counts2/3/4/5, deduplicate,
  and include the 12-lead baseline. Assess all shortlisted configurations over ten
  fixed seeds30000..30009, preserving probabilities for EVERY realization.
- For those candidates use paired record bootstrap2000, seed31415, on fixed seed42
  predictions vs12. Report all four disease metrics and macro metrics; skip and record
  replicates without every class. No reroll to obtain favorable intervals.
- Ranking and intervals reuse the exploration test. No superiority/noninferiority,
  calibration, broad ECG disease coverage, early prediction, independent cohort or
  fabric-electrode validity claim follows from this stage.

## Gates and reproducibility

E4 gates: original parent asset/source identity and saved prediction hashes; current
four-condition forward parity; two expanded-input implementation controls; unique
4095 subsets; all probabilities finite/in range and sum to one; source IDs/labels
identical; independent full-ROC/confusion-matrix metric recalculation; raw-to-summary
regeneration; fail/skip counts. Invalid outputs stop progression to stage2.

Original single-seed historical reproduction remains failed/unverified. Separately
predeclared stochastic reproduction passed39 seeds x5conditions/17metrics; keep both facts.
Record manifests, UTC/KST times, command, code commit/hash, environment, data fingerprints,
raw outcomes, controls, failures and rationale. Append events and preserve interruptions.
Do not overwrite the prior experiment or delete negative findings. Local retention,
Git tracking, review copies and independent off-device backup are distinct statuses.

## Stage2/3 prerequisites

The current regenerated mc_ml splits overlap the old test and are not a fresh holdout.
Restore original train/val IDs without moving originals; trace every warm checkpoint's
training data. Public PTB-XL/other datasets require diagnostic-label compatibility and
training-exposure checks before they can be called independent. Existing binary-only
external preprocessing is insufficient for four-group validation. Freeze candidate
selection and validation thresholds before inspecting the external test results.

Actual recording requires the intended contact locations, circuit/reference/DRL,
sampling and synchronized reference ECG, activity/contact quality and appropriate
record/person-level grouping. If these assets are absent, prepare a concrete acquisition
and evaluation specification; do not simulate them and call the result measured.
