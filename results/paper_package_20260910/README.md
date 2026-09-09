# Manuscript evidence package — September10,2026

Start with [the complete synthesis](../../records/paper_synthesis_20260910.md).
It contains the rationale, rejected alternatives, methods, results, uncertainty,
exposure limits, clinical interpretation boundaries and future-study requirements.
Earlier retrospective records remain unchanged and retain their historical status.

## Contents and traceability

| File | Scope/source |
|---|---|
| candidate_table.csv | All8 fixed candidates; contacts, Macro-F1, repeats, accuracy and paired intervals |
| candidate_disease_counts.csv |8 candidates×5 classes, original seed42 confusion counts |
| macro_f1_intervals.svg/png | Same-test paired bootstrap95% exploratory intervals |
| disease_sensitivity.svg/png |10 feature-mask realizations, disease-specific Sens@95Sp |
| prediction_manifest.csv |4986 preserved NPZ files, relative paths/bytes/SHA256 |
| verification.json | Archive readback, package joins and51 prior manifest entries verified |
| tests_with_scoped_numpy_allowlist.txt |39 tests passed under the explicitly described compatibility context |
| publication_validation.json | Separate readback of tables, raw source identities and public-file scope |
| artifact_manifest.json | Hashes for the new manuscript, preparation and packaging artifacts |

The raw prediction archive includes all4986 NPZ files in the two experiment output
directories, including available controls/preflights and repeats. This total is not
4986 independent patients or4986 primary comparison rows. The main studies comprise
1586 original model/combination rows and4095 expanded P1 rows with794 reused outputs,
plus80 expanded candidate repeats. Each original CSV retains prediction linkage.
The earlier ten-seed metrics-only study has no per-realization probabilities to add;
missing historical probabilities have not been fabricated or silently rerun.

Public research release:
https://github.com/hemlock06/WidU_P1_LoRA-PEFT_Foundation-Model_Adaptation/releases/tag/p1-research-20260910

The release asset `p1-predictions-20260910.zip` preserves original relative paths.
Extract into a separate verification directory first; compare every member against
prediction_manifest.csv before using it. Do not overwrite an existing experiment.
Git stores the source, scientific records, CSVs and figures; the separate release
asset stores these raw probabilities. Clinical source waveforms and model weights
are not included in that asset. See docs/REPRODUCIBILITY.md and recovery/lineage audits
for sources. Dataset licenses and attribution still apply. This release is a research
archive and makes no deployment, independent-cohort or clinical validation claim.

## Reproduction and actual verification limits

`python scripts/build_paper_package.py` joins existing result tables, draws figures,
checks stored NPZ content/hashes and creates the release archive. It performs no new
model inference. Figure captions distinguish same-test ROC operating points from
argmax confusion counts and feature-mask variability from retraining variability.
Both figures were visually opened and checked for legibility and complete labels.

The first packaging run failed because an older manifest used a `files` list rather
than a path-keyed object. Support for both schemas was added; the corrected run
passed all51 retained manifest entries and4986 archive-member hash checks.
One broad-exception lint finding in the new VFDB script was narrowed to OSError and
ValueError; Ruff then passed both added scripts. These are tooling failures, not
ECG inference failures.
The first publication verifier also expected a prediction_path column absent in the
original793 CSV. It now reconstructs that historical path from the recorded run
fingerprint, model and subset, then compares the exact NPZ hash; no original CSV changed.

Unmodified `pytest tests -q` produced37 passes and2 setup errors in the legacy
embedding-contract tests. PyTorch2.7's default weights-only checkpoint loader rejected
`numpy.core.multiarray.scalar` in the existing trusted checkpoint. After checking
the exact a07 SHA, the test process allowed only `np.core.multiarray.scalar`,
`np.dtype` and `np.dtypes.Float64DType` via `torch.serialization.safe_globals`.
All39 tests passed in4.01s. Neither production code, checkpoint nor global serialization
policy was changed. Thus this does not claim the unmodified default command passed.
The inference-dependent contract tests exercise four synthetic zero-input records
(one single and one batch of three); these are software checks, not new ECG study results.

The previous original/expanded independent numerical audits remain intact. New tables
and figures reuse those results. A commit or release publication does not establish
clinical validity; a successful upload is checked against remote ref and downloaded
asset hashes separately in the publication receipt.
