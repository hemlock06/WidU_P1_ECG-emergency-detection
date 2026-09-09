# Historical split and training-lineage audit

Audit date: 2026-09-09. This is preparation for external validation, without new
inference or training. The completed 4095-subset results remain unchanged.

## Recovery and verification

Eight original train/validation ID and label arrays were copied from the home
repository into `work/stage2_original_assets/`. Source and destination sizes and
SHA256 values matched the transfer manifest. Originals were retained. The requested
`outputs/lora_multitask/lora_multitask_best.pt` was absent at the requested location.

`audit_stage2_lineage.py` verified all restored hashes again, array lengths,
record ID syntax and uniqueness, five-class labels and their AF-or-ischemia binary
derivation. Python-set intersections were cross-checked with NumPy intersections
for every pair of the 12 original/regenerated split arrays (66 pairs).

| Original data | Train | Validation | Test | Between-split ID overlaps |
|---|---:|---:|---:|---:|
| Five-class CPSC | 4357 | 933 | 936 | 0 |
| Binary CPSC | 2217 | 474 | 474 | 0 |

Five-class distributions in NSR, AF, ischemia, conduction, ectopy order:
train `[640,820,755,1770,372]`, validation `[136,176,163,379,79]`,
test `[130,180,165,379,82]`. Binary train/validation labels were not part of this
recovery and were not audited. Record-disjoint splits do not prove patient-level
independence, absence of duplicated signals, or an upstream model's training exposure.

## Cross-dataset overlap requiring lineage clarification

The original five-class test contains 343 records from the original binary train,
65 from binary validation and 67 from binary test. All 475 IDs and their five-class
labels are retained in `mc_test_binary_overlap_ids.csv`.

This is **not proof of leakage into the final model**: the binary model might be a
separate comparator. If its adapted weights were inherited by an ancestor of the
final model, some current test records would have been exposed upstream. That
inheritance remains unverified. No subset of the current test is newly declared
independent on the basis of removing these overlaps alone.

The regenerated multilabel train/validation/test overlaps the original five-class
test by 672/125/139 records. This reproduces the earlier reason for excluding that
split as an independent validation set. The regenerated binary splits overlap it
by 319/72/84. See `overlap_matrix.csv` for the complete matrix.

## Checkpoint evidence and search limits

The final a07 checkpoint hash remains
`287148bfd01ac67b5192268c6c69cc4cc230c004bdfdf332285f38a3b43b08dd`.
Its metadata identifies epoch18, alpha0.7, LoRA rank8/alpha16 and the expected
five classes. It does not contain split identities or a warm-start path/hash.
The complete key list and scalar metadata are preserved in `split_audit.json`.

`records/02_training_logs.md` sections6 and8 and `docs/REPRODUCIBILITY.md` describe
a 5d warm start. Current training code conditionally loads it only if the supplied
path exists; source defaults therefore cannot prove what ran historically.
Even the repository's initial version already describes 5d+SNR and does not
establish how the earlier 5d checkpoint was initialized.

A read-only home repository inventory completed23:51:28 KST, returning20 matching
`.pt/.pth/.zip/.7z` paths, including two Python-package files. The requested 5d
checkpoint is absent from that bounded inventory. A separate direct inventory of
the corresponding Drive `outputs` folder returned eight checkpoints, also without
the requested file. These searches do not establish absence from all storage.

The initial Git version (`1c138d4`) names another historical directory,
`D:/WidU_ecg-fm_emergency-detection`, distinct from the home repository ending in
`_git`. A read-only inventory request for that evidenced path was submitted as
`20260910-p1-stage2-legacy-root-inventory-01`; its result is pending at this milestone.
The request ID is an opaque identifier, not an execution timestamp.

## Reproducibility and remaining work

Run `.venv/Scripts/python.exe scripts/audit_stage2_lineage.py` from the repository.
Four ID-validation/intersection tests and Ruff passed. Numerical split validation
passed; training-lineage independence remains unresolved. No GPU inference,
retraining or external performance evaluation was performed in this audit.

Next: inspect the historical-directory result and existing training records;
preserve any recovered checkpoint and its original hashes. Then establish external
dataset label mapping, training/pretraining exposure and a prospective evaluation
protocol before viewing external outcomes. Actual wearable validation still
requires verified hardware/reference recordings or a concrete collection specification.
All current performance remains retrospective internal classification, with
candidate-selection bias and no early-prediction or clinical-optimality claim.
