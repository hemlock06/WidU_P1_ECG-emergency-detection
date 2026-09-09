# Pretraining exposure and external-validation readiness

Date: 2026-09-10. No new model inference or training was performed. This audit
changes the interpretation of independence, not the saved classification results.

## Published pretraining evidence

The [official ECG-FM repository](https://github.com/bowang-lab/ecg-fm) identifies
`mimic_iv_ecg_physionet_pretrained.pt` as pretrained on MIMIC-IV-ECG and PhysioNet2021.
The [paper, sections2.1–2.2](https://arxiv.org/html/2408.05178v2) specifies six included
Challenge datasets and explicitly excludes PTB and INCART. It describes random
PhysioNet record splits and their separation across pretraining and finetuning.

The published record list was downloaded from revision
`9f926f1911bb9f24789b5c6407677d58ad753054`:
[meta_split_physionet.csv](https://raw.githubusercontent.com/bowang-lab/ecg-fm/9f926f1911bb9f24789b5c6407677d58ad753054/splits/meta_split_physionet.csv).
SHA256 `a2d17b0485a8186d227e82f2200ceb8b2ba7423df23a4fd306275f2197b4d88d`.
The list contains CPSC, CPSC-Extra, PTB-XL, Georgia, Ningbo and Chapman-Shaoxing;
no INCART or PTB rows appear. Inclusion in the broader Challenge alone therefore
does not establish that every Challenge dataset was used by ECG-FM.

| Dataset in published list | Train | Validation | Test |
|---|---:|---:|---:|
| CPSC2018 | 5493 | 698 | 668 |
| CPSC-Extra | 2743 | 358 | 344 |
| PTB-XL | 17519 | 2167 | 2150 |
| Georgia | 8263 | 1024 | 1042 |
| Ningbo | 26638 | 3312 | 3378 |
| Chapman-Shaoxing | 8167 | 1044 | 1021 |

## Direct membership audit of original P1 records

`audit_pretraining_exposure.py` verifies the source hash, dataset-qualified unique
IDs, allowed split names and the original ID-file hashes from the previous split
audit. Joins are independently checked with NumPy intersections. Full original
train/validation/test membership is preserved in `record_membership.csv`.

| Original P1 test | n | Published train | Published validation | Published test | Unlisted |
|---|---:|---:|---:|---:|---:|
| Five-class | 936 | 763 | 94 | 79 | 0 |
| Binary reference | 474 | 380 | 51 | 42 | 1 |

Most P1 test records are therefore listed in the backbone's published pretraining
train split. This establishes published-list membership; it is not a reconstruction
of every actual optimizer step. Self-supervised waveform exposure is distinct from
supervised diagnosis-label leakage. The unresolved 5d warm-start history is a
separate issue. An unlisted record is not automatically an independent record.

The 793- and4095-subset results remain valid saved within-cohort measurements, but
must not be presented as performance on ECGs unseen throughout model development.
The relative candidate comparisons remain exploratory and test-selected. The79
records in the published test split were also used in candidate selection; extracting
them now would not create a new independent validation set. No such relabeling or
post-hoc superiority claim was performed.

## PTB-XL metadata feasibility, not an eligible cohort

The [official version1.0.3 page](https://physionet.org/content/ptb-xl/1.0.3/) and its
two metadata CSVs were read. Locally verified:21799 ECG records and18869 patient IDs.
`source_manifest.json` preserves URLs, byte counts and SHA256 values for all five
downloaded source files. Waveforms have not been downloaded for this preparation.

Using the **provisional** numeric join `HRnnnnn`→`ecg_id=nnnnn`, current-version
records map to17490 train,2162 validation,2146 test and1 unlisted. Requiring all
records of each patient **in the current version** to map to published test leaves
1685 records from1676 patients. The table and assumptions are preserved, not used
to select an evaluation cohort.

Two additional checks are required: verify cross-format recording identity, and
account for patient associations of records removed since the older Challenge
release. A patient whose surviving records are all test may still have a removed
training record. Full adapted-model exposure also remains unresolved. These counts
are feasibility bounds under a mapping assumption, not proof of independence.

The [official SCP dictionary](https://physionet.org/files/ptb-xl/1.0.3/scp_statements.csv)
distinguishes diagnostic, form and rhythm statements. Proposed mappings for a
prospective protocol are AFIB; STD_/STE_ for an ST-change surrogate; 1AVB/CLBBB/CRBBB
for a restricted conduction endpoint; PAC/PVC for ectopy. These are proposals, not
clinician-validated equivalences. STD_/STE_ are nonspecific ST changes, not proof of
acute ischemia; MI or all STTC must not silently replace the original STD/STE task.
Incomplete blocks, uncertain diagnoses, coexisting labels and missing labels need
explicit handling. A zero value on a listed form/rhythm code must not automatically
erase its presence. No external outcomes were consulted to choose mappings.

## Alternative validation path

The paper explicitly excludes INCART from pretraining, consistent with the public
split list. This makes it a useful candidate for further exposure checks. The
[original INCART database](https://physionet.org/content/incartdb/1.0.0/) contains
75 recordings from32 Holter sources, unlike the74-record Challenge subset. Its
beat annotations support an ectopy-oriented endpoint. Subject history alone does
not label ischemia or AF in every extracted10-second window. Windows must be grouped
by the documented patient/source and temporally annotated where the task requires.
Prior P1 evaluations already reported INCART findings, so it also is not untouched
by prior analysis. Do not call a reanalysis prospective independent validation.

## Validation and next steps

Three schema/namespace tests passed; Ruff passed after correcting one lint issue.
The two new scripts preserved source/output fingerprints. No frozen inference,
original protocol, checkpoint or historical raw result was changed.

Next: verify PTB-XL cross-version identity and complete patient associations; inspect
INCART annotation and prior-evaluation scope; freeze an explicit validation protocol
with the eight already-selected configurations and12-lead reference before outcomes.
Report per-disease performance and limitations even when a cohort cannot support
all four disease groups. Do not train a replacement model or claim verified ancestry.
Actual wearable recordings and the synchronized-reference acquisition specification
remain a separate third-stage requirement. The full authorized task remains open.
