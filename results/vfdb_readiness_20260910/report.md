# VT/VF expansion: directly verified data readiness

Date: 2026-09-10 KST. Status: data-format feasibility verified; model performance untested.

## Acquisition and observed contents

Official source: [MIT-BIH Malignant Ventricular Ectopy Database v1.0.0](https://physionet.org/content/vfdb/1.0.0/).
The official RECORDS list identifies22 records. All22 hea/atr/dat sets and the list
were downloaded to `data/raw/vfdb_1.0.0/`, without changing CPSC assets. The67 files
have URL, byte count and SHA256 in source_manifest.json. No download failed or was skipped.
The script decodes the digital signals and independently sums each channel modulo65536
to compare with the checksum in the WFDB header; all22 records pass.

All observed headers specify two channels,250Hz,525000 samples (2100 seconds each).
The actual retained files therefore total46200 seconds, or12h50m. The official
overview describes half-hour records; this audit uses the actual35-minute header
lengths and does not silently substitute the overview approximation. Channels are
named ECG/ECG, without verified standard lead identities or electrode positions.

There are592 annotation entries. Raw auxiliary markers include VT93, VF9, VFIB4,
VFL98 and NOISE73. These are marker counts, not independently adjudicated clinical
event counts, patient counts or positive10-second windows. Every raw sample index,
time, annotation symbol and auxiliary string is retained in raw_annotations.csv.
The official NOISE rule preserves the underlying previous rhythm until another
rhythm change; a simple last-label interval parser would mishandle this case.
No training labels or train/validation/test split have been generated.

## Feasibility conclusion

The existing P1 training source separates a backbone and output heads and accepts
the multiclass head size as a parameter. A new rhythm head or separate detector is
technically possible, but changing the output dimension does not confer VT/VF ability.
Waveforms and temporal rhythm annotations are now locally available for a bounded
first feasibility study. No new model was trained or evaluated in this audit.

Before training: resolve annotation boundaries/noise/unknown labels, check patient
and full model-exposure lineage, define development and held-out cohorts, establish
channel/preprocessing compatibility, and predefine episode/threshold metrics.
Current standard-lead shortlist positions cannot be inferred from generic ECG names.
VFDB alone does not provide population-representative normal person-days or prove
seven-day false-alert performance. ECG rhythm classification does not determine
pulselessness, and it must not be called validated cardiac-arrest diagnosis.

## Reproduction and attribution

Run `python scripts/audit_vfdb_readiness.py` with wfdb4.3.1/numpy1.26.4. Existing files
are reused rather than overwritten; their observed hashes remain explicit. This is
an integrity/format audit, not independently verified clinical label adjudication.
The22 source waveforms remain outside Git; the acquisition script and source hashes
permit recovery. Archive membership and off-device status must be checked separately.

Credit: Greenwald SD. *Development and analysis of a ventricular fibrillation detector*.
MIT master's thesis,1986; Albrecht, Moody and Mark, PhysioNet VFDB v1.0.0.
Dataset DOI: [10.13026/C22P44](https://doi.org/10.13026/C22P44).
Source files are under the [Open Data Commons Attribution License v1.0](https://physionet.org/content/vfdb/view-license/1.0.0/).
Use the official dataset page for the current requested PhysioNet citation.
