# Classification counts and longitudinal-volume limits

Date: 2026-09-10. This adds interpretable counts to the frozen experiment. It does
not change the model, threshold, candidate selection, or original predictions.

## Method and verification

All4095 seed42 prediction files and80 repeated candidate files were checked against
their saved hashes and original936 record IDs/labels. Each ECG is assigned its
highest-probability class (argmax). NumPy confusion counts were independently
checked with scikit-learn for all4175 runs and reproduced every saved Macro-F1
within1e-12. No prediction file failed or was skipped. Ruff passed.

`cohort_accuracy.csv` has4175 rows and `disease_counts.csv` has20875 rows covering
all five classes, including normal. TP/FN/FP/TN, precision, sensitivity, specificity
and F1 are preserved per run. These use the historical single-label reference.
The source and script fingerprints are in `verification.json`; every raw CPSC
header's SHA and duration are in `raw_record_durations.csv`.

## Selected configurations

| Input leads | Measurement contacts, excluding DRL | Correct /936, seed42 | Accuracy, seed42 | Mean accuracy, ten repeats |
|---|---:|---:|---:|---:|
| II | 2 | 705 | 75.32% | 75.46% |
| I + aVR | 3 | 716 | 76.50% | 76.38% |
| II + aVR + V1 + V2 | 5 | 723 | 77.24% | 76.98% |
| V1 + V2 | 5 | 722 | 77.14% | 76.53% |
| All12 leads | 9 | 714 | 76.28% | 76.15% |

These comparisons reuse the selection cohort and do not establish superiority.
Repeated mean counts would be fractional; the following table instead reports
actual integer counts from the fixed seed42 run for II+aVR+V1+V2.

| Reference class | Actual positive records | Found TP | Missed FN | Wrongly assigned FP | Sensitivity |
|---|---:|---:|---:|---:|---:|
| AF | 180 | 161 | 19 | 18 | 89.44% |
| ST-change/ischemia surrogate | 165 | 113 | 52 | 50 | 68.48% |
| Conduction | 379 | 318 | 61 | 34 | 83.91% |
| Ectopy | 82 | 30 | 52 | 36 | 36.59% |

FP means a record whose reference label is another class was assigned this class;
it can be another disease, not necessarily a healthy record. FN means the reference
class was not selected. These are classification errors, not operational alerts
or clinical diagnoses. They differ from the earlier Sens@95Sp ROC operating point.
The weak ectopy result remains visible even when overall accuracy is about77%.

## Actual data duration

The on-disk test array has shape936×12×5000 at500Hz. Its nominal input duration is
936×10 seconds =9360 seconds =2h36m, including short-record padding. The936 source
headers total14470.55 seconds (about4h1m) before truncation. All6877 local CPSC raw
headers total109674.976 seconds (about30h28m), with individual records6–144 seconds.
These are sums across separate recordings, not uninterrupted person-days. Unique
person counts are not established by record IDs. Re-evaluating4095 subsets or ten
seeds does not increase the unique data duration.

Seven continuous days for one person is604800 seconds =168 hours, equivalent in
duration to60480 nonoverlapping10-second windows. Our936-record input duration is
about1/64.6 of that, aggregated across records rather than one continuous person.
This time comparison is not a substitute for diversity, prevalence or diagnostic
coverage. The large backbone pretraining corpus does not supply seven-day patch
performance validation for the present adapted classifier.

For comparison, the manufacturer's
[Zio monitor description](https://www.irhythmtech.com/eu/en/solutions-services/irhythm-service/zio-monitor)
describes beat-to-beat collection for up to14 days, within a service that analyzes
arrhythmias. Its [instructions](https://go.irhythmtech.com/hubfs/Instructions%20for%20Use%20%28IFUs%29/EU/Zio%20monitor/LB10143.01-EU-ZIO-MONITOR-INSTRUCTIONS-FOR-USE-ENGLISH-DIGITAL-Cover-Pages-2.pdf)
describe continuous single-channel recording. Duration and lead count do not by
themselves establish matching indications, four-disease coverage or comparable
accuracy between this service and P1. No equivalence is claimed.

## Longitudinal validation still required

Real continuous recordings with person IDs, intended electrode placement,
reference ECG and episode timing are needed to measure event detection, missed
episodes, false alerts per person-day, time to detection, analyzable wear time and
signal loss during sleep/activity/contact changes. Alert persistence and merging
rules must be fixed before evaluation. Per-window FP cannot simply be multiplied
by60480 to claim a seven-day false-alert rate; prevalence, serial correlation,
signal quality and alert logic would differ.

The previously established pretraining exposure and incomplete warm-start lineage
remain. This is additional retrospective accounting, not independent validation,
early prediction, a seven-day study or evidence of clinical product equivalence.
