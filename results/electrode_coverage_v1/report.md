# Electrode-based P1 coverage exploration — verified results

## Scope and primary result

This study evaluates every nonempty subset of the 12 standard ECG leads (4095) using a frozen P1 multitask checkpoint. It reports a tradeoff between physical measurement contacts and four disease groups, not one clinically established optimal wearable configuration.
The historical CPSC test has 936 records: NSR130, AF180, ischemia165, conduction379 and ectopy82. The model was not retrained. All results below reuse this exploration test.

## Acquisition assumptions

I uses RA/LA, II RA/LL, III LA/LL; augmented limb leads require RA/LA/LL. Standard chest leads additionally require their chest site and RA/LA/LL for the reference. Count the union of measurement contacts; add one separately if the circuit needs DRL/ground. V1+V2 therefore represents five measurement contacts, not two electrodes. These are standard acquisition assumptions, not a measured wearable circuit or arbitrary torso placement equivalence.

## Frozen seed42 shortlist

Sens@95Sp is the maximum empirical sensitivity at specificity >=95% on this same test ROC. Thresholds have not been independently fixed for deployment. Macro-F1 and macro recall include NSR and all four disease groups.

|Input leads|Measurement contacts|With extra DRL|Macro-F1|AF Sens95|Ischemia Sens95|Conduction Sens95|Ectopy Sens95|
|---|---:|---:|---:|---:|---:|---:|---:|
|II|2|3|0.689280|0.905556|0.587879|0.794195|0.341463|
|I;III;aVR;aVF|3|4|0.687372|0.905556|0.648485|0.810026|0.426829|
|I;aVR|3|4|0.702420|0.900000|0.636364|0.810026|0.390244|
|I;III;aVR;aVL;aVF;V5|4|5|0.691547|0.911111|0.587879|0.796834|0.414634|
|II;aVR;aVL;aVF;V2|4|5|0.701130|0.900000|0.581818|0.810026|0.365854|
|II;aVR;V1;V2|5|6|0.704307|0.900000|0.606061|0.828496|0.341463|
|V1;V2|5|6|0.702400|0.911111|0.545455|0.852243|0.512195|
|12-lead baseline|9|10|0.694243|0.911111|0.587879|0.825858|0.341463|

## Repeatability and paired record uncertainty

Eight shortlisted configurations, including the baseline, were evaluated with seeds30000–30009. Every realization retains record IDs, labels and probabilities. Feature-mask variation is not training-run or external-cohort variation.
Paired record bootstrap uses 2000 draws, seed31415, on the fixed seed42 predictions. The same sampled records are used for each candidate and baseline. Draws missing any of five classes or either binary label are skipped without replacement draws. Intervals are unadjusted exploratory percentile intervals after test-based selection; they do not establish superiority or noninferiority.

|Input leads|10-seed mean Macro-F1|Seed SD|Mean paired delta vs12|Seed42 delta vs12 [95% CI]|
|---|---:|---:|---:|---|
|II|0.689365|0.006273|-0.000729|-0.004963 [-0.034351, +0.022905]|
|I;III;aVR;aVF|0.691759|0.005475|+0.001664|-0.006871 [-0.031583, +0.016865]|
|I;aVR|0.697627|0.004333|+0.007532|+0.008177 [-0.016613, +0.031367]|
|I;III;aVR;aVL;aVF;V5|0.689078|0.006620|-0.001017|-0.002696 [-0.024355, +0.016079]|
|II;aVR;aVL;aVF;V2|0.698156|0.003759|+0.008061|+0.006887 [-0.012560, +0.025276]|
|II;aVR;V1;V2|0.702679|0.004703|+0.012585|+0.010065 [-0.008248, +0.027450]|
|V1;V2|0.697595|0.004347|+0.007500|+0.008157 [-0.018667, +0.034231]|
|12-lead baseline|0.690094|0.003987|+0.000000|+0.000000 [+0.000000, +0.000000]|

The complete 18 primary metric intervals and seed summaries are in paired_intervals.csv and stability_summary.csv. Disease-specific weaknesses must be considered alongside the aggregate.
For V1+V2 (five contacts), mean repeated Sens95 is AF0.907222, ischemia0.550303, conduction0.848813 and ectopy0.470732 (ectopy range0.426829–0.512195). Thus the fixed-seed result meeting all four0.50 sensitivity floors is not stable across feature-mask realizations. Its seed42 ectopy delta vs12 is +0.170732 with an unadjusted exploratory95% interval[+0.030920,+0.283951], while ischemia delta is -0.042424. These test-selected findings motivate independent validation, not a superiority claim.
For two contacts, II is the minimal-acquisition candidate. At three contacts, I+aVR maximizes Macro-F1 while I+III+aVR+aVF maximizes worst-disease Sens95 under the frozen criterion. At five contacts, II+aVR+V1+V2 maximizes Macro-F1; V1+V2 improves the weakest disease sensitivity at an ischemia tradeoff. All non-baseline Macro-F1 difference intervals include zero. Increasing electrode count does not automatically improve every disease score in this frozen model.

## Coverage and Pareto frontier

The frontier contains 44 nondominated input subsets under measurement-contact count and the four Sens@95Sp objectives. Full available-lead configurations for all 67 distinct physical contact sets are retained separately. Selecting fewer derived input leads with the same contacts changes model input, not physical acquisition cost.
Coverage_grid.csv retains all4095 x5 predefined thresholds (0.50,0.60,0.70,0.80,0.90). These thresholds are sensitivity analyses, not clinical acceptance criteria.

|Sensitivity floor at specificity >=95%|Fewest contacts with all four groups meeting the floor|
|---|---|
|0.50|5|
|0.60|No configuration|
|0.70|No configuration|
|0.80|No configuration|
|0.90|No configuration|

## Integrity, timing and provenance

Run ID: `9dd890a3b933b877aaad4b8cfef1d5c589c46d48e39edb8f936cc75617d345c7`. Repeat ID: `59f35a0254ab1e0fe9f5dc2926f7dd5c781b4a1da02f35cc138699bf537d745a`.
Checkpoint SHA256: `287148bfd01ac67b5192268c6c69cc4cc230c004bdfdf332285f38a3b43b08dd`.
Raw CSV SHA256: `39926084d8de8dc06c4d683e4943c5090ed15df1d1938dbab2b3f8386deb6fee`.
RTX5070Ti, fp32, batch32, TF32 off; frozen legacy feature-mask behavior retained. Parent source/data/runtime identity was checked before inference and again during audit/repeats.
Expanded inference wall span, including interruption: 22279.21 seconds (6.189 hours). Sum of3301 new combination runtimes: 21601.69 seconds. Imported794 results were not rerun as part of this sum; controls, replay, repeats and analysis are additional work.
Original inference completed 2026-09-09 22:19:24 KST. One external process interruption occurred; cause unconfirmed. Saved1610 rows were audited before identical-condition resume. Boundary replay (last before interruption, first after, and12-lead) matched probabilities exactly (maximum error0). Original files were preserved.
Full audit checked4095 NPZ,102375 metrics,6 preflight files, unique subsets, original IDs/labels/hashes, with maximum metric error3.33e-16. Inference failure events0; observed external interruptions1. Repeated80 NPZ and2000 metrics,144 summary rows and144 intervals were independently rechecked via separate NumPy ROC/confusion calculations.
Bootstrap skipped draws per candidate: 0. No reroll.

## Interpretation limits and next stage

A low-contact candidate is a research shortlist, not established clinical reliability. Ectopy and ischemia sensitivity tradeoffs remain visible. The same internal test selected subsets and ROC operating points; bootstrap and feature-mask repeats do not remove selection bias, establish calibration, or prove independent generalization. These are classifications of current ECGs, not early prediction.
The original single-seed historical reproduction failure remains recorded; the separate predeclared39-seed stochastic gate passed. Neither fact should be substituted for the other.
Next, recover original training/validation IDs and warm-start lineage, audit external training exposure and label compatibility, and freeze the external evaluation protocol before viewing external test results. New overlapping CPSC splits are not independent validation. Actual wearable validity requires synchronized reference recordings and confirmed electrode/circuit geometry.
Existing reduced-lead and exhaustive reconstruction/electrode-cost publications are documented in records/paper_related_work_20260909.md. This study does not claim the first exhaustive search or absence of prior work.

## Retained evidence and backup

Protocols, source, CSVs and small verification manifests are locally versioned. Each raw row points to its retained prediction NPZ and SHA. Large predictions, checkpoints and waveforms remain outside Git. Review copies on the same disk are not an independent backup. Off-device backup of newly generated predictions has not yet been verified. Push is held for user review.
