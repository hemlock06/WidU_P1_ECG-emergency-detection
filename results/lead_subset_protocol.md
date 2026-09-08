# 1–4리드 전수평가 사전 검증 기준

## 범위

동결된 P1 α=0.7 멀티태스크 모델의 현재 10초 ECG 분류 성능을 평가한다.
주지표는 Macro-F1, AF·허혈·전도장애·이소성의 AUROC, F1, argmax 민감도 및
one-vs-rest Sens@95Sp다. 응급 이진 헤드 지표는 참고값으로 분리한다.
조기예측, 의류형 전극의 신호품질, 임상적 유효성을 입증하는 실험이 아니다.

## 데이터·모델 적격성

- P1: `outputs/lora_multitask_snr_a07/lora_multitask_snr_best.pt`, epoch 18, α=0.7.
- 참고 ③: `outputs/lora_multisnr/lora_multisnr_best.pt`, epoch 30.
- P1 test: `cpsc2018_mc/test`, 936개, 단일라벨 `labels.npy`와 `labels_bin.npy`.
- 참고 ③ test: `cpsc2018/test`, 474개, `labels.npy`.
- 각 모델 내 모든 조합은 같은 레코드 순서와 같은 신호를 사용한다.
  두 모델의 데이터셋과 label taxonomy가 달라 직접 성능 우열을 합산하지 않는다.
- 입력은 float32 `(N,12,5000)`, raw mV, 정규화·잡음 추가 없음, 미사용 슬롯 0-fill.
- 원 평가 경로의 백본 `forward(mask=True)`는 `eval()`에서도 NumPy 특징 마스킹을
  수행한다. 이를 끄면 원 평가와 다른 조건이 된다. 기존 동작을 유지하되 매 조합의
  NumPy/PyTorch seed를 동일하게 재설정해 같은 특징 마스크 realization을 사용한다.
  배치 크기도 실행 지문에 포함한다. 이 조건은 5개 실제 레코드의 네 리드 구성에서
  원 평가 코드와 확률 `atol=rtol=1e-6` 일치로 검증했다. 전체 성능 재현과 구분한다.
- 레코드 ID, 신호, 라벨의 해시와 체크포인트 해시를 기록한다.
- `cpsc2018_mc_ml`을 `cpsc2018_mc`로 이름만 바꾸거나 936개로 자르지 않는다.
  현재 전처리 코드의 개정 이력과 구 split 차이는 `records/03_eval_results.md` §5e 참조.

## 전수실행 전 대조

문서 수치를 다른 모델에 적용하지 않는다. §⑥의 네 고정 조건 표는 ①·②·no-RLM이고,
§⑥-c의 표는 참고 ③ 모델이다. §⑥-b의 N별 평균은 특정 조합의 정답이 아니다.

참고 ③의 역사적 대조값(§⑥-c clean):

| 리드 | AUROC | Sens@95Sp |
|---|---:|---:|
| 전체 12 | 0.9469 | 0.7734 |
| I,II,V2,V5 | 0.9497 | 0.7875 |
| I,II | 0.9519 | 0.7762 |
| II | 0.9446 | 0.7564 |

P1의 역사적 12리드 대조값(§5f): 이진 AUROC 0.9139, 이진 F1 0.7887,
이진 Sens@95Sp 0.7072, Macro-F1 0.6858, 클래스별 AUROC
`[0.9304,0.9671,0.9066,0.9577,0.8634]`.

**역사적 대조 허용 절대오차를 평가 전에 고정한다:** AUROC 0.0021(문서의 실행 변동
±0.002와 네 자리 반올림), F1 0.005, Sens@95Sp 0.02(계단형 운영점과 기록된
반복 실행 변동). 초과 시 전수평가를 중단하며 통과시키기 위해 오차를 확대하지 않는다.
불일치 시 checkpoint/head, split ID·라벨 매핑, 신호 스케일, seed·precision·라이브러리를 조사한다.

P1의 고정 4·2·1리드 역사적 개별 수치는 현재 보존 표에서 확인되지 않는다.
이 세 조건의 원 코드 대조는 구현 회귀 검증으로 명시하고 역사적 수치 재현으로 부르지 않는다.
역사적 네 조건 재현 gate에는 참고 ③ 체크포인트가 필요하다.

## 열거·저장·분석 요건

- 표준 순서: I,II,III,aVR,aVL,aVF,V1,V2,V3,V4,V5,V6.
- 조합 수: 12 + 66 + 220 + 495 = 793개/모델. 12리드 대조는 별도 보존.
- 조합별 원시 지표와 레코드별 예측을 저장하고, 조합 단위 flush·재개·중복 검증을 한다.
- 식별자에 모델·데이터·전처리·평가설정·코드 해시를 포함하고 다른 실행 결과를 섞지 않는다.
- 계산 불가능한 값은 빈칸과 측정 가능 여부로 표기한다. 0으로 대체하지 않는다.
- N별 분포, 클래스별 top 10, 민감도 우선 Pareto 후보와 3→4리드 한계 이득을 산출한다.
- 최종 후보와 12리드의 paired record bootstrap 95% CI를 산출한다.
  테스트셋에서 후보를 선택한 후의 CI는 탐색적이며, 독립 평가에서의 우월성을 보증하지 않는다.
- 리드 수와 독립 전극 수를 구분한다. 사지 파생유도 중복을 고려해 성능 우선·최소 전극·
  흉부 포함·사지 전용·2채널 후보를 각각 보고한다.
- CSV의 793개 고유 조합 및 N별 이론 개수를 검증하고 원시 CSV에서 요약을 재생성한다.
- 실패·스킵·NaN/Inf 건수를 먼저 공개한다. 아직 평가하지 않은 조합을 실패 결과로 만들지 않는다.

## 실행 명령

복구 환경의 `.venv/Scripts/python.exe`를 사용한다. 필수 자산 누락 상태에서는 첫
명령도 차단된다. 기존 스크립트는 수정하지 않는다.

```powershell
.venv\Scripts\python.exe scripts\verify_lead_subset_assets.py --forward
.venv\Scripts\python.exe scripts\ablation_exhaustive_lead_subsets.py --stage controls
.venv\Scripts\python.exe scripts\ablation_exhaustive_lead_subsets.py --stage run
.venv\Scripts\python.exe scripts\summarize_exhaustive_lead_subsets.py
```

원시 결과: `results/exhaustive_lead_subsets_1to4.csv`.
레코드별 예측: `outputs/exhaustive_lead_subsets/<run_fingerprint>/`.
재개 시 같은 `--stage run` 명령을 사용하며 각 예측 파일의 해시까지 검사한다.
코드·환경·데이터·체크포인트가 달라지면 기존 실행에 덧붙이지 않는다.
