# P1 전수 리드 평가 — 자산 복구·환경 검증

> 후속 갱신: 홈 PC 원본을 복구했다. 현재 상태는 `lead_subset_home_recovery_report.md` 참조.
> 아래는 최초 복구 시점의 기록이며 당시 자산 부재 판정을 보존한다.

**상태: 전수평가 미실행.** 모델 forward는 통과했지만, 역사적 평가셋 및 참고 ③
체크포인트의 복구가 완료되지 않았다. 아래 작은 표본 검증은 성능 재현이 아니다.

## 복구한 자산

- 별도 P1 GitHub 저장소를 로컬 `P1` 디렉터리에 clone했다.
- brain 실제 위치 `C:/brain`에서 `git pull --rebase --autostash`를 완료하고
  `exchange/mac2win/2026-09-08T1525_mac_p1-exhaustive-lead-subsets-lab-pc.md`를 읽었다.
- Google Drive는 볼륨 이름과 실제 마운트를 열거해 찾았다. 이 실행에서 마운트는
  `G:/내 드라이브`였고, 요청한 `대학원/대학원 입시/03_연구프로젝트/WidU_AI논문_자료/models/`
  아래 P1 디렉터리에서 체크포인트를 **복사**했다. 원본은 보존했다.

| 자산 | 바이트 | SHA-256 |
|---|---:|---|
| P1 α=0.7 체크포인트 | 364840844 | `287148bfd01ac67b5192268c6c69cc4cc230c004bdfdf332285f38a3b43b08dd` |
| ECG-FM 공개 백본 | 1090825421 | `4d0142bcb485eb9f0c7845e0c19ff3463f6ae9d0e458eab69136efe90ceb9b7e` |

P1 원본·사본 해시는 일치했다. 백본 해시는 Hugging Face의 LFS SHA-256과 일치했다.
레포 문서의 `bowang-lab/ecg-fm` Hugging Face 주소는 HTTP 401을 반환했다.
[공식 ECG-FM GitHub](https://github.com/bowang-lab/ECG-FM)가 연결하는
[공개 모델](https://huggingface.co/wanglab/ecg-fm/blob/main/mimic_iv_ecg_physionet_pretrained.pt)은
`wanglab/ecg-fm`이며, 이 출처에서 받았다.

참고 ③ `outputs/lora_multisnr/lora_multisnr_best.pt`는 지정된 Drive 보관 폴더와
연구프로젝트·백업 파일명 검색에서 발견되지 않았다. 다른 모델로 대체하지 않았다.

## 환경과 forward

| 항목 | 실제 환경 |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 Ti, 16 GB |
| 드라이버 | 591.86 (nvidia-smi의 CUDA 13.1 표시는 드라이버 지원 수준) |
| Python | 3.10.21, 레포 내부 `.venv` |
| PyTorch / 런타임 CUDA | 2.7.1+cu128 / 12.8 |
| compute capability | 12.0 (`sm_120`); 휠 지원 아키텍처 목록에 포함 |
| NumPy / WFDB | 1.26.4 / 4.3.1 |
| fairseq-signals | f8f0ff1c788a82c2059cb452cd5462898867489e + 레포 LoRA 패치 |

과거 문서 환경(Python 3.9, torch 2.1.2+cu118, RTX 3060)과 다르다.
[PyTorch의 Blackwell 지원 릴리스](https://pytorch.org/blog/pytorch-2-7/)를 근거로
CUDA 12.8 빌드를 선택했다. 기존 requirements/environment 파일은 수정하지 않았다.

fairseq 패치는 Windows 줄바꿈 차이 때문에 일반 `git apply`가 실패했다.
`git apply --check --ignore-whitespace` 검증 후 같은 옵션으로 적용했다.
추론에 사용하지 않는 확장 빌드는 `READTHEDOCS=1`로 생략하고 editable 설치했다.
전체 설치 버전은 `lead_subset_recovery_environment.txt`에 보존했다.

- 원 `scripts/verify_env.py`: import·CUDA·GPU 확인 통과.
- GPU Conv1d: 유한 출력 `(1,16,4996)` 확인.
- 복구한 백본 + LoRA + 두 헤드: **실제 GPU forward 통과**.
  임베딩 `(1,768)`, 이진 `(1,)`, 다중분류 `(1,5)`, 모두 유한값.
- 체크포인트 metadata: epoch 18, α=0.7, rank 8, LoRA α=16,
  val composite 0.8051110704554267. LoRA 모듈 24개. 가중치 누락·추가 키 0개.

## 구현 대조 중 발견한 난수 문제

실제 다운로드된 ECG 중 클래스별 1개씩 총 5개를 사용한 구현 smoke에서
초기 최대 확률 차이는 0.08530116이었다. 원인은 `Wav2Vec2Model.forward(mask=True)`가
`eval()`에서도 `apply_mask()`를 호출하고, 이 함수가 NumPy 난수로 특징 마스크를
만드는 것이었다. 따라서 실행 간 차이를 cuDNN 비결정성만으로 설명할 수 없다.

같은 seed로 원 코드와 새 코드의 특징 마스크를 맞춘 뒤, 12리드 / I·II·V2·V5 /
I·II / II 모두 확률 `atol=rtol=1e-6` 비교를 통과했다.
레코드 ID·출력 shape·대조 결과는 `lead_subset_four_mask_smoke.json`에 있다.
새 스크립트는 매 조합에 같은 seed를 적용해 재개 순서의 영향을 제거한다.
기존 forward를 `mask=False`로 바꾸거나 기존 소스를 덮어쓰지 않았다.

**이 5개 레코드 대조는 구현과 하드웨어 검증이다. 936개 역사적 test 성능의
재현도, 새 성능 추정치도 아니다.**

## CPSC 복구 상태

원 다운로드 및 두 전처리 스크립트 실행을 완료했다. 다운로드는 약 80분,
이진 전처리는 8.60초, multi-label 전처리는 17.99초 걸렸다.
다운로드 성공 13,754개, 스킵 0개, 오류 1개(`REFERENCE.csv` HTTP 404)다.
원 전처리는 각 `.hea`의 `#Dx:`를 사용한다. 신호·헤더는 각각 6,877개이며
짝 누락·빈 파일 0개, 전처리 읽기 오류·스킵 0개다.

| 출력 | train / val / test | 판정 |
|---|---|---|
| `cpsc2018` | 2233 / 478 / 480 | 요구한 test 474개와 불일치 |
| `cpsc2018_mc` | 없음 | 요구한 구 5-class test 936개 미복구 |
| `cpsc2018_mc_ml` | 4813 / 1031 / 1033 | 현재 스크립트의 새 multi-label 데이터 |

이진 test는 `signals.npy` `(480,12,5000)`과 `labels.npy` `(480,)`이며,
정상 145개·응급 335개다. 새 multi-label test는 `signals.npy` `(1033,12,5000)`,
`labels.npy` `(1033,)`, `labels_bin.npy` `(1033,)`, `labels_mc.npy` `(1033,5)`를
실제로 생성했다. 두 데이터의 모든 split에서 NaN·Inf 0개, 중복 ID 0개다.
파일별 SHA-256·분포는 `lead_subset_preprocessing_audit.json`에 보존했다.

- 이진 test 파일 집합 지문: `cb58028111831c8c2dd459e700a698af3fb0a55e515cf2c25a97dee803257189`
- 새 multi-label test 파일 집합 지문: `6a632d25da16bc75811123e6cb6c31b48367fd05d593e18dd48ea7a4d2bb3934`

현재 `scripts/preprocess_cpsc2018_mc.py`는 2026-05-29 multi-label 개정본이다.
기본 출력은 `cpsc2018_mc_ml`이고 `labels_mc.npy`는 `(N,5)` multi-hot이다.
164884008 코드를 추가 복구해 구 단일라벨 split과 다르다.
최초 Git 이력의 파일도 이미 이 개정본이므로, Git에서 개정 전 코드를 복구하지 못했다.
`records/03_eval_results.md` §5e는 새 test 1024개 중 773개가 구 train/val과 겹쳤다고
기록한다. 이 데이터를 구 936개 test 대신 사용해서는 안 된다.

**원본 모집단도 달랐다.** `records/04_run_history.md`의 2026-05-24 다운로드 기록은
13,755개 대상 중 성공 13,704개·타임아웃 51개이며, `.hea` 6,839개와 `.mat` 6,865개만
저장됐다고 명시한다. 현재 공식 목록은 `.hea`·`.mat` 각각 6,877개다.
과거 누락 `.hea` 38개의 전체 목록은 이 기록에 없고, 누락 `.mat`도 일부 5개 이름만
기재돼 있다. `split_records()`는 전체 인식 레코드 목록을 정렬한 뒤 shuffle하므로
원본 집합이 다르면 seed=42만으로 같은 test ID를 복원할 수 없다.
이 문제는 구 단일라벨 매핑을 복원하는 것만으로도 해결되지 않으며 이진 test에도 적용된다.
과거의 `record_ids.npy` 또는 전체 누락 목록 없이 레코드를 임의 제외해 표본 수를 맞추지 않는다.

## 코드·검증·실행 경계

추가한 파일:

- `scripts/verify_lead_subset_assets.py`: 자산 해시·shape·고유 ID·유한값·GPU forward 감사.
- `scripts/smoke_exhaustive_lead_subsets.py`: 고정된 실제 레코드 5개로 네 리드 조건의
  원 평가 코드와 예측 확률 일치 검증. 전체 test 성능 검증으로 사용하지 않는다.
- `scripts/ablation_exhaustive_lead_subsets.py`: 네 조건 대조 gate, 793개/모델 열거,
  지표·레코드별 예측 저장, 파일 해시 및 실행 지문을 검사하는 재개 경로.
- `scripts/summarize_exhaustive_lead_subsets.py`: 조합 수 검증, CSV 요약 재생성,
  클래스별 후보·Pareto·paired record bootstrap CI. 실제 완주 데이터로는 미검증.
- `tests/test_exhaustive_lead_subsets.py`: 열거·마스킹·지표·누락 자산 차단·요약 무결성·
  파생유도 전극 수·재개 난수 고정 검증.

기존 회귀 테스트는 PyTorch 2.6 이후 `torch.load` 기본값 변경으로 최초 2건의 로딩
오류가 발생했다. 검증한 파일에 한해 `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`을 적용하면
기존 테스트 8개가 통과했다. 새 감사·실행 코드의 P1 체크포인트 로딩은
`weights_only=False`를 명시한다. 이 환경 변수는 원 테스트 실행 프로세스에만 설정했다.

사전 대조값·허용오차·실행 명령은 `lead_subset_protocol.md`에 고정했다.
최종 테스트는 기존 8개와 신규 7개를 합쳐 **15 passed, 1 warning (3.86초)**이며,
신규 Python 파일 Ruff 검사와 Git 공백 검사를 통과했다.
실행 gate는 필수 자산 누락으로 중단됐다. **역사적 성능 대조 0건, 전수 조합 평가 0건**이다.
따라서 조합별 성능 CSV·순위·최종 하드웨어 후보를 아직 산출하지 않았다.
P1 결과를 조기예측 또는 의류 전극 성능으로 확대하지 않는다.

재개에는 과거 `cpsc2018/test` 및 `cpsc2018_mc/test`의 `record_ids.npy`와
`labels*.npy`, 참고 ③ 체크포인트가 필요하다. 원 test ID를 확보하면 현재 원신호에서
해당 신호를 복구하고 라벨·지문을 검증할 수 있다. 역사적 성능 대조를 다시 통과하기 전에는
793개 조합 평가를 실행하지 않는다.

모든 기존 추적 소스는 유지했다. 코드·감사 결과를 로컬 커밋 대상으로 검증했으며 push는 미실행이다.
