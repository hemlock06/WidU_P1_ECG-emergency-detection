# 전극 전수평가 후처리 준비 및 중단·재개 기록

2026-09-09 17:49 KST. 상태: **추론 재개 확인, 후처리 코드 준비; 전체 결과 미완료**.

## 실행 중단과 재개

17:37:13 이후 raw/status는1610행에서 멈췄다. 17:46 확인 시 실제 Python 프로세스가
없었고 원 실행 도구 세션40520은 Unknown process id를 반환했다. 원 stdout 로그에
traceback이 없고, 조회한17:35 이후 Windows Application Error/Windows Error Reporting
사건도0개였다. 부팅 시각은13:09:42였다. **중단 원인은 확인하지 못했다.**
상태 파일의 running과 실패0만으로 정상 실행을 판단하지 않는다.

재개 전 독립 검사에서 저장1610행(부모794+새816)의 ID/라벨/예측해시와40250개
지표를 재계산했다. 최대 절대차3.330669e−16, 등록되지 않은 예측파일0개,
모델/데이터/원 실행 코드 identity 일치. 근거는
`results/electrode_resume_audit_20260909_1748.json`이다.
원 raw와 저장예측을 보존하고 같은 runner/프로토콜로 남은2485개를 재개했다.
사건은 `events.jsonl`의 external_interruption_observed로 남겼다. 추론 예외0과
**관측된 프로세스 중단1회**는 별개이며 중단을 실패0 표현으로 감추지 않는다.

17:48 Windows Start-Process Hidden으로 별도 프로세스를 시작했다.
동작 확인 시 wrapper8984/실제23796이 존재했고 raw가1614행까지 증가했다.
새 stdout/stderr:

- `work/electrode_coverage_resume_20260909_1748.stdout.log`
- `work/electrode_coverage_resume_20260909_1748.stderr.log`

기존 `work/electrode_coverage_run_20260909.log`는 최초 실행 기록으로 보존한다.
새 실행도 외부 종료/전원 손실에 면역인 것은 아니다. 이후 status+실제process+새로그를
함께 확인하며 중복 실행하지 않는다. 총 실행시간은 두 run_started 및 중단 간격과
조합별 runtime을 구분해서 계산해야 한다.

## 준비한 후처리와 실행 순서

기존 추론 runner·프로토콜은 수정하지 않았다. 새로 작성한 파일:
`scripts/analyze_electrode_coverage.py`, `scripts/repeat_electrode_candidates.py`,
`tests/test_electrode_analysis.py`.

아래는 P1 루트의 `.venv\Scripts\python.exe`로 실행한다. **전체4095 추론 종료 후에만**
실행하며, 단계별 오류가 나면 다음 명령으로 넘어가지 않는다.

```powershell
.venv\Scripts\python.exe scripts\verify_electrode_restart_parity.py
.venv\Scripts\python.exe scripts\analyze_electrode_coverage.py --stage audit
.venv\Scripts\python.exe scripts\analyze_electrode_coverage.py --stage analyze
.venv\Scripts\python.exe -u scripts\repeat_electrode_candidates.py --stage repeat
.venv\Scripts\python.exe -u scripts\repeat_electrode_candidates.py --stage finish
```

1. audit: OS 실행잠금, complete 상태, parent identity, 정확4095조합, 원본794행,
   새예측경로·ID·라벨·해시,25지표/행 독립 fullROC/confusion 계산,6대조,
   중단/재개 감사와 실패·재사용·신규 수를 확인한다.
   그 전에 verify_electrode_restart_parity.py로 중단 직전/직후 조합 및12리드의
   전체936개 확률을 seed42/batch32로 다시 계산한다. 새NPZ를 별도 경로에 보존하고
   기존확률과1e-6 오차/argmax 일치 대조를 통과해야 최종 audit를 허용한다.
   현재 이 재추론은 미실행이며 원 전체 추론이 끝나면 수행한다.
2. analyze: 감사 해시를 대조한 뒤 모든 조합 요약, 네 질환 Sens@95Sp Pareto,
   전체5개 coverage threshold, 정확전극수별 최고macro/최저질환민감도 최적 후보,
   완전가용 유도67개 물리구성과 반복 shortlist를 저장한다.
3. repeat: 고정seeds30000–30009로 모든 shortlist를 재추론하며 개별 NPZ와 지표를
   매번 보존한다. 이미 있는 예측을 덮어쓰지 않는다. 후보별 매번 동일 seed를 쓴다.
4. finish: 모든 반복 NPZ를 독립 재계산하고 paired record bootstrap2000 seed31415,
  18개 주지표의 차이/CI/스킵 수와 seed변동 요약을 만든다.

macro-F1 정확동률은 사전기준이 별도 명시하지 않았으므로 재현 가능한 사전적
lead-index 순서로 정했다. 최저질환민감도 기준의 동률처리는 고정 프로토콜의
질환평균→Macro-F1→lead-index 순서를 구현했다. 실제 확장 결과로 동률 규칙을
선택한 것은 아니다.

## 검증 범위

`pytest tests/test_electrode_analysis.py tests/test_electrode_coverage.py -q`:
10 passed,2.33초. Ruff 및 whitespace검사 통과.
Pareto의 동률/질환별 상충, 후보 동률처리,4095공간/67물리구성/전체coverage grid,
중복·전극메타데이터변조, 미완료 상태 차단, 예측해시/ID변조, 원 양성대조4조건의
저장확률에서100개 지표 재계산, paired bootstrap의 동일후보0차이 및 미충족표본
스킵(재추출 금지)을 확인했다. 모델 forward를 테스트로 추가 실행하지 않았다.

아직 실제4095 결과의 audit/analyze/repeat/finish를 실행한 것은 아니다.
bootstrap/반복 전체 결과, 요약 독립 재검수와 최종 해석 보고서가 남는다.
`candidate_analysis.json`은 computed_pending_final_analysis_review로 남도록 했으며,
이 파일 생성만으로1단계 완료/독립외부검증 진입을 선언하지 않는다.
후속 최종 검수는 원시값에서 요약·CI 재현 및 모든 산출물 해시·실패 수를 확인하고,
전체 조합/후보의 네 질환별 결과와 한계를 보고한다. 그 후2단계로 간다.

## 연구 해석 경계

coverage grid는 탐색용 민감도 기준이며 임상 허용기준이 아니다. Sens@95Sp는 동일
test의 경험ROC에서 추정된다. 같은test 후보 선택 편향과 기록단위 bootstrap의 한계,
기존 단일seed 역사대조 실패는 유지한다. 프로세스 중단 원인을 GPU/모델/앱 문제로
추정해 확정하지 않는다. P1 push·재학습은 수행하지 않는다.
