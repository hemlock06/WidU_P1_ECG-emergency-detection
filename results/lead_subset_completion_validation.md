# 전수평가 완료 검증

- GPU: NVIDIA GeForce RTX 5070 Ti 16 GB, compute capability 12.0.
- 환경: Python 3.10.21, torch 2.7.1+cu128, CUDA 12.8, float32, batch32, TF32 off.
- 전수평가: 2026-09-08 19:21:34–21:33:55 KST, 약 2시간 12분 21초(벽시계).
- 조합별 runtime 합: P1 5,280.20초, 참고 이진 2,644.12초. 모델 로딩 등은 벽시계 시간과 구분한다.
- 별도 확률적 대조: 195회, 801.99초. 후보 난수 안정성: 4고유 구성 × 10 seeds, 264.19초.
- 전수평가 1,586행 = 각 모델 793행 = 12 + 66 + 220 + 495.
- 전수평가 실패 0, 재개 스킵 0. 이전 사전 단계 중단 로그 3건은 삭제하지 않았다.
- 저장 예측 NPZ 1,586개의 SHA-256·레코드 순서·원 테스트 라벨 전부 일치.
- 별도 confusion-matrix/full-ROC 계산으로 22,204지표 재계산: 최대 절대차 3.33e-16.
- 원 자산·코드 identity 파일 15개 재해시 일치. 전체 identity는 후보 반복 실행에서도 확인.
- raw CSV에서 summary 재생성 바이트 일치.
- 후보별 paired record bootstrap 2,000회, 70개 구간, 스킵 0.
- 후보 안정성 요약 90행을 원 반복 CSV에서 다시 계산: 차이 0.
- `python -m pytest tests -q`: 19 passed, 1 warning, 4.71초.
  기존 테스트의 신뢰된 역사적 체크포인트 로드에 `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` 적용.
- 전체 폴더의 무제한 pytest 수집은 work에 보관한 colorama와 설치된 colorama 테스트 간
  import mismatch 5건으로 중단됐다. 프로젝트가 추적하는 tests 디렉터리 전체를 지정해 위 결과를 확인했다.
- 후처리 해석 스크립트 첫 실행의 Path/str 인자 오류는 수정 후 실제 전체 산출물로 재실행해 통과했다.
  이는 모델 전수 추론의 실패 수와 구분한다.
- 새 스크립트 lint 및 Git whitespace 검사 통과. 원 n-lead 스크립트·동결 실행 identity 파일은 변경하지 않았다.

기계 판독 근거는 exhaustive_lead_subsets_verification.json 및
exhaustive_lead_subsets_analysis_verification.json에 보존한다.
과거 단일 seed 역사값 대조 실패는 미검증 상태로 남으며 stochastic reproduction 통과와 구분한다.
