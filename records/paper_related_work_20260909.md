# 관련 문헌 확인 및 주장 범위

확인일 2026-09-09. 실험 시작 후 수행한 표적 문헌 조회다. 체계적 문헌고찰이나
문헌 부재 증명은 아니다. 최초 설계가 아래 새로 확인한 논문에 근거했다고 소급하지 않는다.
검색어: `ECG lead selection optimal subset electrodes deep learning multi label classification paper`,
`ECG-FM open electrocardiogram foundation models lead agnostic Oh 2022`,
`Optimal ECG-lead selection increases generalizability`, `Comparing ECG Lead Subsets`,
`Optimization of Arrhythmia-based ECG-lead Selection` 및 해당 제목의 PMC 제한 검색.
출처의 수치를 우리 성능과 직접 비교하지 않았다. 라벨·코호트·학습과 선택 절차가 다르다.

| 1차 문헌 | 직접 확인한 관련 방법 | P1 기록에 대한 함의 |
|---|---|---|
| [Lai 등, Optimal ECG-lead selection increases generalizability of deep learning on ECG abnormality classification, 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8805596/) | 8종 이상 분류, forward stepwise selection; II/aVR/V1/V4 조합, 외부 평가 | 최적 리드 선택 자체는 선행연구가 있다. 우리 모델에서 V2가 선택된 것을 문헌 반증이나 생리학적 보편성으로 해석하지 않는다. |
| [Oh 등, Lead-agnostic Self-supervised Learning for Local and Global Representations of Electrocardiogram, CHIL2022](https://proceedings.mlr.press/v174/oh22a.html) | 임의 리드에 강건하도록 random lead masking을 제안 | 임의 리드 대응의 방법론적 근거. 특정 물리 전극 조합의 최적성을 대신 입증하지 않는다. |
| [Comparing ECG Lead Subsets for Heart Arrhythmia/ECG Pattern Classification: Convolutional Neural Networks and Random Forest](https://pmc.ncbi.nlm.nih.gov/articles/PMC11886372/) | 8종 패턴별 CNN/RF 및 RFE; 질환·방법에 따라 선택이 다름; global optimum 한계 명시 | 평균 하나보다 질환별 성능이 중요하다는 비교 문헌. 연구별 패턴 분류와 P1의 묶인5-class 라벨은 동일하지 않다. |
| [Zhang 등, Systematic benchmark of reduced-lead configurations for 12-lead ECG reconstruction: multi-model evaluation across all possible subsets, 2026-06-29](https://www.frontiersin.org/journals/cardiovascular-medicine/articles/10.3389/fcvm.2026.1856211/full) | PTB-XL에서 4094개 부분집합, 재구성 후 동결 분류기, 전극 비용. Table2에서 LR/Ridge/LightCNN은 전수, Transformer는32개라고 구분 | 매우 가까운 비교 대상. ‘전수열거 최초’ 및 ‘전극 수 최초 고려’ 주장을 기각한다. P1은 재구성 없이 동결 모델에 zero-fill 입력하며 네 질환군을 평가한다. |

Frontiers 논문의 전극 비용식은 사지유도가 포함될 때4개 고정 비용을 부과하며 흉부만
선택한 경우 별도 사지 기준 비용을 포함하지 않는다(§3.4). 현재 P1 프로토콜은 표준
흉부유도 기준을 위한 RA/LA/LL도 세고 DRL을 따로 표시한다. **서로 다른 계수 가정**이므로
전극 수 또는 효율 점수를 그대로 비교하지 않는다. 해당 논문의 실제 회로/기준 구현을
검증한 것은 아니다. 우리 쪽도 실제 장치 검증이 추가로 필요하다.

## 방어 가능한 배경 문장

“축소 리드 선택과 전극 부담을 고려한 선행연구가 존재한다. 그러나 이들 연구의
조합 선택을 특정 동결 ECG 기반 분류기에 그대로 적용할 수 있는지는 별도의
검증이 필요하다. 본 연구는 P1 멀티태스크 모델에서 표준 전극 의존관계를 명시하고,
동일 기록의 리드 부분집합에 따른 질환별 분류 성능과 평가 변동을 조사한다.”

이는 연구 질문의 정당화이지 아직 입증하지 않은 문헌적 최초성 선언이 아니다.
논문 기여는 최종 외부 검증·비교 결과와 추가 관련 문헌 검토 후 확정한다.
AF·허혈·전도·이소성이라는 그룹명도 세부 진단과 라벨 매핑을 적어야 한다.

## 후속 문헌 검토 항목

1. 위 논문들의 공개 코드·보충자료에서 데이터 split 및 실제 전극 기준 가정을 대조한다.
2. 같은 frozen direct-classification 설정의 선행 exhaustive benchmark를 추가 검색한다.
3. 착용 위치·reference/DRL·피부 접촉 품질은 실제 회로 문서와 계측 근거를 추가한다.
4. 최종 원고에 DOI/권호/페이지를 1차 출판사 metadata와 대조해 참고문헌을 완성한다.

이번 확인에서는 PMC 원문 검색 색인과 PMLR/Frontiers 본문을 읽었다. PMC 직접 open의
일부는 브라우저 확인 페이지를 반환했으며 해당 문헌은 본문 검색 결과로 확인했다.
검색 성공을 전체 문헌 수집이나 코드 재현으로 확대하지 않는다.
