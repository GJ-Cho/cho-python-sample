# ZISVFM

> 담당 토픽: topics/input-modality.md · 작성일: 2026-10-29

## 요약

학습 없이 SAM을 **컬러맵 깊이 이미지**에 돌려 마스크를 제안하고, DINOv2 ViT 어텐션으로 비물체 마스크를 거른 뒤,
K-Medoids로 만든 점 프롬프트로 SAM(RGB)이 최종 마스크를 그리는 3단계 파이프라인. m1(SAM2 + 기하)과 구조가 가장 가까운
**학습 0 [RGB-D] 방식**이라 3단계 실험의 C4/C5 조건(topics/input-modality.md 7장) 설계 참고로 둔다.

## 분류

| 항목 | 값 |
|---|---|
| 파이프라인 유형 | [PM]에 가까움 (제안 → 필터 → 재프롬프트) |
| 입력 모달리티 | [RGB-D] (깊이로 제안, RGB로 정밀 분할) |
| 맞는 시나리오 | [UNK] — 물체 사이 깊이 차가 있을 때 강함. 밀착 파우치·같은 높이 박스([BOX])는 깊이 제안 단계가 합쳐버릴 위험 (추론, SAD 정성 관찰과 레포 노말맵 실험 근거) |
| 프롬프트 | 없음 (내부적으로 점 프롬프트 생성) |

## 라이선스

| | 라이선스 | 상용 가능 | 출처 |
|---|---|---|---|
| 코드 | **표기 없음** (GitHub API license=null) → 기본적으로 사용 허락 없음 | 불명 — 재구현 권장 | https://github.com/Yinmlmaoliang/zisvfm (확인) |
| 가중치 | 자체 가중치 없음. SAM(Apache-2.0), DINOv2(Apache-2.0) 사용 | 예 (구성요소 기준) | README Acknowledgements (확인), 각 라이선스는 `segmenters.md`/`dino-backbone.md` 참조 |

아이디어 자체는 단순하므로 기존 m1 코드(SAM2 + DINOv2)로 **재구현**하는 것이 라이선스·환경 양쪽에서 현실적이다 (추론).

## 학습 데이터

없음.

## 속도

| 수치 | 하드웨어 | 해상도 | 출처 (확인/추론) |
|---|---|---|---|
| 보고 없음 | RTX A6000 (실험 GPU) | — | [arXiv HTML](https://arxiv.org/html/2502.03266) (확인: 시간 미보고) |

RTX 4060 Laptop 8GB 추정: SAM ViT-H AMG(전체 이미지) + DINOv2 + 프롬프트 SAM → **수 초** (추론). SAM2 + 최상층 밴드 프롬프트로 바꾸면 m1의 약 2배(1~2s) (추론). 속도 예산 초과 가능성 높음.

## 성숙도

- 코드 공개(데모 notebook, python 3.9 conda). 마지막 push 2024-11, 스타 약 30 (GitHub API, 확인). 평가 코드는 UOAIS 것을 사용.
- 2025-02 arXiv. 게재지 미확인.

## 성능 (확인, [arXiv HTML](https://arxiv.org/html/2502.03266))

OCID Overlap / Boundary F / F@.75: SAM(RGB) 40.0 / 36.7 / 63.7 · SAM(컬러맵 깊이) 84.4 / 78.5 / 83.6 · **ZISVFM 89.2 / 83.5 / 87.0** (학습형 UOAIS-Net 89.8과 비슷).
OSD: SAM(RGB) 43.3 / 42.7 / 70.6 · SAM(깊이) 60.0 / 41.3 / 63.6 · ZISVFM 78.1 / 60.0 / 66.0.

## 기준선(m1/m2) 대비

- 기대 이점: 학습 0으로 학습형 UOIS와 동급(탁상 기준). 깊이 제안이 텍스처 과분할을 줄임 — m1 약점(과분할)과 직접 대응.
- 위험: 탁상 장면은 물체 사이 깊이 불연속이 거의 항상 있어 깊이 제안이 유리하다. **밀착 파우치에서는 레포 노말맵 실험처럼 평평한 물체를 놓치거나 합칠 가능성**(추론). SAM을 두 번 돌려 느림. 코드 라이선스 없음.

## 출처

- 논문: "ZISVFM: Zero-Shot Object Instance Segmentation in Indoor Robotic Environments with Vision Foundation Models", arXiv 2502.03266
- 레포: https://github.com/Yinmlmaoliang/zisvfm
- 관련: SAD — Segment Any RGBD (CC BY-NC 4.0, 렌더링 깊이를 SAM에 투입, 정성 비교만), https://github.com/jun-cen/SegmentAnyRGBD
