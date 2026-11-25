# UOAIS-Net

> 담당 토픽: topics/input-modality.md · 작성일: 2026-10-29

## 요약

보이는 마스크·아모달 마스크·가림 여부를 계층적으로 예측하는 RGB-D UOIS 모델(detectron2 기반). 사실적 합성 데이터 UOAIS-Sim(5만 장) 학습.
가림 여부 출력은 "위에 아무것도 없는 물체부터 집기"에 직결되어 피킹에 유용하다. 단 라이선스가 비상업.

## 분류

| 항목 | 값 |
|---|---|
| 파이프라인 유형 | [E2E] |
| 입력 모달리티 | [RGB] / Depth / [RGB-D] (변형별 config: `R50_rgbdconcat_...`, `R50_depth_...`) |
| 맞는 시나리오 | [UNK] (가림 추론이 빈 클러터에 의미 있음). [BOX] 같은 높이 밀착은 불리 (추론) |
| 프롬프트 | 없음 |

## 라이선스

| | 라이선스 | 상용 가능 | 출처 |
|---|---|---|---|
| 코드 | 커스텀: "non-commercial scientific research, non-commercial education ... Any other use, in particular commercial ... prohibited" | **아니오** | [LICENSE.md](https://raw.githubusercontent.com/gist-ailab/uoais/main/LICENSE.md) (확인) |
| 가중치 | 별도 표기 없음(Google Drive). 코드 라이선스 적용으로 보는 것이 안전 | **아니오** (추론) | [README](https://raw.githubusercontent.com/gist-ailab/uoais/main/README.md) |

## 학습 데이터

없음(공개 가중치). UOAIS-Sim 50,000장 / 1,000 클러터 장면, 사실적 렌더링 (확인, 논문).

## 속도

| 수치 | 하드웨어 | 해상도 | 출처 (확인/추론) |
|---|---|---|---|
| 약 0.13s/이미지 | Titan XP | 640x480 추정 | [ar5iv 2109.11103](https://ar5iv.labs.arxiv.org/html/2109.11103) (확인, 해상도는 추론) |

RTX 4060 Laptop 8GB 추정: 0.1~0.2s @640x480 (추론). 이 문서의 RGB-D 모델 중 가장 빠르다.

## 입력 ablation (확인, 표 열 정렬은 추출 결과 기준)

OCID Overlap F: RGB 67.9 / Depth **89.8** / RGB-D 71.9. OSD: RGB 83.8 / Depth 85.5 / RGB-D 85.2.
**RGB-D가 Depth 단독보다 OCID에서 크게 낮다** — 저자: OCID의 복잡한 텍스처 배경이 RGB-D 모델의 전경 분할을 해쳤다.

## 성숙도

- 코드·가중치 공개, RealSense·Azure Kinect 데모, ROS 노드 포함. python 3.7~3.8, torch 1.8~1.9, detectron2 v0.5/0.6, `setup.py build develop` (README, 확인).
- 마지막 push 2025-07, 스타 약 160 (GitHub API, 확인).

## 기준선(m1/m2) 대비

- 기대 이점: 빠름, 가림 여부 출력(최상층 판정을 학습으로 대체 가능), 깊이 단독 변형은 텍스처 과분할이 없음.
- 위험: **상용 불가 라이선스.** 구 환경. 탁상·선반 도메인 → 빈 탑다운 밀착 파우치는 미검증.

## 출처

- 논문: Back et al., "Unseen Object Amodal Instance Segmentation via Hierarchical Occlusion Modeling", ICRA 2022, https://arxiv.org/abs/2109.11103
- 레포: https://github.com/gist-ailab/uoais
