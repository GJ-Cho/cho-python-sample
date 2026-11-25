# MM-Grounding-DINO

> 담당 토픽: topics/detectors.md · 작성일: 2026-10-29

## 요약

OpenMMLab이 Grounding DINO를 오픈 재구현·재학습한 모델. 같은 구조(Swin-T/B/L)에서 zero-shot 성능이 크게 올랐고 transformers에 들어와 있어, **m2의 grounding-dino-tiny를 거의 코드 변경 없이 교체**할 수 있는 1순위 실험 후보다.

## 분류

| 항목 | 값 |
|---|---|
| 파이프라인 유형 | [DS] 앞단 |
| 입력 모달리티 | [RGB] |
| 맞는 시나리오 | [UNK] ○(LVIS 희귀 어휘 성능 향상), [BOX] △, [ONE]/[FEW] △ |
| 프롬프트 | 텍스트 |

## 라이선스

| | 라이선스 | 상용 가능 | 출처 |
|---|---|---|---|
| 코드 | Apache-2.0 (mmdetection / transformers) | 예 | (추론: 두 레포의 라이선스 기준. config README에는 라이선스 문구 없음) |
| 가중치 | 명시 없음 → 레포 Apache-2.0으로 봄 | 대체로 예. O365/GoldG/GRIT/V3Det 약관 쟁점 가능(추론) | [config README](https://raw.githubusercontent.com/open-mmlab/mmdetection/main/configs/mm_grounding_dino/README.md) |

## 학습 데이터

없음 (zero-shot). 필요하면 mmdetection으로 fine-tune할 수 있다 (같은 README, 확인).

## 속도

| 수치 | 하드웨어 | 해상도 | 출처 (확인/추론) |
|---|---|---|---|
| 공식 수치 없음 | — | — | config README (확인) |

RTX 4060 Laptop 8GB 추정: T 기준 GD-T와 비슷한 0.1~0.25s (추론). 같은 Swin-T + DETR 구조이고, 디코더 파라미터 공유 제거와 분류 헤드 개선이 주된 차이다.

## 성숙도

- 가중치 공개. transformers에 2025-08-01 추가, 예: `openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg_v3det` ([transformers doc](https://huggingface.co/docs/transformers/en/model_doc/mm-grounding-dino), 확인).
- B/L 일부 체크포인트는 "not fully trained" 표시가 있다 (확인).
- 성능(T, zero-shot): COCO 50.6, LVIS minival 41.4 / val 31.9. GD-T는 48.4 / 28.8 / 20.1 (확인).

## 기준선(m1/m2) 대비

- 기대 이점: 같은 API와 비슷한 속도로 LVIS val +11.8AP. 어휘 밖 물체 누락이 줄 가능성이 있다.
- 위험: "object." 같은 일반 어휘에서 이득이 있는지는 확인하지 못했다. transformers 버전 업그레이드가 필요할 수 있다.

## 출처

- https://raw.githubusercontent.com/open-mmlab/mmdetection/main/configs/mm_grounding_dino/README.md
- https://huggingface.co/docs/transformers/en/model_doc/mm-grounding-dino
- arXiv 2401.02361 (An Open and Comprehensive Pipeline for Unified Object Grounding and Detection)
