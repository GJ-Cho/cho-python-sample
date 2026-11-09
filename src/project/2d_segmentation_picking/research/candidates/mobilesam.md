# MobileSAM / MobileSAMv2

> 담당 토픽: topics/segmenters.md · 작성일: 2026-10-29

## 요약

SAM-ViT-H 인코더를 TinyViT(5M)로 디커플드 증류한 최초의 대표 경량 SAM. MobileSAMv2는 AMG의 격자 프롬프트를 **object-aware 프롬프트 샘플링**으로 바꿔 segment-everything을 가속한다는 점에서 m1 가속 아이디어로 의미가 있다.

## 분류

| 항목 | 값 |
|---|---|
| 파이프라인 유형 | [DS] 뒷단 / [PM] 앞단(v2 everything) |
| 입력 모달리티 | [RGB] |
| 맞는 시나리오 | 속도 우선 [DS] 뒷단. 정확도는 EfficientViT-SAM·EdgeSAM보다 낮음 |
| 프롬프트 | 박스·점 / (v2) 객체 인식 프롬프트 |

## 라이선스

| | 라이선스 | 상용 가능 | 출처 |
|---|---|---|---|
| 코드 | Apache-2.0 | 가능 | 확인 — https://raw.githubusercontent.com/ChaoningZhang/MobileSAM/master/LICENSE |
| 가중치 | 레포 포함 (`mobile_sam.pt`), 별도 표기 없음 | 가능 (추론) | — |

MobileSAMv2의 object-aware 프롬프트 생성기는 YOLOv8 계열로 알려져 있어 **ultralytics AGPL 승계 가능성** — README에 명시 없음, (추론) 확인 필요.

## 학습 데이터

없음.

## 속도

| 수치 | 하드웨어 | 해상도 | 출처 (확인/추론) |
|---|---|---|---|
| 인코더 8ms + 디코더 4ms = 12ms (SAM-H 452ms) | "single GPU"(모델 미기재) | 1024 | 확인 — https://raw.githubusercontent.com/ChaoningZhang/MobileSAM/master/README.md |
| COCO mask AP 39.4 (EdgeSAM 표), 41.0 (TinySAM 표) | — | — | 확인 — EdgeSAM / TinySAM README |

RTX 4060 Laptop 8GB 추정: 인코더 10~20ms (추론).

## 성숙도

- 코드·가중치·ONNX export 공개. 2023년 이후 사실상 정체. 후속 경량 SAM들이 정확도에서 앞섬.

## 기준선(m1/m2) 대비

- 기대 이점: 매우 빠름.
- 위험: SAM-H 대비 AP 5~7 하락 → 경계·밀착 물체 분리 품질 저하. 단독 후보보다는 **비교 기준**.

## 출처

- https://github.com/ChaoningZhang/MobileSAM · https://arxiv.org/abs/2306.14289 · MobileSAMv2 https://arxiv.org/abs/2312.09579
