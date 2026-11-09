# FastSAM

> 담당 토픽: topics/segmenters.md · 작성일: 2026-10-29

## 요약

YOLOv8-seg를 SA-1B 2%로 학습해 "segment everything"을 CNN 한 번으로 수행하는 모델. AMG를 한 번의 순전파로 대체한다는 점에서 m1 속도 문제의 직접 대안처럼 보이지만, **라이선스 표기 모순(README Apache vs LICENSE AGPL)** 과 낮은 마스크 품질이 문제.

## 분류

| 항목 | 값 |
|---|---|
| 파이프라인 유형 | [PM] 앞단 (everything) / 점·박스·텍스트 선택 |
| 입력 모달리티 | [RGB] |
| 맞는 시나리오 | [UNK] 빠른 제안기 (품질 낮음) |
| 프롬프트 | 없음(everything) / 점 / 박스 / 텍스트(CLIP) |

## 라이선스

| | 라이선스 | 상용 가능 | 출처 |
|---|---|---|---|
| 코드 | **LICENSE 파일은 AGPL-3.0**, README는 "Apache 2.0" 표기 — 모순. 레포는 `CASIA-LMC-Lab/FastSAM`으로 이전 | 불명확 → AGPL로 간주 | 확인 — https://raw.githubusercontent.com/CASIA-IVA-Lab/FastSAM/main/LICENSE , README |
| 가중치 | YOLOv8x/s 기반(ultralytics) | AGPL 간주 | 확인(README "YOLOv8x based") / 해석은 (추론) |

## 학습 데이터

없음.

## 속도

| 수치 | 하드웨어 | 해상도 | 출처 (확인/추론) |
|---|---|---|---|
| ~40ms (프롬프트 수 무관), SAM-H everything 64×64 6972ms | RTX 3090 | 1024 | 확인 — README |
| 전체 파이프라인 ~64ms (MobileSAM 측 측정) | 미상 GPU | — | 확인 — MobileSAM README |

RTX 4060 Laptop 8GB 추정: 60~120ms (추론).

## 성숙도

- 2024-07 이후 정체. ultralytics 패키지에 통합.

## 기준선(m1/m2) 대비

- 기대 이점: everything 모드가 m1 대비 10배 이상 빠름.
- 위험: AGPL 간주, YOLO 프로토타입 마스크라 경계가 SAM보다 거침, 밀착 물체 분리 약함 (추론). 우선순위 낮음.

## 출처

- https://github.com/CASIA-LMC-Lab/FastSAM · https://arxiv.org/abs/2306.12156
