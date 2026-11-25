# SAM-6D ISM (Instance Segmentation Model)

> 담당 토픽: topics/dino-backbone.md · 작성일: 2026-10-29

## 요약

SAM-6D(CVPR 2024)의 앞단. SAM/FastSAM 제안 + DINOv2 의미 점수 + 외관 점수 + **CAD 기반 기하 점수**로 CNOS를 개선. CAD가 없으면 기하 점수를 못 쓰므로 우리 조건에서는 참고용.

## 분류

| 항목 | 값 |
|---|---|
| 파이프라인 유형 | [PM] |
| 입력 모달리티 | [RGB] (FastSAM RGB 변형) / [RGB-D] |
| 맞는 시나리오 | [ONE] [FEW] — **CAD 있을 때만** |
| 프롬프트 | CAD 렌더 템플릿 |

## 라이선스

| | 라이선스 | 상용 가능 | 출처 |
|---|---|---|---|
| 코드 | 명시 없음(GitHub API spdx 없음) | 불명 → 감점 | `JiehongLin/SAM-6D` (확인) |
| 가중치 | SAM Apache-2.0 / FastSAM AGPL-3.0 / DINOv2 Apache-2.0 | 구성별 | (확인) |

## 학습 데이터

ISM은 학습 없음. CAD 필수(데모 입력에 `CAD_PATH`, 확인).

## 속도

| 수치 | 하드웨어 | 해상도 | 출처 (확인/추론) |
|---|---|---|---|
| BOP-Core AP 47.1 / 0.45s (FastSAM, RGB-D), 44.4 / 0.25s (FastSAM, RGB), 44.9 / 2.80s (SAM) | 참가자 PC | BOP | https://arxiv.org/html/2504.02812v2 (확인) |
| BOP-Industrial 0.419 (FastSAM RGB-D) | — | — | 리더보드 (확인) |

RTX 4060 Laptop 8GB 추정: FastSAM 0.5~0.8s, SAM 4s+ (추론).

## 성숙도

- 코드·Docker 공개, 마지막 push 2024-07. 유지보수 정지 상태.

## 기준선(m1/m2) 대비

- 기대 이점: CAD가 생기는 산업 부품 고객 시나리오에서 CNOS보다 정확.
- 위험: CAD 의존, 라이선스 불명, 유지보수 없음.

## 출처

- https://github.com/JiehongLin/SAM-6D · arXiv 2311.15707
