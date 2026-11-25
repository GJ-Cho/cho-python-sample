# CNOS

> 담당 토픽: topics/dino-backbone.md · 작성일: 2026-10-29

## 요약

SAM/FastSAM으로 마스크를 전부 제안하고 DINOv2 임베딩을 CAD 렌더 템플릿과 대조하는 [PM]의 원조(2023). BOP unseen 2D 검출의 표준 기준선. 우리 조건(CAD 없음)에서는 BOP24 model-free 모드(실사진 onboarding)로만 의미가 있다.

## 분류

| 항목 | 값 |
|---|---|
| 파이프라인 유형 | [PM] |
| 입력 모달리티 | [RGB] |
| 맞는 시나리오 | [ONE] [FEW] — 레퍼런스 필요. [UNK] 불가 |
| 프롬프트 | 예시 이미지(CAD 렌더 또는 onboarding 사진) |

## 라이선스

| | 라이선스 | 상용 가능 | 출처 |
|---|---|---|---|
| 코드 | MIT | 가능 | GitHub API `nv-nguyen/cnos` (확인) |
| 가중치 | SAM Apache-2.0 + DINOv2 Apache-2.0; FastSAM 모드는 Ultralytics **AGPL-3.0** | SAM 모드 가능 / FastSAM 감점 | README의 FastSAM 설치(ultralytics 8.0.135) (확인) |

## 학습 데이터

없음. 레퍼런스: CAD(렌더 템플릿) 또는 BOP24 static/dynamic onboarding 사진(확인).

## 속도

| 수치 | 하드웨어 | 해상도 | 출처 (확인/추론) |
|---|---|---|---|
| BOP-Core AP 42.8 / 0.22s (FastSAM), 36.1 / 1.85s (SAM) | 참가자 PC(미상, V100 기재 사례) | BOP | https://arxiv.org/html/2504.02812v2 Track 3 (확인) |
| BOP-Industrial AP 0.27~0.33 | — | — | https://bop.felk.cvut.cz/leaderboards/detection-unseen-bop23/bop-industrial/ (확인) |
| BOP-H3 model-free(static) AP 0.340 / 0.394s | — | — | model-free 리더보드 (확인) |

RTX 4060 Laptop 8GB 추정: SAM 모드 2~4s, FastSAM 모드 0.3~0.5s (추론) — BOP 수치 × 1.5~2.

## 성숙도

- 공식 코드·커스텀 데이터 튜토리얼(2023.7), BOP24 HOPE 튜토리얼(2024.5), 최근 push 2025-08
- 작은 물체는 `stability_score_thresh`를 0.97 → 0.5로 낮추라고 README가 권고(확인)

## 기준선(m1/m2) 대비

- 기대 이점: 레퍼런스 기반 필터링으로 배경·빈 벽 오검출 제거, 제품 라벨 부여.
- 위험: SAM 모드는 m1보다 느림. 인스턴스 분리 품질은 SAM AMG 그대로(m1 과분할 그대로). 무늬 없는 산업 부품에서 약함(BOP-Industrial 하위권).

## 출처

- https://github.com/nv-nguyen/cnos · arXiv 2307.11067
- BOP 2024 보고서 arXiv 2504.02812
