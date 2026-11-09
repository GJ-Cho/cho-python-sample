# NOCTIS

> 담당 토픽: topics/dino-backbone.md · 작성일: 2026-10-29

## 요약

Grounded-SAM 2 제안 + DINOv2 CLS·패치 임베딩 매칭에 **cyclic thresholding**(반복 무늬·유사 패턴 오매칭 완화)과 제안 신뢰도 가중을 추가한 학습 없는 [PM](arXiv 2507.01463, CVPR 2026 프리프린트 표기). 논문은 BOP23 core 7종에서 RGB·RGB-D 최고 AP를 주장하지만 리더보드에는 미등재.

## 분류

| 항목 | 값 |
|---|---|
| 파이프라인 유형 | [PM] |
| 입력 모달리티 | [RGB] |
| 맞는 시나리오 | [ONE] [FEW] — 기본은 CAD 렌더 템플릿, 실사진 사용은 코드 수정 필요(추론) |
| 프롬프트 | 예시 이미지(템플릿) |

## 라이선스

| | 라이선스 | 상용 가능 | 출처 |
|---|---|---|---|
| 코드 | MIT | 가능 | GitHub API `code-iai/noctis` (확인) |
| 가중치 | Grounding DINO / SAM·SAM2.1 / MobileSAM / DINOv2 — 모두 Apache-2.0 계열(추론) | 가능 | README 다운로드 목록 (확인) |

## 학습 데이터

없음.

## 속도

| 수치 | 하드웨어 | 해상도 | 출처 (확인/추론) |
|---|---|---|---|
| 미확인 | — | — | README·초록에 수치 없음 |

RTX 4060 Laptop 8GB 추정: 1~1.5s (추론) — m2(Grounded-SAM 2) + DINOv2 매칭.

## 성숙도

- 코드 공개, 별 4개, 마지막 push 2025-11 (확인). USB 카메라 커스텀 데모 스크립트 있음.

## 기준선(m1/m2) 대비

- 기대 이점: 반복 무늬(같은 제품 여러 개가 깔린 빈) 오매칭 완화 아이디어는 우리 [FEW]에 직접 유용 — 아이디어만 차용 가능.
- 위험: CAD 템플릿 전제, 독립 재현 없음, 속도 개선 없음.

## 출처

- https://github.com/code-iai/noctis · https://arxiv.org/abs/2507.01463
