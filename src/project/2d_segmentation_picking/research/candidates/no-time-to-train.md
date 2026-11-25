# No time to train! (Training-Free Reference-Based Instance Segmentation)

> 담당 토픽: topics/dino-backbone.md · 작성일: 2026-10-29

## 요약

SAM2 제안 + DINOv2-L 메모리뱅크(레퍼런스 특징 집계) + semantic-aware 매칭으로, **레퍼런스 사진 몇 장만으로 다중 클래스 인스턴스 분할**을 학습 없이 수행(Edinburgh·Meta, 2025.7, 2026.2 개정). 코드 Apache-2.0이라 [FEW]의 [PM] 후보 중 라이선스가 가장 깔끔하다.

## 분류

| 항목 | 값 |
|---|---|
| 파이프라인 유형 | [PM] |
| 입력 모달리티 | [RGB] |
| 맞는 시나리오 | [FEW] 최적(다중 클래스 메모리뱅크), [ONE] 가능 |
| 프롬프트 | 예시 이미지 + 마스크(박스만 있어도 변환 스크립트 제공, 확인) |

## 라이선스

| | 라이선스 | 상용 가능 | 출처 |
|---|---|---|---|
| 코드 | Apache-2.0 | 가능 | GitHub API `miquel-espinosa/no-time-to-train` (확인) |
| 가중치 | SAM2 Apache-2.0 / DINOv2 Apache-2.0 | 가능 | README 체크포인트 링크 (확인) |

## 학습 데이터

없음. 레퍼런스: COCO 실험은 1/10/30-shot. 1~3-shot은 레퍼런스 선택에 성능이 민감(확인, 논문 4.3절).

## 속도

| 수치 | 하드웨어 | 해상도 | 출처 (확인/추론) |
|---|---|---|---|
| 0.93s/img 종단간 (SAM AMG 사용 시 3.5s, Matcher 120s) | A100, 20클래스 | COCO | https://arxiv.org/html/2507.02798v4 부록 A.6 (확인) |
| 모델 1.97GiB, 피크 11.5GiB(20클래스×10샷) | A100 | — | 같은 부록 (확인) |

RTX 4060 Laptop 8GB 추정: 2~3s (추론) — A100 대비 2~3배. 0.5/1s 목표 미달. SAM2 32×32 grid를 최상층 ROI·성긴 grid로 줄이거나 경량 SAM으로 바꿔야 함(논문도 MobileSAM 등 권고, 확인).

## 성숙도

- 연구 코드("expect a bit of chaos", 확인), 커스텀 데이터셋 가이드(2025.7), 활발한 push(2026-10).
- conda + 소스 설치. Windows 검증 없음(추론).

## 기준선(m1/m2) 대비

- 기대 이점: 여러 제품 구분과 오검출 제거, 라벨 0장.
- 위험: 느림. 논문 자체 실패 사례가 **밀집·겹침 장면 인스턴스 미검출, 작은 물체 누락, 유사 클래스 혼동**(확인, 부록 A.9) — 빈피킹과 정확히 겹침.

## 출처

- https://github.com/miquel-espinosa/no-time-to-train · arXiv 2507.02798
