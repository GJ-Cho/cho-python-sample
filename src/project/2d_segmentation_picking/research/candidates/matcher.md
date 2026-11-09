# Matcher

> 담당 토픽: topics/dino-backbone.md · 작성일: 2026-10-29

## 요약

DINOv2로 레퍼런스-타깃 대응을 구하고 그 점들을 SAM 프롬프트로 써서 one-shot 분할(ICLR 2024). 개념적으로 중요한 기준점이지만 **속도와 라이선스로 탈락**.

## 분류

| 항목 | 값 |
|---|---|
| 파이프라인 유형 | [PM] (대응 기반 프롬프트 생성) |
| 입력 모달리티 | [RGB] |
| 맞는 시나리오 | [ONE] (one-shot semantic/part 분할) — 빈피킹 실사용 부적합 |
| 프롬프트 | 예시 이미지 1장 + 마스크 |

## 라이선스

| | 라이선스 | 상용 가능 | 출처 |
|---|---|---|---|
| 코드 | 학술용 BSD-2-Clause, **상업용은 저자 문의** | 불가(별도 계약) | https://raw.githubusercontent.com/aim-uofa/Matcher/main/README.md (확인) |
| 가중치 | SAM / DINOv2 Apache-2.0 | 가능 | (추론) |

## 학습 데이터

없음.

## 속도

| 수치 | 하드웨어 | 해상도 | 출처 (확인/추론) |
|---|---|---|---|
| ~120s/img (20클래스) | A100 | COCO | no-time-to-train 부록 표 9 (https://arxiv.org/html/2507.02798v4, 확인) |

RTX 4060 Laptop 8GB 추정: 1클래스여도 수 초 이상 (추론).

## 성숙도

- 코드 공개, 마지막 push 2025-12 (확인). 연구 코드.

## 기준선(m1/m2) 대비

- 기대 이점: 없음(후속 방법들이 대체).
- 위험: 속도, 상업 라이선스.

## 출처

- https://github.com/aim-uofa/Matcher · arXiv 2305.13310
