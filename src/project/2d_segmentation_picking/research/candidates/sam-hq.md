# SAM-HQ (HQ-SAM, Light HQ-SAM, HQ-SAM 2)

> 담당 토픽: topics/segmenters.md · 작성일: 2026-10-29

## 요약

SAM 디코더에 HQ output token과 글로벌-로컬 특징 융합을 추가해 **경계 품질**을 높인 모델(NeurIPS 2023). HQ-SAM 2(2024-11, beta)는 SAM2 기반. m1의 "경계 거침" 약점에 대한 뒷단 교체 후보.

## 분류

| 항목 | 값 |
|---|---|
| 파이프라인 유형 | [DS] 뒷단 / AMG 가능 |
| 입력 모달리티 | [RGB] |
| 맞는 시나리오 | [ONE][FEW] 파우치·얇은 구조 경계 · [BOX]는 박스 경계가 직선이라 이득 작음 (추론) |
| 프롬프트 | 박스·점 |

## 라이선스

| | 라이선스 | 상용 가능 | 출처 |
|---|---|---|---|
| 코드 | Apache-2.0 | 가능 | 확인 — https://raw.githubusercontent.com/SysCV/sam-hq/main/LICENSE |
| 가중치 | 별도 표기 없음(HF `lkeab/hq-sam`), 레포 Apache-2.0을 따른다고 봄 | 가능 (추론) | — |

## 학습 데이터

없음.

## 속도

| 수치 | 하드웨어 | 해상도 | 출처 (확인/추론) |
|---|---|---|---|
| Light HQ-SAM (TinyViT) 41.2 FPS | README 표 (GPU 미기재 확인 못함) | 1024 | 확인 — https://raw.githubusercontent.com/SysCV/sam-hq/main/README.md |

RTX 4060 Laptop 8GB 추정: HQ-SAM 2(SAM2 small급) ≈ SAM2 + 수~10% (추론) — 디코더 경량 추가.

## 성숙도

- 공식 코드·가중치, pip 패키지, 디코더 ONNX export 지원. HQ-SAM 2는 beta. 마지막 push 2025-09.

## 기준선(m1/m2) 대비

- 기대 이점: 경계 정밀도 → 마스크 erode 손실 감소, 평면 피팅 품질↑.
- 위험: 과분할 문제는 해결하지 않음(같은 프롬프트 모호성).

## 출처

- https://github.com/SysCV/sam-hq · https://arxiv.org/abs/2306.01567
