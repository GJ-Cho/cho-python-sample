# NIDS-Net

> 담당 토픽: topics/dino-backbone.md · 작성일: 2026-10-29

## 요약

Grounding DINO("objects" 같은 범용 프롬프트) + SAM으로 제안하고, 마스크 아래 DINOv2 패치 특징 평균(**FFA**, Foreground Feature Averaging) 임베딩을 레퍼런스와 매칭. 선택적으로 수 분 만에 학습되는 weight adapter로 임베딩을 다듬는다. **실사진 + 마스크 레퍼런스**를 전제로 설계돼 우리 조건에 맞는다.

## 분류

| 항목 | 값 |
|---|---|
| 파이프라인 유형 | [PM] (제안은 [DS] 구조) |
| 입력 모달리티 | [RGB] |
| 맞는 시나리오 | [ONE] [FEW] — one-shot/few-shot 명시(확인) |
| 프롬프트 | 예시 이미지(+마스크) |

## 라이선스

| | 라이선스 | 상용 가능 | 출처 |
|---|---|---|---|
| 코드 | MIT | 가능 | GitHub API `IRVLUTD/NIDS-Net` (확인) |
| 가중치 | Grounding DINO Apache-2.0 / SAM Apache-2.0 / DINOv2 Apache-2.0 | 가능 | (추론, 각 원 레포 기준) |

## 학습 데이터

basic: 없음. adapter: 레퍼런스 템플릿만으로 수 분 학습(확인: README "a few minutes").

## 속도

| 수치 | 하드웨어 | 해상도 | 출처 (확인/추론) |
|---|---|---|---|
| BOP-Core AP 46.9 / 0.49s (WA_Sappe), 43.1 / 0.49s (basic) | 참가자 PC | BOP | https://arxiv.org/html/2504.02812v2 (확인) |

RTX 4060 Laptop 8GB 추정: 0.8~1.2s (추론) — 구조가 m2(GDINO+SAM2)와 거의 같고 m2 warm이 0.96~1.22s.

## 성숙도

- 공개 코드, Colab one-shot 데모, 마지막 push 2025-05 (확인). 후속 L2G-Det(같은 연구실)이 2026에 나옴.

## 기준선(m1/m2) 대비

- 기대 이점: m2에 FFA 매칭만 붙이면 거의 NIDS-Net → 구현 비용 최소. 제품 라벨·오검출 필터.
- 위험: GDINO 제안이라 m2의 "어휘 밖 누락" 성격을 공유(단, "objects" 범용 프롬프트). 속도 개선 없음.

## 출처

- https://github.com/IRVLUTD/NIDS-Net · arXiv 2405.17859
