# TinySAM

> 담당 토픽: topics/segmenters.md · 작성일: 2026-10-29

## 요약

Huawei Noah의 경량 SAM(전단계 증류 + 양자화 Q-TinySAM). **"hierarchical segment everything"** 전략으로 everything 모드 프롬프트 수를 줄이는 아이디어가 m1 가속에 직접 참고가 된다.

## 분류

| 항목 | 값 |
|---|---|
| 파이프라인 유형 | [DS] 뒷단 / [PM] 앞단(hierarchical everything) |
| 입력 모달리티 | [RGB] |
| 맞는 시나리오 | [UNK] AMG 가속 아이디어 |
| 프롬프트 | 점·박스 / 없음 |

## 라이선스

| | 라이선스 | 상용 가능 | 출처 |
|---|---|---|---|
| 코드 | Apache-2.0 | 가능 | 확인 — https://raw.githubusercontent.com/xinghaochen/TinySAM/main/README.md |
| 가중치 | 레포 Apache-2.0 따름 | 가능 (추론) | — |

## 학습 데이터

없음.

## 속도

| 수치 | 하드웨어 | 해상도 | 출처 (확인/추론) |
|---|---|---|---|
| COCO AP 42.3 (Q-TinySAM 41.4, MobileSAM 41.0, SAM-H 46.6), 42.0 GFLOPs | — (FLOPs만) | 1024 | 확인 — README |
| hierarchical everything 가속 "성능 저하 거의 없음" — 정량 수치 README엔 없음 | — | — | 확인 — README |

RTX 4060 Laptop 8GB 추정: 인코더 10~25ms (추론).

## 성숙도

- 코드·가중치 공개, 마지막 push 2025-01.

## 기준선(m1/m2) 대비

- 기대 이점: 계층적 everything = 저밀도 격자로 먼저 돌리고 **미커버 영역에만** 고밀도 프롬프트 → m1의 프롬프트 수 절감 기법으로 SAM2에도 이식 가능 (추론).
- 위험: 모델 자체 정확도는 SAM2-small보다 낮을 가능성.

## 출처

- https://github.com/xinghaochen/TinySAM · https://arxiv.org/abs/2312.13789
