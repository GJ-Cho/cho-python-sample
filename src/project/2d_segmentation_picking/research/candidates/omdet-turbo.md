# OmDet-Turbo

> 담당 토픽: topics/detectors.md · 작성일: 2026-10-29

## 요약

실시간을 목표로 한 Transformer 개방어휘 검출기(Efficient Fusion Head). Apache-2.0이고 transformers가 지원한다. Grounding DINO보다 빠른 텍스트 검출기가 필요할 때의 후보.

## 분류

| 항목 | 값 |
|---|---|
| 파이프라인 유형 | [DS] 앞단 |
| 입력 모달리티 | [RGB] |
| 맞는 시나리오 | [UNK]/[BOX]/[ONE]/[FEW] 모두 △ — 텍스트 어휘 의존은 GD와 같다 |
| 프롬프트 | 텍스트 (클래스 목록 + 태스크 문장) |

## 라이선스

| | 라이선스 | 상용 가능 | 출처 |
|---|---|---|---|
| 코드 | Apache-2.0 (transformers) | 예 | (추론) |
| 가중치 | Apache-2.0 | 예 | [HF 카드](https://huggingface.co/omlab/omdet-turbo-swin-tiny-hf) (확인) |

## 학습 데이터

없음.

## 속도

| 수치 | 하드웨어 | 해상도 | 출처 (확인/추론) |
|---|---|---|---|
| 모델 카드에 수치 없음. 논문 제목에 "Real-time" 주장 | — | — | HF 카드 (확인) |

RTX 4060 Laptop 8GB 추정: 0.03~0.1s (추론). 논문 표는 이번에 확인하지 못했다.

## 성숙도

- 가중치 공개, `OmDetTurboForObjectDetection` (확인).
- LVIS 등 정확도 수치는 확인하지 못했다.

## 기준선(m1/m2) 대비

- 기대 이점: 검출 단계 지연 단축.
- 위험: GD/LLMDet 대비 정확도 미확인. m2의 병목이 검출 단계가 아니라면 이득이 작다.

## 출처

- https://huggingface.co/omlab/omdet-turbo-swin-tiny-hf
- arXiv 2403.06892 (Real-time Transformer-based Open-Vocabulary Detection with Efficient Fusion Head)
