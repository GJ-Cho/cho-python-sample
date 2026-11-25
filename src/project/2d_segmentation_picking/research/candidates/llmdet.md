# LLMDet

> 담당 토픽: topics/detectors.md · 작성일: 2026-10-29

## 요약

MM-Grounding-DINO를 LLM이 생성한 이미지·영역 캡션으로 공동학습해 개방어휘 성능을 더 끌어올린 모델(CVPR 2025). transformers 4.55.0+에서 Grounding DINO와 같은 API로 로드된다. MM-GDINO와 함께 m2 앞단 교체 1순위.

## 분류

| 항목 | 값 |
|---|---|
| 파이프라인 유형 | [DS] 앞단 |
| 입력 모달리티 | [RGB] |
| 맞는 시나리오 | [UNK] ○, [BOX] △, [ONE]/[FEW] △ |
| 프롬프트 | 텍스트 |

## 라이선스

| | 라이선스 | 상용 가능 | 출처 |
|---|---|---|---|
| 코드 | Apache-2.0 | 예 | [README](https://raw.githubusercontent.com/iSEE-Laboratory/LLMDet/main/README.md) (확인) |
| 가중치 | 별도 명시 없음 → Apache-2.0으로 봄 | 대체로 예. 학습 데이터 약관 쟁점 가능(추론) | 같은 README |

## 학습 데이터

없음 (zero-shot).

## 속도

| 수치 | 하드웨어 | 해상도 | 출처 (확인/추론) |
|---|---|---|---|
| 공식 수치 없음 | — | — | README (확인) |

RTX 4060 Laptop 8GB 추정: Swin-T 0.1~0.25s, Swin-B는 그 약 2.5배 (추론). LLM은 학습에만 쓰고 추론 구조는 MM-GDINO와 같다(추론: 논문 구조 기준).

## 성숙도

- 가중치 공개(Swin-T/B/L). transformers 4.55.0에 공식 병합, `AutoModelForZeroShotObjectDetection`로 로드 (확인).
- LVIS minival / val: T 44.7 / 34.9, B 48.3 / 38.5, L 51.1 / 42.0 (확인).

## 기준선(m1/m2) 대비

- 기대 이점: GD-T 대비 LVIS val +14.8AP(T). venv의 transformers 버전만 맞으면 config의 `model_id` 교체 수준이다.
- 위험: transformers 업그레이드가 현 venv(SAM2, torch 2.6)와 충돌하는지 확인하지 못했다. 일반 어휘 recall이 오르는지도 미확인.

## 출처

- https://raw.githubusercontent.com/iSEE-Laboratory/LLMDet/main/README.md
- arXiv 2501.18954 (LLMDet: Learning Strong Open-Vocabulary Object Detectors under the Supervision of Large Language Models)
