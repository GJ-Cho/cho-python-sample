# Rex-Omni (3B MLLM 검출기)

> 담당 토픽: topics/detectors.md · 작성일: 2026-10-29

## 요약

IDEA의 Qwen 기반 3B 멀티모달 LLM으로, 검출·포인팅·시각 프롬프트를 "다음 점 예측"으로 통합했다(arXiv 2510.12798). 가중치는 공개됐지만 **IDEA License 1.0 + Qwen Research License**라 상용에 제약이 있고, 3B LLM 생성 방식이라 1초 예산에 맞지 않는다. 참고 후보.

## 분류

| 항목 | 값 |
|---|---|
| 파이프라인 유형 | [DS] 앞단 |
| 입력 모달리티 | [RGB] |
| 맞는 시나리오 | 참고 ([UNK]/[ONE]/[FEW] 기능상 가능, 속도·라이선스 불가) |
| 프롬프트 | 텍스트 / 시각 프롬프트 |

## 라이선스

| | 라이선스 | 상용 가능 | 출처 |
|---|---|---|---|
| 코드 | 확인하지 못함 (GitHub LICENSE 미열람) | ? | https://github.com/IDEA-Research/Rex-Omni |
| 가중치 | **IDEA License 1.0**, Qwen 기반이라 **Qwen RESEARCH LICENSE AGREEMENT** 동시 적용 | 제약 큼 (감점) | HF 모델 카드 (검색 결과 기준 확인) |

## 학습 데이터

없음.

## 속도

공식 수치 확인하지 못함. RTX 4060 Laptop 8GB 추정: 1초 이상, bf16 3B로 VRAM 약 7GB+ → SAM2와 동시 적재 곤란 (추론). 물체 수만큼 토큰을 생성하므로 밀집 빈에서 더 느려진다.

## 성숙도

- 가중치 공개(2025.10). 로컬 실행 가능.

## 기준선(m1/m2) 대비

- 기대 이점: 자연어·시각 프롬프트 유연성.
- 위험: 라이선스, 속도, VRAM.

## 출처

- https://github.com/IDEA-Research/Rex-Omni
- https://rex-omni.github.io
- arXiv 2510.12798 (Detect Anything via Next Point Prediction)
