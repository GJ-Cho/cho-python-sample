# Qwen3-VL (2B / 4B) — VLM 그라운딩 검출

> 담당 토픽: topics/detectors.md · 작성일: 2026-10-29

## 요약

Alibaba의 범용 VLM으로 2D 그라운딩(박스·점 출력)을 지원한다. 2B/4B가 Apache-2.0이라 라이선스는 깨끗하지만, 생성형이라 느리고 밀집 다물체 검출용으로 설계되지 않았다. **오프라인 자동 라벨링·어려운 사례 검증용**으로만 의미가 있다(추론).

## 분류

| 항목 | 값 |
|---|---|
| 파이프라인 유형 | [DS] 앞단 (실시간 경로 부적합) |
| 입력 모달리티 | [RGB] |
| 맞는 시나리오 | [UNK] 참고(자연어로 "집을 수 있는 물체" 같은 지시 가능), [BOX]/[ONE]/[FEW] ✕ — 학습 데이터 자동 라벨 보조로는 가능(추론) |
| 프롬프트 | 텍스트(자연어 지시) |

## 라이선스

| | 라이선스 | 상용 가능 | 출처 |
|---|---|---|---|
| 코드 | Apache-2.0 | 예 | Qwen3-VL GitHub (미러 기준 확인) |
| 가중치 | Apache-2.0 (4B 카드 확인, 2B/8B는 3자 자료) | 예 | [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) (확인) |

참고: Qwen2.5-VL 일부 크기(예: 3B)는 Qwen Research License였던 것으로 알려져 있어 버전·크기별로 확인 필요 (추론).

## 학습 데이터

없음.

## 속도

공식 검출 지연 수치 확인하지 못함. RTX 4060 Laptop 8GB 추정: 2B fp16 1~3초/장(박스 수십 개 출력 시) (추론) — 출력 토큰 수에 비례.

## 성숙도

- 가중치 공개, transformers 지원, 2D grounding cookbook 있음 (검색 결과 기준 확인).

## 기준선(m1/m2) 대비

- 기대 이점: 자연어 조건("맨 위에 있는 상자") 처리.
- 위험: 속도, 박스 좌표 정밀도, 밀집 장면에서 누락·환각(추론).

## 출처

- https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct
- Qwen3-VL Technical Report
