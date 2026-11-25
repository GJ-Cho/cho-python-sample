# UnSAMv2 (granularity 제어 SAM2) · Semantic-SAM 참고

> 담당 토픽: topics/segmenters.md · 작성일: 2026-10-29

## 요약

UC Berkeley(2025-11). SAM-2에 **granularity encoder + granularity-aware mask token**을 붙여 "점 하나 + 연속 granularity 값(0~1)"으로 부분↔전체 수준을 직접 고르게 한 모델. 무라벨 6K장 자기지도 + 파라미터 0.02% 추가. m1의 과분할(점 프롬프트의 부분/전체 모호성)을 **구조적으로** 겨냥하는 거의 유일한 후보.

## 분류

| 항목 | 값 |
|---|---|
| 파이프라인 유형 | [PM] 앞단 (whole-image segmentation at granularity) / [DS] 뒷단 |
| 입력 모달리티 | [RGB] |
| 맞는 시나리오 | [UNK] ◎ (m1 대체 1순위 실험 대상) · [BOX] 라벨·테이프 과분할 억제 기대 |
| 프롬프트 | 점 + granularity 스칼라 / 없음(전체 분할, granularity 지정) |

## 라이선스

| | 라이선스 | 상용 가능 | 출처 |
|---|---|---|---|
| 코드 | Apache-2.0 | 가능 | 확인 — https://raw.githubusercontent.com/yujunwei04/UnSAMv2/main/LICENSE |
| 가중치 | Apache-2.0 (HF 카드) | 가능 | 확인(검색 요약) — https://huggingface.co/yujunwei04/UnSAMv2 |

주의: 전작 **UnSAM(v1)은 CC-BY-NC**(비상업) — 확인 (https://github.com/frank-xwang/UnSAM README License 절). 가중치 학습 파이프라인이 CutLER(CC-BY-NC) 의사라벨을 쓰므로 학습 재현 시 주의 (추론).

## 학습 데이터

없음 (공개 가중치 사용).

## 속도

공개 지연 수치 확인하지 못함. SAM-2 + 경량 모듈이므로 SAM2 동일 크기와 비슷 (추론). 전체 분할 모드는 AMG와 같은 격자 프롬프트 방식이면 m1과 같은 시간대 (추론, 확인 필요).

RTX 4060 Laptop 8GB 추정: m1과 유사(≈1s), granularity로 중복 후보가 줄면 NMS·병합 후처리 감소 (추론).

## 성숙도

- 코드·가중치·HF 데모 공개(인터랙티브 / whole image / video). 연구 코드, 2025-11 이후 활동 적음 (추론).
- 참고: **Semantic-SAM**(UX-Decoder, 6단계 granularity 출력, Apache 계열 추정) — LICENSE 파일 미검출(GitHub API null), 검증 필요. detectron2 기반이라 Windows 설치 부담 (추론).

## 기준선(m1/m2) 대비

- 기대 이점: "객체 수준" granularity 하나로 고정 → 부분 마스크 생성 자체 억제, 평면 병합 후처리 부담 감소.
- 위험: 학습 도메인이 자연 이미지. 박스 위 라벨처럼 "객체 vs 부분" 경계가 애매한 산업 장면에서 고정 granularity가 장면마다 맞을지 미지수.

## 출처

- https://arxiv.org/abs/2511.13714 · https://yujunwei04.github.io/UnSAMv2-Project-Page · https://github.com/yujunwei04/UnSAMv2
- Semantic-SAM: https://github.com/UX-Decoder/Semantic-SAM
