# SEEM / OpenSeeD / X-Decoder (2023 open-vocab 범용 분할 계열)

> 담당 토픽: topics/segmenters.md · 작성일: 2026-10-29

## 요약

Microsoft/IDEA의 2023년 open-vocabulary 범용 분할 모델 묶음. 텍스트·점·박스·참조 이미지 프롬프트로 instance/panoptic 마스크를 낸다. SAM3(2025-11) 이후 [E2E] open-vocab 분할에서 대부분 대체되었다고 판단해 한 장으로 묶는다.

## 분류

| 항목 | 값 |
|---|---|
| 파이프라인 유형 | [E2E] (open-vocab) |
| 입력 모달리티 | [RGB] |
| 맞는 시나리오 | (비교용) |
| 프롬프트 | 텍스트 / 점·박스 / 참조 이미지(SEEM) |

## 라이선스

| | 라이선스 | 상용 가능 | 출처 |
|---|---|---|---|
| 코드 | Apache-2.0 (SEEM, OpenSeeD 확인) | 가능 | 확인 — https://raw.githubusercontent.com/IDEA-Research/OpenSeeD/main/LICENSE , SEEM v1.0 LICENSE |
| 가중치 | 레포 따름 (학습 데이터 조건 별도) | 가능 (추론) | — |

## 학습 데이터

없음 (zero-shot).

## 속도

공식 지연 수치 확인하지 못함. Focal-T/L, Swin 백본 → 수백 ms (추론).

## 성숙도

- detectron2 계열 + 커스텀 op, 2023~2024 이후 정체 (추론).

## 기준선(m1/m2) 대비

- 기대 이점: SEEM의 참조 이미지 프롬프트는 [ONE] 레퍼런스 활용 개념상 유효.
- 위험: SAM3 대비 open-vocab 정확도 낮음(SAM3 논문 "기존 대비 2배" 주장), 설치 부담 → 채택 비추천.

## 출처

- https://github.com/UX-Decoder/Segment-Everything-Everywhere-All-At-Once · https://github.com/IDEA-Research/OpenSeeD · https://github.com/microsoft/X-Decoder
