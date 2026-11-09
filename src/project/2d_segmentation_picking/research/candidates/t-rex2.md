# T-Rex2 — 비교용, 추천 제외

> 담당 토픽: topics/detectors.md · 작성일: 2026-10-29

## 요약

IDEA의 텍스트+시각 프롬프트 검출기(ECCV 2024). 예시 물체에 박스를 그리면 비슷한 물체를 모두 찾는(카운팅) 방식이라 [ONE]/[FEW]에 개념상 정확히 맞지만, 공개된 것은 **API 호출 코드**뿐이다.

## 분류

| 항목 | 값 |
|---|---|
| 파이프라인 유형 | [DS] 앞단 |
| 입력 모달리티 | [RGB] |
| 맞는 시나리오 | 참고용 ([ONE]/[FEW] 개념 일치, 로컬 불가) |
| 프롬프트 | 예시 박스(양성·음성), 교차 이미지 프롬프트, 텍스트 |

## 라이선스

| | 라이선스 | 상용 가능 | 출처 |
|---|---|---|---|
| 코드 | API 예제 코드 (라이선스 미표기) | — | [T-Rex repo](https://github.com/IDEA-Research/T-Rex) |
| 가중치 | **비공개 (DeepDataSpace API)** | API 약관 | 같은 레포, API 토큰 필요 (확인: 검색 결과 기준, 가중치 다운로드 경로 없음) |

## 학습 데이터

없음 (예시 박스 몇 개).

## 속도

공개 수치 없음 (확인하지 못함).

## 성숙도

- API 전용. 같은 아이디어의 공개 대안은 YOLOE 시각 프롬프트(AGPL), OWLv2 `image_guided_detection`(Apache), Rex-Omni(IDEA License).

## 기준선(m1/m2) 대비

- 기대 이점: 시각 프롬프트 검출 성능의 비교 기준.
- 위험: 네트워크·과금·외부 전송.

## 출처

- https://github.com/IDEA-Research/T-Rex
- arXiv 2403.14610 (T-Rex2: Towards Generic Object Detection via Text-Visual Prompt Synergy)
