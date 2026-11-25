# DINO-X (Pro / Edge) — 비교용, 추천 제외

> 담당 토픽: topics/detectors.md · 작성일: 2026-10-29

## 요약

Grounding DINO 1.5를 기반으로 마스크·키포인트·영역 캡션 헤드와 **prompt-free 검출(universal object prompt)**, 시각 프롬프트를 더한 IDEA의 통합 모델. 기능은 [UNK]/[ONE]에 이상적이지만 **API 전용**이다.

## 분류

| 항목 | 값 |
|---|---|
| 파이프라인 유형 | [DS] 앞단 + 마스크 헤드([E2E]에 가까움) |
| 입력 모달리티 | [RGB] |
| 맞는 시나리오 | 참고용 (로컬 실행 불가) |
| 프롬프트 | 텍스트 / 시각 / prompt-free (`<prompt_free>`) |

## 라이선스

| | 라이선스 | 상용 가능 | 출처 |
|---|---|---|---|
| 코드 | API 예제 레포 Apache-2.0 | (API 호출 코드일 뿐) | [DINO-X-API](https://github.com/IDEA-Research/DINO-X-API) (확인) |
| 가중치 | **비공개 (API 전용)** | API 약관 | 같은 레포 "hosted on DeepDataSpace" (확인). 논문도 "demo and API will be released"만 언급 ([arXiv 2411.14347v3](https://arxiv.org/html/2411.14347v3), 확인) |

## 학습 데이터

없음.

## 속도

| 수치 | 하드웨어 | 해상도 | 출처 (확인/추론) |
|---|---|---|---|
| Edge 20.1 FPS (FP16) / 10.0 FPS (FP32) | Jetson Orin NX, TensorRT | 640×640 | 논문 (확인) |

정확도: Pro COCO 56.0, LVIS minival 59.8 / val 52.4. Edge COCO 48.7 @640², LVIS minival 44.5 @640² (확인). Pro 마스크 COCO 37.9 mask AP — 레포 스스로 Grounded SAM보다 낮다고 적음 (확인).

## 성숙도

- API·데모만. Edge 가중치 공개 정황 없음 (확인 범위 내).

## 기준선(m1/m2) 대비

- 기대 이점: prompt-free 검출의 상한 비교 기준.
- 위험: 네트워크·과금·외부 전송·비공개.

## 출처

- https://github.com/IDEA-Research/DINO-X-API
- https://arxiv.org/html/2411.14347v3 (DINO-X: A Unified Vision Model for Open-World Object Detection and Understanding)
