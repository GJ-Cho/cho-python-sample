# Grounding DINO 1.5 (Pro / Edge) · 1.6 (Pro) — 비교용, 추천 제외

> 담당 토픽: topics/detectors.md · 작성일: 2026-10-29

## 요약

IDEA의 Grounding DINO 1.0 후속. 대규모 학습으로 zero-shot 성능을 크게 올렸으나 **가중치가 공개되지 않은 클라우드 API 전용**이다(DeepDataSpace 토큰 필요). research/README 규칙에 따라 성능 비교용 카드만 둔다.

## 분류

| 항목 | 값 |
|---|---|
| 파이프라인 유형 | [DS] 앞단 |
| 입력 모달리티 | [RGB] |
| 맞는 시나리오 | 참고용 (로컬 실행 불가) |
| 프롬프트 | 텍스트 |

## 라이선스

| | 라이선스 | 상용 가능 | 출처 |
|---|---|---|---|
| 코드 | API 예제 레포 Apache-2.0 | (API 호출 코드일 뿐) | [Grounding-DINO-1.5-API](https://github.com/IDEA-Research/Grounding-DINO-1.5-API) (확인) |
| 가중치 | **비공개 (API 전용)** | API 이용 약관에 따름 | 같은 레포: "hosted on DeepDataSpace", API 토큰 필요 (확인) |

## 학습 데이터

없음 (zero-shot, API).

## 속도

| 수치 | 하드웨어 | 해상도 | 출처 (확인/추론) |
|---|---|---|---|
| 1.5 Edge 10.7 FPS | Jetson Orin NX, TensorRT | 640×640 | [arXiv 2405.10300](https://arxiv.org/pdf/2405.10300) 표 (확인) |
| 1.5 Edge 75.2 FPS | A100, TensorRT FP32 | 800×1333 | 같은 논문 (확인. 일부 2차 자료가 이를 Orin NX 수치로 오인) |

정확도: 1.5 Edge COCO 45.0 @800×1333 (확인). 1.6 Pro COCO 55.4, LVIS minival 57.7 / val 51.1 (확인, API 레포).
RTX 4060 추정: 해당 없음 — 클라우드 왕복 지연이 추가돼 0.5초 목표에 불리(추론).

## 성숙도

- 1.6 Edge는 언급 없음. 로컬 실행 불가.

## 기준선(m1/m2) 대비

- 기대 이점: 오픈 모델의 상한 비교 기준(LLMDet-L LVIS val 42.0 vs 1.6 Pro 51.1).
- 위험: 네트워크 의존, 과금, 데이터 외부 전송, 가중치 비공개.

## 출처

- https://github.com/IDEA-Research/Grounding-DINO-1.5-API
- https://arxiv.org/pdf/2405.10300 (Grounding DINO 1.5: Advance the "Edge" of Open-Set Object Detection)
