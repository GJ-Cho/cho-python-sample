# EdgeSAM

> 담당 토픽: topics/segmenters.md · 작성일: 2026-10-29

## 요약

프롬프트-인-더-루프 증류로 SAM을 순수 CNN(RepViT 계열)에 담은 경량 SAM. 정확도·속도는 MobileSAM보다 좋지만 **라이선스가 비상업 전용**이라 추천 후보에서 제외, 비교용 카드.

## 분류

| 항목 | 값 |
|---|---|
| 파이프라인 유형 | [DS] 뒷단 |
| 입력 모달리티 | [RGB] |
| 맞는 시나리오 | (비교용) |
| 프롬프트 | 박스·점 (README에 AMG 언급 없음) |

## 라이선스

| | 라이선스 | 상용 가능 | 출처 |
|---|---|---|---|
| 코드 | **NTU S-Lab License 1.0** — "non-commercial purpose" 한정, 상업용은 저자 문의 | **불가** | 확인 — https://raw.githubusercontent.com/chongzhou96/EdgeSAM/master/LICENSE |
| 가중치 | 동일 라이선스 적용 | **불가** | 확인 — README "Redistribution and use should follow this license" |

## 학습 데이터

없음.

## 속도

| 수치 | 하드웨어 | 해상도 | 출처 (확인/추론) |
|---|---|---|---|
| 164.3 FPS (MobileSAM 103.5, SAM 4.3) | RTX 2080 Ti | 1024 | 확인 — https://raw.githubusercontent.com/chongzhou96/EdgeSAM/master/README.md |
| COCO mask AP 42.2 (SAM 46.1, MobileSAM 39.4), 9.6M params | — | — | 확인 — 동일 |

RTX 4060 Laptop 8GB 추정: 인코더 5~10ms (추론).

## 성숙도

- 코드·가중치·ONNX/CoreML 공개. 마지막 push 2025-05.

## 기준선(m1/m2) 대비

- 기대 이점: 빠르고 MobileSAM보다 정확.
- 위험: **비상업 라이선스 → 감점·제외.** 같은 계열 대안은 RepViT-SAM(Apache-2.0).

## 출처

- https://github.com/chongzhou96/EdgeSAM · https://arxiv.org/abs/2312.06660
