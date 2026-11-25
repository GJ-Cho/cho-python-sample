# DINOv2

> 담당 토픽: topics/dino-backbone.md · 작성일: 2026-10-29

## 요약

Meta의 자기지도 ViT 백본(2023). 마스크를 직접 내지 않지만 [PM]의 매칭 엔진(CNOS·SAM-6D·NIDS-Net·no-time-to-train)과 학습형 검출기(RF-DETR) 백본으로 쓰인다. Apache-2.0이라 이 조사의 **기본 특징 추출기**다.

## 분류

| 항목 | 값 |
|---|---|
| 파이프라인 유형 | [PM] 매칭 단계 / [E2E]·[DS] 학습형의 백본 |
| 입력 모달리티 | [RGB] |
| 맞는 시나리오 | [ONE] [FEW]: 제품 종류 구분에 강함. [UNK]: 병합 보조 신호만(같은 제품 인스턴스 구분 불가) |
| 프롬프트 | 예시 이미지(레퍼런스 임베딩) |

## 라이선스

| | 라이선스 | 상용 가능 | 출처 |
|---|---|---|---|
| 코드 | Apache-2.0 | 가능 | https://raw.githubusercontent.com/facebookresearch/dinov2/main/README.md (확인) |
| 가중치 | Apache-2.0 | 가능 | 같은 README (확인). XRay-DINO·Cell-DINO 등 파생은 비상업 — 혼동 주의 |

## 학습 데이터

없음(매칭) / 제품당 마스크 5~20개(kNN·linear probe 분류, 추론) / 라벨 수십~수백 장(RF-DETR 백본으로 파인튜닝, 추론)

## 속도

| 수치 | 하드웨어 | 해상도 | 출처 (확인/추론) |
|---|---|---|---|
| ViT-S 21M / B 86M / L 300M / g 1.1B, 패치 14 | — | — | README (확인) |
| DINOv2-B ≈ 40ms | RTX A4000 | 미상 | 검색 요약(arXiv 2508.00746), 정확도 낮음 |

RTX 4060 Laptop 8GB 추정: 전체 이미지 1회 + 마스크 패치 평균(FFA) 기준 ViT-B 0.05~0.1s, ViT-L 0.15~0.3s (추론) — FLOPs(ViT-B 224² ≈ 17.6G) × 해상도 비율, 실효 10~20 TFLOPS 가정.

## 성숙도

- 공식 코드·가중치 공개, `torch.hub`·HF `transformers` 정식 지원, registers 버전 제공
- ONNX/TensorRT 변환 사례 다수(RF-DETR가 TRT 배포)

## 기준선(m1/m2) 대비

- 기대 이점: m1/m2 마스크에 "어느 제품인가" 라벨과 오검출(빈 벽·배경) 필터를 붙인다. 오버헤드 작음.
- 위험: 인스턴스 분리 품질은 개선하지 못함(앞단 의존). 무늬 없는 부품 간 구분 약함. 14px 패치라 작은 물체 임베딩이 배경에 오염.

## 출처

- https://github.com/facebookresearch/dinov2 · arXiv 2304.07193
- Registers: arXiv 2309.16588
