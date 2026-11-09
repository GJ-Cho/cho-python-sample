# EfficientSAM (Meta, SAMI 사전학습)

> 담당 토픽: topics/segmenters.md · 작성일: 2026-10-29

## 요약

Meta(yformer)의 경량 SAM. SAM 특징을 복원하는 마스크드 이미지 사전학습(SAMI)으로 ViT-T/ViT-S 인코더를 학습. EfficientSAM3·EfficientTAM과 같은 저자 계열. 이름이 EfficientViT-SAM과 혼동되기 쉬우니 주의.

## 분류

| 항목 | 값 |
|---|---|
| 파이프라인 유형 | [DS] 뒷단 |
| 입력 모달리티 | [RGB] |
| 맞는 시나리오 | [DS] 뒷단 (비교용) |
| 프롬프트 | 점·박스 |

## 라이선스

| | 라이선스 | 상용 가능 | 출처 |
|---|---|---|---|
| 코드 | Apache-2.0 | 가능 | 확인 — https://raw.githubusercontent.com/yformer/EfficientSAM/main/LICENSE |
| 가중치 | 레포 내 배포(`weights/efficient_sam_vitt.pt`), 별도 표기 없음 | 가능 (추론) | 확인(배포 위치) — README |

## 학습 데이터

없음.

## 속도

공식 README에서 GPU 지연 수치를 확인하지 못함. ViT-T/ViT-S 크기라 MobileSAM과 비슷한 급 (추론). ONNX(인코더·디코더 분리) 공개 — 확인 (README, HF Space `yunyangx/EfficientSAM`).

RTX 4060 Laptop 8GB 추정: 인코더 20~50ms (추론).

## 성숙도

- 코드·가중치·ONNX 공개, 2024 이후 정체. 비디오 확장은 EfficientTAM(Apache-2.0, 확인 — https://raw.githubusercontent.com/yformer/EfficientTAM/main/LICENSE).

## 기준선(m1/m2) 대비

- 기대 이점: 라이선스 깨끗, 가벼움.
- 위험: 공식 AMG 도구 없음(직접 구현), EfficientViT-SAM보다 배포 문서 부족 → 우선순위 낮음.

## 출처

- https://github.com/yformer/EfficientSAM · https://arxiv.org/abs/2312.00863
