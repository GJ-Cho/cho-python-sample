# Mask2Former (+ HF transformers 구현)

> 담당 토픽: topics/segmenters.md · 작성일: 2026-10-29

## 요약

Meta의 범용(semantic/instance/panoptic) 마스크 분류 트랜스포머. 실시간 모델은 아니지만 **학습형 [E2E] 인스턴스 분할의 정확도 기준점**이며, CropFormer(엔티티 분할) 등 다수 후속 연구의 베이스다. HF `transformers`에 구현이 있어 커스텀 CUDA op 없이 실행 가능.

## 분류

| 항목 | 값 |
|---|---|
| 파이프라인 유형 | [E2E] (학습형) |
| 입력 모달리티 | [RGB] |
| 맞는 시나리오 | [BOX][ONE][FEW] (학습 필요, 속도 여유 있을 때) |
| 프롬프트 | 없음 |

## 라이선스

| | 라이선스 | 상용 가능 | 출처 |
|---|---|---|---|
| 코드 | MIT (일부 Apache-2.0: Deformable-DETR 연산 등) | 가능 | 확인 — https://raw.githubusercontent.com/facebookresearch/Mask2Former/main/README.md |
| 가중치 | 모델 주 별도 표기 없음 → MIT 간주 | 가능 (추론) | — |

## 학습 데이터

라벨 수백 장 이상 권장 (DETR 계열 수렴이 느림, 추론). COCO 사전학습 가중치에서 fine-tune.

## 속도

| 수치 | 하드웨어 | 해상도 | 출처 (확인/추론) |
|---|---|---|---|
| R50 COCO instance AP 43.7 내외, 수 FPS~10 FPS급 (논문 기준) | V100 | 800~1333 | (추론, 기억 기반 — 원문 미재확인) |

RTX 4060 Laptop 8GB 추정: R50 100~200ms, Swin-L은 VRAM·시간 모두 부담 (추론).

## 성숙도

- 원본은 detectron2 + MSDeformAttn CUDA op 빌드 필요(Windows 부담). HF `Mask2FormerForUniversalSegmentation`으로 우회 가능 (추론). 원본 레포는 아카이브 상태 (추론).

## 기준선(m1/m2) 대비

- 기대 이점: 학습 시 인스턴스 품질 높음.
- 위험: RF-DETR-Seg 대비 느리고 학습 데이터 더 필요. 신규 채택 이유 약함 → 비교용.

## 출처

- https://github.com/facebookresearch/Mask2Former · https://arxiv.org/abs/2112.01527 · https://huggingface.co/docs/transformers/model_doc/mask2former
