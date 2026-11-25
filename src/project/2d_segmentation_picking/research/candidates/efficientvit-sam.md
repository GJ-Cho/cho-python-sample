# EfficientViT-SAM

> 담당 토픽: topics/segmenters.md · 작성일: 2026-10-29

## 요약

MIT Han Lab의 EfficientViT 백본으로 SAM-ViT-H를 대체한 경량 SAM. **TensorRT 경로가 공식 문서화**되어 있고 AMG(`EfficientViTSamAutomaticMaskGenerator`)도 제공한다. 경량 SAM 중 라이선스·정확도·배포성이 가장 균형 잡힌 후보.

## 분류

| 항목 | 값 |
|---|---|
| 파이프라인 유형 | [DS] 뒷단 / [PM] 앞단(AMG) |
| 입력 모달리티 | [RGB] |
| 맞는 시나리오 | [DS] 뒷단으로 전 시나리오 · [UNK] AMG 가속판 |
| 프롬프트 | 박스·점 / 없음(AMG) |

## 라이선스

| | 라이선스 | 상용 가능 | 출처 |
|---|---|---|---|
| 코드 | Apache-2.0 | 가능 | 확인 — https://raw.githubusercontent.com/mit-han-lab/efficientvit/master/LICENSE |
| 가중치 | 별도 표기 없음 → 레포 Apache-2.0 (SA-1B 학습 데이터 조건은 별개) | 가능 (추론) | — |

## 학습 데이터

없음.

## 속도

| 수치 | 하드웨어 | 해상도 | 출처 (확인/추론) |
|---|---|---|---|
| L0 762 img/s, L1 638, L2 538 (COCO mAP 45.7/46.2/46.6, ViTDet 박스 프롬프트) | A100 TensorRT fp16 bs16 | 512 | 확인 — https://raw.githubusercontent.com/mit-han-lab/efficientvit/master/applications/efficientvit_sam/README.md |
| XL0 278 img/s (47.5), XL1 182 img/s (47.8) | 동일 | 1024 | 확인 — 동일 |
| L0 8.2ms / XL0 22.5ms / XL1 37.2ms | Jetson AGX Orin bs1 | — | 확인 — 동일 |

참고: SAM-ViT-H는 같은 지표 46.5 내외(SAM 논문) → **L2/XL0는 정확도 손실이 거의 없다**.

RTX 4060 Laptop 8GB 추정: XL0 인코더 TensorRT fp16 ≈ 15~30ms, PyTorch ≈ 40~80ms (추론) — Orin 22.5ms 기준, 4060L이 Orin보다 빠름.

## 성숙도

- 공식 코드·가중치, ONNX export + `trtexec` TensorRT 빌드 절차 문서화. 마지막 push 2025-09.
- SAM1 계열 디코더(프롬프트 인터페이스가 segment-anything과 동일) → m2 뒷단 교체 비용 낮음 (추론).

## 기준선(m1/m2) 대비

- 기대 이점: 인코더 시간 대폭 단축, TensorRT로 0.5s 목표에 유리. 정확도는 SAM2-small과 비슷하거나 약간 높을 수 있음(박스 프롬프트 COCO 기준, 추론).
- 위험: AMG 시간은 인코더보다 **디코더×프롬프트 수 + NMS**가 지배 → m1 가속 폭은 생각보다 작을 수 있음 (추론). 과분할 구조는 동일.

## 출처

- https://github.com/mit-han-lab/efficientvit · https://arxiv.org/abs/2402.05008
