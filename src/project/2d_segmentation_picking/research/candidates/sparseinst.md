# SparseInst

> 담당 토픽: topics/segmenters.md · 작성일: 2026-10-29

## 요약

HUST의 instance activation map 기반 실시간 인스턴스 분할(CVPR 2022). NMS 없음, MIT. 정확도는 최신 모델보다 낮다. 비교용.

## 분류

| 항목 | 값 |
|---|---|
| 파이프라인 유형 | [E2E] (학습형) |
| 입력 모달리티 | [RGB] |
| 맞는 시나리오 | [BOX][ONE] (학습 필요) |
| 프롬프트 | 없음 |

## 라이선스

| | 라이선스 | 상용 가능 | 출처 |
|---|---|---|---|
| 코드 | MIT | 가능 | 확인 — GitHub API `hustvl/SparseInst` license MIT |
| 가중치 | Google Drive 배포, 별도 표기 없음 | 가능 (추론) | — |

## 학습 데이터

라벨 수십~수백 장 (추론).

## 속도

| 수치 | 하드웨어 | 해상도 | 출처 (확인/추론) |
|---|---|---|---|
| R50 AP 32.8 / 44.3 FPS ~ R50-vd-DCN AP 37.7 / 39.3 FPS | 논문 측정 GPU(README 미기재, 2080Ti로 기억 — 추론) | 608~640 | 확인 — https://raw.githubusercontent.com/hustvl/SparseInst/main/README.md |

RTX 4060 Laptop 8GB 추정: 20~30ms (추론).

## 성숙도

- detectron2 기반, 마지막 push 2023-10. 정체.

## 기준선(m1/m2) 대비

- 기대 이점: 단순·빠름.
- 위험: 정확도가 RF-DETR-Seg·RTMDet-Ins보다 낮고 detectron2 의존 → 채택 이유 약함.

## 출처

- https://github.com/hustvl/SparseInst · https://arxiv.org/abs/2203.12827
