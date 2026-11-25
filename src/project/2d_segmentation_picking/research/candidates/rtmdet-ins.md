# RTMDet-Ins (MMDetection)

> 담당 토픽: topics/segmenters.md · 작성일: 2026-10-29

## 요약

OpenMMLab의 실시간 인스턴스 분할. TRT FP16 1.7~5.3ms로 빠르고 Apache-2.0. 다만 MMDetection/MMCV 생태계 유지보수가 둔화되어 최신 torch(2.6+)·Windows 설치가 부담.

## 분류

| 항목 | 값 |
|---|---|
| 파이프라인 유형 | [E2E] (학습형) |
| 입력 모달리티 | [RGB] |
| 맞는 시나리오 | [BOX][ONE][FEW] (학습 필요) |
| 프롬프트 | 없음 |

## 라이선스

| | 라이선스 | 상용 가능 | 출처 |
|---|---|---|---|
| 코드 | Apache-2.0 | 가능 | 확인 — https://raw.githubusercontent.com/open-mmlab/mmdetection/main/LICENSE |
| 가중치 | download.openmmlab.com 배포, 레포 라이선스 따름 | 가능 (추론) | — |

## 학습 데이터

라벨 수십~수백 장 (추론).

## 속도

| 수치 | 하드웨어 | 해상도 | 출처 (확인/추론) |
|---|---|---|---|
| tiny mask AP 35.4 / 1.70ms · s 38.7 / 1.93 · m 42.1 / 2.69 · l 43.7 / 3.68 · x 44.6 / 5.31ms | RTX 3090 TRT FP16 (NMS 제외 추정) | 640 | 확인 — https://raw.githubusercontent.com/open-mmlab/mmdetection/main/configs/rtmdet/README.md |

RTX 4060 Laptop 8GB 추정: PyTorch 15~40ms (추론).

## 성숙도

- MMDetection 3.x 마지막 릴리스가 2024년 초(3.3.0)로 기억 (추론, 미재확인). mmcv 휠이 최신 torch/CUDA/Windows 조합을 따라가지 못하는 문제가 흔함 (추론).

## 기준선(m1/m2) 대비

- 기대 이점: 매우 빠르고 Apache.
- 위험: 설치·유지보수. 같은 조건이면 RF-DETR-Seg가 더 간편 → 2순위.

## 출처

- https://github.com/open-mmlab/mmdetection/tree/main/configs/rtmdet · https://arxiv.org/abs/2212.07784
