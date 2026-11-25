# Mask DINO

> 담당 토픽: topics/segmenters.md · 작성일: 2026-10-29

## 요약

IDEA Research의 DINO 검출기 + 마스크 분기 통합 모델(CVPR 2023). Mask2Former 대비 instance AP가 높다. 실시간은 아님. 비교용.

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
| 코드 | Apache-2.0 | 가능 | 확인 — https://raw.githubusercontent.com/IDEA-Research/MaskDINO/main/LICENSE |
| 가중치 | detrex-storage 릴리스, 별도 표기 없음 → Apache-2.0 간주 | 가능 (추론) | 확인(배포 위치) — README |

## 학습 데이터

라벨 수백 장 이상 (추론).

## 속도

| 수치 | 하드웨어 | 해상도 | 출처 (확인/추론) |
|---|---|---|---|
| R50 COCO mask AP 46.1~46.3 (box 51.5~51.7) | — | — | 확인 — 체크포인트 파일명, https://raw.githubusercontent.com/IDEA-Research/MaskDINO/main/README.md |

RTX 4060 Laptop 8GB 추정: R50 150~300ms (추론).

## 성숙도

- detectron2 + 커스텀 CUDA op 빌드 필요 → 단일 venv/Windows 부담(PLAN 5장 UOIS 폐기 사유와 같은 유형). 2023 이후 정체.

## 기준선(m1/m2) 대비

- 기대 이점: 학습형 중 높은 정확도.
- 위험: 설치 부담, 속도, RF-DETR-Seg 대비 이점 불명확 → 비교용.

## 출처

- https://github.com/IDEA-Research/MaskDINO · https://arxiv.org/abs/2206.02777
