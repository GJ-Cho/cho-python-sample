# YOLO-seg (Ultralytics YOLOv8 / YOLO11 / YOLO26 -seg)

> 담당 토픽: topics/segmenters.md · 작성일: 2026-10-29

## 요약

가장 널리 쓰이는 실시간 인스턴스 분할 계열. 학습·배포 도구가 가장 성숙하지만 **AGPL-3.0**(상용은 Ultralytics Enterprise 라이선스 필요). 성능 비교용으로 카드만 두고 추천 시 감점.

## 분류

| 항목 | 값 |
|---|---|
| 파이프라인 유형 | [E2E] (학습형) |
| 입력 모달리티 | [RGB] |
| 맞는 시나리오 | [BOX][ONE][FEW] (학습 필요) · [UNK] ✕ |
| 프롬프트 | 없음 |

## 라이선스

| | 라이선스 | 상용 가능 | 출처 |
|---|---|---|---|
| 코드 | **AGPL-3.0** / Enterprise(유료) | AGPL 조건 하에서만 | 확인 — https://www.ultralytics.com/license |
| 가중치 | **AGPL-3.0** (학습한 커스텀 가중치도 AGPL 영향 받는다는 것이 Ultralytics 입장) | 동일 | 확인(라이선스 페이지), 커스텀 가중치 해석은 (추론) |

## 학습 데이터

라벨 수십~수백 장.

## 속도

| 수치 | 하드웨어 | 해상도 | 출처 (확인/추론) |
|---|---|---|---|
| YOLO26n-seg mask mAP 33.9, 2.1ms / YOLO26s-seg 40.0, 3.3ms | T4 TensorRT10 | 640 | 확인(2차) — https://replicate.com/ultralytics/yolo26-seg/readme |

RTX 4060 Laptop 8GB 추정: s/m 크기 5~15ms PyTorch (추론).

## 성숙도

- 매우 성숙(export: ONNX/TensorRT/OpenVINO). Windows 지원 양호.

## 기준선(m1/m2) 대비

- 기대 이점: 빠르고 학습 쉬움.
- 위험: **AGPL** — Zivid 샘플/데모 배포 시 소스 공개 의무 범위 검토 필요. 동급 Apache 대안 RF-DETR-Seg가 있어 우선순위 낮음.

## 출처

- https://docs.ultralytics.com/ · https://www.ultralytics.com/license
