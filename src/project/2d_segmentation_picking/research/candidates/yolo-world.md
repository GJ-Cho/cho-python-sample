# YOLO-World

> 담당 토픽: topics/detectors.md · 작성일: 2026-10-29

## 요약

Tencent AILab의 실시간 개방어휘 YOLO("prompt-then-detect", 어휘를 미리 인코딩해 재파라미터화). 빠르지만 **GPL-3.0**이고 정확도·기능 모두 후속작 YOLOE가 앞선다.

## 분류

| 항목 | 값 |
|---|---|
| 파이프라인 유형 | [DS] 앞단 |
| 입력 모달리티 | [RGB] |
| 맞는 시나리오 | [BOX]/[UNK]/[ONE]/[FEW] 모두 △ — 텍스트 어휘 의존 |
| 프롬프트 | 텍스트 (이미지 프롬프트는 2024.03 지원 추가, YOLO-World-Image는 preview) |

## 라이선스

| | 라이선스 | 상용 가능 | 출처 |
|---|---|---|---|
| 코드 | **GPL-3.0** ("supported for commercial usage", 별도 상용 라이선스 문의) | 조건부 | [README](https://raw.githubusercontent.com/AILab-CVC/YOLO-World/master/README.md) (확인) |
| 가중치 | GPL-3.0으로 봄. Ultralytics판은 **AGPL-3.0** | 조건부 | (추론) |

## 학습 데이터

없음.

## 속도

| 수치 | 하드웨어 | 해상도 | 출처 (확인/추론) |
|---|---|---|---|
| README에 FPS 표 없음. LVIS minival L 26.8 @640, X 28.6 @640 | — | 640/800/1280 | README (확인) |
| ONNX 내보내기 지원, TensorRT "coming soon" | — | — | README (확인) |

RTX 4060 Laptop 8GB 추정: 10~50ms (추론, YOLOv8급 구조).

## 성숙도

- 코드·가중치 공개, v2.1(2025.02) 이미지 프롬프트 학습 코드 포함 (확인).

## 기준선(m1/m2) 대비

- 기대 이점: 검출 속도.
- 위험: GPL-3.0, LVIS AP가 GD-T 수준 이하(minival 26.8 vs GD-T 28.8)로 어휘 밖 대응이 나아지지 않는다. YOLOE가 상위호환에 가깝다(추론).

## 출처

- https://raw.githubusercontent.com/AILab-CVC/YOLO-World/master/README.md
- arXiv 2401.17270 (YOLO-World: Real-Time Open-Vocabulary Object Detection)
