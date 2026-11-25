# RT-DETR / RT-DETRv2

> 담당 토픽: topics/detectors.md · 작성일: 2026-10-29

## 요약

Baidu의 실시간 DETR. D-FINE·DEIM·RF-DETR의 기반이 된 계열로 Apache-2.0이고 transformers에 들어와 있어 학습·추론 도구가 가장 흔하다. 성능은 후속작보다 낮아 2순위.

## 분류

| 항목 | 값 |
|---|---|
| 파이프라인 유형 | [DS] 앞단 |
| 입력 모달리티 | [RGB] |
| 맞는 시나리오 | [BOX]/[ONE]/[FEW] ○(학습 전제), [UNK] ✕ |
| 프롬프트 | 없음 (폐쇄형) |

## 라이선스

| | 라이선스 | 상용 가능 | 출처 |
|---|---|---|---|
| 코드 | Apache-2.0 | 예 | [LICENSE](https://raw.githubusercontent.com/lyuwenyu/RT-DETR/main/LICENSE) (확인) |
| 가중치 | 레포 라이선스를 따른다고 봄. O365 사전학습판은 약관 쟁점 가능 | 대체로 예 (추론) | — |

Ultralytics에도 RT-DETR 구현이 있으나 그 경로는 **AGPL-3.0**이다(추론: Ultralytics 전체 라이선스 기준). 원 레포 또는 transformers 경로를 쓴다.

## 학습 데이터

라벨 수백 장 이상 (추론).

## 속도

| 수치 | 하드웨어 | 해상도 | 출처 (확인/추론) |
|---|---|---|---|
| v2-S 217 FPS (COCO 48.1) / v2-L 108 FPS (53.4) / v2-X 74 FPS (54.3) | T4, TensorRT FP16 | 640 | [README](https://raw.githubusercontent.com/lyuwenyu/RT-DETR/main/README.md) (확인) |

RTX 4060 Laptop 8GB 추정: TRT 5~10ms, eager 20~50ms (추론).

## 성숙도

- 코드·가중치 공개, transformers에 RTDETR/RTDETRv2 지원 (확인). README에 RT-DETRv4(2025.11) 언급 (확인, 세부 미조사).

## 기준선(m1/m2) 대비

- 기대 이점: transformers 단일 환경(현 m2와 같은 라이브러리)에서 학습형 검출기 도입.
- 위험: 같은 지연에서 D-FINE/RF-DETR보다 COCO AP가 낮다.

## 출처

- https://raw.githubusercontent.com/lyuwenyu/RT-DETR/main/README.md
- https://raw.githubusercontent.com/lyuwenyu/RT-DETR/main/LICENSE
