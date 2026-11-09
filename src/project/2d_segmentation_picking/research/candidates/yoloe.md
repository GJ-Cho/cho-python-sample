# YOLOE (-seg)

> 담당 토픽: topics/detectors.md(검출) · topics/segmenters.md(-seg) 공동 · 작성일: 2026-10-29
> 주의: detectors 담당이 먼저 만든 같은 이름의 카드를 segmenters 담당이 실수로 덮어썼다(23:34). 검출 관점 내용(T4 시각 프롬프트 27 FPS 등)은 `topics/detectors.md` 표를 기준으로 보완 필요.

## 요약

THU-MIG의 "Real-Time Seeing Anything". 텍스트·시각(visual prompt)·prompt-free 세 모드의 open-vocab 검출+**분할**을 YOLO 속도로 수행. 같은 이미지 안 예시 박스로 개념을 지정할 수 있어 SAM3의 경량 대안처럼 보이지만 **AGPL-3.0**.

## 분류

| 항목 | 값 |
|---|---|
| 파이프라인 유형 | [E2E] (open-vocab) |
| 입력 모달리티 | [RGB] |
| 맞는 시나리오 | [BOX] 시각 프롬프트 · [FEW] 텍스트 · [UNK] prompt-free(내장 대어휘) |
| 프롬프트 | 텍스트 / 시각(박스) / 없음 |

## 라이선스

| | 라이선스 | 상용 가능 | 출처 |
|---|---|---|---|
| 코드 | **AGPL-3.0** | 조건부(AGPL) | 확인 — https://raw.githubusercontent.com/THU-MIG/yoloe/main/LICENSE |
| 가중치 | AGPL-3.0 (ultralytics 기반) | 조건부 | (추론) |

## 학습 데이터

없음 (zero-shot). fine-tune 가능.

## 속도

| 수치 | 하드웨어 | 해상도 | 출처 (확인/추론) |
|---|---|---|---|
| LVIS minival AP: v8-S 27.9(텍스트)/26.2(시각)/21.0(prompt-free), v8-L 35.9/34.2/27.2 | — | 640 | 확인 — https://raw.githubusercontent.com/THU-MIG/yoloe/main/README.md |
| 텍스트 프롬프트: v8-S 305.8 FPS, v8-L 102.5 FPS | T4 TensorRT | 640 | 확인 — 동일 |
| **시각 프롬프트: v8-S 64.3 FPS, v8-L 27.2 FPS** (detectors 담당 보완) | T4 TensorRT | 640 | 확인 — 동일 |
| prompt-free: v8-S 95.8 FPS, v8-L 25.3 FPS | T4 **PyTorch** | 640 | 확인 — 동일 |

검출 관점 보완(detectors 담당): Ultralytics 문서에 YOLOE-26 체크포인트(`yoloe-26{s,m,l,x}-seg.pt`, prompt-free `-seg-pf.pt`)가 있다 — https://docs.ultralytics.com/models/yoloe (확인). Ultralytics 경로는 AGPL-3.0 또는 Enterprise License.

RTX 4060 Laptop 8GB 추정: 10~30ms (추론).

## 성숙도

- 공식 코드·가중치, ultralytics 패키지에 통합. 마지막 push 2025-06.

## 기준선(m1/m2) 대비

- 기대 이점: m2 대비 수십 배 빠름, 분할 내장.
- 위험: AGPL. open-vocab 정확도는 SAM3보다 크게 낮을 것 (추론, LVIS AP 기준). 마스크 품질은 YOLO-seg 수준(프로토타입 마스크, 경계 거침).

## 출처

- https://github.com/THU-MIG/yoloe · https://arxiv.org/abs/2503.07465
