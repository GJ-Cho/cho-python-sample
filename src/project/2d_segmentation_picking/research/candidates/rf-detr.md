# RF-DETR (검출 모델, N/S/M/L)

> 담당 토픽: topics/detectors.md · 작성일: 2026-10-29
> 분할 변형 RF-DETR-Seg는 `segmenters.md` 담당. 이 카드는 [DS] 앞단으로서의 **박스 검출 모델**만 다룬다.

## 요약

Roboflow의 실시간 DETR(LW-DETR 계열 + DINOv2 백본, 추론: 논문 기준). 소량 라벨로 fine-tune하는 용도로 설계되었고 N~L 가중치가 Apache-2.0이다. **학습 데이터를 만들 수 있는 [BOX]/[ONE]/[FEW]의 1순위 검출기.**

## 분류

| 항목 | 값 |
|---|---|
| 파이프라인 유형 | [DS] 앞단 (Seg 변형은 [E2E]) |
| 입력 모달리티 | [RGB] (직접 학습 시 depth 채널 실험은 가능하나 사전학습과 불일치, 추론) |
| 맞는 시나리오 | [BOX] ◎(1~3종, 라벨·합성 쉬움), [ONE]/[FEW] ◎, [UNK] ✕(학습 클래스만 검출) |
| 프롬프트 | 없음 (폐쇄형) |

## 라이선스

| | 라이선스 | 상용 가능 | 출처 |
|---|---|---|---|
| 코드 | Apache-2.0 (`rfdetr`) | 예 | [README](https://raw.githubusercontent.com/roboflow/rf-detr/develop/README.md) (확인) |
| 가중치 | Nano/Small/Medium/Large 검출·전 Seg 모델: Apache-2.0. **Atto/Femto/Pico/XL/2XL 검출과 `rfdetr_plus`: PML 1.0** | N~L은 예. PML 모델은 감점 | 같은 README (확인) |

사전학습 데이터(Objects365 등) 약관 쟁점은 남는다(추론, `topics/detectors.md` 참조).

## 학습 데이터

라벨 수백 장 이상(클래스당 수십~수백 인스턴스, 추론). RF100-VL(소규모 실세계 데이터셋 100종) 벤치마크에서 N AP50 85.0 / M 87.4 (확인) — 소량 데이터 fine-tune 성능 지표. 합성 데이터·SAM 보조 자동 라벨링은 `data/dataset-plan.md` 참조.

## 속도

| 수치 | 하드웨어 | 해상도 | 출처 (확인/추론) |
|---|---|---|---|
| Medium 4.4ms, COCO 54.7 | T4, TensorRT FP16, bs1 | 576×576 | README (확인) |
| Atto 1.0ms / 2XL 17.2ms | 동일 | 380² / 880² | README (확인) |

RTX 4060 Laptop 8GB 추정: TensorRT FP16 5~10ms, PyTorch eager 20~50ms (추론). 4060 Laptop의 FP16 텐서 처리량을 T4와 비슷한 급으로 보고, eager는 TRT 대비 3~5배로 본 값.

## 성숙도

- 공식 코드·가중치 공개, pip 패키지, ONNX/TensorRT 내보내기, Colab 학습 노트북 (확인: README에 학습 경로 언급).
- 2026년 기준 활발히 업데이트(Atto~2XL, Seg, keypoint 프리뷰 추가, 확인).

## 기준선(m1/m2) 대비

- 기대 이점: 검출이 수 ms로 m2 대비 검출 단계 지연이 사실상 사라진다. 어휘 문제 없음, threshold 튜닝이 단순해진다(추론).
- 위험: 학습 데이터 제작 비용. 학습하지 않은 물체는 못 잡는다. 붙어 있는 같은 박스는 라벨 품질에 좌우된다.

## 출처

- https://raw.githubusercontent.com/roboflow/rf-detr/develop/README.md
- LW-DETR LICENSE(Apache-2.0): https://raw.githubusercontent.com/Atten4Vis/LW-DETR/main/LICENSE
