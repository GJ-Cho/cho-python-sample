# RF-DETR-Seg (Nano ~ 2XL)

> 담당 토픽: topics/segmenters.md · 작성일: 2026-10-29
> 검출 전용 RF-DETR은 `rf-detr.md`(detectors 담당) 참조.

## 요약

Roboflow RF-DETR(DINOv2 백본 실시간 DETR)의 인스턴스 분할 변형. **전 크기 Apache-2.0, T4 TensorRT 3.4~21.8ms**, 소량 라벨 fine-tune 지향. 학습 데이터를 만들 수 있는 [BOX]/[ONE]/[FEW]에서 [E2E] 1순위 후보.

## 분류

| 항목 | 값 |
|---|---|
| 파이프라인 유형 | [E2E] (학습형, 폐쇄 어휘) |
| 입력 모달리티 | [RGB] |
| 맞는 시나리오 | [BOX] ◎ 박스 1~3종을 클래스로 학습 · [ONE][FEW] ◎ 세 제품 유형 모두(반사 부품도 학습으로 흡수) · [UNK] ✕ |
| 프롬프트 | 없음 |

## 라이선스

| | 라이선스 | 상용 가능 | 출처 |
|---|---|---|---|
| 코드 | Apache-2.0 (`rfdetr_plus` 확장은 PML 1.0) | 가능 | 확인 — https://raw.githubusercontent.com/roboflow/rf-detr/develop/README.md |
| 가중치 | **Seg-N/S/M/L/XL/2XL 전부 Apache-2.0** (PML은 검출 XL/2XL 등에만 해당) | 가능 | 확인 — 동일 README 표 |

## 학습 데이터

라벨 수십~수백 장(폴리곤/마스크). 수치 근거는 공식 README에 없음. RF-100VL 지향 설계상 소량 데이터 fine-tune에 강하다는 주장(논문). SAM3/SAM2로 자동 라벨 → 사람 검수 → 학습 흐름이 현실적 (추론, → `data/dataset-plan.md`).

## 속도

| 수치 | 하드웨어 | 해상도 | 출처 (확인/추론) |
|---|---|---|---|
| Seg-N 3.4ms (mask AP 40.3) / S 4.4 (43.1) / M 5.9 (45.3) / L 8.8 (47.1) / XL 13.5 (48.8) / 2XL 21.8ms (49.9) | T4, TensorRT FP16, bs1 | 312 / 384 / 432 / 504 / 624 / 768 | 확인 — README |

RTX 4060 Laptop 8GB 추정: TensorRT ≈ T4와 비슷하거나 약간 빠름(Seg-M 5ms 내외), PyTorch eager 20~60ms + 후처리 (추론). **0.5s 목표를 크게 여유 있게 만족**.

주의: 입력 해상도가 312~768로 낮다. Zivid 2+ MR130 전체(1224×1024 내외)를 넣으면 작은 물체 경계가 거칠 수 있어 빈 ROI 크롭 후 입력 권장 (추론).

## 성숙도

- 2026-10 기준 매우 활발(마지막 push 2026-10-29). pip 패키지 `rfdetr`, ONNX/TensorRT export, Colab 학습 노트북. 논문 arXiv 2511.09554 (ICLR 2026).

## 기준선(m1/m2) 대비

- 기대 이점: 과분할 없음(인스턴스 단위로 학습), 10~100배 빠름, 라이선스 깨끗.
- 위험: 학습·라벨링 비용. 새 제품마다 재학습. 밀착 동일 박스 경계는 데이터 품질에 좌우.

## 출처

- https://github.com/roboflow/rf-detr · https://arxiv.org/abs/2511.09554
