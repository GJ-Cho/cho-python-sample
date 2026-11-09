# MSMFormer (Mean Shift Mask Transformer)

> 담당 토픽: topics/input-modality.md · 작성일: 2026-10-29

## 요약

UCN의 mean-shift 클러스터링을 미분 가능한 트랜스포머(von Mises-Fisher mean shift)로 바꾼 UOIS 모델. RGB / RGB-D 가중치가 모두 있어
같은 모델에서 모달리티 차이를 보여준다. 공개 RGB-D UOIS 중 상위권 수치.

## 분류

| 항목 | 값 |
|---|---|
| 파이프라인 유형 | [E2E] |
| 입력 모달리티 | [RGB] / [RGB-D] |
| 맞는 시나리오 | [UNK] 일부 (깊이 차가 있는 클러터). 밀착·동일 평면 물체는 불리 (추론) |
| 프롬프트 | 없음 |

## 라이선스

| | 라이선스 | 상용 가능 | 출처 |
|---|---|---|---|
| 코드 | MIT (`IRVLUTD/UnseenObjectsWithMeanShift`) | 예 (코드만) | GitHub API license=MIT (확인) |
| 가중치 | 표기 없음. **1단계 백본 가중치를 UCN 체크포인트에서 받도록 지시** → UCN은 NVIDIA 비상업 라이선스 | **사실상 아니오** (추론) | [README](https://raw.githubusercontent.com/IRVLUTD/UnseenObjectsWithMeanShift/master/README.md) (확인: UCN 체크포인트 필요) |

## 학습 데이터

없음(공개 가중치). TOD 합성 약 28만 장 1 epoch, UOAIS-Sim 학습판, 로봇 상호작용 데이터 미세조정판도 공개.

## 속도

| 수치 | 하드웨어 | 해상도 | 출처 (확인/추론) |
|---|---|---|---|
| 1단계 약 0.278s + 2단계 ROI당 약 0.072s | A100 | 480x640 RGB-D | [ar5iv 2211.11679](https://ar5iv.labs.arxiv.org/html/2211.11679) (확인) |

RTX 4060 Laptop 8GB 추정: 1단계만 0.6~1.0s, 정련 포함 물체 20개면 2s 이상 (추론) — A100 대비 3~4배 느리다고 가정. **속도 예산 초과 가능성 높음.**

## 입력 ablation (확인, Overlap F / Boundary F)

OCID: RGB 67.7 / 55.8 → RGB-D 88.5 / 83.0 (정련 시 91.5 / 87.3). OSD: RGB 63.6 / 47.0 → RGB-D 82.8 / 60.6 (정련 86.4 / 68.4).

## 성숙도

- 코드·가중치 공개, 로컬 실행 가능. torch 1.10 + detectron2 + Mask2Former의 **MultiScaleDeformableAttention CUDA op 컴파일** 필요할 수 있음, numpy 1.23 고정 (README, 확인).
- 마지막 push 2024-08 (GitHub API, 확인). ONNX/TensorRT 없음.

## 기준선(m1/m2) 대비

- 기대 이점: 공개 UOIS 중 OCID 최상위권. 깊이 덕에 텍스처 과분할 적음.
- 위험: UCN 가중치 의존으로 상용 불가, 구 환경 + CUDA 빌드(`PLAN.md` 5장 보류 사유 그대로), 4060에서 속도 미달 예상, 탁상 도메인.

## 출처

- 논문: Lu et al., "Mean Shift Mask Transformer for Unseen Object Instance Segmentation", arXiv 2211.11679 (ICRA 2024)
- 레포: https://github.com/IRVLUTD/UnseenObjectsWithMeanShift (구 주소 YoungSean/UnseenObjectsWithMeanShift)
