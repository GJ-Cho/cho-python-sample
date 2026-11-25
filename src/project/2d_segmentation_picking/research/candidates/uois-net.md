# UOIS-Net-3D

> 담당 토픽: topics/input-modality.md · 작성일: 2026-10-29

## 요약

깊이(XYZ)로 물체 중심을 투표·클러스터링해 초기 마스크를 만들고(DSN), RGB로 경계를 정련하는(RRN) 2단계 UOIS 모델.
"깊이로 분리, RGB로 경계"라는 역할 분담의 원형이며, 공개 RGB-D UOIS 모델 중 **코드가 MIT**인 드문 사례라 후보로 둔다.

## 분류

| 항목 | 값 |
|---|---|
| 파이프라인 유형 | [E2E] (2단계 내부 구조) |
| 입력 모달리티 | [RGB-D] (DSN은 XYZ만, RRN은 RGB + 초기 마스크) |
| 맞는 시나리오 | [UNK] 일부 — 물체 사이에 깊이 차가 있을 때. 밀착 파우치·같은 높이 박스([BOX])는 DSN 단계가 분리 근거를 못 얻음 (추론) |
| 프롬프트 | 없음 |

## 라이선스

| | 라이선스 | 상용 가능 | 출처 |
|---|---|---|---|
| 코드 | MIT | 예 | [LICENSE](https://raw.githubusercontent.com/chrisdxie/uois/master/LICENSE), README "released under the MIT license" (확인) |
| 가중치 | 별도 표기 없음 (Google Drive 배포). TOD 합성 + RRN은 Google Open Images 학습판도 있음 | 불명확 — OID 라벨은 CC BY 4.0이나 가중치 조건은 미확인 | [README](https://raw.githubusercontent.com/chrisdxie/uois/master/README.md) (확인: 배포 위치 / 미확인: 가중치 조건) |

## 학습 데이터

없음(공개 가중치 사용 시). 공개 가중치는 탁상 합성 TOD(비사실적 RGB). 빈 탑다운 장면에 맞추려면 합성 데이터 재학습이 필요할 수 있다 (추론).

## 속도

| 수치 | 하드웨어 | 해상도 | 출처 (확인/추론) |
|---|---|---|---|
| 3~5 FPS (전체 파이프라인) | RTX 2080 Ti | 640x480 | [ar5iv 2007.08073](https://ar5iv.labs.arxiv.org/html/2007.08073) (확인) |

RTX 4060 Laptop 8GB 추정: 0.25~0.4s @640x480 (추론) — 2080 Ti와 비슷하거나 약간 느린 급. Zivid 1224x1024를 그대로 넣으면 2~4배 (추론).

## 성숙도

- 공식 코드·가중치 공개, 로컬 실행 가능. 실행 예제는 Jupyter notebook, 환경은 `env.yml`(구버전 conda).
- 마지막 push 2024-10, 스타 약 160 (GitHub API, 확인). ONNX/TensorRT 없음.
- `PLAN.md` 5장은 UOIS 계열 전체를 "torch 1.x + 커스텀 CUDA 빌드"로 묶어 보류했으나, UOIS-Net-3D 자체가 커스텀 CUDA op을 요구하는지는 확인하지 못했다.

## 기준선(m1/m2) 대비

- 기대 이점: 깊이 기반 분리라 텍스처 과분할이 원리적으로 없다. 강체가 흩어진 장면에서 m1보다 깨끗할 가능성.
- 위험: OCID Overlap F 86.4(DSN depth)로 2021 기준이며 이후 모델보다 낮다. 밀착·동일 평면 물체는 DSN이 한 덩어리로 묶는다(5장 순수 기하 폐기와 같은 이유, 추론). 탁상 학습 → 빈 벽·바닥 처리 미검증.

## 출처

- 논문: Xie et al., "Unseen Object Instance Segmentation for Robotic Environments", T-RO 2021, https://arxiv.org/abs/2007.08073
- 레포: https://github.com/chrisdxie/uois
