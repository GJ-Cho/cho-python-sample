# UCN (Unseen Object Clustering)

> 담당 토픽: topics/input-modality.md · 작성일: 2026-10-29

## 요약

RGB·깊이를 별도 인코더로 픽셀 임베딩을 학습하고 mean-shift로 클러스터링한 뒤, 물체별 zoom-in으로 정련하는 UOIS 모델.
**입력 융합 방식(early / add / concat)과 RGB vs Depth를 같은 모델에서 비교한 ablation**이 있어 모달리티 판단의 핵심 근거다. 후보로서는 라이선스 때문에 비교용.

## 분류

| 항목 | 값 |
|---|---|
| 파이프라인 유형 | [E2E] (임베딩 + 클러스터링) |
| 입력 모달리티 | [RGB] / [RGB-D] (변형별 가중치) |
| 맞는 시나리오 | [UNK] 일부 — 깊이 차가 있는 클러터. [BOX]·밀착 파우치는 불리 (추론) |
| 프롬프트 | 없음 |

## 라이선스

| | 라이선스 | 상용 가능 | 출처 |
|---|---|---|---|
| 코드 | NVIDIA Source Code License — "non-commercially, research or evaluation purposes only" | **아니오** | [LICENSE.md](https://raw.githubusercontent.com/NVlabs/UnseenObjectClustering/master/LICENSE.md) (확인) |
| 가중치 | 별도 표기 없음, 같은 레포에서 배포 → 코드 라이선스 적용으로 보는 것이 안전 | **아니오** (추론) | [README](https://raw.githubusercontent.com/NVlabs/UnseenObjectClustering/master/README.md) |

## 학습 데이터

없음(공개 가중치 사용 시). 공개 가중치는 탁상 합성 TOD.

## 속도

| 수치 | 하드웨어 | 해상도 | 출처 (확인/추론) |
|---|---|---|---|
| 1단계 약 0.2s + 2단계 물체당 약 0.05s | TITAN Xp | 640x480 | [ar5iv 2007.15157](https://ar5iv.labs.arxiv.org/html/2007.15157) (확인) |

RTX 4060 Laptop 8GB 추정: 물체 20개 장면에서 정련 포함 약 1~1.3s (추론) — TITAN Xp와 비슷한 급으로 가정. 정련 없이 0.2s대.

## 입력 ablation (확인, OCID / OSD Overlap F, 정련 전)

RGB 59.4 / 63.3 · Depth 86.4 / 81.0 · RGB-D early 82.8 / 79.2 · **RGB-D add 88.5 / 86.2** · RGB-D concat 82.9 / 79.7.
→ 채널을 입력에서 붙이는 early fusion은 깊이 단독보다 낮다. 특징 덧셈(late)만 깊이 단독을 넘는다.

## 성숙도

- 공식 코드·가중치 공개, 로컬 실행 가능. PyTorch 0.4.1+ 시대 코드, ROS 노드는 Python2 언급.
- 마지막 push 2023-12 (GitHub API, 확인). ONNX/TensorRT 없음.

## 기준선(m1/m2) 대비

- 기대 이점: 깊이 임베딩이라 텍스처 과분할이 적다.
- 위험: **비상업 라이선스로 추천 불가.** 환경이 오래돼 단일 venv 요구와 충돌(`PLAN.md` 5장). 탁상 도메인.

## 출처

- 논문: Xiang et al., "Learning RGB-D Feature Embeddings for Unseen Object Instance Segmentation", CoRL 2020, https://arxiv.org/abs/2007.15157
- 레포: https://github.com/NVlabs/UnseenObjectClustering
