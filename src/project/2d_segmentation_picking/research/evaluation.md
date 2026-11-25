# 성능 판단 기준 — 지표, 측정 규칙, 기준선 수치

> 작성: 2026-10-29 (1단계 조사). 표기법([BOX]/[UNK]/[ONE]/[FEW], [RGB]/[RGB-D]/[RGB-D-N])은 `README.md` 참조.
> **확인** = 출처(URL 또는 파일 경로)를 붙인 내용, **(추론)** = 근거는 있으나 직접 검증하지 않은 판단.

## 1. 요약 — 시나리오별 주지표

| 시나리오 | 1순위 지표 | 2순위 | 이유 (추론) |
|---|---|---|---|
| [BOX] | **P@K (K=1,3,5), IoU≥0.75** + **merge 수** | Recall@0.75(최상층), Boundary IoU | 같은 박스가 빈틈없이 붙어 있어 "두 박스를 한 마스크로" 합치는 오류가 치명적이다. 박스 경계가 곧 파지 위치·자세 근거라 0.5 매칭은 너무 느슨하다 |
| [UNK] | **Recall@0.5** (기존 벤치마크와 연속성) + **split 수** | P@3, Precision@0.5(ignore 영역 적용) | GT가 부분 라벨이라 AP/precision의 신뢰도가 낮다(3장). 놓치지 않는 것과 과분할이 m1/m2의 알려진 약점 |
| [ONE] | **Mask AP75** (클래스 무관) | P@K@0.75, Boundary IoU | 학습형 모델이 들어오므로 점수 순위까지 평가하는 AP가 유효. 완전 라벨 가능(1종) |
| [FEW] | **클래스별 mAP (AP50:95)** + **matched 인스턴스의 분류 정확도** | 클래스별 Recall@0.5, P@K | "어느 제품인가"까지 맞혀야 한다. 마스크 품질과 분류 오류를 분리해 봐야 원인을 안다 |
| 공통 게이트 | **warm 총 지연 ≤ 1.0s (목표 0.5s)**, **peak VRAM ≤ 8GB** (한 방법 단독) | 추론/후처리 분리 시간 | `README.md` 공통 제약 |

품질 지표는 "통과/탈락"이 아니라 **지연 게이트를 통과한 후보끼리** 비교한다.

## 2. 지표 정의

공통 표기: GT 인스턴스 집합 G, 예측 집합 P(각 p에 점수 s(p)와 방법 고유 순위), 마스크 IoU `IoU(g,p)=|g∩p|/|g∪p|`.
보조량: `cov_g(p)=|g∩p|/|g|`(GT 커버율), `pur_p(g)=|g∩p|/|p|`(예측 순도).

### 2.1 인스턴스 매칭 (모든 지표의 기반)

- **1:1 매칭**: COCO 방식. 예측을 점수 내림차순으로 보며, 아직 매칭 안 된 GT 중 IoU 최대(≥τ)인 것과 짝짓는다. τ ∈ {0.5, 0.75}.
  - 확인: 현재 `scripts/run_benchmark.py`의 `_evaluate()`는 **1:1이 아니다** — GT마다 최고 IoU만 본다(같은 예측이 두 GT에 동시에 매칭될 수 있음). IoU≥0.5에서는 한 예측이 두 GT와 동시에 0.5 이상일 수 없어 결과가 같지만, τ<0.5나 precision 계산에서는 차이가 난다 (추론, IoU 정의상).
- **ignore 영역**: 라벨되지 않은 영역(부분 라벨, 최상층 밴드 밖)을 COCO `iscrowd=1` 영역으로 두고, 그 영역과 `pur_p ≥ 0.5`인 예측은 TP도 FP도 아닌 "무시"로 친다. pycocotools가 crowd 영역에 대해 이 방식(IoU 대신 예측 면적 기준 겹침)을 쓴다 (추론, COCO 평가 코드의 알려진 동작).

### 2.2 Recall / Precision / F1 @τ

- `Recall@τ = |매칭된 GT| / (|G_labeled| + n_missing)`. **완전 누락 물체(마스크 없이 위치만 기록된 것)를 분모에 포함한다** — 기존 정의 유지 (확인: `scripts/run_benchmark.py` docstring).
- `Precision@τ = |TP| / (|P| − |ignored|)`.
- `F1@τ` = 조화평균. τ=0.5(검출 수준)와 τ=0.75(파지에 쓸 만한 마스크 품질)를 둘 다 보고한다.

### 2.3 Mask AP / AR (COCO)

- `AP` = IoU 0.50:0.05:0.95 평균, `AP50`, `AP75`. `AR@k` (k=1,10,100). 클래스 무관 평가는 모든 GT/예측의 `category_id=1`.
- 점수 의존성: AP는 **예측 점수의 순위**를 평가한다. 확인: m1은 `score=mean_h`(높이), m2는 Grounding DINO `box_score`다 (`methods/m1_sam2_toplayer.py:199`, `methods/m2_grounded_sam.py:208`). m1의 AP는 "높이 순 = 신뢰도 순"이라는 가정에서의 값이므로 **m1/m2의 AP를 직접 비교할 때 주석을 단다.** 대안으로 m1은 SAM2 `predicted_iou`를 점수로 쓴 AP도 병기한다.
- [UNK]에서는 부분 라벨 때문에 FP가 과대 집계되어 AP가 체계적으로 낮게 나온다 → 2순위 이하.

### 2.4 과분할 / 과병합 (split / merge)

현재 정의(GT 하나에 IoU≥0.3 예측이 2개 이상)는 **작은 조각을 놓친다** — GT 면적의 20%짜리 조각은 IoU≤0.2라 세지 않는다 (추론, 정의에서 직접 도출). 대체 정의를 제안한다.

- **split(g)**: `pur_p(g) ≥ 0.8` 이고 `cov_g(p) ≥ 0.1`인 예측 p가 2개 이상인 GT g. → `n_split`, 그리고 `fragments_per_split_gt`(평균 조각 수).
- **merge(p)**: `cov_g(p) ≥ 0.5`인 GT g가 2개 이상인 예측 p. → `n_merge`. [BOX]에서 1순위.
- 기존 정의(`n_oversegmented_gt`, IoU≥0.3)는 연속성을 위해 함께 출력한다.

### 2.5 경계 품질

- **Boundary IoU** (Cheng et al., CVPR 2021, https://arxiv.org/abs/2103.16562): 마스크 경계에서 폭 d 안쪽 띠끼리의 IoU. 확인: 공식 구현 기본값 `dilation_ratio=0.02`, `d = round(0.02 × 영상 대각선)` (https://raw.githubusercontent.com/bowenc0221/boundary-iou-api/master/boundary_iou/utils/boundary_utils.py).
  - 1224×1024 영상이면 대각선 ≈ 1596px → d ≈ 32px (계산). [UNK]의 소형 물체(면적 ~5,000px ≈ 70×70px, `data/eval/01_gt.json`)에서는 띠가 물체 대부분을 덮어 Mask IoU와 거의 같아진다 (추론). → **주 설정 `dilation_ratio=0.005`(≈8px)**, 공개 벤치마크 비교용으로 0.02 병기.
  - 매칭된 쌍(IoU≥0.5)에 대해서만 평균한다. "Boundary AP"(iouType=`boundary`)도 같은 도구로 산출 가능 (확인: `boundary_iou/coco_instance_api/cocoeval.py`의 `iouType` 주석).
- **Boundary F-score (BF)**: 허용 거리 θ px 이내 경계 픽셀의 precision/recall 조화평균 (DAVIS의 F 지표와 동일 계열). θ=3px, 매칭 쌍 평균 (추론, 관행값).
- **주의 — GT 경계 편향**: [UNK] GT는 SAM2 조각의 합집합이다 (확인: `scripts/gen_gt_fragments.py`, `scripts/build_gt_from_groups.py`). 경계가 SAM2 출력 그대로라 **SAM 계열 방법의 Boundary IoU가 구조적으로 유리하다** (추론). [UNK]에서는 경계 지표를 참고치로만 쓰고, 새로 만드는 [BOX]/[ONE]/[FEW] 세트에서 1순위로 쓴다.

### 2.6 Top-K precision (피킹 관점)

로봇은 순위 상위 몇 개만 시도한다. 방법이 **실제 배포에서 쓰는 순위**(m1=높이, m2=신뢰도)를 그대로 쓴다.

- `P@K = (상위 K 예측 중 "피킹 유효" 수) / K`, K ∈ {1,3,5}. 예측 수 < K면 분모는 K 그대로(부족분은 실패).
- **피킹 유효** = (a) 어떤 GT와 IoU≥0.75, (b) merge가 아님(다른 GT를 `cov≥0.2` 이상 덮지 않음), (c) ignore 영역 예측은 제외 후 다음 순위로 채움.
- 장면 평균으로 보고. `P@1`이 사실상 "첫 시도 마스크가 올바른 단일 물체인가"다.

### 2.7 지연 시간 (warm)

- `t_build`: 모델 로드 (참고만).
- `t_pre`: 입력 준비 — depth/법선 정규화·채널 결합 포함. **[RGB-D-N]은 법선 획득 비용을 반드시 여기에 넣는다** (Zivid `copy_data("normals")`, 확인: `C:\Zivid\zivid-python-samples\source\camera\advanced\capture_and_print_normals.py`).
- `t_infer`: 신경망 순전파(들). GPU 구간은 `torch.cuda.synchronize()` 후 경계에서 시간 측정 (추론, 비동기 실행 때문에 필수).
- `t_post`: 마스크 후처리 — NMS, 기하 정련(`core/plane.py`), 필터, 정렬.
- `t_total = t_pre + t_infer + t_post`. **cold 1회 버리고 warm N=10회, 중앙값과 p90**을 보고한다 (PLAN 10장은 warm 6회 중앙값, 확인: `PLAN.md` 10장).

### 2.8 VRAM

- `peak_torch` = `torch.cuda.max_memory_allocated()` (PyTorch 텐서만, `reset_peak_memory_stats()` 후 측정).
- `peak_process` = NVML(`nvidia-ml-py`)의 프로세스 GPU 메모리 최대값 — CUDA 컨텍스트, cuDNN 워크스페이스, ONNX Runtime/TensorRT 할당 포함. **8GB 게이트 판단은 이 값으로 한다** (추론: torch 수치만 보면 ONNX/TensorRT 후보와 비교가 불공정).
- 측정 시 다른 GPU 프로세스(브라우저, Zivid Studio)를 기록해 둔다. 노트북 GPU는 디스플레이와 VRAM을 공유한다 (추론).

## 3. 현재 `run_benchmark.py`가 재는 것과 기준선 수치

### 3.1 측정 항목 (확인: `scripts/run_benchmark.py`)

| 항목 | 정의 | 비고 |
|---|---|---|
| `recall` | IoU≥0.5 매칭 GT / (라벨 GT + 완전 누락) | 1:1 아님 (2.1) |
| `n_oversegmented_gt` | IoU≥0.3 예측이 2개 이상인 GT | 작은 조각 미검출 (2.4) |
| `n_candidates_unmatched_iou0.5` | 어떤 GT와도 IoU<0.5인 예측 | 부분 라벨이라 오탐 아님 — 코드 주석에 명시 |
| `predict_time_s` / `predict_time_warm_s` | cold 1회 / warm **1회** | 분리·반복 없음 |
| `build_time_s` | 모델 로드 | |

### 3.2 기준선 (확인: `data/eval/benchmark_results.json`, 수정일 2026-08-18, 12장면 01~12)

집계는 이 파일을 읽어 직접 계산했다.

| | m1 SAM2 top-layer | m2 Grounded-SAM 2 |
|---|---|---|
| 물체 수 (라벨 108 + 완전 누락 11) | 119 | 119 |
| 매칭 (IoU≥0.5) | 91 | 92 |
| **Recall@0.5 (micro)** | **0.765** | **0.773** |
| Recall (라벨된 GT만, 참고) | 0.843 | 0.852 |
| 장면 평균 recall | 0.815 | 0.833 |
| 최저 장면 recall | 0.50 (06) | 0.556 (06) |
| 예측 총수 / 미매칭 | 140 / 49 | 147 / 49 |
| 매칭 비율 (precision 하한 대용) | 0.65 | 0.667 |
| **과분할 GT (기존 정의)** | **5** | **18** |
| warm 중앙값 / 평균 / 최대 | 0.977 / 1.067 / 1.729s | 0.819 / 0.839 / 1.145s |
| build | 2.7s | 9.35s |

해석 (추론):
- 두 방법의 recall 차이(0.008)는 통계적으로 의미 없다. 119개 물체에서 recall≈0.77의 95% 구간은 이항 근사로 ±0.076이고, 장면 단위 군집을 고려하면 더 넓다. **현재 세트로는 ~8%p 미만 차이를 구분할 수 없다.**
- m1의 과분할 5는 커밋 `6641c95`(포함비율 중복 제거, m1 전용) 이후 값으로 보인다 (확인: git log 메시지, 결과 파일 날짜 기준 추정).
- 장면 06(18개 중 3개 완전 누락, 작은 둥근 물체·쇠 부품)은 두 방법 공통 약점이다 (확인: `scripts/build_gt_from_groups.py` `MISSING`).

### 3.3 갭 (개선 필요)

1. **한 프로세스에서 m1 → m2 순차 실행** (확인: `main()`의 `for method in methods` 루프, m1 객체 해제 없음). `AGENTS.md` "방법 하나만 로드한 독립 프로세스" 규칙 위반이며, m2 warm이 m1 잔류 VRAM 영향을 받았을 수 있다 (PLAN 11장에서 최대 2배 부풀림 확인). → `--methods`를 하나씩 별도 프로세스로 돌리는 래퍼 필요.
2. warm 1회 → 장면 12 m1처럼 warm(1.729s)이 cold(1.153s)보다 큰 이상치가 그대로 남는다 (확인: 결과 파일). N=10 중앙값/p90.
3. 추론/후처리 시간 분리 없음, VRAM 없음.
4. τ=0.75, precision(ignore 적용), AP, P@K, merge, 경계 지표 없음.
5. 예측 마스크를 저장하지 않아 지표를 추가할 때마다 모델을 다시 돌려야 한다. → 예측을 **COCO results JSON(RLE)** 으로 저장하고 채점을 분리.
6. **GT 편향**: (a) SAM2 조각 합집합이라 SAM 계열에 유리, (b) SAM2가 조각을 못 만든 물체는 마스크 없음 → 비-SAM 방법이 그 물체를 잡아도 TP로 셀 수 없음(recall 분모에는 있으나 매칭 불가), (c) GT 범위가 m1과 같은 `top_band_mm`·`bin_roi` 최상층 영역 (확인: `gen_gt_fragments.py`가 `base_cfg["top_layer"]` 재사용). 최상층 밴드 밖을 분할하는 방법은 미매칭이 늘어난다 → 밴드 밖은 ignore 영역으로 처리해야 공정.
7. 장면 `00.zdf`는 GT가 없다 (확인: `data/input/20260811_piece_picking_data/`).

**권장 보완**: 완전 누락 11개 물체에 폴리곤을 직접 그려 넣고(X-AnyLabeling 등, `data/dataset-plan.md` 4장), 기존 GT 중 20%를 수동 경계 보정한 "v2 GT"를 만든다. v1은 기존 결과와의 비교용으로 동결한다.

## 4. 측정 프로토콜

`AGENTS.md` "측정·비교 규칙"을 그대로 따르고 다음을 추가한다.

1. **두 단계 분리**: (a) 실행 단계 — 방법 1개 = 프로세스 1개. build → cold 1회 버림 → warm N=10 (마지막 회의 예측을 저장). (b) 채점 단계 — 저장된 예측 JSON과 GT로 오프라인 계산. 모델 없이 지표 재계산 가능.
2. **입력 동결**: GT 버전·장면 목록·`bin_frame_pp_*.json`·`bin_roi_*.json`의 해시를 결과 JSON에 기록. `build_bin_frame.py` 재실행 금지(RANSAC).
3. **결정성**: 경로에 난수 금지(AGENTS.md). 신경망 쪽은 `torch.manual_seed` 고정, 같은 입력 2회 실행에서 예측 동일 여부를 1회 확인해 기록 (추론: cuDNN 비결정 커널로 미세 차이 가능).
4. **환경 기록**: GPU 드라이버, torch/CUDA 버전, 전원 모드(노트북은 AC 연결·고성능 모드 — 배터리 모드에서 GPU 클럭이 떨어짐, 추론), Zivid SDK 버전.
5. **결과 스키마**: 장면×방법 행에 2장 지표 전부 + `t_pre/t_infer/t_post` 중앙값·p90 + `peak_process_vram_mb`.
6. **불확실성**: 장면 단위 부트스트랩(1,000회)으로 95% 구간. 두 방법 비교는 **같은 장면 쌍 대응 부트스트랩**으로 차이의 구간을 낸다. 구간이 0을 포함하면 "차이 없음"으로 적는다 (추론, 소표본 비교의 표준 방식).

### 4.1 [RGB] vs [RGB-D] vs [RGB-D-N] 공정 비교

| 통제 항목 | 규칙 | 이유 |
|---|---|---|
| GT | 동일 세트, 동일 ignore 영역 | |
| 해상도·ROI | 같은 1224×1024 입력, 같은 `bin_roi_2d` 크롭 | |
| 기하 후처리 | `core/plane.py` 병합·중복 제거를 **켠 결과와 끈 결과를 둘 다** 보고 | m1은 이미 XYZ를 후처리로 쓴다. "depth를 입력에 넣은 효과"와 "depth로 후처리한 효과"를 분리해야 한다 |
| 최상층 게이팅 | 모든 모달리티에 동일하게 적용(또는 동일하게 미적용) | [RGB]만 게이팅 없이 돌리면 불리 |
| depth 표현 | 카메라 Z가 아니라 **bin frame 높이**(림 평면 기준)로 정규화 | AGENTS.md: 기울인 카메라에서 카메라 Z는 높이가 아니다 |
| 법선 표현 | bin frame으로 회전한 법선 또는 카메라 좌표 법선 중 하나로 고정하고 명시 | 절대량/상대량 구분 (AGENTS.md) |
| NaN 처리 | 동일 규칙(예: 0 채움 + 유효 마스크 채널) | Zivid 무효 점 = NaN (`PLAN.md` 2장) |
| 학습형 | 같은 학습 데이터·에폭·증강·시드, 입력 채널만 다르게. 추가 채널 가중치 초기화 방식 명시 | 사전학습 RGB 백본에 채널을 붙이면 초기화가 결과를 좌우 (추론) |
| 시간 | `t_pre`에 depth/법선 준비 비용 포함 | [RGB-D-N]의 실제 비용 |
| 대조 실험 | depth 채널을 상수로 바꾼 입력으로도 평가 | 모델이 depth를 실제로 쓰는지 확인 (추론) |
| 하위 집합 | 반사·투명·검은 물체(depth NaN 다발) 하위 집합을 따로 집계 | depth 이점/손해가 갈리는 지점 (`PLAN.md` 9장 리스크) |

## 5. 도구

| 도구 | 용도 | 라이선스 | 확인 |
|---|---|---|---|
| `pycocotools` (cocodataset/cocoapi) | Mask AP/AR, RLE, crowd(ignore) | BSD 2조항 계열("Simplified BSD") | https://raw.githubusercontent.com/cocodataset/cocoapi/master/license.txt (BSD 조문 확인) |
| `faster-coco-eval` | pycocotools 호환 고속 평가, 경계 평가 기능 | Apache-2.0 | GitHub API `MiXaiLL76/faster_coco_eval` license 필드, README의 "Boundary evaluation" 항목 |
| `boundary-iou-api` | Boundary IoU / Boundary AP | BSD 계열 (`license.txt`) | https://raw.githubusercontent.com/bowenc0221/boundary-iou-api/master/license.txt |
| `bop_toolkit` | BOP 데이터 로딩·마스크 변환 (외부 데이터 사용 시) | MIT | GitHub API `thodan/bop_toolkit` |
| `nvidia-ml-py` (pynvml) | 프로세스 VRAM | BSD (추론) | 미확인 |
| `torch.cuda` 메모리 API | torch 할당 peak | BSD 계열 (PyTorch) (추론) | 미확인 |

권장 구성 (추론): 채점기는 `pycocotools`(AP/AR, ignore) + 직접 구현한 P@K·split/merge·BF(numpy, 수십 줄) + `boundary-iou-api`의 `mask_to_boundary`만 차용. 의존성을 최소로 해 `venv_segpick`을 흔들지 않는다.
