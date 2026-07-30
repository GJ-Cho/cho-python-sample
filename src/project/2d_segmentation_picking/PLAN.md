# 2D Segmentation Picking Demo — 설계 계획서

작성일: 2026-07-27
상태: Phase 0 착수 대기

---

## 1. 프로젝트 목표

Zivid 카메라로 획득한 2D 이미지와 포인트 클라우드를 사용해, **사전 학습되지 않은 임의의 제품(unknown object)** 을 빈(bin) 안에서 세그멘테이션하고, 세그멘테이션된 영역의 평면을 평가해 피킹 포즈를 생성한 뒤 UR3e 로봇으로 피킹한다.

### 입출력 정의

- **입력**: 2D 이미지 1장 (+ 동일 해상도 organized 포인트 클라우드)
- **출력**: 세그멘테이션된 영역 마스크, 그리고 **영역 중심의 픽셀 인덱스** `center_px`
- **후속**: `center_px` → XYZ → 평면 피팅 → normal → 피킹 포즈(4x4)

### 하드웨어

| 항목 | 값 |
|---|---|
| 카메라 | Zivid 2+ MR130 |
| 해상도 | 1224 x 1024 (2x2 subsample) — 2D/3D 동일 |
| 로봇 | UR3e (페이로드 3kg) |
| 카메라 설치 | **기울여 설치** — bin frame(림 평면 기준) 로직 필수 (4장) |
| 엔드이펙터 | **핑거 그리퍼(2지) 유력** (석션 전제에서 변경 — 아래 "그리퍼 변경" 주석 참조) |
| 대상 | 혼재 클러터 빈 — 파우치, 박스, 튜브, 병, 비닐백, 케이블 등 |
| 빈 깊이 | 약 100~200mm (림→바닥) |
| 연산 | RTX 4060 Laptop, 8GB VRAM → SAM2 체크포인트 `sam2.1_hiera_small` 확정 |

> **그리퍼 변경 (2026-07-27)**: 초기 석션 전제에서 **핑거 그리퍼**로 방향이 바뀜. 세그멘테이션(m1/m2)·bin frame·Phase 0~1은 그리퍼와 무관하므로 계획대로 진행하되, **어포던스(m3)와 최종 파지 포즈 생성은 핑거 파지(antipodal 2점, 그리퍼 개폐 폭, 양옆 클리어런스, 파지 축)** 기준으로 재설계한다. 결정 방침: **세그멘테이션 우선, 파지 로직은 그리퍼 확정 후.** `PickCandidate`·`Segmenter` 인터페이스는 gripper-agnostic하게 유지한다.

---

## 2. 공통 데이터 계약

Zivid 네이티브 데이터 형태를 기준으로 한다.

```python
@dataclass
class SceneData:
    rgb: np.ndarray            # (H,W,3) uint8   <- copy_data("rgba_srgb")[:, :, :3]
    xyz: np.ndarray            # (H,W,3) float32, 단위 mm, 무효값 NaN  <- copy_data("xyz")
    snr: np.ndarray | None     # (H,W) float32, 신뢰도 필터용
```

```python
@dataclass
class PickCandidate:
    center_px: tuple[int, int]        # 최종 요구 출력
    mask: np.ndarray | None           # (H,W) bool
    position_mm: np.ndarray | None    # (3,) 카메라 좌표
    normal: np.ndarray | None         # (3,) 카메라 쪽을 향함
    score: float
    plane_rms_mm: float | None
    meta: dict                        # 방법별 부가정보
```

```python
class Segmenter(ABC):
    requires_xyz: bool
    def predict(self, scene: SceneData) -> list[PickCandidate]: ...
```

### 결정 사항: intrinsics는 사용하지 않는다

Zivid는 2D와 3D가 픽셀 단위로 정렬되어 있으므로, 마스크의 `(row, col)`을 `xyz[row, col]`에 그대로 사용할 수 있다. backprojection이 불필요하므로 `SceneData`에서 intrinsics 필드를 **제거**했다.

- m1(SAM2): 불필요
- m3(어포던스): 불필요
- m2(Grounded-SAM 2): 불필요 (마스크는 2D에서 나오고 포즈만 XYZ로 계산)

리사이즈가 필요한 경우 XYZ를 함께 리사이즈하면 되므로 여전히 불필요하다.

### 확정: rgba_srgb 채택 (2026-07-27, Phase 0)

SAM2와 Grounding DINO는 일반 sRGB 자연영상으로 학습되었다. `scripts/inspect_zdf.py`로 `image_test.zdf`의 두 포맷을 PNG로 저장해 비교한 결과, **`rgba_srgb`(luminance 114)가 `rgba`(linear, luminance 54)보다 밝고 자연스러워 `rgba_srgb`를 채택**한다. 데이터 계약(2장)의 `rgb = copy_data("rgba_srgb")[:, :, :3]` 그대로 확정.

참고: 사용자가 제공한 테스트 이미지가 전반적으로 어둡고 주변부 감광이 있다. 3D 캡처의 프로젝터 조명으로 찍힌 RGB일 가능성이 높다. **`Settings2D`로 주변광 기준 별도 2D 캡처를 하고 그것을 세그멘테이션 입력으로 쓰는 것**이 성능에 결정적일 수 있다. 3D 캡처의 RGB는 좌표 매핑용으로만 쓴다.

---

## 3. ROI 정의 — 두 가지를 분리한다

혼동하기 쉬우므로 명확히 구분한다.

### (a) `bin_roi_2d` — 순수 2D 픽셀 ROI

- 빈 벽, 바깥 바닥, 글레어 영역을 제외하는 픽셀 사각형/폴리곤
- 고정 셋업이므로 1회 티칭 후 `config/bin_roi.json`에 저장
- 출력: `(H,W) bool`

### (b) `top_layer_mask` — 3D 높이 기반 필터

픽셀 좌표와 무관하며, **XYZ의 높이값으로 걸러낸다.** 결과만 `(H,W) bool`로 환원된다.

목적: 아래에 깔린 물체는 분할해도 집을 수 없으므로 최상층만 세그멘테이션 대상으로 넘긴다.

높이 정의 — **카메라 좌표계 Z가 아니다**:

```
h = dot(p - p_floor, n_up)     # n_up = 빈 바닥 평면의 위쪽 법선
```

카메라 Z(광축 거리)를 쓰면 카메라가 기울어 설치된 경우 빈 한쪽이 통째로 "높다"고 오판된다.

두 종류 기준을 함께 적용한다:

| 종류 | 정의 | 목적 |
|---|---|---|
| 상대 밴드 | `h > percentile(h, 99) - band_mm` | 현재 장면의 최상층 추적. 빈이 비어가도 자동 적응. 최대값 대신 99퍼센타일로 아웃라이어 회피 |
| 절대 경계 | `floor_margin_mm < h < bin_depth_mm` | 바닥 자체 제외, 림 위/빈 밖 제외 |

### 최종 유효 마스크

```
valid = bin_roi_2d AND 상대밴드 AND 절대경계 AND (snr > snr_min)
```

SAM2의 프롬프트 그리드는 이 마스크 내부에만 배치한다. 프롬프트 수가 크게 줄어 속도도 개선된다.

> **Phase 1 실측 교정 (2026-07-27, image_test.zdf 기준)**
> 1. **bin_roi_2d는 별도 티칭하지 않고 림 4꼭짓점에서 파생**한다. `roi.bin_interior_mask`가 림 안쪽(+ `shrink_mm`로 사면 벽 제외)을 반환. 빈 바깥 표면이 내용물보다 카메라에 가까워(높이 큼) 최상층으로 오검출되므로 **반드시 빈 내부로 한정**해야 한다.
> 2. **절대 경계(`floor_margin < h < bin_depth`)는 기본 사용하지 않는다.** `bin_depth`가 추정값이면 p_floor가 부정확해 절대 게이팅이 전부를 잘라낸다. 실측 전까지는 **상대 밴드(h > p99 − band, ROI 내부에서 계산)** 만으로 최상층을 잡는다. 상대 밴드는 절대 스케일에 무관해 강건하다.
> 3. 빈 사면 안쪽 벽이 림 높이라 최상층에 걸리면 `bin_interior_mask(shrink_mm≈50)`로 벽을 대부분 제외한다. 남는 벽 잔류는 **후속 SAM2/평면성 평가에서 걸러낸다**(별도 색/기하 처리 안 함).
> 4. 채택 파라미터: `rim_band_mm=20`, `interior_shrink_mm=50`, `top_band_mm=50`, 절대 게이팅 off. 결과: 최상층이 더미 최상단 파우치/튜브에 안착.

---

## 4. Bin Frame — 카메라 기울기 불변성

### 문제

카메라가 기울여 설치될 수 있다. 카메라 좌표계 기준의 절대 기하 판정(높이, 기울기)은 모두 왜곡된다.

### 폐기된 대안: 매 캡처 바닥 RANSAC

제품이 빈에 가득 차면 바닥이 거의 보이지 않아 실패한다. **채택하지 않는다.**

### 채택: 빈 림(rim) 평면 기반

빈 상단 테두리는 제품이 얼마나 쌓여도 가려지지 않는다. 테스트 이미지에서도 노란 빈의 테두리가 사방 완전히 노출되어 있다.

1. 1회 티칭: 림 영역을 **환형(annulus) 2D ROI**로 지정 — 외곽/내곽 폴리곤 2개
2. 매 캡처마다 그 환형 픽셀의 XYZ만 뽑아 RANSAC 평면 피팅 → `n_rim`
3. 림 평면은 바닥 평면과 평행하므로 normal을 그대로 사용
4. 바닥 위치는 `p_floor = p_rim + n_down * bin_depth_mm` (빈 내부 깊이는 1회 실측)
5. `config/bin_frame.json`에 캐시하고 매 캡처 갱신

장점: 사전 빈 통 촬영 불필요, 카메라 기울기/빈 미세 이동에 자동 대응.

### 보조 검증: 로베 터치 티칭

같은 레포의 `src/project/touch_pose_estimation_2d`를 재활용해 빈 바닥 3점을 로봇으로 터치하여 평면을 정의하고, 림 기반 추정치와 비교한다. 핸드아이 캘리브레이션 오차까지 함께 드러난다. **Phase 2 이후 옵션.**

### 각도 기준 두 종류를 구분할 것

| 파라미터 | 성격 | 좌표계 의존성 |
|---|---|---|
| `merge_normal_deg` (마스크 병합) | 두 마스크 normal의 **상대** 각도차 | 없음. 상대량이라 안전 |
| `max_tilt_deg` (석션 도달성) | normal과 기준축의 **절대** 각도 | 있음. **반드시 bin frame 기준** |

두 값 모두 하드코딩하지 않고 config로 노출한다. 초기값은 넉넉하게 `merge_normal_deg: 8`, `merge_offset_mm: 3`으로 두고 실데이터 벤치마크로 확정한다. 마스크 병합은 **경계가 실제로 맞닿은 인접 쌍에만** 적용한다.

평면 피팅은 RANSAC 후 인라이어로 재피팅(PCA)하고 `plane_rms_mm`을 신뢰도 지표로 남긴다.

---

## 5. 세그멘테이션 3가지 방법

모두 Python 3.11 + torch 2.x **단일 환경**에서 동작하며, 추정 처리 시간이 1초 이하다.

| ID | 방법 | 입력 | 추정 시간 | 성격 |
|---|---|---|---|---|
| **m1** | 최상층 밴드 + SAM2 AMG + 평면 병합 | RGB + XYZ | 300~800ms | 학습 데이터 0, 진짜 unknown |
| **m2** | Grounded-SAM 2 (Grounding DINO + SAM2) | RGB (+XYZ는 포즈용) | 200~500ms | 인스턴스 단위 깔끔, 텍스트 프롬프트 필요 |
| **m3** | 석션 어포던스 직접 탐색 | XYZ only | 30~100ms | 세그멘테이션 없음. 폴백 |

### m1 — 최상층 밴드 + SAM2 AMG

1. `bin_roi_2d` 크롭
2. bin frame 높이로 최상층 밴드 마스크 생성
3. AMG 포인트 그리드를 **밴드 내부에만** 배치 (프롬프트 1024개 → 100~200개)
4. `multimask_output=False` 로 포인트당 마스크 1개만 생성 (과분할 억제에 가장 효과 큰 설정)
5. 마스크별 평면 피팅 → 인접 마스크 병합 (normal 각도차 + offset 기준)
6. 마스크 erode 후 최종 평면 피팅 (경계 depth 아웃라이어 회피)
7. 최상단 / 최대 면적 / 평면성 기준으로 랭킹

설치: `git clone https://github.com/facebookresearch/sam2.git` → `pip install -e .` → `checkpoints/download_ckpts.sh`

### m2 — Grounded-SAM 2

- Grounding DINO(text → bbox) + SAM2(bbox → mask)
- **중요**: 원본 Grounding DINO 레포는 커스텀 CUDA op 빌드가 필요하다. HuggingFace `transformers`의 `IDEA-Research/grounding-dino-tiny`를 쓰면 **컴파일 없이 순수 PyTorch**로 동작한다. 단일 환경 요구사항을 만족하려면 반드시 이 경로를 쓴다.
- 텍스트 프롬프트 초기값: `"pouch. box. bottle. tube. can. bag."`
- 트레이드오프: 완전한 unknown은 아니다. 카테고리 수준의 프롬프트가 필요하다.

### m3 — 석션 어포던스 직접 탐색

> **핑거 그리퍼 전환 주석**: 아래는 석션 기준 설계다. 핑거 그리퍼로 확정되면 **핑거 파지 어포던스**(대향 파지점 2개, 그리퍼 개폐 폭 내 폭, 양옆 클리어런스/충돌 회피, 파지 축 방향)로 재설계한다. 세그멘테이션 우선 방침에 따라 구현은 그리퍼 확정 후.

1. 포인트 클라우드 법선 추정
2. 석션컵 직경만큼의 원형 패치를 슬라이딩 → 곡률·평면 잔차 계산
3. bin frame 높이, 벽면 거리, 패치 반경 충족 여부로 랭킹
4. 1등 패치 중심 + normal → 피킹 포즈

"물체가 몇 개인지" 알 필요가 없어 중첩·밀착이 문제되지 않는다. ~~데모 실패 방지용으로 먼저 구현~~ → **핑거 그리퍼 전환으로 Phase 4(파지 어포던스)로 미룸.** 핑거 파지 어포던스(대향 2점·개폐 폭·클리어런스·파지 축)로 재설계한다.

### 폐기된 대안

- **순수 기하 (RANSAC + DBSCAN)**: 테스트 이미지의 물체들이 전부 밀착·중첩되어 있고, 연성 파우치가 겹쳐 누워 있어 깊이 불연속이 없다. 하단 분홍 파우치는 바닥과 거의 동일 평면이라 배경 제거 시 함께 지워진다. 클러스터링이 전체를 한 덩어리로 뭉칠 것이므로 폐기.
- **UOIS 계열 (UCN, MSMFormer, UOIS-Net-3D)**: torch 1.x + Python 3.8 + 커스텀 CUDA 확장 빌드가 필수여서 단일 환경 요구사항과 양립 불가. `docs/uois-deferred.md`에 설치 절차만 남기고 후순위로 둔다.
- **SAM2 AMG 단독 (ROI/밴드 없이)**: 3~8초. 속도 미달이며 텍스처 과분할이 심하다.
- **기하 클러스터링으로 SAM2 프롬프트 생성**: 아이디어는 좋으나 이 장면에서는 클러스터링 자체가 실패하므로 프롬프트가 틀린다.

---

## 6. 폴더 구성

기존 레포는 프로젝트당 `README.md` + 단일 `.py` 정도의 가벼운 구조지만, 3안 비교 벤치마크가 목적이므로 모듈을 분리한다. 단 과도한 구조는 피한다.

```
src/project/2d_segmentation_picking/
├── CLAUDE.md
├── PLAN.md                      # 이 문서
├── README.md                    # 실행법, 3안 비교표, 트러블슈팅
├── requirements-add.txt         # 상위 requirements.txt에 추가할 항목만
├── config/
│   ├── common.yaml              # (미생성) 공용 임계는 현재 각 m*.yaml에 있다
│   ├── bin_roi_01.json          # 장면별 림 4꼭짓점 티칭 결과 (teach_rim.py)
│   ├── bin_roi_02.json
│   ├── bin_frame_01.json        # 림 평면 캐시 (build_bin_frame.py 자동 생성, gitignore)
│   ├── m1_sam2.yaml
│   ├── m2_grounded_sam.yaml
│   └── m3_affordance.yaml       # Phase 4
├── data/                        # 전체 gitignore — 용량(.zdf 1장 ≈ 86MB)
│   ├── input/
│   │   └── image_test_01.zdf    # 사용자가 직접 복사 (README "데이터 준비")
│   └── output/
├── core/
│   ├── types.py                 # SceneData, PickCandidate
│   ├── loader.py                # zdf / npy 로딩
│   ├── bin_frame.py             # 림 RANSAC, 높이 계산
│   ├── roi.py                   # 2D ROI, 최상층 밴드
│   ├── plane.py                 # 트리밍 PCA 평면 피팅, 마스크 병합/중복제거/기각 (11장)
│   ├── pose.py                  # (미생성, Phase 4) 후보 -> 4x4 포즈
│   └── viz.py                   # 마스크 오버레이, 포즈 렌더
├── methods/
│   ├── base.py                  # Segmenter 인터페이스
│   ├── m1_sam2_toplayer.py
│   ├── m2_grounded_sam.py
│   └── m3_affordance.py
└── scripts/
    ├── inspect_zdf.py           # 씬 통계 -> scene_stats.json
    ├── teach_bin_roi.py         # 2D ROI + 림 환형 ROI 티칭 GUI
    ├── run_single.py            # --method m1 --input ... --viz
    ├── run_benchmark.py         # 3안 순차 실행 -> 타이밍 CSV + 비교표
    └── capture_zivid.py         # 2D+3D 캡처, rgba/rgba_srgb 비교 저장
```

### 추가 의존성

상위 `requirements.txt`에 이미 `zivid`, `open3d`, `opencv-python`, `scipy`, `pyyaml`, `numpy`, `matplotlib`가 있다. 추가할 것만:

```
torch
torchvision
transformers
huggingface-hub
# sam2 는 git clone 후 pip install -e . (별도 설치)
```

Python **3.12 확정** (2026-07-27). Phase 0~1은 글로벌 3.12로 검증했고, Phase 2(SAM2)부터는 **전용 venv**를 사용한다.

**전용 venv (Phase 2~, 2026-07-27 구축)**: `C:\Zivid\3rdparty\venv_segpick` (Python 3.12.10). 설치: `torch 2.6.0+cu124`·`torchvision 0.21`(CUDA True, RTX 4060), `zivid 2.18`·`open3d 0.19`·`opencv 5.0`·`scipy 1.18`·`matplotlib 3.11`·`huggingface_hub`, SAM2(editable, `C:\Zivid\3rdparty\sam2`), 체크포인트 `C:\Zivid\3rdparty\checkpoints\sam2.1_hiera_small.pt`(config `configs/sam2.1/sam2.1_hiera_s.yaml`). 모든 스크립트는 이 venv의 python으로 실행: `C:\Zivid\3rdparty\venv_segpick\Scripts\python.exe`.

---

## 7. Phase 계획

| Phase | 내용 | 검증 기준 | 상태 |
|---|---|---|---|
| 0 | 스켈레톤 + `types.py` / `loader.py` / `viz.py` / `inspect_zdf.py` | zdf 로딩 성공, `scene_stats.json` 출력, RGB 두 포맷 비교 | ✅ 완료 (2026-07-27): 1224x1024, 유효 75%, rgba_srgb 채택 |
| 1 | `teach_rim.py` + `bin_frame.py` + `roi.py` (+ `build_bin_frame.py`) | 림 평면 피팅 rms < 2mm, 최상층 밴드 오버레이 육안 확인 | ✅ 완료 (2026-07-27): RMS 0.46mm, 최상층이 내용물 최상단에 안착 (아래 교정 참조) |
| 2 | **m1 SAM2 세그멘테이션** (`methods/base.py` + `methods/m1_sam2_toplayer.py`) | 최상층 마스크 → SAM2 분할, 과분할/과병합 육안 평가, 처리 시간 측정 | ✅ 완료 (2026-07-27 구현 `ad547da`, 2026-07-30 튜닝) — 후보 40/62개, warm 1.01/1.39s (10장) |
| 3 | **m2 Grounded-SAM 2 세그멘테이션** | 인스턴스 분리 품질 비교 | ✅ 완료 (2026-07-27 구현 `5114dd3`, 2026-07-30 튜닝) — 후보 28/43개, warm 0.92/1.08s (10장) |
| 3.5 | **기하 후처리** (`core/plane.py`) — XYZ로 RGB 마스크 정련 | 과분할 병합 육안 검증, 사이클 타임 유지 | ✅ 완료 (2026-07-30) — m1 40→30 / 62→50, warm 유지 (11장) |
| 4 | **파지 어포던스 & 포즈** (m3 재설계 + `pose.py`) — **핑거 그리퍼 기준**, 그리퍼 스펙 확정 후 | 피킹 후보 상위 5개 시각화, 파지 포즈 JSON 출력 | 미착수 (그리퍼 스펙 대기: 개폐 폭, 핑거 두께, TCP). `plane.py`는 3.5에서 선구현 |
| 5 | `run_benchmark.py` | 세그멘테이션안 타이밍 + `center_px` 비교표 | 미착수 — 10장 "남은 한계"의 정량 지표(recall, 마스크 중복률)를 여기서 도입 |
| 6 | UR3e 연동 | 별 브랜치로 분리 | 미착수 |

**순서 변경 이유 (2026-07-27)**: 초기 계획은 석션 전제로 "m3(어포던스)를 먼저" 두었으나, 엔드이펙터가 **핑거 그리퍼**로 바뀌면서 파지 어포던스가 복잡해졌다. **세그멘테이션(m1/m2)은 그리퍼와 무관**하므로 먼저 진행하고, 그리퍼에 의존하는 **파지 어포던스(m3)·포즈는 그리퍼 스펙 확정 후(Phase 4)로 미룬다.** m1/m2는 Phase 1의 `top_layer_mask`(빈 내부 최상층)를 입력 프롬프트 영역으로 사용한다.

---

## 8. 확정된 결정 (2026-07-27 사용자 확인 완료)

| # | 항목 | 결정 |
|---|---|---|
| 1 | GPU / VRAM | RTX 4060 Laptop, 8GB → SAM2 체크포인트 **`sam2.1_hiera_small`** 확정 (m2가 Grounding DINO + SAM2 동시 로드하므로 여유 확보) |
| 2 | 인터넷 접근 | **가능** (huggingface.co / github.com 정상). SAM2 체크포인트·`grounding-dino-tiny` 다운로드 문제 없음, 별도 프록시 경로 불필요 |
| 3 | 카메라 설치 각도 | **기울여 설치** → bin frame(림 평면 기준) 로직 필수 (4장 그대로) |
| 4 | 빈 내부 깊이 | 약 **100~200mm** (림→바닥). 정밀값은 티칭 단계에서 실측 |
| 5 | 카메라 모델 | **Zivid 2+ MR130** |
| 6 | 엔드이펙터 | **핑거 그리퍼(2지) 유력** — 석션에서 변경. m3·파지 포즈 재설계 대상 (1장 "그리퍼 변경" 주석 참조) |
| 7 | m2 텍스트 프롬프트 | 기본값 `"pouch. box. bottle. tube. can. bag."`으로 시작, **추후 추가/수정 예정** (config로 노출) |
| 8 | 목표 사이클 타임 | **캡처 ≤ 1s, 탐색(세그멘테이션+포즈) ≤ 1s** 목표. m1(300~800ms)/m2(200~500ms)/m3(30~100ms) 모두 이 범위 내 설계 |

> 남은 확인 항목은 없음. 단, **RGB 포맷(rgba vs rgba_srgb)** 채택은 코드가 아니라 `scripts/inspect_zdf.py`로 두 포맷을 저장해 육안 비교 후 확정한다(2장). 이는 블로킹 질문이 아니라 Phase 0 산출물이다.

---

## 9. 알려진 리스크

- **투명/반사 물체**: 테스트 장면에 투명 용기, 비닐백, 유리병 뚜껑이 있다. NaN 홀과 노이즈가 발생한다. 다중 acquisition HDR 설정 검토 필요.
- **검은 케이블 다발**: 저반사로 포인트 드롭아웃 가능성. 석션으로도 잡히지 않으므로 데모 대상 SKU에서 제외 고려.
- **글레어**: 테스트 이미지 우상단에 빨간 포화 영역이 있다. 노출 재조정 필요.
- **연성 파우치**: 평면 피팅 rms가 크게 나올 수 있다. `max_plane_rms_mm` 임계를 너무 엄격하게 잡으면 후보가 전멸한다.
- **핸드아이 캘리브레이션**: 세그멘테이션 정확도보다 이쪽이 최종 피킹 성공률의 지배 요인일 가능성이 높다.

---

## 10. 튜닝 스윕 결과 (2026-07-30)

Phase 2/3 구현 직후의 검출 수가 너무 적어(m1 4개, m2 2개) 파라미터 스윕을 수행했다. `config/m1_sam2.yaml`·`config/m2_grounded_sam.yaml`의 주석이 이 장을 참조한다.

### 측정 조건

- 장면 2개: `data/input/image_test_01.zdf`(혼재 클러터), `image_test_02.zdf`(더 채워진 빈). 장면별 림 티칭 → `config/bin_roi_01.json`·`bin_roi_02.json`, bin frame은 `bin_frame_01/02.json`(자동 생성).
- 실행: `C:\Zivid\3rdparty\venv_segpick\Scripts\python.exe scripts/run_single.py --method m1 --input ... --bin-roi ... --bin-frame ...`
- 타이밍은 **warm 기준**. `Segmenter.build()`(모델 로드)를 `predict()`에서 분리했고, cold 1회를 버린 뒤 warm 6회의 중앙값을 취했다.

> **타이밍 측정 주의**: 한 프로세스에 m1·m2 모델을 모두 올린 채 재면 8GB VRAM이 압박받아 warm 시간이 최대 2배까지 부풀었다(m2 scene02: 1.08s → 2.16s). **케이스별 독립 프로세스에서 측정해야** 배포 상황(방법 1개만 로드)의 수치가 나온다. 아래 값은 격리 측정치다.

### 베이스라인 → 최종

| 항목 | m1 scene01 | m1 scene02 | m2 scene01 | m2 scene02 |
|---|---|---|---|---|
| 후보 수 (베이스라인 `ad547da`/`5114dd3`) | 4 | 8 | 2 | 5 |
| 후보 수 (최종) | **40** | **62** | **28** | **43** |
| warm 중앙값 (최종) | 1.01s | 1.39s | 0.92s | 1.08s |
| warm 중앙값 (베이스라인) | 0.43s | 0.51s | 0.76s | 1.31s |
| 최상층 면적 px (베이스라인 → 최종) | 63,449 → 315,735 | 92,289 → 376,281 | 동일 | 동일 |

베이스라인 수치는 2026-07-30에 HEAD의 config를 그대로 되돌려 **재실행으로 검증**했다(m1은 `grid_mode: global`을 명시해 당시 동작을 재현). 아래 개별 파라미터의 기여도 순위는 튜닝 세션의 스윕 관찰 결과이며, 재검증한 것은 양 끝점(베이스라인/최종)이다.

### m1 — 지배 인자 순서

`top_band_mm` > `multimask_output` > `pred_iou`/`stability` > `min_overlap` > `points_per_side`

| 파라미터 | 변경 | 근거 |
|---|---|---|
| `top_band_mm` | 50 → **120** | 최상층 면적이 5배로 늘어 프롬프트가 놓일 자리가 생겼다. 검출 수의 지배 인자. 50 = 최상단 한 겹만, 250 ≈ 빈 내부 전체 |
| `multimask_output` | false → **true** | Phase 2의 "과분할 억제" 판단이 과했다. false는 포인트당 후보를 1/3로 줄여 검출이 급감한다. AMG 본래 동작(3개 → IoU/stability/NMS 정리)이 낫다 |
| `pred_iou_thresh` | 0.80 → **0.70** | 흐트러진 파우치·반사 표면이 과도하게 탈락했다 |
| `stability_score_thresh` | 0.90 → **0.85** | 동일 |
| `min_overlap` | 0.5 → **0.15** | 프롬프트가 이미 최상층 내부에만 있으므로 마스크는 최상층에 "닿기만" 하면 된다. 0.5는 밴드 아래로 이어지는 물체를 통째로 탈락시켰다 (베이스라인 scene02에서 overlap 탈락 5건) |
| `grid_mode` | (신규) **bbox** | `global`은 전체 영상에 격자를 깔아 최상층에 살아남는 프롬프트가 몇 개뿐이었다(scene01 50개). 최상층 bbox에 격자를 깔면 같은 프롬프트 예산으로 밀도가 3배 이상(163개) |
| `points_per_side` | 32 → **16** | 효율이 가장 낮은 인자. 시간이 프롬프트 수에 비례하므로 밀도보다 밴드 커버리지에 예산을 쓰는 편이 이득 |

### m2 — 지배 인자 순서

`box_threshold` > 프롬프트 어휘 폭 > NMS/`max_area_frac`(품질 보정)

| 파라미터 | 변경 | 근거 |
|---|---|---|
| `box_threshold` | 0.30 → **0.15** | 어수선한 빈에서 물체 대부분을 놓쳤다. 지배 인자 |
| `text_prompt` | 8단어 → **17단어** | 어휘가 좁으면 그 어휘에 없는 물체를 통째로 놓친다. 미지 물체 빈에서는 넓게 쓴다 |
| `box_nms_thresh` | (신규) **0.7** | Grounding DINO는 텍스트 구(phrase)별로 박스를 내므로 같은 물체가 여러 라벨로 중복된다. threshold를 낮추면 중복이 급증해 클래스 무관 NMS가 필수 |
| `max_area_frac` | (신규) **0.5** | **빈 자체가 "box"/"container"로 검출되어 최상층 전체를 덮는 마스크가 실제로 나왔다.** 이 필터 도입 전 수치는 32/48개였고, 도입 후 28/43개가 실제 물체 기준 값이다 |

**폐기**: `grounding-dino-base`. tiny 대비 2.5배 느리면서 동일 threshold에서 더 낫지 않아 **tiny 유지**.

### 사이클 타임 평가

8장의 목표는 탐색 ≤ 1s다. 최종 warm은 **m1 1.01~1.39s, m2 0.92~1.08s**로 목표를 조금 넘거나 걸치는 수준이다. 장면이 채워질수록(scene02) 느려진다 — m1은 프롬프트 수(163 → 232), m2는 박스 수(30 → 45)에 비례한다. 판단: **Phase 4(파지 포즈)까지 구현한 뒤 전체 사이클로 재평가**한다. 지금 최적화하면 조기 최적화다. 줄일 여지는 m1 `points_per_side`/`top_band_mm`, m2 `box_threshold` 상향인데 모두 검출 수와 직접 상충한다.

### 남은 한계 (정직한 평가)

- **검출 수 증가 = 품질 향상이 아니다.** 이번 튜닝은 "물체를 놓치지 않는 것"에 맞춰 느슨하게 조정했고, 그 대가로 과분할(한 물체가 여러 마스크)과 마스크 중복이 늘었다. 육안으로는 최상층 아이템이 대체로 잡히지만 정량 지표가 없다.
- **정답 라벨이 없다.** 현재 비교는 후보 수와 오버레이 육안 확인뿐이다. Phase 5 벤치마크에서 최소한 (a) 사람이 센 최상층 물체 수 대비 recall, (b) 마스크 중복률(IoU > 0.5 쌍의 비율) 정도는 있어야 m1/m2를 제대로 비교할 수 있다.
- 장면 2개뿐이다. 조명·적재 상태가 다른 장면이 더 필요하다.

---

## 11. 기하 후처리 — XYZ로 RGB 세그멘테이션 정련 (2026-07-30)

10장 튜닝으로 검출 수는 올렸지만 과분할과 마스크 중복이 남았다. 원인은 명확했다:
**XYZ가 최상층 게이팅과 높이 랭킹에만 쓰이고, 세그멘테이션 자체는 순수 RGB였다.**
`core/plane.py`를 만들어 PLAN 5장 m1의 미구현 5·6단계(마스크별 평면 피팅 → 인접 마스크
병합 → erode 후 재피팅)를 구현했다.

### 5장 "순수 기하 폐기"와 모순되지 않는다

5장에서 폐기한 것은 기하를 **주 세그멘터**로 쓰는 경우(RANSAC + DBSCAN)다. 밀착·중첩된
연성 파우치에 깊이 불연속이 없어 클러스터링이 전체를 한 덩어리로 뭉친다는 판단은 유효하다.
여기서 쓰는 것은 **보조 신호**이며 역할이 다르다.

| 용도 | 방법 | 해결 대상 |
|---|---|---|
| 병합 | 인접 마스크의 normal 각도차 + 평면 오프셋이 작으면 같은 물체 | 과분할 |
| 중복 제거 | 마스크 IoU 기준 (AMG의 NMS는 box 기준이라 마스크 중복이 남는다) | 중복 |
| 기각 | 평면성/두께가 나쁜 마스크 = 여러 물체를 걸친 마스크 | 정밀도 |
| 기울기 기각 | bin frame `n_up` 기준 **절대** 기울기 | 빈 사면 벽 잔류 |

병합의 각도차는 두 normal의 **상대량**이라 좌표계와 무관하다. 반면 tilt는 **절대량**이므로
반드시 bin frame 기준으로 계산한다 (4장의 두 각도 구분).

### 결과

| | m1 scene01 | m1 scene02 | m2 scene01 | m2 scene02 |
|---|---|---|---|---|
| 필터 통과 마스크 | 40 | 62 | 28 | 43 |
| → 중복 제거 후 | 36 | 56 | 28 | 42 |
| → 병합 후 = 최종 후보 | **30** (그룹 5) | **50** (그룹 5) | **28** (병합 off) | **42** (병합 off) |
| warm (기하 적용 후) | 0.98s | 1.39s | 0.96s | 1.22s |
| warm (기하 이전, 10장) | 1.01s | 1.39s | 0.92s | 1.08s |

**기하 단계를 추가했는데 m1은 오히려 같거나 빨라졌다.** 아래 성능 항목의 `valid_mask`
캐싱이 기하 비용을 상쇄했기 때문이다.

육안 검증(scene01): 좌상단 종이 파우치 3조각 → 1개, 하단 분홍 박스 2조각 → 1개,
좌측 노란 파우치 2조각 → 1개로 병합. 서로 다른 물체가 잘못 병합된 사례는 확인되지 않았다.

**m2는 병합을 의도적으로 끈다.** m2의 마스크는 이미 인스턴스 단위이고, 겹쳐 쌓인 동일
평면 물체를 병합하면 m2의 장점인 인스턴스 분리를 스스로 훼손한다. 중복 제거와 평면 지표
계산만 공유한다(m1/m2 비교 가능성 확보).

### 구현 중 발견한 것 세 가지

**1. 인라이어 rms는 평면성 지표가 될 수 없다.** 인라이어는 "평면에서 `inlier_thresh_mm`
이내"로 정의되므로 rms는 구조적으로 그 임계 이하로 묶인다. 실측에서 전 후보의
`plane_rms_mm`이 0.09~0.88mm에 몰려 판별력이 전혀 없었다 — **두 물체를 걸친 마스크도
작게 나온다.** 그래서 전체 포인트 기준 `rms_all_mm`과 `inlier_ratio`(지배 평면이 마스크를
설명하는 비율)를 추가했다. 후자가 가장 판별력이 크다.

| 지표 | min | p25 | med | p75 | p90 | max | 판별력 |
|---|---|---|---|---|---|---|---|
| `plane_rms_mm` (인라이어) | 0.09 | 0.42 | 0.75 | 0.82 | 0.87 | 0.88 | 없음 (임계에 묶임) |
| `plane_rms_all_mm` | 0.09 | 0.50 | 2.12 | 3.07 | 5.80 | 10.24 | 있음 |
| `inlier_ratio` | 0.13 | 0.44 | 0.71 | 0.97 | 1.00 | 1.00 | 가장 큼 |
| `depth_span_mm` | 0.37 | 2.89 | 8.25 | 12.27 | 26.29 | 38.09 | 있음 |
| `tilt_deg` | 1.3 | 9.1 | 21.5 | 33.0 | 42.2 | 52.8 | 벽 잔류 탐지용 |

(m1 scene01, 후보 30개)

**2. 성능 — 전체 영상 연산이 병목이었다.** 초기 구현은 warm 2.98s였다. 원인 두 개:
- `SceneData.valid_mask`가 프로퍼티로 매 접근마다 1224x1024 NaN 검사를 다시 했다. 기하
  단계에서 마스크마다 호출되어 수십 회 → **`cached_property`로 변경.**
- erode와 XYZ 추출을 전체 영상에서 했다. 마스크는 보통 영상의 일부다 → **경계 상자 크롭.**

두 수정으로 2.98s → 0.98s. 병합 자체(`merge_coplanar`)는 0.010s로 애초에 값쌌다.

**3. open3d RANSAC → 트리밍 PCA.** 마스크당 20ms대였고 난수를 쓴다. PCA → 임계 밖 포인트
제외 → 재피팅을 3회 반복하는 방식으로 바꿔 1ms 미만이 됐고 **결정적**이 됐다(별개 프로세스
2회 실행에서 후보 30개가 `center_px`·면적·`inlier_ratio`까지 완전 동일). 벤치마크에
난수가 섞이면 안 되므로 이 성질이 중요하다. `core/bin_frame.py`의 림 평면은 1회 계산이고
아웃라이어 비율이 높을 수 있어 RANSAC을 그대로 유지한다.

### 기각 임계는 전부 `null`(끔)로 남겼다

분포는 측정했지만 **정답 라벨 없이 컷오프를 정하면 10장에서 지적한 "애매한 튜닝"을 한 겹
더 쌓는 것**이다. `inlier_ratio 0.13`처럼 분명히 나쁜 후보가 존재하는 것은 확인했으나,
어디서 자를지는 평가 세트가 있어야 정할 수 있다. 병합·중복 제거는 근거가 명확해서 켰고,
절대 임계 기각은 의도적으로 미룬다.
