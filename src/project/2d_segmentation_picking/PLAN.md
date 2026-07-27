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

"물체가 몇 개인지" 알 필요가 없어 중첩·밀착이 문제되지 않는다. **이 장면에서 가장 안 깨지는 방법이므로 데모 실패 방지용으로 먼저 구현할 것.**

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
│   ├── common.yaml              # 단위, SNR 임계, 출력 경로
│   ├── bin_roi.json             # 2D ROI + 림 환형 ROI (티칭 결과)
│   ├── bin_frame.json           # 림 평면 캐시 (자동 갱신)
│   ├── m1_sam2.yaml
│   ├── m2_grounded_sam.yaml
│   └── m3_affordance.yaml
├── data/
│   ├── input/
│   │   └── image_test.zdf       # 사용자가 직접 복사
│   └── output/
├── core/
│   ├── types.py                 # SceneData, PickCandidate
│   ├── loader.py                # zdf / npy 로딩
│   ├── bin_frame.py             # 림 RANSAC, 높이 계산
│   ├── roi.py                   # 2D ROI, 최상층 밴드
│   ├── plane.py                 # RANSAC + PCA 재피팅, 마스크 병합
│   ├── pose.py                  # 후보 -> 4x4 포즈 (카메라 좌표계)
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

Python **3.12 확정** (2026-07-27). 당초 3.11로 계획했으나, 개발 머신의 3.12.10 글로벌 환경에 `zivid 2.18.0`·`open3d 0.19.0`·`opencv 4.12`·`numpy 2.2`·`scipy 1.16`이 모두 정상 설치·동작함을 확인 → 3.12 사용. (torch/SAM2도 3.12 지원)

---

## 7. Phase 계획

| Phase | 내용 | 검증 기준 | 상태 |
|---|---|---|---|
| 0 | 스켈레톤 + `types.py` / `loader.py` / `viz.py` / `inspect_zdf.py` | zdf 로딩 성공, `scene_stats.json` 출력, RGB 두 포맷 비교 | ✅ 완료 (2026-07-27): 1224x1024, 유효 75%, rgba_srgb 채택 |
| 1 | `teach_bin_roi.py` + `bin_frame.py` + `roi.py` | 림 평면 피팅 rms < 2mm, 최상층 밴드 오버레이 육안 확인 | 미착수 |
| 2 | m3 어포던스 + `plane.py` + `pose.py` | 피킹 후보 상위 5개 시각화, 포즈 JSON 출력 | 미착수 |
| 3 | m1 SAM2 | 과분할/과병합 정도 육안 평가, 처리 시간 측정 | 미착수 |
| 4 | m2 Grounded-SAM 2 | 인스턴스 분리 품질 비교 | 미착수 |
| 5 | `run_benchmark.py` | 3안 타이밍 + `center_px` 비교표 | 미착수 |
| 6 | UR3e 연동 | 별 브랜치로 분리 | 미착수 |

**m3를 m1보다 먼저 구현하는 이유**: 세그멘테이션 없이 동작하므로 이 장면에서 가장 안 깨진다. 로봇 피킹 루프의 하한선을 먼저 확보한 뒤 세그멘테이션 기반 방법을 얹는 것이 리스크가 낮다.

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
