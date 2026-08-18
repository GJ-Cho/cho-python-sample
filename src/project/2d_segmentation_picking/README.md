# 2D Segmentation Picking Demo

> 🚧 **WIP — 진행 중인 개인 연구 브랜치입니다.** 아직 main에 머지되지 않았고,
> 파라미터·API가 예고 없이 계속 바뀝니다. PR은 진행 상황 공유·백업 목적이며
> 리뷰/머지 준비가 된 상태가 아닙니다.

Zivid 2+ MR130으로 캡처한 2D 이미지 + organized 포인트 클라우드에서 **사전 학습되지 않은 임의의 물체**를 빈 안에서 세그멘테이션하고, 최종적으로 UR3e로 피킹하는 데모.

전체 설계·결정 근거·폐기된 대안은 [`PLAN.md`](PLAN.md)에 있다. 작업 규칙은 [`CLAUDE.md`](CLAUDE.md).

현재 상태: **Phase 5 진행 중** (m1/m2 세그멘테이션 + 기하 후처리 튜닝 완료, 평가 세트 구축 및 첫 recall 벤치마크 진행 중). Phase 4(파지 포즈)는 그리퍼 스펙 확정 대기, Phase 6(UR3e 연동) 미착수.

---

## 실행 환경

Phase 2부터는 **전용 venv**를 쓴다. 글로벌 Python이 아니다.

```
C:\Zivid\3rdparty\venv_segpick\Scripts\python.exe     # Python 3.12.10, torch 2.6.0+cu124
C:\Zivid\3rdparty\sam2                                # SAM2 (editable install)
C:\Zivid\3rdparty\checkpoints\sam2.1_hiera_small.pt   # 체크포인트
```

구성 내역은 `PLAN.md` 6장 "전용 venv" 참조. 아래 예시의 `$PY`는 위 python 경로다.

## 데이터 준비

`data/`는 전체 gitignore다 (`.zdf` 1장이 약 86MB). 리포를 새로 클론했으면 캡처 파일을 직접 넣는다.

```
data/input/image_test_01.zdf     # 장면 1
data/input/image_test_02.zdf     # 장면 2
```

Zivid Studio나 `zivid-python-samples`의 캡처 샘플로 저장한 `.zdf`를 복사한다 (`scripts/capture_zivid.py`는 `PLAN.md` 6장에 계획돼 있으나 아직 미구현). 파일명은 자유지만 아래 명령의 기본값은 `image_test_01.zdf`다.

## 장면 준비 (장면당 1회)

카메라가 기울여 설치되어 있어 **빈 좌표계(bin frame)** 가 필요하다. 카메라 Z를 높이로 쓰면 빈 한쪽이 통째로 "높다"고 오판된다 (`PLAN.md` 4장).

```bash
# 1) 씬 통계 + RGB 두 포맷 비교 (선택)
$PY scripts/inspect_zdf.py --input data/input/image_test_01.zdf

# 2) 림 4꼭짓점 티칭 — RGB 위에서 빈 외곽을 시계방향으로 클릭
$PY scripts/teach_rim.py --input data/input/image_test_01.zdf --config config/bin_roi_01.json

#    좌표를 이미 알면 비대화형으로:
$PY scripts/teach_rim.py --corners "263,133 263,1143 967,1143 967,133" --no-show

# 3) 림 RANSAC 평면 → bin frame 캐시 + 최상층 마스크 검증 오버레이
$PY scripts/build_bin_frame.py --input data/input/image_test_01.zdf \
   --config-roi config/bin_roi_01.json --config-frame config/bin_frame_01.json
```

`bin_frame_*.json`은 자동 생성물이라 gitignore다. 림 평면 RMS가 2mm를 넘으면 티칭 좌표가 림을 벗어났는지 확인한다 (실측 0.46mm).

> **주의**: 림 평면 피팅은 RANSAC(난수)이라 `build_bin_frame.py`를 다시 돌리면 `bin_frame_*.json`이 미세하게 바뀌고, **그 위에서 계산되는 모든 높이·최상층·기울기 결과가 함께 바뀐다.** 세그멘테이션 자체는 결정적이지만(`PLAN.md` 11장) 이 파일은 그렇지 않다. 기존 결과와 비교할 때는 bin frame을 다시 만들지 말 것.

## 세그멘테이션 실행

```bash
# m1 (SAM2 AMG, 최상층에만 프롬프트)
$PY scripts/run_single.py --method m1 --input data/input/image_test_01.zdf

# m2 (Grounding DINO → bbox → SAM2)
$PY scripts/run_single.py --method m2 --input data/input/image_test_01.zdf

# 다른 장면: ROI/bin frame을 오버라이드 (같은 yaml 재사용)
$PY scripts/run_single.py --method m1 --input data/input/image_test_02.zdf \
   --bin-roi config/bin_roi_02.json --bin-frame config/bin_frame_02.json --tag _s02
```

## 산출물 네이밍 규칙

| 스크립트 | 파일명 |
|---|---|
| `inspect_zdf.py` | `<stem>_rgba_linear.png`, `<stem>_rgba_srgb.png`, `<stem>_scene_stats.json` |
| `teach_rim.py` | `<stem>_rim_annulus.png` |
| `build_bin_frame.py` | `<stem>_bin_frame.png` |
| `run_single.py` | `<stem>_<method><tag>_top<N>.png`, `..._top<N>_report.json` |

`<stem>`은 입력 zdf 이름, `<method>`는 `m1`/`m2`, `<tag>`는 `--tag`로 주는 자유 문자열, `<N>`은 **실제로 그린 마스크 수**다.

`<N>`을 파일명에 넣는 이유: 오버레이는 상위 N개만 그리므로 **같은 결과라도 `--topk`가 다르면 다른 그림이 된다.** 파일명에 없으면 서로 다른 조건의 그림을 같은 조건으로 착각해 비교하게 된다. 후보 전체를 그리려면 `--topk 500`처럼 충분히 크게 준다.

`--tag`는 자유 문자열이라 의미가 파일에 남지 않는다. 그래서 리포트 JSON에 **`config` 전문과 `git`(커밋 해시 + `dirty` 여부)** 를 함께 기록한다. `dirty: true`면 그 커밋만으로는 결과를 재현할 수 없다는 뜻이다.

현재 보관 중인 태그:

| 태그 | 의미 |
|---|---|
| `_base` | Phase 2/3 구현 직후 파라미터 (튜닝 전) |
| `_nogeo` | 튜닝 후 + `geometry.enabled: false` (기하 후처리 기여도 비교용) |
| `_geo` | 현재 상태 — 튜닝 + 기하 후처리 |

### 무엇을 비교하면 되는가

| 알고 싶은 것 | 비교 |
|---|---|
| 기하 후처리 효과 | `image_test_01_m1_nogeo_top40.png` ↔ `image_test_01_m1_geo_top30.png` |
| 파라미터 튜닝 효과 | `image_test_01_m1_base_top4.png` ↔ `image_test_01_m1_nogeo_top40.png` |
| m1 vs m2 | `image_test_01_m1_geo_top30.png` ↔ `image_test_01_m2_geo_top28.png` |
| 장면 난이도 | `image_test_01_m1_geo_top30.png` ↔ `image_test_02_m1_geo_top50.png` |
| 수치·기각 사유 | `*_report.json`의 `stats.rejected`, `topk[].inlier_ratio` |

## 방법 비교 (2026-07-30, 장면 2개)

| | m1 SAM2 top-layer | m2 Grounded-SAM 2 |
|---|---|---|
| 입력 | RGB + XYZ | RGB + XYZ |
| 학습 데이터 / 프롬프트 | 없음 — 진짜 unknown | 텍스트 어휘 필요 |
| 후보 수 (scene01 / scene02) | 30 / 50 | 28 / 42 |
| warm 처리 시간 | 0.98s / 1.39s | 0.96s / 1.22s |
| 모델 로드 (cold, 1회) | 약 3~10s | 약 12~38s |
| 기하 병합 | 사용 (과분할 해소) | **off** — 마스크가 이미 인스턴스 단위 |
| 성격 | 마스크가 많고 경계가 거칠다 | 인스턴스가 깔끔하나 어휘 밖 물체를 놓친다 |

파라미터 스윕 근거는 `PLAN.md` 10장, 기하 후처리는 11장. 타이밍은 **방법 1개만 로드한 독립 프로세스**에서 재야 한다 — m1/m2를 한 프로세스에 함께 올리면 8GB VRAM 압박으로 최대 2배까지 부풀었다.

주의: 후보 수가 많은 것이 곧 품질이 좋은 것은 아니다. 현재 비교는 후보 수와 육안 확인뿐이고, recall·마스크 중복률 같은 정량 지표는 Phase 5(`run_benchmark.py`)에서 도입한다.

## 기하 후처리 (`core/plane.py`)

RGB 세그멘테이션 결과를 XYZ로 정련하는 단계다. m1/m2가 `plane.refine_masks()`를 공유한다.

```
중복 제거(마스크 IoU) → 마스크별 평면 피팅 → 동일 평면 인접 마스크 병합 → 재피팅 → 기각
```

`config/*.yaml`의 `geometry` 블록으로 제어하고, `geometry.enabled: false`로 두면 순수 RGB 결과가 나온다(기여도 비교용). 평면 지표는 후보 `meta`와 리포트 JSON에 실린다.

읽을 때 주의할 점:

- **`plane_rms_mm`(인라이어 기준)만 보고 평면성을 판단하면 안 된다.** 인라이어가 "평면에서 `inlier_thresh_mm` 이내"로 정의되므로 rms는 구조적으로 그 임계 이하로 묶인다. 두 물체를 걸친 마스크도 작게 나온다. 실제 판별력은 **`inlier_ratio`** 와 `plane_rms_all_mm`/`depth_span_mm`(전체 포인트 기준)에 있다.
- `tilt_deg`는 bin frame `n_up` 기준 **절대** 각도다. 병합에 쓰는 normal 각도차는 **상대량**이라 좌표계와 무관하다 — 둘을 섞지 말 것(`PLAN.md` 4장).
- `reject` 임계는 전부 `null`(끔)이다. 정답 라벨 없이 컷오프를 정하면 근거 없는 튜닝이 되므로 평가 세트 도입 후 정한다.

## 트러블슈팅

| 증상 | 원인 / 확인 |
|---|---|
| 후보가 0개 | 최상층 마스크가 비었다. `build_bin_frame.py` 오버레이로 최상층이 내용물에 얹혔는지 확인. `top_band_mm`을 키운다 |
| 빈 바깥 바닥이 최상층으로 잡힘 | 빈 외부 표면이 내용물보다 카메라에 가깝다. 최상층은 반드시 림 안쪽으로 한정해야 한다 (`interior_shrink_mm`) |
| 빈 사면 벽이 최상층에 걸림 | `interior_shrink_mm`을 키운다(기본 50). 잔류는 후속 필터가 처리 |
| m2에서 빈 전체를 덮는 마스크 | 빈 자체가 "box"로 검출된 것. `filter.max_area_frac`(기본 0.5)로 차단 |
| m2 검출이 거의 없음 | `box_threshold`가 높다. 0.30은 어수선한 빈에서 대부분을 놓친다 (현재 0.15) |
| `post_process_grounded_object_detection` 인자 오류 | transformers 5.x는 `threshold=`, 구버전은 `box_threshold=` |
| 절대 높이 게이팅이 전부 잘라냄 | `bin_depth`가 추정값이면 p_floor가 부정확하다. 절대 게이팅은 기본 off, 상대 밴드만 사용 |
| 한 물체가 여러 마스크로 갈라짐 | `geometry.merge`를 켠다. 안 붙으면 `normal_deg`/`offset_mm`/`dilate_px`를 키운다 |
| 서로 다른 물체가 하나로 병합됨 | `merge.dilate_px`를 줄여 실제로 맞닿은 쌍만 남긴다. `normal_deg`/`offset_mm`도 조인다 |
| 후보가 `plane_fit`으로 기각됨 | 유효 XYZ가 `min_plane_points` 미만 — 투명/반사 물체의 NaN 홀. `reject.require_plane: false`로 통과시킬 수 있으나 기하 평가가 불가능한 후보다 |
| 기하 단계가 느림 | 전체 영상 연산이 섞였는지 확인. `valid_mask`는 캐시되어야 하고, 마스크 연산은 경계 상자 안에서 해야 한다(`PLAN.md` 11장) |
