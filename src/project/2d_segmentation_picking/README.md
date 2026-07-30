# 2D Segmentation Picking Demo

Zivid 2+ MR130으로 캡처한 2D 이미지 + organized 포인트 클라우드에서 **사전 학습되지 않은 임의의 물체**를 빈 안에서 세그멘테이션하고, 최종적으로 UR3e로 피킹하는 데모.

전체 설계·결정 근거·폐기된 대안은 [`PLAN.md`](PLAN.md)에 있다. 작업 규칙은 [`CLAUDE.md`](CLAUDE.md).

현재 상태: **Phase 3 완료** (m1/m2 세그멘테이션 동작 + 튜닝). Phase 4(파지 포즈)는 그리퍼 스펙 확정 대기.

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

산출물은 `data/output/`에:

- `<stem>_<method><tag>.png` — 마스크 오버레이 + `center_px` 마커
- `<stem>_<method><tag>_report.json` — 후보 목록, 타이밍(build/cold/warm), 단계별 통계

`--tag`로 설정을 바꿔가며 결과를 나란히 남길 수 있다.

## 방법 비교 (2026-07-30, 장면 2개)

| | m1 SAM2 top-layer | m2 Grounded-SAM 2 |
|---|---|---|
| 입력 | RGB + XYZ | RGB (+XYZ는 포즈용) |
| 학습 데이터 / 프롬프트 | 없음 — 진짜 unknown | 텍스트 어휘 필요 |
| 후보 수 (scene01 / scene02) | 40 / 62 | 28 / 43 |
| warm 처리 시간 (중앙값) | 1.01s / 1.39s | 0.92s / 1.08s |
| 모델 로드 (cold, 1회) | 약 3~10s | 약 12~38s |
| 성격 | 마스크가 많고 경계가 거칠다. 과분할 경향 | 인스턴스가 깔끔하나 어휘 밖 물체를 놓친다 |

파라미터 스윕 근거와 지배 인자 순위는 `PLAN.md` 10장. 타이밍은 **방법 1개만 로드한 독립 프로세스**에서 재야 한다 — m1/m2를 한 프로세스에 함께 올리면 8GB VRAM 압박으로 최대 2배까지 부풀었다.

주의: 후보 수가 많은 것이 곧 품질이 좋은 것은 아니다. 현재 비교는 후보 수와 육안 확인뿐이고, recall·마스크 중복률 같은 정량 지표는 Phase 5(`run_benchmark.py`)에서 도입한다.

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
