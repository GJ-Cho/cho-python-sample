# cho-python-sample

Zivid 카메라, 로봇(UR), 2D/3D 비전 관련 Python 샘플 코드 저장소입니다.

## 설치

```bash
pip install -r requirements.txt
```

`requirements.txt`의 `./modules` 항목이 `modules/` 패키지를 함께 설치합니다.  
또는 개발 모드로 직접 설치:

```bash
cd modules
pip install -e .
```

## 폴더 구조

```
src/
├── zivid/
│   ├── convert_zdf/            # ZDF → PLY, PNG, Depth map, Normal map, SNR map 변환
│   ├── stitching/              # 로컬 포인트 클라우드 레지스트레이션 기반 스티칭
│   ├── stitching_multi_camera/ # 멀티 카메라 캘리브레이션 및 스티칭
│   ├── camera/                 # 카메라 캡처 및 라이브 스트리밍
│   ├── get_camera_intrinsic/   # 카메라 내부 파라미터 추출
│   └── 4x4_matrix/             # XYZRxRyRz → 4x4 변환 행렬 변환 유틸리티
└── project/
    ├── UR_communication_test/                  # UR 로봇 RTDE 통신 테스트
    ├── UR_move_xyzRxRyRz_test/               # UR 로봇 TCP 이동 검증
    ├── pose_estimation/                       # 2D/3D 포즈 추정
    ├── touch_pose_estimation_2d/             # 2D 클릭 기반 터치 포즈 추정 GUI
    └── touch_pose_estimation_cal_board_marker/ # 캘리브레이션 보드 마커 기반 터치 포즈 추정 GUI

modules/
└── zividsamples/               # 공용 유틸리티 모듈 (GUI, 캘리브레이션, 디스플레이 등)

sample/                         # 테스트용 ZDF 샘플 파일
```

## 주요 샘플 실행

```bash
# ZDF 파일 일괄 변환 (PLY, PNG, Depth/Normal/SNR map) — sample/ 폴더의 ZDF를 자동 탐색
python src/zivid/convert_zdf/convert_zdf_file_dir.py

# 포인트 클라우드 스티칭 (2개 프레임) — C:/ProgramData/Zivid/StitchingPointClouds/BlueObject/ 필요
python src/zivid/stitching/stitch_via_local_point_cloud_registration.py

# 회전 물체 연속 스티칭 (카메라 연결 필요)
python src/zivid/stitching/stitch_continuously_rotating_object.py --settings-path path/to/settings.yml

# 멀티 카메라 캘리브레이션 및 스티칭 — imgXX.zdf, img_test_XX.zdf를 같은 폴더에 배치 후 실행
python src/zivid/stitching_multi_camera/multicam_cal.py

# 카메라 내부 파라미터 추출 — sample/sample_MR130.zdf 필요
python src/zivid/get_camera_intrinsic/get_camera_intrinsics_simple.py

# UR 로봇 RTDE 통신 테스트 (IP는 스크립트 내 IP_ROBOT 상수에서 수정)
python src/project/UR_communication_test/universal_robots_comm_test.py

# UR 로봇 TCP 이동 검증 (+10mm XYZ)
python src/project/UR_move_xyzRxRyRz_test/universal_robots_move_test.py --ip <ROBOT_IP>

# 픽셀 기반 피킹 포즈 추정 프로토타입 (ZDF 파일 경로 및 파라미터는 스크립트에서 직접 수정)
python src/project/pose_estimation/pose_estimation_test.py

# 2D 클릭 기반 터치 포즈 추정 GUI
python src/project/touch_pose_estimation_2d/touch_pose_estimation.py
```

## 프로젝트 샘플 상세

### touch_pose_estimation_2d — 2D 클릭 기반 터치 포즈 추정

2D 이미지에서 터치 지점을 클릭하고, 주변 포인트 클라우드에 SVD 평면 피팅을 적용해 4×4 포즈 행렬을 추정하는 PyQt5 GUI 애플리케이션입니다.

**주요 기능**

- ZDF 파일 로드 또는 카메라 직접 연결·캡처
- 구형 반경(mm) 또는 드래그 사각형으로 ROI 선택
- X-axis 방향 선택: SVD 주축(카메라 +X 부호 정렬) 또는 Camera +X 투영
- SVD 평면 피팅 → 카메라 좌표계 4×4 포즈 행렬
- Open3D 3D 뷰어 (포즈 좌표계 + ROI 하이라이트)
- **Advanced 패널**: Hand-Eye 캘리브레이션(YAML 또는 직접 입력) 기반 로봇 베이스 좌표계 포즈 계산
  - Eye-to-Hand / Eye-in-Hand 모드 선택
  - 9가지 6DoF 출력 형식 (4×4, RotVec, Quaternion, Euler 6종)
- 선택 픽셀 NaN 시 자동 대체 및 경고 출력

자세한 내용은 [`src/project/touch_pose_estimation_2d/README.md`](src/project/touch_pose_estimation_2d/README.md)를 참고하세요.

---

### UR_communication_test — UR 로봇 RTDE 통신 테스트

UR 로봇과 RTDE(Real-Time Data Exchange, 포트 30004) 통신을 검증하고, 6개 관절 각도를 실시간으로 모니터링·로깅하는 스크립트입니다.

**주요 기능**

- RTDE 200Hz로 6개 관절 실제 각도(`actual_q`) 수신
- matplotlib 실시간 그래프 (6개 서브플롯, 루프 카운트 표시)
- `s` 키로 안전 종료, 측정 데이터 → `ur_rtde_joint_log.csv` 저장

자세한 내용은 [`src/project/UR_communication_test/README.md`](src/project/UR_communication_test/README.md)를 참고하세요.

---

### UR_move_xyzRxRyRz_test — UR 로봇 TCP 이동 검증

RTDE double 레지스터(24~29)로 목표 TCP 포즈(XYZRxRyRz)를 전송하고, 현재 포즈에서 X/Y/Z 각 +10 mm 이동 후 실제 이동량을 비교해 검증합니다.

자세한 내용은 [`src/project/UR_move_xyzRxRyRz_test/README.md`](src/project/UR_move_xyzRxRyRz_test/README.md)를 참고하세요.

---

### pose_estimation — 픽셀 기반 피킹 포즈 추정 프로토타입

ZDF 파일에서 고정 픽셀을 선택하고, 주변 포인트 클라우드에 SVD 평면 피팅을 적용해 피킹 포즈(4×4 행렬)를 추정하는 탐색용 스크립트입니다. GUI 없이 코드에 직접 파라미터를 입력하며, `touch_pose_estimation_2d`의 전신에 해당하는 프로토타입입니다.

**주요 기능**

- 구형 ROI 마스킹 → SVD 평면 피팅 → 4×4 포즈 행렬 생성
- X축 방향을 이미지 수평 방향으로 보정 (Gram-Schmidt 직교화)
- Open3D로 포인트 클라우드 + 좌표계를 단계별 시각화

자세한 내용은 [`src/project/pose_estimation/README.md`](src/project/pose_estimation/README.md)를 참고하세요.

---

### touch_pose_estimation_cal_board_marker — 캘리브레이션 보드 / ArUco 마커 기반 터치 포즈 추정

ZDF 파일 또는 라이브 캡처에서 Zivid 캘리브레이션 보드 또는 ArUco 마커를 검출하고, 핸드아이 캘리브레이션 행렬을 적용해 로봇 베이스 좌표계 포즈를 계산하는 PyQt5 GUI 애플리케이션입니다.

**주요 기능**

- 검출 대상 전환: Calibration Board(체커보드) 또는 ArUco Marker
- 카메라 설정 전환: Eye-to-Hand / Eye-in-Hand
- ArUco 마커 복수 검출 시 ID가 가장 작은 마커 우선 사용
- 9가지 회전 출력 형식 (4×4, RotVec, Quaternion, Euler 6종)
- Open3D 3D 뷰어 (검출 포즈 좌표계 표시)

자세한 내용은 [`src/project/touch_pose_estimation_cal_board_marker/README.md`](src/project/touch_pose_estimation_cal_board_marker/README.md)를 참고하세요.

---

## 주요 의존성

| 패키지 | 용도 |
|--------|------|
| `zivid` | Zivid SDK — 3D 카메라 인터페이스 |
| `open3d` | 포인트 클라우드 처리 및 시각화 |
| `opencv-python` | 2D 이미지 처리 |
| `numpy` / `scipy` | 수치 계산, 회전 변환 |
| `pyqt5` | GUI 컴포넌트 (modules 패키지) |
| `robodk` | 로봇 시뮬레이션 및 제어 |

## 참고

- ZDF 파일: Zivid 전용 3D 데이터 포맷 (포인트 클라우드 + 2D 이미지 포함)
- `convert_zdf_file_dir.py`는 `sample/` 폴더의 모든 ZDF를 자동으로 처리합니다.
- `get_camera_intrinsics_simple.py`는 `sample/sample_MR130.zdf` 파일이 있어야 실행됩니다.
- `multicam_cal.py`는 ZDF 파일이 `src/zivid/stitching_multi_camera/` 폴더에 있어야 합니다 (`img01.zdf`, `img_test_01.zdf` …).
- 스티칭 샘플 일부는 실험적 SDK API (`zivid.experimental`) 를 사용하므로 SDK 버전에 따라 변경될 수 있습니다.
- 로봇 통신 샘플은 RTDE 프로토콜(포트 30004)을 사용하며, 로봇에 해당 `.urp` 프로그램이 로드된 상태여야 합니다.
