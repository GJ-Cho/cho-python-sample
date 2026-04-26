# 2D Point-Based Touch Pose Estimation

2D 이미지에서 터치 지점을 선택하고, 해당 픽셀 주변의 3D 포인트 클라우드에 SVD 평면 피팅을 적용해 4×4 포즈 행렬을 추정하는 PyQt5 GUI 샘플입니다.  
Hand-Eye 캘리브레이션 결과를 입력하면 로봇 베이스 좌표계 기준의 포즈도 함께 계산합니다.

## 워크플로우

```
ZDF 로드 / 카메라 연결 + 캡처
          ↓
2D RGB 뷰어 — 마우스 클릭 또는 u, v 직접 입력으로 터치 포인트 선택
          ↓
ROI 설정 — 구형 반경(mm) 또는 드래그 사각형 선택
          ↓
X-axis Mode 선택 — SVD 주축(카메라 +X 부호 정렬) 또는 Camera +X 투영
          ↓
"Estimate Pose" → 포인트 클라우드 ROI 추출 → SVD 평면 피팅
          ↓
4×4 포즈 행렬 (카메라 좌표계)
          ↓
(선택) Advanced 패널 → Hand-Eye 캘리브레이션 입력 → "Compute Robot Pose"
          ↓
4×4 로봇 베이스 포즈 + 선택한 6DoF 표현 형식
```

## 실행 방법

```bash
python touch_pose_estimation.py
```

CLI 인수 없이 실행하면 GUI 창이 바로 열립니다.

## 조작 방법

| 위치 | 동작 | 설명 |
|------|------|------|
| 툴바 | `Load ZDF` | ZDF 파일 불러오기 |
| 툴바 | `Connect Camera` | Zivid 카메라 연결 |
| 툴바 | `Capture` | 연결된 카메라로 캡처 |
| 2D 뷰어 | 좌클릭 | 터치 포인트 선택 (Radius 모드) |
| 2D 뷰어 | 클릭 드래그 | ROI 사각형 선택 (Rectangle 모드) |
| 우측 패널 | `u` / `v` 스핀박스 | 픽셀 인덱스 직접 입력 |
| 우측 패널 | ROI — Radius | 구형 반경(mm) 기준 ROI |
| 우측 패널 | ROI — Rectangle | 2D 뷰어 드래그로 사각형 ROI 선택 |
| 우측 패널 | X-axis Mode | SVD 주축 / Camera +X 투영 선택 |
| 우측 패널 | `Estimate Pose` | SVD 평면 피팅 → 카메라 좌표계 포즈 계산 |
| 우측 패널 | `Visualize in 3D` | Open3D 3D 시각화 창 열기 |
| Advanced 패널 | Hand-Eye Calibration | YAML 로드 또는 4×4 직접 입력 |
| Advanced 패널 | Config | Eye-to-Hand / Eye-in-Hand 선택 |
| Advanced 패널 | Robot Capture Pose | Eye-in-Hand 시 캡처 시점 로봇 포즈 입력 |
| Advanced 패널 | 6DoF 형식 | 로봇 포즈 출력 표현 방식 선택 |
| Advanced 패널 | `Compute Robot Pose` | 로봇 베이스 좌표계 포즈 계산 |

## X-axis Mode

| 모드 | 설명 |
|------|------|
| SVD 주축 | SVD 결과의 지배 고유벡터를 X축으로 사용. 카메라 +X 방향과 반대이면 부호 반전 |
| Camera +X 투영 | 카메라 +X 벡터 [1,0,0]을 표면 평면에 투영. 이미지 좌→우 방향에 가까운 X축 생성 |

## Advanced — 로봇 베이스 포즈 계산

Hand-Eye 캘리브레이션 결과(4×4 행렬)를 이용해 카메라 좌표계 포즈를 로봇 베이스 좌표계로 변환합니다.

**Eye-to-Hand**
```
T_robot_base = T_handeye @ T_camera
```

**Eye-in-Hand**
```
T_robot_base = T_robot_capture @ T_handeye @ T_camera
```

행렬 입력 방식은 Zivid YAML 파일(`FloatMatrix` 형식) 로드 또는 4×4 숫자 16개 직접 입력을 지원합니다.

### 6DoF 출력 형식

| 인덱스 | 형식 |
|--------|------|
| 0 | 4×4 Matrix |
| 1 | Rotation Vector [rx ry rz (rad)] |
| 2 | Quaternion [qx qy qz qw] |
| 3 | Euler XYZ extrinsic — RPY (Roll Pitch Yaw) deg |
| 4 | Euler ZYX extrinsic — Yaw-Pitch-Roll deg |
| 5 | Euler ZYX intrinsic — KUKA A-B-C deg |
| 6 | Euler ZYZ extrinsic deg |
| 7 | Euler ZYZ intrinsic deg |
| 8 | Euler XYZ intrinsic deg |

## NaN 터치 포인트 처리

선택한 픽셀에 깊이 값이 없을 경우(NaN), 픽셀 거리 기준으로 가장 가까운 유효 3D 점을 ROI 기준점으로 대체 사용합니다.  
터치 포인트 위치는 ROI 유효 점들의 centroid로 설정되며, 결과 화면과 터미널에 경고가 표시됩니다.

## 주요 파라미터

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| ROI Radius | 10 mm | 구형 ROI 반경. 우측 패널 `r =` 스핀박스로 조정 |
| ROI Rectangle | — | 2D 뷰어에서 드래그로 직접 선택 |
| X-axis Mode | SVD 주축 | 포즈 X축 방향 결정 방식 |

## 출력

- 터치 포인트 3D 좌표 (X, Y, Z, mm)
- 표면 법선 벡터 (카메라를 향하는 방향)
- 카메라 좌표계 4×4 포즈 행렬
- (Advanced) 로봇 베이스 좌표계 4×4 포즈 행렬 + 선택 6DoF 표현
- Open3D 시각화 (포인트 클라우드 + 포즈 좌표계 + ROI 하이라이트)
- 터미널 박스형 결과 출력

## 파일 구조

```
touch_pose_estimation_2d/
├── touch_pose_estimation.py   # 메인 스크립트 (PyQt5 GUI)
├── README.md
└── sample/                    # 테스트용 ZDF 파일 위치
```
