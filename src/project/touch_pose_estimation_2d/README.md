# 2D Point-Based Touch Pose Estimation

2D 이미지에서 터치 지점을 선택하고, 해당 픽셀 주변의 3D 포인트 클라우드에 SVD 평면 피팅을 적용해 4×4 포즈 행렬을 추정하는 GUI 샘플입니다.

## 워크플로우

```
ZDF 로드 / 카메라 연결 + 캡처
          ↓
2D RGB 뷰어 — 마우스 클릭 또는 u,v 직접 입력으로 터치 포인트 선택
          ↓
ROI 설정 — 구형 반경(mm) 또는 드래그 사각형 선택
          ↓
"Estimate Pose" → 포인트 클라우드 ROI 추출 → SVD 평면 피팅
          ↓
4×4 포즈 행렬 생성 (Z = 카메라를 향하는 표면 법선)
          ↓
"Visualize in 3D" → Open3D 창 (포인트 클라우드 + 포즈 좌표계 + ROI 하이라이트)
```

## 실행 방법

```bash
python touch_pose_estimation.py
```

CLI 인수 없이 실행하면 GUI 창이 바로 열립니다.

## 조작 방법

| 위치 | 동작 | 설명 |
|------|------|------|
| 툴바 | `Load ZDF` 버튼 | ZDF 파일 불러오기 |
| 툴바 | `Connect Camera` 버튼 | Zivid 카메라 연결 |
| 툴바 | `Capture` 버튼 | 연결된 카메라로 캡처 |
| 툴바 | `Click Point` 라디오 | 클릭으로 터치 포인트 선택 모드 |
| 툴바 | `Drag Rectangle` 라디오 | 드래그로 ROI 사각형 선택 모드 |
| 2D 뷰어 | 좌클릭 | 터치 포인트 선택 (Click Point 모드) |
| 2D 뷰어 | 클릭 드래그 | ROI 사각형 선택 (Drag Rectangle 모드) |
| 우측 패널 | `u` / `v` 스핀박스 | 픽셀 인덱스 직접 입력 |
| 우측 패널 | `Estimate Pose` 버튼 | SVD 평면 피팅 → 포즈 행렬 계산 |
| 우측 패널 | `Visualize in 3D` 버튼 | Open3D 3D 시각화 창 열기 |

## 출력

- 터치 포인트 3D 좌표 (X, Y, Z, mm)
- 표면 법선 벡터 (카메라를 향하는 방향)
- 4×4 포즈 행렬
- Open3D 시각화 창 (포인트 클라우드 + 포즈 좌표계 + ROI 하이라이트)

## 주요 파라미터

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| ROI Radius | 10 mm | 구형 ROI 반경. 우측 패널 `r =` 스핀박스로 조정 |
| ROI Rectangle | — | 2D 뷰어에서 드래그로 직접 선택 |

## 파일 구조

```
touch_pose_estimation_2d/
├── touch_pose_estimation.py   # 메인 스크립트 (PyQt5 GUI)
├── README.md
└── sample/                    # 테스트용 ZDF 파일 위치
```
