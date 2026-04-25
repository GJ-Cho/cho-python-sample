# 2D Point-Based Touch Pose Estimation

2D 이미지에서 사용자가 클릭한 픽셀을 기반으로 터치 지점의 3D 포즈를 추정하는 샘플입니다.

## 개요

```
ZDF 로드 / 카메라 캡처
       ↓
2D RGB 이미지 표시 → 마우스 클릭으로 터치 포인트 선택
       ↓
선택한 픽셀 → 포인트 클라우드에서 3D 좌표 조회
       ↓
터치 포인트 주변 구형 ROI 추출 → SVD 평면 피팅
       ↓
4x4 포즈 행렬 생성 (Z = 표면 법선)
       ↓
Open3D로 포즈 좌표계 시각화
```

## 실행 방법

```bash
# ZDF 파일로 실행
python touch_pose_estimation.py --zdf sample/your_file.zdf

# 카메라 직접 연결
python touch_pose_estimation.py --live

# ROI 반경 조정 (기본값: 10mm)
python touch_pose_estimation.py --zdf sample/your_file.zdf --roi-radius 15
```

## 조작 방법

| 동작 | 설명 |
|------|------|
| 좌클릭 | 터치 포인트 선택 |
| Enter | 선택 확정 → 포즈 추정 시작 |
| Esc | 취소 및 종료 |

## 출력

- 터치 포인트 3D 좌표 (X, Y, Z, mm)
- 표면 법선 벡터
- 4x4 포즈 행렬
- Open3D 시각화 창 (좌표계 + 포인트 클라우드)

## 주요 파라미터

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `--roi-radius` | 10 mm | 평면 피팅에 사용할 구형 ROI 반경 |

## 파일 구조

```
touch_pose_estimation_2d/
├── touch_pose_estimation.py   # 메인 스크립트
├── README.md
└── sample/                    # 테스트용 ZDF 파일 위치
```
