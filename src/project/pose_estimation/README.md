# pose_estimation — 2D 픽셀 기반 피킹 포즈 추정 프로토타입

ZDF 파일에서 고정 픽셀을 선택하고, 해당 픽셀 주변의 포인트 클라우드에 SVD 평면 피팅을 적용해 피킹 포즈(4×4 행렬)를 추정하는 스크립트입니다.  
GUI 없이 코드에 직접 파라미터를 입력하는 탐색용 프로토타입입니다.

> **참고**: 이 스크립트는 초기 알고리즘 검증용 프로토타입입니다. GUI 기반의 완성된 버전은 [`touch_pose_estimation_2d`](../touch_pose_estimation_2d/) 프로젝트를 사용하세요.

## 파일 구조

```
pose_estimation/
└── pose_estimation_test.py   # 메인 스크립트
```

## 실행 방법

```bash
python pose_estimation_test.py
```

실행 전 스크립트 내 파라미터를 직접 수정합니다.

```python
data_file = "C:/ProgramData/Zivid/BinWithArucoMarker.zdf"  # ZDF 파일 경로
target_height = 600   # 타겟 픽셀 행 인덱스
target_width  = 960   # 타겟 픽셀 열 인덱스
outer_radius_threshold = 10  # ROI 구형 반경 (mm)
```

## 동작 흐름

```
ZDF 파일 로드 → xyz, rgba 배열 추출
    ↓
target_height, target_width 픽셀의 3D 좌표 취득
    ↓
구형 ROI 마스킹 (inner=0mm ~ outer=10mm)
    ↓
[1단계 시각화] ROI 영역 포인트 클라우드 Open3D 표시
    ↓
SVD 평면 피팅 → U 행렬(법선 포함), 평면 내 평균 점 계산
    ↓
[2단계 시각화] SVD 기반 좌표계 표시 (Z = 법선)
    ↓
X축 수정: 이미지 수평 방향(target_width ± 10 픽셀) 벡터를 평면에 투영
    ↓
[3단계 시각화] 원래 좌표계 + 수정된 좌표계를 동시에 표시
    ↓
최종 4×4 피킹 포즈 터미널 출력
```

## 주요 알고리즘

### SVD 평면 피팅 (`_plane_fit`)
- ROI 내 유효 포인트의 중심점을 기준으로 분산 행렬 M 구성
- `np.linalg.svd(M)`으로 U 행렬 계산
- U 행렬의 3번째 열 = 평면 법선 벡터 (Z축)

### X축 방향 결정
- 이미지 상에서 `(target_width-10)` → `(target_width+10)` 픽셀 벡터를 vx로 사용
- vx와 vz(법선)가 수직이 되도록 vx[2] 보정 후 정규화
- vy = vx × vz (오른손 좌표계)

### 좌표계 구성 (`_get_transformation_matrix`)
- Z = SVD 법선, X = 카메라 +X 유사 방향, Y = Z × X

### Z축 회전 (`_get_z_rotation_matrix`)
- 추가 Z축 회전 적용 가능 (`z_rotation` 파라미터, 기본값 0)

## 시각화

| 단계 | 내용 |
|------|------|
| 1단계 | ROI 포인트 클라우드만 표시 |
| 2단계 | SVD 좌표계 (size=30) + 전체 포인트 클라우드 |
| 3단계 | SVD 좌표계 (size=20) + 수정 좌표계 (size=30) 동시 표시 |

각 시각화 창을 닫으면 다음 단계로 진행됩니다.

## 주요 의존성

| 패키지 | 용도 |
|--------|------|
| `zivid` | ZDF 파일 로드, 포인트 클라우드 추출 |
| `open3d` | 포인트 클라우드 및 좌표계 시각화 |
| `numpy` | SVD 계산, 좌표 변환 |
| `nptyping` | 배열 타입 어노테이션 |
