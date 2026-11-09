# ZividAPI.md — Zivid 코딩 레퍼런스 (Python 중심)

> 에이전트용 압축 레퍼런스. Zivid 코드를 쓰기 전에 이 파일을 먼저 읽고, 여기서 부족한 부분만 아래 "확인 순서"대로 찾아본다.
> 이 파일의 API 이름은 로컬 `C:\Zivid\zivid-python-samples`(SDK 2.18.0 기준) 샘플에서 확인한 것이다.
> 마지막 정리: 2026-11-09

## 0. 버전 기준 (가장 먼저 확인)

- 로컬 샘플 클론 기준: **Zivid SDK 2.18.0**. zivid-python은 SDK와 같은 버전 번호를 쓴다(예: 2.18.0, 패치 시 2.18.0.1).
- **실제로 실행되는 환경의 버전이 유일한 기준**이다. 코드 작성 전에 확인:
  ```
  python -c "import zivid; print(zivid.__version__, zivid.SDKVersion.full)"
  ```
- 설치 버전과 GitHub master 버전이 다르면, master 코드를 그대로 쓰지 말고 해당 버전 태그/릴리스의 샘플을 본다.
  (API가 버전마다 바뀐다. 예: 과거 `camera.capture(settings)` → 현재 `capture_2d_3d()` / `capture_3d()`)

## 1. 확인 순서

1. **이 파일** — 기본 패턴과 샘플 위치.
2. **Zivid Knowledge Base** — 개념, 설정값 의미, 튜토리얼. https://support.zivid.com/en/latest/
3. **로컬 클론 grep** — `C:\Zivid\zivid-python-samples` (C#: `C:\Zivid\zivid-csharp-samples`, HALCON: `C:\Zivid\zivid-halcon-samples`, Isaac Sim: `C:\Zivid\zivid-isaac-sim`).
   웹보다 빠르고 토큰이 적으며 버전이 고정돼 있다. 아래 4장 인덱스로 관련 파일 1~2개만 연다.
4. **GitHub (https://github.com/zivid)** — 로컬에 없는 레포(`zivid-python`, `zivid-cpp-samples`, `zivid-ros` 등)이거나 최신 변경을 확인할 때만.
   파일은 `github.com/.../blob/...` 페이지 대신 `raw.githubusercontent.com` 경로로 읽는다(HTML 껍데기 없이 원문만 옴).
   `zivid-python` 레포에서는 README와 Python 모듈 계층만 보고 `src`(C++ 바인딩), `cmake`, `continuous-integration`, `test`는 건너뛴다.

API 이름·인자·설정 항목은 위 출처에서 확인된 것만 쓴다. 확인 못 했으면 추측하지 말고 사용자에게 묻는다.

## 2. 개념 구조 (KB API Reference 기준)

```
Application ──connect()──> Camera
Camera + Settings   ──capture3D()────> Frame   ──> PointCloud, DeviceArray, Hand-Eye
Camera + Settings   ──capture2D3D()──> Frame   (Frame 안에 Frame2D 포함)
Camera + Settings2D ──capture2D()────> Frame2D ──> Image, DeviceArray
Camera ──showImage()──> Projector
```
- Settings = 3D 캡처 설정, Settings2D = 2D 캡처 설정. 2D+3D 캡처는 `Settings.color`에 Settings2D를 넣는다.
- KB 문서 구조: Camera(Basic / Advanced / Info and Maintenance), Applications(Basic / Advanced / Data Types).

## 3. Python 핵심 패턴 (샘플에서 확인됨)

### 연결
```python
import zivid

app = zivid.Application()  # 다른 Zivid 클래스보다 먼저 생성하고, 사용하는 동안 살려둔다
camera = app.connect_camera()                                    # 첫 번째 사용 가능한 카메라
camera = app.connect_camera(serial_number="2020C0DE")
camera = app.connect_camera(address=zivid.CameraAddress("172.28.60.5"))  # IP 또는 hostname
for cam in app.cameras():                                        # 탐색
    print(cam.info.serial_number, cam.state.network.ipv4.address, cam.state.status)
camera.disconnect()
camera = app.create_file_camera("path/to/FileCameraZivid2M70.zfc")  # 카메라 없이 개발할 때
```

### 캡처
```python
settings = zivid.Settings(
    acquisitions=[zivid.Settings.Acquisition()],
    color=zivid.Settings2D(acquisitions=[zivid.Settings2D.Acquisition()]),
)
frame = camera.capture_2d_3d(settings)   # 2D+3D (컬러 포인트 클라우드)
frame = camera.capture_3d(settings)      # 3D만 (color 불필요)

settings = zivid.Settings.load("settings.yml")   # Studio에서 export한 YML 사용 가능
settings = zivid.capture_assistant.suggest_settings(
    camera, zivid.capture_assistant.SuggestSettingsParameters()
)
```
- 2D 전용 캡처(KB의 `capture2D()`)는 Python에서 `capture_2d(settings_2d)`이고 `Frame2D`를 반환한다:
  ```python
  settings_2d = zivid.Settings2D(acquisitions=[zivid.Settings2D.Acquisition()])
  frame_2d = camera.capture_2d(settings_2d)
  image = frame_2d.image_rgba_srgb()   # 또는 image_bgra_srgb() (OpenCV용)
  ```
  (확인: `applications/advanced/capture_undistort_2d.py`, `barcode/barcode_detector.py`)
- **샘플링 설정은 카메라 모델마다 다르다**(Zivid 2 / 2+ / 2+ MR / 3 계열). `source/camera/advanced/capture_2d_and_3d.py`의 `_get_2d_and_3d_settings()` 참고. 모델은 `camera.info.model`, 비교 대상은 `zivid.CameraInfo.Model.*`.

### 데이터 접근·저장
```python
image = frame.frame_2d().image_rgba_srgb()    # 2D 컬러 이미지
image.save("ImageRGBA.png"); rgba = image.copy_data()

pc = frame.point_cloud()
xyz  = pc.copy_data("xyz")          # [H, W, 3] float, 단위 mm
rgba = pc.copy_data("rgba_srgb")    # [H, W, 4] uint8
snr  = pc.copy_data("snr")
h, w = pc.height, pc.width

pc.transform(matrix_4x4)            # in-place 변환 (예: hand-eye 결과로 카메라 → 로봇 베이스)
frame.save("Frame.zdf")             # 확장자로 포맷 결정 (.zdf, .ply 등)
frame = zivid.Frame("Frame.zdf")    # ZDF 읽기 (Application이 먼저 있어야 함)
```

### 버전·카메라 정보
```python
zivid.__version__, zivid.SDKVersion.full
camera.info, camera.state, camera.state.temperature
```

## 4. 작업 → 샘플 인덱스 (`C:\Zivid\zivid-python-samples\source\...`)

| 하려는 작업 | 샘플 |
|---|---|
| 기본 캡처·저장(PNG/ZDF/PLY) | `camera/basic/capture.py` |
| 연결 방법 전체 | `camera/basic/connect.py` |
| YML 설정으로 캡처 | `camera/basic/capture_with_settings_from_yml.py` |
| 설정 전체 항목 예시 | `camera/basic/capture_hdr_complete_settings.py` |
| 파일 카메라 | `camera/basic/capture_from_file_camera.py` |
| 2D/3D 분리 캡처, 모델별 샘플링 | `camera/advanced/capture_2d_and_3d.py` |
| 2D 전용 캡처(`capture_2d`) | `applications/advanced/capture_undistort_2d.py` |
| 버전·카메라 정보·온도 | `camera/info_util_other/camera_info.py` |
| 인트린직(OpenCV 모델) | `camera/info_util_other/get_camera_intrinsics.py` |
| 웜업 / 헬스체크 | `camera/info_util_other/warmup.py`, `check_health.py` |
| 네트워크 IP 설정 | `camera/info_util_other/network/` |
| Infield 검증·보정 | `camera/maintenance/` |
| ZDF 읽기·픽셀 순회 | `applications/basic/file_formats/read_iterate_zdf.py` |
| ZDF 변환 | `applications/basic/file_formats/convert_zdf.py` |
| 시각화 | `applications/basic/visualization/` |
| 다운샘플 / 마스크 / 깊이맵 | `applications/advanced/downsample.py`, `mask_point_cloud.py`, `create_depth_map.py` |
| 2D 언디스토트 / 컬러 밸런스 | `applications/advanced/capture_undistort_2d.py`, `color_balance.py` |
| 좌표 변환(체커보드·ArUco·mm→m) | `applications/advanced/transform/` |
| ROI 박스 | `applications/advanced/roi/` |
| Hand-Eye 캘리브레이션·결과 활용 | `applications/advanced/hand_eye_calibration/` (UR: `ur_hand_eye_calibration/`) |
| 멀티 카메라 | `applications/advanced/multi_camera/` |
| 스티칭 | `applications/advanced/stitching/` |
| 프로젝터 | `applications/advanced/projector/` |
| GPU(CUDA·PyTorch·CuPy) 연동 | `applications/advanced/cuda/` |
| 바코드 | `applications/advanced/barcode/` |
| 공용 유틸(행렬 저장/로드, 샘플 데이터 경로, 디스플레이) | `modules/zividsamples/` |

## 5. 자주 틀리는 점

- `zivid.Application()` 없이 다른 Zivid 객체를 만들거나, Application이 먼저 소멸되면 실패한다.
- 포인트 클라우드 좌표 단위는 **mm**다. 로봇/ROS 쪽이 m이면 변환을 명시한다.
- 유효하지 않은 점은 NaN으로 들어온다. 통계·피팅 전에 NaN을 걸러낸다.
- 컬러는 `rgba`와 `rgba_srgb`가 있다. 표시·저장용은 샘플처럼 sRGB 쪽을 쓴다.
- `zividsamples`는 `zivid`와 별개 패키지다(`zivid-python-samples/modules`에서 설치). 실제 import되는 것은 설치된 site-packages 버전이다.
- 샘플링·설정값은 카메라 모델마다 다르므로, 특정 모델용 값을 범용 코드에 하드코딩하지 않는다.

## 6. 자주 쓰는 KB 링크

- API Reference: https://support.zivid.com/en/latest/camera/api-reference/api-reference.html
- Capture Tutorial: https://support.zivid.com/en/latest/camera/academy/camera/capture-tutorial.html
- 2D+3D Capture Strategy: https://support.zivid.com/en/latest/camera/academy/camera/2d3d-capture-strategy.html
- Point Cloud Tutorial: https://support.zivid.com/en/latest/camera/academy/applications/point-cloud-tutorial.html
- Point cloud 구조·출력 포맷: https://support.zivid.com/en/latest/camera/reference-articles/point-cloud-structure-and-output-formats.html
- Hand-Eye: https://support.zivid.com/en/latest/camera/academy/applications/hand-eye.html
- Camera Settings: https://support.zivid.com/en/latest/camera/reference-articles/camera-settings.html
- Troubleshooting: https://support.zivid.com/en/latest/camera/support/troubleshooting.html

한국어 KB는 `/en/` → `/ko/`로 바꾸면 된다. 용어가 어색하면 영어 원문을 기준으로 한다.
