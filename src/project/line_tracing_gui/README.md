# Line Tracing GUI

UR3e(뾰족한 그리퍼) + Zivid2+ MR130(eye-to-hand, 고정 거치) 조합으로, 캡처한 2D 이미지 위에 그린 라인을
실제 3D 표면에 수직으로 접촉하며 따라가게 하는 PyQt5 GUI입니다.

Zivid 공식 `zivid-python-samples`의 Hand-Eye GUI와 같은 스택/관례를 따릅니다: PyQt5, `zividsamples`
패키지의 위젯 재사용, 다크 테마, QSettings 기반 설정 저장.

## 사전 준비

- 리포 루트에서 `pip install -r requirements.txt` (`modules/zividsamples`, `ur_rtde`, `pyqt5`,
  `pyqtgraph`, `pyopengl` 등 포함).
- **로봇**: UR 티치펜던트에서 **Remote Control 모드**를 켜 둘 것 (e-Series: 우측 상단 아이콘 →
  Remote Control, 또는 Installation → General → Remote Control). 별도로 로드해야 하는 로봇 쪽
  프로그램은 없습니다 — 공식 `ur_rtde` 라이브러리가 PC에서 직접 제어합니다.
- **TCP 오프셋**(그리퍼 팁 위치)은 로봇 컨트롤러에서 설정합니다: PolyScope Installation → TCP
  Configuration (미터 단위). GUI의 Connect 탭에서 현재 값을 조회/수정할 수도 있습니다.
- **카메라**: Zivid2+ MR130, hand-eye 캘리브레이션 결과(YAML)가 필요합니다. eye-to-hand(고정 거치)와
  eye-in-hand(플랜지 장착)를 모두 지원하지만, 실기 검증은 eye-to-hand로만 했습니다.

## 실행

```
python main.py                                  # 이 폴더 안에서
python src/project/line_tracing_gui/main.py      # 리포 루트에서
```

## 탭 구성

- **Connect**: 카메라 연결/라이브 프리뷰, 로봇 IP 연결, 상태(정상/E-stop/protective-stop) 표시,
  현재 TCP/조인트 모니터링, TCP 오프셋 조회/수정.
- **Calibration**: hand-eye 캘리브레이션 YAML 불러오기. eye-in-hand를 선택하면 아래에 **Robot Capture
  Pose** 섹션이 나타납니다 — 캡처 시점의 로봇 pose가 추가로 필요하기 때문입니다. 이 값은 캡처할 때마다
  **자동으로 기록**되므로(캡처 중에는 로봇이 정지해 있으니 그 순간의 pose가 곧 캡처 pose) 따로 입력할
  필요가 없습니다. 기록된 pose로 **Move To Capture Pose** 버튼을 눌러 로봇을 되돌릴 수 있고, 필요하면
  YAML로 불러오거나 로봇에서 수동으로 읽을 수도 있습니다.
- **Line Tracing**: 캡처 → 2D 이미지 위에 라인 드로잉 → 웨이포인트 생성(간격 조절 가능, 3D 프리뷰로 확인) →
  접근/후퇴 거리·속도·가속도·블렌드 설정 → 실행(홈 → 접근 → 라인 트레이싱 → 후퇴 → 홈).

## 오프라인 개발 테스트

카메라/로봇 없이 리포 루트의 `sample/sample_MR130.zdf`, `sample_MR130_2d.png`로 동작을 확인할 수 있는
스크립트입니다 (`src/project/`에서 실행):

```
python -m line_tracing_gui._dev_test_waypoint_builder
python -m line_tracing_gui._dev_test_pointcloud_preview
python -m line_tracing_gui._dev_test_drawable_viewer
```

## 안전

이 GUI는 **순수 위치 제어**입니다 — force/torque로 실제 접촉을 감지하지 않고, 사전에 계산된 좌표로
이동할 뿐입니다. 캘리브레이션 오차나 표면 굴곡이 있으면 그리퍼가 과도하게 파고들거나 허공에서 멈출 수
있습니다. 새 캘리브레이션·새 표면으로 처음 테스트할 때는 저속 + 충분한 접근 오프셋으로, 사람이 즉시
비상정지 가능한 거리에서 진행하세요.

설계 결정과 그 근거는 `PLAN.md`를 참고하세요.
