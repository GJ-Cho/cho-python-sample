# PLAN — Line Tracing GUI

`python-test`(개인 실험 폴더)에서 개발한 뒤 이 리포로 이관한 프로젝트입니다. 이 문서는 최종 설계와
그 근거, 폐기된 대안을 기록합니다 — 개발 중 나왔던 임시 아이디어는 정리하고 최종 결정만 남겼습니다.

## 재사용한 기존 컴포넌트 (모두 `zividsamples` 패키지, 수정 없이 import)

- `zividsamples.gui.qt_application.ZividQtApplication` — 앱 부트스트랩, 다크 테마, `run(win, title)`
- `zividsamples.gui.image_viewer.ImageViewer` — 2D 이미지 표시 (클릭/드래그 오버레이는 없어서
  `DrawableImageViewer`로 상속 확장)
- `zividsamples.gui.live_2d_widget.Live2DWidget` — 카메라 라이브 2D 프리뷰
- `zividsamples.gui.buttons_widget.CameraButtonsWidget` — Connect/Capture 버튼 UI
- `zividsamples.gui.robot_control.RobotControl` / `RobotTarget` — 로봇 제어 추상 인터페이스.
  `RobotControlURRTDE`가 이를 구현.
- `zividsamples.transformation_matrix.TransformationMatrix` — 모든 좌표 변환의 공통 타입
- `zividsamples.gui.pose_widget.PoseWidget` — hand-eye 변환 값 표시

## 주요 설계 결정

### 1. 로봇 제어: 공식 `ur_rtde` (raw RTDE 레지스터 방식은 폐기)

초기에는 사용자가 지정한 레퍼런스(`UR_move_xyzRxRyRz_test`)와 동일한 raw RTDE 레지스터 핸드셰이크 +
로봇 쪽 URScript 폴링 루프 방식으로 시작했다. 실기 검증(연결, 왕복 이동)까지 통과했지만, 이후 공식
`rtde_control.RTDEControlInterface` / `rtde_receive.RTDEReceiveInterface`로 전환했다:

- 로봇 쪽에 별도 프로그램을 미리 로드하고 재생 상태로 둬야 하는 수동 단계가 없어짐 — Remote Control
  모드만 켜면 PC에서 직접 제어 가능.
- `movePath()` + `PathEntry`로 여러 웨이포인트를 한 번에 블렌딩 이동시킬 수 있어, 라인 트레이싱처럼
  수십~수백 개 웨이포인트를 연속으로 매끄럽게 통과시켜야 하는 용도에 적합.
- PyPI에 cp313 wheel이 없어 소스 빌드가 필요했지만(Boost/CMake 필요), 실제로 빌드/설치에 성공함.

`robot/robot_control_ur_rtde.py`의 `move_path()`가 `PathEntry(MoveL, PositionTcpPose, params)`를
웨이포인트마다 추가한다. **`params`의 순서는 `[x,y,z,rx,ry,rz, velocity, acceleration, blend]`**
(ur_rtde 소스로 확인됨) — accel/velocity를 바꿔 넣으면 로봇이 처음엔 매우 느리게 램프업하다가 갑자기
빨라지는 증상이 나타난다.

### 2. TCP 오프셋: 로봇 컨트롤러에서 처리 (소프트웨어에서 중복 적용 금지)

그리퍼 팁 오프셋을 `TransformationMatrix` 체이닝으로 소프트웨어에서 처리하는 대신, 로봇 컨트롤러의
TCP Configuration(`set_tcp`)에서 처리하도록 확정했다. 따라서 `waypoint_builder.py`가 만드는 pose는
TCP 오프셋을 곱하지 않은 채로 그대로 `move_l`/`move_path`에 전달된다 — 로봇이 이미 그 pose를 그리퍼
팁 기준으로 해석하기 때문이다. (주의: TCP Configuration 입력은 **미터 단위** — mm 값을 그대로 넣으면
"payload 무게중심이 오프셋 초과" 에러가 난다.)

### 3. 툴 Z축 방향: +Z가 표면 안쪽(찌르는 방향)

실기로 확인됨: TCP 로컬 +Z로 이동하면 그리퍼 팁이 앞으로(찌르는 방향으로) 나간다. 따라서
`waypoint_builder`는 Zivid `Normals`(표면 바깥쪽, 카메라 방향)의 **반대 부호**를 툴 Z축으로 쓴다 —
버그가 아니라 확인된 하드웨어 관례다.

### 4. 웨이포인트 방향(회전): rotation-minimizing frame — 진행방향에 X축을 고정하지 않음

**증상**: `move_path` 실행 중 특정 구간(주로 손목 마지막 축)에서만 로봇이 느려지고, 그 외에는 정상
속도로 움직여 전체적으로 속도가 들쭉날쭉했다. 웨이포인트 최소 간격 강제, blend 자동조정, Z축(법선)
스무딩을 차례로 적용해도 재발했다.

**원인**: 초기 구현은 매 waypoint마다 툴 X축을 "라인의 진행 방향"에 강제로 맞췄다. 그런데 이 그리퍼는
뾰족한 점 형태로 툴 Z축 둘레로 회전 대칭이라, X축을 어디에 두든 물리적으로 무관하다. 즉 X축을
진행방향에 맞추는 것은 **불필요한 회전**을 매 waypoint마다 강제로 추가하는 것이었고, 라인이 휘어지는
구간마다 손목이 불필요하게 많이 돌아야 했다. `move_l`/`movePath`의 속도(v)/가속도(a) 파라미터는
TCP 선속도/선가속도로 정확히 적용되지만, UR 컨트롤러는 이를 만족시키는 동시에 조인트 속도/가속도
한계도 지켜야 하므로, 필요한 회전량이 커지면 그 구간 전체를 자동으로 감속시킨다.

**해결**: X축을 진행방향에 고정하는 대신, 이전 waypoint의 프레임에서 Z축(법선) 변화량만큼만 **최소
회전**시켜 다음 프레임을 만드는 rotation-minimizing frame(parallel transport) 방식으로 교체했다.
Z축(수직 접촉에 필요한 방향)은 그대로 보존되고, Z축 둘레의 "비틀림"만 최소화된다.

합성 나선형 라인으로 수치 검증(10mm 간격, 20mm/s 기준): 기존 방식은 세그먼트당 최대 282°/s의
요구 각속도가 나왔는데, rotation-minimizing frame으로 바꾸자 107°/s로 감소(누적 회전량
1160°→429°). 직선 구간에서는 새 방식이 Z축이 실제로 요구하는 이론적 최소 회전량과 정확히
일치함을 확인했다(불필요한 낭비 회전 0). 실기 테스트(10mm 간격/2mm 블렌드)에서 떨림 없이 매끄럽게
동작함을 확인함.

구현: `geometry/waypoint_builder.py`의 `_build_rotation_minimizing_frames()`.
`_orthonormal_frame()`은 첫 waypoint의 X축을 정하는 시드(임의지만 결정적인 시작값) 용도로만 남았다.

### 5. 웨이포인트 최소 간격 강제 + blend 자동조정

라인을 픽셀 등간격으로 리샘플링하면, 라인이 촘촘하게 휘어진 구간(루프 등)에서는 실제 3D 간격이 목표
간격보다 훨씬 작아질 수 있다(예: 목표 5mm인데 실제 0.9mm). blend 반경이 이 실제 간격의 절반을
넘으면 인접 세그먼트끼리 겹쳐서 로봇이 앞 구간을 건너뛰고 속도가 튀는 현상이 생긴다.

- `_enforce_minimum_spacing()`: 실제 3D 간격이 목표 간격의 50% 미만인 점은 그리디하게 드롭한다.
- generate 시점에 blend 값을 실측된 최소 간격의 40%로 자동 설정한다.

## 현재 상태

Connect / Calibrate / Trace 3탭 모두 동작 확인됨 (카메라 캡처, 라인 드로잉, 웨이포인트 생성 및 3D
프리뷰, 홈 포지션, E-stop/protective-stop 감지 및 재연결, 실제 로봇 트레이싱 실행까지 실기 검증
완료).

## 향후 개선 여지

- 속도/가속도/블렌드 값을 더 공격적으로(빠르게) 튜닝.
- 전체 시퀀스(홈 → 접근 → 트레이싱 → 후퇴 → 홈)를 다양한 실제 표면으로 반복 검증.
