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

Connect / Calibration / Line Tracing 3탭 모두 동작 확인됨 (카메라 캡처, 라인 드로잉, 웨이포인트 생성
및 3D 프리뷰, 홈 포지션, E-stop/protective-stop 감지 및 재연결, 실제 로봇 트레이싱 실행까지 실기 검증
완료).

단, 위 실기 검증은 전부 **eye-to-hand** 기준이다. `build_waypoints`는 처음부터 `robot_pose` 인자로
eye-in-hand를 지원했지만 GUI에서 그 값을 넣을 방법이 없어 거부하고 있었는데, Calibration 탭의
**Robot Capture Pose** 섹션으로 연결했다.

캡처 pose는 **캡처 시점에 자동으로 기록**한다(`TracePanel.on_capture_clicked` →
`CalibrationPanel.record_capture_pose`). 캡처 중에는 로봇이 정지해 있으므로 그 순간의 현재 pose가 곧
캡처 pose이고, 사용자가 타이밍을 신경 쓸 필요가 없다. 기록된 값은 되돌아갈 위치로도 쓰인다
(**Move To Capture Pose** 버튼 → `RobotConnectionWidget.move_to_pose`). YAML 불러오기와 수동
읽기도 남겨 두었지만, 수동 읽기는 캡처 당시 자세 그대로 서 있을 때만 유효하다.

### RTDE의 command 측과 receive 측은 서로 다른 TCP를 가질 수 있다

**실기에서 발견된 문제**: 팬던트 Installation에 TCP가 정상 지정되어 있는데도 웨이포인트로 이동할 때
**플랜지가 목표 지점에 놓였다**(즉 `base_T_flange = base_T_target`). 팁이 가야 하는데.

원인은 RTDE의 두 인터페이스가 서로 다른 TCP를 참조한다는 점이다:

| | 무엇을 쓰는가 |
|---|---|
| `rtde_receive.getActualTCPPose()` | 컨트롤러의 활성 TCP (= 팬던트 Installation) |
| `rtde_control.moveL` / `movePath` / `getInverseKinematics` | **`RTDEControlInterface` 자신의 TCP** |

`RTDEControlInterface`는 자체 TCP 오프셋을 들고 있고(`getTCPOffset` / `setTcp`), 이것이 팬던트
Installation 값과 자동으로 같아지지는 않는다. `connect()`는 `setTcp`를 호출하지 않았으므로 command
측이 0인 상태로 이동 명령이 나갔고, `moveL`은 **command 측 TCP**를 목표 pose에 놓으므로 플랜지가
목표에 갔다. 읽기(`get_pose`)는 컨트롤러 값을 반영하니 좌표만 봐서는 정상으로 보인다 — 이 비대칭이
원인을 가렸다.

**해결** — Connect 탭 TCP Offset 섹션의 **Sync Command TCP** 버튼 (`sync_command_tcp_offset`):

1. `get_controller_tcp_offset()`이 `getForwardKinematics(q, [0]*6)`로 플랜지 pose를 구한다 — TCP를
   0으로 명시하므로 command 측 TCP가 무엇이든 정확하다.
2. `flange^-1 · getActualTCPPose()` → 컨트롤러가 실제로 쓰는 TCP를 **측정만으로** 복원한다.
3. 그 값을 `setTcp()`로 command 측에 심는다. 이후 `moveL`/`movePath`/IK가 팁을 목표에 놓는다.

보정이 필요했는지 여부를 같은 섹션에 표시한다 — 조용히 고치면 실제 설정 문제를 가리게 되므로.

**버튼인 이유 (한 번 실수한 부분)**: 처음엔 `connect()` 직후 자동 실행하도록 넣었는데, RTDE 왕복
호출 3개(그중 하나는 control-script 호출인 `getForwardKinematics`)를 UI 스레드에서 동기로 돌리는
바람에 **Connect 버튼이 `Connecting...`에서 멈췄다.** 게다가 `setTcp`는 로봇 상태를 바꾸는 쓰기
동작이라 "연결만 했는데" 일어나야 할 일이 아니다. 지금은 명시적 버튼이고, 다른 블로킹 로봇 호출과
같은 워커 스레드 + `processEvents` 패턴을 써서 응답이 느려도 창이 얼지 않는다.

`get_flange_pose()`는 `get_pose() * get_tcp_offset().inv()` 구현으로 유지한다 — eye-in-hand 캡처마다
호출되는 경로라 control-script 왕복을 넣지 않는다. 대신 command 측이 동기화되어 있어야 정확하므로,
eye-in-hand + Pose Reference=Flange 로 쓸 거면 Sync Command TCP 를 먼저 눌러야 한다.

### eye-in-hand의 함정: 로봇이 알려주는 pose는 TCP pose다

결정 #2에 따라 이 프로젝트는 그리퍼 팁을 **0이 아닌 TCP**로 컨트롤러에 설정해 둔다. 그런데
`RobotControlURRTDE.get_pose()`는 `getActualTCPPose()`, 즉 그 TCP가 **적용된** `base_T_tcp`를
돌려준다. 프레임 체인은 이렇게 생겼다:

```
base ──→ flange ──→ tcp ──→ camera ──→ point
              └ flange_T_tcp ┘
```

`flange_T_tcp`(= 팬던트에 설정한 TCP 오프셋)는 체인상 flange와 camera **사이에** 있다. 그래서
캘리브레이션이 TCP 기준(`hand_eye = tcp_T_camera`)이면 필요한 것은 `base_T_tcp`이고,

```
base_T_point = base_T_tcp     · tcp_T_camera · camera_T_point
             = base_T_flange · flange_T_tcp · tcp_T_camera · camera_T_point
```

두 줄은 `base_T_tcp = base_T_flange · flange_T_tcp`를 대입한 **같은 식**이다(항이 추가된 게 아니다).
여기서 `base_T_flange`를 넣으면 `flange_T_tcp` 링크가 빠져 체인이 끊어진다.

- `get_flange_pose()`가 `base_T_tcp * flange_T_tcp^-1`로 오프셋을 되돌린다.
- Calibration 탭의 **Pose Reference**(Flange / TCP)로 어느 규약인지 고른다. 기본값은 Flange.
- **판단 기준**: hand-eye 캘리브레이션을 돌릴 때 팬던트에 TCP가 활성이었다면 기록된 로봇 pose가
  TCP pose이므로 → **TCP**. TCP가 0이었다면 → **Flange**. `hand_eye`와 `robot_pose`가 같은 프레임
  규약이어야 한다는 것이 요점이고, 어느 쪽이 "옳다"가 아니다.
- **틀렸을 때의 증상**: 오차를 base 프레임의 변환 하나로 정리하면

  ```
  base_T_error = base_T_flange · flange_T_tcp · base_T_flange^-1
  base_T_point = base_T_error · base_T_point_wrong
  ```

  `base_T_flange`(캡처 pose)가 경로 전체에 대해 고정값이므로 같은 변환이 모든 웨이포인트에 걸린다 —
  **경로가 찌그러지지 않고 통째로 `flange_T_tcp`만큼 강체 이동**한다. 팁이 그 이동된 경로를 따라가므로
  **원래 라인 위를 플랜지가 지나가는 것처럼 보인다**. (실기에서 실제로 관측됨. `flange_T_tcp`에 회전이
  있으면 밀림에 더해 캡처 pose를 중심으로 한 회전까지 생긴다.)
- 반대로 **Move To Capture Pose**는 `move_j`가 TCP 타깃을 받으므로 flange pose에 `flange_T_tcp`를
  **다시 곱해서** 보낸다. 안 그러면 TCP 오프셋만큼 못 미치는 곳으로 간다.
- Pose Reference를 바꾸면 이미 기록된 pose는 다른 규약의 값이므로 지운다.

eye-to-hand는 `robot_pose`를 아예 쓰지 않으므로(`camera_to_base = hand_eye`) 이 문제가 없다. 실기
검증이 전부 eye-to-hand였던 탓에 드러나지 않았던 부분이다.

### 3D 프리뷰로는 이 부류의 오류를 검증할 수 없다

`TracePanel`은 웨이포인트를 `camera_to_base.inv() * waypoint`로 카메라 프레임에 되돌려 그린다.
웨이포인트 자체가 `camera_to_base * point_camera`로 만들어졌으므로 이 둘이 **대수적으로 상쇄되어
정확히 `point_camera`가 나온다** — hand-eye나 capture pose 체인이 아무리 틀려도 프리뷰는 완벽하게
보인다. 프리뷰는 픽셀→3D 대응과 법선 방향만 검증한다.

체인을 검증할 수 있는 것은 **로봇이 직접 측정한 pose**뿐이다:

- `show_current_position`이 그리는 노란 점은 `camera_to_base.inv() * (로봇이 보고한 pose)`다. 로봇의
  측정값은 캘리브레이션 체인과 독립이므로, **팁을 표면의 알아볼 수 있는 지점에 조그로 대고** 노란 점이
  점 구름의 그 지점에 찍히는지 보면 된다. TCP 오프셋만큼 벗어나 있으면 Pose Reference가 틀린 것이다.
- 웨이포인트 개수 옆에 어떤 체인을 썼는지 표시된다(`_transform_chain_description`).

**eye-in-hand 경로는 아직 실기로 확인하지 않았다** — 첫 사용 시 저속으로, 위의 노란 점 대조로 Pose
Reference부터 확정한 뒤 진행할 것.

## 향후 개선 여지

- 속도/가속도/블렌드 값을 더 공격적으로(빠르게) 튜닝.
- 전체 시퀀스(홈 → 접근 → 트레이싱 → 후퇴 → 홈)를 다양한 실제 표면으로 반복 검증.
