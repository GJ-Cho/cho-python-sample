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

### 6. 실행 전 도달성 검사 (pre-flight)

`move_path` 도중 웨이포인트 하나가 도달 불가로 판명되면 로봇은 **라인 중간에서 팁을 표면 가까이 둔 채로
멈춘다.** 멈추기에 가장 나쁜 위치이고 복구도 번거롭다. 그래서 움직이기 전에 전체 시퀀스를 컨트롤러에게
미리 물어본다 — `robot/reachability.py`.

컨트롤러에 던지는 질문은 pose 하나당 3가지다.

| 질문 | ur_rtde API | 무엇을 잡는가 |
|---|---|---|
| 팔이 닿는가 | `getInverseKinematicsHasSolution` | 작업 반경 밖, IK 해 없음 |
| 안전 평면 안인가 | `isPoseWithinSafetyLimits` | 팬던트에 설정한 safety plane 위반 |
| 조인트 범위 안인가 | `isJointsWithinSafetyLimits` | 조인트 각도 제한 위반 |

도달성과 안전 한계는 **서로 독립**이다. 완벽히 풀리는 pose가 safety plane 밖에 있을 수 있고, 그 반대도
된다. 그래서 셋 다 묻는다.

**qnear 체이닝.** 각 pose의 IK는 *직전 스텝의 조인트 해*를 `qnear`로 넣어 푼다. 로봇이 실제로 경로를
지나가는 방식과 같게 맞춘 것이다. 매번 현재 조인트 값으로 풀면 IK가 웨이포인트마다 다른 팔 자세를 골라도
되므로, 정작 잡아야 할 자세 전환(branch flip)이 검사에서 사라진다.

**linear vs joint 구간 구분.** 접근·후퇴·홈은 `moveJ`라 컨트롤러가 조인트 경로를 자유롭게 고른다 — 여기서
조인트가 크게 변하는 건 정상이므로 보고하지 않는다. 반면 웨이포인트는 `movePath`의 `MoveL` 엔트리들이고,
`movePath`는 **로봇이 서 있는 자리에서 시작**하므로 `접근 pose → 첫 웨이포인트` 구간도 직선 이동이다.
직선 구간에서 인접 웨이포인트 간 조인트 변화가 크면 IK가 다른 팔 자세로 넘어갔다는 뜻이고, 컨트롤러는
그 세그먼트를 거부하거나 크게 휘둘러 지나간다. 임계값(기본 30°)이 휴리스틱이라 **실패가 아니라 경고**로
보고한다.

**결과 처리.** 도달 불가가 하나라도 있으면 실행을 시작하지 않는다(사용자 우회 없음). 경고만 있으면 확인
대화상자에 함께 띄우고 사용자가 판단한다. 3D 뷰에서 도달 불가는 빨강, 경고는 주황으로 표시해 "라인을
어디서 다시 그릴지"를 눈으로 찾게 한다.

**검사 시퀀스는 실행 시퀀스와 반드시 같아야 한다.** 다른 시퀀스를 검사하는 pre-flight는 없느니만 못하다 —
얻지 못한 신뢰를 주기 때문이다. 그래서 접근/후퇴 pose는 `TracePanel._approach_retreat_poses()` 하나에서
만들어 검사와 실행이 같은 값을 쓴다.

**로봇 접근은 5개 조회 메서드로 한정**했다(`get_joint_positions`, `has_inverse_kinematics_solution`,
`get_inverse_kinematics`, `is_pose_within_safety_limits`, `is_joints_within_safety_limits`). 덕분에
시퀀싱/보고 로직을 로봇 없이 스텁으로 검증할 수 있다(`_dev_test_reachability.py`). 스텁이 이 5개 외에는
아무 메서드도 갖고 있지 않으므로, 검사 코드가 로봇을 **움직이려 하면 그 테스트가 AttributeError로 깨진다** —
"움직이는 pre-flight 검사"는 없는 것보다 나쁘기 때문에 일부러 이렇게 잡았다.

### 7. 이동 실패 원인 표시

`moveL`/`moveJ`/`movePath`는 성공 여부를 bool 하나로만 준다. 그래서 예전에는 어떤 원인이든 메시지가
똑같이 "robot may have refused or stopped the motion"이었다. `describe_motion_failure()`가 실패 시점에
로봇 상태를 되읽어 원인을 한 문장으로 붙인다. 확인 순서는 실제로 원인일 확률이 높은 것부터다.

1. RTDE 연결이 끊겼는가
2. `isProgramRunning()` — RTDE control script가 죽었는가 (**1순위 후보**. 실기에서 관측된
   `moveJ (to joints) failed`가 이 모양이었다)
3. E-stop / protective stop
4. `getRobotMode()` / `getSafetyMode()` 숫자를 이름으로 변환해 표시
5. 위가 전부 정상이면 — 타겟이 도달 불가였거나 safety limit 밖이었을 가능성이 크다고 안내

이 함수는 **실패 경로에서 도는 코드라 절대 예외를 던지지 않는다.** 여기서 2차 예외가 나면 원래 원인이
가려진다.

### 8. 워커 스레드는 위젯을 읽지 않는다 (그리고 항상 결과를 보고한다)

`TracePanel`은 이동/검사를 워커 스레드에 돌리고 GUI 스레드는 `while thread.is_alive():
QApplication.processEvents()`로 기다린다. 이 구조에서 **워커가 스핀박스를 읽으면 안 된다** — Qt 위젯은
GUI 스레드 소유다. 도달성 검사를 처음에 그 방식으로 만들었더니 offscreen 테스트가 실제로 멈췄고, 같은
패턴이 `_run_execution`에도 있었다.

규칙은 두 가지다.

1. **입력은 GUI 스레드에서 전부 읽어 평범한 값으로 넘긴다.** `_perform_reachability_check`와
   `on_execute_clicked`가 스핀박스·`config`·웨이포인트를 읽어 `kwargs`로 만들고, 워커
   (`_run_reachability_check` / `_run_execution`)는 인자만 본다.
2. **워커는 무슨 일이 있어도 결과를 큐에 정확히 하나 넣는다.** 호출자가 `result_queue.get()`에서
   블로킹하므로, 워커가 보고 없이 죽으면 GUI가 영구 정지한다. 그래서 인자 검증까지 `try` 안에 넣었다.

접근/후퇴 pose는 `_approach_retreat_poses()`로 한 번 만들어 검사와 실행에 **같은 객체**를 넘긴다
(`_perform_reachability_check`가 이 쌍을 인자로 받는 이유). 각자 다시 계산하면 검사한 경로와 실행되는
경로가 갈라질 수 있고, 그런 pre-flight는 없느니만 못하다.

### 9. 프로젝터로 라인을 실제 표면에 투사 (`geometry/projection.py`)

3D 프리뷰는 `camera_to_base`의 **역변환으로 되돌려 그리기** 때문에 그 체인이 틀려도 오류가 상쇄되어
멀쩡하게 보인다(위 「3D 프리뷰로는 이 부류의 오류를 검증할 수 없다」 참조). 카메라 내장 프로젝터로 빛을
실제 표면에 쏘면 상쇄가 일어나지 않는다 — 계산이 말하는 자리에 빛이 떨어지고, 그게 의도한 자리인지는
눈이 판단한다.

**무엇이 검증되는지는 어느 pose로 되돌리느냐에 달렸다.** 이게 이 기능의 핵심이다.

| 모드 | `camera_to_base` | 검증되는 것 | 어긋났을 때 |
|---|---|---|---|
| **캡처 pose** (기본) | `capture_pose · hand_eye` | 픽셀→3D **만** (hand_eye가 소거됨) | 원인 하나 → 진단 명확 |
| **현재 pose 보정** | `current_pose · hand_eye` | 픽셀→3D **+ hand-eye 체인** | 둘 중 어느 쪽인지 구분 불가 |

`build_waypoints`가 쓴 체인을 그대로 재현하면 `hand_eye`가 소거되므로 캡처 pose 모드는 순수한 픽셀→3D
검사다. 로봇을 다른 자리로 옮겨 보정 투사했는데도 선이 제자리에 떨어지면 **hand-eye 체인까지 맞다는
뜻**이다. 그래서 순서가 중요하다: 먼저 캡처 pose에서 통과시키고, 그 다음에 옮겨서 보정 투사를 본다.
캡처 pose에서 이미 어긋나 있으면 보정 투사는 아무것도 말해주지 못한다.

eye-in-hand는 카메라가 팔에 달려 있으므로 캡처 이후 로봇이 움직이면 투사가 엉뚱한 곳에 간다.
현재 pose와 캡처 pose를 비교해(`PROJECTION_POSE_TOLERANCE_MM/DEG`) 어긋나 있으면
**Move To Capture Pose / Project From Here / Cancel** 중에서 사용자가 고른다. eye-to-hand는 카메라가
고정이라 로봇 pose가 아예 개입하지 않으므로 이 확인을 건너뛴다.

**보정 투사를 크게 신뢰하면 안 되는 이유 3가지** (각도를 조금 바꾸는 정도에서만 유효하다):

- **가림(occlusion)을 처리하지 않는다.** `pixels_from_3d_points`는 depth buffer 없는 순수 투영이므로,
  새 시점에서 가려진 점도 그대로 투사되어 실제로 거기 있는 다른 면에 빛이 떨어진다.
- 프로젝터가 닿지 못하는 점은 **NaN** 또는 해상도 밖 좌표로 돌아온다. 둘 다 버리고 **선을 끊는다** —
  버린 구간을 직선으로 이어 그리면 웨이포인트가 말한 적 없는 경로를 빛으로 그리는 셈이 된다.
- 프로젝터 초점은 고정이다. 캡처 거리에서 크게 벗어나면 기하가 맞아도 선이 흐려진다.

**색은 도달성 검사 결과를 그대로 쓴다** — 도달 가능 초록 / 경고 주황 / 도달 불가 빨강, 검사 전이면 노랑
(`zividsamples`의 hand-eye verification이 confirmed/unconfirmed에 쓰는 초록/노랑 짝과 같다).
세그먼트 색은 **양 끝점 두 개의 더 나쁜 쪽**으로 정한다. 초기 구현은 런 색을 누적 비교해서 도달 불가
웨이포인트 하나가 그 뒤 전체를 빨강으로 삼켜 이후 판정을 다 가렸다 — 이 표시의 목적과 정반대여서
`_dev_test_projection`에 회귀로 못박아 뒀다. 점(dot)은 세그먼트와 달리 **자기 자신의 판정 색**으로
찍어서, 도달 불가 웨이포인트가 정확히 어디인지 표시된다.

**그리기는 `cv2`를 직접 쓰지 않는다 — `CV2Handler`를 쓴다. 선택이 아니라 강제다.**

`ZividQtApplication.__init__`은 `"cv2" in sys.modules`이면 예외를 던진다:

```
When using a ZividQtApplication you cannot directly load/import cv2.
It has conflicting versions of Qt on some platforms.
```

그리고 `main.py`는 앱을 만들기 **전에** `main_window` → `trace_panel` → `geometry/projection.py`까지
전부 import한다. 즉 `projection.py` 최상단에 `import cv2` 한 줄만 있어도 **GUI가 시작조차 못 한다.**
`CV2Handler`가 `__init__` 안에서 cv2를 지연 import하는 이유가 바로 이것이고, 그래서 그리는 시점에
인스턴스를 만든다 — 그때는 앱 생성 이후다. `_dev_test_projection`의 0번 시나리오가 이 불변식을
지킨다(모듈 import 후 `cv2` 미로드 → 그리기 후 로드).

점 반지름은 `CV2Handler.draw_circles`가 2px로 고정한 값을 그대로 쓴다.

> **한 번 틀렸던 경위** (같은 실수 방지용): 처음엔 리포의 `modules/zividsamples/gui/cv2_handler.py`
> 경로로 import했다가 `ModuleNotFoundError`가 났다. 리포 `modules/`는 실행되지 않는 낡은 평면 구조
> 사본이고, 실제로 import되는 것은 `C:/Zivid/zivid-python-samples/modules`에서 설치된 신구조
> 패키지(`gui/widgets/cv2_handler.py`)다. 그걸 "설치본에 cv2_handler가 없다"로 잘못 결론내고 `cv2`를
> 직접 import했다가, 위의 Qt 충돌 가드에 걸려 GUI가 죽었다. **참고할 코드는 설치본에서 읽는다.**

### 10. 여러 라인 = 여러 독립 경로 (합치지 않는다)

한 번의 press-drag-release가 stroke 하나이고, **stroke 하나가 로봇 경로 하나**다. 여러 줄을 그려도
이어붙이지 않는다.

`DrawableImageViewer`는 처음부터 `_strokes: List[Stroke]`로 stroke별 보관을 했고 화면에도 `moveTo`로
끊어 그렸다. 합쳐진 건 딱 한 곳, 옛 `get_line_points()`가 `extend`로 납작하게 만든 지점이었다.
그리고 그 결과는 단순히 "선이 이어져 보인다"가 아니라 **실제로 로봇이 그린 적 없는 구간을 훑는 것**이었다:

- `_resample_polyline`은 연속한 점 쌍을 순회하므로, stroke 1의 끝과 stroke 2의 시작 **사이에 등간격
  샘플을 깐다.** 그게 진짜 웨이포인트가 되어 뾰족한 팁이 표면에 닿은 채 지나간다.
- rotation-minimizing frame이 불연속을 가로질러 전파되어, 두 번째 선의 공구 자세가 첫 번째 선이
  어디서 끝났는지에 좌우된다.
- `_enforce_minimum_spacing`이 이음매를 넘어 비교해, stroke 시작점이 앞 stroke 끝점과 가깝다는 이유로
  버려질 수 있다.

`_dev_test_waypoint_builder`가 이걸 수치로 못박는다 — 떨어뜨려 그린 두 stroke 기준: 분리하면
**2경로 42 웨이포인트**, 납작하게 합치면 **64 웨이포인트이고 그중 14개가 어느 선에도 속하지 않는다.**

**경로마다 자기 접근/후퇴를 갖는다.** 한 선의 끝에서 다음 선의 시작으로 가려면 표면에서 떨어져야 한다.
실행 시퀀스는 `home → (접근 → 트레이싱 → 후퇴) × N → home`이고, `_dev_test_reachability`와 GUI
스모크 테스트가 **트레이싱 바로 뒤에 트레이싱이 오지 않는 것**을 확인한다.

**인덱스는 `(경로, 경로 내 웨이포인트)` 쌍**으로 다룬다. 전역 평탄 인덱스를 쓰면 경로 경계가 사라져서
애초에 고치려던 문제가 표현 층위에서 되살아난다. 도달성 판정·3D 강조·프로젝터 색이 모두 이 쌍을 쓴다.

**blend 반경은 경로 안에서만 판단한다.** 선과 선 사이 점프는 블렌드되는 세그먼트가 아니다(로봇이
후퇴했다가 접근한다). 그 간격이 blend 한도를 정하는 데 끼어들면 안 된다.

**한 줄만 그린 경우 동작은 이전과 완전히 동일하다** — 경로 1개짜리 목록일 뿐이고, 도달성 검사 라벨도
`Path 1` 접두사 없이 예전 그대로 나온다(접두사는 구분할 대상이 둘 이상일 때만 붙는다).

## 현재 상태

Connect / Calibration / Line Tracing 3탭 모두 동작 확인됨 (카메라 캡처, 라인 드로잉, 웨이포인트 생성
및 3D 프리뷰, 홈 포지션, E-stop/protective-stop 감지 및 재연결, 실제 로봇 트레이싱 실행까지 실기 검증
완료).

**eye-in-hand도 실기 검증 완료.** `build_waypoints`는 처음부터 `robot_pose` 인자로 eye-in-hand를
지원했지만 GUI에서 그 값을 넣을 방법이 없어 거부하고 있었는데, Calibration 탭의 **Robot Capture Pose**
섹션으로 연결했고 실제 로봇에서 팁이 그린 라인을 따라가는 것까지 확인했다(Pose Reference = Flange,
플랜지 기준 hand-eye).

캡처 pose는 **캡처 시점에 자동으로 기록**한다(`TracePanel.on_capture_clicked` →
`CalibrationPanel.record_capture_pose`). 캡처 중에는 로봇이 정지해 있으므로 그 순간의 현재 pose가 곧
캡처 pose이고, 사용자가 타이밍을 신경 쓸 필요가 없다. 기록된 값은 되돌아갈 위치로도 쓰인다
(**Move To Capture Pose** 버튼 → `RobotConnectionWidget.move_to_pose`). 캡처 pose는 파일로
불러오지 않는다 — 항상 로봇에서 읽는다. `Read From Robot`으로 수동 재읽기도 가능하지만, 캡처 당시
자세 그대로 서 있을 때만 유효하다.

**실행 전 도달성 검사는 실기 검증 완료** (2026-09-10, 의도적으로 도달 불가 구간을 포함시켜 확인).

**프로젝터 투사도 실기 검증 완료** (2026-09-10). 캡처 pose 기준 투사가 실제 표면의 의도한 자리에
떨어지는 것을 확인했다.

**캡처 pose는 로봇 연결 시 자동으로 미리 채워진다**(`CalibrationPanel._auto_read_capture_pose`).
캡처 전에도 pose 값이 보이므로 잘못된 Pose Reference가 TCP 오프셋만큼 어긋난 값으로 즉시 드러나고,
Move To Capture Pose도 캡처 없이 쓸 수 있다. **이미 캡처로 기록된 pose는 덮어쓰지 않는다** — 그걸
"지금 로봇이 서 있는 자리"로 갈아치우면 모든 웨이포인트가 그 차이만큼 밀리고, 아래 「3D 프리뷰로는
이 부류의 오류를 검증할 수 없다」대로 GUI 어디에서도 보이지 않는다.

같이 고친 잠재 결함: `_camera_to_base_transform`이 `eye_in_hand`를 확인하지 않아서, eye-in-hand로
작업한 뒤 eye-to-hand로 바꾸면 남아 있는 캡처 pose를 계속 곱했다. `build_waypoints`는 그걸 무시하므로
3D 프리뷰와 투사가 실제로 실행될 경로와 어긋나게 된다.

**남은 실기 확인 항목** (프로젝팅 관련): 검증 순서: (1) 캡처 pose에서
선이 실제 의도한 자리에 떨어지는지 → (2) 통과하면 로봇을 옮겨 보정 투사로 hand-eye 체인까지 확인.
확인할 것: 프로젝터 FOV 밖 점의 실제 반환 형태(NaN인지 범위 밖 좌표인지 — 둘 다 처리했지만 어느
쪽인지 안다), 보정 투사가 각도를 얼마나 바꿔도 쓸 만한지, 초점이 흐려지는 거리.

**아래 도달성 검사 관련 항목은 초기 구현 시점 기록이다.** 스텁 기반
`_dev_test_reachability.py`가 시퀀스 구성·판정·보고를 확인하지만, UR의 IK와 safety limit이 실제 셀에서
어떻게 답하는지는 실기로만 확인할 수 있다. 특히 확인할 것: (a) 정상 라인이 오탐 없이 통과하는지,
(b) 30° 조인트 점프 경고 임계값이 실제 경로에서 적절한지, (c) 웨이포인트 100개 기준 검사 소요 시간
(pose당 RTDE 왕복 3회).

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

eye-to-hand는 `robot_pose`를 아예 쓰지 않으므로(`camera_to_base = hand_eye`) 이 문제가 없다. 초기
실기 검증이 eye-to-hand뿐이었던 탓에 eye-in-hand를 붙일 때까지 드러나지 않았던 부분이다.

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

eye-in-hand 경로는 실기 검증됐다. 다만 **새 캘리브레이션이나 새 표면으로 처음 쓸 때는** 위의 노란 점
대조로 Pose Reference부터 확정한 뒤 저속으로 진행할 것 — 규약이 틀리면 경로가 TCP 오프셋만큼 통째로
밀리고, 그건 GUI 어디에서도 보이지 않는다.

### 실기로 측정한 TCP 동작

`setTcp([0]*6)` 전후로 `getActualTCPPose()`를 읽어 확정했다:

```
before : getActualTCPPose = [+0.2696, +0.1151, +0.3499, ...]   getTCPOffset = [-0.2267, +0.1732, +0.0988, ...]
TCP=0  : getActualTCPPose = [+0.2282, -0.1582, +0.4714, ...]   getTCPOffset = [0, 0, 0, 0, 0, 0]
차이   : 301.89 mm
```

- **TCP 오프셋은 하나뿐이고 command 측과 receive 측이 공유한다.** `setTcp`는 `moveL`/`movePath`/IK
  뿐 아니라 `getActualTCPPose()`가 보고하는 값까지 바꾼다. 두 측이 분리돼 있다는 가설은 틀렸다.
- **`setTcp`는 RTDE 세션을 넘어 영속된다.** GUI를 끊고 새 `RTDEControlInterface`를 만들어도 이전에
  쓴 값이 그대로 남아 있었다. 팬던트 Installation 값을 다시 입력해도 자동 반영되지 않고 `setTcp`로
  덮어써야 하며, **한 번 덮어쓰면 원래 Installation 값은 RTDE로 되읽을 수 없다.**
- 따라서 결정 #2는 유효하다: TCP가 팁으로 설정되어 있으면 `moveL`이 팁을 목표에 놓는다. 그리고
  `get_flange_pose()` = `get_pose() * get_tcp_offset().inv()`도 정확하다 — 같은 하나의 TCP를 쓰므로.
- **실기에서 관측된 "플랜지가 라인을 따라간다"의 원인은 활성 TCP가 팁이 아니었던 것**이다(당시
  `getTCPOffset()`이 팬던트 값과 달랐다). 코드가 아니라 설정 문제였다.

`Apply`는 이 영속적이고 되돌릴 수 없는 쓰기를 수행하므로 확인 문구에 그 사실을 명시한다.

### 폐기된 시도: `getForwardKinematics`로 플랜지 pose 구하기 (실기에서 실패)

command 측 TCP를 컨트롤러 값에 자동으로 맞추려고 `getForwardKinematics(q, [0]*6)`으로 플랜지 pose를
구하고, `flange^-1 · getActualTCPPose()`로 컨트롤러 TCP를 복원해 `setTcp()`로 심는 방식을 넣었다가
**되돌렸다**. 두 가지가 잘못됐다:

1. `connect()` 직후 동기 실행 → RTDE 왕복 3개가 UI 스레드를 막아 **Connect가 `Connecting...`에서
   멈췄다.**
2. 버튼으로 분리한 뒤에도 **`getForwardKinematics(q, [0]*6)`이 플랜지 pose를 돌려주지 않았다.**
   실기 관측값으로 역산하면 이 호출의 결과가 `getTCPOffset()` 값(`[0,0,186.59]`, Rz=−180)과 일치했고,
   그 결과 `setTcp()`에 쓰레기 값(`[−226.7, 173.2, 98.8]`)이 기록되어 로봇 상태가 더 꼬였다.

교훈: **검증하지 않은 ur_rtde API 동작을 가정한 채 `setTcp()`처럼 로봇 상태를 쓰는 코드를 만들지
않는다.** TCP 규약이 의심되면 먼저 측정으로 확정한다 — 예: `getActualTCPPose()`를 읽고 →
`setTcp([0]*6)` → 다시 읽어서, 값이 변하면 `getActualTCPPose()`가 `setTcp()`를 따르는 것이고 그 차이가
활성 오프셋이다. 값이 그대로면 command 측과 receive 측이 실제로 분리된 것이다.

## 향후 개선 여지

### 1. 그 외

- 속도/가속도/블렌드 값을 더 공격적으로(빠르게) 튜닝.
- 전체 시퀀스(홈 → 접근 → 트레이싱 → 후퇴 → 홈)를 다양한 실제 표면으로 반복 검증.
- 로봇/카메라 연결·재연결·에러 리셋의 예외 처리 보강. (이동 실패 원인 표시는 「주요 설계 결정 7」로
  구현 완료 — 남은 건 카메라 쪽과 재연결 흐름이다.)
- 도달성 검사의 조인트 점프 경고 임계값(기본 30°)을 실기 경로로 재조정. 지금 값은 근거 있는 추정이지
  측정값이 아니다.
- 도달성 검사가 통과한 웨이포인트만 남기고 실패 구간을 자동으로 잘라내는 옵션. 지금은 사용자가 라인을
  다시 그려야 한다 — 자동으로 자르면 조용히 다른 경로를 실행하게 되므로 일부러 넣지 않았다.
- 웨이포인트 리샘플링을 픽셀 호길이 대신 실제 3D 거리 기준으로. 현재는 근사여서 목표 5 mm인데
  실측 최소 간격이 3.4 mm까지 내려가고 절반 가까이가 `_enforce_minimum_spacing`에서 버려지는 경우가
  관측됐다.
