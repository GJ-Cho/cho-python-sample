# Line Tracing GUI

UR3e(뾰족한 그리퍼) + Zivid 3D 카메라 조합으로, 캡처한 2D 이미지 위에 그린 라인을
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
  eye-in-hand(플랜지 장착)를 모두 지원하며, 둘 다 실기 검증했습니다.

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
  필요가 없습니다.
  **로봇이 연결되면 현재 pose가 미리 채워집니다** (출처란에 `(current robot pose - no capture taken
  yet)`으로 표시). 캡처 전에도 값이 보이므로 Pose Reference가 잘못됐으면 TCP 오프셋만큼 어긋난 값으로
  바로 드러나고, **Move To Capture Pose**도 캡처 없이 쓸 수 있습니다. 이미 캡처로 기록된 pose는
  **덮어쓰지 않습니다** — 다시 읽으려면 **Read From Robot**을 쓰세요. Pose Reference를 바꾸면 새 기준으로
  다시 읽어서, 값이 TCP 오프셋만큼 움직이는 것으로 그 라디오 버튼의 의미가 눈에 보입니다. 기록된 pose로 **Move To Capture Pose** 버튼을 눌러 로봇을 되돌릴 수 있고, 필요하면
  **Read From Robot**으로 현재 pose를 다시 읽을 수도 있습니다(캡처 당시 자세 그대로일 때만 유효).
  캡처 pose는 파일로 불러오지 않습니다 — 항상 로봇에서 읽습니다.
  **Pose Reference**는 hand-eye 캘리브레이션을 무엇을 기준으로 했는지 고르는 항목입니다. 로봇은 TCP가
  적용된 pose(`getActualTCPPose`)만 알려주므로, 어느 프레임으로 캡처 pose를 만들지가 여기서 정해집니다:

  ```
  Flange : hand_eye = flange_T_camera  ->  capture pose = base_T_flange  (활성 TCP 오프셋을 빼줌)
  TCP    : hand_eye = tcp_T_camera     ->  capture pose = base_T_tcp     (로봇 보고값 그대로)
  ```

  `base_T_touch = capture pose · hand_eye · camera_T_point` 에서 중간 프레임이 소거되어야 하므로,
  hand_eye YAML의 프레임과 반드시 일치해야 합니다. Zivid 샘플은 플랜지(6축 끝) 기준으로 캘리브레이션
  하므로 기본값은 `Flange`입니다. 잘못 고르면 모든 웨이포인트가 TCP 오프셋만큼 밀립니다.
- **Line Tracing**: 캡처 → 2D 이미지 위에 라인 드로잉(**여러 줄 가능**) → 웨이포인트 생성(간격 조절 가능, 3D 프리뷰로 확인) →
  접근/후퇴 거리·속도·가속도·블렌드 설정 → **Check Reachability**(선택) → 실행(홈 → 접근 → 라인
  트레이싱 → 후퇴 → 홈).

  **여러 줄을 그리면 각각 별개의 경로가 됩니다.** 한 번의 드래그가 한 줄이고, 줄과 줄은 **절대 이어지지
  않습니다** — 로봇은 한 줄을 마치면 표면에서 떨어졌다가 다음 줄로 이동해 다시 접근합니다. 실행 순서는
  그린 순서이고, 시퀀스는 `홈 → [접근 → 트레이싱 → 후퇴] × N → 홈`입니다. `Undo`는 마지막으로 그린
  줄 하나를 지웁니다.

  > 줄을 이어붙이면 두 줄 사이 빈 공간에도 웨이포인트가 생겨서, **그린 적 없는 구간을 뾰족한 팁이 표면에
  > 닿은 채 지나갑니다.** 그래서 분리는 편의 기능이 아니라 안전 문제입니다.

  **Check Reachability**는 로봇을 움직이지 않고, 시퀀스의 모든 pose(홈·접근·각 웨이포인트·후퇴)를
  컨트롤러에게 직접 물어봅니다 — 팔이 닿는지(IK 해 존재), 설정된 safety plane 안인지, 조인트 범위
  안인지. 3D 뷰에서 **도달 불가는 빨강, 경고는 주황**으로 표시되므로 라인을 어디서 다시 그려야 할지
  바로 보입니다.

  **Project**는 카메라 내장 프로젝터로 그 라인을 **실제 표면에 빛으로 그립니다.** 도달성 검사를
  이미 돌렸다면 같은 색(초록/주황/빨강)이 실물 위에 그대로 올라갑니다. 토글 버튼이라 다시 누르면
  꺼지고, 캡처·웨이포인트 재생성·실행 시작 시에는 자동으로 꺼집니다(같은 카메라에서 capture가
  시작되면 SDK가 투사를 중단시키기 때문입니다).

  이게 **`camera_to_base` 체인 오류를 잡을 수 있는 유일한 화면**입니다. 3D 프리뷰는 웨이포인트를 만들 때
  쓴 변환의 역변환으로 되돌려 그리기 때문에 오류가 상쇄되어 멀쩡하게 보입니다.

  eye-in-hand에서 캡처 이후 로봇이 움직였으면 선택지를 묻습니다:

  | 선택 | 하는 일 | 검증되는 것 |
  |---|---|---|
  | **Move To Capture Pose** | 캡처 자리로 되돌린 뒤 투사 | 픽셀→3D 대응만 (진단 명확) |
  | **Project From Here** | 현재 자리에 맞게 다시 매핑해 투사 | 픽셀→3D **+ hand-eye 체인** |

  권장 순서는 **캡처 pose에서 먼저 통과시키고, 그 다음 로봇을 옮겨 보정 투사를 확인**하는 것입니다.
  캡처 pose에서 이미 어긋나 있으면 보정 투사는 아무것도 말해주지 못합니다. 보정 투사는 가림(occlusion)을
  처리하지 않고 프로젝터 초점도 고정이므로, **각도를 조금 바꾸는 정도**에서만 신뢰할 수 있습니다.

  **Execute도 도달성 검사를 자동으로 먼저 돌립니다.** 도달 불가가 하나라도 있으면 실행을 시작하지 않습니다.
  `move_path` 중간에 실패하면 로봇이 라인 한가운데에서 팁을 표면 가까이 둔 채 멈추기 때문입니다.
  경고(직선 구간에서 팔 자세가 바뀔 것으로 의심되는 지점)만 있는 경우에는 확인 대화상자에 함께 표시되고,
  진행 여부는 사용자가 정합니다. 별도 버튼이 있는 이유는 라인을 실제로 실행하기 전에 여러 번 시험해 볼 수
  있게 하기 위함입니다.

## 오프라인 개발 테스트

카메라/로봇 없이 리포 루트의 `sample/sample_MR130.zdf`, `sample_MR130_2d.png`로 동작을 확인할 수 있는
스크립트입니다 (`src/project/`에서 실행):

```
python -m line_tracing_gui._dev_test_waypoint_builder
python -m line_tracing_gui._dev_test_pointcloud_preview
python -m line_tracing_gui._dev_test_drawable_viewer
python -m line_tracing_gui._dev_test_eye_to_hand_regression
python -m line_tracing_gui._dev_test_reachability
python -m line_tracing_gui._dev_test_projection
```

마지막 것은 **eye-to-hand 회귀 가드**입니다. eye-to-hand가 먼저 실기 검증됐고 eye-in-hand를 그 위에
얹었기 때문에, eye-in-hand 쪽 수정이 eye-to-hand 웨이포인트를 조용히 움직일 수 있습니다. 이 테스트가
`camera_to_base`가 정확히 hand-eye 변환뿐임을 불변식으로 확인하고, 샘플 점군 기준 golden 값과도
대조합니다. eye-in-hand를 손댄 뒤에는 한 번 돌려보세요.

`_dev_test_reachability`는 **도달성 검사(pre-flight)** 를 로봇 없이 검증합니다. 검사 코드는 로봇을
조회 메서드 5개로만 접근하므로, 그 5개만 구현한 스텁을 넣어 "어떤 스텝을 어떤 순서로 검사하는지,
판정과 보고가 맞는지"를 확인합니다. 스텁에는 이동 메서드가 아예 없으므로, 검사 코드가 로봇을 움직이려
하면 이 테스트가 깨집니다. 단, UR의 실제 IK·safety limit이 실제 셀에서 어떻게 답하는지는 여기서 알 수
없습니다 — 그건 실기로만 확인됩니다.

`_dev_test_projection`은 **프로젝터 투사 이미지 생성**을 카메라 없이 검증합니다. 3D→프로젝터 픽셀
매핑을 인자로 주입할 수 있게 만들었으므로, 그 아래 단계(NaN·FOV 밖 걸러내기, 세그먼트 색 결정,
투사 불가 지점에서 **선을 잇지 않고 끊는지**)를 전부 확인합니다. 매핑 자체가 빛이 실제로 떨어지는
자리와 맞는지는 여기서 알 수 없습니다.

## 안전

이 GUI는 **순수 위치 제어**입니다 — force/torque로 실제 접촉을 감지하지 않고, 사전에 계산된 좌표로
이동할 뿐입니다. 캘리브레이션 오차나 표면 굴곡이 있으면 그리퍼가 과도하게 파고들거나 허공에서 멈출 수
있습니다. 새 캘리브레이션·새 표면으로 처음 테스트할 때는 저속 + 충분한 접근 오프셋으로, 사람이 즉시
비상정지 가능한 거리에서 진행하세요.

설계 결정과 그 근거는 `PLAN.md`를 참고하세요.
