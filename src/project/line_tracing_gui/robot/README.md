# RobotControlURRTDE — 로봇(UR3e) 쪽 사전 설정

`RobotControlURRTDE`(`robot_control_ur_rtde.py`)는 공식 `ur_rtde` 라이브러리
(`rtde_control.RTDEControlInterface`, `rtde_receive.RTDEReceiveInterface`)로 Python에서
직접 `moveL`/`moveJ`/`movePath`를 호출합니다. **티치펜던트에 별도 프로그램을 올리거나
재생(▶) 상태로 둘 필요가 없습니다.**

## 필요한 사전 설정: Remote Control 모드

1. 티치펜던트에서 화면 우측 상단 아이콘(또는 Installation → General → Remote Control) →
   **Remote Control** 활성화.
2. Remote Control이 꺼져 있으면 `RTDEControlInterface(ip)`/`RTDEReceiveInterface(ip)` 생성자가
   연결에 실패합니다(예외 발생) — GUI에서 "Failed to connect" 메시지로 뜹니다.

## 이전 방식(raw RTDE 레지스터 + URScript)과의 차이

이전에는 `robot_program/line_tracing.script`를 티치펜던트에 직접 올려서 재생해두고, PC는
RTDE 레지스터로 목표 pose만 주고받는 방식이었습니다. 이제는 그 로봇 쪽 프로그램이 필요 없고,
Python에서 `ur_rtde`로 바로 명령을 보냅니다 — 더 간단하고, 여러 웨이포인트를 한 번에
블렌딩(`move_path`)해서 보낼 수 있어 라인 트레이싱처럼 연속 이동이 필요한 경우에 더 적합합니다.

## 안전 유의사항

- 그리퍼가 뾰족하므로 **첫 실행은 반드시 저속**(`speed=0.02, acceleration=0.1` 정도)으로, 사람이
  로봇 옆에서 즉시 비상정지 가능한 상태로 진행하세요.
- 이 방식은 순수 위치 제어입니다 — force/torque로 접촉을 감지하지 않습니다. 캘리브레이션 오차나
  표면 굴곡이 있으면 그리퍼가 표면을 파고들거나 허공에서 멈출 수 있습니다.
- TCP 오프셋(그리퍼 팁 위치)은 로봇 컨트롤러의 `set_tcp`(PolyScope Installation → TCP Configuration)에서
  관리합니다 — 미터 단위로 입력해야 합니다(밀리미터를 그대로 넣으면 안전 한계 오류가 납니다).
