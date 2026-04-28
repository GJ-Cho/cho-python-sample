# Rainbow Robotics RB Series — Communication Test

rbpodo 라이브러리를 사용해 Rainbow Robotics RB 시리즈 로봇과 통신하는 PyQt5 GUI 샘플입니다.

## 구성 (Phase)

| Phase | 내용 | 상태 |
|-------|------|------|
| Phase 1 | 연결 + 실시간 포즈/관절 모니터링 | ✅ 완료 |
| Phase 2 | Digital I/O 신호 교환 + 타겟 포즈 전송 | 예정 |

## Phase 1 — 실행 방법

```bash
pip install rbpodo
python rainbow_comm_test.py
```

Robot IP는 GUI 상단 입력창에서 직접 수정합니다. (기본값: `192.168.0.1`)

## Phase 1 — 주요 기능

- **연결**: `rb.CobotData(ip)` — Port 5001 데이터 채널, 백그라운드 스레드 50Hz 폴링
- **TCP Pose 실시간 표시**: X Y Z Rx Ry Rz
  - 단위 토글: `mm + deg` (native) ↔ `mm + rad`
  - 단위 전환 시 즉시 화면 반영
- **Joint Angles 실시간 표시**: J1–J6 (항상 degrees)
- **Robot State 표시**: IDLE (초록) / MOVING (노랑) / UNKNOWN (회색)
- **실제 Poll Rate** 표시 (목표 50Hz)
- **Log 패널**: 타임스탬프 포함 이벤트 기록

## 포트 구조

| 포트 | 클래스 | 용도 |
|------|--------|------|
| 5001 | `rb.CobotData` | 로봇 상태 데이터 수신 (Phase 1) |
| 5000 | `rb.Cobot` | 명령 전송 — move, set_dout 등 (Phase 2) |

## 데이터 채널 — `state.sdata` 구조

| 필드 | 내용 | 단위 |
|------|------|------|
| `tcp_pos[0:6]` | 현재 TCP 포즈 [x,y,z,rx,ry,rz] | mm, deg |
| `jnt_cur[0:6]` | 현재 관절 각도 [J1–J6] | deg |
| `robot_state` | RobotState enum (Idle/Moving/Unknown) | — |
| `din[0:15]` | Digital Input 상태 | — |
| `dout[0:15]` | Digital Output 상태 | — |

## Phase 2 예정 기능

- Digital I/O 신호 패널
  - PC → Robot: `capture_complete`, `ready_to_capture` (set_box_dout)
  - Robot → PC: `move_complete`, `at_capture_pos` (din)
- 타겟 포즈 입력 → `robot.move_l(rc, pose, speed, accel)` 직접 전송
- `rb.Cobot(ip)` Port 5000 명령 채널 활성화

## 의존성

| 패키지 | 용도 |
|--------|------|
| `rbpodo` | Rainbow Robotics RB 시리즈 통신 |
| `PyQt5` | GUI |
| `numpy` | rad 변환 (`np.radians`) |
