# UR_move_xyzRxRyRz_test — UR 로봇 TCP 이동 검증

UR5e 로봇의 현재 TCP 포즈를 읽고, X/Y/Z 축으로 각 +10 mm 이동시켜 실제 이동 결과를 검증하는 스크립트입니다.

## 파일 구조

```
UR_move_xyzRxRyRz_test/
├── universal_robots_move_test.py           # TCP 이동 검증 스크립트
├── universal_robots_communication_file.xml # RTDE 통신 레지스터 설정
└── ur_comm_test.urp                        # UR 로봇 프로그램 (티치 펜던트에 로드)
```

---

## `universal_robots_move_test.py` — TCP 이동 검증

RTDE를 통해 현재 TCP 포즈를 읽고, 목표 포즈(현재 + 10 mm in X, Y, Z)를 double 레지스터(24~29)에 전송합니다. 이동 완료 후 실제 포즈를 읽어 예상 이동량과 비교합니다.

### 실행

```bash
python universal_robots_move_test.py --ip <ROBOT_IP>
```

### 동작 흐름

```
RTDE 연결 (Port 30004)
  → 현재 TCP 포즈 읽기 및 출력
  → 목표 포즈 = 현재 + [+10mm, +10mm, +10mm, 0, 0, 0] 계산
  → 목표 포즈를 double 레지스터 24~29에 전송
  → pc_ready 신호 ON (input_bit_register_65)
  → move_status == -1 대기 (output_int_register_24)
  → 이동 후 TCP 포즈 읽기 및 출력
  → 예상 이동량 vs 실제 이동량 비교 출력
  → move_confirmed 신호 ON/OFF (input_bit_register_64)
  → 연결 해제
```

### RTDE 레지스터 매핑

| 방향 | 레지스터 | 이름 | 설명 |
|------|----------|------|------|
| PC → Robot | input_bit_register_64 | move_confirmed | PC가 이동 완료를 확인한 신호 |
| PC → Robot | input_bit_register_65 | pc_ready | PC가 목표 포즈를 설정 완료한 신호 |
| PC → Robot | input_double_register_24~29 | target x,y,z,rx,ry,rz | 목표 TCP 포즈 (m, rad) |
| Robot → PC | output_int_register_24 | move_status | 이동 상태 (-1: 완료) |
| Robot → PC | output_bit_register_64 | robot_ready | 로봇 이동 준비 완료 신호 |

### 출력 예시

```
RTDE connection established.

Current TCP pose:
  X=400.123 mm   Y=-500.010 mm   Z=440.111 mm
  Rx=3.000000 rad   Ry=0.066600 rad   Rz=-0.234000 rad

Target TCP pose (current + 10 mm in X, Y, Z):
  X=410.123 mm   Y=-490.010 mm   Z=450.111 mm
  Rx=3.000000 rad   Ry=0.066600 rad   Rz=-0.234000 rad

Waiting for robot movement to complete...

Final TCP pose:
  X=410.121 mm   Y=-490.008 mm   Z=450.109 mm
  Rx=3.000001 rad   Ry=0.066599 rad   Rz=-0.234001 rad

Movement verification:
  Expected delta XYZ : [10. 10. 10.] mm
  Actual delta   XYZ : [9.998 10.002 9.998] mm
  Error          XYZ : [-0.002  0.002 -0.002] mm

RTDE connection closed.
```

---

## 사전 준비

| 항목 | 내용 |
|------|------|
| 로봇 프로그램 | `ur_comm_test.urp`을 UR 티치 펜던트에 로드하고 실행 |
| 통신 설정 파일 | `universal_robots_communication_file.xml`이 스크립트와 같은 디렉터리에 있어야 함 |
| 네트워크 | PC와 로봇이 같은 네트워크에 연결되어 있어야 함 |

## 주요 의존성

| 패키지 | 용도 |
|--------|------|
| `rtde` | UR RTDE 통신 (포트 30004) |
| `numpy` | 포즈 연산 및 이동량 비교 |

## 참고

- RTDE 프로토콜: [UR RTDE Guide](https://www.universal-robots.com/how-tos-and-faqs/how-to/ur-how-tos/real-time-data-exchange-rtde-guide-22229/)
