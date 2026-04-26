# UR_communication_test — UR 로봇 RTDE 통신 테스트

UR 로봇과 RTDE(Real-Time Data Exchange) 통신을 확인하고, 6개 관절 각도를 실시간으로 모니터링·로깅하는 테스트 스크립트입니다.

## 파일 구조

```
UR_communication_test/
├── universal_robots_comm_test.py        # 메인 스크립트
└── universal_robots_communication_file.xml  # RTDE 통신 설정 파일
```

## 실행 방법

```bash
python universal_robots_comm_test.py
```

로봇 IP는 스크립트 상단 `IP_ROBOT` 상수에서 직접 수정합니다.

```python
IP_ROBOT = "192.168.56.101"  # 실제 로봇 IP로 변경
```

## 동작 흐름

```
RTDE 연결 (포트 30004)
    ↓
로봇 준비 신호 대기 (output_bit_register_64)
    ↓
200Hz로 관절 상태 수신 루프
    ├── 6개 관절 각도 실시간 그래프 업데이트 (matplotlib)
    └── 시간·루프 카운트·관절 각도(°) 누적
    ↓
's' 키 입력 → 로봇에 종료 신호 송신 → 연결 해제
    ↓
ur_rtde_joint_log.csv 저장
```

## 주요 기능

- RTDE 포트(30004)로 UR 로봇 연결 및 프로토콜 버전 확인
- 6개 관절의 실제 각도(`actual_q`)를 도(°) 단위로 200Hz 수신
- matplotlib 실시간 그래프 (6개 서브플롯, 공유 x축, 루프 카운트 표시)
- 측정 데이터 → `ur_rtde_joint_log.csv` (time, loop_count, joint_1 ~ joint_6)
- `s` 키로 안전 종료; `Ctrl+C` 인터럽트도 처리

## 사전 준비

| 항목 | 내용 |
|------|------|
| 로봇 프로그램 | `universal_robots_hand_eye_script.urp`을 UR 티치 펜던트에 로드 |
| 통신 설정 파일 | `universal_robots_communication_file.xml`이 스크립트와 같은 디렉터리에 있어야 함 |
| 네트워크 | PC와 로봇이 같은 네트워크에 연결되어 있어야 함 |

## 출력

- 터미널: 시작 TCP 포즈, 루프 시작/종료 메시지
- 화면: 실시간 관절 각도 그래프 (루프 카운트 표시)
- 파일: `ur_rtde_joint_log.csv`

## 주요 의존성

| 패키지 | 용도 |
|--------|------|
| `rtde` | UR RTDE 통신 |
| `matplotlib` | 실시간 그래프 |
| `pandas` | CSV 저장 |
| `keyboard` | 's' 키 감지 |
| `numpy` | 각도 변환 (rad → deg) |
