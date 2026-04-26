# UR_move_xyzRxRyRz_test — UR 로봇 이동 검증 및 핸드아이 캘리브레이션

UR5e 로봇과 Zivid 카메라를 이용한 두 가지 스크립트를 포함합니다.

1. **`universal_robots_move_test.py`** — RTDE 레지스터를 통한 로봇 이동 명령 검증
2. **`universal_robots_perform_hand_eye_calibration.py`** — 핸드아이 캘리브레이션 데이터셋 생성 및 캘리브레이션 수행

## 파일 구조

```
UR_move_xyzRxRyRz_test/
├── universal_robots_move_test.py                    # 로봇 이동 검증 스크립트
├── universal_robots_perform_hand_eye_calibration.py # 핸드아이 캘리브레이션 스크립트
├── universal_robots_communication_file.xml          # RTDE 통신 설정 파일
└── universal_robots_hand_eye_script.urp             # UR 로봇 프로그램 (티치 펜던트에 로드)
```

---

## 1. universal_robots_move_test.py — 로봇 이동 검증

RTDE double 레지스터(24~29)에 목표 TCP 포즈(x, y, z, rx, ry, rz)를 전송하고, 이동 전후 실제 TCP 포즈를 읽어 이동이 올바르게 수행되었는지 확인합니다.

### 실행

```bash
python universal_robots_move_test.py --eih --ip <ROBOT_IP>
# 또는
python universal_robots_move_test.py --eth --ip <ROBOT_IP>
```

### 동작 흐름

```
RTDE 연결 → 시작 TCP 포즈 출력
    ↓
목표 포즈 레지스터 전송 + camera_ready 신호
    ↓
로봇 이동 완료 대기 (output_int_register_24 == -1)
    ↓
이동 후 TCP 포즈 출력 → 연결 해제
```

### CLI 인수

| 인수 | 설명 |
|------|------|
| `--eih` / `--eye-in-hand` | Eye-in-Hand 모드 선택 |
| `--eth` / `--eye-to-hand` | Eye-to-Hand 모드 선택 |
| `--ip` | 로봇 IP 주소 |

---

## 2. universal_robots_perform_hand_eye_calibration.py — 핸드아이 캘리브레이션

로봇이 미리 정의된 포즈들을 순서대로 이동하며 Zivid 카메라로 체커보드를 촬영하고, 수집된 데이터로 핸드아이 캘리브레이션을 수행합니다.

### 실행

```bash
python universal_robots_perform_hand_eye_calibration.py --eih --ip <ROBOT_IP>
# 또는
python universal_robots_perform_hand_eye_calibration.py --eth --ip <ROBOT_IP>
```

### 동작 흐름

```
RTDE 연결 → Zivid 카메라 연결 (capture assistant로 설정 자동 추천)
    ↓
로봇이 각 포즈로 이동 → 준비 신호(output_bit_register_64) 수신
    ↓
ZDF 캡처 + TCP 포즈 읽기
    ↓
체커보드 특징점 검출 (zivid.calibration.detect_feature_points)
    ↓
img01.zdf, pos01.yaml … 저장 (datasets/YYYY-MM-DD_HH-MM-SS/)
    ↓
모든 포즈 완료 (output_int_register_24 == -1)
    ↓
Eye-in-Hand: zivid.calibration.calibrate_eye_in_hand(inputs)
Eye-to-Hand: zivid.calibration.calibrate_eye_to_hand(inputs)
    ↓
handEyeTransform.yaml + residuals.yaml 저장
```

### 출력 파일

| 파일 | 내용 |
|------|------|
| `datasets/<timestamp>/img01.zdf` … | 각 포즈에서 촬영한 ZDF 파일 |
| `datasets/<timestamp>/pos01.yaml` … | 각 포즈의 4×4 TCP 변환 행렬 |
| `datasets/<timestamp>/handEyeTransform.yaml` | 핸드아이 캘리브레이션 결과 (4×4 변환 행렬) |
| `datasets/<timestamp>/residuals.yaml` | 포즈별 잔차 (rotation °, translation mm) |

### CLI 인수

| 인수 | 설명 |
|------|------|
| `--eih` / `--eye-in-hand` | Eye-in-Hand 캘리브레이션 |
| `--eth` / `--eye-to-hand` | Eye-to-Hand 캘리브레이션 |
| `--ip` | 로봇 IP 주소 |

---

## 사전 준비

| 항목 | 내용 |
|------|------|
| 로봇 프로그램 | `universal_robots_hand_eye_script.urp`을 UR 티치 펜던트에 로드하고 각 포즈를 씬에 맞게 수정 |
| 통신 설정 파일 | `universal_robots_communication_file.xml`이 스크립트와 같은 디렉터리에 있어야 함 |
| 체커보드 | Zivid 캘리브레이션 보드를 카메라 시야 내에 고정 배치 |
| 네트워크 | PC와 로봇이 같은 네트워크에 연결되어 있어야 함 |

## 주요 의존성

| 패키지 | 용도 |
|--------|------|
| `zivid` | Zivid SDK — 카메라 캡처, 특징점 검출, 핸드아이 캘리브레이션 |
| `rtde` | UR RTDE 통신 |
| `scipy` | 회전 벡터 → 변환 행렬 변환 |
| `opencv-python` | 잔차 YAML 저장 |

## 참고

- Zivid 공식 샘플: [UR5 + Python Hand-Eye Calibration](https://support.zivid.com/latest/academy/applications/hand-eye/ur5-robot-%2B-python-generate-dataset-and-perform-hand-eye-calibration.html)
- RTDE 프로토콜: 포트 30004 사용
