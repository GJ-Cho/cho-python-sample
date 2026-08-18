# AGENTS.md — Line Tracing GUI

> 상위 `cho-python-sample/AGENT.md`(Zivid 규칙)와 전역 지침을 상속합니다. 여기엔 이 폴더 고유 규칙만 씁니다.

UR3e(뾰족한 그리퍼) + Zivid2+ MR130(eye-to-hand, 고정 거치)로, 캡처한 2D 이미지 위에 사용자가 그린 라인을
실제 표면에 수직으로 접촉하며 따라가게 하는 PyQt5 GUI다.

## 시작 전 반드시 할 일

1. **`PLAN.md`를 먼저 읽는다.** 설계 결정과 그 근거(특히 폐기된 대안: raw RTDE, 소프트웨어 TCP 오프셋,
   진행방향-고정 X축)가 정리되어 있다. 같은 문제를 다시 그 방향으로 풀지 않도록 확인할 것.
2. `README.md`로 실행 방법·필수 사전조건(로봇 Remote Control 모드 등)을 확인한다.

## 실행 환경

- 리포 루트에서 `pip install -r requirements.txt` (이 프로젝트가 추가한 항목: `ur_rtde`, `pyopengl`).
- 로봇 제어는 공식 `ur_rtde` 라이브러리(`rtde_control`/`rtde_receive`)만 쓴다. raw RTDE 레지스터 방식은
  폐기됨 — 이유는 `PLAN.md` 참조.
- 실행: `python main.py` (이 폴더 안에서) 또는 `python src/project/line_tracing_gui/main.py`
  (리포 루트에서). `python -m line_tracing_gui.main`도 동작 (`sys.path`에 이 폴더의 부모를 자동으로 추가함).
- 오프라인 개발/회귀 테스트: `_dev_test_*.py` (카메라/로봇 없이 `sample/sample_MR130.zdf` 등 리포 루트
  샘플 데이터로 동작). 예: `python -m line_tracing_gui._dev_test_waypoint_builder` (src/project/에서 실행).

## 작업 규칙

- Zivid/`zividsamples` API 이름을 추측하지 않는다. 실제 설치된 site-packages 버전(`modules/zividsamples`)을
  확인하거나 불확실하면 사용자에게 묻는다.
- TCP 오프셋(그리퍼 팁 위치)은 **로봇 컨트롤러**(PolyScope Installation → TCP Configuration)에서만
  설정한다. `waypoint_builder.py`나 다른 소프트웨어 코드에서 다시 곱하면 오프셋이 중복 적용된다.
- 웨이포인트 방향(회전) 계산은 그리퍼가 **툴 Z축 둘레로 회전 대칭**이라는 전제에 의존한다
  (`waypoint_builder._build_rotation_minimizing_frames`). 이 전제가 깨지는 그리퍼(비대칭 팁 등)로
  바뀌면 이 로직 전체를 재검토해야 한다.
- 하드코딩 금지. 속도/가속도/블렌드/간격 등은 GUI 입력값이나 상수로 명시하고, `config.py`(QSettings)로
  영속화가 필요한 값만 저장한다.
- 주석과 문서는 한국어, 코드 식별자와 API 이름은 영어.

## 안전

- 순수 위치 제어이며 force/torque 기반 접촉 검증은 없다 — 사전에 계산된 좌표로 이동하는 것뿐이라
  캘리브레이션 오차나 표면 굴곡이 있으면 그리퍼가 과도하게 파고들거나 허공에서 멈출 수 있다.
- 새 캘리브레이션이나 새 표면으로 처음 테스트할 때는 반드시 저속 + 여유 있는 접근 오프셋으로, 사람이
  즉시 비상정지 가능한 거리에서 진행한다.
