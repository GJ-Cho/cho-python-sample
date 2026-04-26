# AGENT.md — 작업 규칙

## 프로젝트 도메인

이 프로젝트는 Zivid 3D camera, Zivid SDK, Zivid Studio, Zivid API, point cloud, capture settings, calibration, robot vision, ROS 연동, Python/C++ 샘플 코드와 관련된 작업을 다룬다.

---

## 우선 참고 자료

### 1. Zivid 공식 Knowledge Base
- https://support.zivid.com/ko/latest/index.html
- User Guide, Software Installation, Studio Guide, Quick Capture Tutorial, API Reference, Samples, Troubleshooting, Camera Settings, General 3D Topics를 우선 확인한다.
- 한국어 페이지가 어색하거나 부족하면 같은 문서의 영어 원문도 함께 참고한다.

### 2. Zivid 공식 GitHub
- https://github.com/zivid
- 코드 관련 질문은 공식 repo의 README, examples, sample code, package configuration, issue/README 설명을 우선 확인한다.
- 언어별로 필요한 경우 다음 repo를 우선 고려한다.
  - Python: `zivid-python`, `zivid-python-samples`
  - C++: `zivid-cpp-samples`
  - C#: `zivid-csharp-samples`
  - ROS: `zivid-ros`
  - HALCON: `zivid-halcon-samples`
  - MATLAB: `zivid-matlab-samples`
  - Isaac Sim: `zivid-isaac-sim`

---

## 답변 원칙

- Zivid 관련 기술 질문은 일반적인 3D vision 지식보다 Zivid 공식 문서와 공식 GitHub 내용을 우선한다.
- 최신성, 설치 방법, API 사용법, 지원 OS, SDK 버전, 설정값, 샘플 코드, troubleshooting은 반드시 공식 문서 또는 공식 GitHub 기준으로 확인해서 답한다.
- 확인한 내용과 추론한 내용을 명확히 구분한다.
- 문서나 GitHub 내용을 확인하지 못했으면 "확인하지 못했다"고 먼저 말하고, 일반적인 설명은 별도로 구분해서 제공한다.
- 가능하면 참고한 문서 제목, GitHub repo명, 파일명, 함수명, 설정 항목명을 함께 알려준다.
- 코드 예시는 Zivid 공식 샘플 스타일을 우선 따르고, 임의로 API 이름을 만들어내지 않는다.
- 사용자가 사용하는 환경이 중요할 경우 OS, SDK 버전, 카메라 모델, 언어(Python/C++/C#/ROS 등), 오류 메시지를 확인 대상으로 삼는다.
- 답변은 기본적으로 한국어로 하되, API 이름, 함수명, 클래스명, 파일명은 원문 그대로 유지한다.
- 해결 절차가 필요한 경우 "원인 후보 → 확인 방법 → 해결 방법 → 관련 문서/샘플" 순서로 정리한다.
