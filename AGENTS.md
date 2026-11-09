# AGENTS.md — cho-python-sample

> 전역 지침(`~/.claude/CLAUDE.md`)을 자동 상속합니다. 여기엔 이 프로젝트 고유 규칙만 씁니다.
> Claude Code는 `CLAUDE.md`(→ 이 파일 import)를 통해, 다른 에이전트는 이 파일을 직접 읽습니다.

## 프로젝트 도메인

Zivid 3D camera, Zivid SDK, Zivid Studio, Zivid API, point cloud, capture settings,
calibration, robot vision, ROS 연동, Python/C++ 샘플 코드.

## 우선 참고 자료

### 1. Zivid 공식 Knowledge Base

- https://support.zivid.com/ko/latest/index.html
- User Guide, Software Installation, Studio Guide, Quick Capture Tutorial, API Reference,
  Samples, Troubleshooting, Camera Settings, General 3D Topics를 우선 확인한다.
- 한국어 페이지가 어색하거나 부족하면 같은 문서의 영어 원문도 함께 참고한다.

### 2. Zivid 공식 GitHub

- https://github.com/zivid
- 코드 관련 질문은 공식 repo의 README, examples, sample code, package configuration,
  issue 설명을 우선 확인한다.
- 언어별 우선 repo:
  - Python: `zivid-python`, `zivid-python-samples`
  - C++: `zivid-cpp-samples`
  - C#: `zivid-csharp-samples`
  - ROS: `zivid-ros`
  - HALCON: `zivid-halcon-samples`
  - MATLAB: `zivid-matlab-samples`
  - Isaac Sim: `zivid-isaac-sim`

## 작업 규칙

- Zivid 관련 기술 질문은 일반적인 3D vision 지식보다 공식 문서와 공식 GitHub를 우선한다.
- 최신성, 설치 방법, API 사용법, 지원 OS, SDK 버전, 설정값, 샘플 코드, troubleshooting은
  반드시 공식 문서 또는 공식 GitHub 기준으로 확인해서 답한다.
- 참고한 문서 제목, GitHub repo명, 파일명, 함수명, 설정 항목명을 함께 알려준다.
- 코드 예시는 Zivid 공식 샘플 스타일을 따르고, API 이름을 임의로 만들어내지 않는다.
- 환경이 중요한 경우 OS, SDK 버전, 카메라 모델, 언어(Python/C++/C#/ROS), 오류 메시지를
  확인 대상으로 삼는다.
