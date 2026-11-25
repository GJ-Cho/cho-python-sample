# AGENTS.md — cho-python-sample

> 전역 지침(`~/.claude/CLAUDE.md`)을 자동 상속합니다. 여기엔 이 프로젝트 고유 규칙만 씁니다.
> Claude Code는 `CLAUDE.md`(→ 이 파일 import)를 통해, 다른 에이전트는 이 파일을 직접 읽습니다.

## 프로젝트 도메인

Zivid 3D camera, Zivid SDK, Zivid Studio, Zivid API, point cloud, capture settings,
calibration, robot vision, ROS 연동, Python/C++ 샘플 코드.

## Zivid 코딩 워크플로 (필수)

Zivid API를 쓰는 코드를 작성·수정하기 전에 아래 순서를 따른다.

1. **`ZividAPI.md`를 먼저 읽는다.** 버전 기준, 핵심 API 패턴, 작업별 샘플 위치, 자주 틀리는 점이 정리되어 있다.
2. **설치된 버전을 확인한다.** `python -c "import zivid; print(zivid.__version__, zivid.SDKVersion.full)"`
   설치 버전이 코드의 기준이다. GitHub master가 더 최신이면 해당 버전 태그의 샘플을 본다.
3. **Knowledge Base로 개념·설정을 확인한다.** (아래 "우선 참고 자료" 1번)
4. **코딩 직전에 공식 샘플에서 실제 사용법을 확인한다.**
   먼저 로컬 클론(`C:\Zivid\zivid-python-samples` 등)을 grep하고, 로컬에 없을 때만 https://github.com/zivid 를 본다.
   `ZividAPI.md` 4장 인덱스로 관련 파일 1~2개만 연다. 레포 전체를 훑지 않는다.
5. 작성한 코드에 참고한 샘플 파일명이나 KB 문서를 주석/설명으로 남긴다.

## 우선 참고 자료

### 1. Zivid 공식 Knowledge Base

- https://support.zivid.com/en/latest/index.html (한국어: https://support.zivid.com/ko/latest/index.html)
- 현재 경로 구조는 `/en/latest/camera/...`(카메라)와 `/en/latest/motion/...`(Zivid Motion)로 나뉜다.
- User Guide, Software Installation, Studio Guide, Quick Capture Tutorial, API Reference,
  Samples, Troubleshooting, Camera Settings, General 3D Topics를 우선 확인한다.
- 한국어 페이지가 어색하거나 부족하면 같은 문서의 영어 원문을 기준으로 한다.

### 2. Zivid 공식 GitHub

- https://github.com/zivid
- 코드 관련 질문은 공식 repo의 README, examples, sample code, package configuration,
  issue 설명을 우선 확인한다.
- 로컬 클론이 있는 레포(`C:\Zivid\` 아래): `zivid-python-samples`, `zivid-csharp-samples`,
  `zivid-halcon-samples`, `zivid-isaac-sim`. 이 레포들은 로컬 파일을 먼저 본다.
- 언어별 우선 repo:
  - Python: `zivid-python-samples`(사용 예제), `zivid-python`(README·Python 모듈 계층만; `src`·`cmake`·CI 폴더는 무시)
  - C++: `zivid-cpp-samples`
  - C#: `zivid-csharp-samples`
  - ROS: `zivid-ros`
  - HALCON: `zivid-halcon-samples`
  - MATLAB: `zivid-matlab-samples`
  - Isaac Sim: `zivid-isaac-sim`
- 웹에서 파일을 읽을 때는 `github.com/.../blob/...` 페이지 대신 `raw.githubusercontent.com` 원문 경로를 쓴다.

## 작업 규칙

- Zivid 관련 기술 질문은 일반적인 3D vision 지식보다 공식 문서와 공식 GitHub를 우선한다.
- 최신성, 설치 방법, API 사용법, 지원 OS, SDK 버전, 설정값, 샘플 코드, troubleshooting은
  반드시 공식 문서 또는 공식 GitHub 기준으로 확인해서 답한다.
- 참고한 문서 제목, GitHub repo명, 파일명, 함수명, 설정 항목명을 함께 알려준다.
- 코드 예시는 Zivid 공식 샘플 스타일을 따르고, API 이름을 임의로 만들어내지 않는다.
- 환경이 중요한 경우 OS, SDK 버전, 카메라 모델, 언어(Python/C++/C#/ROS), 오류 메시지를
  확인 대상으로 삼는다.
- 작업 중 `ZividAPI.md`와 실제 API가 다른 것을 발견하면(버전 업데이트 등) 코드는 실제 API에 맞추고,
  사용자에게 `ZividAPI.md` 갱신이 필요하다고 알린다.
