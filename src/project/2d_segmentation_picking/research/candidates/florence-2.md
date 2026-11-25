# Florence-2 (base 0.23B / large 0.77B)

> 담당 토픽: topics/detectors.md · 작성일: 2026-10-29

## 요약

Microsoft의 seq2seq 비전 파운데이션 모델. 태스크 토큰으로 검출(`<OD>`), **클래스 무관 영역 제안(`<REGION_PROPOSAL>`)**, 구 그라운딩을 수행한다. MIT 라이선스지만 토큰 생성 방식이라 느리고 박스 정밀도가 DETR류보다 떨어져 보조 후보.

## 분류

| 항목 | 값 |
|---|---|
| 파이프라인 유형 | [DS] 앞단 |
| 입력 모달리티 | [RGB] |
| 맞는 시나리오 | [UNK] △(`<REGION_PROPOSAL>`), [BOX]/[ONE]/[FEW] ✕(밀집 장면에서 출력 토큰 길이 제한·속도, 추론) |
| 프롬프트 | 태스크 토큰 / 텍스트 |

## 라이선스

| | 라이선스 | 상용 가능 | 출처 |
|---|---|---|---|
| 코드 | MIT (HF 원격 코드) | 예 | [HF 카드](https://huggingface.co/microsoft/Florence-2-large) (확인) |
| 가중치 | MIT | 예 | 같은 카드 (확인) |

## 학습 데이터

없음. fine-tune 가능(커뮤니티 예제 다수, 추론).

## 속도

| 수치 | 하드웨어 | 해상도 | 출처 (확인/추론) |
|---|---|---|---|
| `<OPEN_VOCABULARY_DETECTION>` 약 698ms wall (모델 크기 불명) | T4 (Colab) | 1600×1066 입력(내부 768²) | 커뮤니티 Colab 샘플 (일화) |
| caption 128토큰 base 652ms / large 1120ms | L4, bs1 | — | [Roboflow 문서](https://docs.roboflow.com/models/supported-models/florence-2) (확인, 다른 태스크) |

RTX 4060 Laptop 8GB 추정: 0.3~0.8s (추론). 출력 박스 수에 비례해 디코딩 시간이 는다 — 물체가 수십 개인 빈에서 불리.

## 성숙도

- 가중치 공개, transformers(`trust_remote_code`, 최신 버전은 네이티브) 지원(추론).
- `<OPEN_VOCABULARY_DETECTION>` 태스크는 모델 카드 본문엔 없고 공식 예제 노트북에 있음(추론).

## 기준선(m1/m2) 대비

- 기대 이점: MIT, 클래스 무관 영역 제안 기능.
- 위험: 1초 예산을 검출만으로 거의 소진. 밀집 물체 recall 미검증.

## 출처

- https://huggingface.co/microsoft/Florence-2-large
- arXiv 2311.06242 (Florence-2: Advancing a Unified Representation for a Variety of Vision Tasks)
