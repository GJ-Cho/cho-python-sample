# EfficientSAM3 (SAM3 증류 경량판) / SAM3-LiteText

> 담당 토픽: topics/segmenters.md · 작성일: 2026-10-29

## 요약

SAM3를 Progressive Hierarchical Distillation으로 경량 백본(EfficientViT/RepViT/TinyViT)에 증류한 커뮤니티 프로젝트. Stage 3(2026-06-11, v0.4.0)에서 **텍스트 PCS까지 가능한 ~90M 모델**을 공개했다. SAM3가 4060에서 느릴 경우의 대안 후보.

## 분류

| 항목 | 값 |
|---|---|
| 파이프라인 유형 | [E2E] (텍스트/예시) / [DS] 뒷단 |
| 입력 모달리티 | [RGB] |
| 맞는 시나리오 | SAM3와 동일하되 정확도 손실 감수 — [BOX] 우선 검증 대상 |
| 프롬프트 | 텍스트 / 박스·점 |

## 라이선스

| | 라이선스 | 상용 가능 | 출처 |
|---|---|---|---|
| 코드 | README에 Apache-2.0 표기, **GitHub에 LICENSE 파일 미검출**(API `license: null`) | 불명확 | 확인 — https://raw.githubusercontent.com/SimonZeng7108/efficientsam3/main/README.md |
| 가중치 | HF 카드 apache-2.0 표기. 단 SAM3 출력으로 증류 + "SAM/SAM2/SAM3/**EdgeSAM**… 각 라이선스 참조" 문구 | **불명확** — SAM License 승계 가능성, EdgeSAM(비상업) 구성요소 포함 여부 확인 필요 | 확인 — https://huggingface.co/Simon7108528/EfficientSAM3 / 승계 판단은 (추론) |

## 학습 데이터

없음(증류 완료 가중치 사용).

## 속도

| 수치 | 하드웨어 | 해상도 | 출처 (확인/추론) |
|---|---|---|---|
| EV-M 89.2M / RV-M 92.7M / TV-M 95.3M (SAM3 861.5M 대비 ~90%↓) | — | — | 확인 — README |
| 공식 지연 수치 없음 (COCO val 전체 평가 시간만, RTX 4070 Ti) | RTX 4070 Ti | — | 확인 — HF 카드 |
| SAM3-LiteText: SA-Co/Gold cgF1 53.6~53.95 (SAM3 54.1), 텍스트 인코더만 경량화(비전 인코더 ~463M 유지) | — | — | 확인 — HF 카드 |

RTX 4060 Laptop 8GB 추정: 텍스트 프롬프트당 50~150ms (추론) — 파라미터 1/9, 비전 인코더 20~28M.

## 성숙도

- 개인/학계 프로젝트, 가중치 공개, 마지막 push 2026-08. Stage 3 PCS 모델의 **정확도 수치가 README에 없음**.
- ONNX/TensorRT 언급 없음.

## 기준선(m1/m2) 대비

- 기대 이점: SAM3 개념 분할을 1s 예산 안으로.
- 위험: 정확도 공개 부족, 라이선스 불명확, 성숙도 낮음. 조사 단계에서는 "SAM3가 너무 느릴 때의 예비"로만 둔다.

## 출처

- https://github.com/SimonZeng7108/efficientsam3 · https://arxiv.org/abs/2511.15833 · https://arxiv.org/abs/2602.12173
