# SAM 3 / SAM 3.1

> 담당 토픽: topics/segmenters.md · 작성일: 2026-10-29

## 요약

Meta의 3세대 SAM(2025-11). 텍스트 명사구·예시 박스(exemplar)로 지정한 **개념의 모든 인스턴스**를 한 번에 검출+분할(Promptable Concept Segmentation, PCS). 단일 모델로 m2(Grounding DINO → SAM2)를 대체하는 [E2E] open-vocab 후보이자, 점·박스 프롬프트 [DS] 뒷단도 겸한다. SAM 3.1(2026-03-27, Object Multiplex)은 비디오 다중 객체 추적 가속판.

## 분류

| 항목 | 값 |
|---|---|
| 파이프라인 유형 | [E2E] (텍스트/예시 → 전 인스턴스) / [DS] 뒷단 (점·박스) |
| 입력 모달리티 | [RGB] |
| 맞는 시나리오 | [BOX] "box/carton" 텍스트 + 같은 장면 예시 박스 1~3개 — 1~3종 혼합에 자연스러움 · [ONE][FEW] 강체 소비재는 텍스트로 가능성 큼, 무늬 없는 산업 부품은 불확실 · [UNK] 어휘가 없으면 장점 소멸 |
| 프롬프트 | 텍스트 / 예시 박스(positive·negative) / 점·박스·마스크 |

## 라이선스

| | 라이선스 | 상용 가능 | 출처 |
|---|---|---|---|
| 코드 | **SAM License** (Meta 자체, 2025-11-19) | 조건부 가능 — 비상업 제한 조항 없음. 단 군사·무역통제 금지, 역공학 금지, 재배포 시 라이선스 사본 동봉 | 확인 — https://raw.githubusercontent.com/facebookresearch/sam3/main/LICENSE |
| 가중치 | SAM License, **HF gated** (`facebook/sam3`, `facebook/sam3.1`; 접근 신청·승인 후 `hf auth login`) | 동일 | 확인 — https://raw.githubusercontent.com/facebookresearch/sam3/main/README.md |

Apache/MIT보다 한 단계 아래로 본다(표준 OSS 아님, 법무 확인 필요) (추론).

## 학습 데이터

없음 (SA-Co 4M 개념으로 사전학습). 도메인 적응은 10-shot 미세조정으로 ODinW13 AP 71.8, RF-100VL AP 36.5 — 확인 (https://arxiv.org/html/2511.16719v2).

## 속도

| 수치 | 하드웨어 | 해상도 | 출처 (확인/추론) |
|---|---|---|---|
| 이미지 1장, 객체 100+개 ≈ 30ms | H200 | 1008 (추론: 기본 입력) | 확인 — 논문 v2 |
| 4클래스 15.8 FPS (클래스당 순차) | RTX 4080 | 1008×1008 | 확인(2차) — DART 논문, 검색 요약 |
| VRAM ≈ 8000MiB (정밀도·GPU 미기재) | 미상 | 미상 | 확인(블로그) — https://zenn.dev/watanko/articles/ef4370d5d94dac |

RTX 4060 Laptop 8GB 추정: **텍스트 프롬프트 1개당 0.3~0.6s (bf16), 프롬프트 수에 거의 선형** (추론) — 848M 파라미터, H200→4060L 연산량 1/10~1/20, RTX 4080 87ms/클래스 → 4060L은 4080 대비 약 1/3 성능. VRAM: 가중치 bf16 ≈ 1.7GB, fp32 ≈ 3.4GB + 활성값 → **bf16이면 8GB 안에 들어갈 가능성 높고 fp32는 빠듯** (추론). 검출 수가 많으면 A10G 24GB에서도 OOM 보고가 있음(HF discussion, 확인(2차)).

## 성숙도

- 공식 코드·가중치 공개(로컬 실행 가능). 요구사항 **Python 3.12+, PyTorch 2.7+, CUDA 12.6+** — 확인 (README). 현재 venv(torch 2.6+cu124)와 **불일치** → 별도 venv 필요.
- HF `transformers`에 `Sam3Model`/`Sam3Processor` 포함, 텍스트·positive/negative 박스 프롬프트 지원 — 확인 (https://huggingface.co/docs/transformers/main/en/model_doc/sam3).
- **예시(exemplar)는 "같은 이미지 안의 박스"** 다. 다른 이미지의 레퍼런스 사진을 바로 넣는 API는 문서에 없다 — 확인(문서 예제 기준). 레퍼런스 크롭을 대상 영상에 이어 붙이는 우회는 (추론, 검증 필요).
- 공식 "segment everything"(AMG) 모드는 README에 없음 — 확인. 기하 프롬프트로 SAM2식 AMG를 돌릴 수는 있다 (추론).
- 마지막 push 2026-10-23, 활발.
- 산업 부품: 같은 부품이 재질·색이 달라 외관 프롬프트가 불안정하다는 지적(CAD-Prompted SAM3, arXiv 2602.20551) — 확인(초록). bin picking 정량 평가는 찾지 못함.

## 기준선(m1/m2) 대비

- 기대 이점: m2의 2모델을 1모델로. presence head로 어휘 오검출(빈 자체를 "box"로 잡던 문제 등) 감소 기대 (추론). [BOX]에서 같은 장면 예시 박스 1개로 동일 박스 전체를 잡는 흐름이 가장 자연스럽다.
- 위험: 4060에서 1s 예산 초과 가능성(프롬프트 다수 시). 라이선스 비표준. 별도 venv. 무늬 없는 반사 부품·밀착 동일 박스 경계는 미검증.

## 출처

- https://github.com/facebookresearch/sam3 · https://arxiv.org/abs/2511.16719 · https://huggingface.co/facebook/sam3
- 사용자 노트: `07_Literature_Notes/sources/sam3-segment-anything-concepts.md`
- https://arxiv.org/abs/2602.20551 (CAD-Prompted SAM3)
