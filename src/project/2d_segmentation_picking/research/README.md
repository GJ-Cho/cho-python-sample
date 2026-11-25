# 2D 세그멘테이션 방법 조사

> 상태: **1단계(조사) 완료** — 2026-10-29. **2단계(합성 데이터) A 시험 생성 완료** — 2026-11-03 (`../synth/`, `data/synthetic/pilot/` 36장). 2026-11-03 사용자 피드백 반영(박스 1~5호, XL250 4.0 m 고정·작업거리 1.7~4.0 m, 빈 깊이 200/300/400, 점군 피팅 내부 파라미터, 센서 노이즈 후처리) 후 파일럿 재생성(39장). 2026-11-04 센서 노이즈 실측 교정 완료, 노이즈는 학습 때 on-the-fly(저장 안 함), 검은 테이프 추가. **본 생성 완료** — 2026-11-04 (3,900장 + COCO: box 1,500 / one 1,400 / few 1,000).
> **3단계(학습·평가) [RGB] 완료** — 2026-11-09. RF-DETR-Seg Medium, BOX 5종 + 제품 7종 모델, 합성 테스트 결과는 `../train/README.md`. 다음은 [RGB+D]/[RGB+N]/[RGB+D+N] 입력 결합 비교.
> 이 문서는 조사 전체의 진입점이다. 결론 요약은 맨 아래에 있다.

## 목적

현재 데모의 m1(SAM2 AMG, 최상층 프롬프트)과 m2(Grounding DINO → SAM2)를 **기준선**으로 두고,
대상물 조건별로 더 나은 2D 세그멘테이션 방법이 있는지 처음부터 다시 찾는다.
피킹(파지 포즈·로봇 연동)은 범위 밖이다. 결과물은 "마스크를 어떻게 얻을 것인가"까지다.

## 기준선 (README 기준, 2026-07-30, 장면 2개)

| | m1 SAM2 top-layer | m2 Grounded-SAM 2 |
|---|---|---|
| warm 처리 시간 | 0.98s / 1.39s | 0.96s / 1.22s |
| 학습 데이터 | 없음 | 없음 (텍스트 어휘 필요) |
| 약점 | 과분할, 경계 거침 | 어휘 밖 물체 누락 |

정량 recall 벤치마크는 `data/eval/`(GT 12장)과 `scripts/run_benchmark.py` 참조.

## 표기법

**시나리오**

| 표기 | 내용 | 테스트 데이터 |
|---|---|---|
| [BOX] | 팔레트 디팔레타이징, 박스 1종 / 2종 / 3종 혼합 | 새로 생성 |
| [UNK] | 모르는 물건 (학습·어휘 없음) | 기존 Phase 5 GT 세트(`data/eval/`) 재사용 |
| [ONE] | 1종 제품만 | 새로 생성 (학습 데이터 포함) |
| [FEW] | 5종 이하 제품 | 새로 생성 (학습 데이터 포함) |

[ONE]/[FEW]의 제품 유형은 **강체 소비재(병·캔·소형 포장), 산업 부품(무늬 없음·반사), 연성 포장재(파우치)** 세 가지를 모두 다룬다.

**파이프라인 유형**

| 표기 | 의미 | 기존 대응 |
|---|---|---|
| [DS] Detect→Segment | 검출기 박스 → 분할기 마스크 | m2 |
| [E2E] End-to-End | 단일 모델이 인스턴스 마스크 출력 | — |
| [PM] Propose→Match | 마스크 전부 제안 → 레퍼런스와 대조 | m1 앞단과 유사 |

**입력 모달리티**

| 표기 | 입력 |
|---|---|
| [RGB] | Zivid 2D 컬러만 |
| [RGB-D] | 컬러 + depth(Zivid XYZ의 Z 또는 거리) |
| [RGB-D-N] | 컬러 + depth + 법선(Zivid `normals`) |

## 공통 제약

- **속도**: 세그멘테이션 warm 기준 1초 이내, 목표 0.5초 이내. 추론과 후처리를 분리해 기록한다.
- **PC**: Intel Core Ultra 7 155H / RTX 4060 Laptop 8GB VRAM / RAM 16GB
- **라이선스**: 상용 가능(Apache-2.0, MIT, BSD 등)이 최선. AGPL·비상업 라이선스는 표시하고 감점한다. 코드와 가중치 라이선스를 따로 본다.
- **클라우드 API 전용 모델**(Grounding DINO 1.5/1.6, DINO-X 등): 성능 비교용 카드만 쓰고 추천 후보에서는 제외한다.

## 문서 구조

```
research/
├── README.md            ← 이 문서 (목적·표기·결론 요약)
├── evaluation.md        성능 판단 기준: 지표, 측정 규칙, 기준선 수치
├── candidates/          후보 하나당 카드 1장 (_TEMPLATE.md 형식)
├── topics/
│   ├── detectors.md       [DS] 앞단: Grounding DINO 계열과 대안
│   ├── segmenters.md      [DS] 뒷단과 [E2E]: SAM 계열, 경량화, 학습형
│   ├── dino-backbone.md   DINOv2/v3 활용: [PM] 매칭, few-shot, 검출 백본
│   └── input-modality.md  [RGB] vs [RGB-D] vs [RGB-D-N]
├── scenarios/           시나리오별 추천과 근거 (box / unknown / one-product / few-products)
└── data/
    └── dataset-plan.md  테스트·학습 데이터 생성 방법, 출처, 라이선스
```

## 작성 규칙

- 한국어로 쓴다. 모델명·API·라이선스명·논문 제목은 원문 그대로 둔다.
- **확인한 내용과 추론한 내용을 구분한다.** 확인은 출처 링크를 붙이고, 추론은 "(추론)"으로 표시한다.
- 속도 수치는 측정 하드웨어와 해상도를 함께 적는다. 4060 8GB 기준 값은 대부분 추정이므로 근거를 쓴다.
- GitHub 파일은 `raw.githubusercontent.com` 원문 기준으로 확인한다.

## 결론 요약

모든 속도는 RTX 4060 Laptop 기준 **추정**이다. 실측은 3단계에서 한다.

| 시나리오 | 운영 1순위 | 학습 없이 / 라벨 생성용 | 입력 모달리티 판단 (추론) | 상세 |
|---|---|---|---|---|
| [BOX] | RF-DETR-Seg S/M (Apache) | SAM3 텍스트 + 장면 안 예시 박스 | 같은 높이 밀착은 [RGB]로 충분, 층·높이 혼합에 [RGB-D] | `scenarios/box.md` |
| [UNK] | UnSAMv2로 m1 AMG 교체 실험 / m1 프롬프트 수 절감 | (정의상 학습 없음) | m1의 [RGB-D] 후처리 융합 유지, 노말맵 주 입력은 손해(확인) | `scenarios/unknown.md` |
| [ONE] | RF-DETR-Seg 단일 클래스 (Apache) | OWLv2 예시 이미지 / DINOv2 매칭 / SAM3 텍스트 | 강체는 [RGB-D-N] 이득 큼, 반사 부품은 깊이 신뢰도 가중 필요 | `scenarios/one-product.md` |
| [FEW] | RF-DETR-Seg 다중 클래스 (Apache) | DINOv2 매칭 / no-time-to-train / OWLv2 | [ONE]과 같음. 분류는 RGB 외관 | `scenarios/few-products.md` |

### 여러 조사가 공통으로 가리킨 결론 (추론)

1. **"학습 전 방법으로 라벨 → 검수 → RF-DETR-Seg 학습"이 [BOX]·[ONE]·[FEW]의 공통 경로다.** 0.5s 목표를 여유 있게 맞추는 상용 가능 후보는 학습형뿐이다.
2. **Grounding DINO 계열에서 로컬로 쓸 수 있는 것은 1.0뿐이다(확인).** 1.5, 1.6, DINO-X는 API 전용이다. 1.0을 개선하려면 MM-Grounding-DINO나 LLMDet으로 바꾸는 것이 최소 비용이다.
3. **DINOv2/v3는 검출기가 아니라 백본이다.** "무슨 제품인가"(분류)에는 강하지만 "몇 개인가"(인스턴스 분리)에는 약하다.
4. **깊이·법선은 입력 대체보다 후처리 융합이 맞다.** RGB로 제안하고, 깊이·법선으로 쪼개고 합친다.
5. **m1이 느린 원인은 인코더가 아니라 프롬프트 수다.** 경량 SAM으로 바꾸는 것보다 프롬프트 수를 줄이는 것이 먼저다.

### 라이선스 정리

| 구분 | 모델 |
|---|---|
| 상용 가능 (Apache/MIT) | SAM2, RF-DETR-Seg(전 크기), RF-DETR 검출 N~L, Grounding DINO 1.0, MM-GDINO, LLMDet, OWLv2, Florence-2, DINOv2, UnSAMv2(코드), EfficientViT-SAM, SAM-HQ |
| 비표준 (법무 확인) | SAM3(SAM License), DINOv3(DINOv3 License), Objects365 사전학습 가중치 전반 |
| AGPL / GPL | YOLOE, YOLO-seg, FastSAM, YOLO-World |
| 비상업 | EdgeSAM, CropFormer, UCN, UOAIS, CutLER |
| 클라우드 API 전용 (참고만) | Grounding DINO 1.5/1.6, DINO-X, T-Rex2 |

### 3단계 전에 고칠 것 (확인된 결함)

- `run_benchmark.py`가 m1·m2를 한 프로세스에서 돌린다(AGENTS.md 위반). 또 warm을 1회만 재고, 추론/후처리 시간을 나누지 않고, 예측 마스크를 저장하지 않는다.
- **과병합(merge) 지표가 없다.** m1 평면 병합이 서로 다른 물체를 합친 사례(scene03 원통 2개)를 지금 지표로는 잡지 못한다.
- GT가 RGB SAM2 조각을 묶어 만든 것이라 SAM 계열과 RGB 조건에 유리하다.

### 교차검증 기록 (통합 시 직접 확인)

| 주장 | 결과 |
|---|---|
| RF-DETR-Seg 전 크기 Apache-2.0, 검출 XL/2XL은 PML 1.0 | 확인 (rf-detr README) |
| UnSAMv2 Apache-2.0 | 코드는 확인. **가중치 라이선스는 README에 없음** (카드의 HF 카드 근거는 미재확인) |
| SAM3 = SAM License, 비상업 제한 없음, 군사·무역통제 금지, 재배포 시 사본 동봉 | 확인 (sam3 LICENSE). 상용 명시 허가 문구도 없음 |
| Box-IS RGB-D 83.7 AP | 확인. 단 RealSense 촬영, 저자 주장치이고 독립 검증 없음 |

### 2단계 시작 전 결정할 것

1. **데이터 출처:** `data/dataset-plan.md`는 테스트 세트를 실제 Zivid 촬영본으로 권고한다. 합성·공개 데이터만으로 진행할지, 일부 실촬영을 할지
2. [BOX] 규모: 실물 팔레트 vs 탁상 축소 적재
3. 제품 선정: [ONE] 3유형 대표 제품, [FEW] 5종, [BOX] 박스 3종
4. 합성 경로: 노트북 BlenderProc vs 16GB+ GPU 장비 Isaac Sim (노트북은 Isaac Sim 최소 사양 미달)
5. 비상업·CC BY-SA 공개 데이터의 내부 비교용 사용 허용 여부
6. 라벨링 도구: X-AnyLabeling(1인) vs CVAT(다인 검수)
7. [UNK] GT v2 작성 여부
