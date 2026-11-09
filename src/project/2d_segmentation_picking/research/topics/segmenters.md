# [DS] 뒷단과 [E2E]: 분할기 (SAM 계열, 경량화, 학습형)

> 작성일: 2026-10-29 · 담당: segmenters 조사 에이전트
> 범위: 박스·점 프롬프트를 받는 **분할기**([DS] 뒷단), 단일 모델이 인스턴스 마스크를 내는 **[E2E]**, AMG 같은 클래스 무관 제안기.
> 검출기(Grounding DINO·OWLv2·YOLO-World 등)는 `detectors.md`, DINOv2/v3·CNOS·PerSAM 매칭은 `dino-backbone.md`,
> RGB-D 모델(UOIS·UOAIS·MSMFormer 등)은 `input-modality.md` 소관이다. 여기서는 교차 참조만 한다.
> 표기: **확인** = 출처 링크 있음(각 카드에 원문 링크), **(추론)** = 근거는 있으나 직접 확인 못함.

## 0. 기준선과 이 문서의 질문

- m1 = SAM2.1 hiera_small AMG, 프롬프트를 최상층 bbox 격자(163~232개)에만, `multimask_output=True`, bf16 autocast → warm 1.01 / 1.39s. 주 약점은 **과분할·중복** (PLAN 10장, 확인).
- m2 = grounding-dino-tiny → SAM2.1 small 박스 프롬프트 → 0.92 / 1.08s (확인).
- 이 문서가 답하려는 질문 3개:
  1. 뒷단 SAM2를 **바꾸면** 속도·경계가 좋아지는가? (경량 SAM, SAM-HQ)
  2. m1 과분할을 **구조적으로** 줄이는 분할기가 있는가? (granularity 제어, 엔티티 분할, 개념 분할)
  3. 학습 데이터를 만들 수 있는 시나리오에서 [E2E] 학습형이 [DS]를 대체하는가?

## 1. 비교표

"4060L 추정" 열은 RTX 4060 Laptop 8GB 기준 **추정치(전부 추론)**. 근거는 각 카드.

### 1.1 Promptable 분할기 ([DS] 뒷단 / AMG 제안기)

| 모델 (카드) | 코드 / 가중치 라이선스 | 공개 속도 (확인) | 4060L 추정 (추론) | 프롬프트 | 비고 |
|---|---|---|---|---|---|
| [SAM2.1 t/s/b+/l](../candidates/sam2.md) | Apache-2.0 / Apache-2.0 | A100 91/85/64/40 FPS | 인코더 small 50~100ms. m1 실측 1.0~1.4s | 점·박스 / AMG | **기준선** |
| SAM v1 (ViT-B/L/H) | Apache-2.0 / Apache-2.0 | ViT-H 인코더 ~452ms (MobileSAM README) | 0.3~0.6s | 점·박스 / AMG | SAM2로 대체됨, 카드 생략 |
| [SAM3 / 3.1](../candidates/sam3.md) | **SAM License**(비표준, 군사·무역통제 금지) / 동일 + **HF gated** | H200 30ms (객체 100+) | 텍스트 1개당 0.3~0.6s, bf16이면 8GB 적재 가능성 높음 | **텍스트 / 같은 이미지 예시 박스** / 점·박스 | torch 2.7+·Py3.12+·CUDA 12.6+ |
| [EfficientSAM3](../candidates/efficientsam3.md) | README Apache, LICENSE 파일 없음 / **불명확**(SAM3 증류, EdgeSAM 언급) | 지연 수치 없음, ~90M | 50~150ms/프롬프트 | 텍스트 / 점·박스 | 정확도 미공개 |
| [SAM-HQ / HQ-SAM 2](../candidates/sam-hq.md) | Apache-2.0 / Apache-2.0(추론) | Light HQ-SAM 41.2 FPS | SAM2 + 수~10% | 점·박스 | 경계 품질 |
| [EfficientViT-SAM](../candidates/efficientvit-sam.md) | Apache-2.0 / Apache-2.0(추론) | A100 TRT XL0 278 img/s, Orin 22.5ms | XL0 TRT 15~30ms | 점·박스 / AMG | **TRT 공식 지원**, 정확도 손실 거의 없음 |
| [MobileSAM(v2)](../candidates/mobilesam.md) | Apache-2.0 / 동일(추론) | 12ms/장 | 10~20ms | 점·박스 / object-aware everything | COCO AP 39~41 |
| [EdgeSAM](../candidates/edgesam.md) | **S-Lab 비상업** / 동일 | 2080Ti 164 FPS | 5~10ms | 점·박스 | **제외** |
| [RepViT-SAM](../candidates/repvit-sam.md) | Apache-2.0 / 동일(추론) | iPhone12 48.9ms | 10~20ms | 점·박스 | EdgeSAM의 Apache 대안 |
| [EfficientSAM](../candidates/efficientsam.md) | Apache-2.0 / 동일(추론) | 미확인 | 20~50ms | 점·박스 | 공식 AMG 없음 |
| [TinySAM](../candidates/tinysam.md) | Apache-2.0 / 동일(추론) | FLOPs만 (AP 42.3) | 10~25ms | 점·박스 / 계층적 everything | 가속 **기법**이 핵심 |
| [FastSAM](../candidates/fastsam.md) | **LICENSE=AGPL** (README는 Apache, 모순) / YOLOv8 기반 | 3090 ~40ms | 60~120ms | everything / 점·박스·텍스트 | 경계 거침 |
| [UnSAMv2](../candidates/unsamv2.md) | Apache-2.0 / Apache-2.0 | 미확인 | SAM2와 유사 | **점 + granularity 스칼라** / 전체 분할 | **과분할 직접 겨냥** |

### 1.2 [E2E] 인스턴스 분할

| 모델 (카드) | 종류 | 코드 / 가중치 라이선스 | 학습 데이터 | 공개 속도·정확도 (확인) | 4060L 추정 (추론) |
|---|---|---|---|---|---|
| [RF-DETR-Seg N~2XL](../candidates/rf-detr-seg.md) | 학습형 | Apache-2.0 / **전 크기 Apache-2.0** | 라벨 수십~수백 | T4 TRT 3.4~21.8ms, COCO mask AP 40.3~49.9 | TRT ~5ms(M), PyTorch 20~60ms |
| [YOLO-seg (v8/11/26)](../candidates/yolo-seg-ultralytics.md) | 학습형 | **AGPL-3.0** / AGPL | 라벨 수십~수백 | YOLO26s-seg T4 TRT 3.3ms, mask 40.0 | 5~15ms |
| [RTMDet-Ins](../candidates/rtmdet-ins.md) | 학습형 | Apache-2.0 / 동일(추론) | 라벨 수십~수백 | 3090 TRT 1.7~5.3ms, mask 35.4~44.6 | 15~40ms, **mmcv 설치 부담** |
| [SparseInst](../candidates/sparseinst.md) | 학습형 | MIT / 동일(추론) | 라벨 수십~수백 | 39~44 FPS, AP 32.8~37.7 | 20~30ms |
| [Mask2Former](../candidates/mask2former.md) | 학습형 | MIT / 동일(추론) | 라벨 수백+ | R50 AP ~43.7 (추론, 미재확인) | 100~200ms |
| [Mask DINO](../candidates/mask-dino.md) | 학습형 | Apache-2.0 / 동일(추론) | 라벨 수백+ | R50 mask AP 46.1~46.3 | 150~300ms, 커스텀 op |
| [SAM3](../candidates/sam3.md) | open-vocab | SAM License | 없음 (10-shot FT 가능) | LVIS zero-shot mask AP 48.8, SA-Co/Gold cgF1 54.1 (사람 72.8) | 0.3~0.6s/텍스트 |
| [YOLOE-seg](../candidates/yoloe.md) | open-vocab + prompt-free | **AGPL-3.0** | 없음 | LVIS AP v8-L 35.9(텍스트)/27.2(pf), T4 TRT 102 FPS | 10~30ms |
| [SEEM/OpenSeeD/X-Decoder](../candidates/seem-openseed-xdecoder.md) | open-vocab | Apache-2.0 | 없음 | 미확인 | 수백 ms, 설치 부담 |
| [CropFormer](../candidates/cropformer.md) | 클래스 무관 엔티티 | **CC BY-NC 4.0** | 없음 | 미확인 | 0.3~1s |
| UOAIS / UOIS / MSMFormer | 클래스 무관 RGB-D | → `input-modality.md` | — | — | — |

입력 모달리티: 이 문서의 모든 모델은 **[RGB]** 사전학습이다. 깊이/노말맵을 3채널로 렌더링해 넣는 방식과 그 근거는 `input-modality.md` 참조.

## 2. 핵심 관찰

### 2.1 m1의 시간은 인코더가 아니라 "프롬프트 수 × 디코더 + 후처리"다 (추론)

- PLAN 10장: m1 시간은 프롬프트 수(163 → 232)에 비례(확인). SAM2-small 인코더는 1장당 1회이고 A100 85 FPS급이라 4060L에서도 100ms 안쪽일 것(추론).
- 따라서 **경량 SAM으로 인코더만 바꾸면 m1 가속 폭은 제한적**이다. 효과가 클 것으로 보는 순서(추론):
  1. **프롬프트 수 줄이기** — 격자 대신 object-aware 프롬프트(MobileSAMv2 아이디어: 검출기 박스·깊이 클러스터 중심), TinySAM식 계층 샘플링(저밀도 1차 → 미커버 영역만 2차).
  2. `points_per_batch` 상향(기본 64 → 128~256, VRAM 허용 시) — 디코더 호출 횟수 감소. 파라미터 존재는 확인(`sam2/automatic_mask_generator.py`).
  3. 후처리(NMS·RLE·평면 피팅)를 분리 측정한 뒤 최적화. README 규칙대로 추론과 후처리를 나눠 기록해야 판단할 수 있다.
  4. 인코더 ONNX/TensorRT — Hiera-S ONNX 공개본은 있으나 RTX 수치는 공개된 것이 없다(확인). EfficientViT-SAM은 TRT 경로가 공식.

### 2.2 과분할은 "모델 크기"가 아니라 "점 프롬프트의 모호성" 문제다

- SAM 계열은 점 하나에 부분/전체/그룹 3후보를 내도록 설계됐다(`multimask_output`). 라벨·테이프·하이라이트가 많은 산업 장면에서 부분 마스크가 살아남는 것은 구조적이다(추론, m1 관찰과 일치).
- 구조적 해법 후보:
  - **granularity 제어**: UnSAMv2(Apache, SAM-2 기반) — 객체 수준 granularity를 고정해 부분 마스크 생성 자체를 억제. **[UNK] 1순위 실험 대상.** Semantic-SAM은 라이선스 미확인·detectron2 부담.
  - **엔티티 분할**: CropFormer — 비중첩 엔티티 출력이라 이상적이지만 CC BY-NC → 상한 참고용.
  - **개념 분할**: SAM3 PCS — "box", "pouch"처럼 개념을 주면 인스턴스 단위 출력. 어휘가 있으면 과분할 문제를 우회한다.
  - **박스 프롬프트**: 점 대신 박스면 모호성이 크게 준다(m2가 m1보다 깔끔한 이유로 추정). 따라서 [DS]에서는 뒷단 교체보다 앞단 박스 품질이 결과를 좌우한다(추론).
- 후처리(평면 병합)는 위 어느 모델을 써도 재사용할 수 있다.

### 2.3 SAM3 정리

- 확인: SAM License(비표준, 비상업 제한 조항은 없음, 군사·무역통제 금지, 재배포 시 사본 동봉), HF gated, 848M, H200 30ms, torch 2.7+/Py3.12+/CUDA 12.6+ → **현재 venv(torch 2.6+cu124)와 맞지 않아 별도 venv 필요**.
- 확인: transformers `Sam3Model`/`Sam3Processor`가 텍스트 + positive/negative 박스를 지원한다. **예시 박스는 같은 이미지 안에서 지정**하며, 다른 사진의 레퍼런스를 바로 넣는 공식 API는 문서에 없다. `dino-backbone.md`의 cnos25 저자 의견("SAM3는 cross-image exemplar 미지원")과 일치.
  - 함의: [BOX]는 "장면에서 박스 하나를 사람이나 검출기가 찍어 주면 같은 박스 전부" 흐름에 딱 맞다. [ONE]/[FEW]의 "제품 사진 1장 등록" 흐름은 SAM3 단독으로는 안 되고 [PM](CNOS 계열)이나 학습형이 필요하다.
- 확인: 공식 AMG 모드는 없다. [UNK]에서는 SAM2와 다를 바 없다(추론).
- 4060L 8GB: bf16이면 가중치 ~1.7GB라 적재 가능성이 높고, 시간은 텍스트 프롬프트 수에 선형(추론). 블로그 실측 VRAM ~8000MiB(정밀도 미기재)라 fp32는 위험. **직접 측정 전에는 1s 예산 충족을 단정할 수 없다.**
- 산업 부품 정량 평가는 찾지 못했다. CAD-Prompted SAM3(arXiv 2602.20551)는 "같은 부품도 재질·색이 달라 외관 프롬프트가 불안정"하다고 문제를 제기한다(확인, 초록).

## 3. [DS] 조합 (검출기 × 분할기)

| 앞단 (→ `detectors.md`) | 뒷단 | 판단 |
|---|---|---|
| grounding-dino-tiny (현 m2) | SAM2.1 small | 기준선 |
| MM-GDINO-T / LLMDet-T | SAM2.1 small | 앞단만 교체 — 최소 비용 실험 (detectors.md 1순위와 일치) |
| 동일 | **EfficientViT-SAM XL0 (TRT)** | 뒷단 시간 단축용. 박스당 디코더 비용은 비슷해 이득은 인코더분(수십 ms) (추론) |
| 동일 | SAM-HQ / HQ-SAM 2 | 파우치·얇은 물체 경계 개선 기대, 시간 소폭 증가 |
| RF-DETR (학습) | SAM2.1 small | 박스 라벨만 있을 때 유효. 마스크 라벨까지 있으면 RF-DETR-Seg 단일 모델이 더 빠름 |
| OWLv2 image-guided / CNOS 제안 | SAM2.1 | [ONE] 학습 전 데모용 (detectors / dino-backbone 참조) |
| **SAM3 텍스트/예시 (검출+분할 일체)** | (내장) | m2의 2모델 → 1모델. [BOX]와 어휘 있는 [FEW]에서 m2 대체 1순위 |
| 깊이 클러스터·최상층 밴드 중심점 | SAM2 / UnSAMv2 | m1 프롬프트 수 절감(object-aware prompting의 3D 버전) (추론) |

## 4. 시나리오별 추천

### [BOX] 팔레트 박스 1~3종
1. **학습 없이**: **SAM3** — 텍스트 "cardboard box" + 장면 내 예시 박스 1~3개(종류별). 밀착 동일 박스의 경계 분리가 핵심 위험이라 depth 엣지 후처리와 결합해야 할 것(`input-modality.md`) (추론).
2. **학습 가능**: **RF-DETR-Seg (S/M)** — Apache, 4060L에서 수십 ms. 박스는 합성 데이터 생성이 쉽다(→ `data/dataset-plan.md`). SAM3로 자동 라벨 → 검수 → 학습.
3. 비교: m2(GD-T → SAM2), YOLO-seg(AGPL).

### [UNK] 모르는 물건
1. **UnSAMv2**(granularity 고정)로 m1의 AMG를 교체하는 실험 — 같은 `data/eval/` GT로 과분할 지표 비교.
2. m1 유지 + **프롬프트 수 절감**(계층 샘플링/깊이 클러스터 프롬프트) + `points_per_batch` 상향으로 0.5s 목표에 접근.
3. EfficientViT-SAM AMG(TRT)는 인코더 가속분만 기대.
4. 상한 참고: CropFormer(비상업), SAM3에 "object" 같은 일반 어휘(효과 미확인).

### [ONE] 1종 제품
1. **RF-DETR-Seg 단일 클래스** — 강체 소비재·반사 부품·파우치를 모두 학습으로 흡수. 라벨 수십~수백.
2. 학습 전 데모: SAM3 텍스트(강체 소비재처럼 "bottle", "can" 등 어휘가 명확할 때) 또는 장면 내 예시 박스. 무늬 없는 반사 부품은 SAM3 텍스트가 약할 가능성(추론). 레퍼런스 사진 등록형은 `dino-backbone.md`([PM]).
3. 비교: YOLO-seg(AGPL), RTMDet-Ins(설치 부담).

### [FEW] 5종 이하
1. **RF-DETR-Seg 다중 클래스**.
2. 학습 전: SAM3 클래스별 텍스트 프롬프트. 프롬프트 수만큼 시간이 늘어 5종이면 4060L에서 1s를 넘길 수 있다(추론). 또는 SAM3/EfficientSAM3를 **자동 라벨링 도구**로만 쓰고 배포는 RF-DETR-Seg.
3. 파우치: SAM-HQ 계열 뒷단이 경계에 유리할 수 있다(추론).

### 라이선스 요약
- 깨끗함(Apache/MIT): SAM/SAM2, SAM-HQ, EfficientViT-SAM, MobileSAM, RepViT-SAM, EfficientSAM, TinySAM, UnSAMv2, RF-DETR-Seg(전 크기), RTMDet-Ins, SparseInst, Mask2Former, Mask DINO, SEEM/OpenSeeD.
- 비표준: **SAM3**(SAM License, gated), **EfficientSAM3**(불명확).
- AGPL: **YOLO-seg, YOLOE, FastSAM**(LICENSE 파일 기준).
- 비상업: **EdgeSAM**(S-Lab), **CropFormer**(CC BY-NC), UnSAM v1(CC BY-NC).

## 5. 열린 질문

1. **SAM3의 4060L 실측**: bf16 VRAM 피크, 텍스트 프롬프트 1개/5개 시간, 검출 수가 많을 때 OOM 여부. 별도 venv(torch 2.7+) 필요.
2. SAM License가 Zivid 데모·고객 배포에 적합한가(법무). EfficientSAM3 가중치가 SAM License나 EdgeSAM 조건을 승계하는가.
3. m1 시간 분해: 인코더 / 디코더(프롬프트 수) / AMG 후처리 / 평면 병합 각각의 ms. 이것 없이는 경량 SAM 교체 이득을 판단할 수 없다.
4. UnSAMv2 전체 분할 모드의 속도, 그리고 산업 장면(라벨 붙은 박스, 반사 부품)에서 granularity 하나로 고정할 수 있는가.
5. SAM3 예시 박스가 "같은 무늬 밀착 박스"를 인스턴스별로 분리하는가, 한 덩어리로 묶는가.
6. RF-DETR-Seg 입력 해상도(312~768)가 Zivid 2+ MR130 빈 ROI에서 작은 물체 경계에 충분한가.
7. FastSAM 라이선스 모순(README Apache vs LICENSE AGPL)에 대한 공식 입장.
8. MobileSAMv2 프롬프트 생성기가 ultralytics(AGPL)에 의존하는가.
