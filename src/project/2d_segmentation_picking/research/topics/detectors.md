# [DS] 앞단: 검출기 (Grounding DINO 계열과 대안)

> 작성일: 2026-10-29 · 담당: detectors 조사 에이전트
> 범위: 분할기(SAM 계열)에 박스·프롬프트를 넘기는 **검출기**만 다룬다. SAM/SAM3/RF-DETR-Seg 등 분할기는
> `segmenters.md`, DINOv2/v3 백본과 [PM] 매칭은 `dino-backbone.md`가 담당한다.
> 표기: **확인** = 출처 링크 있음, **(추론)** = 근거는 있으나 직접 확인 못함.

## 0. 현재 m2의 검출기

- `IDEA-Research/grounding-dino-tiny`(transformers, `AutoModelForZeroShotObjectDetection`), box 0.15 / text 0.25, 17개 어휘 프롬프트 + 클래스 무관 NMS 0.7 (`config/m2_grounded_sam.yaml`).
- m2 전체 warm 0.96 / 1.22s(SAM2 포함). **검출 단계만의 시간은 기록돼 있지 않다** → 열린 질문 1.
- `grounding-dino-base`는 2.5배 느리면서 이득이 없어 tiny 유지(같은 yaml 주석).

## 1. 용어 정리: "DINO 1, 2, 3"은 서로 다른 두 계열이다

| 이름 | 만든 곳 | 무엇 | 단독 검출 가능? |
|---|---|---|---|
| DINO (2021) / DINOv2 (2023) / DINOv3 (2025) | Meta | 자기지도학습 **ViT 백본**(특징 추출기). 검출 헤드 없음 | 아니오 — 매칭·백본 용도 (→ `dino-backbone.md`) |
| DINO (2022, "DETR with Improved deNoising anchOr boxes") | IDEA | **폐쇄형(closed-set) DETR 검출기** | 예 (학습 필요) |
| Grounding DINO 1.0 (2023) | IDEA | 위 DETR-DINO + 텍스트 그라운딩 → 개방어휘 검출. 백본은 Swin(DINOv2 아님) | 예 (텍스트) |
| Grounding DINO 1.5 Pro/Edge (2024.05), 1.6 Pro (2024) | IDEA | 1.0의 대규모 학습판 | 예, **클라우드 API 전용** |
| DINO-X Pro/Edge (2024.11) | IDEA | 1.5 기반 + 마스크·키포인트·프롬프트 없는 검출 | 예, **클라우드 API 전용** |

즉 Grounding DINO의 "DINO"는 IDEA의 검출기 이름이고, Meta DINOv2/v3와는 이름만 겹친다(추론 아님, 각 논문 저자·구조 기준).
두 계열이 만나는 지점은 DEIMv2처럼 **DINOv3를 백본으로 쓰는 폐쇄형 검출기**다 ([DEIMv2 README](https://raw.githubusercontent.com/Intellindust-AI-Lab/DEIMv2/main/README.md), 확인).

### Grounding DINO 버전별 차이 (확인)

| 버전 | 가중치 | 라이선스 | 성능 (zero-shot) | 속도 | 출처 |
|---|---|---|---|---|---|
| 1.0 T / B | **공개** | Apache-2.0 (코드 LICENSE) | COCO 48.4(T) / 56.7(B, COCO 포함 학습), LVIS minival 28.8(T) | 공식 수치 없음 | [README](https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/README.md), [LICENSE](https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/LICENSE), [MM-GDINO 표](https://raw.githubusercontent.com/open-mmlab/mmdetection/main/configs/mm_grounding_dino/README.md) |
| 1.5 Pro / Edge | **API 전용** (DeepDataSpace 토큰) | 레포(API 예제 코드)만 Apache-2.0 | Edge: COCO 45.0 @800×1333 | Edge: Orin NX 10.7 FPS @640², A100 TRT 75.2 FPS @800×1333 | [1.5 API repo](https://github.com/IDEA-Research/Grounding-DINO-1.5-API), [arXiv 2405.10300](https://arxiv.org/pdf/2405.10300) |
| 1.6 Pro | **API 전용** | 위와 동일 | COCO 55.4, LVIS minival 57.7 / val 51.1 | 공개 수치 없음 | [1.5 API repo](https://github.com/IDEA-Research/Grounding-DINO-1.5-API) |
| DINO-X Pro / Edge | **API 전용** | 레포(API 예제)만 Apache-2.0 | Pro: COCO 56.0, LVIS minival 59.8 / Edge: COCO 48.7 @640² | Edge: Orin NX FP16 TRT 20.1 FPS @640² | [DINO-X API repo](https://github.com/IDEA-Research/DINO-X-API), [arXiv 2411.14347v3](https://arxiv.org/html/2411.14347v3) |

1.5/1.6/DINO-X는 레포에 "Apache-2.0"이 붙어 있지만 **API 호출 코드의 라이선스**이고, 모델 가중치 다운로드 경로는 어디에도 없다(확인). 연구 README 규칙대로 비교용 카드만 둔다.

## 2. 후보 비교표

라이선스: 코드 / 가중치. 4060 추정은 전부 **(추론)** — 근거는 각 카드에. "T4 TRT" = NVIDIA T4, TensorRT FP16, batch 1.

| 후보 | 유형 | 프롬프트 | 라이선스 코드 / 가중치 | 학습 데이터 | 공개 속도 | 4060 Laptop 추정 | [BOX] | [UNK] | [ONE] | [FEW] |
|---|---|---|---|---|---|---|---|---|---|---|
| [Grounding DINO 1.0 T](../candidates/grounding-dino.md) (현 m2) | 개방어휘 | 텍스트 | Apache / Apache* | 없음 | 공식 없음 | 0.1~0.25s (PyTorch) | △ | △ 어휘 의존 | △ | △ |
| [MM-Grounding-DINO T](../candidates/mm-grounding-dino.md) | 개방어휘 | 텍스트 | Apache / Apache* | 없음 | 공식 없음 | GD-T와 비슷 | △ | ○ LVIS +11AP | △ | △ |
| [LLMDet T/B](../candidates/llmdet.md) | 개방어휘 | 텍스트 | Apache / Apache* | 없음 | 공식 없음 | GD-T와 비슷 (T) | △ | ○ LVIS minival 44.7 | △ | △ |
| [OWLv2 B/16](../candidates/owlv2.md) | 개방어휘 | 텍스트 **+ 예시 이미지** | Apache / Apache | 없음~레퍼런스 몇 장 | 공식 없음 | 0.15~0.4s @960² | △ | ○ objectness | ○ | ○ |
| [OmDet-Turbo](../candidates/omdet-turbo.md) | 개방어휘(실시간) | 텍스트 | Apache / Apache | 없음 | 카드에 수치 없음 | 0.03~0.1s | △ | △ | △ | △ |
| [Florence-2 B/L](../candidates/florence-2.md) | VLM(seq2seq) | 텍스트·태스크 토큰 | MIT / MIT | 없음 | T4 ≈0.7s(커뮤니티) | 0.3~0.8s | ✕ | △ `<REGION_PROPOSAL>` | ✕ | ✕ |
| [YOLO-World](../candidates/yolo-world.md) | 개방어휘(실시간) | 텍스트 (+이미지 preview) | **GPL-3.0** / GPL-3.0 | 없음 | 공식 FPS 없음 | 0.01~0.05s | △ | △ | △ | △ |
| [YOLOE (v8/11/26)](../candidates/yoloe.md) | 개방어휘(실시간) | 텍스트 **+ 시각 프롬프트 + prompt-free** | **AGPL-3.0** / AGPL-3.0 | 없음 | T4 TRT L: 102 FPS(텍스트)/27 FPS(시각) | 0.01~0.05s | ○ | ○ prompt-free | ○ | ○ |
| [Rex-Omni 3B](../candidates/rex-omni.md) | MLLM 검출 | 텍스트·시각 | ? / **IDEA License 1.0 + Qwen Research** | 없음 | 공식 없음 | >1s, 8GB 빠듯 | ✕ | 참고 | 참고 | 참고 |
| [Qwen3-VL 2B/4B](../candidates/qwen3-vl.md) | MLLM 그라운딩 | 텍스트 | Apache / Apache | 없음 | 공식 없음 | >1s | ✕ | 참고 | ✕ | ✕ |
| [RF-DETR N/S/M](../candidates/rf-detr.md) | **폐쇄형, 학습** | 없음 | Apache / Apache(N~L)* | 라벨 수백 장 | T4 TRT M 4.4ms @576² | 0.005~0.03s | **◎** | ✕ | **◎** | **◎** |
| [D-FINE / DEIM](../candidates/d-fine.md) | 폐쇄형, 학습 | 없음 | Apache / Apache* (DEIMv2는 자체 License) | 라벨 수백 장 | T4 TRT S 3.49ms | 0.005~0.03s | ◎ | ✕ | ◎ | ◎ |
| [RT-DETRv2](../candidates/rt-detr.md) | 폐쇄형, 학습 | 없음 | Apache / Apache* | 라벨 수백 장 | T4 TRT S 217 FPS @640 | 0.005~0.03s | ○ | ✕ | ○ | ○ |
| [Grounding DINO 1.5/1.6](../candidates/grounding-dino-1.5-1.6.md) | 개방어휘 | 텍스트 | API 전용 | — | Edge Orin NX 10.7 FPS | 해당 없음(클라우드) | 참고 | 참고 | 참고 | 참고 |
| [DINO-X](../candidates/dino-x.md) | 개방어휘 + 마스크 | 텍스트·시각·prompt-free | API 전용 | — | Edge Orin NX 20.1 FPS | 해당 없음 | 참고 | 참고 | 참고 | 참고 |
| [T-Rex2](../candidates/t-rex2.md) | 시각 프롬프트 검출 | 예시 박스 + 텍스트 | API 전용 | — | 공개 없음 | 해당 없음 | 참고 | ✕ | 참고 | 참고 |

◎ 최적 · ○ 적합 · △ 가능하나 한계 · ✕ 부적합 · 참고 = 비교용(추천 제외)
`*` **Objects365 등 학습 데이터 약관 주의**: D-FINE README는 "Objects365로 (사전)학습한 체크포인트는 Objects365 약관을 따를 수 있으며 D-FINE 라이선스로 상용 허가됐다고 가정하지 말라"고 명시한다([D-FINE README](https://raw.githubusercontent.com/Peterande/D-FINE/master/README.md), 확인). Grounding DINO·MM-GDINO·LLMDet(O365/GoldG 등)·RF-DETR·RT-DETR의 O365 사전학습 가중치에도 같은 쟁점이 있을 수 있다(추론). 상용 데모 배포 전 법무 확인 대상.

목록에만 두고 카드를 쓰지 않은 것:
- **APE** (Apache-2.0 [LICENSE](https://raw.githubusercontent.com/shenyunhang/APE/main/LICENSE), 확인): detect+segment everything. Swin-L/ViT-L급이라 8GB·1초 예산에 무겁고, 2024.04 이후 활동 없음([README](https://raw.githubusercontent.com/shenyunhang/APE/main/README.md)). 감점.
- **CutLER** (클래스 무관 비지도 검출): 대부분 **CC-BY-NC**([README](https://raw.githubusercontent.com/facebookresearch/CutLER/main/README.md), 확인) → 상용 불가로 제외.
- **LW-DETR**: Apache-2.0([LICENSE](https://raw.githubusercontent.com/Atten4Vis/LW-DETR/main/LICENSE), 확인). RF-DETR이 LW-DETR 계열을 이어받아 더 활발하므로 RF-DETR 카드로 대표(추론: RF-DETR 논문이 LW-DETR 기반이라고 밝힘, 원문 미열람).
- **DEIMv2**: DINOv3 백본 실시간 검출기. 자체 "DEIMv2 License" + 상용은 문의 요구 → 감점([README](https://raw.githubusercontent.com/Intellindust-AI-Lab/DEIMv2/main/README.md), 확인). D-FINE 카드에 같이 적음.

## 3. 입력 모달리티

조사한 검출기는 **전부 [RGB]**다. 공개 사전학습 가중치 중 depth를 입력으로 받는 개방어휘 검출기는 찾지 못했다(확인하지 못함 — 검색 범위 내에서 없음).
- 폐쇄형 검출기(RF-DETR, D-FINE)를 **직접 학습**한다면 입력 채널을 바꾸는 것(예: RGB 중 1채널을 정규화 depth로, 또는 4채널 stem) 이론상 가능하나, 사전학습 가중치와의 불일치로 데이터가 더 필요하다(추론).
- 실용적인 결합은 검출 후 단계다: 박스 안 depth로 최상층·평면 검증(현 `core/plane.py`), 또는 depth 엣지로 붙은 박스 분리. → `input-modality.md`.

## 4. 시나리오별 추천 (검출기 관점)

### [BOX] 팔레트 박스 1/2/3종
- **1순위: RF-DETR-N/S(또는 M) 폐쇄형 학습** — Apache, T4 TRT 수 ms, 박스는 라벨링·합성이 쉬움. 1~3종이면 클래스 3개 또는 "box" 단일 클래스. 마스크까지 한 번에 원하면 RF-DETR-Seg(→ `segmenters.md`).
- 2순위: D-FINE-S/M (Apache, O365 약관 주의).
- 학습 없이: MM-GDINO/LLMDet "box. carton." — 박스 어휘는 개방어휘 모델의 강점 영역이지만 **같은 무늬 박스가 빈틈없이 붙은 팔레트**에서 여러 박스를 하나로 묶거나 쪼갤 위험이 크다(추론). depth 엣지와의 결합이 필수일 가능성이 높다.

### [UNK] 모르는 물건
- 검출기 기반 [DS]의 구조적 한계: 어휘가 없으면 박스를 못 낸다. 현 m2 약점("어휘 밖 물체 누락")과 같다.
- **1순위(최소 비용 실험): grounding-dino-tiny → MM-GDINO-T 또는 LLMDet-T 교체.** 같은 transformers `AutoModelForZeroShotObjectDetection` API라 코드 변경이 거의 없고(LLMDet은 transformers 4.55.0+, [LLMDet README](https://raw.githubusercontent.com/iSEE-Laboratory/LLMDet/main/README.md) 확인), LVIS val AP가 20.1 → 31.9(MM-GDINO-T)/34.9(LLMDet-T)로 올라 희귀 어휘 대응이 낫다(확인). "object." 같은 일반 어휘 recall도 오를지는 미확인.
- 2순위: OWLv2 — `objectness` 점수로 클래스 무관 제안이 가능(추론: transformers 출력에 objectness logits 존재, 원문 미확인).
- prompt-free가 있는 YOLOE-pf는 기능상 가장 맞지만 **AGPL**.
- 결론: [UNK]는 검출기보다 m1식 [PM]/AMG가 본령일 수 있다. 검출기는 "보조 박스 제안"으로 m1과 합치는 방향을 검토(추론).

### [ONE] 1종 제품
- **1순위: RF-DETR-N/S 단일 클래스 학습** — 라벨 수백 장(합성·SAM 보조 자동 라벨 가능, → `data/dataset-plan.md`).
- 학습 전 즉시 데모: **OWLv2 `image_guided_detection`** (예시 이미지 1장, Apache, [transformers owlv2 doc](https://raw.githubusercontent.com/huggingface/transformers/main/docs/source/en/model_doc/owlv2.md) 확인). 무늬 없는 반사 부품에서는 정확도 저하 우려(추론).
- YOLOE 시각 프롬프트는 성능·속도 모두 좋으나 AGPL. T-Rex2는 API 전용(참고).

### [FEW] 5종 이하
- **1순위: RF-DETR 다중 클래스 학습** (5클래스, 클래스당 수십~수백 인스턴스).
- 보조: OWLv2 예시 이미지 다중 쿼리(종마다 1~수 장)로 학습 데이터 자동 라벨링 → RF-DETR 학습의 부트스트랩으로 쓰는 2단계 구성(추론).
- 연성 파우치는 형상 변형이 커서 예시 1장 매칭보다 학습형이 유리할 것(추론).

## 5. 열린 질문

1. **m2의 검출 단계 시간**이 기록돼 있지 않다. Grounding DINO 대비 대안의 이득을 판단하려면 `predict()` 안에서 GD / SAM2 / 후처리를 분리 측정해야 한다(4060 추정치는 전부 추론).
2. MM-GDINO/LLMDet이 **"object." 같은 일반 어휘**에서 GD-T보다 recall이 나은가 — `data/eval/` GT 12장으로 바로 측정 가능.
3. Zivid 컬러(`rgba_srgb`)의 해상도·조명 특성에서 OWLv2(960² 고정 리사이즈)의 작은 물체 성능.
4. Objects365 사전학습 가중치의 상용 사용 가능 여부(데모 배포 범위에 따라 필요).
5. Ultralytics(YOLOE/YOLO-World Ultralytics판) AGPL — Zivid 내부 데모·고객 배포 시 Enterprise License 필요 여부.
6. 팔레트 박스(붙어 있는 동일 박스)에서 RGB 검출기만으로 인스턴스 분리가 되는가, depth 엣지가 필수인가.
7. DEIMv2 License 원문 미열람 — 상용 조건 확인 필요.
