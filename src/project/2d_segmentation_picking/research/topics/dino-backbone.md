# DINO / DINOv2 / DINOv3 활용 — [PM] 매칭, few-shot, 검출 백본

> 작성일: 2026-10-29 · 담당 카드: `candidates/dinov2.md`, `dinov3.md`, `cnos.md`, `cnos25.md`, `sam-6d-ism.md`,
> `nids-net.md`, `noctis.md`, `no-time-to-train.md`, `matcher.md`, `persam.md`, `l2g-det.md`
> SAM 계열·SAM3 exemplar는 `segmenters.md`, Grounding DINO·YOLOE·RF-DETR 검출기 본체는 `detectors.md` 소관이다. 여기서는 교차 참조만 한다.

## 0. 결론 먼저

| 시나리오 | 1순위 | 2순위 | 근거 한 줄 |
|---|---|---|---|
| [ONE] | **[PM] 부트스트랩 → RF-DETR-Seg 학습** (DINOv2 백본, Apache-2.0) | m1 제안 + **DINOv2 마스크 임베딩 매칭**(CNOS식, 자체 구현) | [PM]은 데이터 0장으로 바로 동작, 학습형은 0.5s 목표를 여유 있게 맞춤 (추론) |
| [FEW] | 위와 동일 (클래스 ≤5) | **no-time-to-train** 방식 메모리뱅크(SAM2+DINOv2, Apache-2.0) | 여러 제품의 구분은 DINO 임베딩이 강점. 단 같은 제품의 인접 인스턴스 분리는 DINO가 못 함 |
| [UNK] | DINO는 **보조 신호**로만 (기존 법선 병합에 "외관 유사도" 조건 추가) | — | 같은 제품 인접 인스턴스를 DINO는 구분 못 함 → 단독 병합 기준으로 쓰면 과병합 (추론, ISAR 관찰과 일치) |

**가장 중요한 한 줄**: DINO 특징은 "무슨 물건인가(클래스·제품 종류)"를 아주 잘 구분하지만 "몇 개인가(같은 제품 두 개의 경계)"는 거의 구분하지 못한다. 그래서 [PM]에서 **인스턴스 분리 품질은 전적으로 앞단 제안기(SAM/FastSAM/YOLOE/GDINO)가 결정**하고, DINO는 필터·라벨링만 한다. m1의 약점(과분할)은 DINO 매칭을 붙여도 그대로 남는다. (추론)

## 1. DINO vs DINOv2 vs DINOv3

| | DINO (2021) | DINOv2 (2023) | DINOv3 (2025.8) |
|---|---|---|---|
| 학습 | ImageNet 자기지도 | LVD-142M 큐레이션 데이터, iBOT+DINO 손실 | LVD-1689M, 7B teacher + **Gram anchoring** + 고해상도 적응 단계 |
| 패치 | 8/16 | 14 | 16 |
| 크기 | ViT-S/B, ResNet-50 | ViT-S 21M / B 86M / L 300M / g 1.1B (+registers 버전) | ViT-S 21M / S+ 29M / B 86M / L 300M / H+ 840M / 7B 6.7B, ConvNeXt T 29M / S 50M / B 89M / L 198M |
| 코드 라이선스 | Apache-2.0 | Apache-2.0 | **DINOv3 License (Meta 커스텀)** |
| 가중치 라이선스 | Apache-2.0 | Apache-2.0 | **DINOv3 License**, 다운로드 신청(게이트) 필요 |
| HF 지원 | 있음 | `transformers` 정식 | `transformers` ≥ 4.56, `timm` ≥ 1.0.20 |

출처: DINO 레포 라이선스 Apache-2.0 확인(GitHub API, `facebookresearch/dino`) · DINOv2 README "code and model weights are released under the Apache License 2.0" 확인(https://raw.githubusercontent.com/facebookresearch/dinov2/main/README.md) · DINOv3 모델 표·게이트·HF 버전 확인(https://raw.githubusercontent.com/facebookresearch/dinov3/main/README.md)

### 1.1 무엇이 바뀌었나 (dense feature 관점)

- **DINOv2**: 패치 특징이 대응(correspondence)·분할에 바로 쓸 만해졌고, CNOS/SAM-6D/NIDS-Net 등 [PM] 계열 거의 전부가 DINOv2를 쓴다. **registers 버전**은 attention map의 고노름 아티팩트를 줄여 dense 특징이 깨끗하다. (확인: dinov2 README의 registers 모델 표)
- **DINOv3**: 긴 학습에서 dense 특징이 무너지는 문제를 **Gram anchoring**(패치 간 유사도 행렬을 초기 teacher에 묶는 정칙화)으로 해결하고, 512/768 crop으로 **고해상도 적응**을 추가했다. ADE20k linear mIoU: DINOv2 ViT-g 49.5 → DINOv3 7B 55.9, NYUv2 depth RMSE 0.372 → 0.309 (확인: https://arxiv.org/html/2508.10104v1). 증류된 ViT-S/B/L의 DINOv2 대비 수치는 표 헤더를 확인하지 못했다.
- **주의(확인)**: no-time-to-train 논문은 DINOv3가 "더 매끄러운 특징이라 유사 클래스 구분력이 떨어진다"고 보고했다(https://arxiv.org/html/2507.02798v4 4.3절). 즉 분할용 dense 품질 ↑가 곧 **인스턴스 매칭 구분력 ↑는 아니다**. 반대로 cnos25는 DINOv3 ViT-L로 BOP 2025 model-free 2D 검출 Fast/Open-source 상을 받았다. → 우리 데이터에서 v2/v3 둘 다 측정해야 한다.

### 1.2 DINOv3 라이선스 (검증)

**확인**(https://raw.githubusercontent.com/facebookresearch/dinov3/main/LICENSE.md, 2025-08-19판):
- Apache가 **아니다**. "DINOv3 License"라는 Meta 커스텀 계약이다.
- 1.a: "non-exclusive, worldwide, non-transferable and royalty-free limited license … to use, reproduce, distribute, copy, create derivative works of, and make modifications". **정의·조항 어디에도 non-commercial/research-only 제한이 없다.** Llama식 MAU 조항도 없다.
- 제한: 재배포 시 계약서 동봉(1.b.i), 논문 발표 시 사용 표기(1.b.ii), ITAR·군사·핵·첩보 용도 금지(1.b.v), Meta 상대 IP 소송 시 종료, 면책(indemnify) 조항, Meta가 약관 변경 가능.
- 판단(추론): **상용 사용은 가능해 보이나 커스텀 계약이라 법무 검토가 필요**하다. README 규칙상 Apache/MIT보다 한 단계 감점. 고객 배포물에 가중치를 포함하면 계약서 동봉 의무가 생긴다. 웹의 "CC BY-NC" 주장(dinov3.pages.dev)은 비공식 사이트라 무시.

### 1.3 8GB 노트북 GPU 속도 (추론)

근거: ViT-B/16 224² ≈ 17.6 GFLOPs, ViT-L ≈ 61 GFLOPs. RTX 4060 Laptop FP16 실효 10~20 TFLOPS 가정. 참고 실측: DINOv2-B ≈ 40ms on RTX A4000(해상도 미상, https://arxiv.org/pdf/2508.00746 검색 요약 — 정확도 낮음).

| 사용 방식 | ViT-S | ViT-B | ViT-L |
|---|---|---|---|
| 마스크 crop 50개 배치(224²)로 CLS 임베딩 | ~15ms | ~40–80ms | ~150–300ms |
| 전체 이미지 1회(≈1024px, 패치 풀링) | ~20ms | ~50–100ms | ~150–300ms |
| VRAM(FP16 가중치) | <0.1GB | ~0.2GB | ~0.6GB |

- **전체 이미지 1회 + 마스크 아래 패치 평균**(NIDS-Net FFA, no-time-to-train 방식)이 crop 방식보다 마스크 수와 무관해 유리하다. 단 패치 14/16px라 작은 물체(<3×3 패치)의 임베딩이 배경에 오염된다.
- SAM2-L과 함께 올려도 VRAM은 여유(no-time-to-train: DINOv2-L 1.13GiB + SAM2 856MiB, 확인). 단 그 논문의 피크 11.5GiB는 20클래스×10샷 메모리뱅크 기준이라 [FEW] 5클래스면 훨씬 작다(추론).

## 2. [PM] Propose→Match 파이프라인 비교

공통 구조: ① 클래스 무관 마스크 제안 → ② 각 마스크의 DINO 임베딩 → ③ 레퍼런스(템플릿) 임베딩과 코사인 유사도 → ④ 임계·NMS.

| 방법 | 제안기 | 매칭 | 레퍼런스 | 학습 | 코드 라이선스 | BOP AP / 시간 | 비고 |
|---|---|---|---|---|---|---|---|
| CNOS | SAM / FastSAM | DINOv2 CLS(+patch) | **CAD 렌더** 또는 BOP24 onboarding 사진 | 없음 | MIT | Core 42.8 / 0.22s(FastSAM) | 원조. FastSAM은 AGPL |
| SAM-6D ISM | SAM / FastSAM | 의미+외관+기하 점수 | **CAD 필수**(기하 점수) | 없음 | 명시 없음 | Core 47.1 / 0.45s(FastSAM) | 기하 점수가 CAD 의존 |
| NIDS-Net | GDINO("objects") + SAM | DINOv2 FFA + 선택적 adapter | **실사진 + 마스크** 가능 | adapter 수 분(선택) | MIT | Core 46.9 / 0.49s | 실사진 레퍼런스 친화 |
| NOCTIS | Grounded-SAM 2 | DINOv2 CLS+patch, cyclic threshold | CAD 렌더(BOP), 커스텀 USB 데모 | 없음 | MIT | 논문 주장 BOP23 SOTA(리더보드 미등재) | 반복 무늬 오매칭 완화 |
| **cnos25** | **YOLOE-11L-seg** | **DINOv3 ViT-L** | **실사진(static onboarding)**, 커스텀 노트북 | 없음 | MIT (YOLOE는 AGPL) | **H3 model-free 44.1 / 0.134s (RTX 4090)** | BOP 2025 Fast·Open-source 수상 |
| **no-time-to-train** | SAM2 (32×32 grid) | DINOv2-L 메모리뱅크 + semantic-aware 매칭 | **실사진 + 마스크/박스** | 없음 | **Apache-2.0** | COCO-FSOD 30-shot nAP 36.8 / 0.93s(A100) | 제품 종류 구분용으로 가장 깔끔한 라이선스 |
| Matcher | SAM | DINOv2 대응 + 점 샘플링 | 1장 + 마스크 | 없음 | BSD-2 **학술만, 상업은 문의** | 120s/img(20클래스, A100) | 느림·라이선스로 탈락 |
| PerSAM(-F) | SAM(프롬프트 자동) | SAM 자체 특징 location prior | 1장 + 마스크 | 없음 / 2파라미터 10초 | MIT | — | 기본은 단일 인스턴스, 멀티는 커뮤니티 스크립트 |
| L2G-Det (2026.3) | 제안 없음, DINOv3 패치 대응 → SAM | DINOv3-L | 실사진 소수 + 마스크 | adapter + 객체 토큰(물체당 합성 500장) | 미확인 | HR-InsDet 76.2 vs NIDS-Net 63.9 | 최신, 코드 공개 여부 미확인 |

출처: BOP 2024 보고서 Track 3/5/7 표(https://arxiv.org/html/2504.02812v2), BOP 리더보드(https://bop.felk.cvut.cz/leaderboards/detection-unseen-bop23/bop-classic-core/, …/bop-industrial/, …/modelfree-detection-unseen-bop24/bop-h3/), cnos25 제출 정보(https://bop.felk.cvut.cz/sub_info/30902/), 각 레포 README·GitHub API 라이선스. BOP 시간은 참가자마다 다른 PC라 직접 비교 불가(보고서 명시).

### 2.1 CAD 필요 여부 — 우리는 "직접 찍은 레퍼런스 사진"

- **CAD 없이 실사진으로 가능(확인)**: NIDS-Net, cnos25(static onboarding), no-time-to-train, Matcher, PerSAM, L2G-Det, CNOS의 BOP24 model-free 모드.
- **CAD 사실상 필수**: SAM-6D ISM(기하 점수에 CAD 점군 사용), CNOS 기본 모드, NOCTIS 기본 설정(PBR 렌더 템플릿).
- 레퍼런스 사진 만들기(추론): Zivid 2D로 제품을 턴테이블/손으로 돌려 **면별 6~12장 + 마스크**(마스크는 SAM2 클릭 1번으로 생성). [ONE] 1종 ≈ 10분, [FEW] 5종 ≈ 1시간. 데모 장면과 **같은 조명·같은 카메라**로 찍는 것이 도메인 격차를 가장 줄인다.

### 2.2 BOP-Industrial 신호 — 무늬 없는/반사 산업 부품

BOP-Industrial(무늬 없는 산업 부품 데이터셋 묶음) 리더보드: CNOS 0.27~0.33, MUSE 0.34, SAM6D-FastSAM 0.42, 1위 3PT-Detection 0.58(2.1s) (확인). → **텍스처 없는 부품에서 DINO 매칭은 체급이 크게 떨어진다.** 부품끼리 모양만 다르고 색·무늬가 같으면 DINO 임베딩이 거의 같아 구분이 안 된다(추론). 이 유형은 학습형(RF-DETR-Seg) 또는 depth 기하 특징이 필요하다 → `input-modality.md` 교차 참조.

### 2.3 제품 유형별 적합도 (추론)

| 유형 | DINO [PM] 적합도 | 이유 |
|---|---|---|
| 강체 소비재(병·캔·소형 포장) | **높음** | 라벨·인쇄 무늬가 강한 판별 특징. 단 같은 브랜드 다른 맛(색만 다름)은 v3에서 혼동 가능 |
| 무늬 없음·반사 산업 부품 | **낮음** | 2.2 참조. 반사 하이라이트가 임베딩을 흔듦 |
| 연성 파우치 | 중간 | 인쇄면은 잘 맞으나 접힘·주름으로 시점별 외관 변화 큼 → 레퍼런스 장수를 늘려야 함. SAM 과분할(주름 경계)이 더 문제 |

### 2.4 [ONE]/[FEW] 순위 (적합도)

1. **자체 구현 "m1/m2 제안 + DINOv2 FFA 매칭"** — 기존 `Segmenter` 인터페이스에 매칭 단계만 추가. 라이선스 Apache, 의존성 최소. NIDS-Net·no-time-to-train의 FFA/메모리뱅크 로직을 참고 구현. (추론)
2. **no-time-to-train** — 실사진 레퍼런스, Apache-2.0, 다중 클래스. 단 SAM2 32×32 grid라 느림(A100 0.93s → 4060L 2s+ 추론).
3. **NIDS-Net** — 실사진, MIT, BOP 검증. GDINO("objects") 제안이라 m2와 같은 계열의 누락 위험.
4. **cnos25** — 가장 빠름(4090 0.134s), DINOv3. 단 YOLOE(Ultralytics AGPL-3.0)·DINOv3 커스텀 라이선스 이중 감점. 제안기만 교체하면 유지 가능.
5. CNOS/SAM-6D ISM/NOCTIS — CAD 전제라 우리 조건에서 후순위. Matcher/PerSAM — 속도·멀티 인스턴스·라이선스로 탈락.

SAM3 exemplar 프롬프트와의 관계: cnos25 저자는 "SAM3는 cross-image exemplar를, YOLOE는 multi-reference를 지원하지 않는다, CNOS 방식이 그 대안"이라고 적었다(확인: https://raw.githubusercontent.com/Vision-Kek/cnos25/main/README.md, 저자 의견). SAM3 실제 지원 범위는 `segmenters.md` 결과로 확인할 것.

## 3. DINOv2/v3를 백본으로 한 학습형 검출·분할

- **RF-DETR / RF-DETR-Seg**: DINOv2 ViT 백본(Atto/Femto/Pico는 PE-Core-T). N/S/M/L 검출과 Seg 전 크기 **Apache-2.0**, Atto/Femto/Pico/XL/2XL 검출은 PML 1.0. RF-DETR-Seg-M COCO AP 45.3, T4 TensorRT FP16 5.9ms (확인: https://raw.githubusercontent.com/roboflow/rf-detr/develop/README.md). 세부는 `detectors.md`/`segmenters.md` 소관.
- DINOv3 공식 레포에는 7B 백본 + Mask2Former(ADE20k), 7B 검출기(COCO) 헤드가 있지만 7B라 8GB에서 비실용(확인/추론).
- **필요 라벨 수 (추론)**:
  - 마스크 임베딩 kNN/linear probe로 **"이 마스크가 어느 제품인가" 분류만**: 제품당 5~20개 마스크로 충분. DINO 임베딩은 소수 샷 kNN 분류가 강점.
  - **RF-DETR-Seg 파인튜닝(인스턴스 분리까지 학습)**: [ONE] 50~150장, [FEW] 클래스당 50~100장(총 200~500장)이 출발점. RF100-VL 같은 소규모 도메인 데이터셋 벤치마크가 이 체급을 겨냥한다.
- **권장 흐름 (추론)**: [PM]으로 자동 라벨 생성 → 사람이 검수(틀린 것만 수정) → RF-DETR-Seg 학습. [PM]은 "라벨링 도구"로, 학습형은 "운영 모델"로 쓰면 0장 시작과 0.5s 목표를 둘 다 만족한다.

## 4. [UNK]·m1 과분할 병합에 DINO가 도움이 되나

**결론(추론)**: 단독 병합 기준으로는 **해롭다**, 기존 기하 병합의 **보조 조건**으로는 도움이 될 수 있다.

- 과분할의 전형: 병의 "라벨 영역 / 뚜껑 / 투명 몸통", 파우치의 "주름 양쪽". 이 조각들은 외관이 **다르다** → DINO 유사도가 낮게 나와 병합 신호를 주지 못한다. ISAR 벤치마크도 "DINOv2는 여러 부분으로 된 물체에서 SAM이 부분마다 마스크를 내면 약하다"고 보고(검색 요약 https://arxiv.org/pdf/2311.02734).
- 반대로 DINO 유사도가 높은 인접 쌍은 대부분 **같은 제품의 서로 다른 인스턴스**다(빈피킹의 기본 상황) → 유사도로 병합하면 **과병합**. 현재 m1의 normal 병합이 m2에서 일부러 꺼진 이유(PLAN 11장, 동일 평면 물체 과병합)와 같은 함정이다.
- 그래서 쓸 수 있는 자리: (a) 기하(법선·오프셋)로는 병합 후보인데 외관이 **완전히 다른** 쌍(박스 위에 놓인 다른 물체 등)을 **병합 거부**하는 조건, (b) 기하 판정이 애매한 쌍의 tie-breaker. 둘 다 GT 12장으로 효과를 측정해야 한다.
- 관련 연구(확인, 검색 요약): ProMerge(자기지도 특징 + 겹침으로 반복 병합), FM-Fusion(의미 유사도 + 3D IoU 동시 조건), SAM-CP(학습형 병합 프롬프트). CutLER(DINO 기반 비지도 인스턴스 발견)는 **CC-BY-NC**라 제외(확인: CutLER README).

## 5. 전체 [PM] 지연 시간 vs 0.5s / 1s

| 구성 | 제안 | DINO 매칭 | 후처리 | 합계(4060L, 추론) | 판정 |
|---|---|---|---|---|---|
| m1(SAM2 AMG 최상층) + DINOv2-B FFA | 기존 ~0.9s | ~0.05–0.1s | 기존 | **~1.0–1.1s** | 1s 경계, 0.5s 불가 |
| m2(GDINO+SAM2) + DINOv2-B FFA | 기존 ~0.9s | ~0.05–0.1s | 기존 | ~1.0s | 1s 경계 |
| no-time-to-train 원본(SAM2-L 1024 grid) | — | — | — | ~2–3s | 탈락(그대로는) |
| cnos25(YOLOE + DINOv3-L) | ~0.1s | ~0.2s | — | **~0.3–0.4s** (4090 0.134s × 2~3배) | 0.5s 가능, 라이선스 감점 |
| 학습형 RF-DETR-Seg-M | — | — | — | **~0.02–0.05s** (T4 TRT 5.9ms, PyTorch는 수 배) | 여유 |

근거(추론): 4060 Laptop은 4090 데스크톱 대비 대략 1/2~1/3 처리량. 기존 m1/m2 수치는 README 기준선(0.96~1.39s) 재사용.
→ **[PM]만으로 0.5s를 맞추려면 제안기를 SAM2 AMG보다 빠른 것으로 바꿔야 한다**(YOLOE/FastSAM은 AGPL, 대안 탐색은 `segmenters.md`). 매칭(DINO) 자체는 병목이 아니다.

## 6. 가장 큰 불확실성

1. DINOv2 vs v3 중 어느 쪽이 **우리 제품의 인스턴스 매칭**에 나은지 — 문헌이 엇갈린다(no-time-to-train: v3 구분력↓, cnos25: v3로 수상). 실측 필요.
2. 4060 Laptop 속도는 전부 추정. DINO-B FFA 단계만 먼저 측정하면 표 5의 불확실성 대부분이 사라진다.
3. DINOv3 커스텀 라이선스의 상용 해석 — 법무 확인 전에는 v2(Apache)를 기본값으로.
4. 무늬 없는/반사 부품은 DINO [PM]의 약점이 리더보드로 확인됐지만, 우리 부품에서의 실제 격차는 미측정.
