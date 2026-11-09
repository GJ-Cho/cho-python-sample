# SAM 2 / SAM 2.1 (tiny · small · base_plus · large)

> 담당 토픽: topics/segmenters.md · 작성일: 2026-10-29

## 요약

Meta의 promptable segmenter. 점·박스·마스크 프롬프트 → 마스크. 현재 m1(AMG, 최상층 프롬프트)과 m2(Grounding DINO 박스 → 마스크)의 뒷단이며 **기준선**이다. 이 조사에서는 "크기 변경·AMG 가속·뒷단 유지" 선택지를 평가하기 위한 기준 카드다.

## 분류

| 항목 | 값 |
|---|---|
| 파이프라인 유형 | [DS] 뒷단 (박스 프롬프트) / [PM] 앞단 (AMG 제안) |
| 입력 모달리티 | [RGB] |
| 맞는 시나리오 | [UNK] AMG 제안기로 유일한 학습 0 경로 · [BOX][ONE][FEW] 검출기 뒷단으로 무난 |
| 프롬프트 | 박스·점·마스크 / 없음(AMG) |

## 라이선스

| | 라이선스 | 상용 가능 | 출처 |
|---|---|---|---|
| 코드 | Apache-2.0 | 가능 | 확인 — https://raw.githubusercontent.com/facebookresearch/sam2/main/README.md |
| 가중치 | Apache-2.0 (체크포인트 명시) | 가능 | 확인 — 같은 README "License" 절 |

## 학습 데이터

없음 (SA-V/SA-1B 사전학습). 도메인 미세조정은 가능하나 필수 아님.

## 속도

| 수치 | 하드웨어 | 해상도 | 출처 (확인/추론) |
|---|---|---|---|
| tiny 91.2 / small 84.8 / base_plus 64.1 / large 39.5 FPS (파라미터 38.9/46/80.8/224.4M) | A100, torch 2.5.1 | 1024 | 확인 — sam2 README (비디오 기준 FPS) |
| m1 (small, AMG 163~232 프롬프트 + 후처리) warm 1.01 / 1.39s | RTX 4060 Laptop 8GB, bf16 autocast | Zivid 2+ MR130 크롭 | 확인 — PLAN.md 10장 |
| m2 (GD-tiny + small) warm 0.92 / 1.08s | 동일 | 동일 | 확인 — PLAN.md 10장 |

RTX 4060 Laptop 8GB 추정: 이미지 인코더 1회 ≈ 50~100ms (small), 박스 프롬프트 디코더는 박스당 수 ms (추론) — A100 대비 Ada 노트북 GPU 연산량 1/4~1/6 가정. **m1 시간의 대부분은 인코더가 아니라 프롬프트 수에 비례하는 디코더 + AMG 후처리**(PLAN 10장 "시간이 프롬프트 수에 비례")로 보인다.

## AMG 동작과 과분할

- `SAM2AutomaticMaskGenerator` 파라미터: `points_per_side`, `points_per_batch`(기본 64), `pred_iou_thresh`, `stability_score_thresh`, `box_nms_thresh`, `crop_n_layers`, `min_mask_region_area`, `use_m2m`, `multimask_output` — 확인 (https://raw.githubusercontent.com/facebookresearch/sam2/main/sam2/automatic_mask_generator.py).
- 점 하나는 본질적으로 모호(부분/전체/그룹)하므로 `multimask_output=True` + NMS는 **부분 마스크와 전체 마스크를 함께 살리는 구조**다. 라벨·인쇄·테이프·반사 하이라이트가 있는 표면에서 과분할이 생기는 원인 (추론, m1 관찰과 일치).
- 가속 여지 (추론): ① `points_per_batch`를 VRAM 허용 범위에서 키우기(128~256), ② `crop_n_layers=0` 유지, ③ `use_m2m=False` 유지, ④ 프롬프트를 격자 대신 검출기 박스/깊이 클러스터 중심으로 대체(MobileSAMv2식 object-aware prompting), ⑤ 인코더 ONNX/TensorRT — Hiera-S ONNX 내보내기 공개본 존재(https://huggingface.co/vietanhdev/segment-anything-2.1-onnx-models), RTX 수치는 공개된 것 없음.

## 성숙도

- 공식 코드·가중치 공개, 로컬 실행 중(venv_segpick). 마지막 push 2026-05 (GitHub API). 2024-12 이후 이미지 쪽 기능 업데이트 없음.
- `torch.compile`은 비디오 경로(`vos_optimized=True`) 위주.

## 기준선(m1/m2) 대비

- 기대 이점: tiny로 내리면 인코더 약간 단축, large는 경계 품질↑ 대신 4060에서 2~3배 느림 (추론).
- 위험: 크기 변경으로는 **과분할 구조 문제는 안 바뀐다**. 개선은 프롬프트 설계·granularity 제어(→ unsamv2.md)·후처리 쪽이다.

## 출처

- https://github.com/facebookresearch/sam2 · https://arxiv.org/abs/2408.00714
