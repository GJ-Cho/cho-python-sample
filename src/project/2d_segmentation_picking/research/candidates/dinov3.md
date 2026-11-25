# DINOv3

> 담당 토픽: topics/dino-backbone.md · 작성일: 2026-10-29

## 요약

DINOv2 후속(2025.8, arXiv 2508.10104). Gram anchoring과 고해상도 적응으로 dense 특징 품질이 크게 올랐다. BOP 2025 model-free 2D 검출에서 cnos25(YOLOE+DINOv3)가 Fast/Open-source 상을 받았다. 라이선스가 Apache가 아닌 Meta 커스텀이라 감점.

## 분류

| 항목 | 값 |
|---|---|
| 파이프라인 유형 | [PM] 매칭 단계 / 학습형 백본 |
| 입력 모달리티 | [RGB] |
| 맞는 시나리오 | [ONE] [FEW] — DINOv2와 동일. 유사 클래스 구분은 오히려 떨어질 수 있음(아래) |
| 프롬프트 | 예시 이미지 |

## 라이선스

| | 라이선스 | 상용 가능 | 출처 |
|---|---|---|---|
| 코드 | DINOv3 License (Meta 커스텀, 2025-08-19) | 가능해 보임 — 법무 검토 필요 | https://raw.githubusercontent.com/facebookresearch/dinov3/main/LICENSE.md (확인) |
| 가중치 | DINOv3 License, **다운로드 신청 게이트** | 위와 동일 | dinov3 README (확인) |

확인한 조항: royalty-free·worldwide 사용/수정/재배포 허용, non-commercial·research-only 문구 없음, MAU 조항 없음. 제한: 재배포 시 계약서 동봉, 논문 사용 표기, ITAR·군사·핵·첩보 금지, Meta 상대 IP 소송 시 종료, 면책 조항, Meta 일방 변경 가능.

## 학습 데이터

DINOv2와 동일(없음 / 소수 마스크 / 수십~수백 장 파인튜닝, 추론)

## 속도

| 수치 | 하드웨어 | 해상도 | 출처 (확인/추론) |
|---|---|---|---|
| ViT-S 21M / S+ 29M / B 86M / L 300M / H+ 840M / 7B, 패치 16; ConvNeXt T/S/B/L 29/50/89/198M | — | — | README (확인) |
| cnos25 전체(YOLOE + DINOv3-L) 0.134s/img | RTX 4090 | HOT3D | https://bop.felk.cvut.cz/sub_info/30902/ (확인) |

RTX 4060 Laptop 8GB 추정: ViT-B 0.05~0.1s, ViT-L 0.15~0.3s (추론) — 패치 16이라 같은 해상도에서 DINOv2(14)보다 토큰 수가 약 25% 적다.

## 성숙도

- 공식 코드·가중치 공개(게이트), HF `transformers` ≥ 4.56, `timm` ≥ 1.0.20
- ConvNeXt 증류 모델은 TensorRT 변환이 ViT보다 쉬울 가능성(추론)

## 기준선(m1/m2) 대비

- 기대 이점: dense 품질↑(ADE20k linear: DINOv2-g 49.5 → DINOv3-7B 55.9, 확인), 고해상도 입력 안정.
- 위험: no-time-to-train 논문은 DINOv3의 매끄러운 특징이 **유사 클래스 구분력을 낮춘다**고 보고(https://arxiv.org/html/2507.02798v4, 확인). 같은 브랜드 다른 맛 같은 [FEW] 조건에서 v2보다 나쁠 수 있다. 라이선스 감점.

## 출처

- https://github.com/facebookresearch/dinov3 · https://arxiv.org/abs/2508.10104
- https://ai.meta.com/resources/models-and-libraries/dinov3-license
