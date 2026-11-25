# Grounding DINO 1.0 (Tiny / Base) — 현 m2 기준선

> 담당 토픽: topics/detectors.md · 작성일: 2026-10-29

## 요약

텍스트 프롬프트로 박스를 내는 개방어휘 검출기. 현재 m2 앞단(`IDEA-Research/grounding-dino-tiny`, transformers)이며 모든 대안의 비교 기준이다.

## 분류

| 항목 | 값 |
|---|---|
| 파이프라인 유형 | [DS] 앞단 |
| 입력 모달리티 | [RGB] |
| 맞는 시나리오 | [UNK] △(어휘 밖 물체 누락), [BOX] △(붙어 있는 같은 박스 분리 약함, 추론), [ONE]/[FEW] △(제품 고유 외관을 텍스트로 표현하기 어려움) |
| 프롬프트 | 텍스트 |

## 라이선스

| | 라이선스 | 상용 가능 | 출처 |
|---|---|---|---|
| 코드 | Apache-2.0 | 예 | [LICENSE](https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/LICENSE) (확인) |
| 가중치 | 별도 명시 없음 → 레포 Apache-2.0으로 봄 | 대체로 예. O365/GoldG/Cap4M 학습 데이터 약관 쟁점 가능(추론) | [README](https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/README.md) (확인: 가중치 라이선스 문구 없음) |

## 학습 데이터

없음 (zero-shot). T는 O365+GoldG+Cap4M으로 사전학습했다 (README, 확인).

## 속도

| 수치 | 하드웨어 | 해상도 | 출처 (확인/추론) |
|---|---|---|---|
| m2 전체(GD-T + SAM2-S + 후처리) warm 0.96 / 1.22s | RTX 4060 Laptop 8GB | Zivid 2+ MR130 컬러 | 프로젝트 README·research/README 기준선 (확인) |
| GD-B는 GD-T보다 약 2.5배 느림 | 동일 | 동일 | `config/m2_grounded_sam.yaml` 주석 (확인) |

RTX 4060 Laptop 8GB 추정: GD-T 단독 0.1~0.25s (추론). m2 전체 ~1s 중 SAM2 다중 박스 디코딩과 후처리가 상당 부분이라는 가정이다. **단계별 실측 분리가 필요하다.**

## 성숙도

- 가중치 공개. transformers로 컴파일 없이 실행(현재 사용 중). 원본 레포는 커스텀 CUDA op 빌드가 필요하다.
- 성능: COCO zero-shot 48.4(T), LVIS minival 28.8 / val 20.1(T) ([MM-GDINO README 비교표](https://raw.githubusercontent.com/open-mmlab/mmdetection/main/configs/mm_grounding_dino/README.md), 확인).

## 기준선(m1/m2) 대비

- 기대 이점: (기준선 자체)
- 위험: 어휘 의존. recall을 위해 box threshold를 0.15까지 낮춰 중복 박스가 급증한다(NMS 필수). 같은 계열의 개선판(MM-GDINO, LLMDet)이 같은 API로 있다.

## 출처

- https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/README.md
- arXiv 2303.05499 (Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection)
