# CropFormer (Entity Segmentation, EntitySeg)

> 담당 토픽: topics/segmenters.md · 작성일: 2026-10-29

## 요약

"High-Quality Entity Segmentation"(ICCV 2023 Oral)의 모델. **클래스 무관 엔티티 분할** — 물체를 부분이 아닌 "엔티티" 단위로 겹침 없이 분할하도록 학습돼, SAM AMG의 과분할 문제와 정확히 반대 성향이다. Mask2Former 기반 + 고해상도 크롭 융합. 그러나 **CC BY-NC 4.0**.

## 분류

| 항목 | 값 |
|---|---|
| 파이프라인 유형 | [E2E] (클래스 무관) / [PM] 앞단 |
| 입력 모달리티 | [RGB] |
| 맞는 시나리오 | [UNK] (비교·연구용) |
| 프롬프트 | 없음 |

## 라이선스

| | 라이선스 | 상용 가능 | 출처 |
|---|---|---|---|
| 코드 | **CC BY-NC 4.0** | **불가** | 확인 — https://raw.githubusercontent.com/qqlu/Entity/main/LICENSE |
| 가중치 | HF dataset `qqlu1992/Adobe_EntitySeg`에 동봉, EntitySeg 이미지 저작권은 원 출처 귀속 | **불가로 간주** | 확인 — https://raw.githubusercontent.com/qqlu/Entity/main/Entityv2/README.md |

## 학습 데이터

없음 (EntitySeg 학습 가중치).

## 속도

공식 지연 수치 확인하지 못함. Mask2Former + 크롭 분기 → Swin 백본 기준 수백 ms 이상 (추론).

RTX 4060 Laptop 8GB 추정: 0.3~1s, Swin-L/Hornet-L은 VRAM 빠듯 (추론).

## 성숙도

- detectron2 기반 비공식 확장 레포, 마지막 push 2023-11. Windows 설치 부담.

## 기준선(m1/m2) 대비

- 기대 이점: 엔티티 단위 비중첩 출력 → 과분할·중복이 구조적으로 적음.
- 위험: **비상업** → 추천 제외. "엔티티 수준 분할이 이 장면에서 얼마나 나은가"를 보는 상한 참고치로만.

## 출처

- https://github.com/qqlu/Entity · https://arxiv.org/abs/2211.05776 · https://github.com/adobe-research/EntitySeg-Dataset
