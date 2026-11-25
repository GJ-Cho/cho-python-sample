# OWLv2 (OWL-ViT v2)

> 담당 토픽: topics/detectors.md · 작성일: 2026-10-29

## 요약

Google의 CLIP 기반 개방어휘 검출기. **텍스트뿐 아니라 예시 이미지로도 질의(`image_guided_detection`)** 할 수 있고 Apache-2.0이라, 학습 없이 [ONE]/[FEW]를 시도할 수 있는 몇 안 되는 상용 가능 후보다.

## 분류

| 항목 | 값 |
|---|---|
| 파이프라인 유형 | [DS] 앞단 |
| 입력 모달리티 | [RGB] |
| 맞는 시나리오 | [ONE]/[FEW] ○(예시 이미지 질의), [UNK] ○(objectness 점수로 클래스 무관 제안, 추론), [BOX] △ |
| 프롬프트 | 텍스트 / 예시 이미지 |

## 라이선스

| | 라이선스 | 상용 가능 | 출처 |
|---|---|---|---|
| 코드 | Apache-2.0 (transformers, google-research/scenic) | 예 | (추론: 레포 라이선스 기준) |
| 가중치 | Apache-2.0 | 예 | [HF 모델 카드](https://huggingface.co/google/owlv2-base-patch16-ensemble) (확인) |

## 학습 데이터

없음 ~ 레퍼런스 이미지 몇 장(예시 질의). fine-tune도 가능하나 도구 지원이 약하다(추론).

## 속도

| 수치 | 하드웨어 | 해상도 | 출처 (확인/추론) |
|---|---|---|---|
| OWLv2 base TensorRT 약 400ms+ (사용자 보고, GPU 불명) | Jetson 계열로 보임 | 900 | [NVIDIA forum](https://forums.developer.nvidia.com/t/nanoowl-inference-takes-more-time-than-nanoowl-official-github-shows/293051) (일화) |
| OWL-ViT v1보다 눈에 띄게 느림 | — | — | [HF forum](https://discuss.huggingface.co/t/owlv2-significantly-slower-than-owlvit/73223) (일화) |

입력 960×960(B/16, [Qualcomm AI Hub](https://aihub.qualcomm.com/compute/models/owlv2) 기준, 논문 표는 미확인).
RTX 4060 Laptop 8GB 추정: 0.15~0.4s (추론). ViT-B/16 @960²는 토큰 3600개로 GD-T보다 무겁다. FP16·SDPA로 줄일 여지가 있다.

## 성숙도

- 가중치 공개. transformers `Owlv2ForObjectDetection.image_guided_detection` 지원 ([transformers doc 원문](https://raw.githubusercontent.com/huggingface/transformers/main/docs/source/en/model_doc/owlv2.md), 확인).
- NVIDIA NanoOWL로 TensorRT 경로가 있다(주로 OWL-ViT v1 대상, 추론).

## 기준선(m1/m2) 대비

- 기대 이점: 학습 없이 "이 제품"을 예시 1장으로 지정한다. m2와 같은 [DS] 구조라 SAM2 뒷단을 그대로 쓴다.
- 위험: 무늬 없는 반사 부품·변형 파우치에서 예시 유사도가 불안정할 수 있다(추론). 960² 리사이즈로 작은 물체 손실. 속도.

## 출처

- https://huggingface.co/google/owlv2-base-patch16-ensemble
- arXiv 2306.09683 (Scaling Open-Vocabulary Object Detection)
