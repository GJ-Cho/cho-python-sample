# RepViT-SAM

> 담당 토픽: topics/segmenters.md · 작성일: 2026-10-29

## 요약

THU-MIG의 RepViT(CNN)로 SAM 인코더를 증류한 경량 SAM. EdgeSAM과 같은 "순수 CNN 경량 SAM" 계열이지만 **Apache-2.0**이라 상용 대안이 된다.

## 분류

| 항목 | 값 |
|---|---|
| 파이프라인 유형 | [DS] 뒷단 / AMG 가능(segment-anything 인터페이스, 추론) |
| 입력 모달리티 | [RGB] |
| 맞는 시나리오 | 속도 우선 [DS] 뒷단 |
| 프롬프트 | 박스·점 |

## 라이선스

| | 라이선스 | 상용 가능 | 출처 |
|---|---|---|---|
| 코드 | Apache-2.0 | 가능 | 확인 — https://raw.githubusercontent.com/THU-MIG/RepViT/main/LICENSE |
| 가중치 | 별도 표기 없음 → Apache-2.0 | 가능 (추론) | — |

## 학습 데이터

없음.

## 속도

| 수치 | 하드웨어 | 해상도 | 출처 (확인/추론) |
|---|---|---|---|
| 인코더 48.9ms (iPhone 12), MobileSAM 대비 ~10배 빠름 주장 | iPhone 12 / M1 Pro, Core ML | 1024 | 확인 — https://raw.githubusercontent.com/THU-MIG/RepViT/main/sam/README.md |

RTX 4060 Laptop 8GB 추정: 인코더 10~20ms (추론). GPU 데스크톱 수치 공개본 확인하지 못함.

## 성숙도

- 코드·가중치·CoreML 공개. ONNX는 SAM 디코더 경로 재사용 가능 (추론). 2024 이후 정체.

## 기준선(m1/m2) 대비

- 기대 이점: EdgeSAM급 속도를 Apache로.
- 위험: 공개 정확도 비교가 모바일 중심. EfficientViT-SAM 대비 GPU/TensorRT 문서 부족 → 2순위 경량 후보.

## 출처

- https://github.com/THU-MIG/RepViT/tree/main/sam · https://arxiv.org/abs/2312.05760
