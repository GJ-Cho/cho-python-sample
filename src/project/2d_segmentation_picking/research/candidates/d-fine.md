# D-FINE (+ DEIM / DEIMv2 참고)

> 담당 토픽: topics/detectors.md · 작성일: 2026-10-29

## 요약

RT-DETR 계열 실시간 폐쇄형 DETR(Fine-grained Distribution Refinement). Apache-2.0, T4 TRT 수 ms. RF-DETR의 대안으로 [BOX]/[ONE]/[FEW] 학습형 검출기 후보. DEIM은 D-FINE의 학습 기법 개선, DEIMv2는 DINOv3 백본판이다.

## 분류

| 항목 | 값 |
|---|---|
| 파이프라인 유형 | [DS] 앞단 |
| 입력 모달리티 | [RGB] |
| 맞는 시나리오 | [BOX]/[ONE]/[FEW] ◎(학습 데이터 제작 전제), [UNK] ✕ |
| 프롬프트 | 없음 (폐쇄형) |

## 라이선스

| | 라이선스 | 상용 가능 | 출처 |
|---|---|---|---|
| 코드 | Apache-2.0 | 예 | [README](https://raw.githubusercontent.com/Peterande/D-FINE/master/README.md) (확인) |
| 가중치 | 레포 Apache-2.0. 단 README가 "Objects365로 (사전)학습한 체크포인트는 Objects365 약관을 따를 수 있으며 상용 허가를 가정하지 말라"고 명시 | COCO 전용 가중치는 예, O365 가중치는 확인 필요 | 같은 README (확인) |
| DEIMv2 | 자체 "DEIMv2 License", 상용은 이메일 문의 | 감점 | [DEIMv2 README](https://raw.githubusercontent.com/Intellindust-AI-Lab/DEIMv2/main/README.md) (확인, LICENSE.md 원문 미열람) |

## 학습 데이터

라벨 수백 장 이상(추론). README는 O365 사전학습 체크포인트에서 fine-tune을 권하지만 "카테고리가 아주 단순하면 과적합할 수 있다"고 적는다 (확인).

## 속도

| 수치 | 하드웨어 | 해상도 | 출처 (확인/추론) |
|---|---|---|---|
| N 2.12ms / S 3.49ms / M 5.62ms / L 8.07ms / X 12.89ms (COCO 42.8~55.8) | T4, TensorRT 10.4 FP16, bs1 | 기본 640 (추론: README에 명시 없음) | README (확인) |
| DEIMv2 S 5.78ms (COCO 50.9) ~ X 13.75ms (57.8) | 하드웨어 미표기 | — | DEIMv2 README (확인) |

RTX 4060 Laptop 8GB 추정: TRT 5~10ms, PyTorch eager 20~50ms (추론, RF-DETR 카드와 같은 근거).

## 성숙도

- 코드·가중치 공개, ONNX/TensorRT 배포 스크립트 있음(추론: 레이턴시를 TRT로 측정).
- 학습 도구는 RF-DETR보다 덜 친화적(설정 파일 중심, 추론).

## 기준선(m1/m2) 대비

- 기대 이점: RF-DETR과 같음(수 ms 검출, 어휘 불필요).
- 위험: O365 가중치 약관. 학습 파이프라인 구성 비용이 RF-DETR보다 클 수 있다.

## 출처

- https://raw.githubusercontent.com/Peterande/D-FINE/master/README.md
- https://raw.githubusercontent.com/Intellindust-AI-Lab/DEIMv2/main/README.md
