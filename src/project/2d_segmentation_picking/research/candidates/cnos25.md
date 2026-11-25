# cnos25 (YOLOE + DINOv3)

> 담당 토픽: topics/dino-backbone.md · 작성일: 2026-10-29

## 요약

CNOS 파이프라인을 2025년 모델로 교체한 버전: 제안기 YOLOE-11L-seg, 매칭 DINOv3 ViT-L/16. BOP 2025 Track 7(model-free 2D 검출, BOP-H3)에서 **Best Fast Method·Best Open Source Method** 수상. 실사진 레퍼런스로 동작하고 커스텀 이미지 노트북(2025.12)을 제공해 우리 조건에 가장 가깝다.

## 분류

| 항목 | 값 |
|---|---|
| 파이프라인 유형 | [PM] |
| 입력 모달리티 | [RGB] |
| 맞는 시나리오 | [ONE] [FEW] — 실사진 레퍼런스(static onboarding), multi-reference·cross-image 지원 |
| 프롬프트 | 예시 이미지 |

## 라이선스

| | 라이선스 | 상용 가능 | 출처 |
|---|---|---|---|
| 코드 | MIT | 가능 | GitHub API `Vision-Kek/cnos25` (확인) |
| 가중치 | YOLOE: Ultralytics **AGPL-3.0**(추론, Ultralytics 배포 가중치 기준) / DINOv3: DINOv3 License | **감점 2건** | README 체크포인트 링크 (확인), 라이선스 해석은 `detectors.md` 참조 |

## 학습 데이터

없음("No training data used", 확인). 레퍼런스 사진(onboarding)에서 템플릿 디스크립터 추출.

## 속도

| 수치 | 하드웨어 | 해상도 | 출처 (확인/추론) |
|---|---|---|---|
| BOP-H3 model-free AP 0.441 / 0.134s | RTX 4090 | BOP-H3 | https://bop.felk.cvut.cz/sub_info/30902/ , 리더보드 (확인) |

RTX 4060 Laptop 8GB 추정: 0.3~0.4s (추론) — 4090 대비 2~3배 느림 가정. **[PM] 계열 중 0.5s 목표를 맞출 유일한 공개 후보.**

## 성숙도

- 공개 코드, 별 13개 수준의 소규모 레포, 마지막 push 2026-03 (확인). 연구 코드.
- DINOv3 가중치 신청 필요.

## 기준선(m1/m2) 대비

- 기대 이점: m1/m2보다 2배 이상 빠를 가능성. 레퍼런스 기반이라 어휘 밖 물체 누락(m2 약점) 없음.
- 위험: AGPL·커스텀 라이선스. YOLOE 제안의 클래스 무관 재현율이 빈피킹 밀집 장면에서 SAM보다 낮을 수 있음(추론). 제안기를 SAM2/EfficientViT-SAM 등으로 바꾸면 라이선스는 해결되나 속도 이점 감소.

## 출처

- https://github.com/Vision-Kek/cnos25 (README 원문: raw.githubusercontent.com/Vision-Kek/cnos25/main/README.md)
- https://bop.felk.cvut.cz/leaderboards/modelfree-detection-unseen-bop24/bop-h3/
