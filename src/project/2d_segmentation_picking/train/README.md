# train — RF-DETR-Seg 학습·평가 (조사 3단계)

합성 데이터(`../synth/`, `data/synthetic/`)로 RF-DETR-Seg 를 학습하고 합성 테스트 세트에서 시나리오별로 평가한다.
방법 선정 근거는 `../research/candidates/rf-detr-seg.md`, 시나리오 정의는 `../research/scenarios/`.

## 환경

`venv_segpick`(SAM2·zivid)과 **분리된** 전용 venv 를 쓴다 (m1/m2 벤치마크 환경을 건드리지 않기 위해).

```
C:\Zivid\3rdparty\venv_rfdetr\Scripts\python.exe     # Python 3.12.10, torch 2.6.0+cu124, rfdetr 1.12.0
```

설치 순서는 `requirements.txt` 머리말 참조 (torch CUDA 빌드 먼저, 그다음 `-r requirements.txt`).

## 모델 구성 (사용자 확정 2026-11-08)

| 모델 | 클래스 | 학습 데이터 | 테스트 |
|---|---|---|---|
| BOX | B1~B5 (5종), 최상층만 정답 (`coco_top.json`, 아래층은 iscrowd) | box/train 1,350장 → train 1,215 / valid 135 | box/test 150 |
| 제품 | can, paper_box, bracket, flange, snack_bag, stick_pack, pouch (7종) | one/train + few/train 2,160장 → train 1,944 / valid 216 | one/test 140 + few/test 100 (평가는 따로 집계) |

valid 는 학습 세트에서 장면 단위로 10% 를 뗀다 (seed 0).

## 실행

```powershell
$PY = "C:\Zivid\3rdparty\venv_rfdetr\Scripts\python.exe"
# 1) RF-DETR 폴더 구조 생성 (이미지 복사 없음, json 의 file_name 이 절대 경로)
& $PY train/prepare_rfdetr.py box
& $PY train/prepare_rfdetr.py product
# 2) 본 학습 (BOX → 제품 순서, 끊기면 다시 실행하면 last.ckpt 에서 이어감, done.txt 있으면 건너뜀)
powershell -ExecutionPolicy Bypass -File train/run_train.ps1
# 3) 평가 → data/eval/rfdetr/<run>.json (+ _dets.json 예측 저장본)
& $PY train/eval_rfdetr.py --run runs/box_m     --dataset data/rfdetr/box
& $PY train/eval_rfdetr.py --run runs/product_m --dataset data/rfdetr/product
```

학습 설정: Medium(432 px), 최대 30 epoch, batch 2 × grad accum 8, lr 1e-4, early stop patience 5
(RF-DETR 기본 `early_stopping_min_delta=0.001`). 센서 노이즈(`synth/config/noise.yaml` rgb 절)는 학습 split 에만 on-the-fly.

## 결과 — [RGB] 기준 모델 (2026-11-09, RTX 4060 Laptop, 합성 테스트)

| 시나리오 | GT | recall | 과분할 | 과병합 (덮음 0.3/0.5/0.7) | 마스크 mAP50:95 | 박스 mAP50:95 | 추론 |
|---|---|---|---|---|---|---|---|
| [BOX] 150장 | 2,033 | 1.000 | 9 | 0 / 0 / 0 | 0.979 | 0.974 | 47 ms |
| [ONE] 140장 | 2,740 | 0.982 | 10 | 21 / 9 / 6 | 0.903 | 0.943 | 39 ms |
| [FEW] 100장 | 2,531 | 0.945 | 18 | 57 / 35 / 20 | 0.875 | 0.909 | 43 ms |

- 학습: BOX 25 epoch(2시간 42분, best epoch 22), 제품 24 epoch(3시간 33분, best epoch 21). 둘 다 early stop.
- 놓친 물체는 대부분 작은(가려진) 물체 — 면적 중앙값 1.3~4k px (전체 10k px). 최저는 [FEW] stick_pack 0.875.
- 합성끼리의 점수다. 실측 성능을 뜻하지 않는다. m1/m2 (실측 [UNK] 12장면)와 직접 비교 불가.
- 추론 시간: `model.predict` 1장(전처리+추론+후처리), 모델 로드 제외, 앞 3장 버림, `optimize_for_inference` 미사용.

함정: RF-DETR 분류 헤드는 클래스 수+1 출력이라 낮은 신뢰도에서 범위 밖 `class_id` 가 나온다 → 평가에서 제외.
`class_id = category_id - 1`.

## 다음: 입력 결합 비교 (사용자 확정 2026-11-09)

[RGB] vs [RGB+D] vs [RGB+N] vs [RGB+D+N] — BOX·제품 모델 각각, [RGB] 는 위 모델을 그대로 기준으로 쓴다 (신규 학습 6개).
depth·법선 노이즈는 `synth/add_sensor_noise.py apply_noise` 로 이미 실측 교정돼 있다 (on-the-fly 적용은 미구현).
구현 과제: 첫 입력층 채널 확장, 좌우 뒤집기 시 법선 x 부호 반전, depth 는 장면별 상대 높이로 정규화.

## 다른 PC 로 이전 (데스크탑 등)

1. 코드: `git clone https://github.com/GJ-Cho/cho-python-sample.git` → `git checkout 2d_segmentation_picking`
2. 데이터 복사 (외장 SSD·네트워크): `data/synthetic/` 전체 (약 7.6 GB). 같은 상대 경로에 둔다.
   - `data/rfdetr/` 는 복사하지 않는다 — json 안 이미지 경로가 절대 경로라 새 PC 에서 `prepare_rfdetr.py` 로 다시 만든다.
   - [RGB] 기준 모델: `runs/box_m/`, `runs/product_m/` 에서 `checkpoint_best_total.pth`(각 약 140 MB), `metrics.csv` 만.
3. venv: 위 "환경" 대로 `C:\Zivid\3rdparty\venv_rfdetr` 생성. 경로가 다르면 `run_train.ps1` 의 `$py` 를 고친다.
4. `prepare_rfdetr.py box` / `product` 실행.
5. 확인: `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"`,
   소량 시험 학습 `prepare_rfdetr.py box --limit 100 --name box_trial` →
   `train_rfdetr.py --dataset data/rfdetr/box_trial --model medium --epochs 2 --out runs/box_trial` 로 it/s 측정
   (노트북 RTX 4060 Laptop: Medium 1.9 it/s, epoch 약 6분).
6. 추론 시간 비교는 같은 PC 에서 재야 하므로 [RGB] 기준 모델도 새 PC 에서 `eval_rfdetr.py` 를 다시 돌린다 (재학습 불필요).
