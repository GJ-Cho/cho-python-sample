"""RF-DETR-Seg 학습 (venv_rfdetr). 센서 노이즈는 학습 split 에만 on-the-fly 로 입힌다.

  python train/train_rfdetr.py --dataset data/rfdetr/box --model small --epochs 30 --out runs/box_s

노이즈: synth/add_sensor_noise.py apply_rgb_noise() (config: synth/config/noise.yaml 의 rgb 절).
RF-DETR 코드는 고치지 않고, 이 스크립트에서 CocoDetection._decode_image 를 감싸 학습 split
(폴더 이름 train) 의 이미지에만 노이즈를 넣는다. valid/test 는 깨끗한 원본 그대로다.
Windows DataLoader 워커는 spawn 이라 이 모듈을 다시 import 하므로, 패치는 모듈 최상단에서 건다.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
sys.path.insert(0, str(PROJ / "synth"))

from add_sensor_noise import apply_rgb_noise  # noqa: E402
from rfdetr.datasets import coco as _rf_coco  # noqa: E402

_NOISE_R = yaml.safe_load(open(PROJ / "synth" / "config" / "noise.yaml", encoding="utf-8"))["rgb"]
_NOISE_ON = os.environ.get("SEGPICK_NOISE", "1") == "1"
_orig_decode = _rf_coco.CocoDetection._decode_image


def _decode_with_noise(self, image_id):
    img, scale = _orig_decode(self, image_id)
    if _NOISE_ON and Path(self.root).name == "train":
        rng = np.random.default_rng()  # 매 호출 다른 노이즈 (epoch 마다 다르게)
        img = Image.fromarray(apply_rgb_noise(np.asarray(img.convert("RGB")), _NOISE_R, rng))
    return img, scale


_rf_coco.CocoDetection._decode_image = _decode_with_noise

MODELS = {"nano": "RFDETRSegNano", "small": "RFDETRSegSmall", "medium": "RFDETRSegMedium",
          "large": "RFDETRSegLarge"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", default="small", choices=list(MODELS))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8, help="batch × grad_accum = 유효 배치 (공식 권장 16)")
    ap.add_argument("--resolution", type=int, default=None, help="미지정 = 모델 기본값")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--no-noise", action="store_true")
    ap.add_argument("--early-stop", type=int, default=0, help="patience (0 = 끔)")
    a = ap.parse_args()
    if a.no_noise:
        os.environ["SEGPICK_NOISE"] = "0"  # 워커(spawn)에도 전달
        global _NOISE_ON
        _NOISE_ON = False

    import rfdetr
    kw = {} if a.resolution is None else {"resolution": a.resolution}
    model = getattr(rfdetr, MODELS[a.model])(**kw)
    out = (PROJ / a.out).resolve() if not os.path.isabs(a.out) else Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[train] model={a.model} dataset={a.dataset} noise={'off' if a.no_noise else 'on'} out={out}", flush=True)
    t0 = time.time()
    model.train(dataset_dir=str((PROJ / a.dataset).resolve()), output_dir=str(out), epochs=a.epochs,
                batch_size=a.batch, grad_accum_steps=a.grad_accum, lr=a.lr, num_workers=a.workers,
                resume=a.resume, early_stopping=a.early_stop > 0,
                early_stopping_patience=max(a.early_stop, 1), progress_bar="tqdm", seed=0)
    print(f"[train] done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
