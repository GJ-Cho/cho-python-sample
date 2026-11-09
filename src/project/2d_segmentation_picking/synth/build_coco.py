"""장면 폴더들(inst.png + meta.json) → COCO instance segmentation JSON (일반 Python, numpy + opencv).

  python build_coco.py <root> [--out <json>] [--top-only] [--min-px 150]

- 마스크는 비압축 RLE (column-major, pycocotools 호환 형식).
- image.file_name 은 <scene>/rgb.png. depth.png / normals.png 는 같은 폴더에 있다.
- --top-only: [BOX]에서 최상층(노출) 박스만 정답으로 쓴다. 나머지는 iscrowd=1 (무시 영역)로 둔다.
"""
import argparse
import json
import os

import cv2
import numpy as np

BOX_CATS = {k: f"B{k}" for k in range(1, 6)}  # 택배 상자 1~5호
PRODUCT_CATS = {1: "can", 2: "paper_box", 3: "bracket", 4: "flange", 5: "snack_bag", 6: "stick_pack", 7: "pouch"}


def rle(mask):
    """비압축 RLE (column-major). 0 구간부터 시작하므로 첫 픽셀이 1이면 counts 앞에 0 이 붙는다.
    값이 바뀌는 위치를 numpy 로 한 번에 찾는다 (예전 itertools.groupby 픽셀 루프는 장당 ~4.4 s)."""
    flat = mask.ravel(order="F").astype(bool)
    change = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    bounds = np.concatenate([[0], change, [flat.size]])
    counts = np.diff(bounds).tolist()
    if flat[0]:
        counts.insert(0, 0)
    return {"counts": counts, "size": list(mask.shape)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", default=None)
    ap.add_argument("--top-only", action="store_true")
    ap.add_argument("--min-px", type=int, default=0)
    a = ap.parse_args()
    scenes = sorted(s for s in os.listdir(a.root) if os.path.exists(os.path.join(a.root, s, "meta.json")))
    images, anns = [], []
    cats = None
    for img_id, s in enumerate(scenes):
        meta = json.load(open(os.path.join(a.root, s, "meta.json"), encoding="utf-8"))
        if cats is None:
            cats = BOX_CATS if meta["scenario"] == "BOX" else PRODUCT_CATS
        inst = cv2.imread(os.path.join(a.root, s, "inst.png"), cv2.IMREAD_UNCHANGED)
        images.append(dict(id=img_id, file_name=f"{s}/rgb.png", width=meta["width"], height=meta["height"],
                           depth_file=f"{s}/depth.png", normals_file=f"{s}/normals.png", K=meta["K"]))
        for ins in meta["instances"]:
            if ins["visible_px"] < a.min_px:
                continue
            m = (inst == ins["id"]).astype(np.uint8)
            crowd = int(a.top_only and not ins.get("top", True))
            anns.append(dict(id=len(anns) + 1, image_id=img_id, category_id=ins["category_id"],
                             segmentation=rle(m), area=int(m.sum()), bbox=ins["bbox_xywh"], iscrowd=crowd))
    out = a.out or os.path.join(a.root, "coco_top.json" if a.top_only else "coco.json")
    json.dump(dict(images=images, annotations=anns,
                   categories=[dict(id=k, name=v) for k, v in (cats or {}).items()]), open(out, "w"))
    print(f"{out}: images={len(images)} annotations={len(anns)}")


if __name__ == "__main__":
    main()
