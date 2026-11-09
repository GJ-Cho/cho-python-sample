"""합성 COCO(synth/build_coco.py 산출물) → RF-DETR 데이터셋 폴더 (이미지 복사 없음).

RF-DETR 은 <dataset>/{train,valid,test}/_annotations.coco.json 구조를 요구하고, 이미지를
Path(split 폴더) / file_name 으로 연다 (rfdetr/datasets/coco.py CocoDetection._decode_image).
file_name 을 절대 경로로 쓰면 Path 결합이 절대 경로를 그대로 돌려주므로 원본 장면 폴더의
rgb.png 를 복사 없이 읽는다. 마스크는 비압축 RLE 그대로 쓴다 (convert_coco_poly_to_mask 가 RLE 지원).

  python train/prepare_rfdetr.py box                 # [BOX] B1~B5, 최상층만(coco_top.json)
  python train/prepare_rfdetr.py product             # [ONE]+[FEW] 합친 7종
  python train/prepare_rfdetr.py box --limit 100 --name box_trial   # 시험 학습용 소량

valid 는 학습 세트에서 장면 단위로 떼어 낸다 (--val-frac, 고정 seed).
product 의 test 는 one/test + few/test 를 합친 것이다. 시나리오별 평가는 따로 한다.
"""
import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
SYN = os.path.join(PROJ, "data", "synthetic")
OUT = os.path.join(PROJ, "data", "rfdetr")

SOURCES = {
    "box": {"train": [("box/train", "coco_top.json")], "test": [("box/test", "coco_top.json")]},
    "product": {"train": [("one/train", "coco.json"), ("few/train", "coco.json")],
                "test": [("one/test", "coco.json"), ("few/test", "coco.json")]},
}


def load_merged(parts):
    """여러 COCO 파일을 하나로 합친다. file_name 은 절대 경로, id 는 새로 매긴다."""
    images, anns, cats = [], [], None
    for rel, fname in parts:
        root = os.path.join(SYN, rel)
        d = json.load(open(os.path.join(root, fname), encoding="utf-8"))
        if cats is None:
            cats = d["categories"]
        assert d["categories"] == cats, f"카테고리 불일치: {rel}"
        idmap = {}
        for im in d["images"]:
            new = dict(im, id=len(images), file_name=os.path.abspath(os.path.join(root, im["file_name"])),
                       source=rel)
            for k in ("depth_file", "normals_file"):
                if k in new:
                    new[k] = os.path.abspath(os.path.join(root, im[k]))
            idmap[im["id"]] = new["id"]
            images.append(new)
        for a in d["annotations"]:
            anns.append(dict(a, id=len(anns) + 1, image_id=idmap[a["image_id"]]))
    return images, anns, cats


def subset(images, anns, cats, keep_ids):
    keep = set(keep_ids)
    return {"images": [im for im in images if im["id"] in keep],
            "annotations": [a for a in anns if a["image_id"] in keep],
            "categories": cats}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=list(SOURCES))
    ap.add_argument("--name", default=None, help="출력 폴더 이름 (기본 = kind)")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--limit", type=int, default=0, help="학습 장면 수 상한 (시험용)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    out = os.path.join(OUT, a.name or a.kind)
    tr_imgs, tr_anns, cats = load_merged(SOURCES[a.kind]["train"])
    te_imgs, te_anns, te_cats = load_merged(SOURCES[a.kind]["test"])
    assert te_cats == cats

    rng = np.random.default_rng(a.seed)
    ids = rng.permutation([im["id"] for im in tr_imgs])
    n_val = max(1, int(round(len(ids) * a.val_frac)))
    val_ids, train_ids = ids[:n_val], ids[n_val:]
    test_ids = [im["id"] for im in te_imgs]
    if a.limit:
        train_ids = train_ids[:a.limit]
        val_ids = val_ids[:max(1, a.limit // 10)]
        test_ids = test_ids[:max(1, a.limit // 10)]

    splits = {"train": subset(tr_imgs, tr_anns, cats, train_ids),
              "valid": subset(tr_imgs, tr_anns, cats, val_ids),
              "test": subset(te_imgs, te_anns, cats, test_ids)}
    for name, d in splits.items():
        os.makedirs(os.path.join(out, name), exist_ok=True)
        json.dump(d, open(os.path.join(out, name, "_annotations.coco.json"), "w", encoding="utf-8"))
        n_gt = sum(1 for x in d["annotations"] if not x.get("iscrowd"))
        print(f"{out}/{name}: images={len(d['images'])} gt={n_gt} crowd={len(d['annotations']) - n_gt}")


if __name__ == "__main__":
    main()
