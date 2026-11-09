"""학습한 RF-DETR-Seg 모델을 합성 테스트 세트에서 시나리오별로 평가한다 (venv_rfdetr).

  python train/eval_rfdetr.py --run runs/box_m     --dataset data/rfdetr/box
  python train/eval_rfdetr.py --run runs/product_m --dataset data/rfdetr/product   # ONE/FEW 따로 집계

지표 (scripts/run_benchmark.py 와 같은 정의 — 비교 가능하게 유지):
- recall      = IoU>=0.5 로 매칭된 정답 물체 / 정답 물체 (iscrowd 제외)
- 과분할      = 정답 하나에 IoU>=0.3 후보가 2개 이상
- 과병합      = 후보 하나가 정답 2개 이상을 각각 덮음비율(교집합/정답 면적) >= 0.5 로 덮음 (0.3/0.7 도 기록)
- 미매칭 후보 = 어떤 정답과도 IoU>=0.5 가 안 되는 후보. [BOX] 는 정답이 최상층만이라 아래층 박스를
  잡은 후보도 여기 들어간다 (그래서 오탐지 수가 아니다).
- 위 지표는 신뢰도 >= --conf 후보로 계산. mAP 는 pycocotools COCOeval (segm/bbox, 상위 100개, iscrowd 무시 영역).
- 추론 시간 = model.predict(threshold=--conf) 1장 (전처리+추론+후처리, GPU 동기화). 앞 --warmup 장은 버리고
  나머지 중앙값. 모델 로드 시간은 제외. optimize_for_inference 는 쓰지 않은 기본 경로다.

예측(신뢰도 >= 0.05, 상위 100개)은 <out>_dets.json (COCO results, 압축 RLE) 로 저장해 후처리 비교에 재사용한다.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

PROJ = Path(__file__).resolve().parent.parent
IOU_MATCH, IOU_PARTIAL, MERGE_COVERS = 0.5, 0.3, (0.3, 0.5, 0.7)
SAVE_MIN_CONF, MAX_DETS = 0.05, 100


def decode(seg, h, w):
    if isinstance(seg, dict) and isinstance(seg["counts"], list):
        seg = mask_utils.frPyObjects(seg, h, w)
    elif isinstance(seg, list):
        seg = mask_utils.merge(mask_utils.frPyObjects(seg, h, w))
    return mask_utils.decode(seg).astype(bool)


def instance_metrics(gt, pred, dev):
    """gt, pred: (N,H,W) bool numpy. run_benchmark._evaluate 와 같은 정의를 GPU 행렬곱으로 계산."""
    ng, npd = len(gt), len(pred)
    out = {"n_gt": ng, "n_pred": npd}
    if ng == 0 or npd == 0:
        out.update(n_matched=0, n_overseg=0, n_unmatched_pred=npd,
                   **{f"n_merged_{c}": 0 for c in MERGE_COVERS})
        return out
    g = torch.from_numpy(gt.reshape(ng, -1)).to(dev, torch.float16)
    p = torch.from_numpy(pred.reshape(npd, -1)).to(dev, torch.float16)
    inter = (p @ g.T).float()                                   # (npd, ng) 픽셀 수 (fp32 누적)
    ga, pa = g.float().sum(1), p.float().sum(1)
    iou = inter / (pa[:, None] + ga[None, :] - inter).clamp(min=1)
    cover = inter / ga.clamp(min=1)[None, :]
    out["n_matched"] = int((iou.max(0).values >= IOU_MATCH).sum())
    out["n_overseg"] = int(((iou >= IOU_PARTIAL).sum(0) >= 2).sum())
    out["n_unmatched_pred"] = int((iou.max(1).values < IOU_MATCH).sum())
    for c in MERGE_COVERS:
        out[f"n_merged_{c}"] = int(((cover >= c).sum(1) >= 2).sum())
    return out


def coco_eval(gt_json, dets, img_ids, iou_type):
    if not dets:
        return {"mAP50_95": 0.0, "mAP50": 0.0}
    cg = COCO()
    cg.dataset = gt_json
    cg.createIndex()
    cd = cg.loadRes(dets)
    ev = COCOeval(cg, cd, iou_type)
    ev.params.imgIds = img_ids
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    return {"mAP50_95": round(float(ev.stats[0]), 4), "mAP50": round(float(ev.stats[1]), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="학습 출력 폴더 (runs/box_m)")
    ap.add_argument("--weights", default="checkpoint_best_total.pth")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", default="RFDETRSegMedium")
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None, help="기본 data/eval/rfdetr/<run 이름>")
    a = ap.parse_args()

    import rfdetr
    run = (PROJ / a.run).resolve()
    gt_json = json.load(open(PROJ / a.dataset / "test" / "_annotations.coco.json", encoding="utf-8"))
    cats = sorted(c["id"] for c in gt_json["categories"])
    model = getattr(rfdetr, a.model)(pretrain_weights=str(run / a.weights), num_classes=len(cats))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    images = gt_json["images"][:a.limit] if a.limit else gt_json["images"]
    anns_by_img = {}
    for an in gt_json["annotations"]:
        anns_by_img.setdefault(an["image_id"], []).append(an)

    rows, dets = [], []
    for k, im in enumerate(images):
        pil = Image.open(im["file_name"]).convert("RGB")
        h, w = im["height"], im["width"]
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.predict(pil, threshold=a.conf)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t_inf = time.perf_counter() - t0

        det = model.predict(pil, threshold=0.0)  # 지표용 전체 후보 (신뢰도 내림차순 정렬)
        # 분류 헤드는 클래스 수+1 출력이라 낮은 신뢰도에서 범위 밖 id(배경 슬롯)가 나온다 → 제외
        valid = np.flatnonzero(det.class_id < len(cats))
        order = valid[np.argsort(-det.confidence[valid])][:MAX_DETS]
        conf, cls, masks, boxes = det.confidence[order], det.class_id[order], det.mask[order], det.xyxy[order]
        for i in np.flatnonzero(conf >= SAVE_MIN_CONF):
            rle = mask_utils.encode(np.asfortranarray(masks[i].astype(np.uint8)))
            rle["counts"] = rle["counts"].decode("ascii")
            x0, y0, x1, y1 = boxes[i].tolist()
            dets.append({"image_id": im["id"], "category_id": cats[int(cls[i])], "score": float(conf[i]),
                         "segmentation": rle, "bbox": [x0, y0, x1 - x0, y1 - y0]})

        gts = [an for an in anns_by_img.get(im["id"], []) if not an.get("iscrowd")]
        gt_masks = np.stack([decode(an["segmentation"], h, w) for an in gts]) if gts else np.zeros((0, h, w), bool)
        keep = conf >= a.conf
        m = instance_metrics(gt_masks, masks[keep], dev)
        m.update(image_id=im["id"], source=im.get("source", ""), t_infer_s=round(t_inf, 4))
        rows.append(m)
        if k % 25 == 0:
            print(f"[{k + 1}/{len(images)}] {im.get('source')} gt={m['n_gt']} pred={m['n_pred']} "
                  f"matched={m['n_matched']} t={t_inf * 1000:.0f}ms", flush=True)

    summary = {}
    for src in sorted({r["source"] for r in rows}):
        rs = [r for r in rows if r["source"] == src]
        ids = [r["image_id"] for r in rs]
        times = [r["t_infer_s"] for r in rs if rows.index(r) >= a.warmup]
        tot = lambda key: sum(r[key] for r in rs)  # noqa: E731
        sub_dets = [d for d in dets if d["image_id"] in set(ids)]
        summary[src] = {
            "images": len(rs), "gt": tot("n_gt"), "matched": tot("n_matched"),
            "recall": round(tot("n_matched") / max(tot("n_gt"), 1), 4),
            "overseg_gt": tot("n_overseg"),
            "merged_pred": {str(c): tot(f"n_merged_{c}") for c in MERGE_COVERS},
            "pred": tot("n_pred"), "unmatched_pred": tot("n_unmatched_pred"),
            "segm": coco_eval(gt_json, sub_dets, ids, "segm"),
            "bbox": coco_eval(gt_json, sub_dets, ids, "bbox"),
            "t_infer_median_s": round(float(np.median(times)), 4) if times else None,
        }
    out = Path(a.out) if a.out else PROJ / "data" / "eval" / "rfdetr" / run.name
    out.parent.mkdir(parents=True, exist_ok=True)
    settings = {"run": str(run), "weights": a.weights, "dataset": a.dataset, "conf": a.conf,
                "warmup": a.warmup, "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu"}
    json.dump({"summary": summary, "settings": settings, "rows": rows},
              open(f"{out}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(dets, open(f"{out}_dets.json", "w", encoding="utf-8"))
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
