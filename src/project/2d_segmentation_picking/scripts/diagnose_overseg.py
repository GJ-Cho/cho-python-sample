"""과분할 진단 — 정답 물체 하나에 후보가 여러 개 매칭된 경우, 그 후보들의
기하 지표(plane_rms_all_mm, inlier_ratio, tilt_deg 등)를 나란히 보여준다.

reject 임계(min_inlier_ratio, max_plane_rms_all_mm)를 켤 근거를 찾기 위한
일회성 진단 스크립트. run_benchmark.py와 같은 방식으로 예측하되 candidate.meta
전체를 남긴다.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_PROJECT_DIR = Path(__file__).resolve().parent.parent
_EVAL_DIR = _PROJECT_DIR / "data" / "eval"
sys.path.insert(0, str(_PROJECT_DIR))


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    import yaml

    from core.loader import load_any
    from core.plane import bbox_of, mask_iou

    method = sys.argv[1] if len(sys.argv) > 1 else "m1"
    scenes = sys.argv[2].split(",") if len(sys.argv) > 2 else \
        [f"{i:02d}" for i in range(1, 13)]

    if method == "m1":
        from methods.m1_sam2_toplayer import M1Sam2TopLayer as Cls
        cfg_path = _PROJECT_DIR / "config/m1_sam2.yaml"
    else:
        from methods.m2_grounded_sam import M2GroundedSam as Cls
        cfg_path = _PROJECT_DIR / "config/m2_grounded_sam.yaml"

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg["top_layer"]["bin_roi"] = str((_PROJECT_DIR / "config/bin_roi_pp.json").resolve())
    segmenter = Cls(cfg, _PROJECT_DIR)
    segmenter.build()

    data_dir = _PROJECT_DIR / "data/input/20260811_piece_picking_data"
    for scene in scenes:
        gt_json = _EVAL_DIR / f"{scene}_gt.json"
        if not gt_json.exists():
            continue
        meta = json.loads(gt_json.read_text(encoding="utf-8"))
        gt_npz = np.load(_EVAL_DIR / f"{scene}_gt.npz")
        gt_masks = [gt_npz[f"obj{o['obj_id']}"] for o in meta["objects"]]

        segmenter.cfg["top_layer"]["bin_frame"] = str(
            (_PROJECT_DIR / f"config/bin_frame_pp_{scene}.json").resolve())
        scene_data = load_any(data_dir / f"{scene}.zdf")
        candidates = segmenter.predict(scene_data)

        cand_boxes = [bbox_of(c.mask) for c in candidates]
        for gi, g in enumerate(gt_masks):
            gb = bbox_of(g)
            hits = []
            for c, cb in zip(candidates, cand_boxes):
                iou = mask_iou(g, c.mask, box_a=gb, box_b=cb)
                if iou >= 0.3:
                    hits.append((iou, c))
            if len(hits) > 1:
                print(f"\n[{method}][scene{scene}] obj{gi} (fragments={meta['objects'][gi]['fragment_ids']}) "
                      f"→ 후보 {len(hits)}개 매칭:")
                hits.sort(key=lambda h: -h[0])
                for iou, c in hits:
                    m = c.meta
                    print(f"  iou={iou:.2f} area={m.get('area_px')} "
                          f"inlier_ratio={m.get('inlier_ratio')} "
                          f"plane_rms_all_mm={m.get('plane_rms_all_mm')} "
                          f"depth_span_mm={m.get('depth_span_mm')} "
                          f"tilt_deg={m.get('tilt_deg')} "
                          f"merged_from={m.get('merged_from')}")


if __name__ == "__main__":
    main()
