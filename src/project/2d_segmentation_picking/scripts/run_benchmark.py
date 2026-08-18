"""Phase 5 — m1/m2 벤치마크: 정답 세트 대비 recall/과분할, 타이밍 (PLAN.md 7장/10장).

`data/eval/<scene>_gt.npz`(build_gt_from_groups.py 산출물) 대비, **현재 배포 설정**
그대로(config/m1_sam2.yaml, config/m2_grounded_sam.yaml)의 m1/m2를 여러 장면에
순차 실행해 recall과 과분할을 처음으로 숫자로 낸다.

지표 정의:
- recall = IoU>=0.5로 매칭된 정답 물체 수 / 정답 물체 수(라벨된 것 + 조각조차 없어
  완전 누락으로 기록된 것). 완전 누락은 어떤 방법으로도 못 찾을 것이 확실하므로
  분모에 포함해야 recall이 부풀려지지 않는다.
- 과분할 물체 수 = 정답 물체 하나에 IoU>=0.3인 후보가 2개 이상 매칭된 경우
  (한 물체가 여러 후보로 쪼개짐).
- 미매칭 후보 수 = 어떤 정답 물체와도 IoU>=0.5가 안 되는 후보. **주의**: 정답
  세트가 부분 라벨(사람이 애매하거나 귀찮은 조각은 건너뜀)이라 이 값을 그대로
  "오탐지"로 해석하면 안 된다 — 라벨 안 된 진짜 물체를 잡았을 수도 있다.

사용 (전용 venv python):
  python scripts/run_benchmark.py                    # 12장면 전부, m1+m2
  python scripts/run_benchmark.py --scenes 05,07      # 일부만
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_PROJECT_DIR = Path(__file__).resolve().parent.parent
_EVAL_DIR = _PROJECT_DIR / "data" / "eval"
sys.path.insert(0, str(_PROJECT_DIR))

_ALL_SCENES = [f"{i:02d}" for i in range(1, 13)]
_IOU_MATCH = 0.5
_IOU_PARTIAL = 0.3  # 과분할 판정용 — 부분 겹침도 "같은 물체를 가리킨다"로 센다


def _load_gt(scene: str):
    meta = json.loads((_EVAL_DIR / f"{scene}_gt.json").read_text(encoding="utf-8"))
    npz = np.load(_EVAL_DIR / f"{scene}_gt.npz")
    masks = [npz[f"obj{o['obj_id']}"] for o in meta["objects"]]
    return masks, meta["n_missing_objects"]


def _build_segmenter(method: str):
    import yaml

    if method == "m1":
        from methods.m1_sam2_toplayer import M1Sam2TopLayer as Cls
        cfg_path = _PROJECT_DIR / "config/m1_sam2.yaml"
    else:
        from methods.m2_grounded_sam import M2GroundedSam as Cls
        cfg_path = _PROJECT_DIR / "config/m2_grounded_sam.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    return Cls, cfg


def _evaluate(gt_masks: list[np.ndarray], n_missing: int, pred_masks: list[np.ndarray]) -> dict:
    from core.plane import bbox_of, mask_iou

    n_true = len(gt_masks) + n_missing
    pred_boxes = [bbox_of(p) for p in pred_masks]
    gt_boxes = [bbox_of(g) for g in gt_masks]

    matched = 0
    oversegmented = 0
    pred_best_iou = [0.0] * len(pred_masks)
    for g, gb in zip(gt_masks, gt_boxes):
        hits = 0
        best = 0.0
        for pi, (p, pb) in enumerate(zip(pred_masks, pred_boxes)):
            iou = mask_iou(g, p, box_a=gb, box_b=pb)
            pred_best_iou[pi] = max(pred_best_iou[pi], iou)
            if iou >= _IOU_PARTIAL:
                hits += 1
            best = max(best, iou)
        if best >= _IOU_MATCH:
            matched += 1
        if hits > 1:
            oversegmented += 1

    unmatched_pred = sum(1 for b in pred_best_iou if b < _IOU_MATCH)
    return {
        "n_true_objects": n_true,
        "n_gt_labeled": len(gt_masks),
        "n_missing_objects": n_missing,
        "n_matched": matched,
        "recall": round(matched / n_true, 3) if n_true else None,
        "n_oversegmented_gt": oversegmented,
        "n_candidates": len(pred_masks),
        "n_candidates_unmatched_iou0.5": unmatched_pred,
    }


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(description="Phase 5 벤치마크 — recall/과분할/타이밍")
    parser.add_argument("--scenes", default=",".join(_ALL_SCENES))
    parser.add_argument("--methods", default="m1,m2")
    parser.add_argument("--data-dir", type=Path,
                        default=_PROJECT_DIR / "data/input/20260811_piece_picking_data")
    parser.add_argument("--bin-roi", type=Path, default=_PROJECT_DIR / "config/bin_roi_pp.json")
    parser.add_argument("--output", type=Path, default=_EVAL_DIR / "benchmark_results.json")
    args = parser.parse_args()

    from core.loader import load_any

    scenes = args.scenes.split(",")
    methods = args.methods.split(",")

    results = []
    for method in methods:
        Cls, base_cfg = _build_segmenter(method)
        base_cfg["top_layer"]["bin_roi"] = str(args.bin_roi.resolve())
        segmenter = Cls(base_cfg, _PROJECT_DIR)
        t0 = time.time()
        segmenter.build()  # 방법당 1회만 로드 — 장면마다 새로 만들면 12배 느려진다
        build_dt = time.time() - t0

        for scene in scenes:
            gt_path = _EVAL_DIR / f"{scene}_gt.npz"
            if not gt_path.exists():
                print(f"[건너뜀] {scene}: GT 없음", file=sys.stderr)
                continue

            segmenter.cfg["top_layer"]["bin_frame"] = str(
                (_PROJECT_DIR / f"config/bin_frame_pp_{scene}.json").resolve())

            scene_path = args.data_dir / f"{scene}.zdf"
            scene_data = load_any(scene_path)

            t0 = time.time()
            candidates = segmenter.predict(scene_data)
            cold_dt = time.time() - t0
            t0 = time.time()
            candidates = segmenter.predict(scene_data)
            warm_dt = time.time() - t0

            gt_masks, n_missing = _load_gt(scene)
            metrics = _evaluate(gt_masks, n_missing, [c.mask for c in candidates])
            metrics.update({"scene": scene, "method": method,
                            "build_time_s": round(build_dt, 3),
                            "predict_time_s": round(cold_dt, 3),
                            "predict_time_warm_s": round(warm_dt, 3)})
            results.append(metrics)
            print(f"[{method}][{scene}] recall={metrics['recall']} "
                  f"matched={metrics['n_matched']}/{metrics['n_true_objects']} "
                  f"oversegmented={metrics['n_oversegmented_gt']} "
                  f"candidates={metrics['n_candidates']} warm={warm_dt:.2f}s", file=sys.stderr)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[저장] {args.output}", file=sys.stderr)

    # --- 방법별 집계 ---
    print("\n=== 방법별 집계 ===", file=sys.stderr)
    for method in methods:
        rows = [r for r in results if r["method"] == method]
        if not rows:
            continue
        total_true = sum(r["n_true_objects"] for r in rows)
        total_matched = sum(r["n_matched"] for r in rows)
        total_over = sum(r["n_oversegmented_gt"] for r in rows)
        total_cand = sum(r["n_candidates"] for r in rows)
        median_warm = sorted(r["predict_time_warm_s"] for r in rows)[len(rows) // 2]
        print(f"{method}: recall {total_matched}/{total_true} = "
              f"{total_matched/total_true:.3f}, 과분할 {total_over}개 물체, "
              f"후보 총 {total_cand}개, warm 중앙값 {median_warm:.2f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
