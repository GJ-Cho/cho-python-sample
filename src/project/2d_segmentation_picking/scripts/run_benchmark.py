"""Phase 5 / 연구 4단계 — 세그멘테이션 방법 벤치마크: recall·과분할·과병합·구간별 시간.

`data/eval/<scene>_gt.npz`(build_gt_from_groups.py 산출물) 대비, 등록된 방법(_METHODS)을
배포 설정 그대로 여러 장면에 실행한다. 새 방법은 _METHODS에 한 줄 추가하면 된다
(methods/base.py Segmenter 인터페이스를 따르면 됨).

측정 규칙 (AGENTS.md "측정·비교 규칙"):
- 방법 하나만 로드한 독립 프로세스에서 잰다. --methods 에 여러 개를 주면 방법마다
  이 스크립트를 자식 프로세스로 다시 실행하고 결과만 합친다 (8GB VRAM 압박으로 warm이
  부풀었던 문제, PLAN.md 11장).
- 장면마다 cold 1회는 버리고 warm을 --repeats 회 재서 중앙값을 쓴다.
- 구간 시간은 방법이 Segmenter.stats 에 남긴 t_*_s (core/timing.py, GPU 동기화 포함).

지표:
- recall = IoU>=0.5로 매칭된 정답 물체 수 / 정답 물체 수(라벨된 것 + 조각조차 없어
  완전 누락으로 기록된 것). 완전 누락은 분모에 포함해야 recall이 부풀려지지 않는다.
- 과분할 물체 수 = 정답 물체 하나에 IoU>=0.3인 후보가 2개 이상 (한 물체가 여러 후보로 쪼개짐).
- 과병합 후보 수 = 후보 하나가 서로 다른 정답 물체 2개 이상을 각각 "덮음 비율"
  (교집합 / 정답 물체 면적) >= --merge-cover 이상 덮는 경우. 0.3/0.5/0.7 값도 함께 기록한다.
  GT가 부분 라벨이라 라벨 안 된 물체끼리의 병합은 못 잡는다 (하한값).
- 미매칭 후보 수 = 어떤 정답 물체와도 IoU>=0.5가 안 되는 후보. GT가 부분 라벨이라
  그대로 "오탐지"로 해석하면 안 된다.

예측 마스크는 data/eval/pred/<method>_<scene>.npz 로 저장한다. --from-pred 로 모델 없이
지표만 다시 계산할 수 있다 (지표 정의를 바꿨을 때).

사용 (전용 venv python):
  python scripts/run_benchmark.py                        # 12장면, m1·m2 (방법마다 독립 프로세스)
  python scripts/run_benchmark.py --methods m1 --scenes 05,07
  python scripts/run_benchmark.py --from-pred             # 저장된 예측으로 지표만 재계산
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

_PROJECT_DIR = Path(__file__).resolve().parent.parent
_EVAL_DIR = _PROJECT_DIR / "data" / "eval"
_PRED_DIR = _EVAL_DIR / "pred"
sys.path.insert(0, str(_PROJECT_DIR))

_ALL_SCENES = [f"{i:02d}" for i in range(1, 13)]
_IOU_MATCH = 0.5
_IOU_PARTIAL = 0.3  # 과분할 판정용 — 부분 겹침도 "같은 물체를 가리킨다"로 센다
_MERGE_COVERS = (0.3, 0.5, 0.7)  # 과병합 덮음 비율 — 기준값 민감도 확인용으로 함께 기록

# 방법 이름 → (모듈, 클래스, 설정 파일). 새 방법은 여기에 추가한다.
_METHODS = {
    "m1": ("methods.m1_sam2_toplayer", "M1Sam2TopLayer", "config/m1_sam2.yaml"),
    "m2": ("methods.m2_grounded_sam", "M2GroundedSam", "config/m2_grounded_sam.yaml"),
}


def _load_gt(scene: str):
    meta = json.loads((_EVAL_DIR / f"{scene}_gt.json").read_text(encoding="utf-8"))
    npz = np.load(_EVAL_DIR / f"{scene}_gt.npz")
    masks = [npz[f"obj{o['obj_id']}"].astype(bool) for o in meta["objects"]]
    return masks, meta["n_missing_objects"]


def _build_segmenter(method: str, bin_roi: Path):
    import importlib

    import yaml

    mod_name, cls_name, cfg_rel = _METHODS[method]
    Cls = getattr(importlib.import_module(mod_name), cls_name)
    cfg = yaml.safe_load((_PROJECT_DIR / cfg_rel).read_text(encoding="utf-8"))
    cfg["top_layer"]["bin_roi"] = str(bin_roi.resolve())
    return Cls(cfg, _PROJECT_DIR)


def _evaluate(gt_masks: list[np.ndarray], n_missing: int, pred_masks: list[np.ndarray],
              merge_cover: float) -> dict:
    from core.plane import bbox_of, mask_iou

    n_true = len(gt_masks) + n_missing
    pred_boxes = [bbox_of(p) for p in pred_masks]
    gt_boxes = [bbox_of(g) for g in gt_masks]
    gt_areas = [max(int(g.sum()), 1) for g in gt_masks]

    matched = 0
    oversegmented = 0
    pred_best_iou = [0.0] * len(pred_masks)
    # cover[pi][gi] = |pred ∩ gt| / |gt|
    cover = np.zeros((len(pred_masks), len(gt_masks)), dtype=np.float32)
    for gi, (g, gb) in enumerate(zip(gt_masks, gt_boxes)):
        hits = 0
        best = 0.0
        for pi, (p, pb) in enumerate(zip(pred_masks, pred_boxes)):
            iou = mask_iou(g, p, box_a=gb, box_b=pb)
            if iou > 0:
                cover[pi, gi] = np.count_nonzero(g & p) / gt_areas[gi]
            pred_best_iou[pi] = max(pred_best_iou[pi], iou)
            if iou >= _IOU_PARTIAL:
                hits += 1
            best = max(best, iou)
        if best >= _IOU_MATCH:
            matched += 1
        if hits > 1:
            oversegmented += 1

    def merges(thr):
        covered = cover >= thr
        merged_pred = np.flatnonzero(covered.sum(axis=1) >= 2)
        n_gt_in = int(covered[merged_pred].any(axis=0).sum()) if len(merged_pred) else 0
        return len(merged_pred), n_gt_in, merged_pred.tolist()

    n_merged, n_gt_merged, merged_idx = merges(merge_cover)
    unmatched_pred = sum(1 for b in pred_best_iou if b < _IOU_MATCH)
    return {
        "n_true_objects": n_true,
        "n_gt_labeled": len(gt_masks),
        "n_missing_objects": n_missing,
        "n_matched": matched,
        "recall": round(matched / n_true, 3) if n_true else None,
        "n_oversegmented_gt": oversegmented,
        "n_merged_pred": n_merged,
        "n_gt_in_merges": n_gt_merged,
        "merged_pred_index": merged_idx,  # 저장된 예측(npz)의 후보 순번
        "n_merged_pred_by_cover": {str(t): merges(t)[0] for t in _MERGE_COVERS},
        "n_candidates": len(pred_masks),
        "n_candidates_unmatched_iou0.5": unmatched_pred,
    }


def _save_pred(method: str, scene: str, masks: list[np.ndarray]) -> None:
    _PRED_DIR.mkdir(parents=True, exist_ok=True)
    arr = np.stack(masks) if masks else np.zeros((0, 1, 1), dtype=bool)
    np.savez_compressed(_PRED_DIR / f"{method}_{scene}.npz", masks=arr.astype(bool))


def _load_pred(method: str, scene: str) -> list[np.ndarray] | None:
    p = _PRED_DIR / f"{method}_{scene}.npz"
    if not p.exists():
        return None
    return list(np.load(p)["masks"])


def _run_one_method(method: str, scenes: list[str], args) -> list[dict]:
    """이 프로세스에 방법 하나만 로드해서 실행한다."""
    from core.loader import load_any

    seg = _build_segmenter(method, args.bin_roi)
    t0 = time.perf_counter()
    seg.build()  # 방법당 1회만 로드
    build_dt = time.perf_counter() - t0

    rows = []
    for scene in scenes:
        if not (_EVAL_DIR / f"{scene}_gt.npz").exists():
            print(f"[건너뜀] {scene}: GT 없음", file=sys.stderr)
            continue
        seg.cfg["top_layer"]["bin_frame"] = str(
            (_PROJECT_DIR / f"config/bin_frame_pp_{scene}.json").resolve())
        scene_data = load_any(args.data_dir / f"{scene}.zdf")

        t0 = time.perf_counter()
        seg.predict(scene_data)  # cold — 버린다
        cold_dt = time.perf_counter() - t0

        warm, stages = [], {}
        candidates = []
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            candidates = seg.predict(scene_data)
            warm.append(time.perf_counter() - t0)
            for k, v in getattr(seg, "stats", {}).items():
                if k.startswith("t_") and k.endswith("_s"):
                    stages.setdefault(k, []).append(v)

        pred_masks = [c.mask.astype(bool) for c in candidates]
        _save_pred(method, scene, pred_masks)
        gt_masks, n_missing = _load_gt(scene)
        m = _evaluate(gt_masks, n_missing, pred_masks, args.merge_cover)
        m.update({
            "scene": scene, "method": method,
            "build_time_s": round(build_dt, 3),
            "predict_time_cold_s": round(cold_dt, 3),
            "predict_time_warm_s": round(float(np.median(warm)), 3),
            "predict_time_warm_all_s": [round(w, 3) for w in warm],
            "stage_time_warm_s": {k: round(float(np.median(v)), 4) for k, v in stages.items()},
        })
        rows.append(m)
        st = " ".join(f"{k[2:-2]}={v:.2f}" for k, v in m["stage_time_warm_s"].items())
        print(f"[{method}][{scene}] recall={m['recall']} matched={m['n_matched']}/{m['n_true_objects']} "
              f"overseg={m['n_oversegmented_gt']} merged={m['n_merged_pred']} "
              f"cand={m['n_candidates']} warm={m['predict_time_warm_s']:.2f}s ({st})", file=sys.stderr)
    return rows


def _from_pred(methods: list[str], scenes: list[str], merge_cover: float) -> list[dict]:
    rows = []
    for method in methods:
        for scene in scenes:
            pred = _load_pred(method, scene)
            if pred is None or not (_EVAL_DIR / f"{scene}_gt.npz").exists():
                continue
            gt_masks, n_missing = _load_gt(scene)
            m = _evaluate(gt_masks, n_missing, pred, merge_cover)
            m.update({"scene": scene, "method": method})
            rows.append(m)
    return rows


def _summary(rows: list[dict], methods: list[str]) -> dict:
    out = {}
    for method in methods:
        r = [x for x in rows if x["method"] == method]
        if not r:
            continue
        tot = lambda k: sum(x[k] for x in r)  # noqa: E731
        s = {
            "n_scenes": len(r),
            "matched": tot("n_matched"), "n_true_objects": tot("n_true_objects"),
            "recall": round(tot("n_matched") / tot("n_true_objects"), 3),
            "oversegmented_gt": tot("n_oversegmented_gt"),
            "merged_pred": tot("n_merged_pred"),
            "merged_pred_by_cover": {c: sum(x["n_merged_pred_by_cover"][c] for x in r)
                                     for c in r[0]["n_merged_pred_by_cover"]},
            "candidates": tot("n_candidates"),
        }
        if "predict_time_warm_s" in r[0]:
            s["warm_median_s"] = round(float(np.median([x["predict_time_warm_s"] for x in r])), 3)
            keys = sorted({k for x in r for k in x["stage_time_warm_s"]})
            s["stage_median_s"] = {k: round(float(np.median([x["stage_time_warm_s"].get(k, 0) for x in r])), 4)
                                   for k in keys}
        out[method] = s
    return out


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(description="세그멘테이션 벤치마크 — recall/과분할/과병합/구간 시간")
    parser.add_argument("--scenes", default=",".join(_ALL_SCENES))
    parser.add_argument("--methods", default="m1,m2", help=f"쉼표 구분. 등록: {','.join(_METHODS)}")
    parser.add_argument("--repeats", type=int, default=3, help="cold 1회 뒤 warm 측정 횟수 (중앙값)")
    parser.add_argument("--merge-cover", type=float, default=0.5,
                        help="과병합 판정: 후보가 정답 물체를 덮는 비율(교집합/정답 면적) 기준")
    parser.add_argument("--data-dir", type=Path,
                        default=_PROJECT_DIR / "data/input/20260811_piece_picking_data")
    parser.add_argument("--bin-roi", type=Path, default=_PROJECT_DIR / "config/bin_roi_pp.json")
    parser.add_argument("--output", type=Path, default=_EVAL_DIR / "benchmark_results.json")
    parser.add_argument("--from-pred", action="store_true", help="모델 없이 저장된 예측으로 지표만 재계산")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    scenes = args.scenes.split(",")
    methods = args.methods.split(",")
    unknown = [m for m in methods if m not in _METHODS]
    if unknown:
        parser.error(f"등록되지 않은 방법: {unknown}")

    if args.from_pred:
        rows = _from_pred(methods, scenes, args.merge_cover)
    elif len(methods) == 1:
        rows = _run_one_method(methods[0], scenes, args)
    else:
        # 방법마다 독립 프로세스 — 한 프로세스에 두 모델을 올리지 않는다
        rows = []
        with tempfile.TemporaryDirectory() as td:
            for method in methods:
                out = Path(td) / f"{method}.json"
                cmd = [sys.executable, str(Path(__file__).resolve()), "--child",
                       "--methods", method, "--scenes", args.scenes,
                       "--repeats", str(args.repeats), "--merge-cover", str(args.merge_cover),
                       "--data-dir", str(args.data_dir), "--bin-roi", str(args.bin_roi),
                       "--output", str(out)]
                print(f"\n=== {method}: 독립 프로세스로 실행 ===", file=sys.stderr)
                subprocess.run(cmd, check=True)
                rows += json.loads(out.read_text(encoding="utf-8"))["results"]

    result = {"results": rows}
    if not args.child:
        result["summary"] = _summary(rows, methods)
        result["settings"] = {"repeats": args.repeats, "merge_cover": args.merge_cover,
                              "iou_match": _IOU_MATCH, "iou_partial": _IOU_PARTIAL,
                              "from_pred": args.from_pred}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.child:
        return

    print(f"\n[저장] {args.output}", file=sys.stderr)
    print("\n=== 방법별 집계 ===", file=sys.stderr)
    for method, s in result["summary"].items():
        line = (f"{method}: recall {s['matched']}/{s['n_true_objects']} = {s['recall']:.3f}, "
                f"과분할 {s['oversegmented_gt']}개 물체, 과병합 후보 {s['merged_pred']}개 "
                f"(덮음 기준별 {s['merged_pred_by_cover']}), 후보 총 {s['candidates']}개")
        if "warm_median_s" in s:
            st = ", ".join(f"{k[2:-2]} {v:.3f}" for k, v in s["stage_median_s"].items())
            line += f", warm 중앙값 {s['warm_median_s']:.2f}s ({st})"
        print(line, file=sys.stderr)


if __name__ == "__main__":
    main()
