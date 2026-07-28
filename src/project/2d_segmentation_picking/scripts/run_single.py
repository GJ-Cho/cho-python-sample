"""단일 방법 실행 + 시각화 (PLAN.md 6장 run_single.py).

사용 (전용 venv python으로 실행):
  C:/Zivid/3rdparty/venv_segpick/Scripts/python.exe \
    src/project/2d_segmentation_picking/scripts/run_single.py --method m1 --topk 10

출력: 후보 요약 JSON(stdout) + 마스크/센터 오버레이 PNG(data/output).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_DIR))


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(description="단일 세그멘테이션 방법 실행 + 시각화")
    parser.add_argument("--method", default="m1", choices=["m1", "m2"])  # m3는 Phase 4
    parser.add_argument("--input", type=Path, default=_PROJECT_DIR / "data/input/image_test.zdf")
    parser.add_argument("--config", type=Path, default=None, help="기본: config/<method>_*.yaml")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=_PROJECT_DIR / "data/output")
    args = parser.parse_args()

    import matplotlib.image as mpimg
    import yaml

    from core.loader import load_any
    from core.viz import draw_candidates_2d, overlay_masks

    # 방법/설정 로드
    if args.method == "m1":
        from methods.m1_sam2_toplayer import M1Sam2TopLayer
        cfg_path = args.config or (_PROJECT_DIR / "config/m1_sam2.yaml")
        config = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        segmenter = M1Sam2TopLayer(config, _PROJECT_DIR)
    elif args.method == "m2":
        from methods.m2_grounded_sam import M2GroundedSam
        cfg_path = args.config or (_PROJECT_DIR / "config/m2_grounded_sam.yaml")
        config = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        segmenter = M2GroundedSam(config, _PROJECT_DIR)
    else:
        raise SystemExit(f"미구현 방법: {args.method}")

    scene = load_any(args.input)

    t0 = time.time()
    candidates = segmenter.predict(scene)
    dt = time.time() - t0

    topk = candidates[: args.topk]
    report = {
        "method": args.method,
        "input": str(args.input),
        "predict_time_s": round(dt, 3),
        "num_candidates": len(candidates),
        "topk": [
            {"rank": i, "center_px": c.center_px, "score": round(c.score, 2), **c.meta}
            for i, c in enumerate(topk)
        ],
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))

    # 시각화: 상위 마스크 오버레이 + center 마커
    args.output_dir.mkdir(parents=True, exist_ok=True)
    overlay = overlay_masks(scene.rgb, [c.mask for c in topk], alpha=0.5)
    overlay = draw_candidates_2d(overlay, topk)
    out_png = args.output_dir / f"{args.input.stem}_{args.method}.png"
    mpimg.imsave(str(out_png), overlay)
    print(f"\n[오버레이 저장] {out_png}", file=sys.stderr)


if __name__ == "__main__":
    main()
