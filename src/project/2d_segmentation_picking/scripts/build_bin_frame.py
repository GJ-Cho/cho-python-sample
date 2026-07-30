"""Bin frame 계산 + 최상층 마스크 검증 (PLAN.md Phase 1, 비대화형).

config/bin_roi.json(림 티칭 결과)을 읽어:
  1) 림 밴드 → RANSAC 평면 피팅 → bin frame (n_up, p_floor) 계산, config/bin_frame.json 저장
  2) 최상층(top_layer) 마스크 자동 계산
  3) 림 + 최상층 오버레이 PNG 저장, 통계 JSON 출력

검증 기준: 림 평면 RMS < 2mm, 최상층 밴드 오버레이 육안 확인.

사용:
  python scripts/build_bin_frame.py --bin-depth-mm 150 --top-band-mm 30
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_DIR))


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(description="Bin frame 계산 + 최상층 마스크 검증")
    parser.add_argument("--input", type=Path, default=_PROJECT_DIR / "data/input/image_test_01.zdf")
    parser.add_argument("--config-roi", type=Path, default=_PROJECT_DIR / "config/bin_roi_01.json")
    parser.add_argument("--config-frame", type=Path, default=_PROJECT_DIR / "config/bin_frame_01.json")
    parser.add_argument("--bin-depth-mm", type=float, default=150.0, help="림→바닥 깊이 (bin frame용)")
    parser.add_argument("--top-band-mm", type=float, default=50.0, help="최상층 상대 밴드 두께")
    parser.add_argument("--floor-margin-mm", type=float, default=None,
                        help="최상층 절대 하한(바닥 제외). bin_depth 부정확 시 끔(기본). 상대 밴드 위주")
    parser.add_argument("--max-height-mm", type=float, default=None,
                        help="최상층 절대 상한(overfill 허용 위해 기본 없음). 반사 아웃라이어 제거 시 지정")
    parser.add_argument("--interior-shrink-mm", type=float, default=0.0,
                        help="빈 내부 ROI를 림에서 더 안쪽으로 수축(사면 벽 제외용, 예: 60)")
    parser.add_argument("--snr-min", type=float, default=None)
    args = parser.parse_args()

    import matplotlib.image as mpimg

    from core import bin_frame as bf_mod
    from core import roi as roi_mod
    from core.loader import load_any
    from core.viz import overlay_masks

    if not args.config_roi.exists():
        raise SystemExit(f"림 config가 없습니다: {args.config_roi}\n먼저 teach_rim.py를 실행하세요.")

    roi_cfg = json.loads(args.config_roi.read_text(encoding="utf-8"))
    corners_rc = roi_cfg["rim_outer_corners_rc"]
    rim_band_mm = roi_cfg["rim_band_mm"]

    scene = load_any(args.input)

    # 1) 림 밴드 → bin frame
    rim_mask = roi_mod.rim_annulus_mask(scene, corners_rc, rim_band_mm)
    bf = bf_mod.fit_bin_frame(scene, rim_mask, args.bin_depth_mm)
    bf_mod.save_bin_frame(bf, args.config_frame)

    # 2) 최상층 마스크 (빈 내부로 한정 — 림 안쪽을 그대로 재사용)
    interior = roi_mod.bin_interior_mask(scene, corners_rc, rim_band_mm, shrink_mm=args.interior_shrink_mm)
    n_up, _, p_floor = bf.as_np()
    top_mask, top_meta = roi_mod.top_layer_mask(
        scene, n_up, p_floor, roi_2d=interior, band_mm=args.top_band_mm,
        min_height_mm=args.floor_margin_mm, max_height_mm=args.max_height_mm, snr_min=args.snr_min)

    # 3) 오버레이 저장 (림=0번색, 최상층=1번색)
    out_png = _PROJECT_DIR / "data/output" / f"{args.input.stem}_bin_frame.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    overlay = overlay_masks(scene.rgb, [rim_mask, top_mask], alpha=0.5)
    mpimg.imsave(str(out_png), overlay)

    report = {
        "bin_frame": {
            "n_up": [round(v, 4) for v in bf.n_up],
            "p_floor_mm": [round(v, 2) for v in bf.p_floor],
            "rms_mm": bf.rms_mm,
            "n_inliers": bf.n_inliers,
            "rms_pass_<2mm": bf.rms_mm < 2.0,
        },
        "top_layer": top_meta,
        "outputs": {"bin_frame_json": str(args.config_frame), "overlay_png": str(out_png)},
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\n[저장됨] {args.config_frame}", file=sys.stderr)


if __name__ == "__main__":
    main()
