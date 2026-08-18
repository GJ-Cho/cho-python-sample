"""평가 세트 준비 — SAM2 조각 생성 (Phase 5 선행, 2026-08-06).

Phase 5(run_benchmark.py)가 필요로 하는 recall/마스크 중복률을 계산하려면 정답
세트가 있어야 하는데, 처음부터 폴리곤을 그리면 느리다. 대신 m1과 같은 SAM2
top-layer 파이프라인을 재사용하되 필터/기하 후처리를 최대한 느슨하게 풀어
"물체 하나가 여러 조각으로 쪼개져도 되니 놓치는 조각이 없게" 원시 후보를
뽑는다. 사람이 번호가 매겨진 오버레이를 보고 "N번+M번은 한 물체"라고
묶어주면 그게 정답 세트가 된다 (core/plane.py의 mask_iou로 나중에 채점).

**맹점**: SAM2가 애초에 조각을 하나도 못 만든 물체는 이 방식으로 못 잡는다.
오버레이에서 번호가 하나도 없는 빈 자리에 물체가 있는지 반드시 육안으로
같이 확인해야 한다.

사용 (전용 venv python):
  python scripts/gen_gt_fragments.py --input data/input/image_test_01.zdf \
    --bin-roi config/bin_roi_01.json --bin-frame config/bin_frame_01.json

출력 (data/eval, gitignore 대상 — data/ 전체와 동일 정책):
  <stem>_fragments.png    번호 매겨진 조각 오버레이 (사람이 보고 그룹 지정)
  <stem>_fragments.npz    조각별 (H,W) bool 마스크, 키 "m<id>"
  <stem>_fragments.json   조각별 id/center_px/area_px/bbox 메타
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_DIR))

# 번호를 눈으로 구분하기 쉽게 20색 순환 팔레트 (BGR, cv2 기준)
_PALETTE_BGR = [
    (255, 56, 56), (56, 200, 255), (56, 255, 106), (255, 56, 220),
    (255, 176, 56), (56, 255, 220), (176, 56, 255), (255, 255, 56),
    (56, 106, 255), (255, 106, 56), (106, 255, 56), (255, 56, 106),
    (56, 255, 56), (255, 176, 176), (176, 255, 56), (56, 176, 255),
    (220, 56, 255), (255, 220, 56), (56, 255, 176), (176, 56, 176),
]


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(description="평가 세트용 SAM2 조각(원시 후보) 생성")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--bin-roi", type=Path, required=True)
    parser.add_argument("--bin-frame", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=_PROJECT_DIR / "config/m1_sam2.yaml",
                        help="sam2 체크포인트 등 공통 설정을 읽어올 베이스 config")
    parser.add_argument("--output-dir", type=Path, default=_PROJECT_DIR / "data/eval")
    args = parser.parse_args()

    import cv2
    import matplotlib.image as mpimg
    import yaml

    from core.loader import load_any
    from core.plane import bbox_of
    from methods.m1_sam2_toplayer import M1Sam2TopLayer

    base_cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    # 최대 회수(recall) 지향 설정 — "놓치지 않는 것"이 유일한 목표다.
    # 최종 파이프라인(m1_sam2.yaml)과는 다른 별개의 설정이며 배포용이 아니다.
    config = {
        "sam2": base_cfg["sam2"],
        "amg": {
            "points_per_side": 24,          # 기본(16)보다 촘촘하게
            "grid_mode": "bbox",
            "pred_iou_thresh": 0.5,          # 기본(0.7)보다 느슨하게 — 애매한 조각도 채택
            "stability_score_thresh": 0.7,   # 기본(0.85)보다 느슨하게
            "min_mask_region_area": 100,     # 기본(200)보다 작은 조각도 허용
            "multimask_output": True,
        },
        "top_layer": {
            "bin_roi": str(args.bin_roi.resolve()),
            "bin_frame": str(args.bin_frame.resolve()),
            "interior_shrink_mm": base_cfg["top_layer"]["interior_shrink_mm"],
            "top_band_mm": base_cfg["top_layer"]["top_band_mm"],
        },
        "filter": {
            "min_area_px": 100,
            "max_area_frac": 0.95,
            "min_overlap": 0.05,             # 경계 조각도 놓치지 않게 최대한 느슨하게
        },
        "geometry": {
            "enabled": True,
            "dedupe_iou": 0.9,               # 거의 동일한 중복만 제거 (multimask 부산물)
            "erode_px": 3,
            "max_fit_points": 2000,
            "inlier_thresh_mm": 1.5,
            "trim_iterations": 3,
            "min_plane_points": 50,
            "merge": {"enabled": True, "normal_deg": 8, "offset_mm": 3, "dilate_px": 5},
            "reject": {  # 사람이 판단할 것이므로 자동 기각은 전부 끈다
                "require_plane": False,
                "max_plane_rms_all_mm": None,
                "min_inlier_ratio": None,
                "max_depth_span_mm": None,
                "max_tilt_deg": None,
            },
        },
    }

    scene = load_any(args.input)
    segmenter = M1Sam2TopLayer(config, _PROJECT_DIR)
    segmenter.build()
    candidates = segmenter.predict(scene)
    print(f"조각 {len(candidates)}개 생성 (stats: {segmenter.stats})", file=sys.stderr)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.input.stem

    # --- 오버레이: 마스크 윤곽선 + 번호만 그린다 (조각이 많아 solid fill은 안 보임) ---
    overlay = np.ascontiguousarray(scene.rgb.copy())
    meta = []
    masks_npz = {}
    for i, c in enumerate(candidates):
        color_bgr = _PALETTE_BGR[i % len(_PALETTE_BGR)]
        color_rgb = color_bgr[::-1]
        mask_u8 = c.mask.astype(np.uint8)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color_rgb, 2)
        row, col = c.center_px
        # 검은 테두리 + 색 글자로 어떤 배경에서도 읽히게
        cv2.putText(overlay, str(i), (int(col), int(row)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(overlay, str(i), (int(col), int(row)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, color_rgb, 1, cv2.LINE_AA)
        r0, r1, c0, c1 = bbox_of(c.mask)
        meta.append({"id": i, "center_px": list(c.center_px), "area_px": int(c.mask.sum()),
                     "bbox_rc": [r0, r1, c0, c1]})
        masks_npz[f"m{i}"] = c.mask

    out_png = args.output_dir / f"{stem}_fragments.png"
    mpimg.imsave(str(out_png), overlay)
    out_npz = args.output_dir / f"{stem}_fragments.npz"
    np.savez_compressed(out_npz, **masks_npz)
    out_json = args.output_dir / f"{stem}_fragments.json"
    out_json.write_text(json.dumps({"input": str(args.input), "n_fragments": len(candidates),
                                     "fragments": meta}, indent=2, ensure_ascii=False),
                         encoding="utf-8")

    print(f"[저장] {out_png}", file=sys.stderr)
    print(f"[저장] {out_npz}", file=sys.stderr)
    print(f"[저장] {out_json}", file=sys.stderr)


if __name__ == "__main__":
    main()
