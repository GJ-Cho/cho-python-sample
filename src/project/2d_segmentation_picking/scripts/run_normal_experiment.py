"""실험: 노말맵 이미지를 세그멘테이션 입력으로 사용 (사용자 제안, 2026-08-06).

`convert_zdf_file_dir.py`의 `_create_normal_map`처럼 픽셀당 법선을 이미지로 인코딩해,
m1/m2가 원래 쓰는 `scene.rgb` 대신 이 노말맵 이미지를 세그멘테이션 입력으로 준다.
ROI(top_layer_mask)·기하 정련(core/plane.py)·높이 계산은 전부 실제 XYZ 그대로이고,
**"무엇을 보고 마스크 경계를 그리는가"만** RGB→노말맵으로 바뀐다.

가설: PLAN 5장이 순수 기하 세그멘테이션(깊이 클러스터링)을 폐기한 이유는 "밀착된 물체가
깊이는 이어져 있어 구분이 안 된다"였다. 하지만 깊이가 이어져도 표면 방향(법선)이 꺾이는
경우(주름/이음새/살짝 다른 각도)가 있어, RGB에도 depth map에도 안 보이는 경계가 노말맵
음영에는 보일 수 있다.

사용 (전용 venv python으로 실행):
  python scripts/run_normal_experiment.py --input data/input/image_test_01.zdf \
    --bin-roi config/bin_roi_01.json --bin-frame config/bin_frame_01.json

출력 (data/output):
  <stem>_normal_map.png                     노말맵 시각화(참고용, 방법 무관 1장)
  <stem>_<method>_normal_top<N>.png          기존 run_single.py와 동일한 오버레이 포맷
  <stem>_<method>_normal_top<N>_report.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

_PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_DIR))


def _git_info(cwd: Path) -> dict:
    def run(*cmd):
        try:
            out = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, timeout=10)
            return out.stdout.strip() if out.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            return None

    commit = run("git", "rev-parse", "--short", "HEAD")
    status = run("git", "status", "--porcelain")
    return {"commit": commit, "dirty": None if status is None else bool(status)}


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(description="노말맵 이미지를 세그멘테이션 입력으로 쓰는 실험")
    parser.add_argument("--input", type=Path, default=_PROJECT_DIR / "data/input/image_test_01.zdf")
    parser.add_argument("--bin-roi", type=Path, default=None)
    parser.add_argument("--bin-frame", type=Path, default=None)
    parser.add_argument("--methods", default="m1,m2", help="쉼표 구분 (m1,m2)")
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--tag", default="_normal")
    parser.add_argument("--output-dir", type=Path, default=_PROJECT_DIR / "data/output")
    args = parser.parse_args()

    import matplotlib.image as mpimg
    import yaml

    from core.loader import load_any
    from core.viz import draw_candidates_2d, normals_to_rgb, overlay_masks

    scene = load_any(args.input)
    if scene.normals is None:
        raise SystemExit("scene.normals가 없습니다 (.zdf 로더 확인 필요)")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    original_rgb = scene.rgb.copy()  # 오버레이는 원본 사진 위에 그려야 눈으로 대조가 된다
    normal_rgb = normals_to_rgb(scene.normals)
    norm_png = args.output_dir / f"{args.input.stem}_normal_map.png"
    mpimg.imsave(str(norm_png), normal_rgb)
    print(f"[노말맵 저장] {norm_png}", file=sys.stderr)

    defaults = {"m1": ("config/m1_sam2.yaml", "M1Sam2TopLayer"),
                "m2": ("config/m2_grounded_sam.yaml", "M2GroundedSam")}

    for method in args.methods.split(","):
        method = method.strip()
        if method not in defaults:
            raise SystemExit(f"미구현 방법: {method}")
        cfg_path = _PROJECT_DIR / defaults[method][0]
        config = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        if args.bin_roi:
            config["top_layer"]["bin_roi"] = str(args.bin_roi.resolve())
        if args.bin_frame:
            config["top_layer"]["bin_frame"] = str(args.bin_frame.resolve())

        if method == "m1":
            from methods.m1_sam2_toplayer import M1Sam2TopLayer
            segmenter = M1Sam2TopLayer(config, _PROJECT_DIR)
        else:
            from methods.m2_grounded_sam import M2GroundedSam
            segmenter = M2GroundedSam(config, _PROJECT_DIR)

        # ROI/기하/높이는 scene.xyz 그대로 쓴다. 세그멘테이션이 "보는" 이미지만 바꾼다.
        scene.rgb = normal_rgb

        t0 = time.time()
        segmenter.build()
        build_dt = time.time() - t0

        t0 = time.time()
        candidates = segmenter.predict(scene)
        cold_dt = time.time() - t0
        t0 = time.time()
        candidates = segmenter.predict(scene)
        warm_dt = time.time() - t0

        topk = candidates[: args.topk]
        stem = f"{args.input.stem}_{method}{args.tag}_top{len(topk)}"

        report = {
            "method": method,
            "input_channel": "normals",
            "input": str(args.input),
            "bin_roi": config["top_layer"]["bin_roi"],
            "build_time_s": round(build_dt, 3),
            "predict_time_s": round(cold_dt, 3),
            "predict_time_warm_s": round(warm_dt, 3),
            "num_candidates": len(candidates),
            "topk_requested": args.topk,
            "topk_drawn": len(topk),
            "stats": getattr(segmenter, "stats", {}),
            "git": _git_info(_PROJECT_DIR),
            "config_path": str(cfg_path),
            "config": config,
            "topk": [
                {"rank": i, "center_px": c.center_px, "score": round(c.score, 2), **c.meta}
                for i, c in enumerate(topk)
            ],
        }
        out_json = args.output_dir / f"{stem}_report.json"
        out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[{method}] 후보 {len(candidates)}개, warm {warm_dt:.2f}s → {out_json}", file=sys.stderr)

        # 오버레이는 원본 RGB 위에 그린다 — "노말맵이 찾은 마스크가 실제 물체에 맞는가"를
        # 눈으로 비교하려면 배경이 원본 사진이어야 한다(노말맵 위에 그리면 대조가 어렵다).
        overlay = overlay_masks(original_rgb, [c.mask for c in topk], alpha=0.5)
        overlay = draw_candidates_2d(overlay, topk)
        out_png = args.output_dir / f"{stem}.png"
        mpimg.imsave(str(out_png), overlay)
        print(f"[{method}] 오버레이 저장 → {out_png}", file=sys.stderr)


if __name__ == "__main__":
    main()
