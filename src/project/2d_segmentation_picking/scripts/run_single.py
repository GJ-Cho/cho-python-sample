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
    parser.add_argument("--input", type=Path, default=_PROJECT_DIR / "data/input/image_test_01.zdf")
    parser.add_argument("--config", type=Path, default=None, help="기본: config/<method>_*.yaml")
    parser.add_argument("--bin-roi", type=Path, default=None,
                        help="장면별 림 티칭 config (기본: yaml의 top_layer.bin_roi)")
    parser.add_argument("--bin-frame", type=Path, default=None,
                        help="장면별 bin frame config (기본: yaml의 top_layer.bin_frame)")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--tag", default="", help="출력 파일명 접미사 (설정 비교용)")
    parser.add_argument("--output-dir", type=Path, default=_PROJECT_DIR / "data/output")
    args = parser.parse_args()

    import matplotlib.image as mpimg
    import yaml

    from core.loader import load_any
    from core.viz import draw_candidates_2d, overlay_masks

    # 방법/설정 로드
    defaults = {"m1": ("config/m1_sam2.yaml", "M1Sam2TopLayer"),
                "m2": ("config/m2_grounded_sam.yaml", "M2GroundedSam")}
    if args.method not in defaults:
        raise SystemExit(f"미구현 방법: {args.method}")
    cfg_path = args.config or (_PROJECT_DIR / defaults[args.method][0])
    config = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    # 장면별 ROI/bin frame 오버라이드 (같은 yaml로 여러 장면을 돌리기 위함)
    # (상대경로는 CLI 기준이므로 절대경로로 고정 — 메서드 쪽은 project_dir 기준으로 해석한다)
    if args.bin_roi:
        config["top_layer"]["bin_roi"] = str(args.bin_roi.resolve())
    if args.bin_frame:
        config["top_layer"]["bin_frame"] = str(args.bin_frame.resolve())

    if args.method == "m1":
        from methods.m1_sam2_toplayer import M1Sam2TopLayer
        segmenter = M1Sam2TopLayer(config, _PROJECT_DIR)
    else:
        from methods.m2_grounded_sam import M2GroundedSam
        segmenter = M2GroundedSam(config, _PROJECT_DIR)

    scene = load_any(args.input)

    # 모델 로드는 1회성이므로 사이클 타임에서 분리한다 (PLAN 8장: 탐색 ≤1s 목표는 warm 기준)
    t0 = time.time()
    segmenter.build()
    build_dt = time.time() - t0

    t0 = time.time()
    candidates = segmenter.predict(scene)
    cold_dt = time.time() - t0

    # 2회차 = warm(캐시/컴파일 완료) 스테디 스테이트
    t0 = time.time()
    candidates = segmenter.predict(scene)
    warm_dt = time.time() - t0

    topk = candidates[: args.topk]
    report = {
        "method": args.method,
        "input": str(args.input),
        "bin_roi": config["top_layer"]["bin_roi"],
        "build_time_s": round(build_dt, 3),
        "predict_time_s": round(cold_dt, 3),
        "predict_time_warm_s": round(warm_dt, 3),
        "num_candidates": len(candidates),
        "stats": getattr(segmenter, "stats", {}),
        "topk": [
            {"rank": i, "center_px": c.center_px, "score": round(c.score, 2), **c.meta}
            for i, c in enumerate(topk)
        ],
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))

    # 리포트도 디스크에 남긴다 — 튜닝/벤치마크 수치가 stdout에만 남으면 재현이 불가능하다
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.output_dir / f"{args.input.stem}_{args.method}{args.tag}_report.json"
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[리포트 저장] {out_json}", file=sys.stderr)

    # 시각화: 상위 마스크 오버레이 + center 마커
    overlay = overlay_masks(scene.rgb, [c.mask for c in topk], alpha=0.5)
    overlay = draw_candidates_2d(overlay, topk)
    out_png = args.output_dir / f"{args.input.stem}_{args.method}{args.tag}.png"
    mpimg.imsave(str(out_png), overlay)
    print(f"\n[오버레이 저장] {out_png}", file=sys.stderr)


if __name__ == "__main__":
    main()
