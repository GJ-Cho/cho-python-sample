"""단일 방법 실행 + 시각화 (PLAN.md 6장 run_single.py).

사용 (전용 venv python으로 실행):
  C:/Zivid/3rdparty/venv_segpick/Scripts/python.exe \
    src/project/2d_segmentation_picking/scripts/run_single.py --method m1 --topk 10

출력 (data/output):
  <stem>_<method><tag>_top<N>.png        마스크/센터 오버레이. N = 실제로 그린 마스크 수
  <stem>_<method><tag>_top<N>_report.json 후보 요약 + 타이밍 + config 스냅샷 + git 커밋

파일명에 N을 넣는 이유: 오버레이는 상위 N개만 그리므로 **같은 결과라도 --topk가 다르면
다른 그림이 된다.** 파일명에 없으면 서로 다른 조건의 그림을 같은 조건으로 착각해 비교하게 된다.
리포트에 config 전문과 git 커밋을 남기는 이유: 어떤 파라미터로 뽑은 결과인지 역추적하려면
파일명 태그만으로는 부족하다(태그는 자유 문자열이라 의미가 기록되지 않는다).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

_PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_DIR))


def _git_info(cwd: Path) -> dict:
    """현재 커밋 해시와 작업트리 변경 여부. git이 없거나 실패하면 None으로 채운다."""
    def run(*cmd):
        try:
            out = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, timeout=10)
            return out.stdout.strip() if out.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            return None

    commit = run("git", "rev-parse", "--short", "HEAD")
    status = run("git", "status", "--porcelain")
    return {
        "commit": commit,
        # dirty면 커밋 해시만으로 결과를 재현할 수 없다는 뜻이므로 반드시 함께 남긴다
        "dirty": None if status is None else bool(status),
    }


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
    # 파일명 접미사는 **실제로 그린 마스크 수**로 만든다. 요청값(--topk)이 후보 수보다
    # 크면 그림에는 후보 수만큼만 담기므로, 요청값을 쓰면 파일명이 내용을 잘못 알린다.
    stem = f"{args.input.stem}_{args.method}{args.tag}_top{len(topk)}"

    report = {
        "method": args.method,
        "input": str(args.input),
        "bin_roi": config["top_layer"]["bin_roi"],
        "build_time_s": round(build_dt, 3),
        "predict_time_s": round(cold_dt, 3),
        "predict_time_warm_s": round(warm_dt, 3),
        "num_candidates": len(candidates),
        "topk_requested": args.topk,
        "topk_drawn": len(topk),
        "stats": getattr(segmenter, "stats", {}),
        # 재현을 위한 출처 정보 — 파일명 태그는 자유 문자열이라 의미가 기록되지 않는다
        "git": _git_info(_PROJECT_DIR),
        "config_path": str(cfg_path),
        "config": config,
        "topk": [
            {"rank": i, "center_px": c.center_px, "score": round(c.score, 2), **c.meta}
            for i, c in enumerate(topk)
        ],
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))

    # 리포트도 디스크에 남긴다 — 튜닝/벤치마크 수치가 stdout에만 남으면 재현이 불가능하다
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.output_dir / f"{stem}_report.json"
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[리포트 저장] {out_json}", file=sys.stderr)

    # 시각화: 상위 마스크 오버레이 + center 마커
    overlay = overlay_masks(scene.rgb, [c.mask for c in topk], alpha=0.5)
    overlay = draw_candidates_2d(overlay, topk)
    out_png = args.output_dir / f"{stem}.png"
    mpimg.imsave(str(out_png), overlay)
    print(f"\n[오버레이 저장] {out_png}", file=sys.stderr)


if __name__ == "__main__":
    main()
