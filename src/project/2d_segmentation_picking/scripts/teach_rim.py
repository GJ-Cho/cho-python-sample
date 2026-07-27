"""림 티칭 (PLAN.md 4장, Phase 1).

RGB 위에서 빈 외곽 사각형의 4꼭짓점을 **시계방향**으로 클릭하면, 안쪽으로 band_mm
두께의 림 밴드(환형 ROI)를 자동 생성해 config/bin_roi.json에 저장한다.
셋업이 고정이면 1회만 하면 된다.

사용:
  python scripts/teach_rim.py                       # 기본 입력 image_test.zdf, band 20mm
  python scripts/teach_rim.py --band-mm 25 --input path/to.zdf

주의: 대화형 클릭이 필요하므로 사용자가 직접 실행한다 (Claude Code는 클릭 불가).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_DIR))


def _load_scene(path: Path):
    from core.loader import load_npz, load_zdf

    if path.suffix.lower() == ".zdf":
        import zivid
        with zivid.Application():
            return load_zdf(path)
    return load_npz(path)


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(description="림 외곽 4꼭짓점 티칭 → 림 밴드 config 저장")
    parser.add_argument("--input", type=Path, default=_PROJECT_DIR / "data/input/image_test.zdf")
    parser.add_argument("--band-mm", type=float, default=20.0, help="림 밴드 두께(mm)")
    parser.add_argument("--config", type=Path, default=_PROJECT_DIR / "config/bin_roi.json")
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    from core import roi
    from core.viz import setup_korean_font

    font = setup_korean_font()  # 한글 폰트 설정 (없으면 None)
    if font is None:
        print("[경고] 한글 폰트를 찾지 못했습니다. 제목이 깨질 수 있습니다.", file=sys.stderr)

    scene = _load_scene(args.input)

    # --- 4꼭짓점 클릭 (시계방향) ---
    fig, ax = plt.subplots(figsize=(11, 9))
    ax.imshow(scene.rgb)
    ax.set_title("빈 외곽 사각형의 4꼭짓점을 시계방향으로 클릭 (좌상단→우상단→우하단→좌하단)\n"
                 "잘못 찍으면 마우스 오른쪽 클릭으로 취소, Enter로 확정")
    pts = plt.ginput(4, timeout=0, show_clicks=True)  # (x=col, y=row)
    plt.close(fig)

    if len(pts) != 4:
        raise SystemExit(f"4점이 필요합니다 (입력: {len(pts)}점)")
    corners_rc = [[int(round(y)), int(round(x))] for (x, y) in pts]

    # --- 림 밴드 미리보기 ---
    annulus = roi.rim_annulus_mask(scene, corners_rc, args.band_mm)
    n_valid = int((annulus & scene.valid_mask).sum())

    config = {
        "rim_outer_corners_rc": corners_rc,
        "rim_band_mm": args.band_mm,
        "input": str(args.input),
        "annulus_px": int(annulus.sum()),
        "annulus_valid_px": n_valid,
        "note": "rim_outer_corners_rc는 [row,col] 시계방향. 밴드는 로드 시 재계산.",
    }
    args.config.parent.mkdir(parents=True, exist_ok=True)
    args.config.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(config, indent=2, ensure_ascii=False))
    print(f"\n[저장됨] {args.config}", file=sys.stderr)
    print(f"림 밴드 유효 포인트: {n_valid}개", file=sys.stderr)

    # 오버레이 저장 + 표시 (육안 확인)
    out_png = _PROJECT_DIR / "data/output" / f"{args.input.stem}_rim_annulus.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    from core.viz import overlay_masks
    import matplotlib.image as mpimg
    overlay = overlay_masks(scene.rgb, annulus, alpha=0.6)
    mpimg.imsave(str(out_png), overlay)
    print(f"[오버레이 저장] {out_png}", file=sys.stderr)

    fig2, ax2 = plt.subplots(figsize=(11, 9))
    ax2.imshow(overlay)
    ax2.set_title(f"림 밴드 미리보기 (band={args.band_mm}mm, 유효 {n_valid}px) — 창을 닫으면 종료")
    plt.show()


if __name__ == "__main__":
    main()
