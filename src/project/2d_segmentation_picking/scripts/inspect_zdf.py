"""ZDF 씬 통계 및 RGB 포맷 비교 (PLAN.md Phase 0 산출물).

수행 내용:
  1) ZDF 로드 → 해상도/유효포인트/깊이(z)/SNR/좌표 bbox 통계 계산
  2) scene_stats.json 저장 + stdout에 JSON 출력 (SDK 없는 협업자와 공유용)
  3) rgba / rgba_srgb 두 포맷을 각각 PNG로 저장 → 밝고 자연스러운 쪽 육안 채택
     (PLAN 2장 "확인 필요: rgba vs rgba_srgb")
  4) (옵션) 장면을 npz로 저장

사용:
  python scripts/inspect_zdf.py                       # 기본 입력 data/input/image_test.zdf
  python scripts/inspect_zdf.py --input path/to.zdf --output-dir data/output --save-npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# 프로젝트 루트(2d_segmentation_picking)를 import 경로에 추가
_PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_DIR))


def _f(x) -> float:
    """JSON 직렬화용 안전 float 변환 (NaN/inf 방어)."""
    x = float(x)
    return x if np.isfinite(x) else None


def _compute_stats(rgb: np.ndarray, xyz: np.ndarray, snr: np.ndarray) -> dict:
    """장면 통계를 dict로 계산한다."""
    h, w = xyz.shape[:2]
    valid = ~np.isnan(xyz).any(axis=2)
    n_valid = int(valid.sum())
    n_total = h * w

    stats: dict = {
        "resolution": {"height": h, "width": w},
        "valid_points": {
            "count": n_valid,
            "total": n_total,
            "ratio": round(n_valid / n_total, 4) if n_total else None,
        },
    }

    if n_valid:
        z = xyz[:, :, 2][valid]
        stats["z_mm"] = {
            "min": _f(np.min(z)), "max": _f(np.max(z)),
            "mean": _f(np.mean(z)), "median": _f(np.median(z)),
            "p1": _f(np.percentile(z, 1)), "p99": _f(np.percentile(z, 99)),
        }
        xyz_valid = xyz[valid]
        stats["bbox_mm"] = {
            "x": [_f(xyz_valid[:, 0].min()), _f(xyz_valid[:, 0].max())],
            "y": [_f(xyz_valid[:, 1].min()), _f(xyz_valid[:, 1].max())],
            "z": [_f(xyz_valid[:, 2].min()), _f(xyz_valid[:, 2].max())],
        }
        if snr is not None:
            s = snr[valid]
            stats["snr"] = {
                "min": _f(np.min(s)), "max": _f(np.max(s)),
                "mean": _f(np.mean(s)), "median": _f(np.median(s)),
            }

    # RGB 채널별 평균 + 전체 밝기(luminance) — rgba vs rgba_srgb 비교의 참고치
    stats["rgb_mean"] = [round(float(rgb[:, :, c].mean()), 2) for c in range(3)]
    stats["luminance_mean"] = round(
        float((0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]).mean()), 2
    )
    return stats


def main() -> None:
    # Windows 콘솔(cp1252)에서 한글 JSON 출력이 깨지지 않도록 utf-8 재설정
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(description="ZDF 씬 통계 및 RGB 포맷 비교")
    parser.add_argument("--input", type=Path, default=_PROJECT_DIR / "data/input/image_test.zdf")
    parser.add_argument("--output-dir", type=Path, default=_PROJECT_DIR / "data/output")
    parser.add_argument("--save-npz", action="store_true", help="장면을 npz로도 저장")
    args = parser.parse_args()

    import cv2
    import zivid

    if not args.input.exists():
        raise SystemExit(f"입력 파일이 없습니다: {args.input}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.input.stem

    with zivid.Application():
        frame = zivid.Frame(args.input)
        point_cloud = frame.point_cloud()

        xyz = point_cloud.copy_data("xyz")
        snr = point_cloud.copy_data("snr")
        rgba = point_cloud.copy_data("rgba")            # linear
        rgba_srgb = point_cloud.copy_data("rgba_srgb")  # sRGB

    # 두 RGB 포맷 PNG 저장 (cv2는 BGR이므로 RGB→BGR 변환)
    png_linear = args.output_dir / f"{stem}_rgba_linear.png"
    png_srgb = args.output_dir / f"{stem}_rgba_srgb.png"
    cv2.imwrite(str(png_linear), cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(png_srgb), cv2.cvtColor(rgba_srgb[:, :, :3], cv2.COLOR_RGB2BGR))

    # 통계는 채택 후보인 rgba_srgb 기준으로 계산 (밝기 비교치는 아래 별도 기록)
    stats = _compute_stats(rgba_srgb[:, :, :3], xyz, snr)
    stats["input"] = str(args.input)
    stats["rgb_format_compare"] = {
        "rgba_linear_luminance_mean": round(
            float((0.299 * rgba[:, :, 0] + 0.587 * rgba[:, :, 1] + 0.114 * rgba[:, :, 2]).mean()), 2),
        "rgba_srgb_luminance_mean": round(
            float((0.299 * rgba_srgb[:, :, 0] + 0.587 * rgba_srgb[:, :, 1] + 0.114 * rgba_srgb[:, :, 2]).mean()), 2),
        "note": "값이 큰(밝은) 쪽이 대체로 자연영상에 가깝다. PNG를 육안 비교해 최종 채택.",
    }
    stats["outputs"] = {"rgba_linear_png": str(png_linear), "rgba_srgb_png": str(png_srgb)}

    if args.save_npz:
        from core.loader import save_npz
        from core.types import SceneData

        npz_path = args.output_dir / f"{stem}.npz"
        save_npz(SceneData(rgb=np.ascontiguousarray(rgba_srgb[:, :, :3]), xyz=xyz, snr=snr), npz_path)
        stats["outputs"]["npz"] = str(npz_path)

    stats_path = args.output_dir / f"{stem}_scene_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"\n[저장됨] {stats_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
