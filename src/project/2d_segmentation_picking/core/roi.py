"""ROI / 최상층 마스크 (PLAN.md 3장, 4장).

- 림 환형(annulus) 마스크: 외곽 4꼭짓점(시계방향) → 안쪽으로 band_mm 수축한 밴드.
  카메라가 기울어 있어 intrinsics 없이 mm↔px를 클릭 꼭짓점의 XYZ로 근사한다.
- top_layer_mask: bin frame(n_up, p_floor) 기준 높이로 최상층만 자동 선별.
  (수동 물체 ROI 아님 — PLAN 3(b))
"""

from __future__ import annotations

import numpy as np


def corner_xyz(scene, row: int, col: int, win: int = 3) -> np.ndarray:
    """(row, col) 주변 win 반경 창에서 NaN을 제외한 중앙값 XYZ를 반환한다."""
    h, w = scene.height, scene.width
    r0, r1 = max(0, row - win), min(h, row + win + 1)
    c0, c1 = max(0, col - win), min(w, col + win + 1)
    patch = scene.xyz[r0:r1, c0:c1].reshape(-1, 3)
    patch = patch[~np.isnan(patch).any(axis=1)]
    if len(patch) == 0:
        return np.array([np.nan, np.nan, np.nan], dtype=float)
    return np.median(patch, axis=0)


def estimate_mm_per_px(scene, corners_rc) -> float:
    """외곽 4꼭짓점의 인접 변에서 mm/px 비율을 추정(유효 변들의 중앙값)한다."""
    ratios = []
    n = len(corners_rc)
    for i in range(n):
        (r1, c1), (r2, c2) = corners_rc[i], corners_rc[(i + 1) % n]
        p1, p2 = corner_xyz(scene, r1, c1), corner_xyz(scene, r2, c2)
        if np.isnan(p1).any() or np.isnan(p2).any():
            continue
        mm = float(np.linalg.norm(p1 - p2))
        px = float(np.hypot(r1 - r2, c1 - c2))
        if px > 1e-6:
            ratios.append(mm / px)
    if not ratios:
        raise ValueError("mm/px 추정 실패: 꼭짓점 근방에 유효한 XYZ가 없음")
    return float(np.median(ratios))


def _outer_fill_and_dist(scene, corners_rc, band_mm: float):
    """외곽 사각형 채움 마스크와 경계 거리맵, band_px를 계산한다 (내부 공용)."""
    import cv2

    h, w = scene.height, scene.width
    corners_cv = np.array([[c, r] for (r, c) in corners_rc], dtype=np.int32)  # (col,row)
    outer = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(outer, [corners_cv], 1)

    mm_per_px = estimate_mm_per_px(scene, corners_rc)
    band_px = max(1.0, band_mm / mm_per_px)
    dist = cv2.distanceTransform(outer, cv2.DIST_L2, 5)  # 외곽 경계까지의 거리
    return outer, dist, band_px


def rim_annulus_mask(scene, corners_rc, band_mm: float) -> np.ndarray:
    """외곽 사각형에서 안쪽으로 band_mm 두께의 림 밴드(annulus) bool 마스크."""
    outer, dist, band_px = _outer_fill_and_dist(scene, corners_rc, band_mm)
    return (outer > 0) & (dist > 0) & (dist <= band_px)


def bin_interior_mask(scene, corners_rc, band_mm: float, shrink_mm: float = 0.0) -> np.ndarray:
    """림 밴드 안쪽 = 빈 내부 영역 bool 마스크.

    별도 티칭 없이 림 4꼭짓점에서 밴드보다 더 안쪽(빈 개구부 내부)을 얻는다.
    최상층 판정을 빈 내부로 한정해 바깥 표면이 잡히는 것을 방지한다.

    shrink_mm > 0 이면 림에서 그만큼 더 안쪽부터 ROI로 삼는다. 빈의 사면 안쪽 벽이
    최상층으로 잡히는 것을 피하려면 벽 폭 정도로 준다(예: 60mm).
    """
    outer, dist, band_px = _outer_fill_and_dist(scene, corners_rc, band_mm)
    mm_per_px = estimate_mm_per_px(scene, corners_rc)
    inner_px = band_px + max(0.0, shrink_mm) / mm_per_px
    return (outer > 0) & (dist > inner_px)


def height_map(scene, n_up: np.ndarray, p_floor: np.ndarray) -> np.ndarray:
    """유효 픽셀의 높이 h = dot(p - p_floor, n_up). 무효는 NaN. (H,W) float."""
    h = np.full((scene.height, scene.width), np.nan, dtype=float)
    valid = scene.valid_mask
    h[valid] = (scene.xyz[valid] - np.asarray(p_floor)) @ np.asarray(n_up)
    return h


def top_layer_mask(
    scene,
    n_up: np.ndarray,
    p_floor: np.ndarray,
    roi_2d: np.ndarray | None = None,
    band_mm: float = 50.0,
    min_height_mm: float | None = None,
    max_height_mm: float | None = None,
    snr_min: float | None = None,
) -> tuple[np.ndarray, dict]:
    """최상층 마스크(자동). 상대 밴드 + 절대 경계 + (옵션) SNR (PLAN 3(b)).

    - roi_2d: 빈 내부 등 분석 영역 제한 마스크. 지정 시 퍼센타일/최상층을 이 영역 안에서만
      계산한다. 빈 바깥 표면이 더 높아 잡히는 것을 막는다(권장: bin_interior_mask).
    - 상대 밴드: h > p99 - band_mm  → 현재 장면의 최상층을 추적(장면 적응).
    - 절대 하한: h > min_height_mm  → 바닥/그 아래 제외.
    - 절대 상한: max_height_mm 지정 시 h < max_height_mm → 반사 등 상단 아웃라이어 제거.
      **주의**: 빈이 림 위로 쌓이면(overfill) 최상층 h가 bin_depth를 넘으므로,
      상한을 bin_depth로 두면 안 된다. None이면 상한 없음(상대 밴드가 상단을 결정).
    """
    region = scene.valid_mask
    if roi_2d is not None:
        region = region & roi_2d

    h = height_map(scene, n_up, p_floor)
    hv = h[region]
    if hv.size == 0:
        return np.zeros((scene.height, scene.width), dtype=bool), {"top_layer_px": 0, "error": "빈 ROI"}
    p99 = float(np.percentile(hv, 99))

    relative = h > (p99 - band_mm)      # 현재 장면(ROI 내) 최상층 추적
    mask = region & relative
    # 절대 경계는 bin_depth/ p_floor가 정확할 때만 유효하므로 지정 시에만 적용
    if min_height_mm is not None:
        mask = mask & (h > min_height_mm)
    if max_height_mm is not None:
        mask = mask & (h < max_height_mm)
    if snr_min is not None and scene.snr is not None:
        mask = mask & (scene.snr > snr_min)

    meta = {
        "roi_restricted": roi_2d is not None,
        "h_min_mm": round(float(np.min(hv)), 2),
        "h_p99_mm": round(p99, 2),
        "h_max_mm": round(float(np.max(hv)), 2),
        "band_mm": band_mm,
        "min_height_mm": min_height_mm,
        "max_height_mm": max_height_mm,
        "top_layer_px": int(mask.sum()),
    }
    return mask, meta
