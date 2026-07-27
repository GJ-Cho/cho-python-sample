"""시각화 유틸 (PLAN.md 6장 core/viz.py).

- 2D: 마스크 오버레이, 후보 center_px 마커
- 3D: SceneData → open3d 포인트 클라우드, 4x4 포즈 → 좌표계 렌더
open3d 창을 띄우는 show_* 함수는 헤드리스 환경에서는 호출하지 말 것.
"""

from __future__ import annotations

import numpy as np

# 마스크/후보 구분용 기본 색상 (BGR, cv2 기준)
_PALETTE_BGR = [
    (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255),
    (255, 0, 255), (255, 255, 0), (0, 128, 255), (128, 0, 255),
]


def overlay_masks(rgb: np.ndarray, masks, alpha: float = 0.5) -> np.ndarray:
    """RGB 이미지 위에 boolean 마스크(들)를 반투명 색으로 겹친다.

    Args:
        rgb: (H, W, 3) uint8
        masks: (H, W) bool 하나 또는 그 리스트
        alpha: 마스크 색 불투명도 0~1
    Returns:
        (H, W, 3) uint8 오버레이 결과
    """
    if isinstance(masks, np.ndarray):
        masks = [masks]
    out = rgb.astype(np.float32).copy()
    for i, mask in enumerate(masks):
        color = np.array(_PALETTE_BGR[i % len(_PALETTE_BGR)][::-1], dtype=np.float32)  # RGB
        sel = mask.astype(bool)
        out[sel] = (1.0 - alpha) * out[sel] + alpha * color
    return out.clip(0, 255).astype(np.uint8)


def draw_candidates_2d(rgb: np.ndarray, candidates, radius: int = 6) -> np.ndarray:
    """후보들의 center_px를 이미지에 마커로 표시한다 (score 순위 라벨 포함)."""
    import cv2

    out = np.ascontiguousarray(rgb.copy())
    for rank, cand in enumerate(candidates):
        row, col = cand.center_px
        color = _PALETTE_BGR[rank % len(_PALETTE_BGR)]
        cv2.circle(out, (int(col), int(row)), radius, color, 2)
        cv2.putText(out, str(rank), (int(col) + radius, int(row)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out


def scene_to_o3d_pointcloud(scene, snr_min: float | None = None):
    """SceneData → open3d PointCloud. NaN 및 (옵션) 저 SNR 포인트는 제외한다."""
    import open3d as o3d

    valid = scene.valid_mask
    if snr_min is not None and scene.snr is not None:
        valid = valid & (scene.snr > snr_min)

    xyz = scene.xyz[valid].astype(np.float64)  # mm
    rgb = scene.rgb[valid].astype(np.float64) / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    return pcd


def make_frame(pose4x4: np.ndarray | None = None, size: float = 30.0):
    """4x4 포즈에 놓인 좌표계 메쉬를 만든다 (size 단위 mm). pose 없으면 원점."""
    import open3d as o3d

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    if pose4x4 is not None:
        frame.transform(np.asarray(pose4x4, dtype=np.float64))
    return frame


def show_scene(scene, poses=None, snr_min: float | None = None) -> None:
    """포인트 클라우드 + (옵션) 포즈 좌표계들을 open3d 창으로 표시한다."""
    import open3d as o3d

    geoms = [scene_to_o3d_pointcloud(scene, snr_min=snr_min), make_frame(size=50.0)]
    for pose in poses or []:
        geoms.append(make_frame(pose, size=30.0))
    o3d.visualization.draw_geometries(geoms)
