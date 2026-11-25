"""Bin frame — 카메라 기울기 불변성 (PLAN.md 4장).

림(rim) 환형 픽셀의 XYZ에 RANSAC 평면을 피팅해 위 방향 n_up과 바닥 기준 p_floor를 얻는다.
카메라 좌표계 Z를 높이로 쓰면 기울인 카메라에서 왜곡되므로 반드시 이 프레임 기준으로
높이/기울기를 판정한다.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass
class BinFrame:
    n_up: list          # (3,) 빈 밖(카메라 쪽)을 향하는 단위 법선
    p_rim: list         # (3,) 림 평면 위의 대표점 (인라이어 평균)
    p_floor: list       # (3,) 바닥 기준점 = p_rim + n_down * bin_depth_mm
    bin_depth_mm: float
    rms_mm: float       # 림 평면 피팅 RMS (검증 지표, < 2mm 목표)
    n_inliers: int

    def as_np(self):
        return np.array(self.n_up), np.array(self.p_rim), np.array(self.p_floor)


def fit_plane_ransac(points: np.ndarray, distance_threshold: float = 1.5,
                     num_iterations: int = 2000):
    """open3d RANSAC로 평면 피팅. 반환: (normal(3,), point_on_plane(3,), rms, inlier_idx)."""
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold, ransac_n=3, num_iterations=num_iterations)
    a, b, c, d = plane_model
    normal = np.array([a, b, c], dtype=float)
    norm = np.linalg.norm(normal)
    normal /= norm
    inlier_pts = points[inliers]
    # 인라이어의 평면까지 부호거리 → RMS
    dist = (inlier_pts @ normal) + d / norm
    rms = float(np.sqrt(np.mean(dist ** 2)))
    p_on_plane = inlier_pts.mean(axis=0)
    return normal, p_on_plane, rms, inliers


def fit_bin_frame(scene, rim_mask: np.ndarray, bin_depth_mm: float,
                  distance_threshold: float = 1.5) -> BinFrame:
    """림 마스크 픽셀의 XYZ로 bin frame을 계산한다."""
    valid = rim_mask & scene.valid_mask
    pts = scene.xyz[valid]
    if len(pts) < 50:
        raise ValueError(f"림 유효 포인트가 너무 적음: {len(pts)}개")

    normal, p_rim, rms, inliers = fit_plane_ransac(pts, distance_threshold)

    # n_up은 빈 밖(카메라 쪽)을 향하도록 정렬. Zivid 카메라 +Z는 장면 방향이므로
    # 카메라 쪽 = -Z. normal[2] > 0 이면 뒤집어 위쪽이 카메라를 향하게 한다.
    if normal[2] > 0:
        normal = -normal
    n_up = normal
    n_down = -n_up
    p_floor = p_rim + n_down * bin_depth_mm

    return BinFrame(
        n_up=n_up.tolist(), p_rim=p_rim.tolist(), p_floor=p_floor.tolist(),
        bin_depth_mm=float(bin_depth_mm), rms_mm=round(rms, 4), n_inliers=int(len(inliers)),
    )


def fit_bin_frame_from_floor(scene, floor_mask: np.ndarray,
                             distance_threshold: float = 1.5) -> BinFrame:
    """빈 내부 바닥 픽셀에 직접 평면을 피팅한다 (림 밴드 대신).

    빈이 비어 있는 캘리브레이션 전용 장면에서만 쓴다. 림은 폭 20mm 안팎의 얇은
    밴드라 코너가 조금만 틀어져도 안쪽 사면 벽을 살짝 물어 n_up이 틀어지고,
    그 오차가 멀리 퍼진 평면(빈 전체)에서 크게 벌어진다. 바닥 전체로 피팅하면
    표본이 훨씬 크고 넓어 코너 오차에 덜 민감하다. 이미 바닥 자체를 피팅하므로
    bin_depth_mm=0, p_rim=p_floor로 둔다(필드 이름과 의미가 다르지만 재사용).
    """
    valid = floor_mask & scene.valid_mask
    pts = scene.xyz[valid]
    if len(pts) < 50:
        raise ValueError(f"바닥 유효 포인트가 너무 적음: {len(pts)}개")

    normal, p_on_plane, rms, inliers = fit_plane_ransac(pts, distance_threshold)
    if normal[2] > 0:
        normal = -normal
    n_up = normal

    return BinFrame(
        n_up=n_up.tolist(), p_rim=p_on_plane.tolist(), p_floor=p_on_plane.tolist(),
        bin_depth_mm=0.0, rms_mm=round(rms, 4), n_inliers=int(len(inliers)),
    )


def save_bin_frame(bf: BinFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(bf), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_bin_frame(path: str | Path) -> BinFrame:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return BinFrame(**data)
