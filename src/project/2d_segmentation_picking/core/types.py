"""공통 데이터 계약 (PLAN.md 2장).

Zivid 네이티브 데이터 형태를 기준으로 하며, gripper-agnostic하게 유지한다.
- 2D와 3D가 픽셀 단위로 정렬되어 있으므로 마스크의 (row, col)을 xyz[row, col]에
  그대로 사용할 수 있다. backprojection이 불필요하므로 intrinsics 필드는 두지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np


@dataclass
class SceneData:
    """한 장면의 정렬된 2D + 3D 데이터.

    Attributes:
        rgb: (H, W, 3) uint8. copy_data("rgba_srgb")[:, :, :3] 로 생성.
        xyz: (H, W, 3) float32. 단위 mm, 무효값은 NaN. copy_data("xyz").
        snr: (H, W) float32 또는 None. 신뢰도 필터용. copy_data("snr").
        normals: (H, W, 3) float32 또는 None. 픽셀당 단위 법선, [-1,1], 무효값 NaN.
            copy_data("normals"). 노말맵 실험(scripts/run_normal_experiment.py)에서만 쓴다.
    """

    rgb: np.ndarray
    xyz: np.ndarray
    snr: np.ndarray | None = None
    normals: np.ndarray | None = None

    @property
    def height(self) -> int:
        return int(self.rgb.shape[0])

    @property
    def width(self) -> int:
        return int(self.rgb.shape[1])

    @cached_property
    def valid_mask(self) -> np.ndarray:
        """유효한(비 NaN) 3D 포인트를 나타내는 (H, W) bool 마스크.

        **캐시된다.** 1224x1024 전체 NaN 검사가 약 10ms인데 기하 후처리에서 마스크마다
        호출되므로(수십 회) 매번 재계산하면 그것만으로 수백 ms를 쓴다.
        xyz를 나중에 바꾸면 캐시가 낡으므로, 바꿀 일이 있으면 SceneData를 새로 만든다.
        """
        return ~np.isnan(self.xyz).any(axis=2)


@dataclass
class PickCandidate:
    """피킹 후보. 최종 요구 출력은 center_px이며, 포즈 관련 필드는 방법별로 채운다."""

    center_px: tuple[int, int]  # (row, col) — 최종 요구 출력
    mask: np.ndarray | None = None  # (H, W) bool
    position_mm: np.ndarray | None = None  # (3,) 카메라 좌표
    normal: np.ndarray | None = None  # (3,) 카메라 쪽을 향함
    score: float = 0.0
    plane_rms_mm: float | None = None
    meta: dict = field(default_factory=dict)  # 방법별 부가정보
