"""m1 — 최상층 밴드 + SAM2 AMG 세그멘테이션 (PLAN.md 5장 m1).

절차:
  1) Phase 1의 top_layer_mask(빈 내부 최상층)를 구함
  2) SAM2 Automatic Mask Generator의 프롬프트 격자를 **최상층 내부에만** 배치
  3) multimask_output=False로 포인트당 마스크 1개 (과분할 억제)
  4) 최상층 겹침/면적으로 마스크 필터
  5) 높이(최상단 우선) 기준 랭킹 → PickCandidate

이 단계는 **순수 세그멘테이션**이다. 평면 피팅/파지 포즈(그리퍼 의존)는 Phase 4로 미룬다.
따라서 normal/plane_rms는 채우지 않는다(None). position_mm(center의 XYZ)만 참고로 채운다.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from core import roi as roi_mod
from core.bin_frame import load_bin_frame
from core.types import PickCandidate
from methods.base import Segmenter


class M1Sam2TopLayer(Segmenter):
    requires_xyz = True

    def __init__(self, config: dict, project_dir: Path):
        self.cfg = config
        self.project_dir = Path(project_dir)
        self._model = None  # 지연 빌드 (모델은 1회)
        self.stats: dict = {}  # 직전 predict의 단계별 통계 (튜닝/벤치마크용)

    # ---- 내부 유틸 ----
    def _resolve(self, p: str) -> Path:
        p = Path(p)
        return p if p.is_absolute() else (self.project_dir / p)

    def build(self):
        """SAM2 모델 로드 (1회). 사이클 타임 측정 시 predict 전에 호출한다."""
        if self._model is not None:
            return
        import torch
        from sam2.build_sam import build_sam2

        s = self.cfg["sam2"]
        self.device = s.get("device", "cuda")
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA를 사용할 수 없습니다. config sam2.device를 확인하세요.")
        self._model = build_sam2(s["config"], str(self._resolve(s["checkpoint"])), device=self.device)
        self._points_per_side = self.cfg["amg"]["points_per_side"]

    def _make_amg(self, grid: np.ndarray):
        """최상층 내부 프롬프트 격자로 AMG를 생성한다 (AMG는 생성 시 격자 필요)."""
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

        a = self.cfg["amg"]
        return SAM2AutomaticMaskGenerator(
            self._model,
            points_per_side=None,
            point_grids=[grid],
            pred_iou_thresh=a["pred_iou_thresh"],
            stability_score_thresh=a["stability_score_thresh"],
            min_mask_region_area=a["min_mask_region_area"],
            multimask_output=a["multimask_output"],
        )

    def _top_layer(self, scene):
        """bin_roi/bin_frame config로 최상층 마스크와 (n_up, p_floor)를 구한다."""
        tl = self.cfg["top_layer"]
        roi_cfg = json.loads(self._resolve(tl["bin_roi"]).read_text(encoding="utf-8"))
        bf = load_bin_frame(self._resolve(tl["bin_frame"]))
        n_up, _, p_floor = bf.as_np()

        interior = roi_mod.bin_interior_mask(
            scene, roi_cfg["rim_outer_corners_rc"], roi_cfg["rim_band_mm"],
            shrink_mm=tl["interior_shrink_mm"])
        top_mask, _ = roi_mod.top_layer_mask(
            scene, n_up, p_floor, roi_2d=interior, band_mm=tl["top_band_mm"])
        return top_mask, n_up, p_floor

    def _point_grid_in_mask(self, mask: np.ndarray) -> np.ndarray:
        """마스크 내부에 놓인 정규화(xy, [0,1]) 프롬프트 격자를 만든다.

        grid_mode:
          - "global": 전체 영상에 points_per_side 격자를 깔고 마스크 내부만 남긴다.
            최상층이 얇으면 살아남는 프롬프트가 몇 개뿐이라 검출이 급감한다.
          - "bbox"(기본): 최상층의 **바운딩 박스** 안에 points_per_side 격자를 깔아
            같은 프롬프트 수로 최상층 밀도를 크게 높인다.
        """
        h, w = mask.shape
        n = self._points_per_side
        mode = self.cfg["amg"].get("grid_mode", "bbox")

        if mode == "bbox":
            ys_idx, xs_idx = np.where(mask)
            r0, r1 = int(ys_idx.min()), int(ys_idx.max())
            c0, c1 = int(xs_idx.min()), int(xs_idx.max())
        else:
            r0, r1, c0, c1 = 0, h - 1, 0, w - 1

        ys = np.linspace(r0, r1, n).round().astype(int)
        xs = np.linspace(c0, c1, n).round().astype(int)
        pts = [[x / w, y / h] for y in ys for x in xs if mask[y, x]]
        return np.array(pts, dtype=np.float32)

    # ---- 메인 ----
    def predict(self, scene) -> list:
        import torch

        self.build()
        self.stats = {"top_layer_px": 0, "n_prompts": 0, "n_masks_amg": 0,
                      "n_candidates": 0, "rejected": {"area": 0, "overlap": 0, "no_height": 0}}
        top_mask, n_up, p_floor = self._top_layer(scene)
        top_area = int(top_mask.sum())
        self.stats["top_layer_px"] = top_area
        if top_area == 0:
            return []

        grid = self._point_grid_in_mask(top_mask)
        self.stats["n_prompts"] = int(len(grid))
        if len(grid) == 0:
            return []
        amg = self._make_amg(grid)  # 최상층 내부에만 프롬프트 배치

        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
            masks = amg.generate(scene.rgb)

        h_map = roi_mod.height_map(scene, n_up, p_floor)
        f = self.cfg["filter"]
        rejected = {"area": 0, "overlap": 0, "no_height": 0}
        candidates = []
        for m in masks:
            seg = m["segmentation"].astype(bool)
            area = int(seg.sum())
            if area < f["min_area_px"] or area > f["max_area_frac"] * top_area:
                rejected["area"] += 1
                continue
            # 프롬프트가 이미 최상층 내부에만 있으므로, 마스크는 최상층에 "닿기만" 하면 된다.
            # (물체는 밴드 아래로 이어지므로 마스크 대부분이 밴드 안에 있을 것을 요구하면 안 된다)
            overlap = (seg & top_mask).sum() / max(area, 1)
            if overlap < f["min_overlap"]:
                rejected["overlap"] += 1
                continue

            ys, xs = np.where(seg)
            cy, cx = ys.mean(), xs.mean()
            k = np.argmin((ys - cy) ** 2 + (xs - cx) ** 2)  # centroid를 마스크 픽셀로 스냅
            center = (int(ys[k]), int(xs[k]))

            hv = h_map[seg]
            hv = hv[~np.isnan(hv)]
            if hv.size == 0:
                rejected["no_height"] += 1
                continue
            mean_h = float(hv.mean())

            pos = scene.xyz[center]
            if np.isnan(pos).any():
                xyz_seg = scene.xyz[seg]
                xyz_seg = xyz_seg[~np.isnan(xyz_seg).any(axis=1)]
                pos = np.median(xyz_seg, axis=0) if len(xyz_seg) else None

            candidates.append(PickCandidate(
                center_px=center,
                mask=seg,
                position_mm=None if pos is None else np.asarray(pos, dtype=float),
                normal=None,          # Phase 4(파지)로 미룸
                plane_rms_mm=None,    # Phase 4
                score=mean_h,         # 최상단 우선
                meta={
                    "area_px": area,
                    "mean_h_mm": round(mean_h, 2),
                    "overlap": round(float(overlap), 3),
                    "predicted_iou": round(float(m.get("predicted_iou", 0)), 3),
                    "stability_score": round(float(m.get("stability_score", 0)), 3),
                },
            ))

        candidates.sort(key=lambda c: c.score, reverse=True)  # 높이 내림차순
        self.stats.update({
            "n_masks_amg": len(masks),
            "n_candidates": len(candidates),
            "rejected": rejected,
        })
        return candidates
