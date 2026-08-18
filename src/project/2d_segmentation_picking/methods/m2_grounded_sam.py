"""m2 — Grounded-SAM 2 (PLAN.md 5장 m2).

Grounding DINO(text→bbox, HuggingFace `grounding-dino-tiny`, 순수 PyTorch) +
SAM2(bbox→mask). 텍스트 프롬프트로 인스턴스 단위 분할을 얻는다.

원본 Grounding DINO 레포는 커스텀 CUDA op 빌드가 필요하므로, transformers의
`IDEA-Research/grounding-dino-tiny`를 써서 컴파일 없이 동작시킨다(단일 환경 요구).

이 단계도 **순수 세그멘테이션**이다. 마스크는 2D(Grounding DINO+SAM2)에서 나오고,
포즈(그리퍼 의존)는 Phase 4로 미룬다. position_mm(center의 XYZ)만 참고로 채운다.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from core import roi as roi_mod
from core.bin_frame import load_bin_frame
from core.types import PickCandidate
from methods.base import Segmenter


class M2GroundedSam(Segmenter):
    requires_xyz = True

    def __init__(self, config: dict, project_dir: Path):
        self.cfg = config
        self.project_dir = Path(project_dir)
        self._gd_model = None  # Grounding DINO
        self._sam_predictor = None
        self.stats: dict = {}  # 직전 predict의 단계별 통계 (튜닝/벤치마크용)

    def _resolve(self, p: str) -> Path:
        p = Path(p)
        return p if p.is_absolute() else (self.project_dir / p)

    def build(self):
        """Grounding DINO + SAM2 로드 (1회). 사이클 타임 측정 시 predict 전에 호출한다."""
        if self._gd_model is not None:
            return
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        gd = self.cfg["grounding_dino"]
        self.device = gd.get("device", "cuda")
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA를 사용할 수 없습니다. config device를 확인하세요.")
        self._gd_processor = AutoProcessor.from_pretrained(gd["model_id"])
        self._gd_model = AutoModelForZeroShotObjectDetection.from_pretrained(gd["model_id"]).to(self.device)

        s = self.cfg["sam2"]
        sam_model = build_sam2(s["config"], str(self._resolve(s["checkpoint"])), device=s.get("device", "cuda"))
        self._sam_predictor = SAM2ImagePredictor(sam_model)

    def _top_layer(self, scene):
        tl = self.cfg["top_layer"]
        roi_cfg = json.loads(self._resolve(tl["bin_roi"]).read_text(encoding="utf-8"))
        bf = load_bin_frame(self._resolve(tl["bin_frame"]))
        n_up, _, p_floor = bf.as_np()
        interior = roi_mod.bin_interior_mask(
            scene, roi_cfg["rim_outer_corners_rc"], roi_cfg["rim_band_mm"], shrink_mm=tl["interior_shrink_mm"])
        top_mask, _ = roi_mod.top_layer_mask(scene, n_up, p_floor, roi_2d=interior, band_mm=tl["top_band_mm"])
        return top_mask, n_up, p_floor

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> np.ndarray:
        """클래스 무관 NMS. 남길 인덱스를 점수 내림차순으로 반환한다.

        Grounding DINO는 텍스트 구(phrase)별로 박스를 내므로 같은 물체가 여러 라벨로
        중복 검출된다. threshold를 낮출 때 중복이 급증하므로 NMS로 정리한다.
        """
        order = scores.argsort()[::-1]
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            if order.size == 1:
                break
            rest = order[1:]
            iw = np.maximum(0.0, np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]))
            ih = np.maximum(0.0, np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest]))
            inter = iw * ih
            iou = inter / np.maximum(areas[i] + areas[rest] - inter, 1e-6)
            order = rest[iou <= iou_thresh]
        return np.array(keep, dtype=int)

    def _detect_boxes(self, rgb):
        """Grounding DINO로 텍스트 프롬프트에 해당하는 bbox(xyxy)/score/label 검출."""
        import torch
        from PIL import Image

        gd = self.cfg["grounding_dino"]
        image = Image.fromarray(rgb)
        inputs = self._gd_processor(images=image, text=self.cfg["text_prompt"], return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self._gd_model(**inputs)
        results = self._gd_processor.post_process_grounded_object_detection(
            outputs, inputs["input_ids"],
            threshold=gd["box_threshold"], text_threshold=gd["text_threshold"],
            target_sizes=[image.size[::-1]],  # (H, W)
        )[0]
        return results  # dict: boxes(xyxy), scores, labels/text_labels

    def predict(self, scene) -> list:
        import torch

        self.build()
        self.stats = {"top_layer_px": 0, "n_boxes": 0, "n_boxes_nms": 0, "n_candidates": 0,
                      "rejected": {"area": 0, "overlap": 0}}
        top_mask, n_up, p_floor = self._top_layer(scene)
        self.stats["top_layer_px"] = int(top_mask.sum())
        if self.stats["top_layer_px"] == 0:
            return []

        results = self._detect_boxes(scene.rgb)
        boxes = results["boxes"].detach().cpu().numpy()
        scores = results["scores"].detach().cpu().numpy()
        labels = results.get("text_labels", results.get("labels"))
        self.stats["n_boxes"] = int(len(boxes))
        if len(boxes) == 0:
            return []

        # 라벨 간 중복 박스 제거 (threshold를 낮추면 필수)
        nms_thresh = self.cfg["grounding_dino"].get("box_nms_thresh")
        if nms_thresh is not None and len(boxes) > 1:
            keep = self._nms(boxes, scores, float(nms_thresh))
            boxes, scores = boxes[keep], scores[keep]
            if labels is not None:
                labels = [labels[i] for i in keep]
        self.stats["n_boxes_nms"] = int(len(boxes))

        # SAM2로 각 bbox → 마스크
        self._sam_predictor.set_image(scene.rgb)
        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
            masks, _, _ = self._sam_predictor.predict(box=boxes, multimask_output=False)
        # masks: (N,1,H,W) 또는 (N,H,W)
        masks = np.asarray(masks)
        if masks.ndim == 4:
            masks = masks[:, 0]
        masks = masks.astype(bool)

        h_map = roi_mod.height_map(scene, n_up, p_floor)
        f = self.cfg["filter"]
        rejected = {"area": 0, "overlap": 0}
        # 빈 자체가 "box"/"container"로 검출되어 최상층 전체를 덮는 마스크를 막는다
        max_area = f.get("max_area_frac", 1.0) * self.stats["top_layer_px"]

        # --- 면적 / 최상층 겹침 필터 ---
        segs, keep_scores, metas = [], [], []
        for i, seg in enumerate(masks):
            area = int(seg.sum())
            if area < f["min_area_px"] or area > max_area:
                rejected["area"] += 1
                continue
            overlap = (seg & top_mask).sum() / max(area, 1)
            if overlap < f["min_overlap"]:
                rejected["overlap"] += 1
                continue
            label = labels[i] if labels is not None and i < len(labels) else ""
            segs.append(seg)
            keep_scores.append(float(scores[i]))
            metas.append({
                "label": str(label),
                "box_score": round(float(scores[i]), 3),
                "overlap": round(float(overlap), 3),
            })
        self.stats["n_masks_filtered"] = len(segs)

        # --- 기하 후처리 (core/plane.py, m1과 동일 코드) ---
        # 병합은 기본 off다: m2의 마스크는 이미 인스턴스 단위이고, 겹쳐 쌓인 동일 평면
        # 물체를 병합하면 m2의 장점인 인스턴스 분리를 스스로 훼손한다.
        from core import plane as plane_mod

        segs, metas, planes, g_stats = plane_mod.refine_masks(
            scene, segs, keep_scores, metas, n_up, self.cfg.get("geometry", {}))
        rejected.update(g_stats.pop("rejected"))
        self.stats.update(g_stats)

        candidates = []
        for seg, meta, pl in zip(segs, metas, planes):
            ys, xs = np.where(seg)
            cy, cx = ys.mean(), xs.mean()
            k = np.argmin((ys - cy) ** 2 + (xs - cx) ** 2)
            center = (int(ys[k]), int(xs[k]))

            hv = h_map[seg]
            hv = hv[~np.isnan(hv)]
            mean_h = float(hv.mean()) if hv.size else float("nan")

            pos = scene.xyz[center]
            if np.isnan(pos).any():
                xyz_seg = scene.xyz[seg]
                xyz_seg = xyz_seg[~np.isnan(xyz_seg).any(axis=1)]
                pos = np.median(xyz_seg, axis=0) if len(xyz_seg) else None

            candidates.append(PickCandidate(
                center_px=center, mask=seg,
                position_mm=None if pos is None else np.asarray(pos, dtype=float),
                normal=None if pl is None else pl.normal,
                plane_rms_mm=None if pl is None else pl.rms_mm,
                score=float(meta["box_score"]),  # Grounding DINO 신뢰도
                meta={
                    "area_px": int(seg.sum()),
                    "mean_h_mm": None if np.isnan(mean_h) else round(mean_h, 2),
                    **meta,
                },
            ))

        candidates.sort(key=lambda c: c.score, reverse=True)  # 검출 신뢰도 내림차순
        self.stats.update({"n_candidates": len(candidates), "rejected": rejected})
        return candidates
