"""마스크 평면 피팅 / 기하 병합·기각 (PLAN.md 4장, 5장 m1 5·6단계).

RGB 세그멘테이션 결과를 XYZ로 정련한다. 기하를 **주 세그멘터**로 쓰는 것(RANSAC+DBSCAN)은
PLAN 5장에서 폐기했지만 — 밀착·중첩된 연성 파우치에 깊이 불연속이 없다 — **보조 신호**로
쓰는 것은 별개이며 다음 네 가지에 유효하다.

  1) 병합: 인접 마스크의 평면 normal 각도차 + offset이 작으면 같은 물체 (과분할 해소)
  2) 기각: plane RMS 과대, 깊이 범위 과대 → 여러 물체를 걸친 마스크 (정밀도)
  3) 중복 제거: 마스크 IoU 기준 (AMG multimask_output=true의 부산물)
  4) 기울기 기각: bin frame n_up 기준 절대 기울기 → 빈 사면 벽 잔류 제거

좌표계 주의 (PLAN 4장): 병합의 각도차는 두 normal의 **상대량**이라 좌표계 무관하다.
반면 tilt는 **절대량**이므로 반드시 bin frame(n_up) 기준으로 계산한다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MaskPlane:
    """한 마스크의 평면 피팅 결과. 단위는 mm.

    지표가 셋인 이유: **인라이어 rms만으로는 평면성을 판정할 수 없다.** 인라이어는
    "평면에서 thresh_mm 이내"로 정의되므로 rms_mm은 구조적으로 thresh_mm 이하로 묶인다.
    두 물체를 걸친 마스크도 rms_mm은 작게 나온다. 실제 판별력은 `inlier_ratio`(지배
    평면이 마스크를 얼마나 설명하는가)와 `rms_all_mm`/`depth_span_mm`(전체 포인트 기준)에 있다.
    """

    normal: np.ndarray        # (3,) 단위 법선, 카메라 쪽을 향함
    centroid: np.ndarray      # (3,) 인라이어 무게중심
    rms_mm: float             # 인라이어 기준 RMS — thresh_mm에 묶인다. 단독 사용 금지
    rms_all_mm: float         # 마스크 전체 포인트 기준 RMS — 평면성의 실제 지표
    inlier_ratio: float       # 인라이어 / 전체 — 지배 평면의 설명력. 낮으면 여러 물체
    n_points: int             # 피팅에 사용한 전체 포인트 수 (서브샘플 후)
    n_inliers: int
    depth_span_mm: float      # 법선 방향 두께(1~99 퍼센타일, 전체 포인트) — 걸치면 커진다
    tilt_deg: float | None    # n_up과의 각도. n_up 미지정이면 None

    def as_meta(self) -> dict:
        """리포트/meta에 넣을 직렬화 가능한 요약."""
        return {
            "plane_rms_mm": round(self.rms_mm, 3),
            "plane_rms_all_mm": round(self.rms_all_mm, 3),
            "inlier_ratio": round(self.inlier_ratio, 3),
            "plane_points": self.n_points,
            "depth_span_mm": round(self.depth_span_mm, 2),
            "tilt_deg": None if self.tilt_deg is None else round(self.tilt_deg, 1),
        }


# ---------------------------------------------------------------- 마스크 유틸

def bbox_of(mask: np.ndarray) -> tuple[int, int, int, int]:
    """마스크의 (r0, r1, c0, c1) 경계 상자(양끝 포함). 빈 마스크는 (-1,-1,-1,-1)."""
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        return (-1, -1, -1, -1)
    return (int(rows[0]), int(rows[-1]), int(cols[0]), int(cols[-1]))


def _boxes_overlap(a: tuple, b: tuple, margin: int = 0) -> bool:
    ar0, ar1, ac0, ac1 = a
    br0, br1, bc0, bc1 = b
    if ar0 < 0 or br0 < 0:
        return False
    return not (ar1 + margin < br0 or br1 + margin < ar0
                or ac1 + margin < bc0 or bc1 + margin < ac0)


def mask_iou(a: np.ndarray, b: np.ndarray,
             box_a: tuple | None = None, box_b: tuple | None = None) -> float:
    """두 마스크의 IoU. 경계 상자로 먼저 걸러 전체 영상 AND를 피한다."""
    box_a = box_a if box_a is not None else bbox_of(a)
    box_b = box_b if box_b is not None else bbox_of(b)
    if not _boxes_overlap(box_a, box_b):
        return 0.0
    r0 = max(box_a[0], box_b[0]); r1 = min(box_a[1], box_b[1])
    c0 = max(box_a[2], box_b[2]); c1 = min(box_a[3], box_b[3])
    sa, sb = a[r0:r1 + 1, c0:c1 + 1], b[r0:r1 + 1, c0:c1 + 1]
    inter = int(np.count_nonzero(sa & sb))
    if inter == 0:
        return 0.0
    union = int(a.sum()) + int(b.sum()) - inter
    return inter / max(union, 1)


def masks_adjacent(a: np.ndarray, b: np.ndarray, dilate_px: int,
                   box_a: tuple | None = None, box_b: tuple | None = None) -> bool:
    """두 마스크의 경계가 dilate_px 이내로 실제로 맞닿는지.

    PLAN 4장: 마스크 병합은 **경계가 실제로 맞닿은 인접 쌍에만** 적용한다.
    떨어져 있는데 우연히 같은 평면인 두 물체를 병합하면 안 된다.
    """
    import cv2

    box_a = box_a if box_a is not None else bbox_of(a)
    box_b = box_b if box_b is not None else bbox_of(b)
    if not _boxes_overlap(box_a, box_b, margin=dilate_px):
        return False

    # 두 상자의 합집합 영역만 잘라서 판정한다 (dilate 경계 손실 방지용 여유 포함)
    pad = dilate_px + 1
    h, w = a.shape
    r0 = max(0, min(box_a[0], box_b[0]) - pad); r1 = min(h, max(box_a[1], box_b[1]) + pad + 1)
    c0 = max(0, min(box_a[2], box_b[2]) - pad); c1 = min(w, max(box_a[3], box_b[3]) + pad + 1)
    sa = a[r0:r1, c0:c1].astype(np.uint8)
    sb = b[r0:r1, c0:c1]
    k = np.ones((2 * dilate_px + 1, 2 * dilate_px + 1), np.uint8)
    grown = cv2.dilate(sa, k, iterations=1).astype(bool)
    return bool(np.any(grown & sb))


# ------------------------------------------------------------- 평면 피팅

def _pca_plane(points: np.ndarray):
    """무게중심 + 최소분산 방향(법선). 반환: (normal, centroid)."""
    centroid = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - centroid, full_matrices=False)
    normal = vt[2]
    return normal / max(np.linalg.norm(normal), 1e-12), centroid


def _fit_points(points: np.ndarray, thresh_mm: float, iterations: int = 3):
    """트리밍 PCA 평면 피팅 — PCA → thresh 밖 포인트 제외 → 재피팅 (반복).

    반환: (normal, centroid, rms_inlier, n_inliers, rms_all, depth_span). 실패 시 None.

    open3d의 RANSAC(`segment_plane`)을 쓰지 않는 이유:
      - 속도. 마스크가 수십 개라 호출 횟수가 많고, RANSAC은 마스크당 20ms대였다.
        트리밍 PCA는 순수 numpy로 1ms 미만이다.
      - 결정성. RANSAC은 난수를 써서 같은 입력에 결과가 흔들린다. 벤치마크에 부적합하다.
    bin frame(core/bin_frame.py)은 1회 계산이고 아웃라이어 비율이 높을 수 있어 RANSAC을
    그대로 유지한다. 여기 마스크는 대부분 단일 표면이라 트리밍 PCA로 충분하다.
    """
    if len(points) < 3:
        return None

    normal, centroid = _pca_plane(points)
    inlier = np.ones(len(points), dtype=bool)
    for _ in range(max(1, iterations)):
        dist = (points - centroid) @ normal
        new_inlier = np.abs(dist) <= thresh_mm
        if int(new_inlier.sum()) < 3:
            break                      # 트리밍이 과했다 — 직전 해를 유지
        inlier = new_inlier
        normal, centroid = _pca_plane(points[inlier])

    dist_all = (points - centroid) @ normal
    rms_all = float(np.sqrt(np.mean(dist_all ** 2)))
    rms_in = float(np.sqrt(np.mean(dist_all[inlier] ** 2)))
    # 법선 방향 두께: 아웃라이어에 흔들리지 않게 1~99 퍼센타일 폭 (전체 포인트 기준)
    depth_span = float(np.percentile(dist_all, 99) - np.percentile(dist_all, 1))
    return normal, centroid, rms_in, int(inlier.sum()), rms_all, depth_span


def fit_mask_plane(
    scene,
    mask: np.ndarray,
    erode_px: int = 3,
    max_fit_points: int = 2000,
    thresh_mm: float = 1.5,
    iterations: int = 3,
    min_points: int = 50,
    n_up: np.ndarray | None = None,
    box: tuple | None = None,
) -> MaskPlane | None:
    """마스크 픽셀의 XYZ에 평면을 피팅한다. 포인트 부족/피팅 실패 시 None.

    erode_px > 0 이면 피팅 전에 마스크를 침식한다 (PLAN 5장 m1 6단계) — 마스크 경계는
    물체와 배경의 depth가 섞여 아웃라이어가 몰리는 자리다. 침식으로 min_points 미만이
    되면 원본 마스크로 되돌린다(작은 물체 보호).

    **경계 상자 안에서만 연산한다.** 마스크는 보통 전체 영상의 일부인데 1224x1024 전체에
    erode/불리언 인덱싱을 하면 마스크당 십수 ms가 낭비된다(마스크 수십 개 × = 수백 ms).
    """
    import cv2

    box = box if box is not None else bbox_of(mask)
    r0, r1, c0, c1 = box
    if r0 < 0:
        return None
    pad = erode_px + 1  # 침식 커널이 상자 경계에서 잘리지 않도록 여유
    h, w = mask.shape
    r0 = max(0, r0 - pad); r1 = min(h - 1, r1 + pad)
    c0 = max(0, c0 - pad); c1 = min(w - 1, c1 + pad)

    sub = mask[r0:r1 + 1, c0:c1 + 1]
    use = sub
    if erode_px > 0:
        k = np.ones((2 * erode_px + 1, 2 * erode_px + 1), np.uint8)
        eroded = cv2.erode(sub.astype(np.uint8), k, iterations=1).astype(bool)
        if eroded.sum() >= min_points:
            use = eroded

    pts = scene.xyz[r0:r1 + 1, c0:c1 + 1][use & scene.valid_mask[r0:r1 + 1, c0:c1 + 1]]
    if len(pts) < min_points:
        return None

    # 균일 스트라이드 서브샘플 — 결정적이고(난수 없음) 피팅 시간을 상수로 묶는다
    if len(pts) > max_fit_points:
        pts = pts[:: max(1, len(pts) // max_fit_points)][:max_fit_points]

    fit = _fit_points(np.asarray(pts, dtype=np.float64), thresh_mm, iterations)
    if fit is None:
        return None
    normal, centroid, rms_in, n_in, rms_all, depth_span = fit

    # 카메라 쪽을 향하도록 정렬 (Zivid 카메라 +Z가 장면 방향 → 카메라 쪽은 -Z)
    if normal[2] > 0:
        normal = -normal

    tilt = None
    if n_up is not None:
        cos = float(np.clip(np.dot(normal, np.asarray(n_up, dtype=float)), -1.0, 1.0))
        tilt = float(np.degrees(np.arccos(abs(cos))))  # 앞뒤 뒤집힘 무관

    return MaskPlane(
        normal=normal, centroid=centroid, rms_mm=rms_in, rms_all_mm=rms_all,
        inlier_ratio=n_in / max(len(pts), 1), n_points=int(len(pts)), n_inliers=n_in,
        depth_span_mm=depth_span, tilt_deg=tilt,
    )


# ------------------------------------------------------- 중복 제거 / 병합

def dedupe_by_iou(masks: list[np.ndarray], scores: list[float], iou_thresh: float) -> list[int]:
    """마스크 IoU 기준 중복 제거. 점수 높은 쪽을 남기고 남길 인덱스를 반환한다.

    AMG는 자체 NMS를 box 기준으로 하므로 multimask_output=true에서 마스크 수준 중복이
    남는다. "많이 잡히지만 중복" 상태를 정리한다.
    """
    boxes = [bbox_of(m) for m in masks]
    order = sorted(range(len(masks)), key=lambda i: scores[i], reverse=True)
    keep: list[int] = []
    for i in order:
        if all(mask_iou(masks[i], masks[j], boxes[i], boxes[j]) <= iou_thresh for j in keep):
            keep.append(i)
    return keep


def _plane_offset_mm(pa: MaskPlane, pb: MaskPlane) -> float:
    """두 평면의 상대 오프셋 — 서로의 무게중심을 상대 평면에 투영한 거리의 최댓값."""
    da = abs(float((pb.centroid - pa.centroid) @ pa.normal))
    db = abs(float((pa.centroid - pb.centroid) @ pb.normal))
    return max(da, db)


def normal_angle_deg(na: np.ndarray, nb: np.ndarray) -> float:
    """두 법선의 각도차(도). 상대량이므로 좌표계와 무관하다 (PLAN 4장)."""
    cos = float(np.clip(np.dot(na, nb), -1.0, 1.0))
    return float(np.degrees(np.arccos(abs(cos))))


def merge_coplanar(
    masks: list[np.ndarray],
    planes: list[MaskPlane | None],
    normal_deg: float = 8.0,
    offset_mm: float = 3.0,
    dilate_px: int = 5,
) -> list[list[int]]:
    """인접하고 동일 평면인 마스크들을 그룹으로 묶는다. 반환: 인덱스 그룹 리스트.

    조건: (1) 경계가 dilate_px 이내로 맞닿음  (2) normal 각도차 <= normal_deg
          (3) 평면 오프셋 <= offset_mm

    union-find 1회 통과다. 병합 후 재피팅하고 다시 병합하는 반복은 하지 않는다 —
    한 물체가 여러 조각으로 쪼개진 경우 대부분 1회로 붙고, 반복하면 서로 다른 물체가
    연쇄적으로 이어붙을 위험이 커진다.
    """
    n = len(masks)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    boxes = [bbox_of(m) for m in masks]
    for i in range(n):
        if planes[i] is None:
            continue
        for j in range(i + 1, n):
            if planes[j] is None or find(i) == find(j):
                continue
            if normal_angle_deg(planes[i].normal, planes[j].normal) > normal_deg:
                continue
            if _plane_offset_mm(planes[i], planes[j]) > offset_mm:
                continue
            if not masks_adjacent(masks[i], masks[j], dilate_px, boxes[i], boxes[j]):
                continue
            union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


# ------------------------------------------------------- 통합 정련 파이프라인

def refine_masks(scene, segs, scores, metas, n_up, cfg: dict):
    """RGB 마스크를 XYZ로 정련하는 공용 단계. m1/m2가 공유한다.

    중복 제거 → 평면 피팅 → 동일 평면 인접 병합 → 재피팅 → 기각.

    Args:
        segs: (H,W) bool 마스크 리스트
        scores: 중복 제거 시 우선순위(높은 쪽을 남긴다)
        metas: 마스크별 부가정보 dict 리스트 (평면 지표가 병합되어 반환된다)
        n_up: bin frame 위 방향. tilt 계산용. None이면 tilt 없음
        cfg: config의 `geometry` 블록

    Returns:
        (segs, metas, planes, stats) — stats는 단계별 개수와 기각 사유별 카운트.
        `cfg["enabled"]`가 false면 입력을 그대로 통과시킨다(순수 RGB 결과 비교용).
    """
    stats = {"n_masks_dedup": len(segs), "n_groups_merged": 0, "n_masks_merged": len(segs),
             "rejected": {"dup": 0, "plane_fit": 0, "plane_rms": 0, "inlier_ratio": 0,
                          "depth_span": 0, "tilt": 0}}
    if not cfg.get("enabled", False) or not segs:
        return segs, metas, [None] * len(segs), stats

    rej = stats["rejected"]
    fit_kw = dict(erode_px=cfg["erode_px"], max_fit_points=cfg["max_fit_points"],
                  thresh_mm=cfg["inlier_thresh_mm"], iterations=cfg["trim_iterations"],
                  min_points=cfg["min_plane_points"], n_up=n_up)

    # (1) 마스크 수준 중복 제거 — AMG/DINO의 NMS는 box 기준이라 마스크 중복이 남는다
    iou_thresh = cfg.get("dedupe_iou")
    if iou_thresh is not None and len(segs) > 1:
        keep = sorted(dedupe_by_iou(segs, scores, float(iou_thresh)))
        rej["dup"] = len(segs) - len(keep)
        segs = [segs[i] for i in keep]
        metas = [metas[i] for i in keep]
    stats["n_masks_dedup"] = len(segs)

    # (2) 마스크별 평면 피팅 (경계 상자는 1회만 계산해 재사용)
    boxes = [bbox_of(s) for s in segs]
    planes = [fit_mask_plane(scene, s, box=b, **fit_kw) for s, b in zip(segs, boxes)]

    # (3) 동일 평면 인접 마스크 병합 → 병합 마스크 재피팅 (PLAN 5장 m1 5·6단계)
    mg = cfg.get("merge", {})
    if mg.get("enabled", False) and len(segs) > 1:
        groups = merge_coplanar(segs, planes, normal_deg=mg["normal_deg"],
                                offset_mm=mg["offset_mm"], dilate_px=mg["dilate_px"])
        new_segs, new_metas, new_planes = [], [], []
        for grp in sorted(groups, key=min):
            grp = sorted(grp)
            if len(grp) == 1:
                new_segs.append(segs[grp[0]])
                new_metas.append(metas[grp[0]])
                new_planes.append(planes[grp[0]])
                continue
            stats["n_groups_merged"] += 1
            union = np.zeros_like(segs[grp[0]])
            for i in grp:
                union |= segs[i]
            new_segs.append(union)
            new_metas.append({**metas[grp[0]], "merged_from": len(grp)})
            new_planes.append(fit_mask_plane(scene, union, **fit_kw))
        segs, metas, planes = new_segs, new_metas, new_planes
    stats["n_masks_merged"] = len(segs)

    # (4) 기각 — 임계는 전부 config. 기본 null(끔)이며 실데이터 분포를 보고 정한다.
    rj = cfg.get("reject", {})
    max_rms, min_ratio = rj.get("max_plane_rms_all_mm"), rj.get("min_inlier_ratio")
    max_span, max_tilt = rj.get("max_depth_span_mm"), rj.get("max_tilt_deg")
    require_plane = rj.get("require_plane", True)

    out_segs, out_metas, out_planes = [], [], []
    for seg, meta, pl in zip(segs, metas, planes):
        if pl is None:
            # 유효 XYZ가 부족한 마스크 — 투명/반사 물체의 NaN 홀에서 발생한다(PLAN 9장).
            # 기하 평가가 불가능하므로 기본은 기각하되, 계측해서 드러낸다.
            if require_plane:
                rej["plane_fit"] += 1
                continue
            out_segs.append(seg); out_metas.append(meta); out_planes.append(None)
            continue
        # rms는 전체 포인트 기준(rms_all). 인라이어 rms는 thresh에 묶여 판별력이 없다
        if max_rms is not None and pl.rms_all_mm > max_rms:
            rej["plane_rms"] += 1
            continue
        if min_ratio is not None and pl.inlier_ratio < min_ratio:
            rej["inlier_ratio"] += 1
            continue
        if max_span is not None and pl.depth_span_mm > max_span:
            rej["depth_span"] += 1
            continue
        if max_tilt is not None and pl.tilt_deg is not None and pl.tilt_deg > max_tilt:
            rej["tilt"] += 1
            continue
        out_segs.append(seg)
        out_metas.append({**meta, **pl.as_meta()})
        out_planes.append(pl)
    return out_segs, out_metas, out_planes, stats
