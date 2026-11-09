"""합성 장면 미리보기 + 자기검증 (venv_segpick 등 일반 Python, numpy + opencv).

  python preview.py <scene_dir 또는 상위 dir> [--out <png dir>] [--scale 0.5]

장면마다 [RGB | 인스턴스 오버레이 | depth | normals] 4분할 PNG 를 만든다.
검증 항목 (콘솔 출력):
  - normals 방향: 렌더 법선과 depth 역투영 법선의 코사인 중앙값 (1에 가까워야 함)
  - normals 부호: 카메라를 향하는 면의 z < 0 비율
  - top 플래그(박스): 최상층 인스턴스의 평균 depth 가 다른 인스턴스보다 작아야 함
"""
import argparse
import json
import os

import cv2
import numpy as np


def depth_normals(d_mm, K):
    z = d_mm.astype(np.float32) / 1000
    h, w = z.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    X = (u - K[0][2]) / K[0][0] * z
    Y = (v - K[1][2]) / K[1][1] * z
    P = np.stack([X, Y, z], -1)
    du = P[:, 2:] - P[:, :-2]
    dv = P[2:] - P[:-2]
    n = np.cross(du[1:-1], dv[:, 1:-1])
    n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-9)
    out = np.zeros((h, w, 3), np.float32)
    out[1:-1, 1:-1] = n
    return out


def check(scene):
    meta = json.load(open(os.path.join(scene, "meta.json"), encoding="utf-8"))
    rgb = cv2.imread(os.path.join(scene, "rgb.png"))
    d = cv2.imread(os.path.join(scene, "depth.png"), cv2.IMREAD_UNCHANGED)
    n8 = cv2.cvtColor(cv2.imread(os.path.join(scene, "normals.png")), cv2.COLOR_BGR2RGB)
    inst = cv2.imread(os.path.join(scene, "inst.png"), cv2.IMREAD_UNCHANGED)
    n = n8.astype(np.float32) / 127.5 - 1
    nd = depth_normals(d, meta["K"])
    # 경계(깊이 불연속)를 피해 평탄한 곳만 비교
    ok = (d > 0) & (np.linalg.norm(nd, axis=-1) > 0.5)
    lap = np.abs(cv2.Laplacian(d.astype(np.float32), cv2.CV_32F))
    ok &= lap < 3
    cos = np.sum(n * nd, -1)[ok]
    # depth 역투영 법선의 부호는 cross 순서에 따라 뒤집힐 수 있어 절대값과 부호를 따로 본다
    stats = dict(cos_abs_median=float(np.median(np.abs(cos))) if cos.size else None,
                 cos_sign_pos=float(np.mean(cos > 0)) if cos.size else None,
                 frac_facing_cam=float(np.mean(n[..., 2][d > 0] < 0)),
                 n_inst=len(meta["instances"]), rgb_mean=float(rgb.mean()))
    if meta.get("scenario") == "BOX":
        top = [i for i in meta["instances"] if i["top"]]
        rest = [i for i in meta["instances"] if not i["top"]]
        md = lambda ii: float(np.mean([np.median(d[inst == i["id"]]) for i in ii])) if ii else None
        stats.update(n_top=len(top), top_depth_mm=md(top), other_depth_mm=md(rest))
    return meta, rgb, d, n8, inst, stats


def colorize(inst, rgb, meta):
    rng = np.random.default_rng(1)
    lut = rng.integers(40, 255, (inst.max() + 1, 3)).astype(np.uint8)
    lut[0] = 0
    col = lut[inst]
    out = np.where(inst[..., None] > 0, (0.45 * rgb + 0.55 * col).astype(np.uint8), (0.5 * rgb).astype(np.uint8))
    b =cv2.dilate((cv2.Laplacian(inst.astype(np.float32), cv2.CV_32F) != 0).astype(np.uint8), np.ones((2, 2), np.uint8)) > 0
    out[b] = (255, 255, 255)
    for i in meta["instances"]:
        if i.get("top"):
            x, y, w, h = i["bbox_xywh"]
            cv2.putText(out, "T", (x + w // 2, y + h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", default=None)
    ap.add_argument("--scale", type=float, default=0.5)
    a = ap.parse_args()
    scenes = [a.root] if os.path.exists(os.path.join(a.root, "meta.json")) else sorted(
        os.path.join(a.root, s) for s in os.listdir(a.root) if os.path.exists(os.path.join(a.root, s, "meta.json")))
    out = a.out or os.path.join(a.root, "_preview")
    os.makedirs(out, exist_ok=True)
    for s in scenes:
        meta, rgb, d, n8, inst, st = check(s)
        dv = d.astype(np.float32)
        valid = dv > 0
        lo, hi = np.percentile(dv[valid], [1, 99]) if valid.any() else (0, 1)
        dimg = cv2.applyColorMap((255 * (1 - np.clip((dv - lo) / max(hi - lo, 1), 0, 1))).astype(np.uint8),
                                 cv2.COLORMAP_TURBO)
        dimg[~valid] = 0
        tiles = [rgb, colorize(inst, rgb, meta), dimg, cv2.cvtColor(n8, cv2.COLOR_RGB2BGR)]
        tiles = [cv2.resize(t, None, fx=a.scale, fy=a.scale, interpolation=cv2.INTER_AREA) for t in tiles]
        grid = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
        name = os.path.basename(s.rstrip("\\/"))
        cv2.imwrite(os.path.join(out, f"{name}.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(name, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in st.items()}))


if __name__ == "__main__":
    main()
