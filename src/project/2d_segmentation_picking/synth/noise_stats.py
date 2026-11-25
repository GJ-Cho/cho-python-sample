"""센서 노이즈 통계: 실측 zdf 와 합성(노이즈 적용) 장면을 같은 지표로 비교한다 (venv_segpick).

  python noise_stats.py real <a.zdf> [<b.zdf> ...]                 # zivid 필요
  python noise_stats.py synth <scene_root> [--seed 0] [--n 5]      # apply_noise() 를 즉석 적용

지표
  missing        : depth 결측 비율 (전체 픽셀)
  speckle        : 면적 ≤ 4 px 인 고립 결측 덩어리의 픽셀 비율
  shadow_*       : 행/열 방향 결측 구간(양끝이 유효, 양끝 깊이차 > 20 mm) 중
                   가까운 쪽(가리는 물체)이 왼/오/위/아래 인 구간의 픽셀 비율과
                   구간 폭으로 역산한 기선 b = w·z_near·z_far / (f·(z_far - z_near)) 의 중앙값
  miss_by_lum    : RGB 밝기 구간별 결측 비율
  sigma_mm_by_z  : 9×9 평면 피팅 잔차 std 중앙값 (거리 구간별)
  normal_missing : 유효 depth 중 normals 결측 비율 (real 은 zdf normals, synth 는 재계산 normals)
  normal_rough_deg : 평탄 영역에서 인접(4px) 법선 각도차 중앙값
읽기 방식 참고: zivid-python-samples read_iterate_zdf.py (zivid.Frame, point_cloud().copy_data)
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def _runs(miss_line):
    """1D bool 배열 → (start, end) 결측 구간 목록 (양끝이 배열 안쪽인 것만)."""
    m = np.concatenate([[False], miss_line, [False]]).astype(np.int8)
    d = np.diff(m)
    s, e = np.where(d == 1)[0], np.where(d == -1)[0]
    keep = (s > 0) & (e < len(miss_line))
    return s[keep], e[keep]


def shadow_stats(z, f, axis, jump=0.02, max_w=80, step=2):
    """axis=1: 행 방향 (왼/오), axis=0: 열 방향 (위/아래). z 단위 m, 결측 0."""
    zz = z if axis == 1 else z.T
    near_first, near_last, b_first, b_last = 0, 0, [], []
    for line in zz[::step]:
        s, e = _runs(line == 0)
        for a, b in zip(s, e):
            w = b - a
            if w > max_w:
                continue
            z0, z1 = line[a - 1], line[b]
            if abs(z1 - z0) < jump:
                continue
            zn, zf = min(z0, z1), max(z0, z1)
            bl = w * zn * zf / (f * (zf - zn))
            if z0 < z1:   # 앞(왼/위)이 가까움 → 그림자는 가리는 물체의 오른쪽/아래쪽
                near_first += w
                b_first.append(bl)
            else:
                near_last += w
                b_last.append(bl)
    tot = zz[::step].size
    med = lambda x: float(np.median(x)) if len(x) else None
    return dict(after=near_first / tot, before=near_last / tot,
                b_after=med(b_first), b_before=med(b_last), n_after=len(b_first), n_before=len(b_last))


def plane_sigma(z, K, win=9, step=6):
    """평탄한 9×9 창에서 평면 피팅 잔차 std (mm) → (z, sigma) 배열."""
    h, w = z.shape
    r = win // 2
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    A = np.stack([xx.ravel(), yy.ravel(), np.ones(win * win)], 1).astype(np.float64)
    pinv = np.linalg.pinv(A)
    out = []
    for v in range(r, h - r, step):
        for u in range(r, w - r, step):
            p = z[v - r:v + r + 1, u - r:u + r + 1]
            if (p <= 0).any():
                continue
            c = pinv @ p.ravel()
            res = p.ravel() - A @ c
            if np.abs(res).max() > 0.01:  # 경계·곡면 제외
                continue
            out.append((p[r, r], res.std() * 1000))
    return np.array(out) if out else np.zeros((0, 2))


def normal_rough(n, valid, d=4):
    a, b = n[:, :-d], n[:, d:]
    ok = valid[:, :-d] & valid[:, d:]
    cos = np.clip(np.sum(a * b, -1)[ok], -1, 1)
    ang = np.degrees(np.arccos(cos))
    ang = ang[ang < 20]  # 실제 모서리는 제외
    return float(np.median(ang)) if ang.size else None


def stats(rgb, z, n, K):
    """rgb uint8 RGB, z (m, 결측 0), n (H,W,3, 결측 0 또는 nan)."""
    miss = z <= 0
    n = np.nan_to_num(n)
    nvalid = np.linalg.norm(n, axis=-1) > 0.5
    ncc, lab, st, _ = cv2.connectedComponentsWithStats(miss.astype(np.uint8), connectivity=8)
    small = np.isin(lab, np.where(st[:, cv2.CC_STAT_AREA] <= 4)[0][1:])
    lum = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    bins = [0, 15, 30, 60, 120, 256]
    by_lum = {}
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (lum >= lo) & (lum < hi)
        by_lum[f"{lo}-{hi}"] = (round(float(miss[m].mean()), 4), round(float(m.mean()), 3)) if m.any() else None
    sat = rgb.max(-1) >= 250
    ps = plane_sigma(z, K)
    sig = {}
    if len(ps):
        for lo in np.arange(0.5, 4.5, 0.5):
            m = (ps[:, 0] >= lo) & (ps[:, 0] < lo + 0.5)
            if m.sum() > 30:
                sig[f"{lo:.1f}"] = round(float(np.median(ps[m, 1])), 3)
    hs = shadow_stats(z, K[0][0], 1)
    vs = shadow_stats(z, K[1][1], 0)
    r4 = lambda x: None if x is None else round(x, 4)
    return dict(
        missing=r4(float(miss.mean())), speckle=r4(float(small.mean())),
        shadow_right=r4(hs["after"]), shadow_left=r4(hs["before"]),
        b_right=r4(hs["b_after"]), b_left=r4(hs["b_before"]),
        shadow_down=r4(vs["after"]), shadow_up=r4(vs["before"]),
        b_down=r4(vs["b_after"]), b_up=r4(vs["b_before"]),
        miss_sat=r4(float(miss[sat].mean())) if sat.any() else None, sat_frac=r4(float(sat.mean())),
        miss_by_lum=by_lum, sigma_mm_by_z=sig,
        normal_missing=r4(float((~nvalid & ~miss).mean() / max((~miss).mean(), 1e-9))),
        normal_rough_deg=r4(normal_rough(n, nvalid)),
    )


def load_real(path):
    import zivid
    with zivid.Frame(path) as fr:
        pc = fr.point_cloud()
        rgb = pc.copy_data("rgba_srgb")[..., :3].copy()
        xyz = pc.copy_data("xyz")
        n = pc.copy_data("normals")
        model = fr.camera_info.model_name
    z = np.nan_to_num(xyz[..., 2] / 1000, nan=0.0).astype(np.float32)
    h, w = z.shape
    # 실측 점군에서 핀홀 K 피팅 (synth/config 와 같은 방식)
    v, u = np.mgrid[0:h, 0:w]
    ok = z > 0
    X, Y, Z = xyz[..., 0][ok] / 1000, xyz[..., 1][ok] / 1000, z[ok]
    ax = np.stack([X / Z, np.ones_like(Z)], 1)
    fx, cx = np.linalg.lstsq(ax, u[ok], rcond=None)[0]
    ay = np.stack([Y / Z, np.ones_like(Z)], 1)
    fy, cy = np.linalg.lstsq(ay, v[ok], rcond=None)[0]
    K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    return model, rgb, z, n, K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["real", "synth"])
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--config", default=os.path.join(HERE, "config", "noise.yaml"))
    ap.add_argument("--json", default=None, help="결과 저장 경로")
    a = ap.parse_args()
    res = {}
    if a.kind == "real":
        import zivid
        app = zivid.Application()  # noqa: F841 (Frame 로드에 필요)
        for p in a.paths:
            model, rgb, z, n, K = load_real(p)
            res[os.path.basename(p)] = dict(model=model, **stats(rgb, z, n, K))
            print(os.path.basename(p), json.dumps(res[os.path.basename(p)], ensure_ascii=False))
    else:
        from add_sensor_noise import apply_noise, decode_normals
        cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
        root = a.paths[0]
        scenes = sorted(s for s in os.listdir(root) if os.path.exists(os.path.join(root, s, "meta.json")))
        scenes = scenes[::max(1, len(scenes) // a.n)][:a.n]
        for k, s in enumerate(scenes):
            sd = os.path.join(root, s)
            meta = json.load(open(os.path.join(sd, "meta.json"), encoding="utf-8"))
            rgb = cv2.cvtColor(cv2.imread(os.path.join(sd, "rgb.png")), cv2.COLOR_BGR2RGB)
            d = cv2.imread(os.path.join(sd, "depth.png"), cv2.IMREAD_UNCHANGED)
            n0 = decode_normals(cv2.cvtColor(cv2.imread(os.path.join(sd, "normals.png")), cv2.COLOR_BGR2RGB))
            rgb_n, d_n, n, _ = apply_noise(rgb, d, meta["K"], cfg["cameras"][meta["camera_model"]], cfg,
                                           np.random.default_rng([a.seed, k]), normals=n0)
            res[s] = stats(rgb_n, d_n.astype(np.float32) / 1000, n, meta["K"])
            print(s, json.dumps(res[s], ensure_ascii=False))
    if a.json:
        json.dump(res, open(a.json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
