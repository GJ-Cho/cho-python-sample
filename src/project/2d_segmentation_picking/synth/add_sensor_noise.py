"""합성 장면에 Zivid 유사 센서 노이즈를 입힌다 (일반 Python, numpy + opencv).

  python add_sensor_noise.py <root 또는 scene_dir> --out <dir> [--seed 0] [--config config/noise.yaml]

입력: gen_*.py 출력 장면 폴더 (rgb/depth/normals/inst.png + meta.json)
출력: <out>/<scene>/ 에 같은 파일 이름으로 노이즈 버전 저장. inst.png, meta.json 은 복사.
      depth 결측은 0 (Zivid NaN 에 해당), normals 결측은 (0,0,0)→ png 값 127.5 근처가 아니라 0 으로 둔다.

학습 코드에서 on-the-fly 로 쓰려면 apply_noise(rgb, depth_mm, K, cam_cfg, cfg, rng) 를 import 해서 쓴다.

모델 (전부 근사):
  depth  : 거리 제곱 비례 가우시안(+저주파 상관 성분), 깊이 불연속 경계 결측,
           프로젝터 그림자 결측(평행 스테레오 가정), 반사 하이라이트·스치는 각도·어두운 영역 결측, 단독 결측
  rgb    : Poisson-Gaussian 노이즈, 약한 블러, 노출·화이트밸런스 흔들림
  normals: 노이즈 depth 에서 다시 계산 (Zivid normals 는 포인트 클라우드에서 계산됨)
파라미터의 근거([실측]/[가정])는 config/noise.yaml 주석 참조. 3단계에서 실측 zdf 통계로 교정한다.
"""
import argparse
import json
import os
import shutil

import cv2
import numpy as np
import yaml


def _lowfreq(h, w, cells, rng):
    small = rng.normal(0, 1, (max(2, cells * h // max(h, w)), max(2, cells * w // max(h, w)))).astype(np.float32)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def backproject(z_m, K):
    h, w = z_m.shape
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    X = (u - K[0][2]) / K[0][0] * z_m
    Y = (v - K[1][2]) / K[1][1] * z_m
    return np.stack([X, Y, z_m], -1)


def normals_from_depth(z_m, K, smooth_px):
    valid = z_m > 0
    z = z_m.copy()
    if smooth_px > 0:
        k = 2 * smooth_px + 1
        num = cv2.blur(np.where(valid, z, 0).astype(np.float32), (k, k))
        den = cv2.blur(valid.astype(np.float32), (k, k))
        z = np.where(valid, num / np.maximum(den, 1e-6), 0)
    P = backproject(z, K)
    du = np.zeros_like(P)
    dv = np.zeros_like(P)
    du[:, 1:-1] = P[:, 2:] - P[:, :-2]
    dv[1:-1] = P[2:] - P[:-2]
    n = np.cross(dv, du)  # 카메라를 향하는 면이 z<0 이 되는 순서 (OpenCV 좌표)
    n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-9)
    ok = valid.copy()
    ok[:, [0, -1]] = False
    ok[[0, -1]] = False
    ok &= cv2.erode(valid.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    n[~ok] = 0
    return n


def projector_shadow(z_m, K, baseline_x_m, tol_px=0.5):
    """프로젝터에서 보이지 않는 픽셀 = True.
    프로젝터가 카메라와 같은 자세로 x 축 방향 baseline_x_m 만큼 떨어져 있다고 본다(평행 스테레오).
    그러면 에피폴라 선이 이미지 행이 되고, 프로젝터 열 u' = u - fx·b/z 의 순서가 뒤집히는 픽셀이 가려진 것이다.
      b > 0 (프로젝터가 +x): 오른쪽 픽셀 중 u' 가 더 작은 것이 있으면 가려짐 → 그림자는 물체의 왼쪽
      b < 0: 반대
    (이전 z-buffer 방식은 3×3 이웃 최소값 비교 때문에 빈 벽 같은 가파른 면을 그림자로 오판했다)"""
    h, w = z_m.shape
    valid = z_m > 0
    u = np.arange(w, dtype=np.float32)[None, :]
    with np.errstate(divide="ignore"):
        up = u - K[0][0] * baseline_x_m / np.where(valid, z_m, np.inf)
    if baseline_x_m >= 0:
        a = np.where(valid, up, np.inf)
        smin = np.minimum.accumulate(a[:, ::-1], axis=1)[:, ::-1]
        other = np.full_like(a, np.inf)
        other[:, :-1] = smin[:, 1:]
        return valid & (up > other + tol_px)
    a = np.where(valid, up, -np.inf)
    smax = np.maximum.accumulate(a, axis=1)
    other = np.full_like(a, -np.inf)
    other[:, 1:] = smax[:, :-1]
    return valid & (up < other - tol_px)


def _clustered_uniform(h, w, sigma_px, rng):
    """공간적으로 뭉친 균등분포 난수 (0~1). 결측이 점이 아니라 덩어리로 생기게 한다."""
    g = rng.normal(0, 1, (h, w)).astype(np.float32)
    if sigma_px > 0:
        g = cv2.GaussianBlur(g, (0, 0), sigma_px)
        g /= max(float(g.std()), 1e-6)
    return 1 / (1 + np.exp(-1.702 * g))  # 표준정규 CDF 근사


def _lowfreq_unit(h, w, cells, rng):
    f = _lowfreq(h, w, cells, rng)
    return f / max(float(f.std()), 1e-6)


def decode_normals(n8):
    """normals.png (RGB uint8) → float32 (H,W,3), 결측 0."""
    n = n8.astype(np.float32) / 127.5 - 1
    n[np.all(n8 == 0, -1)] = 0
    return n


def apply_rgb_noise(rgb, R, rng):
    """RGB 노이즈만 (노출·화이트밸런스 흔들림, Poisson-Gaussian, 약한 블러). R = cfg["rgb"].
    RGB 만 쓰는 학습에서 depth 처리 비용 없이 부르려고 apply_noise 에서 분리했다 (난수 소비 순서 동일)."""
    img = rgb.astype(np.float32)
    img *= rng.uniform(*R["exposure_gain"])
    img *= (1 + rng.uniform(-R["wb_jitter"], R["wb_jitter"], 3))[None, None, :]
    var = R["shot_gain"] * np.clip(img, 0, 255) + R["read_sigma"] ** 2
    img = img + rng.normal(0, 1, img.shape).astype(np.float32) * np.sqrt(var)
    bs = rng.uniform(*R["blur_sigma_px"])
    if bs > 0.05:
        img = cv2.GaussianBlur(img, (0, 0), bs)
    return np.clip(np.round(img), 0, 255).astype(np.uint8)


def apply_noise(rgb, depth_mm, K, cam_cfg, cfg, rng, normals=None):
    """rgb uint8 (H,W,3 RGB), depth_mm uint16, normals (H,W,3 float, 렌더 법선, 선택)
    → (rgb, depth_mm, normals_cv float32, drop_mask)."""
    D, R, N = cfg["depth"], cfg["rgb"], cfg["normals"]
    h, w = depth_mm.shape
    z = depth_mm.astype(np.float32) / 1000
    valid = z > 0

    # ---- depth 잡음: 고주파(픽셀 독립) + 저주파(공간 상관), σ ∝ (z / z_ref)^pow
    scale = (np.maximum(z, 1e-3) / cam_cfg["depth_ref_m"]) ** D["sigma_scale_pow"]
    hf = cam_cfg["depth_sigma_hf_mm"] / 1000 * scale * rng.normal(0, 1, (h, w)).astype(np.float32)
    lf = cam_cfg["depth_sigma_lf_mm"] / 1000 * scale * _lowfreq_unit(h, w, D["lowfreq_cells"], rng)
    zn = z + hf + lf

    # ---- 법선 (렌더 법선이 있으면 그것을 기준으로 쓴다: depth 가 mm 양자화라 재계산하면 계단 잡음이 생김)
    n_clean = normals if normals is not None else normals_from_depth(z, K, 0)

    # ---- 결측
    clu = lambda: _clustered_uniform(h, w, D["cluster_sigma_px"], rng)
    drop = np.zeros((h, w), bool)
    if D["edge_drop_prob"] > 0:
        zmax = cv2.dilate(z, np.ones((3, 3), np.uint8))
        zmin = cv2.erode(np.where(valid, z, 1e6).astype(np.float32), np.ones((3, 3), np.uint8))
        edge = valid & ((zmax - zmin) > D["edge_jump_mm"] / 1000)
        if D["edge_dilate_px"] > 0:
            edge = cv2.dilate(edge.astype(np.uint8), np.ones((2 * D["edge_dilate_px"] + 1,) * 2, np.uint8)) > 0
        drop |= edge & (rng.uniform(size=(h, w)) < D["edge_drop_prob"])
    if D["projector_shadow"]:
        drop |= projector_shadow(z, K, cam_cfg["projector_baseline_x_m"], D["shadow_tol_px"])
    sat = rgb.max(-1) >= D["saturation_level"]
    drop |= sat & (rng.uniform(size=(h, w)) < D["saturation_drop_prob"])
    # 어두운 영역: 장면 중앙 밝기 대비 상대 밝기로 판정 (노출이 장면마다 달라서)
    lum = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    rel = lum / max(float(np.median(lum[valid])) if valid.any() else 1.0, 1.0)
    p_dark = np.zeros((h, w), np.float32)
    lo = 0.0
    for hi, p in zip(D["dark_rel_edges"], D["dark_drop_probs"]):
        p_dark[(rel >= lo) & (rel < hi)] = p
        lo = hi
    drop |= clu() < p_dark
    P = backproject(z, K)
    view = P / np.maximum(np.linalg.norm(P, axis=-1, keepdims=True), 1e-9)
    cosang = np.abs(np.sum(n_clean * view, -1))
    grazing = (cosang < np.cos(np.deg2rad(D["grazing_deg"]))) & (np.linalg.norm(n_clean, axis=-1) > 0.5)
    drop |= grazing & (clu() < D["grazing_drop_prob"])
    drop |= rng.uniform(size=(h, w)) < D["speckle_drop_prob"]
    drop &= valid
    zn[drop | ~valid] = 0
    depth_out = np.clip(np.round(zn * 1000), 0, 65535).astype(np.uint16)

    # ---- rgb
    rgb_out = apply_rgb_noise(rgb, R, rng)

    # ---- normals: 렌더 법선 + 작은 각도 흔들림, depth 결측 주변은 결측
    if normals is not None and not N["recompute_from_depth"]:
        j = rng.normal(0, 1, (h, w, 3)).astype(np.float32)
        if N["jitter_blur_px"] > 0:
            j = cv2.GaussianBlur(j, (0, 0), N["jitter_blur_px"])
            j /= max(float(j.std()), 1e-6)
        n = n_clean + np.deg2rad(cam_cfg["normals_jitter_deg"]) * j
        n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-9)
        ok = depth_out > 0
        if N["missing_dilate_px"] > 0:
            k = 2 * N["missing_dilate_px"] + 1
            ok = cv2.erode(ok.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
        ok &= np.linalg.norm(n_clean, axis=-1) > 0.5
        n[~ok] = 0
        normals_out = n
    else:
        normals_out = normals_from_depth(depth_out.astype(np.float32) / 1000, K, N["smooth_px"])
    return rgb_out, depth_out, normals_out, drop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "noise.yaml"))
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    scenes = [a.root] if os.path.exists(os.path.join(a.root, "meta.json")) else sorted(
        os.path.join(a.root, s) for s in os.listdir(a.root) if os.path.exists(os.path.join(a.root, s, "meta.json")))
    for k, s in enumerate(scenes):
        meta = json.load(open(os.path.join(s, "meta.json"), encoding="utf-8"))
        cam_cfg = cfg["cameras"][meta["camera_model"]]
        rng = np.random.default_rng([a.seed, k])
        rgb = cv2.cvtColor(cv2.imread(os.path.join(s, "rgb.png")), cv2.COLOR_BGR2RGB)
        d = cv2.imread(os.path.join(s, "depth.png"), cv2.IMREAD_UNCHANGED)
        n0 = decode_normals(cv2.cvtColor(cv2.imread(os.path.join(s, "normals.png")), cv2.COLOR_BGR2RGB))
        rgb_n, d_n, n, drop = apply_noise(rgb, d, meta["K"], cam_cfg, cfg, rng, normals=n0)
        od = os.path.join(a.out, os.path.basename(s.rstrip("\\/")))
        os.makedirs(od, exist_ok=True)
        cv2.imwrite(os.path.join(od, "rgb.png"), cv2.cvtColor(rgb_n, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(od, "depth.png"), d_n)
        n8 = np.where(np.linalg.norm(n, axis=-1, keepdims=True) > 0, (n + 1) * 127.5, 0)
        cv2.imwrite(os.path.join(od, "normals.png"), cv2.cvtColor(np.round(n8).astype(np.uint8), cv2.COLOR_RGB2BGR))
        for f in ("inst.png", "meta.json"):
            shutil.copy(os.path.join(s, f), os.path.join(od, f))
        print(f"{os.path.basename(od)}  depth_missing={float(np.mean(d_n == 0)):.3f}")


if __name__ == "__main__":
    main()
