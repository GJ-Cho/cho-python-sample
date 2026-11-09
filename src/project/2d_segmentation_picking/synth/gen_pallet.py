import blenderproc as bproc  # noqa: I001  BlenderProc 규약: 첫 import
"""[BOX] 팔레트 디팔레타이징 합성 장면 생성.

실행 (blenderproc venv):
  C:\\Zivid\\3rdparty\\venv_blenderproc\\Scripts\\blenderproc.exe run gen_pallet.py ^
      --custom-blender-path C:\\Zivid\\3rdparty\\blender\\blender-4.2.1 ^
      -- --out <dir> --n 10 --seed 0 [--types 1|2|3|cycle]

장면 i 의 출력: <out>/<seed>_<i:05d>/{rgb,depth,normals,inst}.png + meta.json
박스 category_id: 1~5 = B1~B5 (택배 상자 1~5호, config/pallet.yaml)
"""
import argparse
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

import geometry as G  # noqa: E402
import scene_common as S  # noqa: E402
import textures as T  # noqa: E402



def build_floor_and_pallet(rng, cfg):
    floor = G.box_mesh("floor", 10, 10, 0.01, [G.make_material("concrete", color_img=T.concrete(rng),
                                                                roughness=rng.uniform(0.6, 0.9))] * 3, bevel=0)
    floor.set_location([0, 0, -0.01])
    wood = G.make_material("wood", color_img=T.wood(rng), roughness=0.8)
    px, py = cfg["pallet"]["size_m"]
    parts = []
    z = 0.0
    for k in range(3):  # 하부 보드
        b = G.box_mesh(f"pal_bot{k}", px, 0.1, 0.022, [wood] * 3, bevel=0.002)
        b.set_location([0, (k - 1) * (py - 0.1) / 2, z])
        parts.append(b)
    z += 0.022
    for i in range(3):  # 블록
        for k in range(3):
            b = G.box_mesh(f"pal_blk{i}{k}", 0.14, 0.1, 0.078, [wood] * 3, bevel=0.003)
            b.set_location([(i - 1) * (px - 0.14) / 2, (k - 1) * (py - 0.1) / 2, z])
            parts.append(b)
    z += 0.078
    for k in range(3):  # 스트링어 보드
        b = G.box_mesh(f"pal_str{k}", 0.1, py, 0.022, [wood] * 3, bevel=0.002)
        b.set_location([(k - 1) * (px - 0.1) / 2, 0, z])
        parts.append(b)
    z += 0.022
    n_top = 7
    bw = 0.13
    for k in range(n_top):  # 상판
        b = G.box_mesh(f"pal_top{k}", px, bw, 0.022, [wood] * 3, bevel=0.002)
        b.set_location([0, -py / 2 + bw / 2 + k * (py - bw) / (n_top - 1), z])
        parts.append(b)
    return cfg["pallet"]["height_m"]


def plan_zones(rng, types, dims, size, gap):
    """팔레트 바닥을 x 방향 띠(zone)로 나누고 종류별로 하나씩 배정한다."""
    px, py = size
    remaining = px
    x0 = -px / 2
    zones = []
    for idx, t in enumerate(types):
        L, W, H = dims[t]
        orients = [(L, W), (W, L)]
        rng.shuffle(orients)
        rest = types[idx + 1:]
        min_rest = sum(min(dims[r][0], dims[r][1]) + gap for r in rest)
        chosen = None
        for dx, dy in orients:
            if idx == len(types) - 1:
                cols = int((remaining + gap) // (dx + gap))
            else:
                max_cols = int((remaining - min_rest + gap) // (dx + gap))
                cols = int(rng.integers(1, max_cols + 1)) if max_cols >= 1 else 0
            rows = int((py + gap) // (dy + gap))
            if cols >= 1 and rows >= 1:
                chosen = (dx, dy, cols, rows)
                break
        if chosen is None:
            continue
        dx, dy, cols, rows = chosen
        width = remaining if idx == len(types) - 1 else cols * (dx + gap)
        zones.append(dict(type=t, x0=x0, width=width, dx=dx, dy=dy, cols=cols, rows=rows, H=H,
                          yaw90=(dx != L)))
        x0 += width
        remaining -= width
    return zones


def build_load(rng, cfg, types, mats_by_type, z0):
    L = cfg["load"]
    dims = {t: tuple(cfg["boxes"][t]) for t in types}
    gap = rng.uniform(*L["gap_m"])
    zones = plan_zones(rng, list(types), dims, cfg["pallet"]["size_m"], gap)
    py = cfg["pallet"]["size_m"][1]
    targets, top_z = {}, z0
    count = 0
    # 혼합 팔레트도 층 단위로 쌓으므로 zone 높이는 비슷하다: 공통 목표 높이 ± 1층
    max_load = L["max_top_from_floor_m"] - z0
    min_h = min(zn["H"] for zn in zones) * L["min_layers"]
    target_h = rng.uniform(min_h, max_load)
    for zi, zn in enumerate(zones):
        n_max = int((max_load + 1e-9) // zn["H"])
        n_layers = int(np.clip(round(target_h / zn["H"]) + rng.integers(-1, 2), L["min_layers"], n_max))
        cells = [(i, j) for i in range(zn["cols"]) for j in range(zn["rows"])]
        n_cells = len(cells)
        n_remove = int(round(rng.uniform(*L["top_remove_frac"]) * n_cells))
        n_remove = min(n_remove, n_cells - 1)
        if rng.uniform() < L["sweep_remove_prob"]:
            key = rng.choice(["x", "y", "xy"])
            order = sorted(cells, key=lambda c: {"x": c[0], "y": c[1], "xy": c[0] + c[1]}[key])
            removed = set(order[:n_remove])
        else:
            removed = set(map(tuple, rng.permutation(cells)[:n_remove].tolist()))
        grid_w = zn["cols"] * (zn["dx"] + gap) - gap
        grid_h = zn["rows"] * (zn["dy"] + gap) - gap
        gx0 = zn["x0"] + (zn["width"] - grid_w) / 2 + zn["dx"] / 2
        gy0 = -grid_h / 2 + zn["dy"] / 2
        for layer in range(n_layers):
            is_top_layer = layer == n_layers - 1
            for (i, j) in cells:
                if is_top_layer and (i, j) in removed:
                    continue
                exposed = is_top_layer or (layer == n_layers - 2 and (i, j) in removed)
                jxy = L["top_jitter_xy_m"] if is_top_layer else L["jitter_xy_m"]
                jyaw = L["top_jitter_yaw_deg"] if is_top_layer else L["jitter_yaw_deg"]
                t = zn["type"]
                Lb, Wb, Hb = dims[t]
                # 이웃 박스와 겹치지 않게: 이동 + 회전으로 모서리가 움직이는 양 < 틈의 절반
                budget = gap * 0.45
                jxy = min(jxy, budget * 0.5)
                half_diag = 0.5 * np.hypot(Lb, Wb)
                jyaw = min(jyaw, np.rad2deg((budget * 0.5) / half_diag))
                name = f"box_{t}_{count:04d}"
                o = G.box_mesh(name, Lb, Wb, Hb, mats_by_type[t], bevel=cfg["box_bevel_m"])
                yaw = (np.pi / 2 if zn["yaw90"] else 0) + np.deg2rad(rng.uniform(-jyaw, jyaw))
                o.set_rotation_euler([0, 0, yaw])
                o.set_location([gx0 + i * (zn["dx"] + gap) + rng.uniform(-jxy, jxy),
                                gy0 + j * (zn["dy"] + gap) + rng.uniform(-jxy, jxy),
                                z0 + layer * Hb])
                targets[name] = dict(cls=t, category_id=int(t[1:]), layer=layer, top=bool(exposed), zone=zi)
                count += 1
        top_z = max(top_z, z0 + n_layers * zn["H"])
    assert py > 0
    return targets, top_z, zones


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--types", default="cycle", help="1~n_types_max | cycle — 장면당 박스 종류 수")
    ap.add_argument("--config", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "pallet.yaml"))
    args = ap.parse_args()
    args.out = os.path.abspath(args.out)
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    bproc.init()
    S.setup_renderer(cfg)
    tex_dir = os.path.join(args.out, f"_tex_{args.seed}")
    C = cfg["camera"]
    W, H = C["width"], C["height"]
    K = np.array([[C["fx"], 0, C["cx"]], [0, C["fy"], C["cy"]], [0, 0, 1]])
    box_types = list(cfg["boxes"])
    n_types_max = cfg["n_types_max"]

    for i in range(args.n):
        # 이어서 만들기: meta.json 은 장면 파일 중 마지막에 저장되므로, 있으면 완성된 장면이다.
        # 장면 i 의 난수는 (seed, i) 로만 정해지므로 건너뛰어도 다른 장면은 바뀌지 않는다.
        if os.path.exists(os.path.join(args.out, f"{args.seed:03d}_{i:05d}", "meta.json")):
            continue
        t0 = time.time()
        rng = np.random.default_rng([args.seed, i])
        S.reset_scene()
        G.set_texture_dir(tex_dir)
        n_types = (i % n_types_max) + 1 if args.types == "cycle" else int(args.types)
        types = sorted(rng.choice(box_types, n_types, replace=False).tolist())
        mats_by_type, designs = {}, {}
        for t in types:
            faces, design = T.make_box_design(rng, [d * 1000 for d in cfg["boxes"][t]])
            mats_by_type[t] = [G.make_material(f"{t}_{k}", color_img=faces[k][0], rough_img=faces[k][1])
                               for k in ("top", "long", "short")]
            designs[t] = design
        z0 = build_floor_and_pallet(rng, cfg)
        targets, top_z, zones = build_load(rng, cfg, types, mats_by_type, z0)

        assert top_z <= cfg["load"]["max_top_from_floor_m"] + 1e-6, top_z
        tilt = np.deg2rad(rng.uniform(0, C["tilt_deg_max"]))
        az = rng.uniform(0, 2 * np.pi)
        fwd = np.array([np.sin(tilt) * np.cos(az), np.sin(tilt) * np.sin(az), -np.cos(tilt)])
        # 이미지 x = 월드 x(팔레트 변), 이미지 y = 월드 -y. 이후 inplane_rot_deg 만큼만 회전
        R = S.look_rotation_cv(fwd, [1, 0, 0])
        rot = np.deg2rad(rng.uniform(-C["inplane_rot_deg"], C["inplane_rot_deg"])) if C["inplane_rot_deg"] else 0.0
        c, s_ = np.cos(rot), np.sin(rot)
        R = R @ np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]])
        # (a) 광학 중심 보정: 거리 Zc 에서 팔레트 중심(월드 원점)이 이미지 중앙 (W/2, H/2) 에 오도록
        #     카메라를 옮긴다. 원점의 카메라 좌표가 (Xc, Yc, Zc) 이려면 cam = p - R·(Xc, Yc, Zc).
        Zc = C["center_pallet_at_distance_m"]
        Xc = (W / 2 - C["cx"]) * Zc / C["fx"]
        Yc = (H / 2 - C["cy"]) * Zc / C["fy"]
        cam_pos = -R @ np.array([Xc, Yc, 0.0])
        cam_pos[2] = C["height_above_floor_m"]
        T_cam = S.set_camera_cv(K, W, H, cam_pos, R)
        S.add_lights(rng, cfg, cam_pos, [0, 0, 0])

        out_dir = os.path.join(args.out, f"{args.seed:03d}_{i:05d}")
        meta = S.render_and_save(out_dir, targets,
                                 dict(scenario="BOX", n_types=n_types, types=types, designs=designs,
                                      top_z_m=float(top_z), camera_model=C["model"],
                                      working_distance_top_m=float(C["height_above_floor_m"] - top_z),
                                      zones=[{k: (float(v) if isinstance(v, (float, np.floating)) else v)
                                              for k, v in z.items()} for z in zones]),
                                 K, W, H, cfg["min_visible_px"], T_cam)
        shutil.rmtree(tex_dir, ignore_errors=True)
        n_top = sum(1 for d in meta["instances"] if d["top"])
        print(f"[gen_pallet] {out_dir}  types={types}  visible={len(meta['instances'])}  top={n_top}  "
              f"{time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
