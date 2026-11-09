import blenderproc as bproc  # noqa: I001  BlenderProc 규약: 첫 import
"""[ONE]/[FEW] 빈 피킹 합성 장면 생성 (빈 + Zivid 2+ MR130, 현재 데모 리그 재현).

실행 (blenderproc venv):
  C:\\Zivid\\3rdparty\\venv_blenderproc\\Scripts\\blenderproc.exe run gen_bin.py ^
      --custom-blender-path C:\\Zivid\\3rdparty\\blender\\blender-4.2.1 ^
      --out <dir> --mode one|few --n 10 --seed 0 [--classes can,flange]

  --mode one : 장면마다 1종 (--classes 미지정 시 7종을 순환)
  --mode few : 장면마다 2~5종 혼합
category_id 는 products.py 참조.
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
import products as P  # noqa: E402
import scene_common as S  # noqa: E402
import textures as T  # noqa: E402


def build_bin(rng, cfg):
    B = cfg["bin"]
    ox, oy = B["outer_m"]
    h = float(rng.choice(B["height_choices_m"], p=B["height_probs"]))
    t = B["wall_m"]
    col = B["colors"][rng.integers(len(B["colors"]))]
    plastic = G.make_material("bin", color_img=T.plastic(rng, col), roughness=rng.uniform(0.35, 0.7))
    table = G.make_material("table", color_img=T.concrete(rng), roughness=rng.uniform(0.5, 0.9))
    floor = G.box_mesh("table", 4, 4, 0.02, [table] * 3, bevel=0)
    floor.set_location([0, 0, -0.02])
    parts = [floor]
    b = G.box_mesh("bin_floor", ox, oy, t, [plastic] * 3, bevel=0.002)
    parts.append(b)
    for k, (L, W, x, y) in enumerate([(ox, t, 0, -(oy - t) / 2), (ox, t, 0, (oy - t) / 2),
                                      (t, oy - 2 * t, -(ox - t) / 2, 0), (t, oy - 2 * t, (ox - t) / 2, 0)]):
        w = G.box_mesh(f"bin_wall{k}", L, W, h, [plastic] * 3, bevel=0.002)
        w.set_location([x, y, 0])
        parts.append(w)
    for p in parts:
        p.enable_rigidbody(active=False, collision_shape="BOX", friction=0.6)
    return parts, t, h  # 빈 내부 바닥 높이, 빈 높이


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["one", "few"], required=True)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--classes", default=None, help="쉼표 구분 클래스 목록 (미지정 = 전체)")
    ap.add_argument("--config", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "bin.yaml"))
    args = ap.parse_args()
    args.out = os.path.abspath(args.out)
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    pool = args.classes.split(",") if args.classes else list(P.CLASSES.values())

    bproc.init()
    S.setup_renderer(cfg)
    tex_dir = os.path.join(args.out, f"_tex_{args.mode}_{args.seed}")
    C = cfg["camera"]
    W, H = C["width"], C["height"]
    K = np.array([[C["fx"], 0, C["cx"]], [0, C["fy"], C["cy"]], [0, 0, 1]])
    B = cfg["bin"]
    ix, iy = B["outer_m"][0] - 2 * B["wall_m"], B["outer_m"][1] - 2 * B["wall_m"]

    for i in range(args.n):
        # 이어서 만들기: meta.json 은 장면 파일 중 마지막에 저장되므로, 있으면 완성된 장면이다.
        # 장면 i 의 난수는 (seed, i) 로만 정해지므로 건너뛰어도 다른 장면은 바뀌지 않는다.
        if os.path.exists(os.path.join(args.out, f"{args.mode}_{args.seed:03d}_{i:05d}", "meta.json")):
            continue
        t0 = time.time()
        rng = np.random.default_rng([args.seed, i, 0 if args.mode == "one" else 1])
        np.random.seed(int(rng.integers(1 << 31)))  # bproc 샘플러가 전역 np.random 을 쓴다
        S.reset_scene()
        G.set_texture_dir(tex_dir)
        if args.mode == "one":
            classes = [pool[i % len(pool)]]
        else:
            k = int(rng.integers(cfg["few"]["n_classes"][0], min(cfg["few"]["n_classes"][1], len(pool)) + 1))
            classes = sorted(rng.choice(pool, k, replace=False).tolist())
        bin_parts, floor_z, bin_h = build_bin(rng, cfg)

        skus = {c: P.SKU(c, rng, cfg) for c in classes}
        fill = rng.uniform(*cfg["fill"]["area_ratio"])
        share = rng.dirichlet(np.ones(len(classes)) * 2)
        objs, targets = [], {}
        for c, sh in zip(classes, share):
            n = int(np.clip(round(fill * sh * ix * iy / skus[c].footprint), 1, cfg["fill"]["n_max"]))
            for j in range(n):
                name = f"{c}_{j:03d}"
                o = skus[c].make(name)
                o.enable_rigidbody(active=True, collision_shape=skus[c].collision, friction=0.6,
                                   linear_damping=0.1, angular_damping=0.2)
                objs.append(o)
                targets[name] = skus[c].info()
        if len(objs) > cfg["fill"]["n_max"]:  # 총 개수 상한
            for o in objs[cfg["fill"]["n_max"]:]:
                targets.pop(o.get_name(), None)
                o.delete()
            objs = objs[:cfg["fill"]["n_max"]]

        ph = cfg["physics"]
        z_lo = bin_h + ph["drop_above_rim_m"][0]
        z_hi = bin_h + ph["drop_above_rim_m"][1] + 0.012 * len(objs)
        margin = 0.05

        def sample_pose(o):
            o.set_location(np.random.uniform([-ix / 2 + margin, -iy / 2 + margin, z_lo],
                                             [ix / 2 - margin, iy / 2 - margin, z_hi]))
            o.set_rotation_euler(bproc.sampler.uniformSO3())

        bproc.object.sample_poses(objs, sample_pose_func=sample_pose, objects_to_check_collisions=objs + bin_parts)
        bproc.object.simulate_physics_and_fix_final_poses(min_simulation_time=ph["min_time"],
                                                          max_simulation_time=ph["max_time"],
                                                          check_object_interval=1)
        # 빈 밖으로 떨어진 물체 제거
        kept = []
        for o in objs:
            x, y, z = o.get_location()
            if abs(x) > ix / 2 + 0.01 or abs(y) > iy / 2 + 0.01 or z < floor_z - 0.01:
                targets.pop(o.get_name(), None)
                o.delete()
            else:
                kept.append(o)
        objs = kept

        # 카메라: 빈 바닥 중심을 거리 D, 수직에서 tilt 만큼 기울여 본다. 빈 긴 변 = 이미지 x
        D = rng.uniform(*C["distance_m"])
        tilt = np.deg2rad(rng.uniform(*C["tilt_deg"]))
        az = rng.uniform(0, 2 * np.pi)
        aim = np.array([*rng.uniform(-1, 1, 2) * C["aim_offset_m"], floor_z])
        cam_pos = aim + D * np.array([np.sin(tilt) * np.cos(az), np.sin(tilt) * np.sin(az), np.cos(tilt)])
        R = S.look_rotation_cv(aim - cam_pos, [1, 0, 0])
        rot = np.deg2rad(rng.uniform(-C["inplane_rot_deg"], C["inplane_rot_deg"]))
        if rng.uniform() < C["flip_180_prob"]:
            rot += np.pi
        c, s = np.cos(rot), np.sin(rot)
        R = R @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        T_cam = S.set_camera_cv(K, W, H, cam_pos, R)
        S.add_lights(rng, cfg, cam_pos, [0, 0, 0])

        out_dir = os.path.join(args.out, f"{args.mode}_{args.seed:03d}_{i:05d}")
        meta = S.render_and_save(out_dir, targets,
                                 dict(scenario=args.mode.upper(), classes=classes, n_dropped=len(objs),
                                      camera_model=C["model"], tilt_deg=float(np.rad2deg(tilt)),
                                      bin_height_m=bin_h,
                                      distance_m=float(D)),
                                 K, W, H, cfg["min_visible_px"], T_cam)
        shutil.rmtree(tex_dir, ignore_errors=True)
        print(f"[gen_bin] {out_dir}  classes={classes}  dropped={len(objs)}  visible={len(meta['instances'])}  "
              f"{time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
