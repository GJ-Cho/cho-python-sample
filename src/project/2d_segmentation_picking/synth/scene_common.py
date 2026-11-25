"""장면 공통: 카메라, 조명, 렌더 설정, 결과 저장 (Blender 내부 전용).

참고 API (BlenderProc 2.8.0 소스 확인):
  bproc.camera.set_intrinsics_from_K_matrix, bproc.camera.rotation_from_forward_vec,
  bproc.math.build_transformation_mat, bproc.camera.add_camera_pose,
  bproc.renderer.enable_depth_output / enable_normals_output / enable_segmentation_output,
  bproc.renderer.render

출력 규약 (장면 폴더 1개당):
  rgb.png      8bit sRGB
  depth.png    uint16, mm, 카메라 광축 방향 z (planar depth). 0 = 무효
  normals.png  uint8, OpenCV 카메라 좌표계 (x 오른쪽, y 아래, z 전방), n = png/127.5 - 1
               카메라를 향하는 면은 z < 0 (Zivid normals와 같은 방향 규약)
  inst.png     uint16, 대상 물체 인스턴스 id (1..N). 0 = 배경(바닥, 팔레트, 빈)
  meta.json    카메라 K, 인스턴스별 클래스·속성
"""
from __future__ import annotations

import json
import os

import blenderproc as bproc
import bpy
import cv2
import numpy as np


def setup_renderer(cfg):
    r = cfg["render"]
    bproc.renderer.set_render_devices(desired_gpu_device_type=r.get("device", "OPTIX"))
    bproc.renderer.set_max_amount_of_samples(r["samples"])
    bproc.renderer.set_noise_threshold(r["noise_threshold"])
    bproc.renderer.set_denoiser(r.get("denoiser", "OPTIX"))
    bproc.renderer.set_light_bounces(diffuse_bounces=3, glossy_bounces=4, max_bounces=6,
                                     transmission_bounces=2, transparent_max_bounces=4)
    bproc.renderer.enable_depth_output(activate_antialiasing=False)
    _enable_world_normals_output()
    bproc.renderer.enable_segmentation_output(map_by=["instance", "name"], default_values={"name": "bg"})


def _enable_world_normals_output():
    """월드 좌표 Normal pass를 EXR로 그대로 저장한다.
    bproc.renderer.enable_normals_output은 호출 시점의 카메라 회전을 compositor에 키프레임으로 굳혀서,
    한 프로세스에서 장면마다 카메라를 바꾸면 법선이 틀어진다(소스 확인). 그래서 회전은 numpy에서 한다."""
    from blenderproc.python.utility.Utility import Utility
    out_dir = Utility.get_temporary_directory()
    bpy.context.view_layer.use_pass_normal = True
    bpy.context.scene.render.use_compositing = True
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    rl = Utility.get_the_one_node_with_type(tree.nodes, "CompositorNodeRLayers")
    of = tree.nodes.new("CompositorNodeOutputFile")
    of.base_path = out_dir
    of.format.file_format = "OPEN_EXR"
    of.format.color_mode = "RGB"
    of.format.color_depth = "16"
    of.file_slots.values()[0].path = "wnormals_"
    # 벡터 소켓을 그대로 연결하면 1채널 EXR로 저장된다(4.2 확인). depth 출력과 같은 방식으로 Combine Color를 거친다.
    sep = tree.nodes.new("CompositorNodeSeparateColor")
    sep.mode = "RGB"
    comb = tree.nodes.new("CompositorNodeCombineColor")
    comb.mode = "RGB"
    tree.links.new(rl.outputs["Normal"], sep.inputs["Image"])
    for k in range(3):
        tree.links.new(sep.outputs[k], comb.inputs[k])
    tree.links.new(comb.outputs["Image"], of.inputs["Image"])
    Utility.add_output_entry({"key": "wnormals", "path": os.path.join(out_dir, "wnormals_") + "%04d" + ".exr",
                              "version": "2.0.0"})


def set_camera(K, width, height, cam_pos, look_at, inplane_rot):
    bproc.camera.set_intrinsics_from_K_matrix(np.asarray(K, float), width, height, clip_start=0.05, clip_end=20)
    fwd = np.asarray(look_at, float) - np.asarray(cam_pos, float)
    R = bproc.camera.rotation_from_forward_vec(fwd, inplane_rot=inplane_rot)
    T = bproc.math.build_transformation_mat(cam_pos, R)
    bpy.context.scene.frame_end = 0
    bproc.camera.add_camera_pose(T)
    return T


def set_camera_cv(K, width, height, cam_pos, R_cv):
    """OpenCV 카메라 회전(열 = x오른쪽, y아래, z전방의 월드 표현)으로 포즈를 지정한다."""
    bproc.camera.set_intrinsics_from_K_matrix(np.asarray(K, float), width, height, clip_start=0.05, clip_end=20)
    R_b = np.asarray(R_cv) @ np.diag([1, -1, -1])  # Blender 카메라 = (x, -y, -z)
    T = bproc.math.build_transformation_mat(cam_pos, R_b)
    bpy.context.scene.frame_end = 0
    bproc.camera.add_camera_pose(T)
    return T


def look_rotation_cv(fwd, right_hint):
    z = np.asarray(fwd, float) / np.linalg.norm(fwd)
    x = np.asarray(right_hint, float) - np.dot(right_hint, z) * z
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z], 1)


def add_lights(rng, cfg, cam_pos, scene_center):
    """천장 조명(면광원) + 카메라 동축 조명(Zivid 프로젝터 백색광 근사) + 환경광."""
    L = cfg["lights"]
    world = bpy.context.scene.world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    a = rng.uniform(*L["ambient"])
    bg.inputs[0].default_value = (a, a, a * rng.uniform(0.95, 1.05), 1)
    bg.inputs[1].default_value = 1.0
    for i in range(rng.integers(L["n_ceiling"][0], L["n_ceiling"][1] + 1)):
        light = bproc.types.Light(light_type="AREA", name=f"ceil_{i}")
        off = rng.uniform(-1, 1, 2) * L["ceiling_spread"]
        light.set_location([scene_center[0] + off[0], scene_center[1] + off[1], L["ceiling_height"]])
        light.set_energy(rng.uniform(*L["ceiling_energy"]))
        light.blender_obj.data.size = L["ceiling_size"]
        t = rng.uniform(4500, 6500)
        light.set_color(_kelvin_rgb(t))
    if rng.uniform() < L["coaxial_prob"]:
        light = bproc.types.Light(light_type="POINT", name="coax")
        light.set_location(np.asarray(cam_pos) + rng.uniform(-0.05, 0.05, 3))
        light.set_energy(rng.uniform(*L["coaxial_energy"]))
        light.blender_obj.data.shadow_soft_size = 0.05


def _kelvin_rgb(t):
    # 4500~6500K 근사 (흰색 주변의 미세한 색온도 변화만)
    x = (t - 5500) / 1000
    return [1.0, 1.0 - 0.03 * max(0, -x), 1.0 - 0.08 * max(0, -x) + 0.02 * max(0, x)]


def world_to_cv_normals(nw, cam2world):
    """월드 법선 → OpenCV 카메라 좌표 단위 벡터.
    Blender 카메라 좌표 = (x 오른쪽, y 위, z 뒤) → OpenCV = (x, -y, -z)."""
    R = np.asarray(cam2world)[:3, :3]
    nb = nw[..., :3].astype(np.float32) @ R  # = (R^T n)^T
    n = np.stack([nb[..., 0], -nb[..., 1], -nb[..., 2]], -1)
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    return n / np.maximum(norm, 1e-6)


def render_and_save(out_dir, targets, extra_meta, K, width, height, min_visible_px, cam2world):
    """targets: {blender object name: dict(class, category_id, ...)}"""
    os.makedirs(out_dir, exist_ok=True)
    # enable_segmentation_output은 호출 시점에 있던 오브젝트에만 pass_index를 준다(소스 확인).
    # 장면을 매번 새로 만들므로 렌더 직전에 다시 부여한다.
    from blenderproc.python.utility.BlenderUtility import get_all_blender_mesh_objects
    for k, o in enumerate(get_all_blender_mesh_objects()):
        o.pass_index = k + 1
    data = bproc.renderer.render(load_keys={"colors", "depth", "segmap", "wnormals"})
    rgb = data["colors"][0][..., :3]
    depth = data["depth"][0]
    normals = world_to_cv_normals(data["wnormals"][0], cam2world)
    seg = data["instance_segmaps"][0]
    attr = data["instance_attribute_maps"][0]

    inst = np.zeros(seg.shape, np.uint16)
    instances = []
    next_id = 1
    for a in attr:
        info = targets.get(a["name"])
        if info is None:
            continue
        m = seg == a["idx"]
        npx = int(m.sum())
        if npx < min_visible_px:
            continue
        inst[m] = next_id
        ys, xs = np.nonzero(m)
        instances.append(dict(id=next_id, name=a["name"], visible_px=npx,
                              bbox_xywh=[int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1),
                                         int(ys.max() - ys.min() + 1)], **info))
        next_id += 1

    d_mm = np.where(np.isfinite(depth) & (depth < 15), depth * 1000, 0).round().clip(0, 65535).astype(np.uint16)
    cv2.imwrite(os.path.join(out_dir, "rgb.png"), cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(out_dir, "depth.png"), d_mm)
    n8 = ((normals + 1) * 127.5).round().clip(0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(out_dir, "normals.png"), cv2.cvtColor(n8, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(out_dir, "inst.png"), inst)
    meta = dict(width=width, height=height, K=np.asarray(K).tolist(), depth_unit="mm",
                cam2world_blender=np.asarray(cam2world).tolist(),
                normals_frame="opencv_camera", instances=instances, **extra_meta)
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1, ensure_ascii=False)
    return meta


def reset_scene():
    """오브젝트·재질·이미지를 비운다. 렌더 출력 설정(compositor)은 유지된다."""
    bproc.clean_up(clean_up_camera=False)
    for coll in (bpy.data.images, bpy.data.materials, bpy.data.meshes):
        for item in list(coll):
            if item.users == 0:
                coll.remove(item)
