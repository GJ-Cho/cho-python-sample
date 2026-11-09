"""절차적 메시 생성 + 재질 (Blender 내부 전용, `blenderproc run`으로만 import).

단위는 m. 모든 메시는 원점 기준으로 만들고, 배치는 호출 측에서 한다.
참고 API (BlenderProc 2.8.0 소스 확인):
  bproc.object.create_with_empty_mesh, MeshObject.get_mesh, MeshObject.set_cp,
  bproc.material.create, Material.set_principled_shader_value(name, bpy.types.Image)
"""
from __future__ import annotations

import math
import os

import blenderproc as bproc  # noqa: F401  (Blender 내부에서 이미 import되어 있어야 함)
import bpy
import numpy as np

# ---------------------------------------------------------------- 재질

_IMG_CACHE_DIR = None
_IMG_COUNTER = [0]


def set_texture_dir(path):
    global _IMG_CACHE_DIR
    _IMG_CACHE_DIR = path
    os.makedirs(path, exist_ok=True)


def pil_to_bpy(img, name, non_color=False):
    """PIL 이미지를 PNG로 저장 후 bpy 이미지로 로드한다 (장면 정리 시 함께 삭제됨)."""
    _IMG_COUNTER[0] += 1
    path = os.path.join(_IMG_CACHE_DIR, f"{_IMG_COUNTER[0]:06d}_{name}.png")
    img.save(path)
    im = bpy.data.images.load(path, check_existing=False)
    if non_color:
        im.colorspace_settings.name = "Non-Color"
    return im


def make_material(name, color_img=None, rough_img=None, base_color=None, roughness=0.5, metallic=0.0,
                  specular=0.5, coat=0.0):
    m = bproc.material.create(name)
    if color_img is not None:
        m.set_principled_shader_value("Base Color", pil_to_bpy(color_img, name + "_c"))
    elif base_color is not None:
        m.set_principled_shader_value("Base Color", [*(np.array(base_color[:3]) / 255.0), 1.0])
    if rough_img is not None:
        m.set_principled_shader_value("Roughness", pil_to_bpy(rough_img, name + "_r", non_color=True))
    else:
        m.set_principled_shader_value("Roughness", float(roughness))
    m.set_principled_shader_value("Metallic", float(metallic))
    m.set_principled_shader_value("Specular IOR Level", float(specular))
    if coat > 0:
        m.set_principled_shader_value("Coat Weight", float(coat))
    return m


# ---------------------------------------------------------------- 메시 공통

def _mesh_object(name, verts, faces, uvs=None, mat_idx=None, materials=(), smooth=False):
    """verts (N,3), faces = list of index tuples, uvs = per-face list of (u,v) per corner."""
    obj = bproc.object.create_with_empty_mesh(name)
    me = obj.get_mesh()
    me.from_pydata([tuple(map(float, v)) for v in verts], [], [tuple(int(i) for i in f) for f in faces])
    me.update()
    if uvs is not None:
        uv = me.uv_layers.new(name="UVMap")
        k = 0
        for poly, fuv in zip(me.polygons, uvs):
            for li, (u, v) in zip(poly.loop_indices, fuv):
                uv.data[li].uv = (float(u), float(v))
                k += 1
    for m in materials:
        obj.add_material(m)
    if mat_idx is not None:
        for poly, mi in zip(me.polygons, mat_idx):
            poly.material_index = int(mi)
    for poly in me.polygons:
        poly.use_smooth = smooth
    me.validate()
    return obj


def add_bevel(obj, width, segments=2):
    mod = obj.blender_obj.modifiers.new("bevel", "BEVEL")
    mod.width = width
    mod.segments = segments
    mod.limit_method = "ANGLE"
    mod.harden_normals = False


def apply_modifiers(obj):
    """모디파이어 결과를 메시로 굳힌다 (bpy.ops 컨텍스트 없이 depsgraph로 처리)."""
    bo = obj.blender_obj
    dg = bpy.context.evaluated_depsgraph_get()
    new = bpy.data.meshes.new_from_object(bo.evaluated_get(dg))
    old = bo.data
    bo.modifiers.clear()
    bo.data = new
    bpy.data.meshes.remove(old)


def boolean_cut(obj, cutters):
    for c in cutters:
        mod = obj.blender_obj.modifiers.new("cut", "BOOLEAN")
        mod.operation = "DIFFERENCE"
        mod.solver = "EXACT"
        mod.object = c.blender_obj
    apply_modifiers(obj)
    for c in cutters:
        c.delete()


# ---------------------------------------------------------------- 박스 (면별 UV 0..1)

def box_mesh(name, L, W, H, mats, bevel=0.003):
    """원점 = 바닥면 중심. mats = [top, long, short] (bottom은 top 재질).
    x = 길이(L), y = 폭(W), z = 높이(H)."""
    x, y = L / 2, W / 2
    v = np.array([[-x, -y, 0], [x, -y, 0], [x, y, 0], [-x, y, 0],
                  [-x, -y, H], [x, -y, H], [x, y, H], [-x, y, H]])
    faces = [(4, 5, 6, 7), (3, 2, 1, 0),           # top, bottom
             (0, 1, 5, 4), (2, 3, 7, 6),           # long (-y, +y)
             (1, 2, 6, 5), (3, 0, 4, 7)]           # short (+x, -x)
    q = [(0, 0), (1, 0), (1, 1), (0, 1)]
    uvs = [q, q, q, q, q, q]
    mi = [0, 0, 1, 1, 2, 2]
    obj = _mesh_object(name, v, faces, uvs, mi, mats)
    if bevel > 0:
        add_bevel(obj, bevel)
    return obj


# ---------------------------------------------------------------- 회전체 (캔, 플랜지)

def lathe(name, profile, n_seg=64, mats=(), mat_of_segment=None, uv_v_of_segment=None, smooth=True):
    """profile: [(r, z), ...] 를 z축 둘레로 회전. r==0 인 끝점은 축 정점 1개로 닫는다.
    mat_of_segment[i]: 프로파일 i→i+1 구간의 재질 index.
    uv_v_of_segment[i]: (v0, v1) — 구간의 v 좌표 (라벨 매핑용)."""
    P = len(profile)
    verts, ring_idx = [], []
    for (r, z) in profile:
        if r < 1e-9:
            ring_idx.append([len(verts)] * n_seg)
            verts.append((0, 0, z))
        else:
            idx = []
            for k in range(n_seg):
                a = 2 * math.pi * k / n_seg
                idx.append(len(verts))
                verts.append((r * math.cos(a), r * math.sin(a), z))
            ring_idx.append(idx)
    faces, uvs, mi = [], [], []
    for i in range(P - 1):
        v0, v1 = (uv_v_of_segment[i] if uv_v_of_segment else (i / (P - 1), (i + 1) / (P - 1)))
        for k in range(n_seg):
            k2 = (k + 1) % n_seg
            a, b, c, d = ring_idx[i][k], ring_idx[i][k2], ring_idx[i + 1][k2], ring_idx[i + 1][k]
            u0, u1 = k / n_seg, (k + 1) / n_seg
            if a == b:      # 아래 끝이 축
                f, fu = (a, c, d), [(u0, v0), (u1, v1), (u0, v1)]
            elif c == d:    # 위 끝이 축
                f, fu = (a, b, c), [(u0, v0), (u1, v0), (u0, v1)]
            else:
                f, fu = (a, b, c, d), [(u0, v0), (u1, v0), (u1, v1), (u0, v1)]
            faces.append(f)
            uvs.append(fu)
            mi.append(mat_of_segment[i] if mat_of_segment else 0)
    return _mesh_object(name, np.array(verts), faces, uvs, mi, mats, smooth=smooth)


def cylinder_cutter(name, r, h, z0=-0.05):
    o = bproc.object.create_primitive("CYLINDER", vertices=32)
    o.set_name(name)
    o.set_scale([r, r, h / 2])
    o.set_location([0, 0, z0 + h / 2])
    return o


# ---------------------------------------------------------------- 압출 다각형 (브래킷)

def extrude_polygon_xz(name, poly_xz, width, mats=()):
    """XZ 평면 다각형(반시계)을 y 방향으로 width만큼 압출."""
    n = len(poly_xz)
    y0, y1 = -width / 2, width / 2
    verts = [(x, y0, z) for x, z in poly_xz] + [(x, y1, z) for x, z in poly_xz]
    faces = [tuple(range(n)), tuple(range(2 * n - 1, n - 1, -1))]
    for i in range(n):
        j = (i + 1) % n
        faces.append((i, j, n + j, n + i))
    return _mesh_object(name, np.array(verts), faces, None, None, mats)


# ---------------------------------------------------------------- 베개형 연포장 (봉지, 스틱, 파우치)

def pillow(name, W, H, T, rng, seal=(0.0, 0.08), wrinkle=0.15, bend=0.0, n=(28, 40), mats=()):
    """W(x)×H(y) 크기, 최대 두께 T의 닫힌 베개형 메시.
    seal = (x방향, y방향) 실링 폭 비율. 실링부는 두께가 거의 0(접합부).
    wrinkle = 두께 대비 구김 진폭, bend = 길이 방향 굽힘(m, 중앙 처짐량).
    천 시뮬레이션 대신 노이즈 변형으로 근사한다."""
    nu, nv = n
    us, vs = np.linspace(-1, 1, nu), np.linspace(-1, 1, nv)
    U, V = np.meshgrid(us, vs, indexing="ij")
    sx, sy = 1 - seal[0], 1 - seal[1]
    gu = np.clip(1 - (np.abs(U) / sx) ** 4, 0, 1) if seal[0] < 1 else np.ones_like(U)
    gv = np.clip(1 - (np.abs(V) / sy) ** 4, 0, 1)
    g = np.sqrt(gu * gv)
    g = np.maximum(g, 0.03)
    g[0, :] = g[-1, :] = g[:, 0] = g[:, -1] = 0.0

    def lowfreq(cells):
        small = rng.normal(0, 1, (cells, cells))
        from scipy.ndimage import zoom
        z = zoom(small, (nu / cells, nv / cells), order=3)
        return z[:nu, :nv]
    wr_top = wrinkle * T * (lowfreq(5) * 0.6 + lowfreq(11) * 0.4)
    wr_bot = wrinkle * T * (lowfreq(5) * 0.6 + lowfreq(11) * 0.4)
    X, Y = U * W / 2, V * H / 2
    bend_z = bend * (1 - V ** 2) + rng.uniform(-0.3, 0.3) * bend * (1 - U ** 2)
    # 실링부 크림프(주름) 무늬
    crimp = np.where(g < 0.05, 0.0008 * np.sin(U * 60), 0.0)
    Zt = T / 2 * g + wr_top * np.sqrt(g) + bend_z + crimp
    Zb = -T / 2 * g + wr_bot * np.sqrt(g) + bend_z + crimp

    verts, top_id, bot_id = [], np.zeros((nu, nv), int), np.zeros((nu, nv), int)
    for i in range(nu):
        for j in range(nv):
            top_id[i, j] = len(verts)
            verts.append((X[i, j], Y[i, j], Zt[i, j]))
    for i in range(nu):
        for j in range(nv):
            if i in (0, nu - 1) or j in (0, nv - 1):
                bot_id[i, j] = top_id[i, j]
            else:
                bot_id[i, j] = len(verts)
                verts.append((X[i, j], Y[i, j], Zb[i, j]))
    faces, uvs, mi = [], [], []
    for i in range(nu - 1):
        for j in range(nv - 1):
            u0, u1, v0, v1 = i / (nu - 1), (i + 1) / (nu - 1), j / (nv - 1), (j + 1) / (nv - 1)
            faces.append((top_id[i, j], top_id[i + 1, j], top_id[i + 1, j + 1], top_id[i, j + 1]))
            uvs.append([(u0, v0), (u1, v0), (u1, v1), (u0, v1)])
            mi.append(0)
            f = (bot_id[i, j + 1], bot_id[i + 1, j + 1], bot_id[i + 1, j], bot_id[i, j])
            if len(set(f)) >= 3:
                f = tuple(dict.fromkeys(f))
                faces.append(f)
                uvs.append([(1 - u0, v1), (1 - u1, v1), (1 - u1, v0), (1 - u0, v0)][:len(f)])
                mi.append(1 if len(mats) > 1 else 0)
    obj = _mesh_object(name, np.array(verts), faces, uvs, mi, mats, smooth=True)
    return obj
