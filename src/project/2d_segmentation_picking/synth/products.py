"""제품 7종 절차적 생성 (Blender 내부 전용).

category_id (모든 [ONE]/[FEW] 데이터 공통):
  1 can         캔 음료         (강체 소비재)
  2 paper_box   종이상자        (강체 소비재)
  3 bracket     금속 브래킷     (산업 부품, 무늬 없음·반사)
  4 flange      플랜지          (산업 부품, 무늬 없음·반사)
  5 snack_bag   과자봉지        (연성 포장)
  6 stick_pack  스틱형 포장     (연성 포장)
  7 pouch       파우치          (연성 포장)

한 장면에서 같은 클래스는 같은 SKU(같은 크기·디자인)로 만든다 — 실제 [ONE] 상황과 같다.
"""
from __future__ import annotations

import numpy as np

import geometry as G
import textures as T

CLASSES = {1: "can", 2: "paper_box", 3: "bracket", 4: "flange", 5: "snack_bag", 6: "stick_pack", 7: "pouch"}
CLASS_IDS = {v: k for k, v in CLASSES.items()}


class SKU:
    """제품 1종의 크기·재질 (장면 내 공유). make()로 인스턴스를 찍어낸다."""

    def __init__(self, cls, rng, cfg):
        self.cls, self.rng, self.cfg = cls, rng, cfg
        getattr(self, f"_init_{cls}")()

    def make(self, name):
        obj = getattr(self, f"_make_{self.cls}")(name)
        obj.set_cp("category_id", CLASS_IDS[self.cls])
        return obj

    # 투영 면적 (m^2) — 개수 산정용
    footprint = 0.0
    collision = "CONVEX_HULL"

    # ---------------- can
    def _init_can(self):
        r = self.rng
        d, h = [(0.066, 0.115), (0.066, 0.168), (0.058, 0.146), (0.053, 0.135)][r.integers(4)]
        self.d, self.h = d, h
        label = T.make_print(r, np.pi * d * 1000, h * 1000 * 0.85)
        self.mats = [G.make_material("can_alu", base_color=(200, 200, 205), roughness=r.uniform(0.15, 0.3),
                                     metallic=1.0),
                     G.make_material("can_label", color_img=label, roughness=r.uniform(0.15, 0.35),
                                     metallic=r.uniform(0.3, 0.7))]
        self.footprint = d * h

    def _make_can(self, name):
        R, h = self.d / 2, self.h
        prof = [(0, 0.004), (R * 0.8, 0.0), (R, 0.008), (R, h - 0.012), (R * 0.88, h - 0.003),
                (R * 0.86, h), (R * 0.80, h - 0.004), (0, h - 0.004)]
        seg_mat = [0, 0, 1, 0, 0, 0, 0]
        uvv = [(0, 0)] * 2 + [(0.0, 1.0)] + [(1, 1)] * 4
        o = G.lathe(name, prof, 48, self.mats, seg_mat, uvv)
        return o

    # ---------------- paper box
    def _init_paper_box(self):
        r = self.rng
        L, W, H = [(0.14, 0.09, 0.045), (0.12, 0.12, 0.06), (0.18, 0.11, 0.05), (0.10, 0.07, 0.07)][r.integers(4)]
        self.dims = (L, W, H)
        bg = T.rand_color(r, (0.2, 0.9), (0.5, 1.0))
        imgs = [T.make_print(r, L * 1000, W * 1000, bg=bg),
                T.make_print(r, L * 1000, H * 1000, bg=bg, panel=True),
                T.make_print(r, W * 1000, H * 1000, bg=bg)]
        rough = r.uniform(0.25, 0.6)
        self.mats = [G.make_material(f"pbox_{i}", color_img=im, roughness=rough, coat=r.uniform(0, 0.5))
                     for i, im in enumerate(imgs)]
        self.footprint = L * W

    def _make_paper_box(self, name):
        L, W, H = self.dims
        o = G.box_mesh(name, L, W, H, self.mats, bevel=0.0012)
        G.apply_modifiers(o)
        # 원점을 중심으로 (물리 시뮬레이션용)
        o.blender_obj.data.transform(__import__("mathutils").Matrix.Translation((0, 0, -H / 2)))
        return o

    # ---------------- bracket
    def _init_bracket(self):
        r = self.rng
        self.a = r.uniform(0.04, 0.08)      # 다리 1 길이
        self.b = r.uniform(0.03, 0.07)      # 다리 2 길이
        self.t = r.uniform(0.003, 0.006)    # 두께
        self.w = r.uniform(0.03, 0.06)      # 폭
        finish = r.choice(["zinc", "stainless", "black", "yellow_zinc"])
        base = {"zinc": (190, 195, 200), "stainless": (175, 175, 172), "black": (30, 30, 32),
                "yellow_zinc": (200, 185, 120)}[finish]
        metallic = 0.0 if finish == "black" else 1.0
        self.mats = [G.make_material("bracket", color_img=T.metal_texture(r, base),
                                     roughness=r.uniform(0.2, 0.5), metallic=metallic)]
        self.finish = str(finish)
        self.footprint = self.a * self.w
        self.collision = "CONVEX_HULL"

    def _make_bracket(self, name):
        a, b, t, w = self.a, self.b, self.t, self.w
        poly = [(0, 0), (a, 0), (a, t), (t, t), (t, b), (0, b)]
        o = G.extrude_polygon_xz(name, poly, w, self.mats)
        cut = [G.cylinder_cutter(name + "_h1", w * 0.15, 0.02), G.cylinder_cutter(name + "_h2", w * 0.15, 0.02)]
        cut[0].set_location([a * 0.65, 0, -0.005])
        cut[1].set_rotation_euler([0, np.pi / 2, 0])
        cut[1].set_location([-0.005, 0, b * 0.65])
        G.boolean_cut(o, cut)
        G.add_bevel(o, 0.0006, 1)
        G.apply_modifiers(o)
        o.blender_obj.data.transform(__import__("mathutils").Matrix.Translation((-a / 3, 0, -b / 3)))
        return o

    # ---------------- flange
    def _init_flange(self):
        r = self.rng
        self.ro = r.uniform(0.05, 0.08)
        self.ri = self.ro * r.uniform(0.3, 0.45)
        self.rh = self.ri + (self.ro - self.ri) * r.uniform(0.35, 0.55)
        self.t = r.uniform(0.01, 0.018)
        self.hh = r.uniform(0.0, 0.025)
        self.nb = int(r.choice([4, 6, 8]))
        finish = r.choice(["steel", "stainless", "oiled"])
        base = {"steel": (150, 150, 152), "stainless": (180, 180, 178), "oiled": (95, 90, 85)}[finish]
        self.mats = [G.make_material("flange", color_img=T.metal_texture(r, base, brushed=False),
                                     roughness=r.uniform(0.25, 0.55), metallic=1.0)]
        self.footprint = np.pi * self.ro ** 2

    def _make_flange(self, name):
        ro, ri, rh, t, hh = self.ro, self.ri, self.rh, self.t, self.hh
        if hh > 0.004:
            prof = [(ri, 0), (ro, 0), (ro, t), (rh, t), (rh, t + hh), (ri, t + hh), (ri, 0)]
        else:
            prof = [(ri, 0), (ro, 0), (ro, t), (ri, t), (ri, 0)]
        o = G.lathe(name, prof, 64, self.mats, smooth=False)
        rp = (rh + ro) / 2
        rb = min((ro - rh) * 0.3, rp * np.sin(np.pi / self.nb) * 0.6)
        cuts = []
        for k in range(self.nb):
            c = G.cylinder_cutter(f"{name}_b{k}", rb, 0.05)
            c.set_location([rp * np.cos(2 * np.pi * k / self.nb), rp * np.sin(2 * np.pi * k / self.nb), -0.01])
            cuts.append(c)
        G.boolean_cut(o, cuts)
        o.blender_obj.data.transform(__import__("mathutils").Matrix.Translation((0, 0, -(t + hh) / 2)))
        return o

    # ---------------- 연포장 3종
    def _init_soft(self, W, H, Tk, seal, wrinkle, metallic, rough, back_panel):
        r = self.rng
        self.W, self.H, self.Tk, self.seal, self.wrinkle = W, H, Tk, seal, wrinkle
        front = T.make_print(r, W * 1000, H * 1000)
        back = T.make_print(r, W * 1000, H * 1000, panel=back_panel)
        self.mats = [G.make_material("soft_f", color_img=front, roughness=rough, metallic=metallic),
                     G.make_material("soft_b", color_img=back, roughness=rough, metallic=metallic)]
        self.footprint = W * H

    def _init_snack_bag(self):
        r = self.rng
        W, H = [(0.16, 0.22), (0.18, 0.26), (0.13, 0.19)][r.integers(3)]
        self._init_soft(W, H, r.uniform(0.03, 0.06), (0.0, 0.09), r.uniform(0.1, 0.25),
                        r.uniform(0.3, 0.8), r.uniform(0.15, 0.35), True)

    def _init_stick_pack(self):
        r = self.rng
        W, H = [(0.02, 0.11), (0.025, 0.13), (0.03, 0.15)][r.integers(3)]
        self._init_soft(W, H, r.uniform(0.004, 0.008), (0.1, 0.12), r.uniform(0.05, 0.15),
                        r.uniform(0.0, 0.6), r.uniform(0.2, 0.45), False)

    def _init_pouch(self):
        r = self.rng
        W, H = [(0.12, 0.17), (0.14, 0.2), (0.10, 0.15)][r.integers(3)]
        self._init_soft(W, H, r.uniform(0.012, 0.025), (0.07, 0.07), r.uniform(0.05, 0.15),
                        0.0, r.uniform(0.08, 0.25), True)

    def _make_soft(self, name):
        r = self.rng
        bend = r.uniform(-0.15, 0.15) * self.H * 0.2
        return G.pillow(name, self.W, self.H, self.Tk, r, seal=self.seal, wrinkle=self.wrinkle, bend=bend,
                        n=(max(8, int(self.W / 0.006)), max(12, int(self.H / 0.006))), mats=self.mats)

    _make_snack_bag = _make_soft
    _make_stick_pack = _make_soft
    _make_pouch = _make_soft

    def info(self):
        d = dict(cls=self.cls, category_id=CLASS_IDS[self.cls])
        if self.cls == "bracket":
            d["finish"] = self.finish
        return d
