"""절차적 텍스처 생성 (PIL + numpy만 사용, Blender 밖에서도 import 가능).

모든 무늬·로고·문구는 난수로 만든 가상 디자인이다. 실제 브랜드를 쓰지 않으며,
외부 이미지 에셋을 내려받지 않는다(라이선스 문제 없음).
폰트는 Windows 기본 폰트(C:\\Windows\\Fonts)를 렌더링에만 쓴다.
"""
from __future__ import annotations

import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_DIR = r"C:\Windows\Fonts"
FONTS = ["arialbd.ttf", "arial.ttf", "impact.ttf", "bahnschrift.ttf", "verdanab.ttf",
         "georgiab.ttf", "seguibl.ttf", "tahomabd.ttf", "consolab.ttf"]
SYLL = ["ka", "ro", "mi", "tex", "vo", "lin", "sa", "dor", "ne", "qu", "zu", "pel", "ar", "is", "on", "bri",
        "fa", "mo", "tri", "ven", "lo", "ga", "ny", "ux"]


def _font(rng, size):
    name = FONTS[rng.integers(len(FONTS))]
    try:
        return ImageFont.truetype(os.path.join(FONT_DIR, name), max(6, int(size)))
    except OSError:
        return ImageFont.load_default()


def fake_word(rng, n_min=2, n_max=3):
    return "".join(SYLL[i] for i in rng.integers(len(SYLL), size=rng.integers(n_min, n_max + 1))).upper()


def noise(h, w, cells, rng):
    """저주파 노이즈 [-1, 1]. cells = 짧은 변 기준 격자 수."""
    ch = max(2, int(cells * h / min(h, w)))
    cw = max(2, int(cells * w / min(h, w)))
    small = rng.uniform(-1, 1, (ch, cw)).astype(np.float32)
    img = Image.fromarray(((small + 1) * 127.5).astype(np.uint8)).resize((w, h), Image.BICUBIC)
    return np.asarray(img, np.float32) / 127.5 - 1


def rand_color(rng, sat=(0.5, 1.0), val=(0.4, 1.0)):
    import colorsys
    r, g, b = colorsys.hsv_to_rgb(rng.uniform(), rng.uniform(*sat), rng.uniform(*val))
    return int(r * 255), int(g * 255), int(b * 255)


def to_img(arr):
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


# ---------------------------------------------------------------- 골판지 박스

def cardboard_base(h, w, rng, white=False):
    base = np.array([228, 224, 214] if white else
                    [rng.uniform(175, 205), rng.uniform(135, 160), rng.uniform(90, 112)], np.float32)
    shade = 1 + 0.06 * noise(h, w, 4, rng) + 0.03 * noise(h, w, 40, rng) + rng.normal(0, 0.015, (h, w))
    return base[None, None, :] * shade[..., None]


def make_box_design(rng, dims_mm, px_per_mm=1.5, max_px=1400):
    """박스 1종(SKU)의 면별 텍스처 세트를 만든다.

    dims_mm = (L, W, H). 반환: dict(face -> (color PIL, roughness PIL))
      face: 'top'(L×W), 'long'(L×H), 'short'(W×H)
    인쇄 옵션: plain / logo / logo+label / label  (같은 SKU는 같은 디자인)
    테이프: 모든 박스에 있다. 윗면 중앙을 길이 방향으로 지나 짧은 면으로 내려온다.
    """
    L, W, H = dims_mm
    s = min(px_per_mm, max_px / max(L, W, H))
    white = rng.uniform() < 0.2
    style = rng.choice(["plain", "logo", "logo_label", "label"], p=[0.2, 0.35, 0.3, 0.15])
    ink = tuple(int(c) for c in (rand_color(rng, (0.6, 1), (0.2, 0.6)) if rng.uniform() < 0.6
                                 else (30, 30, 30)))
    brand = fake_word(rng)
    font_seed = int(rng.integers(1 << 31))  # 같은 SKU는 모든 면에 같은 폰트
    tape_kind = rng.choice(["brown", "clear", "white_print", "black"], p=[0.4, 0.3, 0.15, 0.15])
    tape_w = rng.choice([48, 60, 72])
    tape_drop = rng.uniform(50, 80)
    faces = {}
    for face, (fw, fh) in {"top": (L, W), "long": (L, H), "short": (W, H)}.items():
        pw, ph = max(16, int(fw * s)), max(16, int(fh * s))
        arr = cardboard_base(ph, pw, rng, white)
        rough = np.full((ph, pw), 0.85 if not white else 0.6, np.float32)
        img = to_img(arr)
        d = ImageDraw.Draw(img)
        if style in ("logo", "logo_label") and face in ("long", "short", "top"):
            if face != "top" or rng.uniform() < 0.5:
                fs = min(ph * 0.35, pw / max(3, len(brand)) * 1.6)
                d.text((pw * rng.uniform(0.08, 0.3), ph * rng.uniform(0.2, 0.45)), brand, fill=ink,
                       font=_font(np.random.default_rng(font_seed), fs))
                if face != "top":
                    # 취급 주의 표시 (위 화살표 2개) — 가상 기호
                    ax, ay, a = pw * 0.82, ph * 0.25, ph * 0.12
                    for k in range(2):
                        x0 = ax + k * a * 0.9
                        d.polygon([(x0, ay), (x0 - a * 0.35, ay + a * 0.5), (x0 + a * 0.35, ay + a * 0.5)], fill=ink)
                        d.rectangle([x0 - a * 0.1, ay + a * 0.5, x0 + a * 0.1, ay + a], fill=ink)
                    # 작은 문구 줄
                    for k in range(3):
                        y = ph * (0.7 + 0.08 * k)
                        d.rectangle([pw * 0.08, y, pw * rng.uniform(0.3, 0.55), y + ph * 0.025], fill=ink)
        if style in ("logo_label", "label") and face in ("top", "long"):
            if face == "long" or rng.uniform() < 0.6:
                lw, lh = 100 * s, 70 * s
                lx, ly = pw * rng.uniform(0.55, 0.9) - lw, ph * rng.uniform(0.1, 0.9) - lh * 0.5
                ly = np.clip(ly, 2, ph - lh - 2)
                d.rectangle([lx, ly, lx + lw, ly + lh], fill=(245, 245, 242))
                for k in range(int(lw / (1.2 * s))):  # 바코드 형태
                    if rng.uniform() < 0.5:
                        x = lx + 4 * s + k * 1.2 * s
                        if x < lx + lw - 4 * s:
                            d.rectangle([x, ly + lh * 0.55, x + 0.8 * s, ly + lh * 0.9], fill=(15, 15, 15))
                for k in range(4):
                    y = ly + lh * (0.1 + 0.1 * k)
                    d.rectangle([lx + 4 * s, y, lx + lw * rng.uniform(0.4, 0.9), y + 0.05 * lh], fill=(40, 40, 40))
                ra = np.asarray(img, np.float32)
                rough[int(ly):int(ly + lh), int(lx):int(lx + lw)] = 0.45
                img = to_img(ra)
                d = ImageDraw.Draw(img)
        # 테이프
        arr = np.asarray(img, np.float32).copy()
        tw = tape_w * s
        if face == "top":
            y0, y1 = int(ph / 2 - tw / 2), int(ph / 2 + tw / 2)
            region = (slice(y0, y1), slice(0, pw))
        elif face == "short":
            x0, x1 = int(pw / 2 - tw / 2), int(pw / 2 + tw / 2)
            region = (slice(0, int(tape_drop * s)), slice(x0, x1))
        else:
            region = None
        if region is not None:
            if tape_kind == "brown":
                arr[region] = arr[region] * 0.35 + np.array([175, 125, 60]) * 0.65
            elif tape_kind == "clear":
                arr[region] = arr[region] * 0.92 + 18
            elif tape_kind == "black":  # 검은 PVC/OPP 테이프 (사용자 요청 2026-11-04)
                arr[region] = arr[region] * 0.05 + np.array([22, 22, 24]) * 0.95
            else:
                arr[region] = arr[region] * 0.1 + np.array([235, 235, 230]) * 0.9
            rough[region] = 0.2
            img = to_img(arr)
            if tape_kind == "white_print":
                d = ImageDraw.Draw(img)
                f = _font(rng, tw * 0.45)
                if face == "top":
                    for x in np.arange(10, pw, tw * 4):
                        d.text((x, y0 + tw * 0.2), brand, fill=(200, 30, 30), font=f)
            # 테이프 가장자리 살짝 어둡게
            a = np.asarray(img, np.float32).copy()
            if face == "top":
                a[max(0, y0 - 1):y0 + 1] *= 0.8
                a[y1 - 1:y1 + 1] *= 0.8
            img = to_img(a)
        faces[face] = (img, to_img(rough * 255).convert("L"))
    return faces, dict(style=str(style), tape=str(tape_kind), white=bool(white), brand=brand)


# ---------------------------------------------------------------- 제품 인쇄

def make_print(rng, w_mm, h_mm, px_per_mm=4.0, max_px=1024, bg=None, panel=False):
    """제품 포장 인쇄면 (캔 라벨, 봉지, 파우치, 종이상자 공용). 가상 디자인."""
    s = min(px_per_mm, max_px / max(w_mm, h_mm))
    pw, ph = max(16, int(w_mm * s)), max(16, int(h_mm * s))
    bg = bg or rand_color(rng, (0.3, 1.0), (0.3, 1.0))
    arr = np.ones((ph, pw, 3), np.float32) * np.array(bg, np.float32)
    # 그라데이션
    g = np.linspace(-1, 1, ph)[:, None, None] * rng.uniform(0, 40)
    arr = arr + g
    img = to_img(arr)
    d = ImageDraw.Draw(img)
    acc = rand_color(rng)
    kind = rng.integers(3)
    if kind == 0:  # 띠
        for _ in range(rng.integers(1, 4)):
            y = rng.uniform(0, ph)
            d.rectangle([0, y, pw, y + rng.uniform(0.05, 0.2) * ph], fill=rand_color(rng))
    elif kind == 1:  # 원/타원
        for _ in range(rng.integers(1, 4)):
            cx, cy, r = rng.uniform(0, pw), rng.uniform(0, ph), rng.uniform(0.1, 0.4) * min(pw, ph)
            d.ellipse([cx - r, cy - r * rng.uniform(0.6, 1.2), cx + r, cy + r], fill=rand_color(rng))
    else:  # 사선
        for _ in range(rng.integers(2, 6)):
            x = rng.uniform(-pw, pw)
            t = rng.uniform(0.03, 0.12) * pw
            d.polygon([(x, ph), (x + t, ph), (x + t + ph * 0.6, 0), (x + ph * 0.6, 0)], fill=rand_color(rng))
    brand = fake_word(rng)
    vertical = h_mm > 2.2 * w_mm
    fs = (pw / max(3, len(brand)) * 1.5) if not vertical else (ph / max(3, len(brand)) * 1.3)
    txt = Image.new("RGBA", (int(fs * len(brand) * 0.8) + 10, int(fs * 1.4) + 10), (0, 0, 0, 0))
    ImageDraw.Draw(txt).text((5, 5), brand, fill=acc + (255,), font=_font(rng, fs),
                             stroke_width=max(1, int(fs * 0.05)), stroke_fill=(255, 255, 255, 255))
    if vertical:
        txt = txt.rotate(90, expand=True)
    tx = int(np.clip(pw * rng.uniform(0.05, 0.4), 0, max(0, pw - txt.width)))
    ty = int(np.clip(ph * rng.uniform(0.15, 0.5), 0, max(0, ph - txt.height)))
    img.paste(txt, (tx, ty), txt)
    d = ImageDraw.Draw(img)
    for k in range(rng.integers(2, 5)):  # 작은 문구 줄
        y = ph * rng.uniform(0.65, 0.95)
        d.rectangle([pw * 0.08, y, pw * rng.uniform(0.3, 0.8), y + max(1, ph * 0.012)], fill=(250, 250, 250))
    if panel:  # 뒷면 성분표 흰 패널 + 바코드
        px0, py0 = pw * 0.55, ph * 0.55
        d.rectangle([px0, py0, pw * 0.95, ph * 0.92], fill=(248, 248, 248))
        for k in range(30):
            if rng.uniform() < 0.5:
                x = px0 + 5 + k * (pw * 0.35 / 32)
                d.rectangle([x, ph * 0.78, x + pw * 0.006, ph * 0.9], fill=(10, 10, 10))
    return img


def metal_texture(rng, base, h=256, w=256, brushed=True):
    arr = np.ones((h, w, 3), np.float32) * np.array(base, np.float32)
    if brushed:
        streak = rng.normal(0, 1, (1, w)).repeat(h, 0)
        streak = np.asarray(Image.fromarray(((streak + 3) * 40).clip(0, 255).astype(np.uint8)).resize((w, h)), np.float32) / 40 - 3
        arr *= (1 + 0.04 * streak)[..., None]
    arr *= (1 + 0.05 * noise(h, w, 6, rng))[..., None]
    return to_img(arr)


def concrete(rng, h=1024, w=1024):
    base = rng.uniform(110, 170)
    a = base * (1 + 0.12 * noise(h, w, 5, rng) + 0.05 * noise(h, w, 50, rng) + rng.normal(0, 0.03, (h, w)))
    tint = np.array([1.0, rng.uniform(0.97, 1.02), rng.uniform(0.92, 1.0)])
    return to_img(a[..., None] * tint)


def wood(rng, h=128, w=1024):
    base = np.array([rng.uniform(165, 205), rng.uniform(130, 160), rng.uniform(85, 110)], np.float32)
    y = np.linspace(0, 1, h)[:, None]
    grain = np.sin((y * rng.uniform(20, 40) + 0.3 * noise(h, w, 8, rng)) * np.pi * 2)
    a = base * (1 + 0.07 * grain + 0.06 * noise(h, w, 10, rng))[..., None]
    return to_img(a)


def plastic(rng, color, h=256, w=256):
    a = np.ones((h, w, 3), np.float32) * np.array(color, np.float32)
    a *= (1 + 0.03 * noise(h, w, 8, rng))[..., None]
    return to_img(a)
