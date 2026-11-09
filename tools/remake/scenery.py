"""
Remake backgrounds: the three 256x64 images (distant mountains, hills, city) painted again in
relief, at the pack's resolution, from the original silhouettes and colors.

  mountains  height field (mass of the range + ridged noise for the erosion gullies), lit from
             the top left; the original dark blue (pen 1) faces stay in shadow; haze below
  hills      the same, smoother, with grass-like grain
  city       each row of a building is a slice of a body of revolution: cylinder shading, domes
             from the vertical slope; the black windows (pen 1) glow

Colors: the moon in greys (MOON_PALETTES, the pens of the original kept as shades): light distant
mountains, dark hills and city; the ground (text layer, recolored by the shader) is in between.

The images wrap horizontally (the game scrolls them around), so every filter and noise here is
periodic in x. Bottom rows end in the solid color the game draws below the image (pen 3).
"""

import numpy as np
from PIL import Image

LIGHT = np.array([-0.55, -0.6, 0.58])        # screen coordinates: x right, y down, z to the viewer
LIGHT = LIGHT / np.linalg.norm(LIGHT)


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def pblur(a, sigma):
    """Gaussian blur, periodic in x, zero above and below."""
    if sigma <= 0:
        return a
    r = int(np.ceil(sigma * 3))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    a = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 0, a)
    w = a.shape[1]
    wrapped = np.concatenate([a[:, -r:], a, a[:, :r]], axis=1) if r < w else np.tile(a, (1, 3))
    out = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 1, wrapped)
    return out[:, r:r + w] if r < w else out[:, w:2 * w]


def pnoise(h, w, cell, rng):
    """Value noise, periodic in x (w must be a multiple of cell)."""
    gw, gh = max(1, w // cell), h // cell + 2
    g = rng.random((gh, gw))
    y, x = np.arange(h) / cell, np.arange(w) / cell
    y0, x0 = y.astype(int), x.astype(int)
    fy, fx = y - y0, x - x0
    fy, fx = fy * fy * (3 - 2 * fy), fx * fx * (3 - 2 * fx)
    xa, xb = x0 % gw, (x0 + 1) % gw
    a = g[y0][:, xa] * (1 - fx) + g[y0][:, xb] * fx
    b = g[y0 + 1][:, xa] * (1 - fx) + g[y0 + 1][:, xb] * fx
    return a * (1 - fy)[:, None] + b * fy[:, None]


def pfbm(h, w, cell, rng, octaves=4):
    out, amp, total = np.zeros((h, w)), 1.0, 0.0
    for _ in range(octaves):
        out += amp * pnoise(h, w, max(1, cell), rng)
        total += amp
        amp *= 0.5
        cell = max(1, cell // 2)
    return out / total


def pupscale(a, px):
    """Bicubic upscale, periodic in x."""
    pad = 4
    wrapped = np.concatenate([a[:, -pad:], a, a[:, :pad]], axis=1).astype(np.float32)
    img = Image.fromarray(wrapped, "F").resize((wrapped.shape[1] * px, a.shape[0] * px), Image.BICUBIC)
    return np.asarray(img, np.float64)[:, pad * px:(pad + a.shape[1]) * px]


def lin(c):
    c = np.asarray(c, np.float64) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def normals(height, px):
    """Normals of a height field (native pixel units), periodic in x."""
    gx = (np.roll(height, -1, axis=1) - np.roll(height, 1, axis=1)) * 0.5 * px
    gy = np.gradient(height, axis=0) * px
    n = np.stack([-gx, -gy, np.ones_like(height)], axis=-1)
    return n / np.linalg.norm(n, axis=-1, keepdims=True)


def silhouette(mask, px, rng, rough=0.0, cell=2):
    cover = pupscale(pblur(mask.astype(float), 0.6), px)
    if rough:
        cover += rough * (pfbm(*cover.shape, cell * px, rng) - 0.5)
    return smoothstep(0.42, 0.58, cover)


def haze_to_fill(rgb, alpha, fill, px, start, end):
    """Fade towards the solid fill color between native rows start..end (inside the shapes)."""
    H = rgb.shape[0]
    y = np.arange(H)[:, None] / px
    t = smoothstep(start, end, y)[..., None]
    return rgb * (1 - t) + fill * t


def floating(img, pen):
    """Pixels of `pen` in pieces that touch no other pen: drawn in the sky, away from the
    range (the distant mountains have one, a smudge once in relief)."""
    h, w = img.shape
    seen = np.zeros(img.shape, bool)
    out = np.zeros(img.shape, bool)
    for y0, x0 in zip(*np.nonzero(img == pen)):
        if seen[y0, x0]:
            continue
        piece, stack, touches = [], [(y0, x0)], False
        seen[y0, x0] = True
        while stack:
            y, x = stack.pop()
            piece.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    yy, xx = y + dy, (x + dx) % w
                    if not 0 <= yy < h:
                        continue
                    v = img[yy, xx]
                    if v == pen and not seen[yy, xx]:
                        seen[yy, xx] = True
                        stack.append((yy, xx))
                    elif v not in (0, pen):
                        touches = True
        if not touches:
            out[tuple(np.array(piece).T)] = True
    return out


def terrain(img, pal, px, seed, mass_sigma, mass_k, ridge_k, ridge_cell, rough, haze, shadow_k, peak_tint):
    rng = np.random.default_rng(seed)
    img = np.where(floating(img, 2), 0, img)
    mask = img != 0
    alpha = silhouette(mask, px, rng, rough=rough)
    H, W = alpha.shape
    inside = alpha > 0.5
    # height: mass of the range, plus ridges (in native pixel units)
    mass = pupscale(pblur(mask.astype(float), mass_sigma), px) * mass_k
    # gullies: ridged noise stretched vertically (erosion runs down the slopes)
    s = 3
    g = 1 - np.abs(2 * pfbm(H // s + 2, W, ridge_cell * px, rng) - 1)
    g = np.asarray(Image.fromarray(g.astype(np.float32), "F").resize((W, (H // s + 2) * s), Image.BICUBIC))[:H]
    ridges = 0.65 * g + 0.35 * (1 - np.abs(2 * pfbm(H, W, ridge_cell * px, rng) - 1))
    fine = pfbm(H, W, px, rng, octaves=3)
    height = mass + ridge_k * ridges + 0.15 * fine
    height = pblur(height, 0.4 * px)
    n = normals(height, px)
    ndl = np.clip(n @ LIGHT, 0, 1)
    # original shadow faces (pen 1)
    shadow = np.clip(pupscale(pblur((img == 1).astype(float), 0.55), px) * 1.5, 0, 1)
    # 0.5 = flat (facing the viewer), 1 = facing the light
    lit = np.clip(0.5 + 1.3 * (ndl - LIGHT[2]), 0, 1) * (1 - shadow_k * shadow)

    base, dark = lin(pal[3]), lin(pal[1])
    lit_col = base * 1.6 + 0.02
    shadow_col = base * 0.3 + dark * 0.15            # so that flat ground (lit = 0.5) keeps the color
    col = shadow_col + (lit_col - shadow_col) * lit[..., None]
    col *= (0.9 + 0.2 * fine)[..., None]
    # peaks a little brighter (light from the sky catches them)
    top = np.clip(1 - pupscale(pblur(mask.astype(float), 2.0), px) * 1.4, 0, 1)
    col += np.asarray(peak_tint) * (top * lit)[..., None]
    # pen 2 is black in the original (a little figure drawn on the mountains): keep it dark
    black = np.clip(pupscale(pblur((img == 2).astype(float), 0.5), px) * 1.6, 0, 1)
    col *= (1 - 0.85 * black)[..., None]
    col = haze_to_fill(col, alpha, base, px, *haze)
    return np.concatenate([srgb(col), alpha[..., None]], axis=-1)


def city(img, pal, px, seed):
    rng = np.random.default_rng(seed)
    mask = img != 0
    alpha = silhouette(mask, px, rng)
    H, W = alpha.shape
    inside = alpha > 0.5
    # horizontal slope: each run of a row is a cylinder slice
    nx = np.zeros((H, W))
    for y in range(H):
        row = inside[y]
        if not row.any():
            continue
        edges = np.flatnonzero(np.diff(np.concatenate([[0], row.astype(int), [0]])))
        for a, b in zip(edges[::2], edges[1::2]):
            if b - a < 2:
                continue
            c, half = (a + b - 1) / 2, (b - a) / 2
            nx[y, a:b] = (np.arange(a, b) - c) / half
    nx = np.clip(pblur(nx, 0.6 * px) / np.maximum(pblur(inside.astype(float), 0.6 * px), 1e-3), -1, 1) * inside
    # vertical slope (domes, roofs): gradient of the blurred silhouette
    soft = pblur(inside.astype(float), 1.5 * px)
    ny = np.gradient(soft, axis=0) * px * 1.6
    nz = np.sqrt(np.clip(1 - nx ** 2, 0.02, 1))
    n = np.stack([nx * 0.95, -ny, nz], axis=-1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    ndl = np.clip(n @ LIGHT, 0, 1)
    half = LIGHT + np.array([0, 0, 1.0])
    half /= np.linalg.norm(half)
    spec = np.clip(n @ half, 0, 1) ** 30 * 0.5

    body, glow = lin(pal[3]), lin(pal[2])
    shade = 0.25 + 0.95 * ndl
    col = body * shade[..., None]
    # warm light on the lit side (the yellow of the original), cold reflection on the other one
    col += glow * (np.clip(ndl - 0.55, 0, 1) * 1.1)[..., None]
    col += np.array([0.01, 0.03, 0.05]) * (np.clip(nx, 0, 1) ** 3)[..., None] * inside[..., None]
    col += spec[..., None]
    # panel lines
    bands = pfbm(H, W, 2 * px, rng, octaves=2)
    col *= (0.92 + 0.12 * bands)[..., None]

    # windows: the black pixels light up
    win = pupscale(pblur((img == 1).astype(float), 0.45), px)
    win_a = smoothstep(0.25, 0.55, win)
    lamp = lin((255, 210, 110)) * 1.4
    col = col * (1 - win_a[..., None]) + lamp * win_a[..., None]
    halo = pblur(win_a, 1.2 * px)
    col += lamp * 0.35 * halo[..., None]
    col = haze_to_fill(col, alpha, body, px, 56, 63.5)
    return np.concatenate([srgb(col), alpha[..., None]], axis=-1)


# pens 0..3 of the mountains, hills, city (pen 3 is also the solid fill below each image)
MOON_PALETTES = [
    [(0, 0, 0), (92, 96, 108), (0, 0, 0), (172, 175, 182)],       # mountains: light grey, bluish shadows
    [(0, 0, 0), (42, 43, 48), (0, 0, 0), (86, 88, 94)],           # hills: dark grey
    [(0, 0, 0), (0, 0, 0), (215, 205, 180), (92, 94, 100)],       # city: dark grey, warm light, windows
]


def fills():
    """The solid color the game draws below each image (pen 3), for the shader."""
    return [list(p[3]) for p in MOON_PALETTES]


def paint_all(bg_images, bg_palette, px):
    """bg_images: 3 (rows, 256) pen arrays (only the first 64 rows are drawn); returns 3 RGBA.
    bg_palette: the arcade's colors, unused: the remake paints the moon in greys."""
    imgs = [np.asarray(b[:64]) for b in bg_images]
    pals = [np.asarray(p, np.float64) for p in MOON_PALETTES]
    mountains = terrain(imgs[0], pals[0], px, 1, mass_sigma=2.2, mass_k=14.0, ridge_k=3.0, ridge_cell=3,
                        rough=0.12, haze=(30, 63.5), shadow_k=0.7, peak_tint=(0.2, 0.22, 0.27))
    hills = terrain(imgs[1], pals[1], px, 2, mass_sigma=3.5, mass_k=12.0, ridge_k=0.8, ridge_cell=6,
                    rough=0.0, haze=(40, 63.5), shadow_k=0.75, peak_tint=(0.06, 0.06, 0.07))
    return [mountains, hills, city(imgs[2], pals[2], px, 3)]
