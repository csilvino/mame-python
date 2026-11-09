"""
Painted fire for remake sprites: explosions, flames, shots.

The original sprites draw fire in bands of pens (Moon Patrol: red core, green middle, magenta
rim). Each pen gets a heat; the heat field is smoothed, upscaled, broken up with noise and
colored with a black body ramp (smoke -> red -> orange -> yellow -> white), plus a soft glow
around the hot parts. The shape and timing of the animation stay those of the arcade.
"""

import numpy as np
from PIL import Image

# heat -> color (sRGB), black body like, with dark smoke at the cold end
RAMP = [(0.00, (0.10, 0.08, 0.08)),
        (0.25, (0.45, 0.08, 0.03)),
        (0.45, (0.90, 0.25, 0.03)),
        (0.65, (1.00, 0.55, 0.08)),
        (0.85, (1.00, 0.85, 0.35)),
        (1.00, (1.00, 0.98, 0.85))]


# plasma (the UFO bombs): violet -> magenta -> pink -> white
PLASMA = [(0.00, (0.12, 0.02, 0.20)),
          (0.30, (0.55, 0.05, 0.60)),
          (0.55, (1.00, 0.20, 0.80)),
          (0.80, (1.00, 0.60, 0.95)),
          (1.00, (1.00, 0.95, 1.00))]
# green energy
GREEN = [(0.00, (0.02, 0.12, 0.04)),
         (0.35, (0.15, 0.60, 0.10)),
         (0.65, (0.55, 1.00, 0.25)),
         (1.00, (0.95, 1.00, 0.85))]


def ramp(h, smoke=None, stops=None):
    h = np.clip(h, 0, 1)
    out = np.zeros(h.shape + (3,))
    stops = stops or RAMP
    if smoke is not None:
        stops = [(0.0, smoke)] + stops[1:]
    for (h0, c0), (h1, c1) in zip(stops, stops[1:]):
        t = np.clip((h - h0) / (h1 - h0), 0, 1)[..., None]
        sel = ((h >= h0) & (h <= h1))[..., None]
        out = np.where(sel, np.asarray(c0) * (1 - t) + np.asarray(c1) * t, out)
    return out


def blur(a, sigma):
    """Separable gaussian blur (zero outside)."""
    if sigma <= 0:
        return a
    r = int(np.ceil(sigma * 3))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    a = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 0, a)
    return np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 1, a)


def value_noise(h, w, cell, rng):
    gh, gw = h // cell + 2, w // cell + 2
    g = rng.random((gh, gw))
    y = np.arange(h) / cell
    x = np.arange(w) / cell
    y0, x0 = y.astype(int), x.astype(int)
    fy, fx = y - y0, x - x0
    fy, fx = fy * fy * (3 - 2 * fy), fx * fx * (3 - 2 * fx)
    a = g[y0][:, x0] * (1 - fx) + g[y0][:, x0 + 1] * fx
    b = g[y0 + 1][:, x0] * (1 - fx) + g[y0 + 1][:, x0 + 1] * fx
    return a * (1 - fy)[:, None] + b * fy[:, None]


def fbm(h, w, cell, rng, octaves=4):
    out, amp, total = np.zeros((h, w)), 1.0, 0.0
    for _ in range(octaves):
        out += amp * value_noise(h, w, max(1, cell), rng)
        total += amp
        amp *= 0.5
        cell //= 2
    return out / total


def upscale(a, px):
    img = Image.fromarray(a.astype(np.float32), "F")
    return np.asarray(img.resize((a.shape[1] * px, a.shape[0] * px), Image.BICUBIC), np.float64)


def paint(pens, heat_of_pen, px, pad=4, seed=0, glow=1.0, smoke_alpha=1.0, roughness=1.0, smoke=None,
          sparks=None, stops=None, glow_rgb=(1.0, 0.55, 0.15)):
    """
    pens: (h, w) native pen indices (0 = empty). heat_of_pen: {pen: heat 0..1}.
    smoke: color of heat 0 (default dark smoke; dust clouds are light).
    sparks: True: isolated pixels (at most one neighbor) are drawn as glowing dots (debris flying
            away) instead of flames, with the heat of their pen.
    stops: color ramp (default RAMP, black body), glow_rgb: color of the halo.
    Returns (RGBA float image, straight alpha), covering the pens plus `pad` native pixels.
    """
    pens = np.pad(pens, pad)
    filled = pens != 0
    spark_px = np.zeros(pens.shape, bool)
    if sparks:
        f = np.pad(filled, 1).astype(int)
        neighbors = sum(f[1 + dy:f.shape[0] - 1 + dy, 1 + dx:f.shape[1] - 1 + dx]
                        for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx)
        spark_px = filled & (neighbors <= 1)
    cover = (filled & ~spark_px).astype(np.float64)
    heat = np.zeros(pens.shape)
    for pen, h in heat_of_pen.items():
        heat[(pens == pen) & ~spark_px] = h
    rng = np.random.default_rng(seed)

    # smooth shapes at native size, then upscale
    c = upscale(blur(cover, 0.7), px)
    # heat inside the shape only: normalized blur, so the rim keeps its own heat
    hs = upscale(blur(heat, 0.8), px) / np.maximum(c, 1e-3)
    H, W = c.shape

    n1 = fbm(H, W, 4 * px, rng)
    n2 = fbm(H, W, 2 * px, rng)
    alpha = np.clip((c + roughness * 0.35 * (n1 - 0.5) - 0.32) / 0.2, 0, 1)
    h = np.clip(hs + roughness * 0.35 * (n2 - 0.5), 0, 1) * np.clip(c * 1.6, 0, 1)
    rgb = ramp(h, smoke, stops)
    # hotter parts glow through; smoke is a little transparent
    alpha *= np.clip(smoke_alpha + (1 - smoke_alpha) * h * 3, 0, 1)

    if sparks:
        # each spark pixel: a small gaussian dot, hotter in its center
        dots = np.zeros(pens.shape)
        for pen, sh in heat_of_pen.items():
            dots[spark_px & (pens == pen)] = max(sh, 0.05)
        spot = np.zeros((H, W))
        spot_heat = np.zeros((H, W))
        ys, xs = np.nonzero(dots)
        yy, xx = np.mgrid[0:H, 0:W]
        for y, x in zip(ys, xs):
            jx, jy = rng.normal(0, 0.15, 2)
            g = np.exp(-(((xx + 0.5) / px - x - 0.5 - jx) ** 2 + ((yy + 0.5) / px - y - 0.5 - jy) ** 2) / (2 * 0.42 ** 2))
            spot = np.maximum(spot, g)
            spot_heat = np.maximum(spot_heat, g * dots[y, x])
        sa = np.clip(spot * 1.6 - 0.15, 0, 1)
        sh = np.clip(spot_heat / np.maximum(spot, 1e-6) * (0.7 + 0.4 * spot), 0, 1)
        srgb = ramp(sh, None, stops)
        rgb = (rgb * (alpha * (1 - sa))[..., None] + srgb * sa[..., None]) / np.maximum(alpha * (1 - sa) + sa, 1e-6)[..., None]
        h = np.maximum(h, sh * sa)
        alpha = alpha + sa * (1 - alpha)

    # glow: blurred heat around the fire, blended as a light (small alpha, bright color)
    g = blur(np.clip(h - 0.35, 0, 1) * alpha, 2.5 * px) * 2.0 * glow
    g = np.clip(g, 0, 1)
    glow_rgb = np.asarray(glow_rgb)
    out_a = alpha + g * 0.45 * (1 - alpha)
    out_rgb = (rgb * alpha[..., None] + glow_rgb * (g * 0.45 * (1 - alpha))[..., None]) / np.maximum(out_a, 1e-6)[..., None]
    return np.concatenate([out_rgb, out_a[..., None]], axis=2)
