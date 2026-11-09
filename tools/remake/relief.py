"""
Remake art in relief for sprites of any color: the silhouette is inflated into a height field
(rounded like a pressed metal body), each pen keeps its color (smoothed borders, a groove between
pens) and the result is lit like the 3D models. The colors are given per pen, so the same sprite
can be painted for each color the game may use (remake.json "code/color" entries).
"""

import numpy as np

from fire import blur, upscale
from rocks import inside_distance
from sdf import KEY_COLOR, RIM_COLOR, linear_to_srgb, srgb_to_linear

LIGHT = np.array([-0.45, -0.75, 0.5])
LIGHT = LIGHT / np.linalg.norm(LIGHT)
PAD = 2


def paint(pens, colors, px, roundness=2.5, metal=0.5):
    """pens: (h, w); colors: {pen: sRGB 0..255} (pens without a color are transparent)."""
    pens = np.pad(np.asarray(pens), PAD)
    visible = np.isin(pens, list(colors))
    cover = upscale(blur(visible.astype(float), 0.55), px)
    alpha = np.clip((cover - 0.42) / 0.16, 0, 1)

    # height: rounded profile from the distance to the outline
    dist = inside_distance(cover > 0.5) / px
    t = np.clip(dist / roundness, 0, 1)
    height = roundness * np.sqrt(1 - (1 - t) ** 2)
    # pen coverage, smoothed; a shallow groove where two pens meet
    fields = {p: np.clip(upscale(blur((pens == p).astype(float), 0.5), px), 0, 1) ** 4 for p in colors}
    total = sum(fields.values()) + 1e-9
    w = {p: f / total for p, f in fields.items()}
    border = 1 - np.max(np.stack(list(w.values())), axis=0)
    height -= 0.6 * np.clip(border * 2, 0, 1)
    height = blur(height, 0.2 * px)

    gy, gx = np.gradient(height * px)
    n = np.stack([-gx, -gy, np.ones_like(height)], axis=-1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    ndl = np.clip(n @ LIGHT, 0, 1)
    half = LIGHT + np.array([0, 0, 1.0])
    half /= np.linalg.norm(half)
    spec = np.clip(n @ half, 0, 1) ** 40 * (0.2 + 0.6 * metal)
    rim = (1 - np.clip(n[..., 2], 0, 1)) ** 3 * 0.3

    albedo = sum(srgb_to_linear(colors[p]) * w[p][..., None] for p in colors)
    col = albedo * (KEY_COLOR * ndl[..., None] * 1.05 + 0.22) + spec[..., None] * KEY_COLOR + RIM_COLOR * rim[..., None]
    return np.concatenate([linear_to_srgb(col), alpha[..., None]], axis=-1)
