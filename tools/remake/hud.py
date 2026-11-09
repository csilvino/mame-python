"""
Remake art for the score panel and the messages (text layer above the ground) and the bonus
score sprites.

Text tiles: for every code, the coverage of each of its 4 pens at high resolution, smoothed
(the 8x8 pixel stairs become curves): RGBA = coverage of pens 0, 1, 2, 3. The shader mixes the
palette colors of the tile with these weights, so the game's colors (and their changes) stay.
Shapes that continue into the next tile (panel frame, progress bar) are smoothed with the edge
pixels repeated, so they meet their neighbors.

Bonus scores (sprites 7D "500", 7E "1000"): smooth bright digits with a dark outline and a glow.
"""

import numpy as np
from PIL import Image

from fire import blur


def _up(a, px):
    img = Image.fromarray(a.astype(np.float32), "F")
    return np.asarray(img.resize((a.shape[1] * px, a.shape[0] * px), Image.BICUBIC), np.float64)


def pen_coverage(pens, px, pad=2, sharpness=8.0):
    """pens: (8, 8). Returns (8px, 8px, 4) coverage of pens 0..3 (sums to 1)."""
    p = np.pad(pens, pad, mode="edge")
    fields = []
    for pen in range(4):
        f = _up(blur((p == pen).astype(float), 0.55), px)
        fields.append(np.clip(f, 0, 1) ** sharpness)
    w = np.stack(fields, axis=-1)
    w /= np.maximum(w.sum(axis=-1, keepdims=True), 1e-9)
    c = pad * px
    return w[c:c + 8 * px, c:c + 8 * px]


def text_atlas(tiles, px):
    """All 512 text codes, 32 per row: (16 * 8px, 32 * 8px, 4) uint8."""
    t = 8 * px
    out = np.zeros((16 * t, 32 * t, 4), np.uint8)
    for code in range(512):
        cov = pen_coverage(np.asarray(tiles[code]), px)
        y, x = (code // 32) * t, (code % 32) * t
        out[y:y + t, x:x + t] = np.clip(cov * 255 + 0.5, 0, 255)
    return out


SCORE_TOP = np.array([1.0, 0.98, 0.85])       # white-yellow
SCORE_BOTTOM = np.array([1.0, 0.55, 0.1])     # orange


def score_digits(pens, px, pad=3, visible=(2, 3)):
    """Bonus score sprite: returns straight alpha RGBA covering the pens plus `pad`.
    visible: the pens drawn (the game shows them in color 15, where pen 1 is transparent)."""
    p = np.pad(np.isin(np.asarray(pens), visible).astype(float), pad)
    cover = _up(blur(p, 0.5), px)
    body = np.clip((cover - 0.4) / 0.12, 0, 1)
    outline = np.clip((_up(blur(p, 0.7), px) - 0.2) / 0.12, 0, 1) * 0.8
    glow = np.clip(blur(body, 1.8 * px) * 1.2, 0, 1)
    ys = np.nonzero(body.max(axis=1) > 0.5)[0]
    t = np.clip((np.arange(body.shape[0]) - ys.min()) / max(1, ys.max() - ys.min()), 0, 1)[:, None, None]
    grad = SCORE_TOP * (1 - t) + SCORE_BOTTOM * t
    # a little bevel: lighter on the top left edge
    gy, gx = np.gradient(blur(body, 0.6 * px))
    bevel = np.clip(1 + 6.0 * (-gx - gy), 0.7, 1.4)[..., None]
    rgb = np.clip(grad * bevel, 0, 1)
    dark = np.array([0.25, 0.05, 0.0])
    a_out = np.maximum(outline, glow * 0.5)
    col = rgb * body[..., None] + dark * ((outline - body).clip(0) )[..., None] \
        + np.array([1.0, 0.6, 0.2]) * ((1 - outline) * glow * 0.5)[..., None]
    alpha = np.maximum(body, a_out)
    return np.concatenate([col / np.maximum(alpha, 1e-6)[..., None], alpha[..., None]], axis=-1)
