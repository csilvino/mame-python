"""
Remake rocks: the original silhouette inflated into a 3D relief (height field) with faceted
noise, lit like the pre-rendered models (key light from the top left, cold rim light, ambient
occlusion in the cracks). The silhouette stays the arcade one, so collisions look right.
"""

import numpy as np

from fire import blur, fbm, upscale
from sdf import KEY_COLOR, RIM_COLOR, linear_to_srgb, srgb_to_linear

ROCK = srgb_to_linear((132, 130, 127))      # moon rock, grey like the ground
DARK = srgb_to_linear((72, 71, 70))         # mineral spots (pen 3 of the original)
LIGHT = np.array([-0.45, -0.75, 0.5])       # screen coordinates: x right, y down, z to the viewer
LIGHT = LIGHT / np.linalg.norm(LIGHT)
PAD = 2


def inside_distance(mask):
    """Distance (pixels) from each inside pixel to the outside, brute force (small images)."""
    ys, xs = np.nonzero(~mask)
    out = np.zeros(mask.shape)
    iy, ix = np.nonzero(mask)
    for k in range(0, len(iy), 2048):
        dy = iy[k:k + 2048, None] - ys[None, :]
        dx = ix[k:k + 2048, None] - xs[None, :]
        out[iy[k:k + 2048], ix[k:k + 2048]] = np.sqrt((dy * dy + dx * dx).min(axis=1))
    return out


def paint(pens, px, seed=0, roundness=3.0):
    """pens: (h, w) original pens (0 = empty, 3 = dark spots). Returns RGBA (straight alpha)."""
    pens = np.pad(pens, PAD)
    rng = np.random.default_rng(seed)
    cover = upscale(blur((pens != 0).astype(float), 0.55), px)
    alpha = np.clip((cover - 0.42) / 0.16, 0, 1)
    H, W = alpha.shape

    # height: rounded profile from the edge distance + facets
    dist = inside_distance(cover > 0.5) / px                     # native pixels
    r = roundness
    t = np.clip(dist / r, 0, 1)
    height = r * np.sqrt(1 - (1 - t) ** 2)
    facets = 1 - np.abs(2 * fbm(H, W, 3 * px, rng) - 1)          # ridged noise
    height += 0.9 * facets * t + 0.25 * fbm(H, W, px, rng) * t
    height = blur(height, 0.25 * px)

    gy, gx = np.gradient(height * px)                            # slope per native pixel
    n = np.stack([-gx, -gy, np.ones_like(height)], axis=-1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    ndl = np.clip(n @ LIGHT, 0, 1)
    half = LIGHT + np.array([0, 0, 1.0])
    half /= np.linalg.norm(half)
    spec = np.clip(n @ half, 0, 1) ** 24 * 0.25
    rim = (1 - np.clip(n[..., 2], 0, 1)) ** 3 * 0.35
    ao = np.clip(1 - 0.6 * np.clip(blur(height, 1.2 * px) - height, 0, None), 0.4, 1)

    spots = upscale(blur((pens == 3).astype(float), 0.6), px)
    spots *= fbm(H, W, 2 * px, rng) * 1.5
    albedo = ROCK * (1 - np.clip(spots, 0, 1)[..., None]) + DARK * np.clip(spots, 0, 1)[..., None]
    albedo *= (0.85 + 0.3 * fbm(H, W, 2 * px, rng))[..., None]
    col = albedo * (KEY_COLOR * ndl[..., None] * 1.1 + 0.13) * ao[..., None] + spec[..., None] * KEY_COLOR \
        + RIM_COLOR * rim[..., None]
    return np.concatenate([linear_to_srgb(col), alpha[..., None]], axis=-1)
