"""
Remake art for the craters of the text layer (lunar ground).

The craters are groups of text tiles; the whole group is painted at once and cut back into
tiles, each one replacing its tile code. The art is a color multiplier of the soil color (pen 1
of the tile's color), so it follows the palette, and the shader still adds the ground texture:
  rgb = multiplier / 2, alpha = coverage.
The silhouette is the original one with its stairs smoothed (the game's collisions stay the
same); the walls get rock layers, lit from the top left, darker towards the bottom.
"""

import numpy as np

from fire import blur, fbm, upscale

EMPTY = None      # a tile that is not replaced (blank or plain soil), only context for painting

# Groups as they appear in the tilemap (from row 27); 0x00 = blank, 0xF3 = plain soil
GROUPS = [
    [[0xA2, 0x00, 0x00, 0xA8], [0xA3, 0xA4, 0xA6, 0xA9], [0xF3, 0xA5, 0xA7, 0xF3]],    # big crater
    [[0xAA, 0xAB, 0x00], [0xF3, 0xAC, 0xAE], [0xF3, 0xAD, 0xAF]],                    # small crater
    # made by a bomb: the small crater, wider on the right (only B0/B1 are taken from it)
    [[0xAA, 0xAB, 0x00, 0xB0], [0xF3, 0xAC, 0xAE, 0xB1], [0xF3, 0xAD, 0xAF, 0xF3]],
    # bomb craters of the later sections (rows 27-31): deep (B2-BF) and wide (C0-CC)
    [[0xF2, 0xB2, 0x00, 0x00, 0x00, 0x00, 0xBE],
     [0xF3, 0xB3, 0xB5, 0x00, 0x00, 0xBB, 0xBF],
     [0xF3, 0xB4, 0xB6, 0x00, 0x00, 0xBC, 0xF3],
     [0xF3, 0xF3, 0xB7, 0xB8, 0xBA, 0xBD, 0xF3],
     [0xF3, 0xF3, 0xF3, 0xB9, 0xF3, 0xF3, 0xF3]],
    [[0xF2, 0xC0, 0x00, 0x00, 0x00, 0x00, 0xCC],
     [0xF3, 0xC1, 0xC2, 0x00, 0x00, 0xC9, 0xF3],
     [0xF3, 0xF3, 0xC3, 0x00, 0x00, 0xCA, 0xF3],
     [0xF3, 0xF3, 0xC4, 0xC5, 0xC7, 0xCB, 0xF3],
     [0xF3, 0xF3, 0xF3, 0xC6, 0xC8, 0xF3, 0xF3]],
    # the pit of the carnivorous plant (ROM table at 0x30A2: DA-DC, 3 columns 00/00/F0, DD-DF)
    [[0xF2, 0xDA, 0x00, 0x00, 0x00, 0xDD, 0xF2],
     [0xF3, 0xDB, 0x00, 0x00, 0x00, 0xDE, 0xF3],
     [0xF3, 0xDC, 0xF0, 0xF0, 0xF0, 0xDF, 0xF3]],
]
KEEP = {0x00, 0xF0, 0xF2, 0xF3}      # F0 is also a ground surface tile
SURFACE = 3       # native rows of air above the plain ground surface, in the top tile row
PAD = 4
LIGHT = np.array([-0.6, -0.8])            # towards the light, screen coordinates (y down)


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def paint_group(tiles, rows, px, seed):
    """Returns {code: RGBA (8*px, 8*px, 4)} for the tiles of one group."""
    pens = np.block([[tiles[c] for c in row] for row in rows]).astype(np.int32)
    h, w = pens.shape
    solid = pens != 0
    p = PAD
    solid_p = np.pad(solid, ((p, p), (p, p)), mode="edge")
    solid_p[:p] = False
    yy = np.arange(h + 2 * p)[:, None] - p
    hole = ~solid_p & (yy >= SURFACE)                   # inside the crater (not the sky above)
    air = ~solid_p

    rng = np.random.default_rng(seed)
    # smooth silhouette
    cover = upscale(blur(solid_p.astype(float), 0.65), px)
    alpha = smoothstep(0.42, 0.58, cover)
    H, W = alpha.shape

    # surface normal from the air field (points to the air), wall weight from the crater field
    a = upscale(blur(air.astype(float), 1.1), px)
    gy, gx = np.gradient(a)
    g = np.hypot(gx, gy) + 1e-9
    lit = (gx * LIGHT[0] + gy * LIGHT[1]) / g           # 1 facing the light, -1 away
    edge = smoothstep(0.02, 0.5, a)                      # 1 at the edge, 0 deep in the soil
    wall = smoothstep(0.02, 0.22, upscale(blur(hole.astype(float), 2.4), px))

    y = (np.arange(H)[:, None] / px - p) / h             # 0 top .. 1 bottom of the group
    n1 = fbm(H, W, 3 * px, rng)
    strata = 0.5 + 0.5 * np.sin((y * h * 1.3 + 2.5 * fbm(H, W, 4 * px, rng)) * 2.2)

    mult = np.ones((H, W, 3))
    # walls: darker, browner rock with layers, more shadow deeper down
    rock = np.array([0.62, 0.66, 0.8]) * (0.8 + 0.25 * strata[..., None] + 0.2 * (n1[..., None] - 0.5))
    rock *= (1.0 - 0.45 * np.clip(y, 0, 1) ** 1.5)[..., None]
    mult = mult * (1 - wall[..., None]) + rock * wall[..., None]
    # lighting at the edges (walls and surface): lit side brighter, the other darker
    shade = 1.0 + edge * np.where(lit > 0, 0.35 * lit, 0.3 * lit)
    mult *= shade[..., None]
    # thin bright rim where the surface meets the light
    rim = smoothstep(0.35, 0.5, a) * (1 - smoothstep(0.5, 0.7, a)) * np.clip(lit, 0, 1)
    mult += rim[..., None] * 0.35
    # pebbles on the walls
    peb = (fbm(H, W, px, rng, octaves=2) > 0.72) & (wall > 0.5)
    mult[peb] *= 0.7

    # fade back to plain soil before the tiles that are not replaced (they keep their look)
    keep = np.zeros((h, w))
    for r, row in enumerate(rows):
        for c, code in enumerate(row):
            if code in KEEP:
                keep[r * 8:(r + 1) * 8, c * 8:(c + 1) * 8] = 1
    keep = np.pad(keep, p, mode="constant", constant_values=1)
    fade = np.clip(1 - 2.2 * upscale(blur(keep, 1.2), px), 0, 1)[..., None]
    mult = 1 + (mult - 1) * fade

    out = np.zeros((H, W, 4))
    out[..., :3] = np.clip(mult / 2, 0, 1)
    out[..., 3] = alpha
    result = {}
    for r, row in enumerate(rows):
        for c, code in enumerate(row):
            if code in KEEP:
                continue
            y0, x0 = (p + r * 8) * px, (p + c * 8) * px
            result[code] = out[y0:y0 + 8 * px, x0:x0 + 8 * px]
    return result


def paint_all(tiles, px):
    out = {}
    for i, rows in enumerate(GROUPS):
        for code, img in paint_group(tiles, rows, px, seed=100 + i).items():
            out.setdefault(code, img)          # a tile shared by two groups: the first one wins
    return out
