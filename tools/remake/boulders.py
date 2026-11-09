"""
Remake rolling boulders: lumpy rock spheres (3D noise displacement), rendered by sdf.py.

  31      small boulder (one frame)
  32-34   big boulder rolling: the original moves a mark around clockwise, the model turns by a
          third of a turn per frame (the 3 frames loop)
  36, 37  giant boulder (later stages), 2 frames: half a turn each
World units are native pixels: x = sprite x, v = 16 - sprite y; seen like the buggy (PITCH).
"""

import numpy as np

from sdf import Material, srgb_to_linear

BOULDER, BOULDER_DARK = 30, 31
MATERIALS = {
    BOULDER: Material(srgb_to_linear((132, 130, 127)), specular=0.25, shininess=18, rim=0.15),
    BOULDER_DARK: Material(srgb_to_linear((74, 73, 72)), specular=0.15, shininess=12, rim=0.1),
}
# code -> (center x, center v on screen, radius, frame of the rolling animation)
BOULDERS = {0x31: (8.0, 8.0, 3.4, 0), 0x32: (7.5, 7.5, 5.3, 0), 0x33: (7.5, 7.5, 5.3, 1), 0x34: (7.5, 7.5, 5.3, 2),
            0x36: (7.5, 8.0, 6.7, 0), 0x37: (7.5, 8.0, 6.7, 1.5)}


def _hash(i):
    """Pseudo random 0..1 for integer lattice points (n, 3)."""
    h = (i[:, 0] * 73856093) ^ (i[:, 1] * 19349663) ^ (i[:, 2] * 83492791)
    h = (h ^ (h >> 13)) * 1274126177
    return ((h ^ (h >> 16)) & 0xFFFF) / 65535.0


def noise3(p):
    """Value noise in 3D, p: (n, 3)."""
    i = np.floor(p).astype(np.int64)
    f = p - i
    f = f * f * (3 - 2 * f)
    out = 0.0
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = (f[:, 0] if dx else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1]) * (f[:, 2] if dz else 1 - f[:, 2])
                out = out + w * _hash(i + np.array([dx, dy, dz]))
    return out


def fbm3(p, octaves=3):
    s, a, total = 0.0, 1.0, 0.0
    for _ in range(octaves):
        s = s + a * noise3(p)
        total += a
        p = p * 2.07 + 5.3
        a *= 0.5
    return s / total


def boulder_model(cx, cy, r, frame, seed=0.0):
    """cy: world y of the center. Turns clockwise (as seen) by 120 degrees per frame."""
    a = -frame * 2 * np.pi / 3
    ca, sa = np.cos(a), np.sin(a)
    scale = 1.3 / r                                   # lumps: about the same count on both sizes

    def model(p):
        q = p - np.array([cx, cy, 0.0])
        # into the rock's own frame (rotation around the z axis, the rolling axis)
        x = q[:, 0] * ca - q[:, 1] * sa
        y = q[:, 0] * sa + q[:, 1] * ca
        local = np.stack([x, y, q[:, 2]], axis=1) * scale + seed
        bumps = fbm3(local) - 0.5
        ridges = 1 - np.abs(2 * noise3(local * 2.3 + 11.0) - 1)
        d = np.linalg.norm(q, axis=1) - r * (1 + 0.2 * bumps - 0.06 * ridges)
        dark = fbm3(local * 1.1 + 31.0, octaves=2) > 0.66
        m = np.where(dark, BOULDER_DARK, BOULDER)
        return d * 0.8, m                              # displaced: shorter steps
    return model
