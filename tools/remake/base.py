"""
Remake of the moon base the buggy leaves at the start of each stage (text layer tiles 68-91):
a 3D model (sdf.py) of the dome with its antennas and portholes, the launch platform with its
lights and the ramp down to the ground.

The base is drawn by the game as text tiles in rows 24-28 (it scrolls with the ground); the
render covers the whole base and is cut back into those tiles. The tiles are full color art
(remake.json text "mode": "color"): the shader keeps the soil pixels of the original tiles under
the art, so the ground around the base stays the game's.

Local coordinates: x = native pixels from tile column 2, ground surface at local y 27; world
units are native pixels, y up from the ground, seen like the buggy (PITCH).
"""

import numpy as np

from sdf import Material, Scene, capsule, cylinder, ellipsoid, round_box, rotate, sphere, srgb_to_linear, subtract

# tile layout of the base: row -> {column: code}
LAYOUT = {
    24: {4: 0x68},
    25: {4: 0x69, 5: 0x6A, 6: 0x6B},
    26: {**{3 + i: 0x6C + i for i in range(5)}, **{8 + i: 0x71 + i for i in range(11)}},
    27: {3 + i: 0x7C + i for i in range(17)},
    28: {9 + i: 0x8D + i for i in range(5)},
}
COL0, ROW0, GROUND = 2, 24, 27.0
X_RANGE, V_RANGE = (0.0, 144.0), (-13.0, 27.0)      # local x 0..144, local y = 27 - v

PANEL, PANEL_DARK, BLUE_PANEL, STRIPE, PORTHOLE, LAMP, BEACON, METAL = range(50, 58)
MATERIALS = {
    PANEL: Material(srgb_to_linear((205, 232, 212)), specular=0.8, shininess=60, metal=0.15, rim=0.3),
    PANEL_DARK: Material(srgb_to_linear((95, 104, 112)), specular=0.5, shininess=40, metal=0.4, rim=0.2),
    BLUE_PANEL: Material(srgb_to_linear((40, 150, 220)), specular=0.9, shininess=80, metal=0.3, rim=0.3),
    STRIPE: Material(srgb_to_linear((230, 30, 25)), specular=0.6, shininess=50, emissive=0.25, rim=0.2),
    PORTHOLE: Material(srgb_to_linear((150, 230, 255)), specular=1.2, shininess=120, emissive=0.9, rim=0.4),
    LAMP: Material(srgb_to_linear((255, 235, 170)), specular=0.5, shininess=40, emissive=1.5, rim=0.0),
    BEACON: Material(srgb_to_linear((255, 40, 30)), specular=0.5, shininess=40, emissive=1.6, rim=0.0),
    METAL: Material(srgb_to_linear((70, 74, 84)), specular=0.7, shininess=60, metal=0.6, rim=0.25),
}

DOME_C, DOME_R = np.array([27.0, 0.0, 0.0]), np.array([17.0, 15.0, 9.0])
PAD_X0, PAD_X1, PAD_TOP, PAD_Z = 46.0, 97.0, 7.0, 4.5


def on_dome(x, y):
    """Point on the front of the dome at (x, y)."""
    q = 1 - ((x - DOME_C[0]) / DOME_R[0]) ** 2 - (y / DOME_R[1]) ** 2
    return np.array([x, y, DOME_R[2] * np.sqrt(max(q, 0.0))])


def base_model(p):
    s = Scene(len(p))
    # dome, cut at the ground, with a darker ring at its foot
    dome = ellipsoid(p, DOME_C, DOME_R)
    dome = subtract(dome, round_box(p, (27.0, -10.0, 0.0), (30.0, 10.0, 20.0)))
    s.add(dome, PANEL)
    s.add(cylinder(p, (27.0, 0.6, 0.0), 1, 17.2, 0.7, rounding=0.3), PANEL_DARK)
    # portholes and the ring of small lamps (the window and dot rows of the original)
    for x in (24.5, 30.0, 35.0):
        s.add(sphere(p, on_dome(x, 9.0), 1.25), PORTHOLE)
    for x in np.arange(12.0, 42.0, 3.0):
        s.add(sphere(p, on_dome(x, 4.5), 0.45), LAMP)
    # antennas (on the left of the dome top), red beacon on the tall one
    s.add(capsule(p, np.array([17.0, 9.0, 0.0]), (17.0, 26.0, 0.0), 0.35), METAL)
    s.add(sphere(p, (17.0, 26.2, 0.0), 0.8), BEACON)
    s.add(capsule(p, np.array([19.5, 10.0, 0.0]), (19.5, 18.0, 0.0), 0.3), METAL)
    s.add(cylinder(p, (19.5, 18.2, 0.0), 1, 1.3, 0.2, rounding=0.1), METAL)               # small dish
    # launch platform: deck, red edge, lamps along the front, blue door block on the right
    cx, hx = (PAD_X0 + PAD_X1) / 2, (PAD_X1 - PAD_X0) / 2
    s.add(round_box(p, (cx, PAD_TOP / 2, 0.0), (hx, PAD_TOP / 2, PAD_Z), 0.6), PANEL)
    s.add(round_box(p, (cx - 7.0, PAD_TOP - 0.35, PAD_Z + 0.05), (hx - 7.0, 0.35, 0.3), 0.15), STRIPE)
    s.add(round_box(p, (89.5, PAD_TOP / 2 + 0.2, 0.0), (7.5, PAD_TOP / 2 + 0.2, PAD_Z + 0.2), 0.6), BLUE_PANEL)
    for x in np.arange(49.0, 82.0, 4.0):
        s.add(round_box(p, (x, 3.6, PAD_Z + 0.1), (0.9, 0.45, 0.3), 0.15), LAMP)
    # hull going down into the pit under the platform (the tapering green part of the original)
    hull = ellipsoid(p, np.array([75.0, 1.0, 0.0]), np.array([21.0, 9.0, PAD_Z - 0.6]))
    s.add(subtract(hull, round_box(p, (75.0, 10.0, 0.0), (30.0, 9.0, 10.0))), PANEL_DARK)
    for x in np.arange(60.0, 92.0, 6.0):
        s.add(round_box(p, (x, -3.0, PAD_Z - 1.2), (0.5, 3.0, 1.0), 0.2), METAL)            # ribs
    # ramp down to the right, on two struts
    a = -np.arctan2(6.0, 40.0)
    q = rotate(p, 2, a, center=(117.0, 3.3, 0.0))
    s.add(round_box(q, (117.0, 3.3, 0.0), (20.5, 0.7, PAD_Z - 0.5), 0.35), PANEL)
    for x, top in ((106.0, 4.3), (124.0, 1.6)):
        s.add(capsule(p, np.array([x, 0.0, 1.5]), (x, top, 1.5), 0.4), METAL)
        s.add(capsule(p, np.array([x, 0.0, -1.5]), (x, top, -1.5), 0.4), METAL)
    return s.d, s.m


def cut_tiles(rgba, px):
    """The base render (local coordinates) cut into the tiles of LAYOUT: {code: (8px, 8px, 4)}."""
    out = {}
    for row, cols in LAYOUT.items():
        for col, code in cols.items():
            y, x = (row - ROW0) * 8 * px, (col - COL0) * 8 * px
            out[code] = rgba[y:y + 8 * px, x:x + 8 * px]
    return out
