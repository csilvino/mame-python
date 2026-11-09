"""
Remake models of the later stages' ground enemies (sdf.py).

  3D  the mine: a flat metal dome with a lamp; the game flashes it by changing its color
      (10: grey, 11: red), so it has two images, the lamp off and on (remake "3D/0B")
World units are native pixels: x = sprite x, y = 16 - sprite y (up); seen like the buggy.
"""

import numpy as np

from sdf import Material, Scene, cylinder, ellipsoid, round_box, sphere, srgb_to_linear, subtract

MINE_METAL, MINE_DARK, MINE_LAMP_OFF, MINE_LAMP_ON = range(40, 44)
MATERIALS = {
    MINE_METAL: Material(srgb_to_linear((170, 175, 180)), specular=0.9, shininess=60, metal=0.6, rim=0.3),
    MINE_DARK: Material(srgb_to_linear((40, 42, 50)), specular=0.4, shininess=30, metal=0.4, rim=0.2),
    MINE_LAMP_OFF: Material(srgb_to_linear((90, 20, 20)), specular=0.8, shininess=80, rim=0.1),
    MINE_LAMP_ON: Material(srgb_to_linear((255, 40, 20)), specular=0.8, shininess=80, emissive=1.8, rim=0.0),
}
# the mine covers x 0..6, sprite y 0..3 (its base on the ground at sprite y 3)
MINE_BASE = 13.0


def mine_model(lit):
    def model(p):
        s = Scene(len(p))
        dome = ellipsoid(p, np.array([3.0, MINE_BASE, 0.0]), np.array([3.0, 2.1, 3.0]))
        dome = subtract(dome, round_box(p, (3.0, MINE_BASE - 3.0, 0.0), (5.0, 3.0, 5.0)))
        s.add(dome, MINE_METAL)
        s.add(cylinder(p, (3.0, MINE_BASE + 0.25, 0.0), 1, 3.1, 0.3, rounding=0.15), MINE_DARK)    # rim
        s.add(sphere(p, (3.0, MINE_BASE + 2.0, 0.6), 0.75), MINE_LAMP_ON if lit else MINE_LAMP_OFF)
        return s.d, s.m
    return model
