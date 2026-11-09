"""
Remake UFOs: 3D models (sdf.py), fitted to the original 16x16 sprites.

World units are native pixels: x = sprite x, y = 16 - sprite y (up), z towards the viewer.
They are seen almost from the side (UFO_PITCH), a little from above, like the buggy.
Lights and orbs are placed from the original pens of each frame, so the animations (lights
running around the rim, the orbs turning) stay those of the arcade.
  42      small saucer: yellow hull (pen 1), glass dome (pen 2) with red windows, red thrusters
  43, 44  three blue orbs (pen 2) with red cores, joined by red energy links (pen 3): 2 frames
  45-47   big flat saucer (pen 1), red lights (pen 3) running along the rim: 3 frames
"""

import numpy as np

from sdf import Material, Scene, capsule, ellipsoid, round_box, sphere, srgb_to_linear, subtract

UFO_PITCH = 6.0
HULL, DOME, REDLAMP, ORB, LINK, WHITELAMP, HULL_DARK = range(20, 27)
MATERIALS = {
    HULL: Material(srgb_to_linear((240, 220, 60)), specular=0.9, shininess=70, metal=0.15, rim=0.35, emissive=0.1),
    HULL_DARK: Material(srgb_to_linear((120, 105, 35)), specular=0.5, shininess=30, metal=0.3, rim=0.2),
    DOME: Material(srgb_to_linear((70, 160, 230)), specular=1.3, shininess=150, metal=0.25, rim=0.6, emissive=0.2),
    REDLAMP: Material(srgb_to_linear((255, 40, 20)), specular=0.8, shininess=80, emissive=1.4, rim=0.0),
    ORB: Material(srgb_to_linear((60, 140, 220)), specular=1.2, shininess=110, metal=0.45, rim=0.5, emissive=0.08),
    LINK: Material(srgb_to_linear((255, 50, 30)), specular=0.3, shininess=20, emissive=1.6, rim=0.0),
    WHITELAMP: Material(srgb_to_linear((255, 240, 190)), specular=0.5, shininess=40, emissive=1.5, rim=0.0),
}


def front(center, radii, x, v):
    """Point on the front of an ellipsoid that is seen at screen (x, v)."""
    s, c = np.sin(np.radians(UFO_PITCH)), np.cos(np.radians(UFO_PITCH))
    cx, cy, cz = center
    rx, ry, rz = radii
    y, z = v, 0.0
    for _ in range(4):                    # screen v = y cos - z sin
        y = (v + z * s) / c
        z = cz + rz * np.sqrt(max(0.0, 1 - ((x - cx) / rx) ** 2 - ((y - cy) / ry) ** 2))
    return np.array([x, y, z])


def pixels(pens, pen):
    """Screen coordinates (x, v) of the centers of the pixels of a pen."""
    ys, xs = np.nonzero(pens == pen)
    return [(x + 0.5, 16.0 - (y + 0.5)) for y, x in zip(ys, xs)]


def components(mask):
    """8-connected components of a small mask, as lists of (y, x)."""
    seen = np.zeros(mask.shape, bool)
    out = []
    for y0, x0 in zip(*np.nonzero(mask)):
        if seen[y0, x0]:
            continue
        stack, comp = [(y0, x0)], []
        seen[y0, x0] = True
        while stack:
            y, x = stack.pop()
            comp.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < mask.shape[0] and 0 <= xx < mask.shape[1] and mask[yy, xx] and not seen[yy, xx]:
                        seen[yy, xx] = True
                        stack.append((yy, xx))
        out.append(comp)
    return out


def saucer_model(pens):
    """42: small saucer with a glass dome."""
    pens = np.asarray(pens)
    hull_c, hull_r = (7.5, 6.4, 0.0), (7.6, 1.45, 7.6)
    dome_c, dome_r = (7.5, 7.6, 0.0), (3.4, 4.1, 3.4)
    windows = [front(dome_c, dome_r, x, v) for x, v in pixels(pens, 3) if v > 7.5]
    thrusters = [(x, v) for x, v in pixels(pens, 3) if v <= 7.5]

    def model(p):
        s = Scene(len(p))
        hull = ellipsoid(p, np.array(hull_c), np.array(hull_r))
        hull = subtract(hull, sphere(p, (7.5, 4.3, 0.0), 1.9))          # the opening underneath
        s.add(hull, HULL)
        dome = ellipsoid(p, np.array(dome_c), np.array(dome_r))
        s.add(subtract(dome, round_box(p, (7.5, 3.0, 0.0), (6.0, 3.6, 6.0))), DOME)     # cut below the hull
        for w in windows:
            s.add(sphere(p, w, 0.55), REDLAMP)
        # thrusters: one pod per group of red pixels under the hull
        for side in (thrusters[: len(thrusters) // 2], thrusters[len(thrusters) // 2:]):
            if side:
                x = np.mean([t[0] for t in side])
                s.add(ellipsoid(p, np.array([x, 4.9, 0.8]), np.array([1.5, 0.75, 1.5])), REDLAMP)
        # white lamps around the rim
        for x in (1.8, 4.6, 10.4, 13.2):
            s.add(sphere(p, front(hull_c, hull_r, x, 6.4 - 0.6), 0.45), WHITELAMP)
        return s.d, s.m
    return model


def orbs_model(pens):
    """43, 44: three orbs joined by energy links."""
    pens = np.asarray(pens)
    orbs = []
    for comp in components(pens == 2):
        if len(comp) < 6:
            continue
        ys, xs = np.array(comp).T
        # the red core pixel inside the orb (if any) belongs to it too
        area = len(comp) + sum(1 for y, x in zip(*np.nonzero(pens == 3))
                               if ys.min() < y < ys.max() and xs.min() < x < xs.max())
        cx, cv = xs.mean() + 0.5, 16.0 - (ys.mean() + 0.5)
        r = np.sqrt(area / np.pi) + 0.1
        orbs.append((np.array([cx, cv, 0.0]), r))

    def model(p):
        s = Scene(len(p))
        for c, r in orbs:
            s.add(sphere(p, c, r), ORB)
            s.add(sphere(p, c + np.array([0.0, 0.0, r - 0.35]), 0.9), REDLAMP)
        for i in range(len(orbs)):
            for j in range(i + 1, len(orbs)):
                s.add(capsule(p, orbs[i][0], orbs[j][0], 0.42), LINK)
        return s.d, s.m
    return model


def flat_saucer_model(pens):
    """45-47: big flat saucer with running lights."""
    pens = np.asarray(pens)
    top_c, top_r = (8.0, 7.9, 0.0), (8.0, 3.1, 8.0)
    lights = [front(top_c, top_r, x, v) for x, v in pixels(pens, 3)]

    def model(p):
        s = Scene(len(p))
        s.add(ellipsoid(p, np.array(top_c), np.array(top_r)), HULL)
        s.add(ellipsoid(p, np.array([8.0, 9.6, 0.0]), np.array([2.6, 2.0, 2.6])), HULL)       # top hub
        s.add(ellipsoid(p, np.array([8.0, 6.0, 0.0]), np.array([3.2, 1.4, 3.2])), HULL_DARK)  # underside
        for q in lights:
            s.add(sphere(p, q, 0.62), REDLAMP)
        return s.d, s.m
    return model


# code -> model factory (the argument is the code's 16x16 pens)
MODELS = {0x42: saucer_model, 0x43: orbs_model, 0x44: orbs_model,
          0x45: flat_saucer_model, 0x46: flat_saucer_model, 0x47: flat_saucer_model}
