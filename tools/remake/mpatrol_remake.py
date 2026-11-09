"""
Moon Patrol remake graphics: pre-rendered 2.5D replacements for the game's sprites.

    python tools/remake/mpatrol_remake.py          -> hdpack/mpatrol/remake.json + remake_sprites.png

Each replacement is attached to a sprite code: whenever the game shows that sprite, the image is
drawn instead, at the sprite's position plus an offset (in native pixels), with its own size.
A code can also be hidden, when one replacement covers a group of sprites (the buggy body is 4).

Buggy layout in the original (native pixels, relative to sprite 01):
  01 (0,0)  02 (16,0)          body: x 0..31, y 10..24, rear gun up to y 5
  03 (0,16) 04 (16,16)
  wheels: sprites 05/06 (rear, 2 rotation frames) and 07/08 (front, smaller) at their own
  positions; their centers are at sprite + (4, 4.5) and sprite + (3, 5). They bounce on the
  terrain independently, so they stay separate sprites.
Other groups (2x2 or 3x2 sprites, the art is attached to the top left one, the others hidden):
  09-0C buggy nose up (climbing out of a crater), 0D-10 nose down, wheels included; their angle
        and position were fitted to the original silhouettes (see TILTS)
  11-16 burning wreck, 17-1C, 1D-22, 23-24, 25-26, 27: the explosion, from big to gone
  28, 29 forward shot, 2A-2C its impact. Fire is painted by fire.py from the original pens.
  2D-30 rocks (rocks.py), 3E-41 shot hitting a rock, 61-6F the rock blowing up in dust,
  4E-60 a bomb hitting the ground.
  42-47 the UFOs (3D models from ufos.py, fitted to each frame), 48-4A a UFO blowing up, 4B-4D
  their bombs falling (plasma), 35 the bomb of the orbs, 7A a bomb shot down (green sparks).
  31-34 rolling boulders (boulders.py: lumpy 3D spheres, the big one turning frame by frame).
  7D "500", 7E "1000" bonus scores (hud.py).
  70-79 the carnivorous plant of the champion course (relief, see PLANT_*).
  Later stages: 36, 37 giant boulder (boulders.py), 3D mine (ground.py, lamp on in its flash
  color), 38 tank and 3A/7B/7C + 3B low flying craft (relief.py, one image per color: entries
  "code/color"), 39, 3C shots (fire).
Text layer tiles (remake.json "text"): the craters, painted by craters.py.
Backgrounds (remake.json "backgrounds"): mountains, hills and city painted again by scenery.py.
Score panel and messages (remake.json "text_pens"): smoothed pen coverage of every text tile (hud.py).
3D renders are cached in tools/remake/cache (the key covers the models and the renderer).
"""

import hashlib
import inspect
import json
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import base  # noqa: E402
import boulders  # noqa: E402
import craters  # noqa: E402
import hud  # noqa: E402
import relief  # noqa: E402
import fire  # noqa: E402
import ground  # noqa: E402
import rocks  # noqa: E402
import scenery  # noqa: E402
import sdf  # noqa: E402
import ufos  # noqa: E402
from sdf import (Material, Scene, capsule, cylinder, ellipsoid, round_box, rotate, smooth_union,  # noqa: E402
                 sphere, subtract, srgb_to_linear)

PX = 8            # output pixels per native pixel (same as the HD pack)
PITCH = 18.0      # camera looks down by this angle: we see a little of the roofs

# Materials
PAINT, GLASS, GUNMETAL, TIRE, HUB, LAMP, STRIPE, CHASSIS = range(1, 9)
MATERIALS = {
    PAINT: Material(srgb_to_linear((210, 22, 28)), specular=0.6, shininess=50, metal=0.25, rim=0.3),
    GLASS: Material(srgb_to_linear((60, 210, 240)), specular=1.2, shininess=150, metal=0.25, rim=0.6, emissive=0.15),
    GUNMETAL: Material(srgb_to_linear((60, 64, 75)), specular=0.7, shininess=60, metal=0.6, rim=0.25),
    TIRE: Material(srgb_to_linear((28, 28, 34)), specular=0.15, shininess=12, rim=0.2),
    HUB: Material(srgb_to_linear((40, 200, 220)), specular=0.9, shininess=80, metal=0.7, rim=0.3),
    LAMP: Material(srgb_to_linear((255, 230, 120)), specular=1.0, shininess=100, emissive=1.2, rim=0.0),
    STRIPE: Material(srgb_to_linear((40, 220, 240)), specular=0.5, shininess=40, emissive=0.35, rim=0.2),
    CHASSIS: Material(srgb_to_linear((72, 18, 20)), specular=0.3, shininess=20, rim=0.15),
}
BUGGY_MATERIALS = dict(MATERIALS)
MATERIALS.update(ufos.MATERIALS)
MATERIALS.update(boulders.MATERIALS)
MATERIALS.update(ground.MATERIALS)
MATERIALS.update(base.MATERIALS)

# Tilted buggy groups: (top left code, other codes, angle, world (x, v) of the group's top left),
# fitted to the original 32x32 silhouettes (scratch fit: IoU 0.75 for both)
TILTS = [(0x09, (0x0A, 0x0B, 0x0C), 39.0, (-0.5, 22.0)),
         (0x0D, (0x0E, 0x0F, 0x10), -36.0, (0.5, 22.0))]

# Fire groups: (codes as rows of the group, pen -> heat, paint options)
FIRE_HEAT = {3: 1.0, 1: 0.62, 2: 0.3}         # red core, green middle, magenta rim


def cooled(heat, k):
    """The explosion cools down as it fades away."""
    return {pen: h * k for pen, h in heat.items()}


FIRES = [
    ([[0x11, 0x12, 0x13], [0x14, 0x15, 0x16]], {1: 0.7, 2: 0.04}, dict(glow=0.8, roughness=0.6)),   # wreck
    ([[0x17, 0x18, 0x19], [0x1A, 0x1B, 0x1C]], FIRE_HEAT, dict(glow=1.3)),
    ([[0x1D, 0x1E, 0x1F], [0x20, 0x21, 0x22]], FIRE_HEAT, dict(glow=1.0)),
    ([[0x23, 0x24]], cooled(FIRE_HEAT, 0.85), dict(glow=0.8)),
    ([[0x25, 0x26]], cooled(FIRE_HEAT, 0.62), dict(glow=0.5)),
    ([[0x27]], cooled(FIRE_HEAT, 0.5), dict(glow=0.4)),
    ([[0x28]], {3: 1.0, 1: 0.8}, dict(glow=1.5, roughness=0.3)),
    ([[0x29]], {3: 1.0, 1: 0.8}, dict(glow=1.5, roughness=0.3)),
    ([[0x2A]], {1: 0.9}, dict(glow=1.5, roughness=0.4)),
    ([[0x2B]], {3: 1.0, 1: 0.75}, dict(glow=1.5, roughness=0.5)),
    ([[0x2C]], {3: 1.0, 1: 0.7}, dict(glow=1.3, roughness=0.6)),
]
DUST = (0.78, 0.76, 0.72)                     # light grey moon dust
DUST_HEAT = {2: 0.0, 1: 0.45, 3: 0.8}          # grey cloud, dark red and yellow sparks
FIRES += [([[c]], FIRE_HEAT, dict(glow=1.2, roughness=0.5)) for c in range(0x3E, 0x42)]
FIRES += [([[c]], DUST_HEAT, dict(glow=0.6, smoke=DUST, smoke_alpha=0.85)) for c in (0x61, 0x62, 0x63)]
FIRES += [(rows, DUST_HEAT, dict(glow=0.5, smoke=DUST, smoke_alpha=0.85)) for rows in
          ([[0x64, 0x65], [0x66, 0x67]], [[0x68, 0x69], [0x6A, 0x6B]], [[0x6C], [0x6D]], [[0x6E], [0x6F]])]
# bomb from a UFO hitting the ground (it leaves a crater)
FIRES += [([[0x4E]], FIRE_HEAT, dict(glow=1.0))]
FIRES += [(rows, FIRE_HEAT, dict(glow=1.2)) for rows in
          ([[0x4F, 0x50], [0x51, 0x52]], [[0x53, 0x54], [0x55, 0x56]], [[0x57, 0x58], [0x59, 0x5A]],
           [[0x5B, 0x5C], [0x5D, 0x5E], [0x5F, 0x60]])]
# UFOs blowing up (green core, red flames, magenta debris), bombs
UFO_BLAST = {1: 1.0, 3: 0.75, 2: 0.4}
FIRES += [([[c]], UFO_BLAST, dict(glow=1.3, sparks=True)) for c in (0x48, 0x49, 0x4A)]
FIRES += [([[c]], {3: 1.0, 2: 0.6}, dict(glow=1.4, roughness=0.3, stops=fire.PLASMA, glow_rgb=(1.0, 0.3, 0.8)))
          for c in (0x4B, 0x4C, 0x4D)]
FIRES += [([[0x35]], {3: 1.0, 2: 0.8}, dict(glow=1.4, roughness=0.3))]
FIRES += [([[0x7A]], {1: 0.8}, dict(glow=1.0, sparks=True, stops=fire.GREEN, glow_rgb=(0.4, 1.0, 0.3)))]
# later stages: shots (39, 3C)
FIRES += [([[c]], FIRE_HEAT, dict(glow=1.4, roughness=0.3)) for c in (0x39, 0x3C)]
# carnivorous plant in a pit (champion course): head 70 (bud), 73-75 (jaws opening; drawn twice,
# the second one flipped) in color 2; stalk 71/72 or 76/77 and the stump left when shot, 78/79,
# in color 8. Painted in relief with the arcade's hues, organic (no metal).
PLANT_HEADS = (0x70, 0x73, 0x74, 0x75)
PLANT_STALKS = ((0x71, 0x72), (0x76, 0x77), (0x78, 0x79))
PLANT_HEAD_COLORS = {1: (205, 34, 52), 2: (78, 176, 64), 3: (112, 14, 34)}
PLANT_STALK_COLORS = {1: (190, 40, 50), 2: (46, 150, 138), 3: (34, 86, 44)}
FIRE_PAD = 4
# relief art painted once per color (the colors these use in the game are not known)
RELIEFS = [[0x38], [0x3A, 0x3B], [0x7B, 0x3B], [0x7C, 0x3B]]
RELIEF_DEFAULT_COLOR = 12
ROCKS = range(0x2D, 0x31)                      # small to big

# Wheel centers (world x, y, radius); world x = native x relative to sprite 01, y up from the ground
REAR_WHEELS = [(5.0, 4.5, 4.5), (14.0, 4.5, 4.5)]
FRONT_WHEEL = (26.0, 4.0, 3.0)
NEAR_Z = 4.0       # wheels on the visible side
# native y (relative to sprite 01) = Y_ANCHOR - v, chosen so a near rear wheel center lands at 21.5
Y_ANCHOR = 21.5 + sdf.project_v((0.0, 4.5, NEAR_Z), PITCH)


def tire(p, cx, cy, cz, r, angle, treads=10, width=1.7):
    """Tire with tread blocks; angle rotates the treads (animation frames)."""
    q = rotate(p, 2, angle, center=(cx, cy, cz))
    d = cylinder(q, (cx, cy, cz), 2, r, width, rounding=min(0.9, r * 0.25))
    # tread grooves: thin boxes around the rim, cut into the tire
    for k in range(treads):
        a = 2 * np.pi * k / treads
        qq = rotate(q, 2, -a, center=(cx, cy, cz))
        groove = round_box(qq, (cx, cy + r, cz), (r * 0.09, r * 0.16, width + 0.2))
        d = subtract(d, groove)
    return d


def add_wheel(s, p, cx, cy, cz, r, angle):
    s.add(tire(p, cx, cy, cz, r, angle), TIRE)
    s.add(cylinder(p, (cx, cy, cz + 0.6), 2, r * 0.48, 1.4, rounding=0.3), HUB)
    # 5 bolts on the hub, rotating with the tire
    for k in range(5):
        a = angle + 2 * np.pi * k / 5
        s.add(sphere(p, (cx + np.cos(a) * r * 0.28, cy + np.sin(a) * r * 0.28, cz + 2.0), r * 0.07 + 0.08), GUNMETAL)
    s.add(sphere(p, (cx, cy, cz + 2.0), r * 0.12), GUNMETAL)


def wheel_model(r, angle):
    def model(p):
        s = Scene(len(p))
        add_wheel(s, p, 0.0, 0.0, 0.0, r, angle)
        return s.d, s.m
    return model


def body_model(p):
    s = Scene(len(p))
    # hull with wheel wells
    hull = round_box(p, (16.0, 6.9, 0.0), (15.8, 3.6, 4.8), 1.8)
    nose = round_box(rotate(p, 2, -0.35, center=(29.5, 8.0, 0)), (29.5, 7.6, 0.0), (3.2, 2.3, 4.6), 1.2)
    hull = smooth_union(hull, nose, 1.0)
    for x, y, r in REAR_WHEELS + [FRONT_WHEEL]:
        hull = subtract(hull, cylinder(p, (x, y, 0), 2, r + 0.9, 6.0))
    s.add(hull, PAINT)
    # dark chassis under the body
    s.add(round_box(p, (16.0, 3.9, 0.0), (13.0, 0.8, 3.4), 0.5), CHASSIS)
    # cabin (glass bubble) with a frame
    cabin = ellipsoid(p, np.array([18.5, 10.0, 0.0]), np.array([7.2, 4.4, 3.9]))
    s.add(cabin, GLASS)
    # rear anti-aircraft gun (shoots up)
    s.add(sphere(p, (8.5, 10.6, 0.0), 2.0), GUNMETAL)
    s.add(capsule(p, np.array([8.5, 11.0, 0.0]), (8.5, 20.0, 0.0), 0.6), GUNMETAL)
    s.add(cylinder(p, (8.5, 19.6, 0.0), 1, 0.85, 0.5, rounding=0.2), GUNMETAL)
    # front cannon
    s.add(round_box(p, (25.5, 9.8, 1.6), (2.0, 0.9, 1.0), 0.4), GUNMETAL)
    s.add(capsule(p, np.array([26.0, 9.6, 1.6]), (34.0, 9.6, 1.6), 0.75), GUNMETAL)
    # headlight and side stripe
    s.add(sphere(p, (31.4, 7.2, 3.3), 0.9), LAMP)
    s.add(round_box(p, (15.5, 7.3, 4.75), (12.5, 0.35, 0.2), 0.15), STRIPE)
    # wheels on the far side: static, mostly hidden under the body
    for x, y, r in REAR_WHEELS + [FRONT_WHEEL]:
        s.add(tire(p, x, y, -NEAR_Z, r, 0.0), TIRE)
    return s.d, s.m


def tilted_buggy_model(angle_deg, pivot=(16.0, 4.5, 0.0)):
    """Whole buggy (body + near wheels), rotated by angle_deg (nose up > 0) around pivot."""
    a = np.radians(angle_deg)

    def model(p):
        q = rotate(p, 2, a, center=pivot)
        d, m = body_model(q)
        s = Scene(len(p))
        s.d, s.m = d, m
        for x, y, r in REAR_WHEELS + [FRONT_WHEEL]:
            add_wheel(s, q, x, y, NEAR_Z, r, 0.0)
        return s.d, s.m
    return model


def to_image(rgba: np.ndarray) -> Image.Image:
    out = np.zeros(rgba.shape, np.uint8)
    out[..., :3] = np.clip(rgba[..., :3] * 255 + 0.5, 0, 255)
    out[..., 3] = np.clip(rgba[..., 3] * 255 + 0.5, 0, 255)
    return Image.fromarray(out, "RGBA")


def crop_alpha(rgba: np.ndarray, px: int):
    """Crop to the visible part, on whole native pixels. Returns (image, x0, y0) in native pixels."""
    ys, xs = np.nonzero(rgba[..., 3] > 1 / 255)
    y0, x0 = ys.min() // px, xs.min() // px
    y1, x1 = ys.max() // px + 1, xs.max() // px + 1
    return rgba[y0 * px:y1 * px, x0 * px:x1 * px], int(x0), int(y0)


def load_sprites():
    from drivers.mpatrol import MoonPatrol
    driver = MoonPatrol()
    if not driver.load(os.path.join(ROOT, "roms")):
        sys.exit("ROMs not found")
    return driver


CACHE = os.path.join(HERE, "cache")


def render_cached(name, model_factory, *args, **kw):
    """sdf.render(model_factory(*args), ...), cached on disk."""
    # the key only covers what the render depends on (a UFO change does not render the buggy again)
    if name.startswith("ufo"):
        deps, mats = [ufos], ufos.MATERIALS
    elif name.startswith("boulder"):
        deps, mats = [boulders], boulders.MATERIALS
    elif name.startswith("mine"):
        deps, mats = [ground], ground.MATERIALS
    elif name.startswith("base"):
        deps, mats = [base], base.MATERIALS
    else:
        deps, mats = [tire, add_wheel, wheel_model, body_model, tilted_buggy_model], BUGGY_MATERIALS
    sources = [inspect.getsource(sdf)] + [inspect.getsource(f) for f in deps]
    pitch = kw.get("pitch", PITCH)
    materials = repr(sorted((k, vars(m).__repr__()) for k, m in mats.items()))
    key = hashlib.sha1("\n".join(sources + [materials, name, repr(args), repr(kw), repr(PX), repr(pitch)])
                       .encode()).hexdigest()[:16]
    path = os.path.join(CACHE, f"{name}_{key}.npy")
    if os.path.isfile(path):
        return np.load(path)
    print(f"rendering {name}...")
    model = model_factory(*args) if args else model_factory
    rgba = sdf.render(model, MATERIALS, kw["x"], kw["v"], PX, pitch)
    os.makedirs(CACHE, exist_ok=True)
    np.save(path, rgba)
    return rgba


def build(out_dir: str):
    images = []          # (code, PIL image, offset (native), size (native))
    hidden = [0x01, 0x02, 0x04]
    driver = load_sprites()
    tiles = driver.tiles_sprites

    # Buggy body: attached to sprite 03 (at sprite 01 + (0, 16)); 01, 02, 04 are hidden
    x0, x1, v0, v1 = -3.0, 36.0, -1.5, 22.0
    body = to_image(render_cached("body", body_model, x=(x0, x1), v=(v0, v1)))
    images.append((0x03, body, (x0, Y_ANCHOR - v1 - 16.0), (x1 - x0, v1 - v0)))

    # Wheels: 2 rotation frames each; image centered on the original wheel center
    for codes, r, center in (((0x05, 0x06), 4.5, (4.0, 4.5)), ((0x07, 0x08), 3.0, (3.0, 5.0))):
        for frame, code in enumerate(codes):
            angle = frame * np.pi / 10          # half a tread step between frames
            m = r + 1.2
            img = to_image(render_cached(f"wheel{code:02X}", wheel_model, r, angle, x=(-m, m), v=(-m, m)))
            images.append((code, img, (center[0] - m, center[1] - m), (2 * m, 2 * m)))

    # Tilted buggy (wheels included)
    X0, X1, V0, V1 = -8.0, 40.0, -10.0, 34.0
    for code, others, angle, (gx, gv) in TILTS:
        rgba = render_cached(f"tilt{code:02X}", tilted_buggy_model, angle, x=(X0, X1), v=(V0, V1))
        rgba, cx, cy = crop_alpha(rgba, PX)
        h, w = rgba.shape[0] // PX, rgba.shape[1] // PX
        images.append((code, to_image(rgba), (X0 + cx - gx, gv - V1 + cy), (w, h)))
        hidden += others

    # UFOs: world units = native pixels, v = 16 - sprite y
    for code, factory in ufos.MODELS.items():
        rgba = render_cached(f"ufo{code:02X}", factory, tiles[code].tolist(), x=(-2.0, 18.0), v=(-2.0, 18.0),
                             pitch=ufos.UFO_PITCH)
        rgba, cx, cy = crop_alpha(rgba, PX)
        h, w = rgba.shape[0] // PX, rgba.shape[1] // PX
        images.append((code, to_image(rgba), (cx - 2, cy - 2), (w, h)))

    # Rolling boulders: on the ground, seen like the buggy
    cos_p = np.cos(np.radians(PITCH))
    for code, (cx, cv, r, frame) in boulders.BOULDERS.items():
        m = r * 1.3
        rgba = render_cached(f"boulder{code:02X}", boulders.boulder_model, cx, cv / cos_p, r, frame,
                             x=(cx - m, cx + m), v=(cv - m, cv + m))
        rgba, ox, oy = crop_alpha(rgba, PX)
        h, w = rgba.shape[0] // PX, rgba.shape[1] // PX
        images.append((code, to_image(rgba), (cx - m + ox, 16.0 - (cv + m) + oy), (w, h)))

    # Mine: lamp off (its normal color, 10) and on (the flash color, 11)
    for lit, key in ((False, 0x3D), (True, "3D/0B")):
        rgba = render_cached(f"mine{int(lit)}", ground.mine_model, lit, x=(-1.0, 7.0), v=(10.0, 17.0))
        rgba, cx, cy = crop_alpha(rgba, PX)
        h, w = rgba.shape[0] // PX, rgba.shape[1] // PX
        images.append((key, to_image(rgba), (-1.0 + cx, -1.0 + cy), (w, h)))

    # Relief art, one image per color
    pal, transparent = driver.sprite_palette, driver.sprite_transparent
    for row in RELIEFS:
        pens = np.concatenate([tiles[c] for c in row], axis=1)
        code = row[0]
        for color in range(16):
            colors = {p: tuple(int(v) for v in pal[color * 8 + p]) for p in (1, 2, 3) if not transparent[color * 8 + p]}
            if not colors:
                continue
            colors = {p: c if max(c) > 0 else (24, 24, 30) for p, c in colors.items()}    # black: dark metal
            rgba, cx, cy = crop_alpha(relief.paint(pens, colors, PX), PX)
            h, w = rgba.shape[0] // PX, rgba.shape[1] // PX
            entry = (to_image(rgba), (cx - relief.PAD, cy - relief.PAD), (w, h))
            images.append((f"{code:02X}/{color:02X}", *entry))
            if color == RELIEF_DEFAULT_COLOR:
                images.append((code, *entry))
        hidden += [c for c in row[1:] if c not in hidden]

    # Carnivorous plant
    for codes, colors in ([((c,), PLANT_HEAD_COLORS) for c in PLANT_HEADS] +
                          [(pair, PLANT_STALK_COLORS) for pair in PLANT_STALKS]):
        pens = np.concatenate([tiles[c] for c in codes], axis=1)
        rgba, cx, cy = crop_alpha(relief.paint(pens, colors, PX, roundness=2.0, metal=0.0), PX)
        h, w = rgba.shape[0] // PX, rgba.shape[1] // PX
        images.append((codes[0], to_image(rgba), (cx - relief.PAD, cy - relief.PAD), (w, h)))
        hidden += [c for c in codes[1:] if c not in hidden]

    # Bonus scores
    for code in (0x7D, 0x7E):
        rgba, cx, cy = crop_alpha(hud.score_digits(tiles[code], PX), PX)
        h, w = rgba.shape[0] // PX, rgba.shape[1] // PX
        images.append((code, to_image(rgba), (cx - 3, cy - 3), (w, h)))

    # Fire: explosion frames, wreck, shots
    for rows, heat, opts in FIRES:
        pens = np.block([[tiles[c] for c in row] for row in rows])
        code = rows[0][0]
        rgba = fire.paint(pens, heat, PX, pad=FIRE_PAD, seed=code, **opts)
        rgba, cx, cy = crop_alpha(rgba, PX)
        h, w = rgba.shape[0] // PX, rgba.shape[1] // PX
        images.append((code, to_image(rgba), (cx - FIRE_PAD, cy - FIRE_PAD), (w, h)))
        hidden += [c for row in rows for c in row if c != code]

    # Rocks
    for code in ROCKS:
        rgba = rocks.paint(tiles[code], PX, seed=code)
        rgba, cx, cy = crop_alpha(rgba, PX)
        h, w = rgba.shape[0] // PX, rgba.shape[1] // PX
        images.append((code, to_image(rgba), (cx - rocks.PAD, cy - rocks.PAD), (w, h)))

    # Text layer tiles: craters (rgb = soil color multiplier / 2)
    tile_images = [(code, to_image(rgba)) for code, rgba in sorted(craters.paint_all(driver.tiles_text, PX).items())]
    # the moon base at the start of the stages: full color art over the soil of its tiles
    rgba = render_cached("base", base.base_model, x=base.X_RANGE, v=base.V_RANGE)
    color_tiles = base.cut_tiles(rgba, PX)
    tile_images += [(code, to_image(t)) for code, t in sorted(color_tiles.items())]

    # Pack the images in one atlas (simple shelf packing)
    width = 1024
    x = y = shelf = 0
    placed = []
    for kind, code, img, off, size in ([("sprite", *e) for e in images] +
                                       [("text", code, img, None, None) for code, img in tile_images]):
        if x + img.width > width:
            x, y, shelf = 0, y + shelf, 0
        placed.append((kind, code, img, (x, y), off, size))
        x += img.width + 2
        shelf = max(shelf, img.height + 2)
    atlas = Image.new("RGBA", (width, y + shelf), (0, 0, 0, 0))
    entries, text_entries = {}, {}
    for kind, code, img, (ax, ay), off, size in placed:
        atlas.paste(img, (ax, ay))
        if kind == "text":
            text_entries[f"{code:03X}"] = {"rect": [ax, ay, img.width, img.height]}
            if code in color_tiles:
                text_entries[f"{code:03X}"]["mode"] = "color"
            continue
        entries[code if isinstance(code, str) else f"{code:02X}"] = {"rect": [ax, ay, img.width, img.height],
                                  "offset": [round(off[0], 3), round(off[1], 3)],
                                  "size": [round(size[0], 3), round(size[1], 3)]}
    for code in hidden:
        entries[f"{code:02X}"] = {"hide": True}

    atlas.save(os.path.join(out_dir, "remake_sprites.png"))
    Image.fromarray(hud.text_atlas(driver.tiles_text, PX), "RGBA").save(os.path.join(out_dir, "remake_text_pens.png"))
    backgrounds = []
    for i, rgba in enumerate(scenery.paint_all(driver.bg_images, driver.bg_palette, PX)):
        name = f"remake_bg{i}.png"
        to_image(rgba).save(os.path.join(out_dir, name))
        backgrounds.append(name)
    with open(os.path.join(out_dir, "remake.json"), "w", encoding="utf-8") as f:
        json.dump({"atlas": "remake_sprites.png", "sprites": entries, "text": text_entries,
                   "backgrounds": backgrounds, "background_fill": scenery.fills(),
                   "text_pens": {"image": "remake_text_pens.png", "tile": 8 * PX}}, f, indent=2)
    print(f"remake art written to {out_dir} ({len(images)} sprite images, {len(tile_images)} tiles)")


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "hdpack", "mpatrol"))
