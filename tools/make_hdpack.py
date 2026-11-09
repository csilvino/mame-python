"""
Generate an HD pack for Moon Patrol from the original ROM graphics.

    python tools/make_hdpack.py                 -> hdpack/mpatrol, scale 8, original pixels, 3D lighting
    python tools/make_hdpack.py --method smooth -> same, with rounded contours
    python tools/make_hdpack.py --style flat    -> smooth contours only, original flat colors
    python tools/make_hdpack.py --scale 1 --method nearest --style flat --out hdpack/mpatrol_1x
                                                -> exact copy of the original graphics (for testing)

Method "smooth": each pen of an image becomes a mask; the masks are enlarged, blurred, and every
HD pixel takes the pen whose blurred mask is strongest. Staircase edges become curves and
diagonal lines become straight; the center of every original pixel keeps its pen, so single
pixels (stars, bullets, dots of letters) survive as round dots.

Style "3d" (pseudo-3D lighting, like normal-mapped 2D games): every shape gets a relief
(higher in the middle, lower at its edges), lit by a light from the top left. For sprites the
lighting is stored apart from the pens (sprites_light.png: red = brightness x2, green = specular
highlight), because the game recolors them at run time. The backgrounds are baked: relief, rock
texture, light gradient. The text layer is lit by the shader at run time (its tiles join into
larger shapes, like the ground, so per-tile lighting would show seams).

The resulting PNGs can be edited or replaced by hand-drawn art (see drivers/mpatrol_hd.py).
"""

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from drivers.mpatrol import MoonPatrol   # noqa: E402


def _gaussian_kernel(sigma: float) -> np.ndarray:
    r = max(1, int(np.ceil(3 * sigma)))
    x = np.arange(-r, r + 1)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def _blur(a: np.ndarray, sigma: float) -> np.ndarray:
    """Separable gaussian blur over the last two axes (a is already padded)."""
    k = _gaussian_kernel(sigma)
    r = len(k) // 2
    out = np.zeros_like(a)
    for i, w in enumerate(k):                       # rows
        out += w * np.roll(a, i - r, axis=-2)
    a, out = out, np.zeros_like(a)
    for i, w in enumerate(k):                       # columns
        out += w * np.roll(a, i - r, axis=-1)
    return out


PAD = 4       # native pixels of context around tiles and sprites
BG_PAD = 12   # backgrounds: wider, their relief uses a broad blur


def upscale_pens(images: np.ndarray, scale: int, n_pens: int, pad_mode, method: str,
                 sigma: float = 0.45, keep_center: float = 0.3, pad: int = PAD) -> np.ndarray:
    """
    images: (n, h, w) pens. pad_mode: see _pad().
    Returns (n, (h+2*pad)*scale, (w+2*pad)*scale) pens, still padded (crop with crop()).
    The blurs wrap around the padded array, so `pad` must exceed their radius.
    """
    padded = np.stack([_pad(img, pad, pad_mode) for img in images])        # (n, h+4, w+4)
    big = padded.repeat(scale, axis=1).repeat(scale, axis=2)
    if method == "nearest" or scale == 1:
        return big
    masks = np.stack([(big == k).astype(np.float32) for k in range(n_pens)])  # (pens, n, H, W)
    strength = _blur(masks, sigma * scale)
    out = strength.argmax(axis=0).astype(np.uint8)

    # Keep the pen of every original pixel around its center
    yy, xx = np.mgrid[0:big.shape[1], 0:big.shape[2]]
    fy = (yy % scale + 0.5) / scale - 0.5
    fx = (xx % scale + 0.5) / scale - 0.5
    center = (fx * fx + fy * fy) <= keep_center * keep_center
    out[:, center] = big[:, center]
    return out


def crop(a: np.ndarray, scale: int, pad: int = PAD) -> np.ndarray:
    p = pad * scale
    return a[:, p:a.shape[1] - p, p:a.shape[2] - p]


# --- Pseudo-3D lighting ---

LIGHT = np.array([-0.5, -0.75, 0.9]) / np.linalg.norm([-0.5, -0.75, 0.9])   # from the top left
HALF = (LIGHT + [0, 0, 1]) / np.linalg.norm(LIGHT + [0, 0, 1])               # Blinn half vector


def own_pen_height(pens: np.ndarray, n_pens: int, sigma: float) -> np.ndarray:
    """Relief of every region of equal pen: 1 deep inside, ~0.5 on its borders."""
    masks = np.stack([(pens == k).astype(np.float32) for k in range(n_pens)])
    blurred = _blur(masks, sigma)
    return np.take_along_axis(blurred, pens[None].astype(np.int64), axis=0)[0]


def lighting(height: np.ndarray, relief: float, diffuse: float = 0.55,
             specular: float = 0.8, shininess: float = 30.0):
    """height (n, H, W) -> (brightness multiplier, specular) arrays; flat areas give (1, 0)."""
    gy, gx = np.gradient(height, axis=(-2, -1))
    nx, ny, nz = -gx * relief, -gy * relief, np.ones_like(height)
    norm = np.sqrt(nx * nx + ny * ny + nz * nz)
    nx, ny, nz = nx / norm, ny / norm, nz / norm
    ndl = nx * LIGHT[0] + ny * LIGHT[1] + nz * LIGHT[2]
    ndh = np.clip(nx * HALF[0] + ny * HALF[1] + nz * HALF[2], 0, 1)
    mult = 1.0 + diffuse * (ndl - LIGHT[2]) / (1 - LIGHT[2] * 0.5)
    spec = specular * np.clip(ndh ** shininess - HALF[2] ** shininess, 0, None)
    return np.clip(mult, 0.25, 1.9), np.clip(spec, 0, 1)


def light_map(pens: np.ndarray, scale: int, flat_pen0: bool) -> np.ndarray:
    """Padded HD pens -> cropped RGB light map (R = multiplier * 127.5, G = specular * 255)."""
    sigma = 0.9 * scale
    h = own_pen_height(pens, 4, sigma)
    mult, spec = lighting(h, relief=sigma * 1.6)
    if flat_pen0:
        mult[pens == 0], spec[pens == 0] = 1.0, 0.0
    out = np.zeros(pens.shape + (3,), np.uint8)
    out[..., 0] = np.clip(mult * 127.5 + 0.5, 0, 255)
    out[..., 1] = np.clip(spec * 255 + 0.5, 0, 255)
    return np.stack([crop(out[..., c], scale) for c in range(3)], axis=-1)


def periodic_noise(h: int, w: int, cells_x: int, seed: int, octaves: int = 4) -> np.ndarray:
    """Fractal value noise in [-1, 1], seamless horizontally (the backgrounds wrap)."""
    rng = np.random.default_rng(seed)
    total = np.zeros((h, w))
    amp, norm = 1.0, 0.0
    for o in range(octaves):
        cx = cells_x * 2 ** o
        cy = max(2, int(round(cx * h / w)))
        grid = rng.random((cy + 1, cx))
        y = np.linspace(0, cy, h, endpoint=False)
        x = np.linspace(0, cx, w, endpoint=False)
        y0, x0 = y.astype(int), x.astype(int)
        ty, tx = y - y0, x - x0
        ty, tx = ty * ty * (3 - 2 * ty), tx * tx * (3 - 2 * tx)       # smoothstep
        x1 = (x0 + 1) % cx
        g = grid
        top = g[y0][:, x0] * (1 - tx) + g[y0][:, x1] * tx
        bot = g[y0 + 1][:, x0] * (1 - tx) + g[y0 + 1][:, x1] * tx
        total += amp * (top * (1 - ty[:, None]) + bot * ty[:, None])
        norm += amp
        amp *= 0.5
    return (total / norm) * 2 - 1


def shade_background(pens: np.ndarray, rgb: np.ndarray, scale: int, index: int) -> np.ndarray:
    """Padded HD pens + flat colors (both padded) -> lit RGB, cropped."""
    solid = (pens != 0).astype(np.float32)[None]
    # broad volume of the whole silhouette + a finer bevel between color areas
    height = 0.75 * _blur(solid, 3.0 * scale)[0] + 0.25 * own_pen_height(pens[None], 4, 0.9 * scale)[0]
    mult, spec = lighting(height[None], relief=3.0 * scale * 1.2, diffuse=0.6, specular=0.35, shininess=12)
    mult, spec = mult[0], spec[0]
    hh, ww = pens.shape
    p = BG_PAD * scale
    rows = np.arange(hh) - p                                   # row inside the 64-row image
    t = np.clip(rows / (64 * scale), 0, 1)[:, None]
    gradient = 1.12 - 0.12 * t                                 # tops catch more light
    fade = np.clip((64 * scale - rows) / (10 * scale), 0, 1)[:, None]   # no texture at the seam
    # period = the real image width, then padded like the image, so it wraps without a seam
    noise = np.pad(periodic_noise(hh, ww - 2 * p, cells_x=24, seed=7 + index), ((0, 0), (p, p)), mode="wrap")
    texture = 1.0 + 0.10 * noise * fade
    out = rgb * (mult * gradient * texture)[..., None] + 255 * (spec * fade)[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)[p:hh - p, p:ww - p]


def _pad(img: np.ndarray, pad: int, mode):
    """mode: (vertical, horizontal); 'edge', 'wrap', or constant pens: (top, bottom) / (left, right)."""
    v, hz = mode
    if isinstance(hz, str):
        out = np.pad(img, ((0, 0), (pad, pad)), mode=hz)
    else:
        left, right = hz
        out = np.concatenate([np.full((img.shape[0], pad), left, img.dtype), img,
                              np.full((img.shape[0], pad), right, img.dtype)], axis=1)
    if isinstance(v, str):
        return np.pad(out, ((pad, pad), (0, 0)), mode=v)
    top, bottom = v
    return np.concatenate([np.full((pad, out.shape[1]), top, out.dtype), out,
                           np.full((pad, out.shape[1]), bottom, out.dtype)])


def atlas_rgb(tiles: np.ndarray, columns: int) -> np.ndarray:
    return np.stack([atlas(tiles[..., c], columns) for c in range(tiles.shape[-1])], axis=-1)


def atlas(tiles: np.ndarray, columns: int) -> np.ndarray:
    n, th, tw = tiles.shape
    rows = n // columns
    return tiles.reshape(rows, columns, th, tw).transpose(0, 2, 1, 3).reshape(rows * th, columns * tw)


def build(out_dir: str, scale: int, method: str, style: str):
    driver = MoonPatrol()
    if not driver.load(os.path.join(ROOT, "roms")):
        raise SystemExit(1)
    os.makedirs(out_dir, exist_ok=True)

    text = upscale_pens(driver.tiles_text, scale, 4, ("edge", "edge"), method)
    # sprites stand alone: around them is transparent pen 0
    sprites = upscale_pens(driver.tiles_sprites, scale, 4, ((0, 0), (0, 0)), method)
    Image.fromarray(atlas(crop(text, scale), 32) * 85).save(os.path.join(out_dir, "text.png"))
    Image.fromarray(atlas(crop(sprites, scale), 16) * 85).save(os.path.join(out_dir, "sprites.png"))
    for name in ("text_light.png", "sprites_light.png"):
        if os.path.exists(os.path.join(out_dir, name)):
            os.remove(os.path.join(out_dir, name))
    if style == "3d":
        light = light_map(sprites, scale, flat_pen0=True)
        Image.fromarray(atlas_rgb(light, 16)).save(os.path.join(out_dir, "sprites_light.png"))

    for i in range(3):
        # Only the 64 rows of real graphics; the background wraps horizontally, nothing is
        # drawn above it (pen 0) and the solid pen 3 area continues below it
        img = driver.bg_images[i][:64][None]
        hd_padded = upscale_pens(img, scale, 4, ((0, 3), "wrap"), method, pad=BG_PAD)[0]
        hd = crop(hd_padded[None], scale, BG_PAD)[0]
        rgba = np.zeros(hd.shape + (4,), np.uint8)
        if style == "3d":
            rgba[..., :3] = shade_background(hd_padded, driver.bg_palette[i][hd_padded].astype(np.float32), scale, i)
        else:
            rgba[..., :3] = driver.bg_palette[i][hd]
        rgba[..., 3] = np.where(hd == 0, 0, 255)
        Image.fromarray(rgba, "RGBA").save(os.path.join(out_dir, f"bg{i}.png"))

    with open(os.path.join(out_dir, "pack.json"), "w", encoding="utf-8") as f:
        json.dump({"game": "mpatrol", "scale": scale, "method": method, "style": style,
                   "smooth_backgrounds": method != "nearest",
                   # shader extras: night sky with stars, moving ground texture
                   "effects": style == "3d"}, f, indent=2)
    print(f"HD pack written to {out_dir} (scale {scale}, {method}, {style})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scale", type=int, default=8)
    ap.add_argument("--method", choices=("smooth", "nearest"), default="nearest")
    ap.add_argument("--style", choices=("3d", "flat"), default="3d")
    ap.add_argument("--out", default=os.path.join(ROOT, "hdpack", "mpatrol"))
    args = ap.parse_args()
    build(args.out, args.scale, args.method, args.style)


if __name__ == "__main__":
    main()
