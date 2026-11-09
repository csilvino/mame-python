"""
Tiny offline 3D renderer (signed distance fields + sphere tracing, vectorized with numpy),
used to pre-render remake sprites, the way Donkey Kong Country pre-rendered its 3D models.

World units are native screen pixels of the game; y points up, z towards the camera.
A model is a function  points (N, 3) -> (distance (N,), material id (N,)).
"""

import numpy as np

# ---------------------------------------------------------------- primitives (points: (N, 3))

def sphere(p, center, r):
    return np.linalg.norm(p - center, axis=1) - r


def round_box(p, center, half, r=0.0):
    q = np.abs(p - center) - (np.asarray(half) - r)
    return np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(axis=1), 0) - r


def ellipsoid(p, center, radii):
    q = (p - center) / radii
    k0 = np.linalg.norm(q, axis=1)
    k1 = np.linalg.norm(q / radii, axis=1)
    return k0 * (k0 - 1.0) / np.maximum(k1, 1e-6)


def cylinder(p, center, axis, r, half_len, rounding=0.0):
    """Capped cylinder along axis 0/1/2."""
    q = p - center
    a = q[:, axis]
    rest = np.delete(q, axis, axis=1)
    d = np.stack([np.linalg.norm(rest, axis=1) - (r - rounding), np.abs(a) - (half_len - rounding)], axis=1)
    return np.minimum(d.max(axis=1), 0) + np.linalg.norm(np.maximum(d, 0), axis=1) - rounding


def capsule(p, a, b, r):
    pa, ba = p - a, np.asarray(b) - a
    h = np.clip((pa @ ba) / (ba @ ba), 0, 1)
    return np.linalg.norm(pa - h[:, None] * ba, axis=1) - r


def smooth_union(d1, d2, k):
    h = np.clip(0.5 + 0.5 * (d2 - d1) / k, 0, 1)
    return d2 * (1 - h) + d1 * h - k * h * (1 - h)


def subtract(d, cut):
    return np.maximum(d, -cut)


def rotate(p, axis, angle, center=(0, 0, 0)):
    """Points expressed in a frame rotated by `angle` around `axis` (to rotate a shape by +angle)."""
    c, s = np.cos(-angle), np.sin(-angle)
    q = p - center
    i, j = [(1, 2), (0, 2), (0, 1)][axis]
    out = q.copy()
    out[:, i] = c * q[:, i] - s * q[:, j]
    out[:, j] = s * q[:, i] + c * q[:, j]
    return out + center


class Scene:
    """Collects (distance, material) pairs; the closest wins."""

    def __init__(self, n):
        self.d = np.full(n, 1e9)
        self.m = np.zeros(n, dtype=np.int32)

    def add(self, d, material):
        closer = d < self.d
        self.d = np.where(closer, d, self.d)
        self.m = np.where(closer, material, self.m)


# ---------------------------------------------------------------- materials

class Material:
    def __init__(self, color, specular=0.4, shininess=40.0, metal=0.0, rim=0.35, emissive=0.0):
        self.color = np.asarray(color, dtype=np.float64)       # linear RGB
        self.specular, self.shininess, self.metal = specular, shininess, metal
        self.rim, self.emissive = rim, emissive


def srgb_to_linear(c):
    c = np.asarray(c, dtype=np.float64) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


# ---------------------------------------------------------------- renderer

KEY_LIGHT = np.array([-0.45, 0.75, 0.5]) / np.linalg.norm([-0.45, 0.75, 0.5])     # top left, front
FILL_LIGHT = np.array([0.7, 0.2, 0.6]) / np.linalg.norm([0.7, 0.2, 0.6])
RIM_COLOR = np.array([0.55, 0.65, 1.0])          # cold light from behind (space)
KEY_COLOR = np.array([1.0, 0.95, 0.85])
FILL_COLOR = np.array([0.25, 0.3, 0.5])


def environment(r):
    """What metal reflects: black space above, the orange moon soil below."""
    up = np.clip(r[:, 1], -1, 1)[:, None]
    space = np.array([0.02, 0.025, 0.06]) + np.array([0.15, 0.15, 0.3]) * np.clip(1 - up * 3, 0, 1)
    soil = np.array([0.55, 0.28, 0.12])
    return np.where(up > 0, space, soil * np.clip(-up * 2 + 0.3, 0, 1))


def render(model, materials, x_range, v_range, px_per_unit, pitch_deg=18.0, ss=2,
           max_dist=200.0, steps=96, shade=True):
    """
    Orthographic render of `model` seen from the front, tilted down by pitch_deg (2.5D view).
    x_range: world x covered by the image. v_range: (bottom, top) along the camera's up axis.
    Returns RGBA float image (H, W, 4), straight alpha, sRGB colors.
    shade=False: coverage only (black), for fitting silhouettes.
    """
    pitch = np.radians(pitch_deg)
    right = np.array([1.0, 0.0, 0.0])
    up = np.array([0.0, np.cos(pitch), -np.sin(pitch)])
    fwd = np.array([0.0, -np.sin(pitch), -np.cos(pitch)])

    w = int(round((x_range[1] - x_range[0]) * px_per_unit))
    h = int(round((v_range[1] - v_range[0]) * px_per_unit))
    W, H = w * ss, h * ss
    jj, ii = np.mgrid[0:H, 0:W]
    u = x_range[0] + (ii.ravel() + 0.5) / W * (x_range[1] - x_range[0])
    v = v_range[1] - (jj.ravel() + 0.5) / H * (v_range[1] - v_range[0])
    origin = u[:, None] * right + v[:, None] * up - fwd * (max_dist / 2)
    n = W * H

    # sphere tracing
    t = np.zeros(n)
    hit = np.zeros(n, dtype=bool)
    active = np.ones(n, dtype=bool)
    eps = 0.5 / px_per_unit
    for _ in range(steps):
        idx = np.nonzero(active)[0]
        if len(idx) == 0:
            break
        p = origin[idx] + fwd * t[idx, None]
        d, _ = model(p)
        t[idx] += d * 0.9
        done = d < eps
        hit[idx[done]] = True
        gone = t[idx] > max_dist
        active[idx[done | gone]] = False

    rgb = np.zeros((n, 3))
    idx = np.nonzero(hit)[0] if shade else []
    if len(idx):
        p = origin[idx] + fwd * t[idx, None]
        _, mat = model(p)
        # normal by central differences
        e = 0.02
        nrm = np.zeros((len(idx), 3))
        for a in range(3):
            off = np.zeros(3); off[a] = e
            nrm[:, a] = model(p + off)[0] - model(p - off)[0]
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)

        # ambient occlusion
        ao = np.ones(len(idx))
        for k in range(1, 6):
            hk = 0.35 * k
            ao -= (hk - model(p + nrm * hk)[0]) * (0.5 ** k) * 1.2
        ao = np.clip(ao, 0.25, 1.0)

        # soft shadow from the key light
        shadow = np.ones(len(idx))
        ts = np.full(len(idx), 0.15)
        for _ in range(24):
            ds = model(p + KEY_LIGHT * ts[:, None])[0]
            shadow = np.minimum(shadow, np.clip(6.0 * ds / ts, 0, 1))
            ts += np.clip(ds, 0.05, 1.0)
        shadow = np.clip(shadow, 0.2, 1.0)

        view = -fwd
        refl = fwd - 2 * (nrm @ fwd)[:, None] * nrm
        env = environment(refl)
        fresnel = (0.04 + 0.96 * (1 - np.clip(nrm @ view, 0, 1)) ** 5)[:, None]
        for mid, m in materials.items():
            sel = mat == mid
            if not sel.any():
                continue
            N = nrm[sel]
            base = m.color
            ndl = np.clip(N @ KEY_LIGHT, 0, 1)[:, None] * shadow[sel, None]
            ndf = np.clip(N @ FILL_LIGHT, 0, 1)[:, None]
            half_v = (KEY_LIGHT + view) / np.linalg.norm(KEY_LIGHT + view)
            spec = (np.clip(N @ half_v, 0, 1) ** m.shininess)[:, None] * shadow[sel, None]
            rim = (1 - np.clip(N @ view, 0, 1))[:, None] ** 3 * m.rim
            diffuse = base * (1 - m.metal)
            col = (diffuse * (KEY_COLOR * ndl + FILL_COLOR * ndf + 0.12) * ao[sel, None]
                   + m.specular * spec * KEY_COLOR
                   + env[sel] * (m.metal * base + (1 - m.metal) * fresnel[sel] * 0.6) * ao[sel, None]
                   + RIM_COLOR * rim
                   + base * m.emissive)
            rgb[idx[sel]] = col

    rgb = linear_to_srgb(rgb).reshape(H, W, 3)
    alpha = hit.astype(np.float64).reshape(H, W)
    # supersampling: average premultiplied color
    pre = (rgb * alpha[..., None]).reshape(h, ss, w, ss, 3).mean(axis=(1, 3))
    a = alpha.reshape(h, ss, w, ss).mean(axis=(1, 3))
    col = np.where(a[..., None] > 0, pre / np.maximum(a[..., None], 1e-9), 0)
    return np.concatenate([col, a[..., None]], axis=2)


def project_v(point, pitch_deg=18.0):
    """Vertical image coordinate (camera up axis) of a world point."""
    pitch = np.radians(pitch_deg)
    return point[1] * np.cos(pitch) - point[2] * np.sin(pitch)
