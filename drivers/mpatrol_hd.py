"""
Moon Patrol (Irem M52) video rebuilt on the GPU, with high resolution graphics.

The CPU renderer in drivers/mpatrol.py composes a 240x252 image and the frontend scales it.
This one composes the screen directly at the window's resolution: every frame it uploads
the video RAM, sprite list and scroll registers (a few KB) and a shader evaluates the M52
layers (backdrop, 3 backgrounds, text tilemap, sprites) for each screen pixel, sampling the
graphics from an HD pack instead of the original 8x8 / 16x16 / 256x64 bitmaps.

HD pack: hdpack/mpatrol/
  pack.json     {"scale": N}   resolution multiplier of every image below
  text.png      512 text tiles, 32 x 16 grid of (8N x 8N) tiles, grayscale pens: value = pen * 85
  sprites.png   128 sprites, 16 x 8 grid of (16N x 16N), grayscale pens
  bg0.png       mountains, (256N x 64N) RGBA, true color, alpha 0 = transparent
  bg1.png       hills
  bg2.png       city
  sprites_light.png   (optional) lighting for the sprite atlas, same size, RGB:
                red = brightness (128 = unchanged, 255 = x2), green = added white highlight
Text tiles and sprites keep pens (not colors) because the game recolors them through the
palette PROMs at run time; backgrounds always use the same colors, so they are plain images.
remake.json + remake_sprites.png (optional): new full color artwork replacing sprites by code,
  {"atlas": "...", "sprites": {"03": {"rect": [x, y, w, h], "offset": [ox, oy], "size": [w, h]},
                               "01": {"hide": true}, ...}}
  rect is in atlas pixels; offset/size in native pixels relative to the sprite's 16x16 box
  (the art may be larger than the sprite). "hide": the sprite is covered by another one's art.
  "3D/0B": art for that code in one color only (the game flashes some sprites by changing their
  color); it takes precedence over "3D" for that color.
  "text": {"0A2": {"rect": [x, y, w, h]}, ...} replaces text tiles by code (any color): the art
  is a multiplier of pen 1 of the tile's color (rgb = multiplier / 2), so it follows the palette.
  "backgrounds": ["remake_bg0.png", ...] full color backgrounds (same layout as bg0-2.png, any
  resolution), drawn instead of the pack's ones, over the effects' sky; "background_fill": their
  solid color below the image. With the remake on, the lunar ground is drawn in grey.
  "text_pens": {"image": "remake_text_pens.png", "tile": T} the text tiles above the ground (score
  panel, messages) smoothed: for each code (32 per row, T x T texels each), RGBA = coverage of
  pens 0..3, mixed with the tile's palette colors. With the remake artwork
  on, the effects' sky also has the Earth (procedural, turning slowly, drifting with the stars).
  Made by tools/remake/mpatrol_remake.py (pre-rendered 3D models).
pack.json "effects": true adds shader extras that do not exist on the arcade: a night sky
with twinkling stars (with parallax) behind the mountains, relief lighting of the text layer
computed from the composed screen (seamless across tiles) and a moving texture on the ground.
tools/make_hdpack.py generates a pack from the ROMs.
"""

import json
import os
import numpy as np

from drivers.mpatrol import VIS_X0, VIS_Y0

# Remake: trails behind the UFOs (sprite codes of the UFO models, tools/remake/ufos.py)
TRAIL_CODES = set(range(0x42, 0x48))
TRAIL_POINTS = 40            # positions kept per UFO
MAX_TRAIL = 96               # segments sent to the shader (as in the shader)

# Remake: a smooth hill in the first part of the first stage (see MoonPatrolHD._update_hill)
HILL_EVENT = 0x2168          # obstacle script pointer that brings it (~10 s after the start)
HILL_LENGTH = 760.0          # ground pixels, up and down
HILL_HEIGHT = 26.0           # native pixels at the top
HILL_PUSH = 3.0              # buggy x pushed per frame per unit of slope (speed up / down)

VERTEX_SHADER = """
#version 330
in vec2 in_pos;
out vec2 v_uv;
void main() {
    v_uv = vec2(in_pos.x * 0.5 + 0.5, 0.5 - in_pos.y * 0.5);   // (0,0) = top-left of the game screen
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

FRAGMENT_SHADER = """
#version 330
#define VIS_X0 @VIS_X0@.0
#define VIS_Y0 @VIS_Y0@.0
#define MAX_SPRITES 64
#define MAX_TRAIL 96

uniform vec2 native;              // 240 x 252
uniform vec2 px_size;             // native units per screen pixel
uniform int  samples;             // supersampling grid (1 = one sample per pixel)
uniform int  scale;               // HD pack resolution multiplier
uniform bool scanlines;

uniform usampler2D tilemap;       // 32x32: r = video RAM (tile code), g = color RAM
uniform usampler2D text_atlas;    // pens
uniform usampler2D sprite_atlas;  // pens
uniform sampler2D  sprite_light;  // lighting map (see module docstring)
uniform sampler2D  remake_atlas;  // replacement artwork, premultiplied alpha
uniform sampler2D  remake_table;  // 128 x 34: row 0 = atlas rect (u0, v0, u1, v1),
                                  //          row 1 = offset.xy, size.xy (size.x < 0: none, 0: hidden)
                                  // rows 2 + 2 * color, 3 + 2 * color: the same for one color
                                  //          (size.x = -2: no art for that color, use rows 0-1)
uniform sampler2D  remake_text;   // 512 x 2: atlas rect of the text tile replacements (u1 = 0: none),
                                  //   row 1 r = 1: full color art (else a soil color multiplier)
uniform bool has_remake;
uniform sampler2D remake_pens;   // text tiles: coverage of pens 0..3 (see "text_pens")
uniform bool has_pens;
uniform float pens_tile;          // texels per tile in remake_pens
uniform bool has_light;
uniform bool effects;
uniform float time;               // seconds, for twinkling
uniform float star_scroll;        // parallax offset of the star field (native pixels)
uniform sampler2D  text_pal;      // 512 x 1
uniform sampler2D  sprite_pal;    // 256 x 1, alpha = opaque
uniform sampler2D  bg0;
uniform sampler2D  bg1;
uniform sampler2D  bg2;
uniform vec3 bg_fill[3];          // pen 3 of each background: the solid area below the image
uniform vec3 backdrop;

uniform int tx_scroll;
uniform int bg_xpos[2];
uniform int bg_ypos[2];
uniform int bg_control;
uniform bool flip;
uniform int n_sprites;
uniform ivec4 sprites[MAX_SPRITES];   // x, y, code, attributes  (in drawing order)
uniform int n_trail;                  // remake: UFO trails, segments (x0, y0, x1, y1) in native pixels
uniform vec4 trail_seg[MAX_TRAIL];
uniform float trail_age[MAX_TRAIL];   // 0 = at the UFO, 1 = faded
uniform vec4 trail_box;               // bounds of all the segments (x0, y0, x1, y1)
uniform vec3 hill;                    // remake: a hill on the course; x = course position of
                                      // screen x 0, y = length, z = height (0: no hill)

in vec2 v_uv;
out vec4 color;

bool drawn;                       // something other than the backdrop covers this sample
bool text_hit;                    // ... and it is the text layer
bool text_remade;                 // ... drawn with remake artwork (lit already)
bool plain_soil = false;          // text_layer: draw the plain soil tile wherever it samples
vec2 text_pos;                    // tilemap coordinates of that text pixel (move with the scroll)

float hash(vec2 v) { return fract(sin(dot(v, vec2(127.1, 311.7))) * 43758.5453); }

float vnoise(vec2 x, float period) {
    // value noise, periodic in x (the ground wraps every 256 native pixels)
    vec2 i = floor(x), f = fract(x);
    f = f * f * (3.0 - 2.0 * f);
    float x0 = mod(i.x, period), x1 = mod(i.x + 1.0, period);
    return mix(mix(hash(vec2(x0, i.y)), hash(vec2(x1, i.y)), f.x),
               mix(hash(vec2(x0, i.y + 1.0)), hash(vec2(x1, i.y + 1.0)), f.x), f.y);
}

float hash3(vec3 v) { return fract(sin(dot(v, vec3(127.1, 311.7, 74.7))) * 43758.5453); }

float vnoise3(vec3 x) {
    vec3 i = floor(x), f = fract(x);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(mix(hash3(i), hash3(i + vec3(1, 0, 0)), f.x),
                   mix(hash3(i + vec3(0, 1, 0)), hash3(i + vec3(1, 1, 0)), f.x), f.y),
               mix(mix(hash3(i + vec3(0, 0, 1)), hash3(i + vec3(1, 0, 1)), f.x),
                   mix(hash3(i + vec3(0, 1, 1)), hash3(i + vec3(1, 1, 1)), f.x), f.y), f.z);
}

float fbm3(vec3 x) {
    float s = 0.0, a = 0.5;
    for (int i = 0; i < 5; i++) { s += a * vnoise3(x); x = x * 2.03 + 0.17; a *= 0.5; }
    return s / 0.97;
}

// Remake only: the Earth in the sky, turning slowly, drifting with the stars (far away)
const float EARTH_R = 11.0;
const vec3 SUN = normalize(vec3(-0.75, -0.45, 0.5));      // from the left, like the scene's key light

vec4 earth(vec2 p) {
    // returns premultiplied color + coverage (atmosphere glow included)
    vec2 center = vec2(mod(178.0 - star_scroll * 0.35, 360.0) - 60.0, 80.0);
    vec2 d = (p - center) / EARTH_R;
    float r2 = dot(d, d);
    if (r2 > 1.6) return vec4(0.0);
    float r = sqrt(r2);
    float aa = 1.2 / EARTH_R;
    float disc = 1.0 - smoothstep(1.0 - aa, 1.0, r);
    vec3 n_edge = normalize(vec3(d, 0.0) + 1e-5);
    vec3 col = vec3(0.0);
    if (r < 1.0) {
        vec3 n = vec3(d.x, d.y, sqrt(1.0 - r2));
        // turn around the (tilted) polar axis
        float a = time * 0.12;
        mat2 rot = mat2(cos(a), -sin(a), sin(a), cos(a));
        vec3 q = n;
        q.xz = rot * q.xz;
        float lat = abs(q.y);
        float land = fbm3(q * 2.2 + 3.0);
        float hi = fbm3(q * 6.0 + 11.0);
        vec3 ocean = mix(vec3(0.02, 0.10, 0.35), vec3(0.03, 0.22, 0.55), hi);
        vec3 ground = mix(vec3(0.16, 0.36, 0.10), vec3(0.50, 0.42, 0.24), smoothstep(0.45, 0.75, hi));
        float is_land = smoothstep(0.47, 0.51, land);
        vec3 surf = mix(ocean, ground, is_land);
        surf = mix(surf, vec3(0.92, 0.95, 1.0), smoothstep(0.78, 0.86, lat + 0.08 * hi));   // ice caps
        // clouds, turning a little faster
        vec3 qc = n;
        float b = time * 0.16;
        qc.xz = mat2(cos(b), -sin(b), sin(b), cos(b)) * qc.xz;
        float cloud = smoothstep(0.55, 0.75, fbm3(qc * vec3(3.0, 6.0, 3.0) + 7.0));
        surf = mix(surf, vec3(1.0), cloud * 0.85);
        float ndl = dot(n, SUN);
        float day = smoothstep(-0.12, 0.25, ndl);
        float spec = (1.0 - is_land) * (1.0 - cloud) * pow(max(dot(n, normalize(SUN + vec3(0, 0, 1))), 0.0), 40.0) * 0.6;
        vec3 lit = surf * (0.08 + 1.1 * max(ndl, 0.0)) + spec;
        // night side: a few city lights on the land
        vec3 night = vec3(1.0, 0.75, 0.35) * is_land * (1.0 - cloud) * smoothstep(0.7, 0.8, hi) * 0.5;
        col = mix(night, lit, day);
        // atmosphere: blue at the rim of the lit side
        col += vec3(0.25, 0.45, 1.0) * pow(1.0 - n.z, 3.0) * smoothstep(-0.2, 0.5, ndl) * 0.8;
    }
    vec3 glow = vec3(0.25, 0.45, 1.0) * 0.35;
    float lit_side = smoothstep(-0.4, 0.6, dot(n_edge.xy, SUN.xy));
    float g = (1.0 - smoothstep(1.0, 1.25, r)) * (1.0 - disc) * lit_side;
    return vec4(col * disc + glow * g, disc);
}

float hill_lift(float x) {
    // how much the ground is raised at screen x: a smooth bump (no slope at its ends)
    float u = x + hill.x;
    if (u <= 0.0 || u >= hill.y) return 0.0;
    return hill.z * (0.5 - 0.5 * cos(6.2831853 * u / hill.y));
}

float hill_slope(float x) {
    // d(lift)/dx; > 0 = going up to the right
    float u = x + hill.x;
    if (u <= 0.0 || u >= hill.y) return 0.0;
    return hill.z * 3.14159265 / hill.y * sin(6.2831853 * u / hill.y);
}

vec3 ufo_trails(vec2 p) {
    // glowing exhaust, widening and fading behind each UFO
    if (n_trail == 0 || any(lessThan(p, trail_box.xy)) || any(greaterThan(p, trail_box.zw)))
        return vec3(0.0);
    float glow = 0.0;
    for (int i = 0; i < n_trail; i++) {
        vec4 s = trail_seg[i];
        vec2 ab = s.zw - s.xy;
        float h = clamp(dot(p - s.xy, ab) / max(dot(ab, ab), 1e-4), 0.0, 1.0);
        float d = length(p - s.xy - ab * h);
        float age = trail_age[i];
        float w = 0.9 + 3.0 * age;
        glow = max(glow, exp(-d * d / (w * w)) * pow(1.0 - age, 1.2));
    }
    return vec3(0.35, 0.75, 1.0) * glow * 0.9 + vec3(0.9, 0.95, 1.0) * pow(glow, 4.0) * 0.5;
}

vec3 sky(vec2 p) {
    float t = clamp(p.y / 200.0, 0.0, 1.0);
    vec3 c = mix(vec3(0.0, 0.0, 0.02), vec3(0.07, 0.03, 0.17), t * t);   // purple glow at the horizon
    vec2 q = vec2(p.x - star_scroll, p.y) * 0.75;
    vec2 cell = floor(q);
    float h = hash(cell);
    if (h > 0.982) {
        float h2 = hash(cell + 17.0);
        vec2 center = vec2(hash(cell + 3.1), hash(cell + 7.7)) * 0.6 + 0.2;
        float d = length(fract(q) - center);
        float size = 0.10 + 0.16 * h2;
        float twinkle = 0.55 + 0.45 * sin(time * (1.0 + 3.0 * h2) + h * 50.0);
        c += vec3(0.85, 0.9, 1.0) * smoothstep(size, 0.0, d) * twinkle * (0.45 + 0.55 * h2);
    }
    if (has_remake) {
        vec4 e = earth(p);
        c = c * (1.0 - e.a) + e.rgb;
        c += ufo_trails(p);
    }
    return c;
}

const vec3 LIGHT = normalize(vec3(-0.5, -0.75, 0.9));    // from the top left (y grows downwards)
const vec3 HALF_V = normalize(LIGHT + vec3(0.0, 0.0, 1.0));

vec3 relight(vec3 c, vec2 grad, float relief, float diffuse, float specular) {
    // grad: gradient of the shape's height; flat areas (grad = 0) keep their color
    vec3 n = normalize(vec3(-grad * relief, 1.0));
    float mult = 1.0 + diffuse * (dot(n, LIGHT) - LIGHT.z) / (1.0 - LIGHT.z * 0.5);
    float spec = specular * max(0.0, pow(max(dot(n, HALF_V), 0.0), 30.0) - pow(HALF_V.z, 30.0));
    return c * clamp(mult, 0.25, 1.9) + spec;
}

vec4 bg_sample(int image, vec2 uv) {
    if (image == 0) return texture(bg0, uv);
    if (image == 1) return texture(bg1, uv);
    return texture(bg2, uv);
}

vec3 background(int image, int xpos, int ypos, vec2 p, vec3 c) {
    float top = float(ypos + 16) - VIS_Y0;
    float iy = p.y - top;
    if (iy >= 0.0 && iy < 64.0) {
        float u = mod(p.x + VIS_X0 - float(xpos + 124), 256.0) / 256.0;
        vec4 t = bg_sample(image, vec2(u, iy / 64.0));
        if (t.a > 0.5) drawn = true;
        return mix(c, t.rgb, t.a);
    }
    if (iy >= 64.0 && iy < 256.0) {
        drawn = true;
        return bg_fill[image];
    }
    return c;
}

ivec2 atlas_texel(int code, int columns, float size, vec2 f) {
    // f: position inside the tile, 0..1
    vec2 cell = vec2(code % columns, code / columns);
    f = clamp(f, 0.0, 0.99999);
    return ivec2(floor((cell + f) * size * float(scale)));
}

vec3 lit(vec3 c, ivec2 texel) {
    if (!has_light) return c;
    vec2 l = texelFetch(sprite_light, texel, 0).rg;
    return c * (l.r * 2.0) + l.g;
}

vec3 text_layer(vec2 p, vec3 c) {
    float sy = mod(p.y + VIS_Y0 - 16.0, 256.0);
    float sx = sy >= 192.0 ? mod(p.x + VIS_X0 - 128.0 - float(tx_scroll), 256.0)   // scrolling rows
                           : mod(p.x + VIS_X0 + 128.0, 256.0);
    ivec2 cell = ivec2(floor(sx / 8.0), floor(sy / 8.0));
    uvec2 vc = texelFetch(tilemap, cell, 0).rg;
    int code = int(vc.r) | ((int(vc.g) & 0x80) << 1);
    int col = int(vc.g) & 0x7F;
    if (plain_soil) {
        code = 0xF3;                       // the soil tile that fills the rows under the ground
        col = 4;
    }
    if (has_remake && has_pens && sy >= 192.0 && col == 5 && code >= 0x41 && code <= 0x5A) {
        // checkpoint letter in the ground (a letter-shaped window in the soil in the arcade):
        // a dark sign with the letter in red light, blinking
        vec2 f = vec2(sx, sy) / 8.0 - vec2(cell);
        vec2 origin = vec2(code % 32, code / 32) * pens_tile;
        vec2 uv = (origin + clamp(f * pens_tile, 0.5, pens_tile - 0.5)) / vec2(32.0, 16.0) / pens_tile;
        vec4 cov = texture(remake_pens, uv);
        float letter = cov.g / max(cov.r + cov.g + cov.b + cov.a, 1e-4);
        float on = smoothstep(0.35, 0.5, abs(fract(time * 0.9) - 0.5) * 2.0);     // 0 / 1, soft edges
        vec2 e = min(f, 1.0 - f);
        float bevel = smoothstep(0.0, 0.12, min(e.x, e.y));
        vec3 plate = mix(vec3(0.20, 0.21, 0.24), vec3(0.06, 0.06, 0.08), bevel);
        vec3 glow = vec3(1.0, 0.08, 0.04) * mix(0.22, 1.35, on);
        drawn = true;
        text_remade = true;
        return mix(plate, glow, letter);
    }
    if (has_remake && sy >= 192.0) {
        vec4 rect = texelFetch(remake_text, ivec2(code, 0), 0);
        if (rect.z > 0.0) {
            vec4 t = texture(remake_atlas, mix(rect.xy, rect.zw, vec2(sx, sy) / 8.0 - vec2(cell)));
            if (texelFetch(remake_text, ivec2(code, 1), 0).r > 0.5) {
                // full color art (the moon base), over the soil pixels of the original tile
                ivec2 texel = atlas_texel(code, 32, 8.0, vec2(sx, sy) / 8.0 - vec2(cell));
                int pen = int(texelFetch(text_atlas, texel, 0).r);
                vec3 o = texelFetch(text_pal, ivec2(col * 4 + pen, 0), 0).rgb;
                bool soil = pen != 0 && o.r > 0.5 && o.r > o.g;
                if (soil || t.a > 0.5) drawn = true;
                if (soil && t.a < 0.5) {
                    text_hit = true;
                    text_pos = vec2(sx, sy);
                } else if (t.a > 0.0)
                    text_remade = true;
                return (soil ? o : c) * (1.0 - t.a) + t.rgb;
            }
            if (t.a > 0.5) drawn = true;
            if (t.a > 0.0) {
                text_hit = t.a > 0.5;
                text_remade = true;
                text_pos = vec2(sx, sy);
            }
            vec3 soil = texelFetch(text_pal, ivec2(col * 4 + 1, 0), 0).rgb;
            return c * (1.0 - t.a) + t.rgb * 2.0 * soil;
        }
    }
    if (has_remake && has_pens && sy < 192.0) {
        vec2 f = vec2(sx, sy) / 8.0 - vec2(cell);
        vec2 origin = vec2(code % 32, code / 32) * pens_tile;
        vec2 uv = (origin + clamp(f * pens_tile, 0.5, pens_tile - 0.5)) / vec2(32.0, 16.0) / pens_tile;
        vec4 cov = texture(remake_pens, uv);
        cov /= max(cov.r + cov.g + cov.b + cov.a, 1e-4);
        bool opaque0 = cell.y < 7;               // pen 0 is opaque on the top 7 tile rows
        vec3 under = opaque0 ? texelFetch(text_pal, ivec2(col * 4, 0), 0).rgb : c;
        vec3 r = under * cov.r + texelFetch(text_pal, ivec2(col * 4 + 1, 0), 0).rgb * cov.g
               + texelFetch(text_pal, ivec2(col * 4 + 2, 0), 0).rgb * cov.b
               + texelFetch(text_pal, ivec2(col * 4 + 3, 0), 0).rgb * cov.a;
        if (opaque0 || cov.r < 0.5) {
            drawn = true;
            text_hit = true;
            text_pos = vec2(sx, sy);
        }
        return r;
    }
    ivec2 texel = atlas_texel(code, 32, 8.0, vec2(sx, sy) / 8.0 - vec2(cell));
    int pen = int(texelFetch(text_atlas, texel, 0).r);
    if (pen != 0 || cell.y < 7) {          // pen 0 is opaque on the top 7 tile rows
        drawn = true;
        text_hit = true;
        text_pos = vec2(sx, sy);
        return texelFetch(text_pal, ivec2(col * 4 + pen, 0), 0).rgb;
    }
    return c;
}

vec3 sprite_layer(vec2 p0, vec3 c) {
    for (int i = 0; i < n_sprites; i++) {
        ivec4 s = sprites[i];
        vec2 p = p0;
        if (has_remake && hill.z > 0.0) {
            // on the hill: what is on the ground goes up with it (less and less higher up), the
            // buggy's body tilts with the slope (around its axles)
            float x = float(s.x) + 8.0;
            bool body = s.z == 0x03 || (s.z >= 0x09 && s.z <= 0x10);
            if (body) x += 8.0;
            float k = clamp((float(s.y) - 120.0) / 50.0, 0.0, 1.0);
            p.y += hill_lift(x) * k;
            if (body) {
                vec2 pivot = vec2(float(s.x) + 16.0, float(s.y) + 12.0);
                float a = atan(hill_slope(x)) * k;
                vec2 d = p - pivot;
                p = pivot + vec2(cos(a) * d.x - sin(a) * d.y, sin(a) * d.x + cos(a) * d.y);
            }
        }
        if (has_remake) {
            int row = 2 + 2 * ((s.w & 0x3F) % 16);
            vec4 geo = texelFetch(remake_table, ivec2(s.z, row + 1), 0);
            if (geo.z < -1.5) {
                row = 0;
                geo = texelFetch(remake_table, ivec2(s.z, 1), 0);
            }
            if (geo.z == 0.0)
                continue;                       // covered by another sprite's artwork
            if (geo.z > 0.0) {
                vec2 off = geo.xy, size = geo.zw;
                if ((s.w & 0x40) != 0) off.x = 16.0 - off.x - size.x;
                if ((s.w & 0x80) != 0) off.y = 16.0 - off.y - size.y;
                vec2 f = (p - vec2(s.xy) - off) / size;
                if (f.x < 0.0 || f.y < 0.0 || f.x >= 1.0 || f.y >= 1.0)
                    continue;
                if ((s.w & 0x40) != 0) f.x = 1.0 - f.x;
                if ((s.w & 0x80) != 0) f.y = 1.0 - f.y;
                vec4 rect = texelFetch(remake_table, ivec2(s.z, row), 0);
                vec4 t = texture(remake_atlas, mix(rect.xy, rect.zw, f));
                if (t.a > 0.5) drawn = true;
                c = c * (1.0 - t.a) + t.rgb;
                continue;
            }
        }
        vec2 local = p - vec2(s.xy);
        if (local.x < 0.0 || local.y < 0.0 || local.x >= 16.0 || local.y >= 16.0)
            continue;
        vec2 f = local / 16.0;
        if ((s.w & 0x40) != 0) f.x = 1.0 - f.x;
        if ((s.w & 0x80) != 0) f.y = 1.0 - f.y;
        ivec2 texel = atlas_texel(s.z, 16, 16.0, f);
        int pen = int(texelFetch(sprite_atlas, texel, 0).r);
        vec4 pc = texelFetch(sprite_pal, ivec2(((s.w & 0x3F) % 16) * 8 + pen, 0), 0);
        if (pc.a > 0.5) {
            drawn = true;
            c = lit(pc.rgb, texel);
        }
    }
    return c;
}

vec3 base_color(vec2 p) {
    // backgrounds + text layer, as the arcade draws them (p already flipped)
    drawn = false;
    text_hit = false;
    text_remade = false;
    // remake backgrounds have soft edges: blend them over the sky, not over the black backdrop
    vec3 c = (effects && has_remake) ? sky(p) : backdrop;
    if ((bg_control & 0x20) == 0) {
        if ((bg_control & 0x10) == 0)
            c = background(0, bg_xpos[1], bg_ypos[1], p, c);       // distant mountains
        if ((bg_control & 0x02) == 0)
            c = background(1, bg_xpos[0], bg_ypos[0], p, c);       // hills
        else if ((bg_control & 0x04) == 0)
            c = background(2, bg_xpos[0], bg_ypos[0], p, c);       // city
    }
    // remake hill: the ground (text layer) comes up, the backgrounds stay
    if (has_remake && hill.z > 0.0 && p.y > 100.0) {
        float y = p.y + hill_lift(p.x);
        // below the tilemap's last row: plain soil (every row may hold crater bottoms or letters)
        plain_soil = y >= 250.0;
        if (plain_soil) y = 242.0 + mod(y - 250.0, 8.0);
        vec3 r = text_layer(vec2(p.x, y), c);
        plain_soil = false;
        return r;
    }
    return text_layer(p, c);
}

vec3 text_effects(vec2 p, vec3 c) {
    // Relief: the height of a text shape is how much of its neighborhood has its color, so its
    // gradient points to where the same color continues. Neighbors are taken from the composed
    // screen, so shapes made of several tiles (the ground) are lit as one.
    vec2 tp = text_pos;
    if (!text_remade) {
        vec2 grad = vec2(0.0);
        for (int i = 0; i < 8; i++) {
            float a = float(i) * 0.785398;
            vec2 o = vec2(cos(a), sin(a));
            vec3 n = base_color(p + o * 0.8);
            float same = 1.0 - smoothstep(0.02, 0.15, distance(n, c));
            grad += o * same;
        }
        c = relight(c, grad / 8.0, 3.5, 0.5, 0.55);
    }
    if (tp.y >= 192.0 && c.r > c.g) {
        // lunar ground (the orange/brown soil): soft dunes + fine grain, moving with the scroll
        float n = 0.6 * vnoise(tp * 0.25, 64.0) + 0.4 * vnoise(tp * 1.5, 384.0);
        float grain = hash(vec2(mod(floor(tp.x * 3.0), 768.0), floor(tp.y * 3.0)));
        c *= 0.86 + 0.22 * n + 0.06 * grain;
    }
    return c;
}

vec3 moon_grey(vec3 c) {
    // remake: the orange soil of the arcade as grey moon dust (same brightness steps)
    return vec3(dot(c, vec3(0.299, 0.587, 0.114)) * 0.78) * vec3(0.98, 1.0, 1.03);
}

vec3 shade(vec2 p) {
    if (flip) p = native - p;
    vec3 c = base_color(p);
    bool covered = drawn;
    bool ground = text_hit && text_pos.y >= 192.0;    // (text_effects samples the neighbors)
    if (effects && text_hit)
        c = text_effects(p, c);
    if (has_remake && ground && c.r > c.g + 0.02)
        c = moon_grey(c);
    drawn = covered;
    c = sprite_layer(p, c);
    if (effects && !drawn)
        c = sky(p);
    return c;
}

void main() {
    vec2 p = v_uv * native;
    vec3 c = vec3(0.0);
    for (int j = 0; j < samples; j++)
        for (int i = 0; i < samples; i++)
            c += shade(p + (vec2(i, j) + 0.5 - 0.5 * float(samples)) / float(samples) * px_size);
    c /= float(samples * samples);
    if (scanlines) {
        float fy = fract(p.y);
        c *= fy < 0.5 ? 1.0 : (fy < 0.75 ? 0.9 : 0.55);
    }
    color = vec4(c, 1.0);
}
""".replace("@VIS_X0@", str(VIS_X0)).replace("@VIS_Y0@", str(VIS_Y0))


def _load_png(path: str, mode: str) -> np.ndarray:
    from PIL import Image
    with Image.open(path) as img:
        return np.array(img.convert(mode))


def load_pack(pack_dir: str):
    """Returns (scale, text pens, sprite pens, [bg RGBA x3]) or None if there is no pack."""
    meta_path = os.path.join(pack_dir, "pack.json")
    if not os.path.isfile(meta_path):
        return None
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    s = int(meta["scale"])
    text = (_load_png(os.path.join(pack_dir, "text.png"), "L").astype(np.int32) + 42) // 85
    sprites = (_load_png(os.path.join(pack_dir, "sprites.png"), "L").astype(np.int32) + 42) // 85
    bgs = [_load_png(os.path.join(pack_dir, f"bg{i}.png"), "RGBA") for i in range(3)]
    path = os.path.join(pack_dir, "sprites_light.png")
    sprite_light = _load_png(path, "RGB") if os.path.isfile(path) else None
    if sprite_light is not None and sprite_light.shape[:2] != sprites.shape:
        raise ValueError("sprites_light.png: must be the same size as sprites.png")
    expected = {"text.png": (128 * s, 256 * s), "sprites.png": (128 * s, 256 * s)}
    for name, arr in (("text.png", text), ("sprites.png", sprites)):
        if arr.shape != expected[name]:
            raise ValueError(f"{name}: expected {expected[name][1]}x{expected[name][0]}, got {arr.shape[1]}x{arr.shape[0]}")
    for i, bg in enumerate(bgs):
        if bg.shape[:2] != (64 * s, 256 * s):
            raise ValueError(f"bg{i}.png: expected {256 * s}x{64 * s}, got {bg.shape[1]}x{bg.shape[0]}")
    remake = None
    remake_path = os.path.join(pack_dir, "remake.json")
    if os.path.isfile(remake_path):
        with open(remake_path, encoding="utf-8") as f:
            remake = json.load(f)
        remake["image"] = _load_png(os.path.join(pack_dir, remake["atlas"]), "RGBA")
        if "text_pens" in remake:
            remake["pens_image"] = _load_png(os.path.join(pack_dir, remake["text_pens"]["image"]), "RGBA")
        remake["bg_images"] = [_load_png(os.path.join(pack_dir, name), "RGBA") for name in remake.get("backgrounds", [])]
    return {"remake": remake, "scale": s, "text": text.astype(np.uint8), "sprites": sprites.astype(np.uint8), "bgs": bgs,
            "sprites_light": sprite_light,
            "smooth": bool(meta.get("smooth_backgrounds", True)), "effects": bool(meta.get("effects", False))}


class MoonPatrolHD:
    """GPU renderer for a MoonPatrol driver instance; draw() replaces driver.render()."""

    def __init__(self, ctx, driver, pack):
        import moderngl
        self._mgl = moderngl
        self.ctx = ctx
        self.driver = driver
        scale, text, sprites, bgs, smooth = (pack[k] for k in ("scale", "text", "sprites", "bgs", "smooth"))
        has_light = pack["sprites_light"] is not None
        self.program = ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)
        quad = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4").tobytes())
        self.vao = ctx.vertex_array(self.program, [(quad, "2f", "in_pos")])

        def utex(arr, components=1):
            t = ctx.texture((arr.shape[1], arr.shape[0]), components, np.ascontiguousarray(arr).tobytes(), dtype="u1")
            t.filter = (moderngl.NEAREST, moderngl.NEAREST)
            return t

        def ftex(arr, components, linear=False):
            t = ctx.texture((arr.shape[1], arr.shape[0]), components, np.ascontiguousarray(arr).tobytes())
            f = moderngl.LINEAR if linear else moderngl.NEAREST
            t.filter = (f, f)
            return t

        self.tilemap = ctx.texture((32, 32), 2, dtype="u1")
        self.tilemap.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.textures = {
            "tilemap": self.tilemap,
            "text_atlas": utex(text),
            "sprite_atlas": utex(sprites),
            "text_pal": ftex(driver.text_palette[None, :, :], 3),
            "sprite_pal": ftex(np.concatenate(
                [driver.sprite_palette, np.where(driver.sprite_transparent, 0, 255).astype(np.uint8)[:, None]],
                axis=1)[None, :, :], 4),
        }
        if has_light:
            self.textures["sprite_light"] = ftex(pack["sprites_light"], 3)
        remake = pack.get("remake")
        if remake is not None:
            img = remake["image"].astype(np.float32) / 255.0
            img[..., :3] *= img[..., 3:4]                    # premultiply: clean edges when filtered
            t = ctx.texture((img.shape[1], img.shape[0]), 4, img.astype("f4").tobytes(), dtype="f4")
            t.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self.textures["remake_atlas"] = t
            table = np.zeros((34, 128, 4), np.float32)
            table[1, :, 2] = -1.0                            # no replacement
            table[3::2, :, 2] = -2.0                         # no art for one color only
            aw, ah = img.shape[1], img.shape[0]
            for key, e in remake["sprites"].items():
                code_hex, _, color_hex = key.partition("/")
                code = int(code_hex, 16)
                row = 2 + 2 * int(color_hex, 16) if color_hex else 0
                if e.get("hide"):
                    table[row + 1, code] = (0, 0, 0, 0)
                    continue
                x, y, w, h = e["rect"]
                table[row, code] = (x / aw, y / ah, (x + w) / aw, (y + h) / ah)
                table[row + 1, code] = (*e["offset"], *e["size"])
            t = ctx.texture((128, 34), 4, table.tobytes(), dtype="f4")
            t.filter = (moderngl.NEAREST, moderngl.NEAREST)
            self.textures["remake_table"] = t
            text_table = np.zeros((2, 512, 4), np.float32)
            for code_hex, e in remake.get("text", {}).items():
                x, y, w, h = e["rect"]
                # half a texel inside: bilinear filtering must not reach the neighbors in the atlas
                text_table[0, int(code_hex, 16)] = ((x + 0.5) / aw, (y + 0.5) / ah,
                                                    (x + w - 0.5) / aw, (y + h - 0.5) / ah)
                text_table[1, int(code_hex, 16), 0] = e.get("mode") == "color"
            t = ctx.texture((512, 2), 4, text_table.tobytes(), dtype="f4")
            t.filter = (moderngl.NEAREST, moderngl.NEAREST)
            self.textures["remake_text"] = t
            if "pens_image" in remake:
                t = ctx.texture((remake["pens_image"].shape[1], remake["pens_image"].shape[0]), 4,
                                np.ascontiguousarray(remake["pens_image"]).tobytes())
                t.filter = (moderngl.LINEAR, moderngl.LINEAR)
                self.textures["remake_pens"] = t
        for i, bg in enumerate(bgs):
            t = ftex(bg, 4, linear=smooth and scale > 1)
            t.repeat_x, t.repeat_y = True, False      # backgrounds wrap horizontally
            self.textures[f"bg{i}"] = t
        for unit, (name, tex) in enumerate(self.textures.items()):
            self.program[name].value = unit
        self._units = list(self.textures.values())
        # remake backgrounds: bound instead of the pack's ones while the remake artwork is on
        self._remake_units = {}
        names = list(self.textures)
        for i, bg in enumerate(remake["bg_images"] if remake is not None else []):
            t = ftex(bg, 4, linear=True)
            t.repeat_x, t.repeat_y = True, False
            self._remake_units[names.index(f"bg{i}")] = t

        p = self.program
        p["native"].value = (float(driver.screen_width), float(driver.screen_height))
        p["scale"].value = scale
        self._bg_fill = [tuple(driver.bg_palette[i][3] / 255.0) for i in range(3)]
        self._remake_fill = self._bg_fill
        if remake is not None and "background_fill" in remake:
            self._remake_fill = [tuple(v / 255.0 for v in c) for c in remake["background_fill"]]
        p["bg_fill"].value = self._bg_fill
        p["backdrop"].value = tuple(driver.sprite_palette[0] / 255.0)
        p["has_light"].value = has_light
        self.has_remake = remake is not None
        self.remake_enabled = self.has_remake
        p["has_remake"].value = self.has_remake
        has_pens = remake is not None and "pens_image" in remake
        if "has_pens" in p:
            p["has_pens"].value = has_pens
            p["pens_tile"].value = float(remake["text_pens"]["tile"]) if has_pens else 1.0
        p["effects"].value = pack["effects"]
        self.scale = scale
        self.frames = 0
        self.star_scroll = 0.0
        self._last_mountain_x = None
        self._trails = {}
        self._ground = 0                 # ground pixels travelled (from the text layer scroll)
        self._last_scroll = None
        self._hill_anchor = None         # self._ground when the hill's event was reached
        self._hill_start = None          # course position of screen x 0 while the hill is on
        self._push = 0.0                 # hill_physics: fraction of a pixel not applied yet

    def _visible_sprites(self):
        """Sprite list in the driver's drawing order (later entries end on top)."""
        d = self.driver
        spr = d.memory
        out = []
        for group in (0x00, 0x40, 0x80, 0xC0):
            for offs in range(group + 0x3C, group - 1, -4):
                base = 0xC800 + offs
                sy = 257 - spr[base] - VIS_Y0
                sx = spr[base + 3] + 129 - VIS_X0
                # wide margin: replacement artwork can extend beyond the 16x16 box
                if sx >= d.screen_width + 48 or sy >= d.screen_height + 48 or sx <= -64 or sy <= -64:
                    continue
                out.append((sx, sy, spr[base + 2] & 0x7F, spr[base + 1]))
        return out

    def _update_trails(self):
        """Remake: the UFOs leave a trail in space. Keeps the last centers of each UFO (per sprite
        slot); a slot that no longer holds a UFO loses one point per frame, so its trail fades."""
        spr = self.driver.memory
        for slot in range(0, 0x100, 4):
            base = 0xC800 + slot
            trail = self._trails.setdefault(slot, [])
            if spr[base + 2] & 0x7F in TRAIL_CODES:
                point = (spr[base + 3] + 129 - VIS_X0 + 8.0, 257 - spr[base] - VIS_Y0 + 8.0)
                if trail and (abs(point[0] - trail[-1][0]) > 24 or abs(point[1] - trail[-1][1]) > 24):
                    trail.clear()                                  # a new UFO, or wrapped around
                if not trail or point != trail[-1]:
                    trail.append(point)
                del trail[:-TRAIL_POINTS]
            elif trail:
                trail.pop(0)
        segs, ages = [], []
        for trail in self._trails.values():
            n = len(trail)
            for i in range(n - 1):
                segs.append((*trail[i], *trail[i + 1]))
                ages.append(1.0 - (i + 1) / TRAIL_POINTS)          # 0 = at the UFO
        segs, ages = segs[-MAX_TRAIL:], ages[-MAX_TRAIL:]
        p = self.program
        if "n_trail" not in p:
            return
        p["n_trail"].value = len(segs)
        if segs:
            p["trail_seg"].write(np.array(segs + [(0, 0, 0, 0)] * (MAX_TRAIL - len(segs)), "f4").tobytes())
            p["trail_age"].write(np.array(ages + [0.0] * (MAX_TRAIL - len(ages)), "f4").tobytes())
            a = np.array(segs)
            p["trail_box"].value = (a[:, [0, 2]].min() - 6, a[:, [1, 3]].min() - 6,
                                    a[:, [0, 2]].max() + 6, a[:, [1, 3]].max() + 6)

    def _update_hill(self):
        """Remake: a hill on the course (the game itself stays flat). It is anchored to an event
        of the stage's obstacle script (pointer at E516/E517): when the game reaches it, the hill
        comes in from the right edge, measured in ground pixels from there. A crash sends the
        pointer back to the checkpoint, before the event, so the hill comes again."""
        d, mem = self.driver, self.driver.memory
        if self._last_scroll is not None:
            self._ground += ((self._last_scroll - d.tx_scroll + 128) % 256) - 128   # forward > 0
        self._last_scroll = d.tx_scroll
        pointer = mem[0xE516] | mem[0xE517] << 8
        if not HILL_EVENT <= pointer < HILL_EVENT + 0x20:
            if pointer < HILL_EVENT or pointer >> 8 != HILL_EVENT >> 8:
                self._hill_anchor = None                  # before the event (or another course)
        elif self._hill_anchor is None:
            self._hill_anchor = self._ground
        start = None if self._hill_anchor is None else self._ground - self._hill_anchor - d.screen_width
        self._hill_start = start if start is not None and start < HILL_LENGTH else None
        if self._hill_start is not None:
            self.program["hill"].value = (start, HILL_LENGTH, HILL_HEIGHT)
        else:
            self.program["hill"].value = (0.0, 1.0, 0.0)

    def hill_physics(self):
        """Remake, called by the frontend before each emulated frame while the player plays (not
        during a replay: it changes the game): the buggy slows down going up the hill and speeds
        up going down. The game's speed follows the buggy's x on the screen (E303, pushed forward
        by the stick and drifting back to its rest place), so the slope pushes that x."""
        if not (self.has_remake and self.remake_enabled):
            return
        self._update_hill()
        if self._hill_start is None:
            self._push = 0.0
            return
        mem = self.driver.memory
        x = mem[0xE303]
        u = x + 26 + self._hill_start                          # course position of the body's center
        if not 0 < u < HILL_LENGTH:
            return
        slope = HILL_HEIGHT * np.pi / HILL_LENGTH * np.sin(2 * np.pi * u / HILL_LENGTH)
        self._push -= HILL_PUSH * slope
        step = int(self._push)                                  # whole pixels, the rest is kept
        if step:
            self._push -= step
            mem[0xE303] = int(np.clip(x + step, 40, 92))

    def draw(self, viewport, scanlines: bool = False, samples: int = 2):
        d = self.driver
        p = self.program
        if "hill" in p:
            if self.has_remake and self.remake_enabled:
                self._update_hill()
            else:
                p["hill"].value = (0.0, 1.0, 0.0)
        if self.has_remake and self.remake_enabled:
            self._update_trails()
        elif "n_trail" in p:
            p["n_trail"].value = 0
            self._trails.clear()
        mem = np.frombuffer(d.memory, dtype=np.uint8)
        self.tilemap.write(np.stack([mem[0x8000:0x8400], mem[0x8400:0x8800]], axis=1).tobytes())

        sprites = self._visible_sprites()
        p["n_sprites"].value = len(sprites)
        if sprites:
            p["sprites"].write(np.array(sprites + [(0, 0, 0, 0)] * (64 - len(sprites)), dtype="i4").tobytes())
        p["tx_scroll"].value = d.tx_scroll
        p["bg_xpos"].value = tuple(d.bg_xpos)
        p["bg_ypos"].value = tuple(d.bg_ypos)
        p["bg_control"].value = d.bg_control
        p["flip"].value = d.flip_screen
        p["px_size"].value = (d.screen_width / viewport[2], d.screen_height / viewport[3])
        p["samples"].value = samples
        p["scanlines"].value = scanlines
        p["has_remake"].value = self.has_remake and self.remake_enabled

        # Stars drift at a quarter of the mountains' speed (their scroll register wraps at 256)
        x = d.bg_xpos[1]
        if self._last_mountain_x is not None:
            self.star_scroll += 0.25 * (((x - self._last_mountain_x + 128) % 256) - 128)
        self._last_mountain_x = x
        self.frames += 1
        for name, value in (("time", self.frames / 60.0), ("star_scroll", self.star_scroll % 4096.0)):
            if name in p:
                p[name].value = value

        remade = self.has_remake and self.remake_enabled
        p["bg_fill"].value = self._remake_fill if remade else self._bg_fill
        for unit, tex in enumerate(self._units):
            (self._remake_units.get(unit, tex) if remade else tex).use(unit)
        self.ctx.viewport = viewport
        self.vao.render(self._mgl.TRIANGLE_STRIP)
