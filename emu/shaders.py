"""
GPU upscaling filters (GLSL, run through ModernGL).

The game draws at its native resolution (240x252 for Moon Patrol); these filters decide how
each native pixel becomes a block of screen pixels:

  nearest  - hard square pixels (the original look)
  smooth   - bilinear: soft but blurry
  xbr      - edge-directed pixel-art scaler in the style of Hyllian's xBR (level 1):
             detects diagonal edges between pixels and redraws them as smooth lines,
             keeping flat areas and straight edges sharp
"""

import numpy as np

VERTEX_SHADER = """
#version 330
in vec2 in_pos;
out vec2 v_uv;
void main() {
    v_uv = vec2(in_pos.x * 0.5 + 0.5, 0.5 - in_pos.y * 0.5);   // row 0 of the texture at the top
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

FRAGMENT_SHADER = """
#version 330
uniform sampler2D tex;      // native frame (GL_LINEAR in smooth mode, GL_NEAREST otherwise)
uniform vec2 src_size;      // native resolution
uniform float scale;        // screen pixels per native pixel
uniform int mode;           // 0 = nearest, 1 = smooth, 2 = xbr
uniform bool scanlines;
in vec2 v_uv;
out vec4 color;

const vec3 YUV_WEIGHT = vec3(48.0, 7.0, 6.0);

vec3 px(vec2 cell) { return texture(tex, (cell + 0.5) / src_size).rgb; }

float dist(vec3 a, vec3 b) {
    // weighted YUV distance: luma differences matter most
    vec3 d = a - b;
    vec3 yuv = vec3(dot(d, vec3(0.299, 0.587, 0.114)),
                    dot(d, vec3(-0.169, -0.331, 0.5)),
                    dot(d, vec3(0.5, -0.419, -0.081)));
    return dot(abs(yuv), YUV_WEIGHT);
}

vec3 xbr(vec2 cell, vec2 fp) {
    // Work on the corner of the native pixel this screen pixel lies in
    vec2 d = vec2(fp.x < 0.5 ? -1.0 : 1.0, fp.y < 0.5 ? -1.0 : 1.0);

    //   C  B  C4        E = current pixel, corner towards I
    //   D  E  F  F4     (layout shown for the bottom-right corner;
    //   G  H  I  I4      the others are mirrored through d)
    //   D5 H5 I5
    vec3 E  = px(cell);
    vec3 F  = px(cell + vec2(d.x, 0.0));
    vec3 H  = px(cell + vec2(0.0, d.y));
    vec3 I  = px(cell + d);
    vec3 C  = px(cell + vec2(d.x, -d.y));
    vec3 G  = px(cell + vec2(-d.x, d.y));
    vec3 F4 = px(cell + vec2(2.0 * d.x, 0.0));
    vec3 I4 = px(cell + vec2(2.0 * d.x, d.y));
    vec3 C4 = px(cell + vec2(2.0 * d.x, -d.y));
    vec3 H5 = px(cell + vec2(0.0, 2.0 * d.y));
    vec3 I5 = px(cell + vec2(d.x, 2.0 * d.y));
    vec3 D5 = px(cell + vec2(-d.x, 2.0 * d.y));

    // Edge along F-H (crossing the corner) is stronger than along E-I?
    float e = dist(E, C) + dist(E, G) + dist(I, F4) + dist(I, H5) + 4.0 * dist(H, F);
    float i = dist(H, D5) + dist(H, I5) + dist(F, I4) + dist(F, C4) + 4.0 * dist(E, I);
    bool edge = (e < i) && dist(E, F) > 1.0 && dist(E, H) > 1.0;

    if (!edge)
        return E;
    // distance of this screen pixel from the corner, along the diagonal (1.5 = the edge line)
    vec2 q = vec2(d.x > 0.0 ? fp.x : 1.0 - fp.x, d.y > 0.0 ? fp.y : 1.0 - fp.y);
    float aa = 1.0 / scale;                     // one screen pixel of anti-aliasing
    float k = smoothstep(1.5 - aa, 1.5 + aa, q.x + q.y);
    vec3 other = dist(E, F) <= dist(E, H) ? F : H;
    return mix(E, other, k);
}

void main() {
    vec2 pos = v_uv * src_size;
    vec2 cell = floor(pos);
    vec2 fp = pos - cell;                       // position inside the native pixel, 0..1

    vec3 c;
    if (mode == 2)      c = xbr(cell, fp);
    else if (mode == 1) c = texture(tex, v_uv).rgb;
    else                c = px(cell);

    if (scanlines) {
        // dark gap at the bottom of each native line, faint aperture grille
        float row = fp.y < 0.5 ? 1.0 : (fp.y < 0.75 ? 0.9 : 0.55);
        float col = fp.x < 0.75 ? 1.0 : 0.9;
        c *= row * col;
    }
    color = vec4(c, 1.0);
}
"""

FILTERS = ("nearest", "smooth", "xbr")
FILTER_NAMES = {"nearest": "original (square pixels)", "smooth": "smooth (bilinear)",
                "xbr": "xBR (smoothed edges)"}


class ShaderScaler:
    """Draws a native-resolution RGB frame onto the current framebuffer with a filter."""

    def __init__(self, ctx, src_w: int, src_h: int):
        import moderngl
        self._mgl = moderngl
        self.ctx = ctx
        self.src_size = (src_w, src_h)
        self.texture = ctx.texture((src_w, src_h), 3)
        self.texture.repeat_x = self.texture.repeat_y = False   # clamp at the borders
        self.program = ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)
        self.program["src_size"].value = (float(src_w), float(src_h))
        quad = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4").tobytes())
        self.vao = ctx.vertex_array(self.program, [(quad, "2f", "in_pos")])

    def draw(self, frame: np.ndarray, filter_name: str, viewport, scanlines: bool = False):
        """frame: (h, w, 3) uint8. viewport: (x, y, w, h) in framebuffer pixels."""
        mgl = self._mgl
        self.texture.write(np.ascontiguousarray(frame).tobytes())
        linear = filter_name == "smooth"
        self.texture.filter = (mgl.LINEAR, mgl.LINEAR) if linear else (mgl.NEAREST, mgl.NEAREST)
        self.program["mode"].value = FILTERS.index(filter_name)
        self.program["scale"].value = viewport[3] / self.src_size[1]
        self.program["scanlines"].value = scanlines
        self.ctx.viewport = viewport
        self.texture.use(0)
        self.vao.render(mgl.TRIANGLE_STRIP)
