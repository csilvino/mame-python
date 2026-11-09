"""
Moon Patrol ROM Loader
Based on Computer Archeology documentation & MAME M52 Driver (src/mame/irem/m52.cpp)
"""

import os
import numpy as np
from typing import Dict, List, Optional, Tuple


def compute_resistor_weights(maxval: int, scaler: float, networks) -> Tuple[List[List[float]], float]:
    """
    Port of MAME's compute_resistor_weights() (minval = 0).
    networks: list of (resistances, pulldown, pullup).
    Returns (weights per network, scale used).
    """
    raw = []
    for resistances, pulldown, pullup in networks:
        w = []
        for n in range(len(resistances)):
            r0 = 1.0 / 1e12 if pulldown == 0 else 1.0 / pulldown
            r1 = 1.0 / 1e12 if pullup == 0 else 1.0 / pullup
            for j, r in enumerate(resistances):
                if j == n:
                    r1 += 1.0 / r
                else:
                    r0 += 1.0 / r
            r0 = 1.0 / r0
            r1 = 1.0 / r1
            vout = maxval * r0 / (r1 + r0)
            w.append(min(max(vout, 0.0), maxval))
        raw.append(w)

    if scaler < 0.0:
        scale = maxval / max(sum(w) for w in raw)
    else:
        scale = scaler
    return [[scale * v for v in w] for w in raw], scale


def combine_weights(weights: List[float], *bits: int) -> int:
    return int(sum(w * b for w, b in zip(weights, bits)) + 0.5)


class MoonPatrolROMLoader:
    """
    Loads and decodes Moon Patrol arcade ROMs.
    Uses PROMs for accurate color emulation (Irem M52 Hardware).
    """

    def __init__(self, rom_path: str = "roms/moonpatrol/"):
        self.rom_path = rom_path

        # Program ROMs (Z80 code)
        self.program_roms: Dict[str, Optional[bytearray]] = {}
        self.sound_rom: Optional[bytearray] = None

        # Graphics ROMs
        self.gfx_text: List[Optional[bytearray]] = []
        self.gfx_sprites: List[Optional[bytearray]] = []
        self.gfx_bg: List[Optional[bytearray]] = [None, None, None]

        # Color PROMs
        self.prom_text: Optional[bytearray] = None         # mpc-4.2a: 512 text colors
        self.prom_bg: Optional[bytearray] = None           # mpc-3.1m: 32 bg colors
        self.prom_sprites: Optional[bytearray] = None      # mpc-1.1f: 32 sprite colors
        self.prom_sprite_clut: Optional[bytearray] = None  # mpc-2.2h: 256 sprite lookup entries

        # Decoded graphics (pixel values, not colors)
        self.tiles_text: Optional[np.ndarray] = None     # (512, 8, 8)   2bpp
        self.tiles_sprites: Optional[np.ndarray] = None  # (128, 16, 16) 2bpp
        # bg0 = mountains (mpe-1), bg1 = hills (mpe-2), bg2 = city (mpe-3); each (128, 256)
        self.bg_images: List[Optional[np.ndarray]] = [None, None, None]

        # Palettes (RGB uint8)
        self.text_palette: Optional[np.ndarray] = None    # (512, 3), index = color * 4 + pixel
        self.bg_palette: Optional[np.ndarray] = None      # (3, 4, 3), [image][pixel]
        self.sprite_palette: Optional[np.ndarray] = None  # (256, 3), index = color * 8 + pixel
        self.sprite_transparent: Optional[np.ndarray] = None  # (256,) bool

        # Performance Cache
        self.tile_cache = {}

    def load_all(self) -> bool:
        """Load all ROM files"""
        print("Loading Moon Patrol (Irem) ROMs...")

        # Program ROMs
        self.program_roms['A'] = self._load_rom('mpa-1.3m')
        self.program_roms['B'] = self._load_rom('mpa-2.3l')
        self.program_roms['C'] = self._load_rom('mpa-3.3k')
        self.program_roms['D'] = self._load_rom('mpa-4.3j')
        self.sound_rom = self._load_rom('mp-s1.1a')

        # Graphics, ordered LSB bitplane first
        self.gfx_text = [self._load_rom('mpe-4.3f'), self._load_rom('mpe-5.3e')]
        self.gfx_sprites = [self._load_rom('mpb-1.3n'), self._load_rom('mpb-2.3m')]
        self.gfx_bg = [self._load_rom('mpe-1.3l'),   # Mountains
                       self._load_rom('mpe-2.3k'),   # Hills
                       self._load_rom('mpe-3.3h')]   # City

        # Color PROMs (MAME m52.cpp)
        self.prom_text = self._load_rom('mpc-4.2a')
        self.prom_bg = self._load_rom('mpc-3.1m')
        self.prom_sprites = self._load_rom('mpc-1.1f')
        self.prom_sprite_clut = self._load_rom('mpc-2.2h')

        self._decode_tiles()
        self._decode_backgrounds()
        self._init_palettes()

        return self._verify_roms()

    def _load_rom(self, filename: str) -> Optional[bytearray]:
        filepath = os.path.join(self.rom_path, filename)
        if not os.path.exists(filepath):
            print(f"⚠️  ROM not found: {filename}")
            return None
        try:
            with open(filepath, 'rb') as f:
                return bytearray(f.read())
        except Exception:
            return None

    def _verify_roms(self) -> bool:
        return all(self.program_roms.values()) and \
               all(self.gfx_text) and all(self.gfx_sprites) and all(self.gfx_bg) and \
               all(p is not None for p in (self.prom_text, self.prom_bg,
                                           self.prom_sprites, self.prom_sprite_clut))

    # --- Graphics decoding ---

    @staticmethod
    def _bits(rom: bytearray) -> np.ndarray:
        """ROM bytes -> bit array, MSB first (MAME bit offset 0 = bit 7)."""
        return np.unpackbits(np.frombuffer(bytes(rom), dtype=np.uint8))

    def _decode_tiles(self):
        print("Decoding tiles...")
        if all(self.gfx_text):
            # gfx_8x8x2_planar: 8 bytes per tile, one ROM per bitplane (mpe-5 = MSB)
            planes = [self._bits(r).reshape(-1, 8, 8) for r in self.gfx_text]
            self.tiles_text = (planes[0] | (planes[1] << 1)).astype(np.uint8)

        if all(self.gfx_sprites):
            # 16x16, 32 bytes per sprite: bytes 0-15 = left 8 pixels of each row,
            # bytes 16-31 = right 8 pixels. mpb-2 = bit 1, mpb-1 = bit 0
            # (MAME declares 3bpp, but the top bitplane is all zeros on Moon Patrol).
            planes = []
            for rom in self.gfx_sprites:
                b = self._bits(rom).reshape(-1, 2, 16, 8)       # sprite, half, row, px
                planes.append(b.transpose(0, 2, 1, 3).reshape(-1, 16, 16))
            self.tiles_sprites = (planes[0] | (planes[1] << 1)).astype(np.uint8)

    def _decode_backgrounds(self):
        print("Decoding backgrounds (256x64 2bpp packed images)...")
        for i, rom in enumerate(self.gfx_bg):
            if not rom:
                continue
            # 64 bytes per row, 4 pixels per byte: pixel k of a byte uses
            # bit (7-k) as LSB and bit (3-k) as MSB.
            b = self._bits(rom).reshape(64, 64, 2, 4)            # row, byte, nibble, px
            img = (b[:, :, 0, :] | (b[:, :, 1, :] << 1)).reshape(64, 256)
            # MAME's region is 0x2000 bytes with the upper half filled with 0xFF:
            # rows 64-127 are solid pixel value 3.
            full = np.full((128, 256), 3, dtype=np.uint8)
            full[:64] = img
            self.bg_images[i] = full

    # --- Palettes ---

    def _init_palettes(self):
        res3 = [1000, 470, 220]
        res2 = [470, 220]

        (w_r, w_g, w_b), scale = compute_resistor_weights(
            255, -1.0, [(res3, 0, 0), (res3, 0, 0), (res2, 0, 0)])

        def char_color(val: int) -> Tuple[int, int, int]:
            bit = lambda n: (val >> n) & 1
            return (combine_weights(w_r, bit(0), bit(1), bit(2)),
                    combine_weights(w_g, bit(3), bit(4), bit(5)),
                    combine_weights(w_b, bit(6), bit(7)))

        if self.prom_text:
            self.text_palette = np.array([char_color(v) for v in self.prom_text], dtype=np.uint8)

        if self.prom_bg:
            bg_colors = [char_color(v) for v in self.prom_bg]
            # PROM address: xbb00 = mountains, 0xxbb = hills, 1xxbb = city
            pen_map = [[0, 4, 8, 12],
                       [0, 1, 2, 3],
                       [0, 17, 18, 19]]
            self.bg_palette = np.array([[bg_colors[p] for p in pens] for pens in pen_map],
                                       dtype=np.uint8)

        if self.prom_sprites and self.prom_sprite_clut:
            # Sprites: different bit order and a 470 ohm pulldown, same scale as chars
            (s_r, s_g, s_b), _ = compute_resistor_weights(
                255, scale, [(res2, 470, 0), (res3, 470, 0), (res3, 470, 0)])
            indirect = []
            for val in self.prom_sprites:
                bit = lambda n: (val >> n) & 1
                indirect.append((combine_weights(s_r, bit(6), bit(7)),
                                 combine_weights(s_g, bit(3), bit(4), bit(5)),
                                 combine_weights(s_b, bit(0), bit(1), bit(2))))
            clut = np.array(self.prom_sprite_clut, dtype=np.uint8) & 0x1F
            self.sprite_palette = np.array(indirect, dtype=np.uint8)[clut]
            # A sprite pen is transparent when it maps to indirect color 0
            self.sprite_transparent = clut == 0

    # --- Helpers kept for debug scripts ---

    def get_tile(self, tile_num: int, tile_type: str = 'sprite') -> Optional[np.ndarray]:
        if tile_type == 'text' and self.tiles_text is not None:
            if 0 <= tile_num < len(self.tiles_text):
                return self.tiles_text[tile_num]
        elif tile_type == 'sprite' and self.tiles_sprites is not None:
            if 0 <= tile_num < len(self.tiles_sprites):
                return self.tiles_sprites[tile_num]
        return None

    def render_tile(self, tile_num: int, tile_type: str = 'sprite',
                    color: int = 0, flip_x: bool = False,
                    flip_y: bool = False) -> Optional[np.ndarray]:
        """Render a text tile (8x8) or sprite (16x16) to RGB. `color` is the hardware color code."""
        cache_key = (tile_num, tile_type, color, flip_x, flip_y)
        if cache_key in self.tile_cache:
            return self.tile_cache[cache_key]

        tile = self.get_tile(tile_num, tile_type)
        if tile is None:
            return None

        if flip_x: tile = np.fliplr(tile)
        if flip_y: tile = np.flipud(tile)

        if tile_type == 'text':
            rgb_tile = self.text_palette[(color & 0x7F) * 4 + tile]
        else:
            rgb_tile = self.sprite_palette[(color % 16) * 8 + tile]

        self.tile_cache[cache_key] = rgb_tile
        return rgb_tile

    def get_program_memory(self) -> bytearray:
        memory = bytearray(16384)
        offset = 0
        for rom_name in ['A', 'B', 'C', 'D']:
            rom = self.program_roms.get(rom_name)
            if rom:
                size = len(rom)
                memory[offset:offset + size] = rom
                offset += size
        return memory
