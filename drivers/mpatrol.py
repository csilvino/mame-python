"""
Moon Patrol (Irem, 1982) - Irem M52 hardware
Based on MAME src/mame/irem/m52.cpp (driver) and irem.cpp (sound board)

  Main CPU   Z80 @ 3.072 MHz, IRQ (RST 10h) once per frame
  Video      240x252 visible, 56.75 Hz
             - text tilemap 32x32 (8x8, 2bpp), bottom quarter scrolls horizontally
             - three 256x64 background images (mountains, hills, city), 2bpp
             - 64 sprites 16x16, 2bpp, through a color lookup PROM
             - colors from PROMs through resistor networks
  Sound      Irem M52 SoundC board (drivers.irem_audio)
  Protection custom chip on the scroll board, read at 0x8800
"""

from typing import Optional, Tuple
import numpy as np
import pygame

from chips.cpu.z80 import Z80
from chips.video.resnet import compute_resistor_weights, combine_weights
from emu.driver import Driver
from emu.rom import RomSet
from emu.audio import SAMPLE_RATE

MAIN_CLOCK = 3072000

# Visible area inside MAME's 384x282 raster (m52.cpp: set_raw(..., 384, 136, 376, 282, 22, 274))
VIS_X0, VIS_Y0 = 136, 22

# ROM_START( mpatrol )
ROMS_MAINCPU = ["mpa-1.3m", "mpa-2.3l", "mpa-3.3k", "mpa-4.3j"]   # 4 x 4 KB at 0x0000
ROM_SOUNDCPU = "mp-s1.1a"                                           # 4 KB at 0x7000 (6803)
ROMS_TX = ["mpe-4.3f", "mpe-5.3e"]           # text tiles, bitplane 0 / 1
ROMS_SPRITES = ["mpb-1.3n", "mpb-2.3m"]      # sprites, bitplane 0 / 1
ROMS_BG = ["mpe-1.3l", "mpe-2.3k", "mpe-3.3h"]   # mountains, hills, city
PROM_TX = "mpc-4.2a"           # 512 text colors
PROM_BG = "mpc-3.1m"           # 32 background colors
PROM_SPRITES = "mpc-1.1f"      # 32 sprite colors
PROM_SPRITE_CLUT = "mpc-2.2h"  # 256 sprite color lookup entries
ALL_ROMS = (ROMS_MAINCPU + [ROM_SOUNDCPU] + ROMS_TX + ROMS_SPRITES + ROMS_BG
            + [PROM_TX, PROM_BG, PROM_SPRITES, PROM_SPRITE_CLUT])


# --- Graphics decoding (MAME gfx layouts) ---

def _bits(rom: bytes) -> np.ndarray:
    """ROM bytes -> bit array, MSB first (MAME bit offset 0 = bit 7)."""
    return np.unpackbits(np.frombuffer(rom, dtype=np.uint8))


def decode_text_tiles(plane0: bytes, plane1: bytes) -> np.ndarray:
    """gfx_8x8x2_planar: 8 bytes per tile, one ROM per bitplane. Returns (512, 8, 8)."""
    planes = [_bits(r).reshape(-1, 8, 8) for r in (plane0, plane1)]
    return (planes[0] | (planes[1] << 1)).astype(np.uint8)


def decode_sprites(plane0: bytes, plane1: bytes) -> np.ndarray:
    """
    16x16, 32 bytes per sprite: bytes 0-15 = left 8 pixels of each row, bytes 16-31 = right
    8 pixels. (MAME declares 3bpp, but the top bitplane is all zeros on Moon Patrol.)
    Returns (128, 16, 16).
    """
    planes = []
    for rom in (plane0, plane1):
        b = _bits(rom).reshape(-1, 2, 16, 8)       # sprite, half, row, px
        planes.append(b.transpose(0, 2, 1, 3).reshape(-1, 16, 16))
    return (planes[0] | (planes[1] << 1)).astype(np.uint8)


def decode_background(rom: bytes) -> np.ndarray:
    """
    256x64 2bpp packed image, 64 bytes per row, 4 pixels per byte: pixel k of a byte uses
    bit (7-k) as LSB and bit (3-k) as MSB. MAME's region is 0x2000 bytes with the upper half
    filled with 0xFF, so rows 64-127 are solid pixel value 3. Returns (128, 256).
    """
    b = _bits(rom).reshape(64, 64, 2, 4)            # row, byte, nibble, px
    img = (b[:, :, 0, :] | (b[:, :, 1, :] << 1)).reshape(64, 256)
    full = np.full((128, 256), 3, dtype=np.uint8)
    full[:64] = img
    return full


def build_palettes(prom_tx: bytes, prom_bg: bytes, prom_sprites: bytes, prom_clut: bytes):
    """
    m52_palette(): returns (text_palette (512,3), bg_palette (3,4,3),
    sprite_palette (256,3), sprite_transparent (256,) bool).
    """
    res3 = [1000, 470, 220]
    res2 = [470, 220]
    (w_r, w_g, w_b), scale = compute_resistor_weights(
        255, -1.0, [(res3, 0, 0), (res3, 0, 0), (res2, 0, 0)])

    def char_color(val: int) -> Tuple[int, int, int]:
        bit = lambda n: (val >> n) & 1
        return (combine_weights(w_r, bit(0), bit(1), bit(2)),
                combine_weights(w_g, bit(3), bit(4), bit(5)),
                combine_weights(w_b, bit(6), bit(7)))

    text_palette = np.array([char_color(v) for v in prom_tx], dtype=np.uint8)

    # PROM address: xbb00 = mountains, 0xxbb = hills, 1xxbb = city
    bg_colors = [char_color(v) for v in prom_bg]
    pen_map = [[0, 4, 8, 12],
               [0, 1, 2, 3],
               [0, 17, 18, 19]]
    bg_palette = np.array([[bg_colors[p] for p in pens] for pens in pen_map], dtype=np.uint8)

    # Sprites: different bit order and a 470 ohm pulldown, same scale as chars
    (s_r, s_g, s_b), _ = compute_resistor_weights(
        255, scale, [(res2, 470, 0), (res3, 470, 0), (res3, 470, 0)])
    indirect = []
    for val in prom_sprites:
        bit = lambda n: (val >> n) & 1
        indirect.append((combine_weights(s_r, bit(6), bit(7)),
                         combine_weights(s_g, bit(3), bit(4), bit(5)),
                         combine_weights(s_b, bit(0), bit(1), bit(2))))
    clut = np.frombuffer(prom_clut, dtype=np.uint8) & 0x1F
    sprite_palette = np.array(indirect, dtype=np.uint8)[clut]
    # A sprite pen is transparent when it maps to indirect color 0
    sprite_transparent = clut == 0
    return text_palette, bg_palette, sprite_palette, sprite_transparent


class MoonPatrol(Driver):
    name = "mpatrol"
    description = "Moon Patrol"
    year = "1982"
    manufacturer = "Irem"
    rom_sets = ("mpatrol", "moonpatrol")

    screen_width = 240
    screen_height = 252
    refresh_rate = 56.75

    controls = "5=Coin, 1=Start, Left/Right=Move, Up=Jump, Space=Fire"

    # 1C = the tune at the start of the game, 18 = the stage music (looped); it ends with 00
    # (reset), 1F (crash), 1D (stage end) and 1B (game over). The other commands (shots,
    # jumps, checkpoints...) play over it.
    music_commands = {"loop": 0x18, "intro": 0x1C, "stops": (0x00, 0x1B, 0x1D, 0x1F), "game_over": 0x1B}

    # Keyboard -> (input port, bit) per MAME m52 input ports
    KEY_MAP = {
        pygame.K_1: ('in0', 0x01),       # P1 Start
        pygame.K_KP1: ('in0', 0x01),
        pygame.K_2: ('in0', 0x02),       # P2 Start
        pygame.K_KP2: ('in0', 0x02),
        pygame.K_5: ('in0', 0x08),       # Coin 1
        pygame.K_KP5: ('in0', 0x08),
        pygame.K_RIGHT: ('in1', 0x01),
        pygame.K_LEFT: ('in1', 0x02),
        pygame.K_DOWN: ('in1', 0x04),
        pygame.K_UP: ('in1', 0x20),      # Jump (Button 2)
        pygame.K_SPACE: ('in1', 0x80),   # Fire (Button 1)
    }

    def __init__(self):
        super().__init__()
        # Input state (active low)
        self.in0 = 0xFF
        self.in1 = 0xFF
        self.in2 = 0xFF
        # DIP Switches - MAME factory defaults (3 lives, upright cabinet)
        self.dsw1 = 0xFE
        self.dsw2 = 0xFD

        # Hardware registers
        self.tx_scroll = 0        # port 0x00: text layer scroll (bottom quarter only)
        self.bg_xpos = [0, 0]     # [0] = hills/city (0x40), [1] = mountains (0x80)
        self.bg_ypos = [0, 0]     # [0] = hills/city (0x60), [1] = mountains (0xA0)
        self.bg_control = 0x00    # port 0xC0
        self.flip_screen = False

        self.memory = bytearray(65536)
        self.framebuffer = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)

        self.sound = None            # in-process sound board, created on demand
        self.sound_commands = []     # (fraction of frame, value) written this frame
        self.cycles_per_frame = int(MAIN_CLOCK / self.refresh_rate)
        self.frame_start_cycles = 0

        # Text tilemap row sampled by each screen line (scrolldy = 16)
        self.tx_src_y = (np.arange(self.screen_height) + VIS_Y0 - 16) % 256
        self.tx_dest_x = np.arange(self.screen_width) + VIS_X0
        # Pen 0 is opaque on tilemap rows 0-6 (tiles 0..223), transparent elsewhere
        self.tx_force_opaque = np.zeros((256, 256), dtype=bool)
        self.tx_force_opaque[:7 * 8] = True

    # --- Machine setup ---

    def load(self, roms_root: str) -> bool:
        roms = RomSet(roms_root, self.rom_sets)
        if not roms.load(ALL_ROMS):
            where = roms.directory or f"{roms_root}/{self.rom_sets[0]}"
            print(f"❌ Missing ROMs in {where}: {', '.join(roms.missing)}")
            return False

        program = b"".join(roms[f] for f in ROMS_MAINCPU)
        self.memory[:len(program)] = program
        self.sound_rom = roms[ROM_SOUNDCPU]

        self.tiles_text = decode_text_tiles(*(roms[f] for f in ROMS_TX))
        self.tiles_sprites = decode_sprites(*(roms[f] for f in ROMS_SPRITES))
        self.bg_images = [decode_background(roms[f]) for f in ROMS_BG]
        (self.text_palette, self.bg_palette,
         self.sprite_palette, self.sprite_transparent) = build_palettes(
            roms[PROM_TX], roms[PROM_BG], roms[PROM_SPRITES], roms[PROM_SPRITE_CLUT])

        self.cpu = Z80(self.memory, debug_log=False)
        self.cpu._read_byte = self._read_byte
        self.cpu._write_byte = self._write_byte
        self.cpu._read_port = self._read_port
        self.cpu._write_port = self._write_port
        return True

    def create_hd_renderer(self, ctx, hdpack_root: str):
        import os
        from drivers.mpatrol_hd import MoonPatrolHD, load_pack
        pack = load_pack(os.path.join(hdpack_root, self.name))
        return MoonPatrolHD(ctx, self, pack) if pack else None

    def sound_board(self):
        return ("drivers.irem_audio", "IremM52Sound", (bytes(self.sound_rom),))

    def shutdown(self):
        self.cpu.print_missing_opcodes_report()

    # --- Memory map ---

    def _read_byte(self, address: int) -> int:
        address &= 0xFFFF

        # ROM / Video RAM / Color RAM (0x0000 - 0x87FF) - Highly frequent case
        if address < 0x8800:
            return self.memory[address]

        # Protection Read (0x8800 - 0x8FFF)
        if address < 0x9000:
            return self._read_protection()

        # Input Ports (0xD000 - 0xD7FF, mirrored every 8 bytes)
        if 0xD000 <= address < 0xD800:
            port = address & 0x07
            if port == 0: return self.in0
            if port == 1: return self.in1
            if port == 2: return self.in2
            if port == 3: return self.dsw1
            if port == 4: return self.dsw2
            return 0xFF

        return self.memory[address]

    def _write_byte(self, address: int, value: int):
        address &= 0xFFFF
        value &= 0xFF

        # Prevent writing to ROM
        if address < 0x4000:
            return

        # Sprite RAM (0xC800 - 0xCBFF, mirrored at 0xCC00)
        if 0xC800 <= address < 0xD000:
            self.memory[0xC800 + (address & 0x3FF)] = value
            return

        # 0xD000: sound command, 0xD001: flip screen (mirrored)
        if 0xD000 <= address < 0xD800:
            if address & 0x03 == 0:
                self._sound_command(value)
            elif address & 0x03 == 1:
                self.flip_screen = bool(value & 0x01)
            return

        self.memory[address] = value

    def _read_protection(self) -> int:
        """
        Custom chip on the scroll board: it mangles the value written to the
        hills/city X scroll port (MAME: popcount(value & 0x7f) ^ (value >> 7)).
        """
        value = self.bg_xpos[0]
        return bin(value & 0x7F).count('1') ^ (value >> 7)

    # --- Port I/O ---

    def _read_port(self, port: int) -> int:
        """M52 has no readable ports."""
        return 0xFF

    def _write_port(self, port: int, value: int):
        """Each port is mirrored over 0x20 addresses."""
        port &= 0xE0
        if port == 0x00:
            self.tx_scroll = value
        elif port == 0x40:
            self.bg_xpos[0] = value
        elif port == 0x60:
            self.bg_ypos[0] = value
        elif port == 0x80:
            self.bg_xpos[1] = value
        elif port == 0xA0:
            self.bg_ypos[1] = value
        elif port == 0xC0:
            self.bg_control = value

    # --- Sound ---

    def _sound_command(self, command: int):
        """Command for the sound board, timestamped within the current frame."""
        frac = (self.cpu.cycles - self.frame_start_cycles) / self.cycles_per_frame
        self.sound_commands.append((min(max(frac, 0.0), 0.999), command))

    # --- Frame ---

    def run_frame(self) -> Optional[np.ndarray]:
        self.frame_start_cycles = self.cpu.cycles
        if not self.cpu.halted:
            self.cpu.run_for_cycles(self.cycles_per_frame)

        # VBLANK interrupt (RST 10h)
        self.cpu.request_interrupt(0xD7)

        commands, self.sound_commands = self.sound_commands, []
        if self.sound_sink is not None:
            self.sound_sink(commands)
            return None
        # No sound process (tests, tools): run the board here and return its samples
        if self.sound is None:
            from drivers.irem_audio import IremM52Sound
            self.sound = IremM52Sound(bytes(self.sound_rom), sample_rate=SAMPLE_RATE)
        for frac, value in commands:
            self.sound.cmd_w(value, frac)
        return self.sound.run_frame(1.0 / self.refresh_rate)

    # --- Video (m52_state::screen_update) ---

    def draw_background(self, image: int, xpos: int, ypos: int):
        """
        Draw one of the 256x64 background images (MAME draw_background).
        Pixel 0 is transparent; below the image a solid fill uses pen 3.
        """
        fb = self.framebuffer
        img = self.bg_images[image]
        pal = self.bg_palette[image]

        xpos += 124
        ypos += 16
        top = ypos - VIS_Y0

        # Image (128 rows: 64 rows of graphics + 64 rows of pen 3)
        y0, y1 = max(top, 0), min(top + 128, self.screen_height)
        if y0 < y1:
            cols = (self.tx_dest_x - xpos) % 256
            pix = img[y0 - top:y1 - top][:, cols]
            mask = pix != 0
            fb[y0:y1][mask] = pal[pix[mask]]

        # Solid fill below the image
        y0, y1 = max(top + 128, 0), min(top + 256, self.screen_height)
        if y0 < y1:
            fb[y0:y1] = pal[3]

    def draw_text_layer(self):
        """
        Text/foreground layer. VRAM: 0x8000-0x83FF, Color RAM: 0x8400-0x87FF.
        Layout: Row-Major (32x32), 128 colors of 4 pens.
        """
        vram = np.frombuffer(self.memory, dtype=np.uint8, count=1024, offset=0x8000)
        cram = np.frombuffer(self.memory, dtype=np.uint8, count=1024, offset=0x8400)

        # Tile number: bit 7 of color attr is the MSB of tile code
        codes = vram.astype(np.int32) | ((cram & 0x80).astype(np.int32) << 1)
        colors = (cram & 0x7F).astype(np.int32).reshape(32, 32)

        pix = self.tiles_text[codes.reshape(32, 32)]           # (32, 32, 8, 8)
        pix = pix.transpose(0, 2, 1, 3).reshape(256, 256)
        pens = np.repeat(np.repeat(colors, 8, axis=0), 8, axis=1) * 4 + pix
        opaque = (pix != 0) | self.tx_force_opaque

        # Rows 0-191 use scroll 255, rows 192-255 use the scroll register (scrolldx = 127)
        sx_fixed = (self.tx_dest_x + 128) % 256
        sx_scroll = (self.tx_dest_x - 128 - self.tx_scroll) % 256
        sy = self.tx_src_y[:, None]
        sx = np.where(sy >= 192, sx_scroll[None, :], sx_fixed[None, :])

        mask = opaque[sy, sx]
        self.framebuffer[mask] = self.text_palette[pens[sy, sx][mask]]

    def draw_sprites(self):
        """
        The 64 16x16 sprites. Sprite RAM is at 0xC800 (4 bytes each).
        Drawn in 4 groups of 16; inside a group the lower offset ends on top.
        """
        fb = self.framebuffer
        spr = self.memory

        for group in (0x00, 0x40, 0x80, 0xC0):
            for offs in range(group + 0x3C, group - 1, -4):
                base = 0xC800 + offs
                sy = 257 - spr[base] - VIS_Y0
                attrs = spr[base + 1]
                code = spr[base + 2] & 0x7F
                sx = spr[base + 3] + 129 - VIS_X0

                if sx >= self.screen_width or sy >= self.screen_height or sx <= -16 or sy <= -16:
                    continue

                tile = self.tiles_sprites[code]
                if attrs & 0x40: tile = tile[:, ::-1]
                if attrs & 0x80: tile = tile[::-1, :]

                pens = ((attrs & 0x3F) % 16) * 8 + tile

                # Clip to screen
                x0, y0 = max(sx, 0), max(sy, 0)
                x1, y1 = min(sx + 16, self.screen_width), min(sy + 16, self.screen_height)
                pens = pens[y0 - sy:y1 - sy, x0 - sx:x1 - sx]

                mask = ~self.sprite_transparent[pens]
                fb[y0:y1, x0:x1][mask] = self.sprite_palette[pens[mask]]

    def render(self) -> np.ndarray:
        """Render the frame (layer order from MAME screen_update)."""
        # Backdrop is sprite pen 0
        self.framebuffer[:] = self.sprite_palette[0]

        ctrl = self.bg_control
        if not (ctrl & 0x20):
            if not (ctrl & 0x10):
                self.draw_background(0, self.bg_xpos[1], self.bg_ypos[1])   # distant mountains
            # Hills and city share the same scroll registers; only one is drawn
            if not (ctrl & 0x02):
                self.draw_background(1, self.bg_xpos[0], self.bg_ypos[0])   # hills
            elif not (ctrl & 0x04):
                self.draw_background(2, self.bg_xpos[0], self.bg_ypos[0])   # cityscape

        self.draw_text_layer()
        self.draw_sprites()

        frame = self.framebuffer
        if self.flip_screen:
            frame = frame[::-1, ::-1]
        return frame
