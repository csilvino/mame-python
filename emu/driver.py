"""
Base class for game drivers (the role of MAME's driver_device + GAME() macro).

A driver describes one arcade machine: its ROMs, screen, inputs, and how to emulate
one video frame. The frontend (emu.frontend) does everything else: window, timing,
keyboard and the sound process.
"""

from typing import Dict, Optional, Tuple
import numpy as np


class Driver:
    # --- Metadata (GAME() macro) ---
    name = ""                 # short name, as in MAME ("mpatrol")
    description = ""
    year = ""
    manufacturer = ""
    rom_sets: Tuple[str, ...] = ()   # directory names searched under roms/

    # --- Screen ---
    screen_width = 0
    screen_height = 0
    refresh_rate = 60.0

    controls = ""            # help line shown at startup

    # Background music, for its replacement by the remake (emu.music):
    # {"loop": start command, "intro": command or None, "stops": commands that end it,
    #  "game_over": command after which it starts over}, or None
    music_commands = None

    # Keyboard -> input port bit (ports are active low): {pygame key: (attribute, bit)}
    KEY_MAP: Dict[int, Tuple[str, int]] = {}

    def __init__(self):
        self._pending_releases = []
        # Set by the frontend when the sound board runs in its own process: called once
        # per frame with that frame's sound commands
        self.sound_sink = None

    # --- Inputs ---

    def key_event(self, key: int, pressed: bool):
        """
        The game samples the inputs once per frame, so releases are applied one frame
        late: a quick tap whose press and release arrive in the same frame would
        otherwise never be seen by the CPU.
        """
        mapping = self.KEY_MAP.get(key)
        if not mapping:
            return
        port, bit = mapping
        if pressed:
            setattr(self, port, getattr(self, port) & ~bit & 0xFF)
            self._pending_releases = [m for m in self._pending_releases if m != mapping]
        else:
            self._pending_releases.append(mapping)

    def apply_pending_releases(self):
        for port, bit in self._pending_releases:
            setattr(self, port, getattr(self, port) | bit)
        self._pending_releases = []

    # --- To implement ---

    def load(self, roms_root: str) -> bool:
        """Load and decode the ROMs, build the machine. False if ROMs are missing."""
        raise NotImplementedError

    def run_frame(self) -> Optional[np.ndarray]:
        """Emulate one video frame. Returns audio samples when sound runs in-process."""
        raise NotImplementedError

    def render(self) -> np.ndarray:
        """The current frame as a (screen_height, screen_width, 3) uint8 array."""
        raise NotImplementedError

    def create_hd_renderer(self, ctx, hdpack_root: str):
        """Optional GPU renderer drawing the game from an HD pack (an object with
        draw(viewport, scanlines, samples)), or None when the driver or pack has none."""
        return None

    def sound_board(self) -> Optional[Tuple[str, str, tuple]]:
        """(module, class name, constructor args) of the sound board, or None."""
        return None

    def shutdown(self):
        pass
