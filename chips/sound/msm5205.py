"""
OKI MSM5205 ADPCM speech synthesizer (MAME: src/devices/sound/msm5205.cpp)

The host feeds 4-bit (or 3-bit) samples; on every VCK cycle the chip decodes one.
VCK = clock / prescaler (96, 48 or 64), and the host is told the period so that it can
schedule the decode and any interrupt the VCK line drives.
"""

import numpy as np


class MSM5205:
    """MSM5205 ADPCM decoder (MAME msm5205.cpp)."""

    INDEX_SHIFT = [-1, -1, -1, -1, 2, 4, 6, 8]
    PRESCALERS = [96, 48, 64, 0]       # S1/S2 select: 4 kHz, 8 kHz, 6 kHz, slave

    def __init__(self, clock: float = 384000, time_clock: float = 384000):
        """clock: the chip's oscillator. time_clock: rate of the timestamps the host uses."""
        self.clock = clock
        self.time_clock = time_clock
        nbl2bit = [[1, 0, 0, 0], [1, 0, 0, 1], [1, 0, 1, 0], [1, 0, 1, 1],
                   [1, 1, 0, 0], [1, 1, 0, 1], [1, 1, 1, 0], [1, 1, 1, 1],
                   [-1, 0, 0, 0], [-1, 0, 0, 1], [-1, 0, 1, 0], [-1, 0, 1, 1],
                   [-1, 1, 0, 0], [-1, 1, 0, 1], [-1, 1, 1, 0], [-1, 1, 1, 1]]
        self.diff = []
        for step in range(49):
            sv = int(16.0 * (11.0 / 10.0) ** step)
            for b in nbl2bit:
                self.diff.append(b[0] * (sv * b[1] + sv // 2 * b[2] + sv // 4 * b[3] + sv // 8))
        self.data = 0
        self.signal = 0
        self.step = 0
        self.reset = False
        self.bitwidth = 4
        self.select = 0                # S96
        self.events = []               # (time, signal)
        self.last_signal = 0           # signal at the start of the next render

    def playmode_w(self, mode: int):
        """Returns True if the VCK rate changed."""
        sel = mode & 3
        self.bitwidth = 4 if mode & 4 else 3
        changed = sel != self.select
        self.select = sel
        return changed

    @property
    def period(self) -> float:
        """VCK period in host time units (0 = slave mode, no VCK)."""
        p = self.PRESCALERS[self.select]
        return p * self.time_clock / self.clock if p else 0.0

    def data_w(self, value: int):
        self.data = value & 0x0F if self.bitwidth == 4 else (value & 0x07) << 1

    def update(self, time: float):
        """Decode one sample (VCK falling edge)."""
        if self.reset:
            new = 0
            self.step = 0
        else:
            v = self.data
            new = self.signal + self.diff[self.step * 16 + (v & 15)]
            new = max(-2048, min(2047, new))
            self.step = max(0, min(48, self.step + self.INDEX_SHIFT[v & 7]))
        if new != self.signal:
            self.signal = new
            self.events.append((time, new))

    def render(self, t0: float, t1: float, n: int) -> np.ndarray:
        out = np.full(n, self.last_signal, dtype=np.float64)
        if self.events:
            times = t0 + (np.arange(n) + 0.5) * ((t1 - t0) / n)
            for t, s in self.events:
                out[np.searchsorted(times, t):] = s
            self.last_signal = self.events[-1][1]
            self.events = []
        return out / 2048.0
