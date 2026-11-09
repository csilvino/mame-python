"""
General Instrument AY-3-8910 PSG (MAME: src/devices/sound/ay8910.cpp)

Register writes are logged with a timestamp from the `now` callback, and the audio is
rendered afterwards with numpy, one segment per register write. Timestamps must be in
units of the chip's input clock (the tone counter advances once every 8 clocks).
"""

import numpy as np

# AY-3-8910 output levels, normalized (measured, as used by most emulators)
AY_LEVELS = np.array([0.0, 0.0100, 0.0145, 0.0211, 0.0307, 0.0455, 0.0644, 0.1073,
                      0.1265, 0.2049, 0.2922, 0.3728, 0.4925, 0.6353, 0.8055, 1.0])

OVERSAMPLE = 4


def _build_lfsr():
    """17-bit AY noise generator sequence (period 131071): new bit16 = bit0 ^ bit3."""
    n = 131071
    out = np.empty(n, dtype=np.uint8)
    rng = 1
    for i in range(n):
        out[i] = rng & 1
        rng = (rng >> 1) | ((((rng & 1) ^ ((rng >> 3) & 1))) << 16)
    return out


_LFSR = None


class AY8910:
    """AY-3-8910 PSG. Register writes are logged with their CPU time for rendering."""

    REG_MASK = [0xFF, 0x0F, 0xFF, 0x0F, 0xFF, 0x0F, 0x1F, 0xFF,
                0x1F, 0x1F, 0x1F, 0xFF, 0xFF, 0x0F, 0xFF, 0xFF]

    def __init__(self, now, port_a_read=None, port_a_write=None, port_b_write=None):
        global _LFSR
        if _LFSR is None:
            _LFSR = _build_lfsr()
        self.now = now
        self.port_a_read = port_a_read
        self.port_a_write = port_a_write
        self.port_b_write = port_b_write
        self.regs = [0] * 16
        self.latch = 0
        self.events = []          # (time, reg, value)

        # Rendering state
        self.rregs = [0] * 16
        self.tone_pos = [0.0, 0.0, 0.0]
        self.noise_pos = 0.0
        self.env_pos = 0.0
        self.env_attack = 0
        self.env_hold = True
        self.env_alt = False

    # --- Bus interface ---

    def address_w(self, value: int):
        self.latch = value & 0x0F

    def data_w(self, value: int):
        r = self.latch
        value &= self.REG_MASK[r]
        old_enable = self.regs[7]
        self.regs[r] = value
        self.events.append((self.now(), r, value))

        if r == 7:
            if (value ^ old_enable) & 0x40 and self.port_a_write:
                self.port_a_write(self.regs[14] if value & 0x40 else 0xFF)
            if (value ^ old_enable) & 0x80 and self.port_b_write:
                self.port_b_write(self.regs[15] if value & 0x80 else 0xFF)
        elif r == 14 and (self.regs[7] & 0x40) and self.port_a_write:
            self.port_a_write(value)
        elif r == 15 and (self.regs[7] & 0x80) and self.port_b_write:
            self.port_b_write(value)

    def data_r(self) -> int:
        r = self.latch
        if r == 14 and self.port_a_read:
            self.regs[14] = self.port_a_read()
        return self.regs[r]

    # --- Rendering ---

    def _apply(self, r: int, v: int):
        self.rregs[r] = v
        if r == 13:
            self.env_pos = 0.0
            self.env_attack = 0x0F if v & 0x04 else 0
            if not (v & 0x08):
                self.env_hold = True
                self.env_alt = bool(self.env_attack)
            else:
                self.env_hold = bool(v & 0x01)
                self.env_alt = bool(v & 0x02)

    def _env_volume(self, steps: np.ndarray) -> np.ndarray:
        cyc = steps >> 4
        level = 15 - (steps & 15)
        if self.env_hold:
            final = (self.env_attack ^ 15) if self.env_alt else self.env_attack
            vol = np.where(cyc >= 1, final, level ^ self.env_attack)
        else:
            toggle = (cyc & 1) * 15 if self.env_alt else 0
            vol = level ^ (self.env_attack ^ toggle)
        return vol

    def _segment(self, taus: np.ndarray, duration: float) -> np.ndarray:
        """Render subsamples at times `taus` (AY clocks from segment start), then advance state."""
        rr = self.rregs
        out = np.zeros(len(taus))

        noise_period = max(1, rr[6])
        noise_bits = _LFSR[(np.floor(self.noise_pos + taus / (16 * noise_period)).astype(np.int64)) % 131071]
        env_period = max(1, rr[11] | (rr[12] << 8))
        env_steps = np.floor(self.env_pos + taus / (16 * env_period)).astype(np.int64)
        env_vol = None

        enable = rr[7]
        for ch in range(3):
            period = max(1, rr[ch * 2] | (rr[ch * 2 + 1] << 8))
            vol_reg = rr[8 + ch]
            if vol_reg & 0x10:
                if env_vol is None:
                    env_vol = AY_LEVELS[self._env_volume(env_steps)]
                amp = env_vol
            else:
                amp = AY_LEVELS[vol_reg & 0x0F]
                if amp == 0.0:
                    self.tone_pos[ch] = (self.tone_pos[ch] + duration / (8 * period)) % 2.0
                    continue

            if enable & (1 << ch):
                tone = 1
            else:
                tone = (np.floor(self.tone_pos[ch] + taus / (8 * period)).astype(np.int64) & 1)
            if enable & (8 << ch):
                noise = 1
            else:
                noise = noise_bits
            out += (tone & noise) * amp
            self.tone_pos[ch] = (self.tone_pos[ch] + duration / (8 * period)) % 2.0

        self.noise_pos = (self.noise_pos + duration / (16 * noise_period)) % 131071
        self.env_pos += duration / (16 * env_period)
        if self.env_hold and self.env_pos > 64:
            self.env_pos = 64.0      # holding: keep the counter bounded
        elif self.env_pos > 1 << 20:
            self.env_pos %= 32
        return out / 3.0

    def render(self, t0: float, t1: float, n: int) -> np.ndarray:
        """n output samples (oversampled internally) covering CPU time [t0, t1)."""
        dt = (t1 - t0) / (n * OVERSAMPLE)
        times = t0 + (np.arange(n * OVERSAMPLE) + 0.5) * dt
        out = np.empty(n * OVERSAMPLE)

        events = self.events
        self.events = []
        start_idx = 0
        seg_t = t0
        for (t, r, v) in events + [(t1, None, None)]:
            t = min(max(t, t0), t1)
            end_idx = int(np.searchsorted(times, t))
            if end_idx > start_idx or t > seg_t:
                taus = times[start_idx:end_idx] - seg_t
                out[start_idx:end_idx] = self._segment(taus, t - seg_t)
                start_idx = end_idx
                seg_t = t
            if r is not None:
                self._apply(r, v)
        return out.reshape(n, OVERSAMPLE).mean(axis=1)
