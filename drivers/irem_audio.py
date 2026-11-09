"""
Irem M52 "SoundC" audio board (MAME: src/mame/irem/irem.cpp, m52_soundc_audio_device)

Used by Moon Patrol and other Irem games of the era.
  - MC6803 @ 3.579545 MHz (E clock 894886 Hz), program ROM mapped at 0x7000
  - 2x AY-3-8910 @ 894886 Hz ("45M": port A = sound latch, port B = MSM5205 control;
    "45L": port A = analog drums, not present on the SoundC board)
  - 1x MSM5205 ADPCM @ 384 kHz; its VCK output drives the 6803 NMI
  - The main CPU writes commands to a latch; bit 7 = 0 asserts the 6803 IRQ

Board interface used by emu.sound_process: cmd_w(value, frame_fraction) and
run_frame(seconds, rate_scale) -> int16 samples.
"""

import math
import numpy as np
from chips.cpu.m6803 import M6803
from chips.sound.ay8910 import AY8910
from chips.sound.msm5205 import MSM5205

CPU_CLOCK = 3579545 / 4          # 6803 E clock, also the AY clock (timestamps are in E clocks)
MSM_CLOCK = 384000


class IremM52Sound:
    """The complete sound board. Feed it commands, get audio per frame."""

    def __init__(self, rom: bytes, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.rom = bytearray(0x8000)
        self.rom[0x7000:0x7000 + len(rom)] = rom

        self.port1 = 0
        self.port2 = 0
        self.soundlatch = 0

        self.cpu = M6803(self._read, self._write,
                         self._port1_r, self._port1_w, self._port2_r, self._port2_w)
        self.cpu.code_mem = bytes(self.rom) * 2        # 0x8000-0xFFFF mirrors the ROM space
        self.cpu.code_base = 0xA000
        # Spin loops "LDAA dir / BEQ *-2" (wait for the NMI handler), skipped in run_frame
        self.spin_loops = {0x8000 + a for a in range(0x2000, 0x7FFC)
                           if self.rom[a] == 0x96 and self.rom[a + 2] == 0x27 and self.rom[a + 3] == 0xFC}
        now = lambda: self.cpu.cycles
        self.ay_m = AY8910(now, port_a_read=lambda: self.soundlatch, port_b_write=self._ay_m_portb_w)
        self.ay_l = AY8910(now)
        self.msm = MSM5205(MSM_CLOCK, CPU_CLOCK)

        self.next_vck = self.msm.period
        self.next_decode = None
        self.commands = []             # (fraction of frame, value)
        self.sample_frac = 0.0

        # Output filters
        self.lp_state = 0.0
        self.dc_x = 0.0
        self.dc_y = 0.0

        self.cpu.reset()
        self.cpu.irq_line = True       # MAME asserts the IRQ on reset

    # --- Main CPU side ---

    def cmd_w(self, value: int, frame_fraction: float = 0.0):
        self.commands.append((frame_fraction, value))

    def _latch_command(self, value: int):
        self.soundlatch = value
        if not (value & 0x80):
            self.cpu.irq_line = True

    # --- 6803 memory map (global mask 0x7FFF) ---

    def _read(self, addr: int) -> int:
        a = addr & 0x7FFF
        if a >= 0x2000:
            return self.rom[a]
        return 0

    def _write(self, addr: int, value: int):
        a = addr & 0x7FFF
        if a < 0x1000:
            if a & 1:
                self.msm.data_w(value)
        elif a < 0x2000:
            if self.soundlatch & 0x80:   # IRQ acknowledge
                self.cpu.irq_line = False

    # --- 6803 ports ---

    def _port1_r(self) -> int:
        if self.port2 & 0x08:
            return self.ay_m.data_r()
        if self.port2 & 0x10:
            return self.ay_l.data_r()
        return 0xFF

    def _port1_w(self, value: int):
        self.port1 = value

    def _port2_r(self) -> int:
        return 0

    def _port2_w(self, value: int):
        if (self.port2 & 0x01) and not (value & 0x01):    # write strobe (falling edge)
            if self.port2 & 0x04:                           # address
                if self.port2 & 0x08: self.ay_m.address_w(self.port1)
                if self.port2 & 0x10: self.ay_l.address_w(self.port1)
            else:                                           # data
                if self.port2 & 0x08: self.ay_m.data_w(self.port1)
                if self.port2 & 0x10: self.ay_l.data_w(self.port1)
        self.port2 = value

    def _ay_m_portb_w(self, value: int):
        # bits 2-4: MSM5205 rate & 3/4 bit mode; bit 0: MSM5205 reset
        if self.msm.playmode_w((value >> 2) & 7):
            period = self.msm.period
            self.next_vck = self.cpu.cycles + period if period else None
        self.msm.reset = bool(value & 1)

    # --- Frame execution ---

    def run_frame(self, seconds: float, rate_scale: float = 1.0) -> np.ndarray:
        """Run `seconds` of board time (one video frame); returns int16 mono samples.
        rate_scale slightly stretches the sample count (dynamic rate control)."""
        frame_cycles = seconds * CPU_CLOCK
        cpu = self.cpu
        t0 = float(cpu.cycles)
        t1 = t0 + frame_cycles
        cmds = sorted(self.commands)
        self.commands = []
        cmd_idx = 0
        spin_loops = self.spin_loops

        while cpu.cycles < t1:
            # Next scheduled event
            nxt = t1
            if cmd_idx < len(cmds):
                nxt = min(nxt, t0 + cmds[cmd_idx][0] * frame_cycles)
            if self.next_vck is not None:
                nxt = min(nxt, self.next_vck)
            if self.next_decode is not None:
                nxt = min(nxt, self.next_decode)

            if cpu.waiting and not cpu.nmi_pending and not (cpu.irq_line and not (cpu.cc & 0x10)):
                cpu.cycles = max(cpu.cycles, int(nxt))   # idle until something happens
            else:
                while cpu.cycles < nxt:
                    # Spin loop waiting for the NMI handler: skip to the next event
                    if (cpu.pc in spin_loops and cpu.read(cpu.code_mem[cpu.pc + 1]) == 0 and not cpu.nmi_pending
                            and not (cpu.irq_line and not (cpu.cc & 0x10))):
                        loops = max(1, math.ceil((nxt - cpu.cycles) / 6))   # 3 + 3 cycles per iteration
                        cpu.cycles += loops * 6
                        cpu.a = 0
                        cpu.cc = (cpu.cc & 0xF1) | 0x04
                        break
                    cpu.step()

            now = cpu.cycles
            while cmd_idx < len(cmds) and t0 + cmds[cmd_idx][0] * frame_cycles <= now:
                self._latch_command(cmds[cmd_idx][1])
                cmd_idx += 1
            if self.next_vck is not None and self.next_vck <= now:
                cpu.set_nmi()                                   # VCK rising edge
                self.next_decode = self.next_vck + self.msm.period / 2
                self.next_vck += self.msm.period
            if self.next_decode is not None and self.next_decode <= now:
                self.msm.update(self.next_decode)
                self.next_decode = None

        for i in range(cmd_idx, len(cmds)):        # commands that landed exactly at t1
            self._latch_command(cmds[i][1])

        # Samples for this frame
        exact = frame_cycles * self.sample_rate / CPU_CLOCK * rate_scale + self.sample_frac
        n = int(exact)
        self.sample_frac = exact - n
        if n <= 0:
            return np.zeros(0, dtype=np.int16)
        t_end = float(cpu.cycles)
        ay = (self.ay_m.render(t0, t_end, n) + self.ay_l.render(t0, t_end, n)) * 0.5
        msm = self._lowpass(self.msm.render(t0, t_end, n))
        mix = ay * 0.7 + msm * 0.3
        mix = self._dc_block(mix)
        return np.clip(mix * 26000, -32768, 32767).astype(np.int16)

    def _lowpass(self, x: np.ndarray) -> np.ndarray:
        """One-pole low pass (~1.3 kHz, the Sallen-Key on the MSM5205 path)."""
        alpha = 1.0 - np.exp(-2 * np.pi * 1300 / self.sample_rate)
        y = np.empty_like(x)
        s = self.lp_state
        for i in range(len(x)):
            s += alpha * (x[i] - s)
            y[i] = s
        self.lp_state = s
        return y

    def _dc_block(self, x: np.ndarray) -> np.ndarray:
        r = 0.995
        y = np.empty_like(x)
        px, py = self.dc_x, self.dc_y
        for i in range(len(x)):
            py = x[i] - px + r * py
            px = x[i]
            y[i] = py
        self.dc_x, self.dc_y = px, py
        return y
