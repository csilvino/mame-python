"""
Motorola MC6803 CPU (6800 instruction set + 6801 extensions)
Instruction semantics and cycle counts follow MAME's m6800/m6801 cores.
Used by the Irem M52 sound board.
"""

from typing import Callable

# Condition code bits
C_FLAG = 0x01
V_FLAG = 0x02
Z_FLAG = 0x04
N_FLAG = 0x08
I_FLAG = 0x10
H_FLAG = 0x20

# Cycle counts (MAME cycles_6803; illegal opcodes = 4)
_XX = 4
CYCLES = [
    _XX, 2, _XX, _XX, 3, 3, 2, 2, 3, 3, 2, 2, 2, 2, 2, 2,
    2, 2, _XX, _XX, _XX, _XX, 2, 2, _XX, 2, _XX, 2, _XX, _XX, _XX, _XX,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 4, 4, 3, 3, 3, 3, 5, 5, 3, 10, 4, 10, 9, 12,
    2, _XX, _XX, 2, 2, _XX, 2, 2, 2, 2, 2, _XX, 2, 2, _XX, 2,
    2, _XX, _XX, 2, 2, _XX, 2, 2, 2, 2, 2, _XX, 2, 2, _XX, 2,
    6, _XX, _XX, 6, 6, _XX, 6, 6, 6, 6, 6, _XX, 6, 6, 3, 6,
    6, _XX, _XX, 6, 6, _XX, 6, 6, 6, 6, 6, _XX, 6, 6, 3, 6,
    2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 4, 6, 3, 3,
    3, 3, 3, 5, 3, 3, 3, 3, 3, 3, 3, 3, 5, 5, 4, 4,
    4, 4, 4, 6, 4, 4, 4, 4, 4, 4, 4, 4, 6, 6, 5, 5,
    4, 4, 4, 6, 4, 4, 4, 4, 4, 4, 4, 4, 6, 6, 5, 5,
    2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 3, _XX, 3, 3,
    3, 3, 3, 5, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4,
    4, 4, 4, 6, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5,
    4, 4, 4, 6, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5,
]


def _nz8(r: int) -> int:
    return (N_FLAG if r & 0x80 else 0) | (0 if r & 0xFF else Z_FLAG)


def _nz16(r: int) -> int:
    return (N_FLAG if r & 0x8000 else 0) | (0 if r & 0xFFFF else Z_FLAG)


def _flags8(a: int, b: int, r: int) -> int:
    """N, Z, V, C for an 8-bit add/sub; r is the unmasked (>= 16 bit) result."""
    return (_nz8(r) | ((((a ^ b ^ r ^ (r >> 1)) & 0x80) >> 6)) | ((r & 0x100) >> 8))


def _flags16(a: int, b: int, r: int) -> int:
    return (_nz16(r) | ((((a ^ b ^ r ^ (r >> 1)) & 0x8000) >> 14)) | ((r & 0x10000) >> 16))


class M6803:
    """
    MC6803 core. The owner supplies memory access callbacks; the internal
    registers (0x00-0x1F) and internal RAM (0x80-0xFF) are handled here.
    """

    def __init__(self, read: Callable[[int], int], write: Callable[[int, int], None],
                 port1_r: Callable[[], int], port1_w: Callable[[int], None],
                 port2_r: Callable[[], int], port2_w: Callable[[int], None]):
        self.ext_read = read
        self.ext_write = write
        self.port1_r = port1_r
        self.port1_w = port1_w
        self.port2_r = port2_r
        self.port2_w = port2_w

        self.a = self.b = 0
        self.x = 0
        self.sp = 0
        self.pc = 0
        self.cc = 0xC0 | I_FLAG
        self.cycles = 0

        self.iram = bytearray(128)       # 0x80-0xFF
        self.regs = bytearray(32)        # other internal registers (storage only)
        self.ddr = [0, 0]
        self.port_data = [0, 0]
        self.port2_written = False
        self.tcsr = 0
        self.ocr = 0xFFFF

        self.irq_line = False
        self.nmi_pending = False
        self.waiting = False             # WAI executed, waiting for an interrupt

        # Optional fast path for opcode fetches: code_mem[addr] for addr >= code_base (read-only ROM)
        self.code_mem = None
        self.code_base = 0x10000

        self.table = self._build_table()

    # --- Memory ---

    def read(self, addr: int) -> int:
        if addr < 0x100:
            if addr >= 0x80:
                return self.iram[addr - 0x80]
            if addr < 0x20:
                return self._io_read(addr)
        return self.ext_read(addr)

    def write(self, addr: int, value: int):
        if addr < 0x100:
            if addr >= 0x80:
                self.iram[addr - 0x80] = value
                return
            if addr < 0x20:
                self._io_write(addr, value)
                return
        self.ext_write(addr, value)

    def read16(self, addr: int) -> int:
        return (self.read(addr) << 8) | self.read((addr + 1) & 0xFFFF)

    def write16(self, addr: int, value: int):
        self.write(addr, (value >> 8) & 0xFF)
        self.write((addr + 1) & 0xFFFF, value & 0xFF)

    def fetch(self) -> int:
        pc = self.pc
        v = self.code_mem[pc] if pc >= self.code_base else self.read(pc)
        self.pc = (pc + 1) & 0xFFFF
        return v

    def fetch16(self) -> int:
        pc = self.pc
        if self.code_base <= pc < 0xFFFF:
            v = (self.code_mem[pc] << 8) | self.code_mem[pc + 1]
        else:
            v = self.read16(pc)
        self.pc = (pc + 2) & 0xFFFF
        return v

    # --- Internal I/O registers ---

    def _io_read(self, addr: int) -> int:
        if addr == 0x00: return self.ddr[0]
        if addr == 0x01: return self.ddr[1]
        if addr == 0x02:
            ddr = self.ddr[0]
            if ddr == 0xFF: return self.port_data[0]
            return (self.port1_r() & ~ddr & 0xFF) | (self.port_data[0] & ddr)
        if addr == 0x03:
            ddr = self.ddr[1]
            if ddr == 0xFF: return self.port_data[1]
            return (self.port2_r() & ~ddr & 0xFF) | (self.port_data[1] & ddr)
        if addr == 0x08: return self.tcsr
        if addr == 0x09: return (self.cycles >> 8) & 0xFF   # free running counter
        if addr == 0x0A: return self.cycles & 0xFF
        if addr == 0x0B: return self.ocr >> 8
        if addr == 0x0C: return self.ocr & 0xFF
        if addr == 0x11: return 0x20                         # TRCSR: TDRE
        return self.regs[addr]

    def _io_write(self, addr: int, value: int):
        if addr == 0x00:
            if self.ddr[0] != value:
                self.ddr[0] = value
                self._write_port1()
        elif addr == 0x01:
            if self.ddr[1] != value:
                self.ddr[1] = value
                self._write_port2()
        elif addr == 0x02:
            self.port_data[0] = value
            self._write_port1()
        elif addr == 0x03:
            self.port_data[1] = value
            self.port2_written = True
            self._write_port2()
        elif addr == 0x08:
            self.tcsr = (self.tcsr & 0xE0) | (value & 0x1F)
        elif addr == 0x0B:
            self.ocr = (value << 8) | (self.ocr & 0xFF)
        elif addr == 0x0C:
            self.ocr = (self.ocr & 0xFF00) | value
        else:
            self.regs[addr] = value

    def _write_port1(self):
        ddr = self.ddr[0]
        self.port1_w((self.port_data[0] & ddr) | (ddr ^ 0xFF))

    def _write_port2(self):
        if not self.port2_written:
            return
        data = self.port_data[1]
        ddr = self.ddr[1] & 0x1F
        if ddr != 0x1F and ddr:
            data = (data & ddr) | (ddr ^ 0xFF)
        self.port2_w(data & 0x1F)

    # --- Stack ---

    def push8(self, v: int):
        self.write(self.sp, v)
        self.sp = (self.sp - 1) & 0xFFFF

    def push16(self, v: int):
        self.push8(v & 0xFF)
        self.push8(v >> 8)

    def pull8(self) -> int:
        self.sp = (self.sp + 1) & 0xFFFF
        return self.read(self.sp)

    def pull16(self) -> int:
        hi = self.pull8()
        return (hi << 8) | self.pull8()

    # --- Control ---

    def reset(self):
        self.cc = 0xC0 | I_FLAG
        self.ddr = [0, 0]
        self.port2_written = False
        self.tcsr = 0
        self.ocr = 0xFFFF
        self.waiting = False
        self.nmi_pending = False
        self.pc = self.read16(0xFFFE)

    def set_nmi(self):
        """Edge-triggered NMI request."""
        self.nmi_pending = True

    def _enter_interrupt(self, vector: int):
        if self.waiting:
            self.waiting = False
            self.cycles += 4
        else:
            self.push16(self.pc)
            self.push16(self.x)
            self.push8(self.a)
            self.push8(self.b)
            self.push8(self.cc)
            self.cycles += 12
        self.cc |= I_FLAG
        self.pc = self.read16(vector)

    def step(self):
        """Service pending interrupts, then execute one instruction."""
        if self.nmi_pending:
            self.nmi_pending = False
            self._enter_interrupt(0xFFFC)
        elif self.irq_line and not (self.cc & I_FLAG):
            self._enter_interrupt(0xFFF8)

        if self.waiting:
            self.cycles += 1
            return

        op = self.fetch()
        self.cycles += CYCLES[op]
        self.table[op]()

    def run(self, target_cycles: int):
        while self.cycles < target_cycles:
            self.step()

    # --- Opcode table ---

    def _build_table(self):
        t = [self._illegal] * 256
        s = self

        # Inherent / misc
        t[0x01] = lambda: None                                           # NOP
        t[0x04] = s._lsrd
        t[0x05] = s._asld
        t[0x06] = s._tap
        t[0x07] = s._tpa
        t[0x08] = s._inx
        t[0x09] = s._dex
        t[0x0A] = lambda: setattr(s, 'cc', s.cc & ~V_FLAG)               # CLV
        t[0x0B] = lambda: setattr(s, 'cc', s.cc | V_FLAG)                # SEV
        t[0x0C] = lambda: setattr(s, 'cc', s.cc & ~C_FLAG)               # CLC
        t[0x0D] = lambda: setattr(s, 'cc', s.cc | C_FLAG)                # SEC
        t[0x0E] = lambda: setattr(s, 'cc', s.cc & ~I_FLAG)               # CLI
        t[0x0F] = lambda: setattr(s, 'cc', s.cc | I_FLAG)                # SEI
        t[0x10] = s._sba
        t[0x11] = s._cba
        t[0x16] = s._tab
        t[0x17] = s._tba
        t[0x19] = s._daa
        t[0x1B] = s._aba

        # Branches
        conds = [
            lambda c: True,                                              # BRA
            lambda c: False,                                             # BRN
            lambda c: not (c & (C_FLAG | Z_FLAG)),                       # BHI
            lambda c: bool(c & (C_FLAG | Z_FLAG)),                       # BLS
            lambda c: not (c & C_FLAG),                                  # BCC
            lambda c: bool(c & C_FLAG),                                  # BCS
            lambda c: not (c & Z_FLAG),                                  # BNE
            lambda c: bool(c & Z_FLAG),                                  # BEQ
            lambda c: not (c & V_FLAG),                                  # BVC
            lambda c: bool(c & V_FLAG),                                  # BVS
            lambda c: not (c & N_FLAG),                                  # BPL
            lambda c: bool(c & N_FLAG),                                  # BMI
            lambda c: not ((c ^ (c << 2)) & N_FLAG),                     # BGE
            lambda c: bool((c ^ (c << 2)) & N_FLAG),                     # BLT
            lambda c: not (((c ^ (c << 2)) & N_FLAG) or (c & Z_FLAG)),   # BGT
            lambda c: bool(((c ^ (c << 2)) & N_FLAG) or (c & Z_FLAG)),   # BLE
        ]
        for i, cond in enumerate(conds):
            t[0x20 + i] = s._make_branch(cond)

        t[0x30] = lambda: setattr(s, 'x', (s.sp + 1) & 0xFFFF)          # TSX
        t[0x31] = lambda: setattr(s, 'sp', (s.sp + 1) & 0xFFFF)         # INS
        t[0x32] = lambda: setattr(s, 'a', s.pull8())                     # PULA
        t[0x33] = lambda: setattr(s, 'b', s.pull8())                     # PULB
        t[0x34] = lambda: setattr(s, 'sp', (s.sp - 1) & 0xFFFF)         # DES
        t[0x35] = lambda: setattr(s, 'sp', (s.x - 1) & 0xFFFF)          # TXS
        t[0x36] = lambda: s.push8(s.a)                                   # PSHA
        t[0x37] = lambda: s.push8(s.b)                                   # PSHB
        t[0x38] = lambda: setattr(s, 'x', s.pull16())                    # PULX
        t[0x39] = lambda: setattr(s, 'pc', s.pull16())                   # RTS
        t[0x3A] = lambda: setattr(s, 'x', (s.x + s.b) & 0xFFFF)         # ABX
        t[0x3B] = s._rti
        t[0x3C] = lambda: s.push16(s.x)                                  # PSHX
        t[0x3D] = s._mul
        t[0x3E] = s._wai
        t[0x3F] = s._swi

        # Read-modify-write group: 0x40 = A, 0x50 = B, 0x60 = indexed, 0x70 = extended
        rmw = {0x0: s._neg, 0x3: s._com, 0x4: s._lsr, 0x6: s._ror, 0x7: s._asr,
               0x8: s._asl, 0x9: s._rol, 0xA: s._dec, 0xC: s._inc, 0xD: s._tst, 0xF: s._clr}
        for low, fn in rmw.items():
            t[0x40 + low] = s._make_rmw_reg('a', fn)
            t[0x50 + low] = s._make_rmw_reg('b', fn)
            t[0x60 + low] = s._make_rmw_mem(s._ea_ix, fn, low)
            t[0x70 + low] = s._make_rmw_mem(s._ea_ex, fn, low)
        t[0x6E] = lambda: setattr(s, 'pc', s._ea_ix())                  # JMP ix
        t[0x7E] = lambda: setattr(s, 'pc', s._ea_ex())                  # JMP ex

        # ALU group: 0x80-0xBF (A), 0xC0-0xFF (B)
        modes = [s._ea_im8, s._ea_di, s._ea_ix, s._ea_ex]
        modes16 = [s._ea_im16, s._ea_di, s._ea_ix, s._ea_ex]
        alu = {0x0: s._sub, 0x1: s._cmp, 0x2: s._sbc, 0x4: s._and, 0x5: s._bit,
               0x6: s._ld, 0x8: s._eor, 0x9: s._adc, 0xA: s._or, 0xB: s._add}
        for m in range(4):
            for low, fn in alu.items():
                t[0x80 + m * 0x10 + low] = s._make_alu('a', fn, modes[m])
                t[0xC0 + m * 0x10 + low] = s._make_alu('b', fn, modes[m])
            t[0x87 + m * 0x10] = s._make_st8('a', modes[m])
            t[0xC7 + m * 0x10] = s._make_st8('b', modes[m])
            t[0x83 + m * 0x10] = s._make_d16(s._subd, modes16[m])
            t[0xC3 + m * 0x10] = s._make_d16(s._addd, modes16[m])
            t[0x8C + m * 0x10] = s._make_d16(s._cpx, modes16[m])
            t[0xCC + m * 0x10] = s._make_d16(s._ldd, modes16[m])
            t[0x8E + m * 0x10] = s._make_d16(s._lds, modes16[m])
            t[0xCE + m * 0x10] = s._make_d16(s._ldx, modes16[m])
            t[0x8F + m * 0x10] = s._make_st16(lambda: s.sp, modes16[m])
            t[0xCF + m * 0x10] = s._make_st16(lambda: s.x, modes16[m])
            t[0xCD + m * 0x10] = s._make_st16(lambda: (s.a << 8) | s.b, modes16[m])
        t[0x8D] = s._bsr
        t[0x9D] = s._make_jsr(s._ea_di)
        t[0xAD] = s._make_jsr(s._ea_ix)
        t[0xBD] = s._make_jsr(s._ea_ex)
        return t

    # --- Addressing modes (return effective address) ---

    def _ea_im8(self) -> int:
        ea = self.pc
        self.pc = (self.pc + 1) & 0xFFFF
        return ea

    def _ea_im16(self) -> int:
        ea = self.pc
        self.pc = (self.pc + 2) & 0xFFFF
        return ea

    def _ea_di(self) -> int:
        return self.fetch()

    def _ea_ix(self) -> int:
        return (self.x + self.fetch()) & 0xFFFF

    def _ea_ex(self) -> int:
        return self.fetch16()

    # --- Builders ---

    def _make_branch(self, cond):
        def op():
            off = self.fetch()
            if cond(self.cc):
                self.pc = (self.pc + (off - 256 if off & 0x80 else off)) & 0xFFFF
        return op

    def _make_rmw_reg(self, reg, fn):
        def op():
            r = fn(getattr(self, reg))
            if r is not None:
                setattr(self, reg, r)
        return op

    def _make_rmw_mem(self, mode, fn, low):
        if low == 0xF:   # CLR: dummy read like the real chip
            def op():
                ea = mode()
                self.read(ea)
                self.write(ea, fn(0))
            return op

        def op():
            ea = mode()
            r = fn(self.read(ea))
            if r is not None:
                self.write(ea, r)
        return op

    def _make_alu(self, reg, fn, mode):
        def op():
            r = fn(getattr(self, reg), self.read(mode()))
            if r is not None:
                setattr(self, reg, r)
        return op

    def _make_st8(self, reg, mode):
        def op():
            v = getattr(self, reg)
            self.cc = (self.cc & 0xF1) | _nz8(v)
            self.write(mode(), v)
        return op

    def _make_d16(self, fn, mode):
        def op():
            fn(self.read16(mode()))
        return op

    def _make_st16(self, get, mode):
        def op():
            v = get()
            self.cc = (self.cc & 0xF1) | _nz16(v)
            self.write16(mode(), v)
        return op

    def _make_jsr(self, mode):
        def op():
            ea = mode()
            self.push16(self.pc)
            self.pc = ea
        return op

    # --- Read-modify-write ops (return new value, or None for TST) ---

    def _neg(self, t):
        r = (-t) & 0xFFFF
        self.cc = (self.cc & 0xF0) | _flags8(0, t, r)
        return r & 0xFF

    def _com(self, t):
        r = (~t) & 0xFF
        self.cc = (self.cc & 0xF1) | _nz8(r) | C_FLAG
        return r

    def _lsr(self, t):
        c = t & 1
        r = t >> 1
        self.cc = (self.cc & 0xF0) | c | (0 if r else Z_FLAG) | (V_FLAG if c else 0)
        return r

    def _ror(self, t):
        c = t & 1
        r = ((self.cc & C_FLAG) << 7) | (t >> 1)
        n = r & 0x80
        self.cc = (self.cc & 0xF0) | c | _nz8(r) | (V_FLAG if bool(n) != bool(c) else 0)
        return r

    def _asr(self, t):
        c = t & 1
        r = (t >> 1) | (t & 0x80)
        n = r & 0x80
        self.cc = (self.cc & 0xF0) | c | _nz8(r) | (V_FLAG if bool(n) != bool(c) else 0)
        return r

    def _asl(self, t):
        r = t << 1
        self.cc = (self.cc & 0xF0) | _flags8(t, t, r)
        return r & 0xFF

    def _rol(self, t):
        r = (t << 1) | (self.cc & C_FLAG)
        self.cc = (self.cc & 0xF0) | _flags8(t, t, r)
        return r & 0xFF

    def _dec(self, t):
        r = (t - 1) & 0xFF
        self.cc = (self.cc & 0xF1) | _nz8(r) | (V_FLAG if r == 0x7F else 0)
        return r

    def _inc(self, t):
        r = (t + 1) & 0xFF
        self.cc = (self.cc & 0xF1) | _nz8(r) | (V_FLAG if r == 0x80 else 0)
        return r

    def _tst(self, t):
        self.cc = (self.cc & 0xF0) | _nz8(t)
        return None

    def _clr(self, t):
        self.cc = (self.cc & 0xF0) | Z_FLAG
        return 0

    # --- ALU ops: (register value, operand) -> new value or None ---

    def _sub(self, a, t):
        r = (a - t) & 0xFFFF
        self.cc = (self.cc & 0xF0) | _flags8(a, t, r)
        return r & 0xFF

    def _cmp(self, a, t):
        r = (a - t) & 0xFFFF
        self.cc = (self.cc & 0xF0) | _flags8(a, t, r)
        return None

    def _sbc(self, a, t):
        r = (a - t - (self.cc & C_FLAG)) & 0xFFFF
        self.cc = (self.cc & 0xF0) | _flags8(a, t, r)
        return r & 0xFF

    def _and(self, a, t):
        r = a & t
        self.cc = (self.cc & 0xF1) | _nz8(r)
        return r

    def _bit(self, a, t):
        self.cc = (self.cc & 0xF1) | _nz8(a & t)
        return None

    def _ld(self, a, t):
        self.cc = (self.cc & 0xF1) | _nz8(t)
        return t

    def _eor(self, a, t):
        r = a ^ t
        self.cc = (self.cc & 0xF1) | _nz8(r)
        return r

    def _adc(self, a, t):
        r = a + t + (self.cc & C_FLAG)
        self.cc = (self.cc & 0xD0) | _flags8(a, t, r) | (((a ^ t ^ r) & 0x10) << 1)
        return r & 0xFF

    def _or(self, a, t):
        r = a | t
        self.cc = (self.cc & 0xF1) | _nz8(r)
        return r

    def _add(self, a, t):
        r = a + t
        self.cc = (self.cc & 0xD0) | _flags8(a, t, r) | (((a ^ t ^ r) & 0x10) << 1)
        return r & 0xFF

    # --- 16-bit ops ---

    def _subd(self, v):
        d = (self.a << 8) | self.b
        r = (d - v) & 0x3FFFF
        self.cc = (self.cc & 0xF0) | _flags16(d, v, r)
        self.a, self.b = (r >> 8) & 0xFF, r & 0xFF

    def _addd(self, v):
        d = (self.a << 8) | self.b
        r = d + v
        self.cc = (self.cc & 0xF0) | _flags16(d, v, r)
        self.a, self.b = (r >> 8) & 0xFF, r & 0xFF

    def _cpx(self, v):
        r = (self.x - v) & 0x3FFFF
        self.cc = (self.cc & 0xF0) | _flags16(self.x, v, r)

    def _ldd(self, v):
        self.a, self.b = v >> 8, v & 0xFF
        self.cc = (self.cc & 0xF1) | _nz16(v)

    def _lds(self, v):
        self.sp = v
        self.cc = (self.cc & 0xF1) | _nz16(v)

    def _ldx(self, v):
        self.x = v
        self.cc = (self.cc & 0xF1) | _nz16(v)

    # --- Inherent ops ---

    def _lsrd(self):
        d = (self.a << 8) | self.b
        c = d & 1
        r = d >> 1
        self.cc = (self.cc & 0xF0) | c | (0 if r else Z_FLAG) | (V_FLAG if c else 0)
        self.a, self.b = r >> 8, r & 0xFF

    def _asld(self):
        d = (self.a << 8) | self.b
        r = d << 1
        self.cc = (self.cc & 0xF0) | _flags16(d, d, r)
        self.a, self.b = (r >> 8) & 0xFF, r & 0xFF

    def _tap(self):
        self.cc = self.a

    def _tpa(self):
        self.a = self.cc

    def _inx(self):
        self.x = (self.x + 1) & 0xFFFF
        self.cc = (self.cc & ~Z_FLAG) | (0 if self.x else Z_FLAG)

    def _dex(self):
        self.x = (self.x - 1) & 0xFFFF
        self.cc = (self.cc & ~Z_FLAG) | (0 if self.x else Z_FLAG)

    def _sba(self):
        self.a = self._sub(self.a, self.b)

    def _cba(self):
        self._cmp(self.a, self.b)

    def _aba(self):
        self.a = self._add(self.a, self.b)

    def _tab(self):
        self.b = self.a
        self.cc = (self.cc & 0xF1) | _nz8(self.b)

    def _tba(self):
        self.a = self.b
        self.cc = (self.cc & 0xF1) | _nz8(self.a)

    def _daa(self):
        a = self.a
        msn, lsn = a & 0xF0, a & 0x0F
        cf = 0
        if lsn > 0x09 or self.cc & H_FLAG: cf |= 0x06
        if msn > 0x80 and lsn > 0x09: cf |= 0x60
        if msn > 0x90 or self.cc & C_FLAG: cf |= 0x60
        t = cf + a
        self.cc = (self.cc & 0xF1) | _nz8(t) | ((t & 0x100) >> 8)
        self.a = t & 0xFF

    def _mul(self):
        t = self.a * self.b
        self.cc = (self.cc & ~C_FLAG) | (C_FLAG if t & 0x80 else 0)
        self.a, self.b = t >> 8, t & 0xFF

    def _rti(self):
        self.cc = self.pull8()
        self.b = self.pull8()
        self.a = self.pull8()
        self.x = self.pull16()
        self.pc = self.pull16()

    def _wai(self):
        self.push16(self.pc)
        self.push16(self.x)
        self.push8(self.a)
        self.push8(self.b)
        self.push8(self.cc)
        self.waiting = True

    def _swi(self):
        self.push16(self.pc)
        self.push16(self.x)
        self.push8(self.a)
        self.push8(self.b)
        self.push8(self.cc)
        self.cc |= I_FLAG
        self.pc = self.read16(0xFFFA)

    def _bsr(self):
        off = self.fetch()
        self.push16(self.pc)
        self.pc = (self.pc + (off - 256 if off & 0x80 else off)) & 0xFFFF

    def _illegal(self):
        pass
