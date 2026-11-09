"""
Z80 CPU Emulator Core
This will simulate the Z80 processor.
"""

import ast
import inspect
import types
from typing import Dict, List, Optional

class Z80:
    """
    A basic implementation of the Z80 CPU.
    This class will hold the CPU's state (registers) and execute instructions.
    """
    # Z80 Flags
    FLAG_C = 0x01  # Carry
    FLAG_N = 0x02  # Add/Subtract
    FLAG_PV = 0x04 # Parity/Overflow
    FLAG_H = 0x10  # Half Carry
    FLAG_Z = 0x40  # Zero
    FLAG_S = 0x80  # Sign

    def __init__(self, memory: bytearray, debug_log: bool = False):
        self.memory = memory
        self.debug_log = debug_log

        # Registers
        self.a: int = 0
        self.f: int = 0
        self.b: int = 0
        self.c: int = 0
        self.d: int = 0
        self.e: int = 0
        self.h: int = 0
        self.l: int = 0
        self.pc: int = 0
        self.sp: int = 0
        self.ix: int = 0
        self.iy: int = 0
        self.i: int = 0
        self.r: int = 0
        self.iff1: bool = False
        self.iff2: bool = False
        self.im: int = 0
        self.halted: bool = False
        self.cycles: int = 0

        # Alternate registers
        self.a_alt: int = 0
        self.f_alt: int = 0
        self.b_alt: int = 0
        self.c_alt: int = 0
        self.d_alt: int = 0
        self.e_alt: int = 0
        self.h_alt: int = 0
        self.l_alt: int = 0

        self.parity_table = self._build_parity_table()
        
        # Debugging missing opcodes
        self.missing_opcodes = set()
        self.skip_missing_opcodes = True
        
        # Build opcode dispatch table for performance (256 methods for O(1) dispatch)
        self.opcode_table = [self._op_invalid] * 256
        self.cycle_table = self._build_cycle_table()
        self.ed_cycle_table = self._build_ed_cycle_table()
        self._build_opcode_table()
        
        print("Z80 CPU initialized.")

    def _read_byte(self, address: int) -> int:
        """Reads a byte from memory. Can be overridden for memory mapping."""
        return self.memory[address & 0xFFFF]

    def _write_byte(self, address: int, value: int):
        """Writes a byte to memory. Can be overridden for memory mapping."""
        self.memory[address & 0xFFFF] = value & 0xFF

    def _get_af(self) -> int: return (self.a << 8) | self.f
    def _get_hl(self) -> int: return (self.h << 8) | self.l
    def _set_hl(self, value: int):
        self.h = (value >> 8) & 0xFF
        self.l = value & 0xFF

    def _get_bc(self) -> int: return (self.b << 8) | self.c
    def _set_bc(self, value: int):
        self.b = (value >> 8) & 0xFF
        self.c = value & 0xFF

    def _get_de(self) -> int: return (self.d << 8) | self.e
    def _set_de(self, value: int):
        self.d = (value >> 8) & 0xFF
        self.e = value & 0xFF

    
    def _build_cycle_table(self) -> List[int]:
        """Official Z80 T-states per opcode (conditional instructions: not-taken value)."""
        return [
            4, 10,  7,  6,  4,  4,  7,  4,  4, 11,  7,  6,  4,  4,  7,  4,   # 00
            8, 10,  7,  6,  4,  4,  7,  4, 12, 11,  7,  6,  4,  4,  7,  4,   # 10
            7, 10, 16,  6,  4,  4,  7,  4,  7, 11, 16,  6,  4,  4,  7,  4,   # 20
            7, 10, 13,  6, 11, 11, 10,  4,  7, 11, 13,  6,  4,  4,  7,  4,   # 30
            4,  4,  4,  4,  4,  4,  7,  4,  4,  4,  4,  4,  4,  4,  7,  4,   # 40
            4,  4,  4,  4,  4,  4,  7,  4,  4,  4,  4,  4,  4,  4,  7,  4,   # 50
            4,  4,  4,  4,  4,  4,  7,  4,  4,  4,  4,  4,  4,  4,  7,  4,   # 60
            7,  7,  7,  7,  7,  7,  4,  7,  4,  4,  4,  4,  4,  4,  7,  4,   # 70
            4,  4,  4,  4,  4,  4,  7,  4,  4,  4,  4,  4,  4,  4,  7,  4,   # 80
            4,  4,  4,  4,  4,  4,  7,  4,  4,  4,  4,  4,  4,  4,  7,  4,   # 90
            4,  4,  4,  4,  4,  4,  7,  4,  4,  4,  4,  4,  4,  4,  7,  4,   # A0
            4,  4,  4,  4,  4,  4,  7,  4,  4,  4,  4,  4,  4,  4,  7,  4,   # B0
            5, 10, 10, 10, 10, 11,  7, 11,  5, 10, 10,  4, 10, 17,  7, 11,   # C0
            5, 10, 10, 11, 10, 11,  7, 11,  5,  4, 10, 11, 10,  4,  7, 11,   # D0
            5, 10, 10, 19, 10, 11,  7, 11,  5,  4, 10,  4, 10,  4,  7, 11,   # E0
            5, 10, 10,  4, 10, 11,  7, 11,  5,  6, 10,  4, 10,  4,  7, 11,   # F0
        ]

    @staticmethod
    def _build_ed_cycle_table() -> List[int]:
        """T-states of ED opcodes beyond the 4 already counted for the prefix.
        Block instructions (A0-BB) count their own cycles per iteration."""
        table = [4] * 256
        for op in range(0x40, 0x80):
            col, low = op & 0x07, op & 0x0F
            if col in (0, 1): table[op] = 8            # IN r,(C) / OUT (C),r
            elif low in (0x02, 0x0A): table[op] = 11   # SBC / ADC HL,rr
            elif low in (0x03, 0x0B): table[op] = 16   # LD (nn),rr / LD rr,(nn)
            elif col == 5: table[op] = 10              # RETN / RETI
            elif col == 4 or col == 6: table[op] = 4   # NEG / IM
        for op in (0x47, 0x4F, 0x57, 0x5F): table[op] = 5   # LD I,A / LD R,A / LD A,I / LD A,R
        for op in (0x67, 0x6F): table[op] = 14              # RRD / RLD
        for op in (0xA0, 0xA1, 0xA2, 0xA3, 0xA8, 0xA9, 0xAA, 0xAB,
                   0xB0, 0xB1, 0xB2, 0xB3, 0xB8, 0xB9, 0xBA, 0xBB):
            table[op] = 0
        return table

    def _build_parity_table(self) -> List[bool]:
        table = [False] * 256
        for i in range(256):
            parity = 0
            n = i
            while n > 0:
                parity += (n & 1)
                n >>= 1
            table[i] = (parity % 2 == 0)
        return table
    
    def _op_invalid(self):
        """Handler for unimplemented or invalid opcodes"""
        opcode = self.memory[(self.pc - 1) & 0xFFFF]
        if opcode not in self.missing_opcodes:
            self.missing_opcodes.add(opcode)
            if not self.skip_missing_opcodes:
                print(f"⚠️ MISSING OPCODE: {opcode:02X} at {self.pc-1:04X}")
    
    _fallback_cache = None

    @classmethod
    def _compiled_fallback(cls) -> Dict[int, types.FunctionType]:
        """Split _execute_opcode's if/elif chain into one function per opcode."""
        if cls._fallback_cache is not None:
            return cls._fallback_cache
        funcs = {}
        try:
            fn = ast.parse(inspect.getsource(cls._execute_opcode).strip()).body[0]
            node = next(b for b in fn.body if isinstance(b, ast.If))
            branches = []
            while isinstance(node, ast.If):
                branches.append((node.test, node.body))
                node = node.orelse[0] if len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If) else None
            for op in range(256):
                for test, body in branches:
                    if eval(compile(ast.Expression(test), "<test>", "eval"), {"opcode": op}):
                        code = f"def _op_{op:02X}(self, opcode={op}):\n" + "\n".join(
                            "    " + line for line in ast.unparse(body).splitlines())
                        namespace = {}
                        exec(compile(code, f"<z80 opcode {op:02X}>", "exec"), inspect.getmodule(cls).__dict__, namespace)
                        funcs[op] = namespace[f"_op_{op:02X}"]
                        break
        except (OSError, TypeError, SyntaxError):
            funcs = {}                     # source unavailable: keep the generic fallback
        cls._fallback_cache = funcs
        return funcs

    def _build_opcode_table(self):
        """Populate the 256-entry jump table with method references"""
        # Default: one function per opcode, compiled from the branches of the old if/elif chain
        # (same code, but O(1) dispatch instead of walking up to ~200 comparisons)
        fallback = self._compiled_fallback()
        for i in range(256):
            if i in fallback:
                self.opcode_table[i] = types.MethodType(fallback[i], self)
            else:
                self.opcode_table[i] = lambda op=i: self._execute_opcode(op)

        # --- 8-bit Load Group ---
        for dst in range(8):
            for src in range(8):
                if dst == 6 and src == 6: 
                    self.opcode_table[0x76] = self._op_halt
                    continue
                opcode = 0x40 + (dst << 3) + src
                self.opcode_table[opcode] = self._make_ld_handler(dst, src)
        
        # LD r, n
        self.opcode_table[0x06] = self._op_ld_b_n
        self.opcode_table[0x0E] = self._op_ld_c_n
        self.opcode_table[0x16] = self._op_ld_d_n
        self.opcode_table[0x1E] = self._op_ld_e_n
        self.opcode_table[0x26] = self._op_ld_h_n
        self.opcode_table[0x2E] = self._op_ld_l_n
        self.opcode_table[0x3E] = self._op_ld_a_n
        
        # 16-bit loads
        self.opcode_table[0x01] = self._op_ld_bc_nn
        self.opcode_table[0x11] = self._op_ld_de_nn
        self.opcode_table[0x21] = self._op_ld_hl_nn
        self.opcode_table[0x31] = self._op_ld_sp_nn
        
        # Special LD
        self.opcode_table[0x0A] = self._op_ld_a_bc_ptr
        self.opcode_table[0x1A] = self._op_ld_a_de_ptr
        self.opcode_table[0x3A] = self._op_ld_a_nn_ptr
        self.opcode_table[0x02] = self._op_ld_bc_ptr_a
        self.opcode_table[0x12] = self._op_ld_de_ptr_a
        self.opcode_table[0x32] = self._op_ld_nn_ptr_a

        # --- Arithmetic / Logic Group ---
        # ADD A, r / ADC A, r / SUB r / SBC A, r / AND r / XOR r / OR r / CP r
        for i in range(8):
            self.opcode_table[0x80 + i] = self._make_alu_handler(0, i) # ADD
            self.opcode_table[0x88 + i] = self._make_alu_handler(1, i) # ADC
            self.opcode_table[0x90 + i] = self._make_alu_handler(2, i) # SUB
            self.opcode_table[0x98 + i] = self._make_alu_handler(3, i) # SBC
            self.opcode_table[0xA0 + i] = self._make_alu_handler(4, i) # AND
            self.opcode_table[0xA8 + i] = self._make_alu_handler(5, i) # XOR
            self.opcode_table[0xB0 + i] = self._make_alu_handler(6, i) # OR
            self.opcode_table[0xB8 + i] = self._make_alu_handler(7, i) # CP

        # Immediate ALU
        self.opcode_table[0xC6] = lambda: self._execute_add_8(self._fetch_byte())
        self.opcode_table[0xCE] = lambda: self._execute_add_8(self._fetch_byte(), self.f & self.FLAG_C)
        self.opcode_table[0xD6] = lambda: self._execute_sub_8(self._fetch_byte())
        self.opcode_table[0xDE] = lambda: self._execute_sub_8(self._fetch_byte(), self.f & self.FLAG_C)
        self.opcode_table[0xE6] = lambda: self._execute_and(self._fetch_byte())
        self.opcode_table[0xEE] = lambda: self._execute_xor(self._fetch_byte())
        self.opcode_table[0xF6] = lambda: self._execute_or(self._fetch_byte())
        self.opcode_table[0xFE] = lambda: self._execute_cp(self._fetch_byte())

        # INC / DEC r
        self.opcode_table[0x3C] = self._op_inc_a
        self.opcode_table[0x04] = self._op_inc_b
        self.opcode_table[0x0C] = self._op_inc_c
        self.opcode_table[0x14] = self._op_inc_d
        self.opcode_table[0x1C] = self._op_inc_e
        self.opcode_table[0x24] = self._op_inc_h
        self.opcode_table[0x2C] = self._op_inc_l
        self.opcode_table[0x34] = self._op_inc_hl_ptr
        
        self.opcode_table[0x3D] = self._op_dec_a
        self.opcode_table[0x05] = self._op_dec_b
        self.opcode_table[0x0D] = self._op_dec_c
        self.opcode_table[0x15] = self._op_dec_d
        self.opcode_table[0x1D] = self._op_dec_e
        self.opcode_table[0x25] = self._op_dec_h
        self.opcode_table[0x2D] = self._op_dec_l
        self.opcode_table[0x35] = self._op_dec_hl_ptr

        # --- Jump Group ---
        self.opcode_table[0xC3] = self._op_jp_nn
        self.opcode_table[0xE9] = self._op_jp_hl
        self.opcode_table[0x18] = self._op_jr_e
        self.opcode_table[0x28] = self._op_jr_z_e
        self.opcode_table[0x20] = self._op_jr_nz_e
        self.opcode_table[0x38] = self._op_jr_c_e
        self.opcode_table[0x30] = self._op_jr_nc_e
        self.opcode_table[0x10] = self._op_djnz_e
        
        # --- Call / Return Group ---
        self.opcode_table[0xCD] = self._op_call_nn
        self.opcode_table[0xC9] = self._op_ret

        # --- Prefix Handlers ---
        self.opcode_table[0xCB] = self._handle_cb_prefix
        self.opcode_table[0xED] = self._handle_ed_prefix
        self.opcode_table[0xDD] = self._handle_ix_prefix
        self.opcode_table[0xFD] = self._handle_iy_prefix

        # Conditional CALL / RET take extra T-states when the condition is met
        for op in (0xC4, 0xCC, 0xD4, 0xDC, 0xE4, 0xEC, 0xF4, 0xFC):
            self.opcode_table[op] = self._make_taken_extra(self.opcode_table[op], 7)
        for op in (0xC0, 0xC8, 0xD0, 0xD8, 0xE0, 0xE8, 0xF0, 0xF8):
            self.opcode_table[op] = self._make_taken_extra(self.opcode_table[op], 6)

    def _make_taken_extra(self, handler, extra: int):
        def op():
            sp = self.sp
            handler()
            if self.sp != sp:
                self.cycles += extra
        return op

    def _op_jp_nn(self): self.pc = self._fetch_word()
    def _op_jp_hl(self): self.pc = self._get_hl()
    
    def _op_jr_e(self):
        offset = self._fetch_byte()
        if offset > 127: offset -= 256
        self.pc = (self.pc + offset) & 0xFFFF

    def _op_jr_z_e(self):
        offset = self._fetch_byte()
        if self.f & self.FLAG_Z:
            self.cycles += 5
            if offset > 127: offset -= 256
            self.pc = (self.pc + offset) & 0xFFFF

    def _op_jr_nz_e(self):
        offset = self._fetch_byte()
        if not (self.f & self.FLAG_Z):
            self.cycles += 5
            if offset > 127: offset -= 256
            self.pc = (self.pc + offset) & 0xFFFF

    def _op_jr_c_e(self):
        offset = self._fetch_byte()
        if self.f & self.FLAG_C:
            self.cycles += 5
            if offset > 127: offset -= 256
            self.pc = (self.pc + offset) & 0xFFFF

    def _op_jr_nc_e(self):
        offset = self._fetch_byte()
        if not (self.f & self.FLAG_C):
            self.cycles += 5
            if offset > 127: offset -= 256
            self.pc = (self.pc + offset) & 0xFFFF

    def _op_djnz_e(self):
        offset = self._fetch_byte()
        self.b = (self.b - 1) & 0xFF
        if self.b != 0:
            self.cycles += 5
            if offset > 127: offset -= 256
            self.pc = (self.pc + offset) & 0xFFFF

    def _op_call_nn(self):
        dest = self._fetch_word()
        self._push_to_stack(self.pc)
        self.pc = dest

    def _op_ret(self):
        self.pc = self._pop_from_stack()

    def _make_ld_handler(self, dst, src):
        """Return a specific method reference for LD r, r' to avoid getattr overhead"""
        # We map these to the specific methods defined below
        reg_methods = [
            [None, self._op_ld_b_c, self._op_ld_b_d, self._op_ld_b_e, self._op_ld_b_h, self._op_ld_b_l, self._op_ld_b_hl_ptr, self._op_ld_b_a],
            [self._op_ld_c_b, None, self._op_ld_c_d, self._op_ld_c_e, self._op_ld_c_h, self._op_ld_c_l, self._op_ld_c_hl_ptr, self._op_ld_c_a],
            [self._op_ld_d_b, self._op_ld_d_c, None, self._op_ld_d_e, self._op_ld_d_h, self._op_ld_d_l, self._op_ld_d_hl_ptr, self._op_ld_d_a],
            [self._op_ld_e_b, self._op_ld_e_c, self._op_ld_e_d, None, self._op_ld_e_h, self._op_ld_e_l, self._op_ld_e_hl_ptr, self._op_ld_e_a],
            [self._op_ld_h_b, self._op_ld_h_c, self._op_ld_h_d, self._op_ld_h_e, None, self._op_ld_h_l, self._op_ld_h_hl_ptr, self._op_ld_h_a],
            [self._op_ld_l_b, self._op_ld_l_c, self._op_ld_l_d, self._op_ld_l_e, self._op_ld_l_h, None, self._op_ld_l_hl_ptr, self._op_ld_l_a],
            [self._op_ld_hl_ptr_b, self._op_ld_hl_ptr_c, self._op_ld_hl_ptr_d, self._op_ld_hl_ptr_e, self._op_ld_hl_ptr_h, self._op_ld_hl_ptr_l, None, self._op_ld_hl_ptr_a],
            [self._op_ld_a_b, self._op_ld_a_c, self._op_ld_a_d, self._op_ld_a_e, self._op_ld_a_h, self._op_ld_a_l, self._op_ld_a_hl_ptr, None]
        ]
        
        handler = reg_methods[dst][src]
        return handler if handler else (lambda: None)

    def _make_alu_handler(self, op_type, src):
        """Return a specific method reference for ALU operations to avoid getattr overhead"""
        # op_type: 0=ADD, 1=ADC, 2=SUB, 3=SBC, 4=AND, 5=XOR, 6=OR, 7=CP
        # src: 0=B, 1=C, 2=D, 3=E, 4=H, 5=L, 6=(HL), 7=A
        
        alu_map = [
            # ADD A, r
            [self._op_add_b, self._op_add_c, self._op_add_d, self._op_add_e, self._op_add_h, self._op_add_l, self._op_add_hl_ptr, self._op_add_a],
            # ADC A, r
            [self._op_adc_b, self._op_adc_c, self._op_adc_d, self._op_adc_e, self._op_adc_h, self._op_adc_l, self._op_adc_hl_ptr, self._op_adc_a],
            # SUB r
            [self._op_sub_b, self._op_sub_c, self._op_sub_d, self._op_sub_e, self._op_sub_h, self._op_sub_l, self._op_sub_hl_ptr, self._op_sub_a],
            # SBC A, r
            [self._op_sbc_b, self._op_sbc_c, self._op_sbc_d, self._op_sbc_e, self._op_sbc_h, self._op_sbc_l, self._op_sbc_hl_ptr, self._op_sbc_a],
            # AND r
            [self._op_and_b, self._op_and_c, self._op_and_d, self._op_and_e, self._op_and_h, self._op_and_l, self._op_and_hl_ptr, self._op_and_a],
            # XOR r
            [self._op_xor_b, self._op_xor_c, self._op_xor_d, self._op_xor_e, self._op_xor_h, self._op_xor_l, self._op_xor_hl_ptr, self._op_xor_a],
            # OR r
            [self._op_or_b, self._op_or_c, self._op_or_d, self._op_or_e, self._op_or_h, self._op_or_l, self._op_or_hl_ptr, self._op_or_a],
            # CP r
            [self._op_cp_b, self._op_cp_c, self._op_cp_d, self._op_cp_e, self._op_cp_h, self._op_cp_l, self._op_cp_hl_ptr, self._op_cp_a]
        ]
        
        return alu_map[op_type][src]

    def _op_halt(self):
        self.halted = True
        self.pc = (self.pc - 1) & 0xFFFF

    # --- Opcode Handlers ---
    def _op_ld_a_n(self): self.a = self._fetch_byte()
    def _op_ld_b_n(self): self.b = self._fetch_byte()
    def _op_ld_c_n(self): self.c = self._fetch_byte()
    def _op_ld_d_n(self): self.d = self._fetch_byte()
    def _op_ld_e_n(self): self.e = self._fetch_byte()
    def _op_ld_h_n(self): self.h = self._fetch_byte()
    def _op_ld_l_n(self): self.l = self._fetch_byte()
    
    def _op_ld_bc_nn(self): self._set_bc(self._fetch_word())
    def _op_ld_de_nn(self): self._set_de(self._fetch_word())
    def _op_ld_hl_nn(self): self._set_hl(self._fetch_word())
    def _op_ld_sp_nn(self): self.sp = self._fetch_word()
    
    def _op_ld_a_b(self): self.a = self.b
    def _op_ld_a_c(self): self.a = self.c
    def _op_ld_a_d(self): self.a = self.d
    def _op_ld_a_e(self): self.a = self.e
    def _op_ld_a_h(self): self.a = self.h
    def _op_ld_a_l(self): self.a = self.l
    def _op_ld_a_hl_ptr(self): self.a = self._read_byte(self._get_hl())
    
    def _op_ld_b_c(self): self.b = self.c
    def _op_ld_b_d(self): self.b = self.d
    def _op_ld_b_e(self): self.b = self.e
    def _op_ld_b_h(self): self.b = self.h
    def _op_ld_b_l(self): self.b = self.l
    def _op_ld_b_a(self): self.b = self.a
    def _op_ld_b_hl_ptr(self): self.b = self._read_byte(self._get_hl())

    # Destination C
    def _op_ld_c_b(self): self.c = self.b
    def _op_ld_c_d(self): self.c = self.d
    def _op_ld_c_e(self): self.c = self.e
    def _op_ld_c_h(self): self.c = self.h
    def _op_ld_c_l(self): self.c = self.l
    def _op_ld_c_a(self): self.c = self.a
    def _op_ld_c_hl_ptr(self): self.c = self._read_byte(self._get_hl())

    # Destination D
    def _op_ld_d_b(self): self.d = self.b
    def _op_ld_d_c(self): self.d = self.c
    def _op_ld_d_e(self): self.d = self.e
    def _op_ld_d_h(self): self.d = self.h
    def _op_ld_d_l(self): self.d = self.l
    def _op_ld_d_a(self): self.d = self.a
    def _op_ld_d_hl_ptr(self): self.d = self._read_byte(self._get_hl())

    # Destination E
    def _op_ld_e_b(self): self.e = self.b
    def _op_ld_e_c(self): self.e = self.c
    def _op_ld_e_d(self): self.e = self.d
    def _op_ld_e_h(self): self.e = self.h
    def _op_ld_e_l(self): self.e = self.l
    def _op_ld_e_a(self): self.e = self.a
    def _op_ld_e_hl_ptr(self): self.e = self._read_byte(self._get_hl())

    # Destination H
    def _op_ld_h_b(self): self.h = self.b
    def _op_ld_h_c(self): self.h = self.c
    def _op_ld_h_d(self): self.h = self.d
    def _op_ld_h_e(self): self.h = self.e
    def _op_ld_h_l(self): self.h = self.l
    def _op_ld_h_a(self): self.h = self.a
    def _op_ld_h_hl_ptr(self): self.h = self._read_byte(self._get_hl())

    # Destination L
    def _op_ld_l_b(self): self.l = self.b
    def _op_ld_l_c(self): self.l = self.c
    def _op_ld_l_d(self): self.l = self.d
    def _op_ld_l_e(self): self.l = self.e
    def _op_ld_l_h(self): self.l = self.h
    def _op_ld_l_a(self): self.l = self.a
    def _op_ld_l_hl_ptr(self): self.l = self._read_byte(self._get_hl())

    # Destination (HL)
    def _op_ld_hl_ptr_b(self): self._write_byte(self._get_hl(), self.b)
    def _op_ld_hl_ptr_c(self): self._write_byte(self._get_hl(), self.c)
    def _op_ld_hl_ptr_d(self): self._write_byte(self._get_hl(), self.d)
    def _op_ld_hl_ptr_e(self): self._write_byte(self._get_hl(), self.e)
    def _op_ld_hl_ptr_h(self): self._write_byte(self._get_hl(), self.h)
    def _op_ld_hl_ptr_l(self): self._write_byte(self._get_hl(), self.l)
    def _op_ld_hl_ptr_a(self): self._write_byte(self._get_hl(), self.a)
    
    def _op_ld_a_bc_ptr(self): self.a = self._read_byte(self._get_bc())
    def _op_ld_a_de_ptr(self): self.a = self._read_byte(self._get_de())
    def _op_ld_a_nn_ptr(self): self.a = self._read_byte(self._fetch_word())
    def _op_ld_bc_ptr_a(self): self._write_byte(self._get_bc(), self.a)
    def _op_ld_de_ptr_a(self): self._write_byte(self._get_de(), self.a)
    def _op_ld_nn_ptr_a(self): self._write_byte(self._fetch_word(), self.a)

    # --- ALU Operations Wrappers ---
    def _op_add_a(self): self._execute_add_8(self.a)
    def _op_add_b(self): self._execute_add_8(self.b)
    def _op_add_c(self): self._execute_add_8(self.c)
    def _op_add_d(self): self._execute_add_8(self.d)
    def _op_add_e(self): self._execute_add_8(self.e)
    def _op_add_h(self): self._execute_add_8(self.h)
    def _op_add_l(self): self._execute_add_8(self.l)
    def _op_add_hl_ptr(self): self._execute_add_8(self._read_byte(self._get_hl()))

    def _op_adc_a(self): self._execute_add_8(self.a, self.f & self.FLAG_C)
    def _op_adc_b(self): self._execute_add_8(self.b, self.f & self.FLAG_C)
    def _op_adc_c(self): self._execute_add_8(self.c, self.f & self.FLAG_C)
    def _op_adc_d(self): self._execute_add_8(self.d, self.f & self.FLAG_C)
    def _op_adc_e(self): self._execute_add_8(self.e, self.f & self.FLAG_C)
    def _op_adc_h(self): self._execute_add_8(self.h, self.f & self.FLAG_C)
    def _op_adc_l(self): self._execute_add_8(self.l, self.f & self.FLAG_C)
    def _op_adc_hl_ptr(self): self._execute_add_8(self._read_byte(self._get_hl()), self.f & self.FLAG_C)

    def _op_sub_a(self): self._execute_sub_8(self.a)
    def _op_sub_b(self): self._execute_sub_8(self.b)
    def _op_sub_c(self): self._execute_sub_8(self.c)
    def _op_sub_d(self): self._execute_sub_8(self.d)
    def _op_sub_e(self): self._execute_sub_8(self.e)
    def _op_sub_h(self): self._execute_sub_8(self.h)
    def _op_sub_l(self): self._execute_sub_8(self.l)
    def _op_sub_hl_ptr(self): self._execute_sub_8(self._read_byte(self._get_hl()))

    def _op_sbc_a(self): self._execute_sub_8(self.a, self.f & self.FLAG_C)
    def _op_sbc_b(self): self._execute_sub_8(self.b, self.f & self.FLAG_C)
    def _op_sbc_c(self): self._execute_sub_8(self.c, self.f & self.FLAG_C)
    def _op_sbc_d(self): self._execute_sub_8(self.d, self.f & self.FLAG_C)
    def _op_sbc_e(self): self._execute_sub_8(self.e, self.f & self.FLAG_C)
    def _op_sbc_h(self): self._execute_sub_8(self.h, self.f & self.FLAG_C)
    def _op_sbc_l(self): self._execute_sub_8(self.l, self.f & self.FLAG_C)
    def _op_sbc_hl_ptr(self): self._execute_sub_8(self._read_byte(self._get_hl()), self.f & self.FLAG_C)

    def _op_and_a(self): self._execute_and(self.a)
    def _op_and_b(self): self._execute_and(self.b)
    def _op_and_c(self): self._execute_and(self.c)
    def _op_and_d(self): self._execute_and(self.d)
    def _op_and_e(self): self._execute_and(self.e)
    def _op_and_h(self): self._execute_and(self.h)
    def _op_and_l(self): self._execute_and(self.l)
    def _op_and_hl_ptr(self): self._execute_and(self._read_byte(self._get_hl()))

    def _op_xor_a(self): self._execute_xor(self.a)
    def _op_xor_b(self): self._execute_xor(self.b)
    def _op_xor_c(self): self._execute_xor(self.c)
    def _op_xor_d(self): self._execute_xor(self.d)
    def _op_xor_e(self): self._execute_xor(self.e)
    def _op_xor_h(self): self._execute_xor(self.h)
    def _op_xor_l(self): self._execute_xor(self.l)
    def _op_xor_hl_ptr(self): self._execute_xor(self._read_byte(self._get_hl()))

    def _op_or_a(self): self._execute_or(self.a)
    def _op_or_b(self): self._execute_or(self.b)
    def _op_or_c(self): self._execute_or(self.c)
    def _op_or_d(self): self._execute_or(self.d)
    def _op_or_e(self): self._execute_or(self.e)
    def _op_or_h(self): self._execute_or(self.h)
    def _op_or_l(self): self._execute_or(self.l)
    def _op_or_hl_ptr(self): self._execute_or(self._read_byte(self._get_hl()))

    def _op_cp_a(self): self._execute_cp(self.a)
    def _op_cp_b(self): self._execute_cp(self.b)
    def _op_cp_c(self): self._execute_cp(self.c)
    def _op_cp_d(self): self._execute_cp(self.d)
    def _op_cp_e(self): self._execute_cp(self.e)
    def _op_cp_h(self): self._execute_cp(self.h)
    def _op_cp_l(self): self._execute_cp(self.l)
    def _op_cp_hl_ptr(self): self._execute_cp(self._read_byte(self._get_hl()))

    # --- INC / DEC Handlers ---
    def _op_inc_a(self): self.a = self._execute_inc(self.a)
    def _op_inc_b(self): self.b = self._execute_inc(self.b)
    def _op_inc_c(self): self.c = self._execute_inc(self.c)
    def _op_inc_d(self): self.d = self._execute_inc(self.d)
    def _op_inc_e(self): self.e = self._execute_inc(self.e)
    def _op_inc_h(self): self.h = self._execute_inc(self.h)
    def _op_inc_l(self): self.l = self._execute_inc(self.l)
    def _op_inc_hl_ptr(self):
        addr = self._get_hl()
        self._write_byte(addr, self._execute_inc(self._read_byte(addr)))

    def _op_dec_a(self): self.a = self._execute_dec(self.a)
    def _op_dec_b(self): self.b = self._execute_dec(self.b)
    def _op_dec_c(self): self.c = self._execute_dec(self.c)
    def _op_dec_d(self): self.d = self._execute_dec(self.d)
    def _op_dec_e(self): self.e = self._execute_dec(self.e)
    def _op_dec_h(self): self.h = self._execute_dec(self.h)
    def _op_dec_l(self): self.l = self._execute_dec(self.l)
    def _op_dec_hl_ptr(self):
        addr = self._get_hl()
        self._write_byte(addr, self._execute_dec(self._read_byte(addr)))

    def _fetch_byte(self) -> int:
        # Optimized fetch: ROM is always safe to read directly from memory buffer
        if self.pc < 0x8000:
            val = self.memory[self.pc]
            self.pc = (self.pc + 1) & 0xFFFF
            return val
        
        # Fallback for RAM or I/O area
        val = self._read_byte(self.pc)
        self.pc = (self.pc + 1) & 0xFFFF
        return val

    def _fetch_word(self) -> int:
        low = self._fetch_byte()
        high = self._fetch_byte()
        return (high << 8) | low

    def _read_port(self, port: int) -> int:
        if self.debug_log:
            print(f"DEBUG: IN from port {port:02X}")
        return 0xFF

    def _write_port(self, port: int, value: int):
        if self.debug_log:
            print(f"DEBUG: OUT to port {port:02X}, value {value:02X}")

    def request_interrupt(self, opcode: int = 0xFF):
        """
        Request a maskable interrupt.
        opcode: The instruction to execute for IM 0 or IM 1 (default 0xFF = RST 38H)
        """
        if not self.iff1:
            return

        self.halted = False
        self.iff1 = self.iff2 = False

        if self.im == 0 or self.im == 1:
            # IM 0/1: Execute the provided opcode (usually RST)
            self._push_to_stack(self.pc)
            
            # RST Opcode decoding (0xC7, 0xCF, ..., 0xFF)
            # 0xD7 is RST 10H
            target = opcode & 0x38
            self.pc = target

            # Simple assumption: opcode is a RST
            # For IM 1, it's always 0x38 (0xFF)
            if self.im == 1:
                self.pc = 0x0038
            else:
                self.pc = target

    # --- ALU Helpers ---
    def _execute_add_16(self, val1: int, val2: int) -> int:
        result = val1 + val2
        self.f &= ~(self.FLAG_N | self.FLAG_H | self.FLAG_C)
        if (val1 & 0x0FFF) + (val2 & 0x0FFF) > 0x0FFF:
            self.f |= self.FLAG_H
        if result > 0xFFFF:
            self.f |= self.FLAG_C
        return result & 0xFFFF

    def _execute_adc_16(self, val1: int, val2: int) -> int:
        carry = 1 if (self.f & self.FLAG_C) else 0
        result = val1 + val2 + carry
        
        self.f = 0
        if (result & 0xFFFF) == 0: self.f |= self.FLAG_Z
        if (result & 0x8000): self.f |= self.FLAG_S
        if result > 0xFFFF: self.f |= self.FLAG_C
        if (val1 & 0x0FFF) + (val2 & 0x0FFF) + carry > 0x0FFF: self.f |= self.FLAG_H
        
        val1_s = (val1 >> 15) & 1
        val2_s = (val2 >> 15) & 1
        res_s = ((result & 0xFFFF) >> 15) & 1
        if (val1_s == val2_s) and (res_s != val1_s):
            self.f |= self.FLAG_PV
            
        return result & 0xFFFF

    def _execute_sbc_16(self, val1: int, val2: int) -> int:
        carry = 1 if (self.f & self.FLAG_C) else 0
        result = val1 - val2 - carry
        
        self.f = self.FLAG_N
        if (result & 0xFFFF) == 0: self.f |= self.FLAG_Z
        if (result & 0x8000): self.f |= self.FLAG_S
        if result < 0: self.f |= self.FLAG_C
        if (val1 & 0x0FFF) < (val2 & 0x0FFF) + carry: self.f |= self.FLAG_H
        
        val1_s = (val1 >> 15) & 1
        val2_s = (val2 >> 15) & 1
        res_s = ((result & 0xFFFF) >> 15) & 1
        if (val1_s != val2_s) and (res_s != val1_s):
            self.f |= self.FLAG_PV
            
        return result & 0xFFFF

    def _execute_add_8(self, value: int, carry: int = 0):
        result = self.a + value + carry
        self.f = 0
        if (self.a & 0x0F) + (value & 0x0F) + carry > 0x0F: self.f |= self.FLAG_H
        if result > 0xFF: self.f |= self.FLAG_C
        
        old_a = self.a
        self.a = result & 0xFF
        
        if self.a == 0: self.f |= self.FLAG_Z
        if self.a & 0x80: self.f |= self.FLAG_S
        
        # PV: set if overflow (signed arithmetic)
        if ((old_a ^ value) & 0x80) == 0 and ((old_a ^ self.a) & 0x80) != 0:
            self.f |= self.FLAG_PV

    def _execute_sub_8(self, value: int, carry: int = 0):
        result = self.a - value - carry
        self.f = self.FLAG_N
        if (self.a & 0x0F) < (value & 0x0F) + carry: self.f |= self.FLAG_H
        if result < 0: self.f |= self.FLAG_C
        
        old_a = self.a
        self.a = result & 0xFF
        
        if self.a == 0: self.f |= self.FLAG_Z
        if self.a & 0x80: self.f |= self.FLAG_S
        
        # PV: set if overflow (signed arithmetic)
        if ((old_a ^ value) & 0x80) != 0 and ((old_a ^ self.a) & 0x80) != 0:
            self.f |= self.FLAG_PV

    def _execute_neg(self):
        val = self.a
        self.a = 0
        self._execute_sub_8(val)

    def _execute_and(self, value: int):
        self.a &= value
        self.f = self.FLAG_H
        if self.a == 0: self.f |= self.FLAG_Z
        if self.a & 0x80: self.f |= self.FLAG_S
        if self.parity_table[self.a]: self.f |= self.FLAG_PV

    def _execute_or(self, value: int):
        self.a |= value
        self.f = 0
        if self.a == 0: self.f |= self.FLAG_Z
        if self.a & 0x80: self.f |= self.FLAG_S
        if self.parity_table[self.a]: self.f |= self.FLAG_PV

    def _execute_xor(self, value: int):
        self.a ^= value
        self.f = 0
        if self.a == 0: self.f |= self.FLAG_Z
        if self.a & 0x80: self.f |= self.FLAG_S
        if self.parity_table[self.a]: self.f |= self.FLAG_PV

    def _execute_inc(self, value: int) -> int:
        result = (value + 1) & 0xFF
        # Preserve C, clear others
        self.f = (self.f & self.FLAG_C)
        if result == 0: self.f |= self.FLAG_Z
        if result & 0x80: self.f |= self.FLAG_S
        if (value & 0x0F) == 0x0F: self.f |= self.FLAG_H
        if value == 0x7F: self.f |= self.FLAG_PV
        return result

    def _execute_dec(self, value: int) -> int:
        result = (value - 1) & 0xFF
        # Preserve C, clear others, set N
        self.f = (self.f & self.FLAG_C) | self.FLAG_N
        if result == 0: self.f |= self.FLAG_Z
        if result & 0x80: self.f |= self.FLAG_S
        if (value & 0x0F) == 0x00: self.f |= self.FLAG_H
        if value == 0x80: self.f |= self.FLAG_PV
        return result

    def _execute_cp(self, value: int):
        result = (self.a - value) & 0xFF
        self.f = self.FLAG_N
        if result == 0: self.f |= self.FLAG_Z
        if result & 0x80: self.f |= self.FLAG_S
        if self.a < value: self.f |= self.FLAG_C
        if (self.a & 0x0F) < (value & 0x0F): self.f |= self.FLAG_H
        if ((self.a ^ value) & 0x80) and ((self.a ^ result) & 0x80):
             self.f |= self.FLAG_PV

    def _execute_daa(self):
        old_a = self.a
        old_c = self.f & self.FLAG_C
        old_h = self.f & self.FLAG_H
        
        correction = 0
        
        if (old_a & 0x0F) > 9 or old_h:
            correction |= 0x06
        
        if old_a > 0x99 or old_c:
            correction |= 0x60
            self.f |= self.FLAG_C
        
        if self.f & self.FLAG_N:
            new_h = old_h and (old_a & 0x0F) < 6
            self.a = (self.a - correction) & 0xFF
        else:
            new_h = (old_a & 0x0F) > 9
            self.a = (self.a + correction) & 0xFF

        self.f &= ~(self.FLAG_S | self.FLAG_Z | self.FLAG_H | self.FLAG_PV)
        if new_h: self.f |= self.FLAG_H
        if self.a & 0x80: self.f |= self.FLAG_S
        if self.a == 0: self.f |= self.FLAG_Z
        if self.parity_table[self.a]: self.f |= self.FLAG_PV

    def step(self):
        if self.halted:
            self.cycles += 4
            return

        current_pc = self.pc
        opcode = self._fetch_byte()
        self.r = (self.r & 0x80) | ((self.r + 1) & 0x7F)  # memory refresh counter

        # Add T-states for this opcode
        self.cycles += self.cycle_table[opcode]

        if self.debug_log:
            state = (f"PC:{current_pc:04X} SP:{self.sp:04X} "
                     f"AF:{self._get_af():04X} BC:{self._get_bc():04X} DE:{self._get_de():04X} HL:{self._get_hl():04X} "
                     f"IX:{self.ix:04X} IY:{self.iy:04X} | Op:{opcode:02X}")
            print(state)

        # Dispatch via O(1) jump table
        self.opcode_table[opcode]()

    def _execute_opcode(self, opcode):
        """Fallback O(n) handler for opcodes not yet moved to the jump table."""
        # --- 8-bit Load Group ---
        if opcode == 0x7F: self.a = self.a
        elif opcode == 0x78: self.a = self.b
        elif opcode == 0x79: self.a = self.c
        elif opcode == 0x7A: self.a = self.d
        elif opcode == 0x7B: self.a = self.e
        elif opcode == 0x7C: self.a = self.h
        elif opcode == 0x7D: self.a = self.l
        elif opcode == 0x7E: self.a = self._read_byte(self._get_hl())
        elif opcode == 0x3E: self.a = self._fetch_byte()
        elif opcode == 0x0A: self.a = self._read_byte(self._get_bc())
        elif opcode == 0x1A: self.a = self._read_byte(self._get_de())
        elif opcode == 0x3A: self.a = self._read_byte(self._fetch_word())
        
        elif opcode == 0x47: self.b = self.a
        elif opcode == 0x46: self.b = self._read_byte(self._get_hl())
        elif opcode == 0x06: self.b = self._fetch_byte()
        elif opcode == 0x4F: self.c = self.a
        elif opcode == 0x4E: self.c = self._read_byte(self._get_hl())
        elif opcode == 0x0E: self.c = self._fetch_byte()
        
        elif opcode == 0x51: self.d = self.c
        elif opcode == 0x57: self.d = self.a
        elif opcode == 0x54: self.d = self.h
        elif opcode == 0x56: self.d = self._read_byte(self._get_hl())
        elif opcode == 0x16: self.d = self._fetch_byte()
        elif opcode == 0x59: self.e = self.c
        elif opcode == 0x5F: self.e = self.a
        elif opcode == 0x5D: self.e = self.l
        elif opcode == 0x5E: self.e = self._read_byte(self._get_hl())
        elif opcode == 0x1E: self.e = self._fetch_byte()

        elif opcode == 0x67: self.h = self.a
        elif opcode == 0x60: self.h = self.b
        elif opcode == 0x66: self.h = self._read_byte(self._get_hl())
        elif opcode == 0x26: self.h = self._fetch_byte()
        elif opcode == 0x69: self.l = self.c
        elif opcode == 0x6F: self.l = self.a
        elif opcode == 0x6E: self.l = self._read_byte(self._get_hl())
        elif opcode == 0x2E: self.l = self._fetch_byte()

        elif opcode == 0x77: self._write_byte(self._get_hl(), self.a)
        elif opcode == 0x70: self._write_byte(self._get_hl(), self.b)
        elif opcode == 0x71: self._write_byte(self._get_hl(), self.c)
        elif opcode == 0x72: self._write_byte(self._get_hl(), self.d)
        elif opcode == 0x73: self._write_byte(self._get_hl(), self.e)
        elif opcode == 0x74: self._write_byte(self._get_hl(), self.h)
        elif opcode == 0x75: self._write_byte(self._get_hl(), self.l)
        elif opcode == 0x36: self._write_byte(self._get_hl(), self._fetch_byte())
        
        elif opcode == 0x02: self._write_byte(self._get_bc(), self.a)
        elif opcode == 0x12: self._write_byte(self._get_de(), self.a)
        elif opcode == 0x32: self._write_byte(self._fetch_word(), self.a)

        # --- 16-bit Load Group ---
        elif opcode == 0x01: self._set_bc(self._fetch_word())
        elif opcode == 0x11: self._set_de(self._fetch_word())
        elif opcode == 0x21: self._set_hl(self._fetch_word())
        elif opcode == 0x22: # LD (nn), HL
            addr = self._fetch_word()
            self._write_byte(addr, self.l)
            self._write_byte((addr + 1) & 0xFFFF, self.h)
        elif opcode == 0x2A: # LD HL, (nn)
            addr = self._fetch_word()
            low = self._read_byte(addr)
            high = self._read_byte((addr + 1) & 0xFFFF)
            self._set_hl((high << 8) | low)
        elif opcode == 0x31: self.sp = self._fetch_word()
        
        # --- Exchange ---
        elif opcode == 0xEB:
            self.d, self.h = self.h, self.d
            self.e, self.l = self.l, self.e
        elif opcode == 0x08: # EX AF, AF'
            self.a, self.a_alt = self.a_alt, self.a
            self.f, self.f_alt = self.f_alt, self.f
        elif opcode == 0xD9: # EXX
            self.b, self.b_alt = self.b_alt, self.b
            self.c, self.c_alt = self.c_alt, self.c
            self.d, self.d_alt = self.d_alt, self.d
            self.e, self.e_alt = self.e_alt, self.e
            self.h, self.h_alt = self.h_alt, self.h
            self.l, self.l_alt = self.l_alt, self.l

        # --- Arithmetic Group ---
        elif opcode >= 0x80 and opcode <= 0x87:
            regs = [self.b, self.c, self.d, self.e, self.h, self.l, self._read_byte(self._get_hl()), self.a]
            self._execute_add_8(regs[opcode & 0x07])
        elif opcode == 0xC6: self._execute_add_8(self._fetch_byte())
        elif opcode >= 0x88 and opcode <= 0x8F:
            regs = [self.b, self.c, self.d, self.e, self.h, self.l, self._read_byte(self._get_hl()), self.a]
            self._execute_add_8(regs[opcode & 0x07], self.f & self.FLAG_C)
        elif opcode == 0xCE: self._execute_add_8(self._fetch_byte(), self.f & self.FLAG_C)
        
        elif opcode >= 0x90 and opcode <= 0x97:
            regs = [self.b, self.c, self.d, self.e, self.h, self.l, self._read_byte(self._get_hl()), self.a]
            self._execute_sub_8(regs[opcode & 0x07])
        elif opcode == 0xD6: self._execute_sub_8(self._fetch_byte())

        elif opcode == 0x27: self._execute_daa()

        elif opcode == 0x3C: self.a = self._execute_inc(self.a)
        elif opcode == 0x04: self.b = self._execute_inc(self.b)
        elif opcode == 0x0C: self.c = self._execute_inc(self.c)
        elif opcode == 0x14: self.d = self._execute_inc(self.d)
        elif opcode == 0x1C: self.e = self._execute_inc(self.e)
        elif opcode == 0x24: self.h = self._execute_inc(self.h)
        elif opcode == 0x2C: self.l = self._execute_inc(self.l)
        elif opcode == 0x34:
            addr = self._get_hl()
            self._write_byte(addr, self._execute_inc(self._read_byte(addr)))

        elif opcode == 0x3D: self.a = self._execute_dec(self.a)
        elif opcode == 0x05: self.b = self._execute_dec(self.b)
        elif opcode == 0x0D: self.c = self._execute_dec(self.c)
        elif opcode == 0x15: self.d = self._execute_dec(self.d)
        elif opcode == 0x1D: self.e = self._execute_dec(self.e)
        elif opcode == 0x25: self.h = self._execute_dec(self.h)
        elif opcode == 0x2D: self.l = self._execute_dec(self.l)
        elif opcode == 0x35:
            addr = self._get_hl()
            self._write_byte(addr, self._execute_dec(self._read_byte(addr)))

        elif opcode == 0xBF: self._execute_cp(self.a)
        elif opcode == 0xB8: self._execute_cp(self.b)
        elif opcode == 0xB9: self._execute_cp(self.c)
        elif opcode == 0xBA: self._execute_cp(self.d)
        elif opcode == 0xBB: self._execute_cp(self.e)
        elif opcode == 0xBC: self._execute_cp(self.h)
        elif opcode == 0xBD: self._execute_cp(self.l)
        elif opcode == 0xBE: self._execute_cp(self._read_byte(self._get_hl()))
        elif opcode == 0xFE: self._execute_cp(self._fetch_byte())

        # --- 16-bit Arithmetic ---
        elif opcode == 0x09: self._set_hl(self._execute_add_16(self._get_hl(), self._get_bc()))
        elif opcode == 0x19: self._set_hl(self._execute_add_16(self._get_hl(), self._get_de()))
        elif opcode == 0x29: self._set_hl(self._execute_add_16(self._get_hl(), self._get_hl()))
        elif opcode == 0x39: self._set_hl(self._execute_add_16(self._get_hl(), self.sp))
        elif opcode == 0x23: self._set_hl((self._get_hl() + 1) & 0xFFFF)
        elif opcode == 0x2B: self._set_hl((self._get_hl() - 1) & 0xFFFF)
        elif opcode == 0x13: self._set_de((self._get_de() + 1) & 0xFFFF)
        elif opcode == 0x1B: self._set_de((self._get_de() - 1) & 0xFFFF)
        elif opcode == 0x03: self._set_bc((self._get_bc() + 1) & 0xFFFF)
        elif opcode == 0x0B: self._set_bc((self._get_bc() - 1) & 0xFFFF)

        # --- Rotate Group ---
        elif opcode == 0x07:
            bit7 = (self.a >> 7) & 1
            self.a = ((self.a << 1) | bit7) & 0xFF
            self.f = (self.f & ~(self.FLAG_H | self.FLAG_N | self.FLAG_C)) | (bit7 * self.FLAG_C)
        
        elif opcode == 0x0F:
            bit0 = self.a & 1
            self.a = ((self.a >> 1) | (bit0 << 7)) & 0xFF
            self.f = (self.f & ~(self.FLAG_H | self.FLAG_N | self.FLAG_C)) | (bit0 * self.FLAG_C)

        elif opcode == 0x17:
            bit7 = (self.a >> 7) & 1
            old_c = 1 if (self.f & self.FLAG_C) else 0
            self.a = ((self.a << 1) | old_c) & 0xFF
            self.f = (self.f & ~(self.FLAG_H | self.FLAG_N | self.FLAG_C)) | (bit7 * self.FLAG_C)

        elif opcode == 0x1F:
            bit0 = self.a & 1
            old_c = 1 if (self.f & self.FLAG_C) else 0
            self.a = ((self.a >> 1) | (old_c << 7)) & 0xFF
            self.f = (self.f & ~(self.FLAG_H | self.FLAG_N | self.FLAG_C)) | (bit0 * self.FLAG_C)

        # --- Logic Group ---
        elif opcode == 0x2F: # CPL
            self.a = ~self.a & 0xFF
            self.f |= self.FLAG_H | self.FLAG_N
        elif opcode == 0xA7: self._execute_and(self.a)
        elif opcode == 0xA0: self._execute_and(self.b)
        elif opcode == 0xA1: self._execute_and(self.c)
        elif opcode == 0xA2: self._execute_and(self.d)
        elif opcode == 0xA3: self._execute_and(self.e)
        elif opcode == 0xA4: self._execute_and(self.h)
        elif opcode == 0xA5: self._execute_and(self.l)
        elif opcode == 0xA6: self._execute_and(self._read_byte(self._get_hl()))
        elif opcode == 0xE6: self._execute_and(self._fetch_byte())

        elif opcode == 0xB7: self._execute_or(self.a)
        elif opcode == 0xB0: self._execute_or(self.b)
        elif opcode == 0xB1: self._execute_or(self.c)
        elif opcode == 0xB2: self._execute_or(self.d)
        elif opcode == 0xB3: self._execute_or(self.e)
        elif opcode == 0xB4: self._execute_or(self.h)
        elif opcode == 0xB5: self._execute_or(self.l)
        elif opcode == 0xB6: self._execute_or(self._read_byte(self._get_hl()))
        elif opcode == 0xF6: self._execute_or(self._fetch_byte())

        elif opcode == 0xAF: self._execute_xor(self.a)
        elif opcode == 0xA8: self._execute_xor(self.b)
        elif opcode == 0xA9: self._execute_xor(self.c)
        elif opcode == 0xAA: self._execute_xor(self.d)
        elif opcode == 0xAB: self._execute_xor(self.e)
        elif opcode == 0xAC: self._execute_xor(self.h)
        elif opcode == 0xAD: self._execute_xor(self.l)
        elif opcode == 0xAE: self._execute_xor(self._read_byte(self._get_hl()))
        elif opcode == 0xEE: self._execute_xor(self._fetch_byte())

        # --- Jump Group ---
        elif opcode == 0xC3: self.pc = self._fetch_word()
        elif opcode == 0xE9: self.pc = self._get_hl() # JP (HL)
        
        elif opcode == 0xC2:
            addr = self._fetch_word()
            if not (self.f & self.FLAG_Z): self.pc = addr
        elif opcode == 0xCA:
            addr = self._fetch_word()
            if (self.f & self.FLAG_Z): self.pc = addr
        elif opcode == 0xD2:
            addr = self._fetch_word()
            if not (self.f & self.FLAG_C): self.pc = addr
        elif opcode == 0xDA:
            addr = self._fetch_word()
            if (self.f & self.FLAG_C): self.pc = addr
        elif opcode == 0xE0: # RET PO
            if not (self.f & self.FLAG_PV): self.pc = self._pop_from_stack()
        elif opcode == 0xF2:
            addr = self._fetch_word()
            if not (self.f & self.FLAG_S): self.pc = addr
        elif opcode == 0xFA:
            addr = self._fetch_word()
            if (self.f & self.FLAG_S): self.pc = addr

        elif opcode == 0x18:
            offset = self._fetch_byte()
            if offset > 127: offset -= 256
            self.pc = (self.pc + offset) & 0xFFFF
        
        elif opcode == 0x10: # DJNZ e
            offset = self._fetch_byte()
            self.b = (self.b - 1) & 0xFF
            if self.b != 0:
                if offset > 127: offset -= 256
                self.pc = (self.pc + offset) & 0xFFFF

        elif opcode == 0x20:
            offset = self._fetch_byte()
            if not (self.f & self.FLAG_Z):
                if offset > 127: offset -= 256
                self.pc = (self.pc + offset) & 0xFFFF
        elif opcode == 0x28:
            offset = self._fetch_byte()
            if (self.f & self.FLAG_Z):
                if offset > 127: offset -= 256
                self.pc = (self.pc + offset) & 0xFFFF
        elif opcode == 0x30:
            offset = self._fetch_byte()
            if not (self.f & self.FLAG_C):
                if offset > 127: offset -= 256
                self.pc = (self.pc + offset) & 0xFFFF
        elif opcode == 0x38:
            offset = self._fetch_byte()
            if (self.f & self.FLAG_C):
                if offset > 127: offset -= 256
                self.pc = (self.pc + offset) & 0xFFFF

        # --- Call & Return ---
        elif opcode == 0xCD:
            addr = self._fetch_word()
            self._push_to_stack(self.pc)
            self.pc = addr

        elif opcode == 0xC9: self.pc = self._pop_from_stack()
        elif opcode == 0xC0: # RET NZ
            if not (self.f & self.FLAG_Z): self.pc = self._pop_from_stack()
        elif opcode == 0xC8: # RET Z
            if (self.f & self.FLAG_Z): self.pc = self._pop_from_stack()
        elif opcode == 0xD0: # RET NC
            if not (self.f & self.FLAG_C): self.pc = self._pop_from_stack()
        elif opcode == 0xD8: # RET C
            if (self.f & self.FLAG_C): self.pc = self._pop_from_stack()
        elif opcode == 0xF0: # RET P
            if not (self.f & self.FLAG_S): self.pc = self._pop_from_stack()
        elif opcode == 0xF8: # RET M
            if (self.f & self.FLAG_S): self.pc = self._pop_from_stack()

        # --- RST ---
        elif opcode == 0xEF: # RST 28H
            self._push_to_stack(self.pc)
            self.pc = 0x0028
        elif opcode == 0xFF: # RST 38H
            self._push_to_stack(self.pc)
            self.pc = 0x0038

        # --- Stack ---
        elif opcode == 0xF5: self._push_to_stack(self._get_af())
        elif opcode == 0xC5: self._push_to_stack(self._get_bc())
        elif opcode == 0xD5: self._push_to_stack(self._get_de())
        elif opcode == 0xE5: self._push_to_stack(self._get_hl())

        elif opcode == 0xF1:
            val = self._pop_from_stack()
            self.a = (val >> 8) & 0xFF
            self.f = val & 0xFF
        elif opcode == 0xC1: self._set_bc(self._pop_from_stack())
        elif opcode == 0xD1: self._set_de(self._pop_from_stack())
        elif opcode == 0xE1: self._set_hl(self._pop_from_stack())

        # --- Control ---
        elif opcode == 0x00: pass
        elif opcode == 0x76: self.halted = True
        elif opcode == 0xF3: self.iff1 = self.iff2 = False
        elif opcode == 0xFB: self.iff1 = self.iff2 = True
        elif opcode == 0xD3:
            port = self._fetch_byte()
            self._write_port(port, self.a)
        elif opcode == 0xDB: # IN A, (n)
            port = self._fetch_byte()
            self.a = self._read_port(port)
            
        # === MISSING MAIN OPCODES ===

        elif opcode == 0x33: # INC SP
            self.sp = (self.sp + 1) & 0xFFFF

        elif opcode == 0x3B: # DEC SP
            self.sp = (self.sp - 1) & 0xFFFF

        elif opcode == 0x3F: # CCF - Complement Carry Flag (H = old carry)
            carry = self.f & self.FLAG_C
            self.f &= ~(self.FLAG_N | self.FLAG_H)
            if carry:
                self.f &= ~self.FLAG_C  # Clear carry
                self.f |= self.FLAG_H
            else:
                self.f |= self.FLAG_C   # Set carry

        elif opcode == 0x37: # SCF - Set Carry Flag
            self.f = (self.f & ~(self.FLAG_N | self.FLAG_H)) | self.FLAG_C

        elif opcode == 0xE2: # JP PO, nn
            addr = self._fetch_word()
            if not (self.f & self.FLAG_PV): self.pc = addr

        elif opcode == 0xEA: # JP PE, nn
            addr = self._fetch_word()
            if self.f & self.FLAG_PV: self.pc = addr

        elif opcode == 0xF9: # LD SP, HL
            self.sp = self._get_hl()

        elif opcode == 0x40: # LD B,B
            self.b = self.b  # NOP essentially

        elif opcode == 0x41: # LD B,C
            self.b = self.c

        elif opcode == 0x48: # LD C,B
            self.c = self.b

        elif opcode == 0x49: # LD C,C
            self.c = self.c  # NOP essentially

        elif opcode == 0x50: # LD D,B
            self.d = self.b

        elif opcode == 0x58: # LD E,B
            self.e = self.b

        elif opcode == 0x62: # LD H,D
            self.h = self.d

        elif opcode == 0x64: # LD H,H
            self.h = self.h  # NOP essentially

        elif opcode == 0x68: # LD L,B
            self.l = self.b

        elif opcode == 0x6C: # LD L,H
            self.l = self.h

        elif opcode == 0xC4: # CALL NZ,nn
            addr = self._fetch_word()
            if not (self.f & self.FLAG_Z):
                self._push_to_stack(self.pc)
                self.pc = addr
        
        elif opcode == 0xCC: # CALL Z,nn
            addr = self._fetch_word()
            if self.f & self.FLAG_Z:
                self._push_to_stack(self.pc)
                self.pc = addr

        elif opcode == 0xD4: # CALL NC,nn
            addr = self._fetch_word()
            if not (self.f & self.FLAG_C):
                self._push_to_stack(self.pc)
                self.pc = addr

        elif opcode == 0xDC: # CALL C,nn
            addr = self._fetch_word()
            if self.f & self.FLAG_C:
                self._push_to_stack(self.pc)
                self.pc = addr

        elif opcode == 0xDE: # SBC A,n
            value = self._fetch_byte()
            carry = 1 if (self.f & self.FLAG_C) else 0
            self._execute_sub_8(value, carry)

        elif opcode == 0xE3: # EX (SP),HL
            low = self._read_byte(self.sp)
            high = self._read_byte((self.sp + 1) & 0xFFFF)
            self._write_byte(self.sp, self.l)
            self._write_byte((self.sp + 1) & 0xFFFF, self.h)
            self.l = low
            self.h = high

        elif opcode == 0xE4: # CALL PO,nn
            addr = self._fetch_word()
            if not (self.f & self.FLAG_PV):
                self._push_to_stack(self.pc)
                self.pc = addr

        elif opcode == 0xE8: # RET PE
            if self.f & self.FLAG_PV:
                self.pc = self._pop_from_stack()

        elif opcode == 0xEC: # CALL PE,nn
            addr = self._fetch_word()
            if (self.f & self.FLAG_PV):
                self._push_to_stack(self.pc)
                self.pc = addr

        elif opcode == 0xF4: # CALL P,nn
            addr = self._fetch_word()
            if not (self.f & self.FLAG_S):
                self._push_to_stack(self.pc)
                self.pc = addr

        elif opcode == 0xFC: # CALL M,nn
            addr = self._fetch_word()
            if self.f & self.FLAG_S:
                self._push_to_stack(self.pc)
                self.pc = addr
                
        # --- RST Group ---
        elif opcode == 0xC7: # RST 00H
            self._push_to_stack(self.pc)
            self.pc = 0x0000
        elif opcode == 0xCF: # RST 08H
            self._push_to_stack(self.pc)
            self.pc = 0x0008
        elif opcode == 0xD7: # RST 10H
            self._push_to_stack(self.pc)
            self.pc = 0x0010
        elif opcode == 0xDF: # RST 18H
            self._push_to_stack(self.pc)
            self.pc = 0x0018
        elif opcode == 0xE7: # RST 20H
            self._push_to_stack(self.pc)
            self.pc = 0x0020
        elif opcode == 0xEF: # RST 28H
            self._push_to_stack(self.pc)
            self.pc = 0x0028
        elif opcode == 0xF7: # RST 30H
            self._push_to_stack(self.pc)
            self.pc = 0x0030
        elif opcode == 0xFF: # RST 38H
            self._push_to_stack(self.pc)
            self.pc = 0x0038

        else:
            opcode_str = f"MAIN {opcode:02X}"
            if self.skip_missing_opcodes:
                self.missing_opcodes.add(opcode_str)
                return
            else:
                addr = (self.pc - 1) & 0xFFFF
                print(f"Unhandled opcode: {opcode:02X} at address {addr:04X}")
                self.halted = True

    def _handle_cb_prefix(self):
        self.r = (self.r & 0x80) | ((self.r + 1) & 0x7F)
        opcode = self._fetch_byte()
        if (opcode & 0x07) != 6: self.cycles += 4
        elif (opcode & 0xC0) == 0x40: self.cycles += 8
        else: self.cycles += 11
        self._execute_cb(opcode)

    def _execute_cb(self, opcode: int):
        # BIT b, r
        if (opcode & 0xC0) == 0x40:
            bit = (opcode >> 3) & 0x07
            reg_idx = opcode & 0x07
            
            val = 0
            if reg_idx == 0: val = self.b
            elif reg_idx == 1: val = self.c
            elif reg_idx == 2: val = self.d
            elif reg_idx == 3: val = self.e
            elif reg_idx == 4: val = self.h
            elif reg_idx == 5: val = self.l
            elif reg_idx == 6: val = self._read_byte(self._get_hl())
            elif reg_idx == 7: val = self.a
            
            is_set = (val >> bit) & 1
            self.f = (self.f & self.FLAG_C) | self.FLAG_H
            if not is_set:
                self.f |= self.FLAG_Z | self.FLAG_PV
            elif bit == 7:
                self.f |= self.FLAG_S

        # RES b, r
        elif (opcode & 0xC0) == 0x80:
            bit = (opcode >> 3) & 0x07
            reg_idx = opcode & 0x07
            mask = ~(1 << bit)
            
            if reg_idx == 0: self.b &= mask
            elif reg_idx == 1: self.c &= mask
            elif reg_idx == 2: self.d &= mask
            elif reg_idx == 3: self.e &= mask
            elif reg_idx == 4: self.h &= mask
            elif reg_idx == 5: self.l &= mask
            elif reg_idx == 6: 
                addr = self._get_hl()
                self._write_byte(addr, self._read_byte(addr) & mask)
            elif reg_idx == 7: self.a &= mask

        # SET b, r
        elif (opcode & 0xC0) == 0xC0:
            bit = (opcode >> 3) & 0x07
            reg_idx = opcode & 0x07
            mask = 1 << bit
            
            if reg_idx == 0: self.b |= mask
            elif reg_idx == 1: self.c |= mask
            elif reg_idx == 2: self.d |= mask
            elif reg_idx == 3: self.e |= mask
            elif reg_idx == 4: self.h |= mask
            elif reg_idx == 5: self.l |= mask
            elif reg_idx == 6: 
                addr = self._get_hl()
                self._write_byte(addr, self._read_byte(addr) | mask)
            elif reg_idx == 7: self.a |= mask

        # RLC r
        elif (opcode & 0xF8) == 0x00:
            reg_idx = opcode & 0x07
            regs = [self.b, self.c, self.d, self.e, self.h, self.l, self._read_byte(self._get_hl()), self.a]
            val = regs[reg_idx]
            
            bit7 = (val >> 7) & 1
            res = ((val << 1) | bit7) & 0xFF
            
            self.f = bit7 * self.FLAG_C
            if res == 0: self.f |= self.FLAG_Z
            if res & 0x80: self.f |= self.FLAG_S
            if self.parity_table[res]: self.f |= self.FLAG_PV
            
            if reg_idx == 0: self.b = res
            elif reg_idx == 1: self.c = res
            elif reg_idx == 2: self.d = res
            elif reg_idx == 3: self.e = res
            elif reg_idx == 4: self.h = res
            elif reg_idx == 5: self.l = res
            elif reg_idx == 6: self._write_byte(self._get_hl(), res)
            elif reg_idx == 7: self.a = res

        # RRC r
        elif (opcode & 0xF8) == 0x08:
            reg_idx = opcode & 0x07
            regs = [self.b, self.c, self.d, self.e, self.h, self.l, self._read_byte(self._get_hl()), self.a]
            val = regs[reg_idx]

            bit0 = val & 1
            res = ((val >> 1) | (bit0 << 7)) & 0xFF

            self.f = bit0 * self.FLAG_C
            if res == 0: self.f |= self.FLAG_Z
            if res & 0x80: self.f |= self.FLAG_S
            if self.parity_table[res]: self.f |= self.FLAG_PV

            if reg_idx == 0: self.b = res
            elif reg_idx == 1: self.c = res
            elif reg_idx == 2: self.d = res
            elif reg_idx == 3: self.e = res
            elif reg_idx == 4: self.h = res
            elif reg_idx == 5: self.l = res
            elif reg_idx == 6: self._write_byte(self._get_hl(), res)
            elif reg_idx == 7: self.a = res

        # RL r
        elif (opcode & 0xF8) == 0x10:
            reg_idx = opcode & 0x07
            regs = [self.b, self.c, self.d, self.e, self.h, self.l, self._read_byte(self._get_hl()), self.a]
            val = regs[reg_idx]
            
            old_c = 1 if (self.f & self.FLAG_C) else 0
            new_c = (val >> 7) & 1
            res = ((val << 1) | old_c) & 0xFF
            
            self.f = new_c * self.FLAG_C
            if res == 0: self.f |= self.FLAG_Z
            if res & 0x80: self.f |= self.FLAG_S
            if self.parity_table[res]: self.f |= self.FLAG_PV

            if reg_idx == 0: self.b = res
            elif reg_idx == 1: self.c = res
            elif reg_idx == 2: self.d = res
            elif reg_idx == 3: self.e = res
            elif reg_idx == 4: self.h = res
            elif reg_idx == 5: self.l = res
            elif reg_idx == 6: self._write_byte(self._get_hl(), res)
            elif reg_idx == 7: self.a = res

        # RR r
        elif (opcode & 0xF8) == 0x18:
            reg_idx = opcode & 0x07
            regs = [self.b, self.c, self.d, self.e, self.h, self.l, self._read_byte(self._get_hl()), self.a]
            val = regs[reg_idx]

            old_c = 1 if (self.f & self.FLAG_C) else 0
            new_c = val & 1
            res = ((val >> 1) | (old_c << 7)) & 0xFF

            self.f = new_c * self.FLAG_C
            if res == 0: self.f |= self.FLAG_Z
            if res & 0x80: self.f |= self.FLAG_S
            if self.parity_table[res]: self.f |= self.FLAG_PV

            if reg_idx == 0: self.b = res
            elif reg_idx == 1: self.c = res
            elif reg_idx == 2: self.d = res
            elif reg_idx == 3: self.e = res
            elif reg_idx == 4: self.h = res
            elif reg_idx == 5: self.l = res
            elif reg_idx == 6: self._write_byte(self._get_hl(), res)
            elif reg_idx == 7: self.a = res
        
        # SLA r
        elif (opcode & 0xF8) == 0x20:
            reg_idx = opcode & 0x07
            regs = [self.b, self.c, self.d, self.e, self.h, self.l, self._read_byte(self._get_hl()), self.a]
            val = regs[reg_idx]

            new_c = (val >> 7) & 1
            res = (val << 1) & 0xFF

            self.f = new_c * self.FLAG_C
            if res == 0: self.f |= self.FLAG_Z
            if res & 0x80: self.f |= self.FLAG_S
            if self.parity_table[res]: self.f |= self.FLAG_PV

            if reg_idx == 0: self.b = res
            elif reg_idx == 1: self.c = res
            elif reg_idx == 2: self.d = res
            elif reg_idx == 3: self.e = res
            elif reg_idx == 4: self.h = res
            elif reg_idx == 5: self.l = res
            elif reg_idx == 6: self._write_byte(self._get_hl(), res)
            elif reg_idx == 7: self.a = res

        # SRA r
        elif (opcode & 0xF8) == 0x28:
            reg_idx = opcode & 0x07
            regs = [self.b, self.c, self.d, self.e, self.h, self.l, self._read_byte(self._get_hl()), self.a]
            val = regs[reg_idx]

            new_c = val & 1
            res = ((val >> 1) | (val & 0x80)) & 0xFF

            self.f = new_c * self.FLAG_C
            if res == 0: self.f |= self.FLAG_Z
            if res & 0x80: self.f |= self.FLAG_S
            if self.parity_table[res]: self.f |= self.FLAG_PV

            if reg_idx == 0: self.b = res
            elif reg_idx == 1: self.c = res
            elif reg_idx == 2: self.d = res
            elif reg_idx == 3: self.e = res
            elif reg_idx == 4: self.h = res
            elif reg_idx == 5: self.l = res
            elif reg_idx == 6: self._write_byte(self._get_hl(), res)
            elif reg_idx == 7: self.a = res

        # SLL r (undocumented: shift left, bit 0 = 1)
        elif (opcode & 0xF8) == 0x30:
            reg_idx = opcode & 0x07
            regs = [self.b, self.c, self.d, self.e, self.h, self.l, self._read_byte(self._get_hl()), self.a]
            val = regs[reg_idx]

            new_c = (val >> 7) & 1
            res = ((val << 1) | 1) & 0xFF

            self.f = new_c * self.FLAG_C
            if res == 0: self.f |= self.FLAG_Z
            if res & 0x80: self.f |= self.FLAG_S
            if self.parity_table[res]: self.f |= self.FLAG_PV

            if reg_idx == 0: self.b = res
            elif reg_idx == 1: self.c = res
            elif reg_idx == 2: self.d = res
            elif reg_idx == 3: self.e = res
            elif reg_idx == 4: self.h = res
            elif reg_idx == 5: self.l = res
            elif reg_idx == 6: self._write_byte(self._get_hl(), res)
            elif reg_idx == 7: self.a = res

        # SRL r
        elif (opcode & 0xF8) == 0x38:
            reg_idx = opcode & 0x07
            regs = [self.b, self.c, self.d, self.e, self.h, self.l, self._read_byte(self._get_hl()), self.a]
            val = regs[reg_idx]

            new_c = val & 1
            res = (val >> 1) & 0xFF

            self.f = new_c * self.FLAG_C
            if res == 0: self.f |= self.FLAG_Z
            if res & 0x80: self.f |= self.FLAG_S
            if self.parity_table[res]: self.f |= self.FLAG_PV

            if reg_idx == 0: self.b = res
            elif reg_idx == 1: self.c = res
            elif reg_idx == 2: self.d = res
            elif reg_idx == 3: self.e = res
            elif reg_idx == 4: self.h = res
            elif reg_idx == 5: self.l = res
            elif reg_idx == 6: self._write_byte(self._get_hl(), res)
            elif reg_idx == 7: self.a = res

        else:
            opcode_str = f"CB {opcode:02X}"
            if self.skip_missing_opcodes:
                self.missing_opcodes.add(opcode_str)
                return
            else:
                print(f"Unhandled opcode: CB {opcode:02X} at address {self.pc-2:04X}")
                self.halted = True

    def _handle_ed_prefix(self):
        sub_opcode = self._fetch_byte()
        self.r = (self.r & 0x80) | ((self.r + 1) & 0x7F)
        self.cycles += self.ed_cycle_table[sub_opcode]
        if sub_opcode == 0x44: self._execute_neg()
        elif sub_opcode == 0x4D: # RETI
            self.pc = self._pop_from_stack()
            self.iff1 = self.iff2
        elif sub_opcode == 0x52: # SBC HL, DE
            self._set_hl(self._execute_sbc_16(self._get_hl(), self._get_de()))
        elif sub_opcode == 0x53: # LD (nn), DE
            addr = self._fetch_word()
            self._write_byte(addr, self.e)
            self._write_byte((addr + 1) & 0xFFFF, self.d)
        elif sub_opcode == 0x5A: # ADC HL, DE
            self._set_hl(self._execute_adc_16(self._get_hl(), self._get_de()))
        elif sub_opcode == 0x5B: # LD DE, (nn)
            addr = self._fetch_word()
            low = self._read_byte(addr)
            high = self._read_byte((addr + 1) & 0xFFFF)
            self._set_de((high << 8) | low)
        elif sub_opcode == 0x56: self.im = 1
        elif sub_opcode == 0x5E: self.im = 2
        elif sub_opcode == 0x46: self.im = 0
        elif sub_opcode == 0x5F: # LD A, R
            self.a = self.r
            self.f &= ~(self.FLAG_S | self.FLAG_Z | self.FLAG_N | self.FLAG_H | self.FLAG_PV)
            if self.a & 0x80: self.f |= self.FLAG_S
            if self.a == 0: self.f |= self.FLAG_Z
            if self.iff2: self.f |= self.FLAG_PV
        elif sub_opcode == 0x79: self._write_port(self.c, self.a) # OUT (C), A
        else:
            self._execute_ed_extended(sub_opcode)

    def _set_in_flags(self, val: int):
        """Flags for IN r,(C) / RRD / RLD: S, Z, P/V = parity, H = N = 0, C preserved."""
        self.f &= self.FLAG_C
        if val == 0: self.f |= self.FLAG_Z
        if val & 0x80: self.f |= self.FLAG_S
        if self.parity_table[val]: self.f |= self.FLAG_PV

    def _set_sp(self, value: int):
        self.sp = value & 0xFFFF

    def _execute_ed_extended(self, op: int):
        """Remaining ED opcodes. Undefined ED opcodes behave as a NOP on a real Z80."""
        if 0x40 <= op <= 0x7F:
            r = ('b', 'c', 'd', 'e', 'h', 'l', None, 'a')[(op >> 3) & 7]
            get_rr, set_rr = ((self._get_bc, self._set_bc), (self._get_de, self._set_de),
                              (self._get_hl, self._set_hl), (lambda: self.sp, self._set_sp))[(op >> 4) & 3]
            low = op & 0x0F
            col = op & 0x07
            if col == 0:    # IN r, (C)
                val = self._read_port(self.c)
                self._set_in_flags(val)
                if r: setattr(self, r, val)
            elif col == 1:  # OUT (C), r
                self._write_port(self.c, getattr(self, r) if r else 0)
            elif low == 0x02:   # SBC HL, rr
                self._set_hl(self._execute_sbc_16(self._get_hl(), get_rr()))
            elif low == 0x0A:   # ADC HL, rr
                self._set_hl(self._execute_adc_16(self._get_hl(), get_rr()))
            elif low == 0x03:   # LD (nn), rr
                addr = self._fetch_word()
                val = get_rr()
                self._write_byte(addr, val & 0xFF)
                self._write_byte((addr + 1) & 0xFFFF, val >> 8)
            elif low == 0x0B:   # LD rr, (nn)
                addr = self._fetch_word()
                set_rr(self._read_byte(addr) | (self._read_byte((addr + 1) & 0xFFFF) << 8))
            elif col == 4:  # NEG (mirrors)
                self._execute_neg()
            elif col == 5:  # RETN / RETI (mirrors)
                self.pc = self._pop_from_stack()
                self.iff1 = self.iff2
            elif col == 6:  # IM 0/1/2 (mirrors)
                self.im = (0, 0, 1, 2)[(op >> 3) & 3]
            elif op == 0x47: self.i = self.a          # LD I, A
            elif op == 0x4F: self.r = self.a          # LD R, A
            elif op == 0x57:                          # LD A, I
                self.a = self.i
                self.f &= self.FLAG_C
                if self.a & 0x80: self.f |= self.FLAG_S
                if self.a == 0: self.f |= self.FLAG_Z
                if self.iff2: self.f |= self.FLAG_PV
            elif op == 0x67 or op == 0x6F:            # RRD / RLD
                addr = self._get_hl()
                m = self._read_byte(addr)
                if op == 0x67:
                    new_m = ((self.a << 4) | (m >> 4)) & 0xFF
                    self.a = (self.a & 0xF0) | (m & 0x0F)
                else:
                    new_m = ((m << 4) | (self.a & 0x0F)) & 0xFF
                    self.a = (self.a & 0xF0) | (m >> 4)
                self._write_byte(addr, new_m)
                self._set_in_flags(self.a)
            return

        step = 1 if (op & 0x08) == 0 else -1
        repeat = op >= 0xB0

        if op in (0xA0, 0xA8, 0xB0, 0xB8):  # LDI / LDD / LDIR / LDDR
            while True:
                self._write_byte(self._get_de(), self._read_byte(self._get_hl()))
                self._set_hl((self._get_hl() + step) & 0xFFFF)
                self._set_de((self._get_de() + step) & 0xFFFF)
                self._set_bc((self._get_bc() - 1) & 0xFFFF)
                self.cycles += 12
                if not repeat or self._get_bc() == 0:
                    break
                self.cycles += 5
            self.f &= ~(self.FLAG_N | self.FLAG_H | self.FLAG_PV)
            if self._get_bc() != 0: self.f |= self.FLAG_PV
        elif op in (0xA1, 0xA9, 0xB1, 0xB9):  # CPI / CPD / CPIR / CPDR
            while True:
                val = self._read_byte(self._get_hl())
                res = (self.a - val) & 0xFF
                self._set_hl((self._get_hl() + step) & 0xFFFF)
                self._set_bc((self._get_bc() - 1) & 0xFFFF)
                self.f = (self.f & self.FLAG_C) | self.FLAG_N
                if res == 0: self.f |= self.FLAG_Z
                if res & 0x80: self.f |= self.FLAG_S
                if (self.a & 0x0F) < (val & 0x0F): self.f |= self.FLAG_H
                if self._get_bc() != 0: self.f |= self.FLAG_PV
                self.cycles += 12
                if not repeat or res == 0 or self._get_bc() == 0:
                    break
                self.cycles += 5
        elif op in (0xA2, 0xAA, 0xB2, 0xBA, 0xA3, 0xAB, 0xB3, 0xBB):  # INI/IND/OUTI/OUTD (+repeat)
            while True:
                if op & 0x01:   # OUT: B is decremented before the port write
                    val = self._read_byte(self._get_hl())
                    self.b = (self.b - 1) & 0xFF
                    self._write_port(self.c, val)
                    self._set_hl((self._get_hl() + step) & 0xFFFF)
                    k = val + self.l
                else:
                    val = self._read_port(self.c)
                    self._write_byte(self._get_hl(), val)
                    self._set_hl((self._get_hl() + step) & 0xFFFF)
                    self.b = (self.b - 1) & 0xFF
                    k = val + ((self.c + step) & 0xFF)
                self.cycles += 12
                if not repeat or self.b == 0:
                    break
                self.cycles += 5
            # Flags as documented in "The Undocumented Z80 Documented"
            self.f = 0
            if self.b == 0: self.f |= self.FLAG_Z
            if self.b & 0x80: self.f |= self.FLAG_S
            if val & 0x80: self.f |= self.FLAG_N
            if k > 0xFF: self.f |= self.FLAG_H | self.FLAG_C
            if self.parity_table[(k & 0x07) ^ self.b]: self.f |= self.FLAG_PV
        # else: undefined ED opcode -> NOP


    # --- DD / FD prefixes (IX / IY) ---
    # Main opcodes whose (HL) operand becomes (IX+d) / (IY+d)
    _INDEX_MEM_OPS = frozenset(
        [0x34, 0x35, 0x36] +
        [0x46 + 8 * i for i in range(8) if i != 6] +   # LD r, (HL)
        [0x70, 0x71, 0x72, 0x73, 0x74, 0x75, 0x77] +   # LD (HL), r
        [0x86 + 8 * i for i in range(8)])              # ALU (HL)
    # Main opcodes that use HL / H / L without (HL): HL is replaced by IX / IY
    _INDEX_HL_OPS = frozenset([
        0x09, 0x19, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x29, 0x2A, 0x2B, 0x2C, 0x2D, 0x2E, 0x39,
        0x44, 0x45, 0x4C, 0x4D, 0x54, 0x55, 0x5C, 0x5D, 0x60, 0x61, 0x62, 0x63, 0x64, 0x65, 0x67,
        0x68, 0x69, 0x6A, 0x6B, 0x6C, 0x6D, 0x6F, 0x7C, 0x7D,
        0x84, 0x85, 0x8C, 0x8D, 0x94, 0x95, 0x9C, 0x9D, 0xA4, 0xA5, 0xAC, 0xAD, 0xB4, 0xB5, 0xBC, 0xBD,
        0xE1, 0xE3, 0xE5, 0xE9, 0xF9])

    def _handle_ix_prefix(self):
        self.ix = self._execute_index(self.ix)

    def _handle_iy_prefix(self):
        self.iy = self._execute_index(self.iy)

    def _fetch_displacement(self) -> int:
        d = self._fetch_byte()
        return d - 256 if d > 127 else d

    def _execute_index(self, index: int) -> int:
        """
        Execute a DD/FD prefixed instruction using `index` (IX or IY) in place of HL.
        Returns the (possibly modified) index register value.
        """
        opcode = self._fetch_byte()
        self.r = (self.r & 0x80) | ((self.r + 1) & 0x7F)
        self.cycles += self.cycle_table[opcode]

        if opcode == 0xCB:
            # DD CB d op: rotate/shift/BIT/RES/SET on (IX+d)
            addr = (index + self._fetch_displacement()) & 0xFFFF
            cb_opcode = self._fetch_byte()
            self.cycles += 12 if (cb_opcode & 0xC0) == 0x40 else 15
            saved_hl = self._get_hl()
            self._set_hl(addr)
            self._execute_cb((cb_opcode & 0xF8) | 6)
            self._set_hl(saved_hl)
            reg_idx = cb_opcode & 0x07
            if reg_idx != 6 and (cb_opcode & 0xC0) != 0x40:
                # Undocumented: result is also copied into a register
                res = self._read_byte(addr)
                if reg_idx == 0: self.b = res
                elif reg_idx == 1: self.c = res
                elif reg_idx == 2: self.d = res
                elif reg_idx == 3: self.e = res
                elif reg_idx == 4: self.h = res
                elif reg_idx == 5: self.l = res
                elif reg_idx == 7: self.a = res
            return index

        if opcode in self._INDEX_MEM_OPS:
            addr = (index + self._fetch_displacement()) & 0xFFFF
            self.cycles += 5 if opcode == 0x36 else 8
            if opcode == 0x74:   # LD (IX+d), H
                self._write_byte(addr, self.h)
            elif opcode == 0x75: # LD (IX+d), L
                self._write_byte(addr, self.l)
            else:
                h, l = self.h, self.l
                self._set_hl(addr)
                self.opcode_table[opcode]()
                if opcode == 0x66:   # LD H, (IX+d)
                    self.l = l
                elif opcode == 0x6E: # LD L, (IX+d)
                    self.h = h
                else:
                    self.h, self.l = h, l
            return index

        if opcode in self._INDEX_HL_OPS:
            h, l = self.h, self.l
            self._set_hl(index)
            self.opcode_table[opcode]()
            index = self._get_hl()
            self.h, self.l = h, l
            return index

        # Any other opcode: prefix has no effect (includes chained DD/FD/ED prefixes)
        self.opcode_table[opcode]()
        return index

    def _push_to_stack(self, value: int):
        self.sp = (self.sp - 1) & 0xFFFF
        self._write_byte(self.sp, (value >> 8) & 0xFF)
        self.sp = (self.sp - 1) & 0xFFFF
        self._write_byte(self.sp, value & 0xFF)

    def _pop_from_stack(self) -> int:
        low = self._read_byte(self.sp)
        self.sp = (self.sp + 1) & 0xFFFF
        high = self._read_byte(self.sp)
        self.sp = (self.sp + 1) & 0xFFFF
        return (high << 8) | low

    def run_for_cycles(self, cycles: int):
        target_cycles = self.cycles + cycles
        if self.debug_log:
            while self.cycles < target_cycles:
                self.step()
            return
        # Same as calling step() in a loop, with the hot path inlined
        table = self.opcode_table
        cycle_table = self.cycle_table
        memory = self.memory
        while self.cycles < target_cycles:
            if self.halted:
                self.cycles += -(-(target_cycles - self.cycles) // 4) * 4   # HALT: 4 T-states per NOP
                return
            pc = self.pc
            if pc < 0x8000:
                opcode = memory[pc]
                self.pc = pc + 1
            else:
                opcode = self._fetch_byte()
            self.r = (self.r & 0x80) | ((self.r + 1) & 0x7F)
            self.cycles += cycle_table[opcode]
            table[opcode]()

    def get_state(self) -> Dict[str, int]:
        return {
            "A": self.a, "F": self.f, "B": self.b, "C": self.c,
            "D": self.d, "E": self.e, "H": self.h, "L": self.l,
            "PC": self.pc, "SP": self.sp, "IX": self.ix, "IY": self.iy,
        }

    def print_missing_opcodes_report(self):
        """Imprime relatório de TODOS os opcodes faltantes"""
        if not self.missing_opcodes:
            print("\n✅ Nenhum opcode faltante!")
            return
        
        print("\n" + "="*70)
        print(f"🔍 {len(self.missing_opcodes)} OPCODES ÚNICOS FALTANDO")
        print("="*70)
        
        # Agrupar por prefixo
        by_prefix = {'MAIN': [], 'ED': [], 'DD': [], 'FD': [], 'CB': []}
        
        for op in sorted(self.missing_opcodes):
            parts = op.split()
            prefix = parts[0]
            code = parts[1] if len(parts) > 1 else op
            if prefix in by_prefix:
                by_prefix[prefix].append(code)
        
        # Imprimir
        for prefix, opcodes in by_prefix.items():
            if opcodes:
                print(f"\n{prefix}: {' '.join(opcodes)}")
        
        print("\n" + "="*70)
