"""
Moon Patrol Boot Sequence Analyzer
Analisa o que acontece durante o boot e por que pode estar pulando a tela de título
"""

import sys
from moon_patrol_rom_loader import MoonPatrolROMLoader
from z80_cpu import Z80


class BootAnalyzer:
    def __init__(self, rom_path: str):
        self.loader = MoonPatrolROMLoader(rom_path)
        if not self.loader.load_all():
            print("❌ Erro ao carregar ROMs!")
            sys.exit(1)

        # Inicializar memória e CPU
        self.memory = bytearray(65536)
        program_rom = self.loader.get_program_memory()
        self.memory[0x0000:len(program_rom)] = program_rom

        # Inputs (todos unpressed)
        self.memory[0xD000] = 0xFF  # IN0
        self.memory[0xD001] = 0xFF  # IN1
        self.memory[0xD002] = 0xFF  # IN2
        self.memory[0xD003] = 0xFF  # DSW1
        self.memory[0xD004] = 0xFF  # DSW2

        self.cpu = Z80(self.memory, debug_log=False)

        # Port handlers
        self.cpu._read_port = self.read_port
        self.cpu._write_port = self.write_port

        self.port_writes = []
        self.calls_made = []
        self.trace = []
        self.current_call_depth = 0

    def read_port(self, port):
        """Lê porta de I/O"""
        if port == 0xD0:
            return self.memory[0xD000]
        elif port == 0xD1:
            return self.memory[0xD001]
        elif port == 0xD2:
            return self.memory[0xD002]
        elif port == 0xD3:
            return self.memory[0xD003]
        elif port == 0xD4:
            return self.memory[0xD004]
        return 0xFF

    def write_port(self, port, value):
        """Escreve em porta de I/O"""
        self.port_writes.append((self.cpu.pc, port, value))

    def write_byte_hook(self, address, value):
        self.memory[address & 0xFFFF] = value & 0xFF
        if address == 0xE046:
            self.trace.append(f"WRITE $E046 = {value:02X}")

    def analyze_boot(self, max_cycles=100000):
        """Analisa o boot sequence"""
        print("\n🔍 Analisando Boot Sequence...\n")
        
        # Monitor memory writes
        self.cpu._write_byte = self.write_byte_hook

        prev_pc = 0
        cycle_count = 0
        call_addresses = []

        while cycle_count < max_cycles and not self.cpu.halted:
            current_pc = self.cpu.pc

            # Detectar CALLs importantes
            if current_pc != prev_pc:
                opcode = self.memory[current_pc]

                # Detectar CALL (0xCD)
                if opcode == 0xCD:
                    call_addr = self.memory[current_pc + 1] | (self.memory[current_pc + 2] << 8)
                    call_addresses.append((current_pc, call_addr, cycle_count))

                # Detectar JP (0xC3)
                if opcode == 0xC3:
                    jp_addr = self.memory[current_pc + 1] | (self.memory[current_pc + 2] << 8)
                    if cycle_count < 100:
                        print(f"  {cycle_count:5d} cycles: JP ${jp_addr:04X} (from ${current_pc:04X})")

            prev_pc = current_pc
            self.cpu.step()
            cycle_count += 1
            
            # Simulate VBLANK interrupt every ~51200 cycles (60Hz)
            if cycle_count % 51200 == 0:
                print(f"⚡ VBLANK Interrupt generated at cycle {cycle_count}")
                self.cpu.request_interrupt()

            # Parar se chegou no main loop (0x00B7)
            if current_pc == 0x00B7:
                print(f"\n✅ Chegou no main loop em ${current_pc:04X} após {cycle_count} cycles")
                break
        
            # Update trace
            self.trace.append(f"${current_pc:04X}")
            if len(self.trace) > 500: # Keep last 500
                self.trace.pop(0)

        print(f"\n📊 Total de cycles: {cycle_count}")
        print(f"📊 PC final: ${self.cpu.pc:04X}")
        
        with open("trace.txt", "w") as f:
            f.write("\n".join(self.trace))
        print("Trace salvo em trace.txt")

        # Análise de CALLs
        print(f"\n📞 CALLs importantes durante boot (primeiros 10):")
        for i, (from_addr, to_addr, cycles) in enumerate(call_addresses[:10]):
            print(f"  {i + 1:2d}. ${from_addr:04X} → CALL ${to_addr:04X} (cycle {cycles})")

        # Análise de port writes
        if self.port_writes:
            print(f"\n🔌 Escritas em portas I/O:")
            for pc, port, value in self.port_writes[:20]:
                print(f"  PC=${pc:04X}: OUT (${port:02X}), ${value:02X}")

        # Verificar áreas de memória modificadas
        print(f"\n💾 Verificando áreas de memória importantes:")
        self.check_memory_regions()

    def check_memory_regions(self):
        """Verifica se áreas importantes foram inicializadas"""
        regions = [
            (0x8000, 0x83FF, "Text VRAM"),
            (0x8400, 0x87FF, "Color RAM"),
            (0x9000, 0x93FF, "BG1 VRAM"),
            (0x9400, 0x97FF, "BG2 VRAM"),
            (0xC800, 0xC8FF, "Sprite RAM"),
        ]

        for start, end, name in regions:
            non_zero = sum(1 for b in self.memory[start:end + 1] if b != 0)
            total = end - start + 1
            pct = (non_zero / total) * 100

            status = "✅" if non_zero > 0 else "❌"
            print(f"  {status} {name:15s}: {non_zero:4d}/{total:4d} bytes não-zero ({pct:5.1f}%)")

    def find_title_screen_code(self):
        """Procura código relacionado à tela de título"""
        print(f"\n🔍 Procurando referências a 'INSERT COIN'...")

        # String está em 0x2C5A
        insert_coin_addr = 0x2C5A

        # Procurar por referências a este endereço
        for addr in range(0x0000, len(self.memory) - 2):
            word = self.memory[addr] | (self.memory[addr + 1] << 8)
            if word == insert_coin_addr:
                print(f"  Referência em ${addr:04X}")

    def analyze_inputs(self):
        """Analisa se inputs estão sendo lidos corretamente"""
        print(f"\n🎮 Estado dos inputs:")
        print(f"  IN0 (0xD000): ${self.memory[0xD000]:02X}")
        print(f"  IN1 (0xD001): ${self.memory[0xD001]:02X}")
        print(f"  IN2 (0xD002): ${self.memory[0xD002]:02X}")
        print(f"  DSW1 (0xD003): ${self.memory[0xD003]:02X}")
        print(f"  DSW2 (0xD004): ${self.memory[0xD004]:02X}")


def main():
    print("🎮 Moon Patrol Boot Sequence Analyzer\n")

    rom_path = "roms/moonpatrol/"

    if len(sys.argv) > 1:
        rom_path = sys.argv[1]

    analyzer = BootAnalyzer(rom_path)
    # Run for more cycles to see if it breaks the loop
    analyzer.analyze_boot(max_cycles=500000) 
    
    print("\n🔍 Desmontando loop ($12B0 - $12E0)...")
    for addr in range(0x12B0, 0x12E0):
        val = analyzer.memory[addr]
        print(f"  ${addr:04X}: {val:02X}")

    analyzer.find_title_screen_code()
    analyzer.analyze_inputs()
    
    print("\n" + "=" * 60)
    print("✅ Análise completa!")
    print("=" * 60)


if __name__ == "__main__":
    main()