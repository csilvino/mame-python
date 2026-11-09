"""
Moon Patrol ROM Analyzer
Ferramenta para examinar o conteúdo das ROMs e encontrar dados importantes
"""

import sys
from pathlib import Path
from moon_patrol_rom_loader import MoonPatrolROMLoader


class ROMAnalyzer:
    def __init__(self, rom_path: str):
        self.loader = MoonPatrolROMLoader(rom_path)
        if not self.loader.load_all():
            print("❌ Erro ao carregar ROMs!")
            sys.exit(1)

        self.program_rom = self.loader.get_program_memory()

    def dump_program_rom(self, output_file: str = "program_rom_dump.txt"):
        """Dump completo da Program ROM em formato legível"""
        print(f"📝 Gerando dump da Program ROM...")

        with open(output_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("MOON PATROL - PROGRAM ROM DUMP\n")
            f.write("=" * 80 + "\n\n")

            # Dump em formato hexadecimal
            for addr in range(0, len(self.program_rom), 16):
                # Endereço
                f.write(f"{addr:04X}: ")

                # Bytes em hex
                hex_bytes = []
                for i in range(16):
                    if addr + i < len(self.program_rom):
                        hex_bytes.append(f"{self.program_rom[addr + i]:02X}")
                    else:
                        hex_bytes.append("  ")
                f.write(" ".join(hex_bytes[:8]) + "  " + " ".join(hex_bytes[8:]))

                # ASCII representation
                f.write("  |")
                for i in range(16):
                    if addr + i < len(self.program_rom):
                        byte = self.program_rom[addr + i]
                        if 32 <= byte <= 126:
                            f.write(chr(byte))
                        else:
                            f.write(".")
                    else:
                        f.write(" ")
                f.write("|\n")

        print(f"✅ Dump salvo em: {output_file}")

    def find_text_strings(self, min_length: int = 4):
        """Procura strings de texto ASCII nas ROMs"""
        print(f"\n🔍 Procurando strings de texto (mín {min_length} chars)...\n")

        strings = []
        current_string = ""
        start_addr = 0

        for addr, byte in enumerate(self.program_rom):
            # Caracteres ASCII imprimíveis (incluindo espaço)
            if 32 <= byte <= 126:
                if not current_string:
                    start_addr = addr
                current_string += chr(byte)
            else:
                if len(current_string) >= min_length:
                    strings.append((start_addr, current_string))
                current_string = ""

        # Última string
        if len(current_string) >= min_length:
            strings.append((start_addr, current_string))

        # Mostrar strings encontradas
        for addr, string in strings:
            print(f"  {addr:04X}: \"{string}\"")

        print(f"\n✅ {len(strings)} strings encontradas")
        return strings

    def analyze_boot_code(self, max_instructions: int = 50):
        """Analisa as primeiras instruções do boot"""
        print(f"\n🔍 Analisando código de boot (primeiras {max_instructions} instruções)...\n")

        pc = 0
        count = 0

        while pc < len(self.program_rom) and count < max_instructions:
            opcode = self.program_rom[pc]
            instruction = f"{pc:04X}: {opcode:02X}"

            # Tentar identificar a instrução
            if opcode == 0x00:
                instruction += " NOP"
                pc += 1
            elif opcode == 0xC3:  # JP nn
                if pc + 2 < len(self.program_rom):
                    addr = self.program_rom[pc + 1] | (self.program_rom[pc + 2] << 8)
                    instruction += f" {self.program_rom[pc + 1]:02X} {self.program_rom[pc + 2]:02X}  JP ${addr:04X}"
                    pc += 3
                else:
                    pc += 1
            elif opcode == 0xCD:  # CALL nn
                if pc + 2 < len(self.program_rom):
                    addr = self.program_rom[pc + 1] | (self.program_rom[pc + 2] << 8)
                    instruction += f" {self.program_rom[pc + 1]:02X} {self.program_rom[pc + 2]:02X}  CALL ${addr:04X}"
                    pc += 3
                else:
                    pc += 1
            elif opcode == 0x31:  # LD SP, nn
                if pc + 2 < len(self.program_rom):
                    addr = self.program_rom[pc + 1] | (self.program_rom[pc + 2] << 8)
                    instruction += f" {self.program_rom[pc + 1]:02X} {self.program_rom[pc + 2]:02X}  LD SP,${addr:04X}"
                    pc += 3
                else:
                    pc += 1
            elif opcode == 0x21:  # LD HL, nn
                if pc + 2 < len(self.program_rom):
                    addr = self.program_rom[pc + 1] | (self.program_rom[pc + 2] << 8)
                    instruction += f" {self.program_rom[pc + 1]:02X} {self.program_rom[pc + 2]:02X}  LD HL,${addr:04X}"
                    pc += 3
                else:
                    pc += 1
            elif opcode == 0x3E:  # LD A, n
                if pc + 1 < len(self.program_rom):
                    instruction += f" {self.program_rom[pc + 1]:02X}     LD A,${self.program_rom[pc + 1]:02X}"
                    pc += 2
                else:
                    pc += 1
            else:
                instruction += "        ???"
                pc += 1

            print(instruction)
            count += 1

    def find_tile_references(self):
        """Procura referências a tiles e gráficos"""
        print(f"\n🔍 Procurando padrões de tiles...\n")

        # Procurar sequências que parecem ser tiles
        # Tiles geralmente têm padrões de 8 bytes

        patterns = {}
        for addr in range(0, len(self.program_rom) - 8, 8):
            chunk = self.program_rom[addr:addr + 8]

            # Verificar se parece um tile (tem variação mas não é aleatório)
            unique_bytes = len(set(chunk))
            if 2 <= unique_bytes <= 6:  # Tiles têm alguma variação
                pattern = tuple(chunk)
                if pattern not in patterns:
                    patterns[pattern] = []
                patterns[pattern].append(addr)

        # Mostrar padrões que aparecem múltiplas vezes
        print("Padrões de 8 bytes que se repetem:")
        for pattern, addresses in sorted(patterns.items(), key=lambda x: len(x[1]), reverse=True)[:10]:
            if len(addresses) > 2:
                hex_pattern = " ".join(f"{b:02X}" for b in pattern)
                print(f"  {hex_pattern} - aparece {len(addresses)}x em: {', '.join(f'{a:04X}' for a in addresses[:5])}")

    def check_palette_data(self):
        """Verifica os dados de paleta"""
        print(f"\n🎨 Analisando dados de paleta...\n")

        palette = self.loader.palette

        print(f"Total de cores na paleta: {len(palette)}")
        print("\nPrimeiras 32 cores:")
        for i in range(min(32, len(palette))):
            r, g, b = palette[i]
            # Converter 4-bit para 8-bit
            r8 = (r << 4) | r
            g8 = (g << 4) | g
            b8 = (b << 4) | b

            # Nome aproximado da cor
            color_name = "?"
            if r8 < 50 and g8 < 50 and b8 < 50:
                color_name = "Preto/Escuro"
            elif r8 > 200 and g8 > 200 and b8 > 200:
                color_name = "Branco/Claro"
            elif r8 > 200 and g8 < 100 and b8 < 100:
                color_name = "Vermelho"
            elif r8 < 100 and g8 > 200 and b8 < 100:
                color_name = "Verde"
            elif r8 < 100 and g8 < 100 and b8 > 200:
                color_name = "Azul"
            elif r8 > 200 and g8 > 200 and b8 < 100:
                color_name = "Amarelo"
            elif r8 < 100 and g8 > 200 and b8 > 200:
                color_name = "Ciano"
            elif r8 > 200 and g8 < 100 and b8 > 200:
                color_name = "Magenta"
            elif r8 > 150 and g8 > 100 and b8 < 100:
                color_name = "Laranja"
            elif r8 > 100 and g8 > 50 and b8 < 50:
                color_name = "Marrom"

            print(f"  [{i:2d}] R={r:X} G={g:X} B={b:X} → RGB({r8:3d},{g8:3d},{b8:3d}) - {color_name}")

    def memory_map_visualization(self):
        """Visualiza o memory map do jogo"""
        print(f"\n🗺️  MEMORY MAP VISUALIZATION\n")
        print("=" * 60)

        regions = [
            (0x0000, 0x3FFF, "Program ROM", "Código do jogo"),
            (0x8000, 0x83FF, "Video RAM (Text)", "Tiles de texto/foreground"),
            (0x8400, 0x87FF, "Color RAM", "Atributos de cor"),
            (0x8800, 0x8FFF, "Work RAM?", "Proteção/trabalho"),
            (0x9000, 0x93FF, "Video RAM (BG1)?", "Background 1 - Estrelas?"),
            (0x9400, 0x97FF, "Video RAM (BG2)?", "Background 2 - Montanhas?"),
            (0xC800, 0xC8FF, "Sprite RAM", "64 sprites × 4 bytes"),
            (0xD000, 0xD000, "IN0", "Input port 0"),
            (0xD001, 0xD001, "IN1", "Input port 1"),
            (0xD002, 0xD002, "IN2", "Input port 2"),
            (0xD003, 0xD003, "DSW1", "DIP switches 1"),
            (0xD004, 0xD004, "DSW2", "DIP switches 2"),
        ]

        for start, end, name, description in regions:
            size = end - start + 1
            print(f"${start:04X}-${end:04X} ({size:5d} bytes) : {name:20s} - {description}")

    def generate_full_report(self):
        """Gera relatório completo"""
        print("\n" + "=" * 80)
        print("MOON PATROL ROM ANALYSIS - RELATÓRIO COMPLETO")
        print("=" * 80)

        self.memory_map_visualization()
        self.check_palette_data()
        self.find_text_strings(min_length=3)
        self.analyze_boot_code(max_instructions=30)
        self.find_tile_references()
        self.dump_program_rom()

        print("\n" + "=" * 80)
        print("✅ Análise completa!")
        print("=" * 80)


def main():
    print("🎮 Moon Patrol ROM Analyzer\n")

    rom_path = "roms/moonpatrol/"

    if len(sys.argv) > 1:
        rom_path = sys.argv[1]

    analyzer = ROMAnalyzer(rom_path)
    analyzer.generate_full_report()


if __name__ == "__main__":
    main()