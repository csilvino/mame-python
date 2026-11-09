"""
Moon Patrol Emulator - Modo Debug
Captura TODOS os opcodes faltantes de uma vez
"""

import pygame
import numpy as np
from moon_patrol_rom_loader import MoonPatrolROMLoader
from z80_cpu import Z80


class MoonPatrolEmulatorDebug:
    """
    Versão DEBUG do emulador que captura todos os opcodes faltantes
    """

    def __init__(self, rom_loader: MoonPatrolROMLoader):
        self.loader = rom_loader

        # Display settings
        self.screen_width = 256
        self.screen_height = 240
        self.scale = 3
        pygame.init()
        self.display = pygame.display.set_mode(
            (self.screen_width * self.scale, self.screen_height * self.scale)
        )
        pygame.display.set_caption("Moon Patrol - Debug Mode")

        # Framebuffer
        self.framebuffer = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)

        # Input
        self.in0 = 0xFF
        self.in1 = 0xFF
        self.in2 = 0xFF
        self.dsw1 = 0xFF
        self.dsw2 = 0xFF

        # Memory & CPU
        self.memory = bytearray(65536)
        self.cpu = Z80(self.memory, debug_log=False)
        self.cpu._read_port = self._read_port

        # ATIVA MODO DEBUG
        self.cpu.skip_missing_opcodes = True

        self._initialize_memory()

        self.clock = pygame.time.Clock()
        self.running = True

        # Contador de frames para relatório
        self.frame_count = 0
        self.max_debug_frames = 300  # 5 segundos a 60fps

    def _initialize_memory(self):
        """Load ROM"""
        print("Initializing memory...")
        program_rom = self.loader.get_program_memory()
        self.memory[0x0000:len(program_rom)] = program_rom

    def _read_port(self, port: int) -> int:
        return 0xFF

    def update_memory_mapped_io(self):
        """Update input"""
        self.memory[0xD000] = self.in0
        self.memory[0xD001] = self.in1
        self.memory[0xD002] = self.in2
        self.memory[0xD003] = self.dsw1
        self.memory[0xD004] = self.dsw2

    def draw_debug_overlay(self):
        """Draw debug info"""
        self.framebuffer[:, :] = (20, 20, 40)  # Dark blue background

        # Title
        font = pygame.font.Font(None, 36)
        title = font.render("DEBUG MODE - Scanning Opcodes...", True, (255, 255, 0))

        # Progress
        progress = min(100, int((self.frame_count / self.max_debug_frames) * 100))

        font2 = pygame.font.Font(None, 28)
        info1 = font2.render(f"Progress: {progress}%", True, (255, 255, 255))
        info2 = font2.render(f"Frame: {self.frame_count}/{self.max_debug_frames}", True, (200, 200, 200))
        info3 = font2.render(f"Unique Opcodes Found: {len(self.cpu.missing_opcodes)}", True, (0, 255, 0))

        surface = pygame.surfarray.make_surface(
            np.transpose(self.framebuffer, (1, 0, 2))
        )
        scaled = pygame.transform.scale(
            surface,
            (self.screen_width * self.scale, self.screen_height * self.scale)
        )

        self.display.blit(scaled, (0, 0))
        self.display.blit(title, (100, 150))
        self.display.blit(info1, (250, 250))
        self.display.blit(info2, (250, 290))
        self.display.blit(info3, (180, 330))

        # Progress bar
        bar_width = 500
        bar_height = 30
        bar_x = (self.screen_width * self.scale - bar_width) // 2
        bar_y = 400

        pygame.draw.rect(self.display, (100, 100, 100), (bar_x, bar_y, bar_width, bar_height), 2)
        fill_width = int(bar_width * progress / 100)
        pygame.draw.rect(self.display, (0, 255, 0), (bar_x, bar_y, fill_width, bar_height))

        pygame.display.flip()

    def handle_events(self):
        """Handle events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False

    def run_debug_scan(self):
        """Run in debug mode to collect all missing opcodes"""
        print("\n" + "=" * 70)
        print("🔍 MOON PATROL - DEBUG MODE")
        print("=" * 70)
        print(f"\nScanning for missing opcodes...")
        print(f"Running for {self.max_debug_frames} frames (~5 seconds)")
        print("Press ESC to stop early\n")
        
        cycles_per_frame = 3072000 // 60
        
        try:
            while self.running and self.frame_count < self.max_debug_frames:
                self.handle_events()

                # Update input
                self.update_memory_mapped_io()

                # Run CPU
                if not self.cpu.halted:
                    self.cpu.run_for_cycles(cycles_per_frame)
                    self.cpu.request_interrupt()

                # Draw debug overlay
                self.draw_debug_overlay()

                self.frame_count += 1
                self.clock.tick(60)
        except KeyboardInterrupt:
             print("\nStopped by user.")
        finally:
            pygame.quit()

            # Print report
            print("\n✅ Scan complete!\n")
            self.cpu.print_missing_opcodes_report()


def main():
    """Main entry point"""
    print("Loading Moon Patrol ROMs...")
    loader = MoonPatrolROMLoader("roms/moonpatrol/")

    if not loader.load_all():
        print("\n❌ ERROR: Failed to load ROMs")
        return

    emulator = MoonPatrolEmulatorDebug(loader)
    emulator.run_debug_scan()


if __name__ == "__main__":
    main()