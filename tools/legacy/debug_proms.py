from moon_patrol_rom_loader import MoonPatrolROMLoader

def debug_prom_colors(name, data):
    print(f"\n--- {name} ({len(data)} bytes) ---")
    # Interpret as RGB (3-3-2)
    for i, b in enumerate(data):
        r = ((b >> 0) & 1) * 36 + ((b >> 1) & 1) * 73 + ((b >> 2) & 1) * 146
        g = ((b >> 3) & 1) * 36 + ((b >> 4) & 1) * 73 + ((b >> 5) & 1) * 146
        b_val = ((b >> 6) & 1) * 85 + ((b >> 7) & 1) * 170
        print(f"[{i:02X}] {b:02X} -> RGB({r},{g},{b_val})")
        if i >= 31: break

def main():
    loader = MoonPatrolROMLoader("roms/moonpatrol/")
    loader.load_all()
    
    # Analyze 32-byte PROMs (likely RGB Palettes)
    if loader.prom_sprite_pal:
        debug_prom_colors("mpc-1.1f (Sprite/Char RGB?)", loader.prom_sprite_pal)
    
    if loader.prom_bg_pal:
        debug_prom_colors("mpc-3.1m (BG RGB?)", loader.prom_bg_pal)
        
    # Analyze CLUTs
    if loader.prom_sprite_clut:
        print(f"\n--- mpc-2.2h (Sprite CLUT) ---")
        print([hex(b) for b in loader.prom_sprite_clut[:32]])

    if loader.prom_text_pal:
        print(f"\n--- mpc-4.2a (Text Palette - First 64 bytes) ---")
        debug_prom_colors("mpc-4.2a", loader.prom_text_pal[:64])

    # Dump Color RAM (simulate loader memory or just check ROMs?)
    # We can't check RAM unless we run the emulator.
    # But we can check the *default* values if they were in ROM, but they are in RAM initialized by code.
    # So we can't dump RAM here easily without running the CPU.
    # However, we can look at the pattern of mpc-4.
    pass

if __name__ == "__main__":
    main()
