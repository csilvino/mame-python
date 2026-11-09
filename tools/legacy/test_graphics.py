from moon_patrol_rom_loader import MoonPatrolROMLoader
import numpy as np

def main():
    print("Initializing Loader...")
    loader = MoonPatrolROMLoader("roms/moonpatrol/")
    if not loader.load_all():
        print("Failed to load ROMs")
        return

    print("Testing render_tile...")
    try:
        # Test text tile
        print("Rendering Text Tile 0...")
        tile = loader.render_tile(0, 'text', 0)
        max_val = np.max(tile)
        print(f"Text tile 0 rendered. Shape: {tile.shape}, Max val: {max_val}")
        
        # Test sprite tile
        print("Rendering Sprite Tile 0...")
        tile = loader.render_tile(0, 'sprite', 0)
        max_val = np.max(tile)
        print(f"Sprite tile 0 rendered. Shape: {tile.shape}, Max val: {max_val}")
        
        print("✅ render_tile test passed!")
    except Exception as e:
        print(f"❌ render_tile failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
