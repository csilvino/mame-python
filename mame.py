"""
Python arcade emulator in the spirit of MAME.

    python mame.py                     run Moon Patrol
    python mame.py mpatrol             run a game by its short name
    python mame.py -listfull           list the supported games
    python mame.py mpatrol --original-speed
                                       exact arcade frame rate instead of syncing to the monitor
    python mame.py mpatrol --filter=nearest
                                       start with another video filter (nearest, smooth, xbr)
    python mame.py mpatrol --no-hd     start with the original graphics instead of the HD pack
    python mame.py mpatrol --replay=tools/bot/mpatrol_replay.json
                                       watch a recorded game (F6 = fast forward)
"""

import os
import sys

from drivers import DRIVERS, get_driver

ROMS_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "roms")
HDPACK_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hdpack")


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    flags = {a for a in args if a.startswith("-")}
    names = [a for a in args if not a.startswith("-")]

    if "-listfull" in flags or "--list" in flags:
        for short_name in DRIVERS:
            driver_class = get_driver(short_name)
            print(f"{short_name:12} \"{driver_class.description}\" ({driver_class.manufacturer} {driver_class.year})")
        return 0

    name = names[0] if names else "mpatrol"
    if name not in DRIVERS:
        print(f"Unknown game '{name}'. Supported: {', '.join(DRIVERS)}")
        return 1

    driver = get_driver(name)()
    print(f"Loading {driver.description}...")
    if not driver.load(ROMS_ROOT):
        return 1

    replay = None
    replay_path = next((f.split("=", 1)[1] for f in flags if f.startswith("--replay=")), None)
    if replay_path:
        import json
        with open(replay_path, encoding="utf-8") as f:
            replay = json.load(f)
        print(f"Replay: {replay_path} ({replay['frames']} frames)")

    from emu.frontend import Frontend
    video_filter = next((f.split("=", 1)[1] for f in flags if f.startswith("--filter=")), "nearest")
    Frontend(driver, original_speed="--original-speed" in flags, video_filter=video_filter,
             hdpack_root=HDPACK_ROOT, hd="--no-hd" not in flags, replay=replay).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
