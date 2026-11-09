# mame-python

An arcade emulator written in Python, in the spirit of [MAME](https://www.mamedev.org/):
reusable chip emulations (`chips/`) and one driver file per game (`drivers/`).

The first supported game is **Moon Patrol** (Irem, 1982), emulated bit-exact (Z80 main CPU,
M6803 sound CPU with two AY-3-8910 and two MSM5205), plus an optional GPU-rendered HD mode and
a "remake" look with redrawn graphics.

## Running

```
pip install -r requirements.txt
python mame.py                      # Moon Patrol
python mame.py -listfull            # supported games
python mame.py mpatrol --no-hd      # original graphics
python mame.py mpatrol --original-speed
python mame.py mpatrol --replay=tools/bot/mpatrol_replay.json   # watch a recorded game (F6 = fast forward)
```

**ROMs are not included.** Use your own legally obtained `mpatrol` ROM set and put its files
in `roms/moonpatrol/` (or `roms/mpatrol/`).

Keys: F2 scanlines, F3 video filter, F4 HD pack, F5 remake look, F6 fast forward.

## HD pack and remake art

They are generated from your ROMs (they are not in the repository):

```
python tools/make_hdpack.py                 # hdpack/mpatrol: HD graphics
python tools/remake/mpatrol_remake.py       # hdpack/mpatrol: remake art (slow the first time)
```

Optional remake music: put `remake_music.mp3` (stage loop) and `remake_music_intro.mp3`
(start jingle) in `hdpack/mpatrol/`.

## Layout

| Path | Contents |
|---|---|
| `mame.py` | entry point |
| `emu/` | frontend (window, timing, input), driver base class, audio, ROM loading, music replacement |
| `chips/` | CPUs (Z80, M6803), sound (AY-3-8910, MSM5205), video helpers |
| `drivers/` | `mpatrol.py` (the machine), `irem_audio.py` (sound board), `mpatrol_hd.py` (GPU renderer) |
| `tools/bot/` | a bot that plays Moon Patrol and records replays; MP4 recording |
| `tools/remake/` | painters for the remake art (3D models, terrain, craters, fire...) |

## Bot and videos

```
python tools/bot/mpatrol_bot.py 36000 replay.json            # play, rewinding on deaths
python tools/bot/mpatrol_bot.py 120000 replay.json --resume --invincible
python tools/bot/mpatrol_video.py out.mp4 --replay=replay.json --from=180 --to=5600
```

`--invincible` uses the game's own cheat DIP switch; the setting is stored in the replay.
`tools/bot/mpatrol_replay_full.json` plays both courses (beginner and champion).

Moon Patrol is a trademark of its owners; this project is not affiliated with them or with MAME.
