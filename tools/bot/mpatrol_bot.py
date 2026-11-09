"""
Moon Patrol bot: plays the game far, with save states, and records a replay.

    python tools/bot/mpatrol_bot.py [frames] [replay.json] [-v]
    python tools/bot/mpatrol_bot.py --check replay.json      replays it, counts deaths
    python tools/bot/mpatrol_bot.py 90000 replay.json --resume  plays the replay, then goes on
    --invincible: the game's own cheat DIP switch (DSW2 bit 6 low: the buggy is not destroyed by
    rocks, mines, shots...; checked at 0x16F6), stored in the replay as a setting

The player is simple (fires all the time, jumps when a crater or a rock comes, changes speed at
random); it does not need to be good: every second the machine state is saved, and when the
buggy is destroyed the bot goes back a few seconds and tries again with other random choices
(further back if it keeps failing at the same place). The inputs that survived are the replay:
replaying them from power on gives the same game, without a single death.

Replay file: {"game": "mpatrol", "frames": N, "ports": ["in0", "in1"], "inputs": [[count, in0, in1], ...]}
(run length), optional "settings": {"dsw2": value} applied from power on. Watch it with: python mame.py mpatrol --replay=tools/bot/mpatrol_replay.json
"""

import json
import os
import random
import sys
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from drivers.mpatrol import MoonPatrol, VIS_X0, VIS_Y0  # noqa: E402

SIMPLE = (int, bool, float)
# sprites on the ground that are not obstacles: the buggy, shots, hits, dust, scores
HARMLESS = set(range(0x01, 0x11)) | set(range(0x28, 0x2D)) | set(range(0x3E, 0x42)) | set(range(0x61, 0x70)) | {0x7D, 0x7E}
DEATH = set(range(0x11, 0x1D))                  # wreck and explosion of the buggy
FIRE, JUMP, RIGHT, LEFT = 0x80, 0x20, 0x01, 0x02
SNAP_EVERY = 60
INVINCIBLE = 0x40                               # DSW2 bit, active low


# --- Machine state -------------------------------------------------------------------------

def snapshot(m):
    return (bytes(m.memory), {k: v for k, v in vars(m.cpu).items() if type(v) in SIMPLE},
            {k: (list(v) if isinstance(v, list) else v) for k, v in vars(m).items()
             if type(v) in SIMPLE or k in ("bg_xpos", "bg_ypos")})


def restore(m, state):
    memory, cpu, driver = state
    m.memory[:] = memory
    for k, v in cpu.items():
        setattr(m.cpu, k, v)
    for k, v in driver.items():
        setattr(m, k, list(v) if isinstance(v, list) else v)
    m.sound_commands = []


def new_machine():
    m = MoonPatrol()
    if not m.load(os.path.join(ROOT, "roms")):
        sys.exit("ROMs not found")
    m.sound_sink = lambda c: None
    return m


# --- What the bot sees ----------------------------------------------------------------------

def sprites(m):
    mem = m.memory
    out = []
    for base in range(0xC800, 0xC900, 4):
        code, attr = mem[base + 2] & 0x7F, mem[base + 1]
        if code == 0 and attr == 0:
            continue
        out.append((code, mem[base + 3] + 129 - VIS_X0, 257 - mem[base] - VIS_Y0))
    return out


def hole_tiles(m):
    """Text codes with empty pixels in their top half: in ground row 28, a crater or a ditch
    (the stages use different tiles for them)."""
    if not hasattr(m, "_holes"):
        m._holes = {code for code in range(512) if (m.tiles_text[code][:4] == 0).any()}
    return m._holes


def craters_x(m):
    mem = m.memory
    holes = hole_tiles(m)
    xs = []
    for c in range(32):
        code = mem[0x8000 + 28 * 32 + c] | ((mem[0x8400 + 28 * 32 + c] & 0x80) << 1)
        if code in holes:
            xs.append((c * 8 - VIS_X0 + 128 + m.tx_scroll) % 256)
    return xs


def score(m):
    digits = [m.memory[0x8000 + 4 * 32 + c] - 0x30 for c in range(4, 10)]
    return int("".join(str(d) for d in digits)) if all(0 <= d <= 9 for d in digits) else 0


def checkpoint(m):
    """Letter of the last checkpoint passed (POINT x in the panel)."""
    code = m.memory[0x8000 + 2 * 32 + 18]
    return chr(code + 1) if 0x40 <= code < 0x5A else "?"


# --- Player ---------------------------------------------------------------------------------

class Player:
    def __init__(self, seed):
        self.rng = random.Random(seed)
        self.hold = 0
        self.lead = 4                         # frames before an obstacle reaches the buggy's front
        self.speed = 0
        self.cooldown = 0
        self.scroll = None                    # ground scroll of the previous frame
        self.v = 1.5                          # ground speed, native pixels per frame

    def reseed(self, seed):
        self.rng = random.Random(seed)
        self.hold = 0

    def __call__(self, f, m):
        rng = self.rng
        if f % 30 == 0:                       # new plan every half second
            self.lead = rng.uniform(0.0, 7.0)
            self.speed = rng.choice((0, 0, 0, RIGHT, LEFT))
        in1 = 0xFF
        if f % 8 < 2:
            in1 &= ~FIRE
        if self.speed:
            in1 &= ~self.speed
        # ground speed from the text layer scroll (a jump covers more ground when faster)
        if self.scroll is not None:
            dv = (m.tx_scroll - self.scroll + 128) % 256 - 128
            if 0 < abs(dv) < 8:
                self.v = 0.8 * self.v + 0.2 * abs(dv)
        self.scroll = m.tx_scroll
        sp = sprites(m)
        buggy = [s for s in sp if s[0] == 0x03]
        if buggy and self.hold == 0 and self.cooldown == 0:
            bx = buggy[0][1]
            holes = [x - bx for x in craters_x(m)]
            if sum(1 for d in holes if 0 < d < 120) >= 4:
                self.speed = RIGHT                # a wide ditch: full speed to jump over it
            ahead = list(holes)
            # rocks, boulders, mines, bomb blasts...: anything else low on the screen
            ahead += [s[1] - bx for s in sp if s[0] not in HARMLESS and s[2] > 175]
            # time until each obstacle reaches the front of the buggy (32 pixels long)
            soon = [(d - 32) / max(self.v, 0.5) for d in ahead]
            if any(self.lead <= t <= self.lead + 5 for t in soon) or rng.random() < 0.004:
                self.hold = rng.randint(4, 8)
                self.cooldown = 16
        if self.hold > 0:
            in1 &= ~JUMP
            self.hold -= 1
        if self.cooldown > 0:
            self.cooldown -= 1
        return in1


def start_inputs(f):
    """Coin at frame 200, start at 260 (in0, active low)."""
    in0 = 0xFF
    if 200 <= f < 206:
        in0 &= ~0x08
    if 260 <= f < 266:
        in0 &= ~0x01
    return in0


# --- Search ---------------------------------------------------------------------------------

def play(frames, log_every=1800, verbose=False, out=None, resume=None, settings=None):
    """Searches a game without deaths for `frames` frames; saves the replay so far to `out` at
    each progress line (a long search can be stopped and still give a replay)."""
    m = new_machine()
    for name, value in (settings or {}).items():
        setattr(m, name, value)
    player = Player(1)
    inputs = []                 # (in0, in1) for frames 1..
    snaps = {}
    fails = {}                  # death place (second) -> failures
    f = 0
    restores = 0
    t0 = time.time()
    # --resume: the replay's inputs first (no deaths there), the search goes on from its end
    for in0, in1 in resume or ():
        if f % SNAP_EVERY == 0:
            snaps[f] = (snapshot(m), 0, 0)
        f += 1
        m.in0, m.in1 = in0, in1
        inputs.append((in0, in1))
        m.run_frame()
        if f > 300:
            player(f, m)                      # keeps its view of the ground speed up to date
    player.hold = player.cooldown = 0
    if resume:
        print(f"resumed at frame {f}  point {checkpoint(m)}  score {score(m)}", flush=True)
    while f < frames:
        if f % SNAP_EVERY == 0:
            snaps[f] = (snapshot(m), player.hold, player.cooldown)
        f += 1
        in0, in1 = start_inputs(f), (player(f, m) if f > 300 else 0xFF)
        m.in0, m.in1 = in0, in1
        inputs.append((in0, in1))
        m.run_frame()
        if f > 300 and any(s[0] in DEATH for s in sprites(m)):
            place = f // 60
            fails[place] = fails.get(place, 0) + 1
            back = 120 + 60 * fails[place]
            target = max(k for k in snaps if k <= max(0, f - back))
            state, player.hold, player.cooldown = snaps[target]
            restore(m, state)
            for k in [k for k in snaps if k > target]:
                del snaps[k]
            del inputs[target:]
            f = target
            restores += 1
            if verbose:
                print(f"  died at {place * 60}, back to {target} (try {fails[place]})", flush=True)
            player.reseed(restores * 7919 + f)
            continue
        if f % log_every == 0:
            print(f"frame {f:6d}  point {checkpoint(m)}  score {score(m):6d}  restores {restores:4d}"
                  f"  {time.time() - t0:6.0f}s", flush=True)
            if out:
                save_replay(inputs, out, settings)
    return inputs


def save_replay(inputs, path, settings=None):
    rle = []
    for in0, in1 in inputs:
        if rle and rle[-1][1] == in0 and rle[-1][2] == in1:
            rle[-1][0] += 1
        else:
            rle.append([1, in0, in1])
    with open(path, "w", encoding="utf-8") as f:
        data = {"game": "mpatrol", "frames": len(inputs), "ports": ["in0", "in1"], "inputs": rle}
        if settings:
            data["settings"] = settings
        json.dump(data, f)


def load_replay(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [(in0, in1) for count, in0, in1 in data["inputs"] for _ in range(count)]


def replay_settings(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f).get("settings", {})


def check(path):
    """Plays a replay from power on: counts the buggy's deaths, prints where it ends."""
    m = new_machine()
    for name, value in replay_settings(path).items():
        setattr(m, name, value)
    deaths, dead = 0, False
    inputs = load_replay(path)
    for f, (in0, in1) in enumerate(inputs, 1):
        m.in0, m.in1 = in0, in1
        m.run_frame()
        now = any(s[0] in DEATH for s in sprites(m))
        deaths += now and not dead
        dead = now
        if f % 3600 == 0:
            print(f"frame {f:6d}  point {checkpoint(m)}  score {score(m):6d}  deaths {deaths}", flush=True)
    print(f"end: frame {len(inputs)}  point {checkpoint(m)}  score {score(m)}  deaths {deaths}")
    return deaths


if __name__ == "__main__":
    if "--check" in sys.argv:
        sys.exit(1 if check(sys.argv[sys.argv.index("--check") + 1]) else 0)
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    frames = int(args[0]) if args else 36000
    out = args[1] if len(args) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "mpatrol_replay.json")
    resume = load_replay(out) if "--resume" in sys.argv and os.path.exists(out) else None
    settings = replay_settings(out) if resume else {}
    if "--invincible" in sys.argv:
        settings["dsw2"] = MoonPatrol().dsw2 & ~INVINCIBLE
    inputs = play(frames, verbose="-v" in sys.argv, out=out, resume=resume, settings=settings)
    save_replay(inputs, out, settings)
    print("replay written to", out, f"({len(inputs)} frames)")
