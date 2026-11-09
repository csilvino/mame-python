"""
Records a replay as an MP4 video: HD graphics with the remake artwork, the sound board's audio
and the remake music (hdpack/mpatrol/remake_music*.mp3, mixed like emu.music does live).

    python tools/bot/mpatrol_video.py [out.mp4] [--from=180] [--to=5600] [--replay=...] [--classic]

--classic: the arcade look and sound (HD pack without the remake). Needs imageio-ffmpeg.
"""

import os
import subprocess
import sys
import time
import wave

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import imageio_ffmpeg  # noqa: E402
import moderngl  # noqa: E402
import numpy as np  # noqa: E402
import pygame  # noqa: E402

from drivers.irem_audio import IremM52Sound  # noqa: E402
from drivers.mpatrol import MoonPatrol  # noqa: E402
from drivers.mpatrol_hd import MoonPatrolHD, load_pack  # noqa: E402
from emu import music  # noqa: E402
from emu.audio import SAMPLE_RATE  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mpatrol_bot import load_replay, replay_settings  # noqa: E402

W, H = 1028, 1080                # the game's aspect ratio, 1080 lines


class Voice:
    """One audio file playing in the offline mix, with fades."""

    def __init__(self, data):
        self.data, self.pos, self.gain, self.target, self.step, self.active = data, 0, 0.0, 0.0, 1.0, False

    def play(self, pos=0, fade_ms=0):
        self.pos, self.active = int(pos) % len(self.data), True
        self.gain, self.target = (0.0 if fade_ms else 1.0), 1.0
        self.step = 1.0 / max(1, fade_ms * SAMPLE_RATE // 1000)

    def fade(self, ms):
        if self.active:
            self.target, self.step = 0.0, self.gain / max(1, ms * SAMPLE_RATE // 1000)

    def render(self, n, loop):
        out = np.zeros(n)
        if not self.active:
            return out
        idx = self.pos + np.arange(n)
        if loop:
            idx %= len(self.data)
        else:
            idx = idx[idx < len(self.data)]
        out[:len(idx)] = self.data[idx]
        self.pos += n
        ramp = self.gain + np.sign(self.target - self.gain) * self.step * np.arange(1, n + 1)
        ramp = np.clip(ramp, min(self.gain, self.target), max(self.gain, self.target))
        self.gain = ramp[-1]
        out *= ramp
        if (self.target == 0.0 and self.gain <= 0.0) or (not loop and self.pos >= len(self.data)):
            self.active = False
        return out


class OfflineMusic:
    """emu.music.MusicReplacement's logic, rendered into samples instead of played."""

    def __init__(self, loop_path, intro_path, commands):
        pygame.mixer.init(SAMPLE_RATE)
        load = lambda p: pygame.sndarray.array(pygame.mixer.Sound(p)).astype(np.float64).reshape(-1, 2).mean(axis=1) / 32768
        self.loop = Voice(load(loop_path))
        self.intro = Voice(load(intro_path)) if intro_path else None
        self.loop_cmd, self.intro_cmd = commands["loop"], commands.get("intro")
        self.stops, self.game_over = set(commands["stops"]), commands.get("game_over")
        self.playing, self.position = False, 0

    def filter(self, commands):
        out = []
        for frac, value in commands:
            command, strobe = value & 0x7F, value & 0x80 == 0
            if strobe and command in (self.intro_cmd, self.game_over):
                self.position = 0
            if command == self.loop_cmd:
                if strobe and not self.playing:
                    intro_on = self.intro is not None and self.intro.active
                    if intro_on:
                        self.intro.fade(music.INTRO_FADE_MS)
                    self.loop.play(self.position, music.LOOP_FADE_IN_MS if intro_on or self.position else 0)
                self.playing = True
                continue
            if command == self.intro_cmd and self.intro is not None:
                if strobe:
                    self.intro.play()
                continue
            if command in self.stops and strobe:
                if self.loop.active and self.loop.target > 0:
                    self.position = self.loop.pos % len(self.loop.data)
                    self.loop.fade(music.FADE_MS)
                if self.intro is not None:
                    self.intro.fade(music.FADE_MS)
                if command == self.game_over:
                    self.position = 0
                self.playing = False
            out.append((frac, value))
        return out

    def render(self, n):
        mix = self.loop.render(n, loop=True)
        if self.intro is not None:
            mix += self.intro.render(n, loop=False)
        return mix * music.VOLUME


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) if "=" in a else (a[2:], "1") for a in sys.argv[1:] if a.startswith("--"))
    out = args[0] if args else os.path.join(ROOT, "videos", "mpatrol.mp4")
    first, last = int(opts.get("from", 180)), int(opts.get("to", 5600))
    remake = "classic" not in opts
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)

    m = MoonPatrol()
    if not m.load(os.path.join(ROOT, "roms")):
        sys.exit("ROMs not found")
    commands = []
    m.sound_sink = commands.extend
    board = IremM52Sound(bytes(m.sound_rom), sample_rate=SAMPLE_RATE)
    hdpack = os.path.join(ROOT, "hdpack")
    files = music.MusicReplacement.find(m, hdpack) if remake else None
    mus = OfflineMusic(*files, m.music_commands) if files else None

    ctx = moderngl.create_standalone_context()
    fbo = ctx.simple_framebuffer((W, H))
    fbo.use()
    hd = MoonPatrolHD(ctx, m, load_pack(os.path.join(hdpack, m.name)))
    hd.remake_enabled = remake

    replay = opts.get("replay", os.path.join(ROOT, "tools", "bot", "mpatrol_replay.json"))
    inputs = load_replay(replay)
    for name, value in replay_settings(replay).items():
        setattr(m, name, value)
    last = min(last, len(inputs))
    silent = out + ".video.mp4"
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    fps = str(m.refresh_rate)
    enc = subprocess.Popen([ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                            "-s", f"{W}x{H}", "-r", fps, "-i", "-", "-vf", "vflip", "-c:v", "libx264",
                            "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", silent],
                           stdin=subprocess.PIPE)
    audio = []
    frame_samples = SAMPLE_RATE / m.refresh_rate
    acc = 0.0
    t0 = time.time()
    for f, (in0, in1) in enumerate(inputs[:last], 1):
        m.in0, m.in1 = in0, in1
        commands.clear()
        m.run_frame()
        cmds = mus.filter(commands) if mus else list(commands)
        for frac, value in cmds:
            board.cmd_w(value, frac)
        samples = board.run_frame(1.0 / m.refresh_rate).astype(np.float64) / 32768
        # exactly frame_samples per frame on average (the video's clock)
        acc += frame_samples
        n = int(acc) - int(acc - frame_samples)
        samples = np.resize(samples, n) if len(samples) != n else samples
        if mus:
            samples = samples + mus.render(n)
        if f >= first:
            audio.append(samples)
            fbo.clear()
            hd.draw((0, 0, W, H), samples=2)
            enc.stdin.write(fbo.read(components=3))
        if f % 600 == 0:
            print(f"frame {f}/{last}  {time.time() - t0:.0f}s", flush=True)
    enc.stdin.close()
    enc.wait()

    pcm = np.clip(np.concatenate(audio), -1, 1)
    wav = out + ".wav"
    with wave.open(wav, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes((pcm * 32767).astype(np.int16).tobytes())
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", silent, "-i", wav, "-c:v", "copy",
                    "-c:a", "aac", "-b:a", "192k", "-shortest", out], check=True)
    os.remove(silent)
    os.remove(wav)
    print(f"video written to {out} ({(last - first + 1) / m.refresh_rate:.0f} s)")


if __name__ == "__main__":
    main()
