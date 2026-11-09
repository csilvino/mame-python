"""
Sound board in its own process.

On arcade hardware the sound board is usually a separate computer that only receives
command bytes from the main board. Here it runs the same way: a child process (its own
CPU core, no GIL sharing with the main CPU and the video) receives each frame's commands,
emulates the board and streams the audio straight to the sound card.

The video runs at its own rate, so the two clocks drift slightly. The sound process
compensates by producing up to +-0.5% more or fewer samples per frame depending on how full
the sound card's buffer is (dynamic rate control, inaudible).

Set SOUND_DEBUG=1 to print buffer statistics every 2 seconds.
"""

import importlib
import multiprocessing as mp
import os
import time
from emu.audio import AudioOutput, SAMPLE_RATE


def _sound_worker(board_spec, refresh_rate: float, conn):
    module_name, class_name, args = board_spec
    board_class = getattr(importlib.import_module(module_name), class_name)
    sound = board_class(*args, sample_rate=SAMPLE_RATE)
    audio = AudioOutput(SAMPLE_RATE)
    frame_seconds = 1.0 / refresh_rate
    frame_samples = SAMPLE_RATE / refresh_rate
    target = 4 * frame_samples                     # ~70 ms of queued audio
    started = False
    debug = bool(os.environ.get("SOUND_DEBUG") or os.environ.get("MPATROL_SOUND_DEBUG"))
    frames = 0
    last_report = time.perf_counter()

    while True:
        try:
            if started and not conn.poll(frame_seconds):
                # The main process is late (window being moved or resized...). A real sound
                # board keeps playing on its own, so run a frame without new commands rather
                # than let the sound card run dry.
                if audio.buffered_samples() >= target * 0.8:
                    continue
                commands = []
            else:
                commands = conn.recv()
        except (EOFError, OSError):
            break
        if commands is None:
            break
        for frac, value in commands:
            sound.cmd_w(value, frac)

        level = audio.buffered_samples()
        error = (target - level) / target          # >0: buffer running low
        scale = 1.0 + 0.005 * max(-1.0, min(1.0, error))
        samples = sound.run_frame(frame_seconds, scale)

        if audio.active:
            audio.push(samples)
            if audio.buffered_samples() > 2 * target:   # backlog (startup, stall): keep latency low
                audio.trim(int(target))
            if not started and audio.buffered_samples() >= target:
                audio.start()
                started = True
        frames += 1
        if debug and time.perf_counter() - last_report >= 2.0:
            last_report = time.perf_counter()
            print(f"[sound] frames {frames} buffer {audio.buffered_samples()} underruns {audio.underruns} scale {scale:.4f}", flush=True)
    audio.close()


class SoundProcess:
    """Main-process handle: send each frame's commands, the child does the rest."""

    def __init__(self, board_spec, refresh_rate: float):
        """board_spec: (module, class name, args) from Driver.sound_board()."""
        ctx = mp.get_context("spawn")
        self._recv, self._send = ctx.Pipe(duplex=False)
        self.process = ctx.Process(target=_sound_worker, args=(board_spec, refresh_rate, self._recv),
                                   daemon=True)
        self.process.start()

    def send_frame(self, commands):
        """commands: list of (fraction of frame 0..1, value) written during this frame."""
        try:
            self._send.send(commands)
        except (BrokenPipeError, OSError):
            pass

    def close(self):
        try:
            self._send.send(None)
        except (BrokenPipeError, OSError):
            pass
        self.process.join(timeout=2)
        if self.process.is_alive():
            self.process.terminate()
