"""
Sound card output: an SDL audio callback fed from a ring buffer of int16 mono samples.
"""

import threading
import numpy as np

SAMPLE_RATE = 44100


class AudioOutput:
    """Streams int16 mono samples to the sound card through an SDL audio callback."""

    def __init__(self, sample_rate: int = SAMPLE_RATE, max_samples: int = SAMPLE_RATE // 4):
        self.buffer = bytearray()
        self.lock = threading.Lock()
        self.max_bytes = max_samples * 2
        self.device = None
        self.underruns = 0
        self.started_output = False
        try:
            import pygame
            from pygame._sdl2.audio import AudioDevice, AUDIO_S16, get_audio_device_names
            pygame.init()
            names = get_audio_device_names(False)   # pygame 2.6 rejects devicename=None
            self.device = AudioDevice(devicename=names[0] if names else None, iscapture=False,
                                      frequency=sample_rate, audioformat=AUDIO_S16, numchannels=1,
                                      chunksize=512, allowed_changes=0, callback=self._callback)
        except Exception as e:
            print(f"⚠️  Audio unavailable ({e}); running without sound.")
            self.device = None

    @property
    def active(self) -> bool:
        return self.device is not None

    def start(self):
        if self.device is not None:
            self.device.pause(0)
            self.started_output = True

    def _callback(self, device, stream):
        n = len(stream)
        with self.lock:
            chunk = self.buffer[:n]
            del self.buffer[:n]
        if len(chunk) < n:
            if self.started_output:
                self.underruns += 1
            chunk += bytes(n - len(chunk))   # underrun: pad with silence
        stream[:] = chunk

    def push(self, samples: np.ndarray):
        with self.lock:
            self.buffer += samples.tobytes()
            excess = len(self.buffer) - self.max_bytes
            if excess > 0:                    # after a long stall: drop the oldest audio
                del self.buffer[:excess & ~1]

    def trim(self, samples: int):
        """Drop the oldest audio so that at most `samples` remain queued."""
        with self.lock:
            excess = len(self.buffer) - samples * 2
            if excess > 0:
                del self.buffer[:excess]

    def buffered_samples(self) -> int:
        with self.lock:
            return len(self.buffer) // 2

    def close(self):
        if self.device is not None:
            self.device.pause(1)
            self.device.close()
            self.device = None
