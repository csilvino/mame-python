"""
Music replacement for the remake: audio files play instead of the game's own music, while the
sound board keeps making every other sound.
    hdpack/<game>/remake_music.mp3         the background music (looped)
    hdpack/<game>/remake_music_intro.mp3   optional: the tune before it (start of the game)

The main CPU starts and stops the music with sound commands (Driver.music_commands: the loop's
start command, the intro's, and the commands after which the music is over). While the
replacement is on, the start commands never reach the board and the files are played instead;
the stop commands stop them. The loop starts when the game starts its music; an intro file
longer than the game's intro fades out then. The state of the music is followed all the
time, so switching the remake on or off in the middle of a stage hands the music over from one
to the other. After a crash or a stage end the loop goes on from where it stopped; a new game
(intro, game over) starts it over.
"""

import os

import pygame

VOLUME = 0.55
FADE_MS = 400            # music stopped by the game (crash, stage end...)
INTRO_FADE_MS = 900      # the intro, when the loop starts
LOOP_FADE_IN_MS = 250


class MusicReplacement:
    def __init__(self, loop_path: str, intro_path, commands: dict, sink):
        """commands: {"loop": command, "intro": command or None, "stops": [...], "game_over": command};
        sink: where the (filtered) sound commands go, the sound board."""
        self.loop_cmd, self.intro_cmd = commands["loop"], commands.get("intro")
        self.stops, self.sink = set(commands["stops"]), sink
        self.game_over = commands.get("game_over")
        self.enabled = False
        self.playing = False           # the game's loop music is on (whoever plays it)
        self.position = 0.0            # where the loop file stopped (seconds): it goes on from there
        self.started_at = 0.0          # position at the last play()
        self.loaded = False
        self.intro = None
        self.intro_channel = None
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(44100)
            pygame.mixer.music.load(loop_path)
            pygame.mixer.music.set_volume(VOLUME)
            self.length = pygame.mixer.Sound(loop_path).get_length()   # (decoded once, for the wrap)
            if intro_path:
                self.intro = pygame.mixer.Sound(intro_path)
                self.intro.set_volume(VOLUME)
            self.loaded = True
        except pygame.error as e:
            print(f"Remake music not available ({e})")

    @staticmethod
    def find(driver, hdpack_root: str):
        """(loop file, intro file or None), or None without a loop file."""
        base = os.path.join(hdpack_root, driver.name, "remake_music")
        if not os.path.isfile(base + ".mp3"):
            return None
        return base + ".mp3", (base + "_intro.mp3" if os.path.isfile(base + "_intro.mp3") else None)

    def _intro_on(self):
        return self.intro_channel is not None and self.intro_channel.get_busy()

    def _play_loop(self, fade_ms=0):
        """The loop file, from where it stopped."""
        pygame.mixer.music.play(loops=-1, start=self.position, fade_ms=fade_ms)
        self.started_at = self.position

    def _start_loop(self):
        # on time, with the game: the intro fades out while the loop fades in
        if self._intro_on():
            self.intro_channel.fadeout(INTRO_FADE_MS)
            self._play_loop(LOOP_FADE_IN_MS)
        else:
            self._play_loop(LOOP_FADE_IN_MS if self.position else 0)

    def _stop_files(self):
        if pygame.mixer.music.get_busy():
            # get_pos: milliseconds since play(), loops included
            self.position = (self.started_at + pygame.mixer.music.get_pos() / 1000.0) % self.length
            pygame.mixer.music.fadeout(FADE_MS)
        if self._intro_on():
            self.intro_channel.fadeout(FADE_MS)

    def __call__(self, commands, forward=True):
        """The driver's sound sink, called every frame: commands = [(fraction of frame, value), ...]."""
        out = []
        for frac, value in commands:
            command = value & 0x7F
            strobe = value & 0x80 == 0
            if strobe and command in (self.intro_cmd, self.game_over):
                self.position = 0.0                  # a new game starts the music from the beginning
            if command == self.loop_cmd:
                if self.enabled:
                    if strobe and not self.playing:
                        self._start_loop()
                    self.playing = True
                    continue                     # (both the command and its strobe)
                self.playing = True
            elif command == self.intro_cmd and self.enabled and self.intro is not None:
                if strobe:
                    self.intro_channel = self.intro.play()
                continue
            elif command in self.stops and strobe:
                if self.enabled:
                    self._stop_files()
                    if command == self.game_over:
                        self.position = 0.0
                self.playing = False
            out.append((frac, value))
        if forward:
            self.sink(out)

    def set_enabled(self, on: bool):
        on = on and self.loaded
        if on == self.enabled:
            return
        self.enabled = on
        if not on:
            self._stop_files()
        if not self.playing:
            return
        # hand the loop music over mid-stage
        if on:
            self.sink([(0.0, 0x00), (0.5, 0x80)])                  # board: everything off
            self._play_loop(LOOP_FADE_IN_MS)
        else:
            self.sink([(0.0, self.loop_cmd), (0.5, self.loop_cmd | 0x80)])   # board: its own music again
