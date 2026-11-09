"""
Frontend: window, video timing, keyboard and the sound process for any Driver
(the role of MAME's OSD layer + running_machine::run).

Keys: Esc = quit, F11 / Alt+Enter = fullscreen, F2 = CRT scanlines, F3 = upscaling filter,
F4 = HD graphics (when the driver has an HD pack in hdpack/<game>/), F5 = remake artwork (and
its music, hdpack/<game>/remake_music.mp3, see emu.music).
With a replay (recorded inputs, see tools/bot), the game plays by itself from power on;
F6 = fast forward (8x, no sound), and the player gets the controls when it ends.

Video goes through OpenGL shaders (emu.shaders) when ModernGL is available, so filters
like xBR cost no CPU time; otherwise it falls back to plain pygame scaling.
"""

import time
import numpy as np
import pygame

from emu.driver import Driver
from emu.music import MusicReplacement
from emu.sound_process import SoundProcess
from emu.shaders import ShaderScaler, FILTERS, FILTER_NAMES


class Frontend:
    def __init__(self, driver: Driver, original_speed: bool = False, video_filter: str = "nearest",
                 hdpack_root: str = "hdpack", hd: bool = True, replay=None):
        """
        original_speed False: lock to the monitor's refresh when it is within 7% of the game's
        (smooth scrolling, like MAME -syncrefresh). True: always run at the game's exact rate.
        """
        self.driver = driver
        self.original_speed = original_speed
        # replay: {"ports": [driver attribute names], "inputs": [[count, value, ...], ...],
        #          "settings": {driver attribute: value} while it plays (e.g. DIP switches)}
        self.replay = None
        self.replay_restore = {}
        if replay is not None:
            self.replay_ports = replay["ports"]
            self.replay = [tuple(v) for count, *v in replay["inputs"] for _ in range(count)]
            for name, value in replay.get("settings", {}).items():
                self.replay_restore[name] = getattr(driver, name)
                setattr(driver, name, value)
        self.frame = 0
        self.fast = False
        self.running = True

        pygame.init()
        w, h = driver.screen_width, driver.screen_height
        # Largest integer scale that fits the desktop height (minus title bar and taskbar)
        desktop_h = pygame.display.get_desktop_sizes()[0][1]
        self.scale = max(1, (desktop_h - 72) // h)
        self.window_size = (w * self.scale, h * self.scale)
        self.fullscreen = False
        self._window = None          # pygame._sdl2 Window, for borderless fullscreen

        # Optional CRT scanlines (F2); off by default for a sharp, full-brightness picture
        self.scanlines = False
        self.video_filter = video_filter if video_filter in FILTERS else "nearest"

        self.gl = None               # ShaderScaler when the OpenGL path works
        self._open_window()
        if self.gl is None:
            # Fallback: SCALED = GPU scaling by SDL (square pixels only); vsync: no tearing
            try:
                self.display = pygame.display.set_mode(self.window_size, pygame.SCALED, vsync=1)
            except pygame.error:
                self.display = pygame.display.set_mode(self.window_size)
            self.screen = pygame.Surface((w, h))   # native-resolution frame
            self.scanline_surface = self._create_scanline_surface()
        pygame.display.set_caption(driver.description or driver.name)

        # HD pack renderer (GPU composes the screen from high resolution graphics)
        self.hd = None
        if self.gl is not None:
            try:
                self.hd = driver.create_hd_renderer(self.gl.ctx, hdpack_root)
            except Exception as e:
                print(f"HD pack not loaded ({e})")
        self.hd_enabled = hd and self.hd is not None
        self.hdpack_root = hdpack_root
        self.music = None            # MusicReplacement, set up with the sound process

        self.clock = pygame.time.Clock()

    def _open_window(self):
        """Create the OpenGL window; leaves self.gl None if OpenGL is unavailable."""
        try:
            import moderngl
            self.display = pygame.display.set_mode(self.window_size, pygame.OPENGL | pygame.DOUBLEBUF, vsync=1)
            ctx = moderngl.create_context()
            self.gl = ShaderScaler(ctx, self.driver.screen_width, self.driver.screen_height)
        except Exception as e:                      # no ModernGL / OpenGL 3.3
            print(f"OpenGL video unavailable ({e}); using basic scaling.")
            self.gl = None

    def _toggle_fullscreen(self):
        if self.gl is None:
            pygame.display.toggle_fullscreen()
            return
        # Borderless window covering the desktop: instant, no video mode change (an exclusive
        # fullscreen switch stalls the emulation for about a second)
        if self._window is None:
            # Keep one reference for the program's lifetime: a short-lived wrapper can take
            # SDL's window down with it when it is garbage collected
            from pygame._sdl2.video import Window
            self._window = Window.from_display_module()
        window = self._window
        self.fullscreen = not self.fullscreen
        if self.fullscreen:
            self._windowed_pos = window.position
            window.borderless = True
            window.position = (0, 0)
            window.size = pygame.display.get_desktop_sizes()[0]
        else:
            window.borderless = False
            window.size = self.window_size
            window.position = self._windowed_pos

    def _viewport(self):
        """Largest area with the game's aspect ratio, centered (integer scale for square pixels)."""
        win_w, win_h = pygame.display.get_window_size()
        w, h = self.driver.screen_width, self.driver.screen_height
        s = min(win_w / w, win_h / h)
        if self.video_filter == "nearest" and not self.hd_enabled and s >= 1:
            s = int(s)                              # uneven pixel sizes look bad without a filter
        vw, vh = int(w * s), int(h * s)
        return ((win_w - vw) // 2, (win_h - vh) // 2, vw, vh)

    def _create_scanline_surface(self):
        """CRT look: a multiply mask with dark gaps between scanlines and a faint aperture grille."""
        w, h = self.display.get_size()
        rows = np.array([255, 255, 230, 140], dtype=np.float32)[np.arange(h) % self.scale * 4 // self.scale]
        cols = np.where(np.arange(w) % self.scale == self.scale - 1, 0.9, 1.0).astype(np.float32)
        mask = (cols[:, None] * rows[None, :]).astype(np.uint8)
        surface = pygame.Surface((w, h))
        pygame.surfarray.blit_array(surface, np.repeat(mask[:, :, None], 3, axis=2))
        return surface

    # --- Video ---

    def update_display(self):
        if self.gl is not None:
            self.gl.ctx.viewport = (0, 0) + pygame.display.get_window_size()
            self.gl.ctx.clear(0.0, 0.0, 0.0)
            if self.hd_enabled:
                # 2x2 supersampling: anti-aliased edges
                self.hd.draw(self._viewport(), self.scanlines, samples=2)
                pygame.display.flip()
                return
            self.gl.draw(self.driver.render(), self.video_filter, self._viewport(), self.scanlines)
            pygame.display.flip()
            return
        pygame.surfarray.blit_array(self.screen, self.driver.render().transpose(1, 0, 2))
        pygame.transform.scale(self.screen, self.display.get_size(), self.display)
        if self.scanlines:
            self.display.blit(self.scanline_surface, (0, 0), special_flags=pygame.BLEND_MULT)
        pygame.display.flip()

    def _measure_refresh(self):
        """Monitor refresh rate in Hz if flip() is vsync-blocked, else None."""
        if self.gl is not None:
            self.gl.ctx.clear(0.0, 0.0, 0.0)
        else:
            self.display.fill((0, 0, 0))
        for _ in range(5):
            pygame.display.flip()
        stamps = []
        for _ in range(61):
            pygame.display.flip()
            stamps.append(time.perf_counter())
        intervals = sorted(b - a for a, b in zip(stamps, stamps[1:]))
        if intervals[len(intervals) // 2] < 0.004:  # flip returned immediately: no vsync
            return None
        return (len(stamps) - 1) / (stamps[-1] - stamps[0])

    # --- Input ---

    def handle_events(self):
        self.driver.apply_pending_releases()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type in (pygame.KEYDOWN, pygame.KEYUP):
                pressed = event.type == pygame.KEYDOWN
                if pressed and event.key == pygame.K_ESCAPE:
                    self.running = False
                elif pressed and (event.key == pygame.K_F11 or
                                  (event.key == pygame.K_RETURN and event.mod & pygame.KMOD_ALT)):
                    self._toggle_fullscreen()
                elif pressed and event.key == pygame.K_F2:
                    self.scanlines = not self.scanlines
                elif pressed and event.key == pygame.K_F4:
                    if self.hd is not None:
                        self.hd_enabled = not self.hd_enabled
                        print("HD graphics:", "on" if self.hd_enabled else "off")
                        self._update_music()
                elif pressed and event.key == pygame.K_F5:
                    if self.hd is not None and getattr(self.hd, "has_remake", False):
                        self.hd.remake_enabled = not self.hd.remake_enabled
                        print("Remake artwork:", "on" if self.hd.remake_enabled else "off")
                        self._update_music()
                elif pressed and event.key == pygame.K_F6:
                    if self.replaying():
                        self.fast = not self.fast
                        print("Fast forward:", "on" if self.fast else "off")
                elif pressed and event.key == pygame.K_F3:
                    if self.gl is not None and not self.hd_enabled:
                        self.video_filter = FILTERS[(FILTERS.index(self.video_filter) + 1) % len(FILTERS)]
                        print(f"Filter: {FILTER_NAMES[self.video_filter]}")
                elif not self.replaying():
                    self.driver.key_event(event.key, pressed)

    def _update_music(self):
        """The remake's music plays while the remake is on screen."""
        if self.music is not None:
            self.music.set_enabled(self.hd_enabled and getattr(self.hd, "remake_enabled", False))

    def replaying(self):
        return self.replay is not None and self.frame < len(self.replay)

    def _run_frame(self):
        """One emulated frame, with the replay's inputs while it lasts."""
        if self.replaying():
            for port, value in zip(self.replay_ports, self.replay[self.frame]):
                setattr(self.driver, port, value)
            if self.frame + 1 == len(self.replay):
                for port in self.replay_ports:
                    setattr(self.driver, port, 0xFF)       # released: the player takes over
                for name, value in self.replay_restore.items():
                    setattr(self.driver, name, value)
                print("Replay finished: you have the controls")
                self.fast = False
        elif self.hd_enabled and hasattr(self.hd, "hill_physics"):
            self.hd.hill_physics()             # (changes the game: never during a replay)
        self.frame += 1
        self.driver.run_frame()

    # --- Main loop ---

    def run(self):
        driver = self.driver
        print("\n" + "=" * 60)
        print(f"{driver.description} ({driver.manufacturer} {driver.year})")
        if driver.controls:
            print(driver.controls)
        print("Esc=Quit, F11=Fullscreen, F2=Scanlines, F3=Filter, F4=HD graphics, F5=Remake art")
        if self.hd is not None:
            print("HD graphics:", "on" if self.hd_enabled else "off")
        if self.gl is not None:
            print(f"Filter: {FILTER_NAMES[self.video_filter]}")
        print("=" * 60)

        # Video timing: with vsync working and a monitor within 7% of the game's rate,
        # emulate one frame per monitor refresh -> perfectly smooth scrolling (the game runs
        # slightly faster or slower; the sound board keeps its own clock, so pitch is unchanged).
        monitor_hz = self._measure_refresh()
        sync_to_monitor = (not self.original_speed and monitor_hz is not None
                           and abs(monitor_hz - driver.refresh_rate) / driver.refresh_rate < 0.07)
        video_rate = monitor_hz if sync_to_monitor else driver.refresh_rate
        if sync_to_monitor:
            print(f"Video: synced to the monitor ({monitor_hz:.2f} Hz)")
        else:
            print(f"Video: arcade timing {driver.refresh_rate} Hz"
                  + (f" (monitor {monitor_hz:.1f} Hz, vsync)" if monitor_hz else ""))

        # The sound board runs on its own CPU core and talks to the sound card directly
        sound_process = None
        board = driver.sound_board()
        if board is not None:
            sound_process = SoundProcess(board, video_rate)
            driver.sound_sink = sound_process.send_frame
            music_files = MusicReplacement.find(driver, self.hdpack_root) if driver.music_commands else None
            if music_files and self.hd is not None:
                self.music = MusicReplacement(*music_files, driver.music_commands, sound_process.send_frame)
                driver.sound_sink = self.music
                self._update_music()

        frame_period = 1.0 / driver.refresh_rate
        next_frame = time.perf_counter()
        try:
            while self.running:
                self.handle_events()
                if self.fast:
                    # fast forward: 7 frames without sound or video, then a normal one
                    sink = driver.sound_sink
                    # (the music still follows the commands: it must know if it is on)
                    driver.sound_sink = ((lambda commands: self.music(commands, forward=False))
                                         if self.music else (lambda commands: None))
                    for _ in range(7):
                        if self.fast:
                            self._run_frame()
                    driver.sound_sink = sink
                self._run_frame()
                self.update_display()

                self.clock.tick()
                if sync_to_monitor:
                    continue                        # flip() already waited for the vblank
                # Precise frame pacing at the game's rate (clock.tick rounds to whole ms)
                next_frame += frame_period
                delay = next_frame - time.perf_counter()
                if delay > 0.002:
                    time.sleep(delay - 0.0015)
                while time.perf_counter() < next_frame:
                    pass
                if delay < -0.1:                    # fell far behind (window drag...): resync
                    next_frame = time.perf_counter()
        except KeyboardInterrupt:
            print("\nEmulation stopped by user.")
        finally:
            if sound_process is not None:
                sound_process.close()
                driver.sound_sink = None
            driver.shutdown()
            pygame.quit()
