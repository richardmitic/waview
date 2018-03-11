import curses
import curses.panel
import sys
import time
import queue
import logging
import asyncio
import traceback
import os
import numpy as np
import scipy.signal
from .core import WaviewCore, LoggerWriter, WavType, INT16_MAX

LOG = logging.getLogger(__name__)

controls = """\
'↑' - Zoom in
'↓' - Zoom out
'→' - Shift right
'←' - Shift left
'o' - Increase Y scale
'k' - Decrease Y scale
'h' - Display this dialogue
'r' - Reset display
'q' - Quit"""

def text_dims(text):
    h = text.count(os.linesep)
    w = max(len(line) for line in text.split(os.linesep))
    return h, w


class PopupWindow():
    def __init__(self, h, w, y, x):
        self.h, self.w, self.y, self.x = h, w, y, x
        self.outer_win = curses.newwin(h, w, y, x)
        self.inner_win = self.outer_win.derwin(h-2, w-2, 1, 1)
        self.outer_win.erase()
        self.outer_win.box()
        self.panel = curses.panel.new_panel(self.outer_win)
        self.set_text("", show=False)

    def draw(self):
        LOG.debug(f"Drawing popup h:{self.h} w:{self.w} y:{self.y} x:{self.x}")
        self.inner_win.erase()
        self.inner_win.addstr(self.text)

    def resize(self, h, w, y, x):
        LOG.debug(f"resizing popup {h} {w} {y} {x}")
        self.__init__(h,w,y,x)

    def set_text(self, text, show=True):
        LOG.debug(f"Setting popup text to {text}")
        self.text = text
        if show:
            self.show()

    def show(self):
        if self.panel.hidden():
            self.panel.show()

    def hide(self):
        if not self.panel.hidden():
            self.panel.hide()

    def toggle_visible(self):
        if self.panel.hidden():
            self.show()
        else:
            self.hide()


class ChannelDisplay():
    def __init__(self, win, index):
        self.win = win
        self.index = index

    @staticmethod
    def reshape_peaks_to_window_size(peaks, drawable_width):
        if peaks.shape[0] == drawable_width:
            return peaks
        else:
            return scipy.signal.resample(peaks, drawable_width)

    def __draw_peaks(self, peaks, y_scale):
        "Draw peaks inside the box"
        h,w = self.win.getmaxyx()
        drawable_peaks = self.reshape_peaks_to_window_size(peaks, w-2)
        free_h = h - 2
        for i, peak in enumerate(drawable_peaks):
            x = i+1
            norm_height = min(int(peak * free_h * y_scale), free_h)
            start_pos = ((free_h - norm_height) // 2) + 1
            if norm_height > 0:
                self.win.vline(start_pos, x, curses.ACS_CKBOARD, norm_height)
            else:
                y = start_pos if h % 2 == 1 else start_pos - 1
                char = curses.ACS_HLINE if h % 2 == 1 else curses.ACS_S9
                self.win.addch(y, x, char)

    def __draw_samples(self, samples, y_scale):
        h,w = self.win.getmaxyx()
        free_h = h - 2
        ys = ((np.copy(samples) * free_h * y_scale) + free_h) / 2
        clipped = ((ys < 1) * 1) + ((ys > h-1) * -1)
        ys = np.clip(ys, 1, h-2).astype(np.int)
        xs = np.arange(1, w-1, dtype=np.int)

        chars = {-1:'▁', 0:'•', 1:'▔'}
        for x, y, c in zip(xs, ys, clipped):
            self.win.addch(y, x, chars[c])

    def __draw_text(self, start, end, y_scale):
        self.win.addstr(0, 0, f"Channel {self.index} range:{start:.3}-{end:.3} scale:{y_scale:.3} ")

    def draw(self, point_type, points, start=0., end=1., y_scale=0.1):
        self.win.border()
        if point_type == WavType.PEAKS:
            self.__draw_peaks(points, y_scale)
        elif point_type == WavType.SAMPLES:
            self.__draw_samples(points, y_scale)
        self.__draw_text(start, end, y_scale)


def with_error_popup(f):
    async def wrapped_f(*args, **kwargs):
        app = args[0]
        try:
            await f(*args, **kwargs)
            if app.popup:
                app.popup.hide()
        except Exception as e:
            app = args[0]
            if app.popup:
                app.popup.set_text(f"{type(e)}: {e}")
                app.redraw()
    return wrapped_f

class WaviewApp():
    def __init__(self, wav=None, yscale=1., centroid=0.5, range=1.,  channels=1, deltashift=0.2, deltazoom=0.2):
        self.core = WaviewCore()
        self.running = True
        self.update = False
        self.height = 0
        self.width = 0
        self.points = None
        self.popup = None
        self.wavfilepath = wav
        self.channels = channels

        # Drawing parameters
        self.y_scale = yscale
        self.centroid = centroid
        self.range = range
        self.delta_shift = deltashift
        self.delta_zoom = deltazoom
        self.delta_zoom_y = deltazoom

    @property
    def start(self):
        return self.centroid - (self.range * 0.5)

    @property
    def end(self):
        return self.centroid + (self.range * 0.5)

    def quit(self):
        for task in asyncio.Task.all_tasks():
            task.cancel()
        self.running = False

    def redraw(self):
        self.draw()

    def set_text(self, txt):
        self.text = txt
        self.update = True

    def channel_draw_width(self):
        return self.width - 2

    @with_error_popup
    async def refresh_points(self):
        if self.wavfilepath:
            self.point_type, self.points = await self.core.get_wav(
                self.wavfilepath, start=self.start, end=self.end,
                num_points=self.channel_draw_width(), channels=self.channels)
        else:
            return None, None

    async def shift_left(self):
        self.centroid -= (self.delta_shift * self.range)
        await self.refresh_points()
        self.redraw()

    async def shift_right(self):
        self.centroid += (self.delta_shift * self.range)
        await self.refresh_points()
        self.redraw()

    async def zoom_in(self):
        self.range *= (1 - self.delta_zoom)
        await self.refresh_points()
        self.redraw()

    async def zoom_out(self):
        self.range *= (1 + self.delta_zoom)
        await self.refresh_points()
        self.redraw()

    async def zoom_in_y(self):
        self.y_scale += self.delta_zoom_y
        await self.refresh_points()
        self.redraw()

    async def zoom_out_y(self):
        self.y_scale -= self.delta_zoom_y
        await self.refresh_points()
        self.redraw()

    async def reset(self):
        self.range = 1.0
        self.centroid = 0.5
        await self.refresh_points()
        self.redraw()

    async def analyze(self, path):
        self.wavfilepath = path
        self.popup.set_text(f"Analyzing {os.path.basename(path)}")
        self.redraw()
        await self.refresh_points()
        self.popup.hide()
        self.redraw()

    def toggle_controls(self):
        h,w = self.screen.getmaxyx()
        ht,wt = text_dims(controls)
        hp,wp = ht+3, wt+3
        y = (h - hp) // 2
        x = (w - wp) // 2
        self.popup.resize(hp, wp, y, x)
        self.popup.set_text(controls)
        self.toggle_popup()

    @with_error_popup
    async def trigger_exception(self):
        raise Exception("Exception raised for debugging")

    def handle_key_press(self, key):
        "Handle key events. Do not call draw() directly from here."
        if key == ord('q'):
            self.quit()
        elif key == ord('h'):
            self.toggle_controls()
        elif key == ord('p'):
            self.toggle_popup()
        elif key == ord('t'):
            asyncio.ensure_future(self.trigger_exception())
        elif key == ord('r'):
            asyncio.ensure_future(self.reset())
        elif key == ord('o'):
            asyncio.ensure_future(self.zoom_in_y())
        elif key == ord('k'):
            asyncio.ensure_future(self.zoom_out_y())
        elif key == curses.KEY_LEFT:
            asyncio.ensure_future(self.shift_left())
        elif key == curses.KEY_RIGHT:
            asyncio.ensure_future(self.shift_right())
        elif key == curses.KEY_UP:
            asyncio.ensure_future(self.zoom_in())
        elif key == curses.KEY_DOWN:
            asyncio.ensure_future(self.zoom_out())

    def toggle_popup(self):
        self.popup.toggle_visible()
        self.redraw()

    def make_channels(self, screen, n):
        drawable_height, drawable_width = self.screen.getmaxyx()
        channels = []
        for i in range(n):
            y = int((i * drawable_height) / n)
            x = 0
            h = int(((i+1) * drawable_height) / n) - y
            w = drawable_width
            LOG.info(f"creating channel y:{y} x:{x} h:{h} w:{w} {drawable_height} {drawable_width}")
            win = screen.derwin(h, w, y, x)
            channels.append(ChannelDisplay(win, i))
        return channels

    def draw_peaks(self):
        num_channels = self.points.shape[0]
        channels = self.make_channels(self.screen, num_channels)
        for peaks, channel in zip(self.points, channels):
            channel.draw(WavType.PEAKS, peaks, start=self.start, end=self.end, y_scale=self.y_scale)

    def draw_samples(self):
        num_channels = self.points.shape[0]
        channels = self.make_channels(self.screen, num_channels)
        for samples, channel in zip(self.points, channels):
            channel.draw(WavType.SAMPLES, samples, start=self.start, end=self.end, y_scale=self.y_scale)

    def draw(self):
        LOG.debug("draw")
        self.height, self.width = self.screen.getmaxyx()
        self.screen.clear()
        # self.screen.border()
        if self.points is not None:
            if self.point_type == WavType.PEAKS:
                self.draw_peaks()
            elif self.point_type == WavType.SAMPLES:
                self.draw_samples()
        self.popup.draw()
        curses.panel.update_panels()
        self.screen.refresh()

    def make_popup(self):
        self.height, self.width = self.screen.getmaxyx()
        popup_height = 10
        popup_width = 40
        x = (self.width - popup_width) // 2
        y = (self.height - popup_height) // 2
        LOG.debug(f"{popup_height, popup_width, y, x}")
        self.popup = PopupWindow(popup_height, popup_width, y, x)

    async def wait_for_key(self):
        loop = asyncio.get_event_loop()
        LOG.debug("waiting for key")
        key = await loop.run_in_executor(None, self.screen.getch)
        LOG.debug(f"got key {key}")
        return key

    async def async_main(self, screen):
        self.screen = screen
        self.height, self.width = screen.getmaxyx()
        self.set_text(f"{self.width}x{self.height}")

        self.make_popup()
        self.popup.panel.top()
        self.redraw()

        while self.running:
            key = await self.wait_for_key()
            self.handle_key_press(key)

    async def async_wrapper(self):
        """Wrapper function that initializes curses and calls another function,
        restoring normal keyboard/screen behavior on error.
        The callable object 'func' is then passed the main window 'stdscr'
        as its first argument, followed by any other arguments passed to
        wrapper().

        Async version of the normal curses.wrapper(), which is blocking.
        see https://github.com/python/cpython/blob/3.6/Lib/curses/__init__.py
        """

        result = 0
        try:
            # Initialize curses
            stdscr = curses.initscr()

            # Turn off echoing of keys, and enter cbreak mode,
            # where no buffering is performed on keyboard input
            curses.noecho()
            curses.cbreak()

            # In keypad mode, escape sequences for special keys
            # (like the cursor keys) will be interpreted and
            # a special value like curses.KEY_LEFT will be returned
            stdscr.keypad(1)

            # Start color, too.  Harmless if the terminal doesn't have
            # color; user can test with has_color() later on.  The try/catch
            # works around a minor bit of over-conscientiousness in the curses
            # module -- the error return from C start_color() is ignorable.
            try:
                curses.start_color()
            except:
                pass

            try:
                result = await self.async_main(stdscr)
            except Exception as e:
                traceback.print_exc()
                LOG.error(e)

        finally:
            # Set everything back to normal
            if 'stdscr' in locals():
                stdscr.keypad(0)
                curses.echo()
                curses.nocbreak()
                curses.endwin()

        return result
