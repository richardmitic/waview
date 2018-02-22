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
from core import WaviewCore, LoggerWriter, INT16_MAX

LOG = logging.getLogger(__name__)

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
        LOG.debug("draw popup")
        self.inner_win.erase()
        self.inner_win.addstr(self.text)

    def set_text(self, text, show=True):
        LOG.debug(f"Setting popup text to {text}")
        self.text = text
        if show:
            self.show()

    def show(self):
        self.panel.show()

    def hide(self):
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

    def __draw_peaks(self, peaks, y_scale=0.1):
        "Draw peaks inside the box"
        h,w = self.win.getmaxyx()
        drawable_peaks = self.reshape_peaks_to_window_size(peaks, w-2)
        free_h = h - 2
        for i, peak in enumerate(drawable_peaks):
            # LOG.debug(f"{i} {peak}")
            x = i+1
            norm_height = min(int(peak * free_h * y_scale), free_h)
            start_pos = ((free_h - norm_height) // 2) + 1
            if norm_height > 0:
                self.win.vline(start_pos, x, curses.ACS_CKBOARD, norm_height)
            else:
                y = start_pos if h % 2 == 1 else start_pos - 1
                char = curses.ACS_HLINE if h % 2 == 1 else curses.ACS_S9
                self.win.addch(y, x, char)

    def __draw_text(self, start, end, y_scale):
        self.win.addstr(0, 0, f"Channel {self.index} range:{start:.3}-{end:.3} scale:{y_scale:.3} ")

    def draw(self, peaks, start=0., end=1., y_scale=0.1):
        self.win.border()
        self.__draw_peaks(peaks, y_scale=y_scale)
        self.__draw_text(start, end, y_scale)


class WaviewApp():
    def __init__(self):
        self.core = WaviewCore()
        self.running = True
        self.update = False
        self.height = 0
        self.width = 0
        self.msg_counter = 0
        self.popup_window = None
        self.peaks = None

        # Drawing parameters
        self.y_scale = 1.
        self.start = 0.0
        self.end = 1.0
        self.delta_shift = 0.1

    def quit(self):
        for task in asyncio.Task.all_tasks():
            task.cancel()
        self.running = False

    def redraw(self):
        self.update = True

    def set_text(self, txt):
        self.text = txt
        self.update = True

    def channel_draw_width(self):
        return self.width - 2

    async def shift_left(self):
        self.start += self.delta_shift
        self.end += self.delta_shift
        self.peaks = await self.core.get_peaks(self.wavfilepath,
                                               start=self.start,
                                               end=self.end,
                                               num_peaks=self.channel_draw_width())
        self.redraw()

    async def shift_right(self):
        self.start -= self.delta_shift
        self.end -= self.delta_shift
        self.peaks = await self.core.get_peaks(self.wavfilepath,
                                               start=self.start,
                                               end=self.end,
                                               num_peaks=self.channel_draw_width())
        self.redraw()

    async def analyze(self, path):
        self.wavfilepath = path
        self.popup.set_text(f"Analyzing {os.path.basename(path)}")
        self.redraw()
        self.peaks = await self.core.get_peaks(path,
                                               start=self.start,
                                               end=self.end,
                                               num_peaks=self.channel_draw_width())
        self.popup.hide()
        self.redraw()

    def handle_key_press(self, key):
        "Handle key events. Do not call draw() directly from here."
        if key == ord('q'):
            self.quit()
        elif key == ord('a'):
            # path = "/Users/richard/Developer/waview/resources/a2002011001-e02.wav"
            # path = "/Users/richard/Developer/waview/resources/4-channels.wav"
            path = "/Users/richard/Developer/waview/resources/395192__killyourpepe__duskwolf.wav"
            asyncio.ensure_future(self.analyze(path))
        elif key == ord('p'):
            self.toggle_popup()
        elif key == curses.KEY_LEFT:
            asyncio.ensure_future(self.shift_left())
        elif key == curses.KEY_RIGHT:
            asyncio.ensure_future(self.shift_right())

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
        num_channels = self.peaks.shape[0]
        LOG.debug(f"{self.peaks.shape}")
        channels = self.make_channels(self.screen, num_channels)

        for peaks, channel in zip(self.peaks, channels):
            LOG.debug(f"{channel} {peaks.shape}")
            channel.draw(peaks, start=self.start, end=self.end, y_scale=self.y_scale)

    def draw(self):
        LOG.debug("draw")
        self.height, self.width = self.screen.getmaxyx()
        self.screen.clear()
        # self.screen.border()
        if self.peaks is not None:
            self.draw_peaks()
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

    async def async_main(self, screen):
        self.screen = screen
        self.screen.timeout(0)
        self.height, self.width = screen.getmaxyx()
        self.set_text(f"{self.width}x{self.height}")

        self.make_popup()
        self.popup.panel.top()
        self.redraw()

        while self.running:
            await asyncio.sleep(0)

            key = screen.getch()
            if (key > -1):
                self.handle_key_press(key)

            if self.update:
                self.draw()
                self.update = False

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



def main():
    logging.basicConfig(level=logging.DEBUG,
                        filename="./waview.log",
                        format='%(asctime)s.%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                        datefmt='%Y%m%d-%H%M%S')
    sys.stdout = LoggerWriter(LOG.debug)
    sys.stderr = LoggerWriter(LOG.error)

    loop = asyncio.get_event_loop()
    app = WaviewApp()

    try:
        tasks = asyncio.gather(
            app.async_wrapper()
        )
        results = loop.run_until_complete(tasks)
        LOG.info(results)
    finally:
        loop.close()

if __name__ == '__main__':
    main()
