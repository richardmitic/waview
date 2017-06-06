#!/usr/bin/env python3

import curses
import sys
import time
import argparse
import itertools
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import resample

INT16_MAX = int(2**15)-1

with open("log.txt", "w") as f:
    print("log starting", file=f)

def log(*args):
    with open("log.txt", "a") as f:
        print(*args, file=f)

def clip(n, _min, _max):
    return min(max(n,_min), _max)

def absmax(arr):
    return max(np.abs(arr)) if len(arr) > 0 else 0 # be careful for zero-length arrays



class Wave():
    def __init__(self):
        self.samples = np.empty((1,0)) # 1 channel, 0 samples
        self.sr = None
        self.nchannels = 1

    def nsamples():
        doc = "The nsamples property."
        def fget(self):
            return len(self.samples[0])
        return locals()
    nsamples = property(**nsamples())

    @staticmethod
    def frames_to_channels(samples):
        "Convert frame-based samples to channel-based"
        return np.transpose(samples)

    def load_file(self, filename):
        "Load a WAV file"
        self.sr, self.samples = wavfile.read(filename)
        if isinstance(self.samples[0], np.ndarray):
            self.nchannels = len(self.samples[0])
            self.samples = self.frames_to_channels(self.samples)
        else:
            self.nchannels = 1
            self.samples = np.array([self.samples]) # Se we can still index the only channel

    def get_samples(self, offset=0, num_samples=None, channel=0, num_chunks=1):
        samps = self.samples[channel]
        num_samples = len(samps)-offset if num_samples is None else num_samples

        # Pad samples with zeros if we go outside the range of the wave
        pre_padding = np.zeros(-offset if offset < 0 else 0)
        stop = offset + num_samples
        post_padding = np.zeros(stop-len(samps) if stop >= len(samps) else 0)

        start = clip(offset, 0, len(samps))
        stop = clip(stop, 0, len(samps))
        samples_to_display = np.concatenate((pre_padding, samps[start:stop], post_padding))
        chunks = np.array_split(samples_to_display, num_chunks)
        return chunks

    def get_peaks(self, offset, num_samples, channel, num_peaks):
        log("get_peaks {} {} {} {}".format(offset, num_samples, channel, num_peaks))
        return list(map(absmax, self.get_samples(offset, num_samples, channel, num_chunks=num_peaks)))


class ChannelDisplay():
    def __init__(self, parent, window_h, window_w, begin_y, begin_x):
        self.width = window_w
        self.height = window_h
        self.begin_y = begin_y
        self.begin_x = begin_x
        self.screen = parent.subwin(window_h, window_w, begin_y, begin_x)
        self.border_size = 1
        # The free space we have available for drawing, i.e. inside the borders
        self.draw_width = self.width - (2 * self.border_size)
        self.draw_height = self.height - (2 * self.border_size)
        self.peaks_threshold = 50 # choose whether to display peaks or waveform

    def set_wave(self, wave, channel):
        "Wave object to draw. A window can only draw 1 channel."
        self.wave = wave
        self.channel = channel

    def scale_sample(self, samp):
        half_height = self.draw_height / 2
        return int(((samp/INT16_MAX) * half_height) + half_height)

    # @staticmethod
    # def gradient_to_symbol(gradient):
    #     if gradient == 0:
    #         return curses.ACS_S1
    #     elif gradient == 1:
    #         return '\\'
    #     elif gradient == -1:
    #         return '/'
    #     elif (gradient >= 2) or (gradient <= -2):
    #         return curses.ACS_VLINE
    #     else:
    #         raise Exception("This should never happen")

    def draw_samples(self, offset, nsamples):
        # Make sure we don't try to draw outside the drawing area
        samples = self.wave.get_samples(offset, nsamples, self.channel)[0] # get_samples returns a list of chunks
        samples = resample(samples, self.draw_width+1) # we don't actually draw the last point
        points = [self.scale_sample(s) + self.border_size for s in samples]
        for i in range(len(points)-1):
            x = i + self.border_size
            y = points[i]
            # gradient = points[i+1] - points[i]
            # symbol = self.gradient_to_symbol(gradient)
            symbol = curses.ACS_DEGREE
            try:
                self.screen.addch(y, x, symbol)
            except curses.error as e:
                log("addch error: {} {} {} {} {}".format(y,x,symbol,self.draw_width,self.draw_height))

    def scale_peak(self, peak):
        half_height = self.draw_height / 2
        length = int((peak/INT16_MAX) * half_height)
        top = int(half_height-length)
        return top, length

    def draw_peaks(self, offset, nsamples):
        # Make sure we don't try to draw outside the drawing area
        peaks = self.wave.get_peaks(offset, nsamples, self.channel, self.draw_width)
        log("draw_peaks {} {} {}".format(offset, nsamples, self.channel))
        for x, peak in enumerate(peaks, self.border_size):
            top, length = self.scale_peak(peak)
            top += self.border_size
            reflected_length = 2 * length
            # log("peaks {top} {x} {reflected_length}".format(**locals()))
            self.screen.vline(top, x, curses.ACS_CKBOARD, reflected_length)

    def draw(self, start, end):
        """ Draw the given section of the wave
            start: Starting point as proportion of total length, i.e. from 0. to 1.
            end: Ending point as proportion of total length, i.e. from 0. to 1.
        """
        self.screen.box()
        offset = int(self.wave.nsamples * start)
        nsamples = int(self.wave.nsamples * (end-start))
        samples_per_column = nsamples / self.draw_width
        self.screen.addstr("samples[{0}:{1}] {2:.4} {3:.4}:{4:.4}:{5:.4}".format(offset, nsamples, samples_per_column, start, end, end-start))
        if samples_per_column < self.peaks_threshold:
            self.draw_samples(offset, nsamples)
        else:
            self.draw_peaks(offset, nsamples)


class App():
    def __init__(self, zoom=1.):
        self.wave = Wave()
        self.wave_centroid = 0.5 # Point of the wave at the centre of the screen
        self.wave_centroid_delta = 0.2 # Proportion of the displayed area to move
        self.zoom = zoom # 1. means entire wave
        self.zoom_delta_multipler = 0.2 # change in zoom value for each key press
        self.running = True

    def quit(self):
        self.running = False

    def shift_left(self):
        current_range = 1. / self.zoom
        self.wave_centroid += (current_range*self.wave_centroid_delta)
        log("shift left", self.wave_centroid)

    def shift_right(self):
        current_range = 1. / self.zoom
        self.wave_centroid -= (current_range*self.wave_centroid_delta)
        log("shift right", self.wave_centroid)

    def zoom_in(self):
        coeff = 1+self.zoom_delta_multipler
        self.zoom = clip(self.zoom * coeff, 1., float('inf'))
        log("zoom in", self.zoom)

    def zoom_out(self):
        coeff = 1-self.zoom_delta_multipler
        self.zoom = clip(self.zoom * coeff , 1., float('inf'))
        log("zoom out", self.zoom)

    def get_window_rect(self, screen, channel):
        "Calculate x, y, width, height for a given channel window"
        total_h, total_w = screen.getmaxyx()
        begin_x, begin_y = 0, int( (total_h / self.wave.nchannels) *  channel    )
        end_x,   end_y =   0, int( (total_h / self.wave.nchannels) * (channel+1) )
        window_w, window_h = total_w, end_y - begin_y
        return begin_x, begin_y, window_w, window_h

    def get_channel_windows(self, screen):
        for n in range(self.wave.nchannels):
            begin_x, begin_y, window_w, window_h = self.get_window_rect(screen, n)
            yield ChannelDisplay(screen, window_h, window_w, begin_y, begin_x)

    def handle_key_press(self, key):
        if key == "q":
            self.quit()
        elif key == "KEY_LEFT":
            self.shift_left()
        elif key == "KEY_RIGHT":
            self.shift_right()
        elif key == "KEY_UP":
            self.zoom_in()
        elif key == "KEY_DOWN":
            self.zoom_out()

    def draw(self, screen):
        screen.clear()
        screen.border()
        _, max_width = screen.getmaxyx()
        for channel, window in enumerate(self.get_channel_windows(screen)):
            window.set_wave(self.wave, channel)
            wave_start = self.wave_centroid - (1./(self.zoom*2))
            wave_end   = self.wave_centroid + (1./(self.zoom*2))
            # log(wave_start, wave_end)
            window.draw(wave_start, wave_end)
        screen.refresh()

    def main(self, stdscr):
        self.draw(stdscr)
        while self.running:
            self.handle_key_press(stdscr.getkey())
            self.draw(stdscr)


def get_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("-w", "--wavfile", help="WAV file to display")
    p.add_argument("-z", "--zoom", help="Initial zoom value", default=1, type=float)
    return p

if __name__ == '__main__':
    argparser = get_argparser()
    args = argparser.parse_args()

    app = App(zoom=args.zoom)
    app.wave.load_file(args.wavfile)

    curses.wrapper(app.main)