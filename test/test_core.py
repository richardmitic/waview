from waview.core import WaviewCore, INT16_MAX, WavType
from waview.waview_async import WaviewApp
import asyncio
import os
import wave
import math
import pytest
from scipy.io import wavfile
import numpy as np

INT16_ONE = 1 / INT16_MAX

def resource(filename):
    return os.path.join(os.path.dirname(__file__), "..", "resources", filename)

def write_test_wav_file_1():
    wav = np.repeat(np.arange(0, 10, dtype=np.int16), 1024)
    wavfile.write(resource("test1.wav"), 8000, wav)

def write_test_wav_file_2():
    wav = np.repeat(np.arange(0, -10, -1, dtype=np.int16), 1024)
    wavfile.write(resource("test2.wav"), 8000, wav)

def write_test_wav_file_3():
    left = np.repeat(np.arange(0, 10, 1, dtype=np.int16), 1024)
    right = np.repeat(np.arange(0, -10, -1, dtype=np.int16), 1024)
    wav = np.array([left, right]).T
    wavfile.write(resource("test3.wav"), 8000, wav)

def setup_module(module):
    write_test_wav_file_1()
    write_test_wav_file_2()
    write_test_wav_file_3()

def step(n):
    "Symetric step function with <n> points"
    return np.heaviside(np.linspace(-1, 1, num=n), 0)

class BaseTest:
    def setup(self):
        self.loop = asyncio.get_event_loop()
        self.core = WaviewCore()

    def test1_samples(self):
        return np.expand_dims(np.repeat(np.arange(0, 10, dtype=np.int16), 1024), 0) / INT16_MAX


class TestPeaks(BaseTest):

    def test_peak_values(self):
        task = self.core.get_peaks(resource("test1.wav"))
        result = self.loop.run_until_complete(task)
        expected_peaks = np.expand_dims(np.arange(10) / INT16_MAX, 0)
        assert (np.array_equiv(result, expected_peaks))

    def test_peak_values_negative_samples(self):
        task = self.core.get_peaks(resource("test2.wav"))
        result = self.loop.run_until_complete(task)
        expected_peaks = np.expand_dims(np.arange(10) / INT16_MAX, 0)
        assert (np.array_equiv(result, expected_peaks))

    def test_peak_values_specific_num_peaks(self):
        task = self.core.get_peaks(resource("test1.wav"), num_peaks=20)
        result = self.loop.run_until_complete(task)
        expected_peaks = np.expand_dims(np.repeat(np.arange(10) / INT16_MAX, 2), 0)
        assert (np.array_equiv(result, expected_peaks))

    def test_peak_values_part_file(self):
        task = self.core.get_peaks(resource("test1.wav"), start=0.2, end=0.5)
        result = self.loop.run_until_complete(task)
        expected_peaks = np.expand_dims(np.arange(2,5) / INT16_MAX, 0)
        assert (np.array_equiv(result, expected_peaks))

    def test_peak_values_part_file_specific_num_peaks(self):
        task = self.core.get_peaks(resource("test1.wav"), start=0.2, end=0.5, num_peaks=6)
        result = self.loop.run_until_complete(task)
        expected_peaks = np.expand_dims(np.repeat(np.arange(2,5) / INT16_MAX, 2), 0)
        assert (np.array_equiv(result, expected_peaks))

    def test_peak_values_negative_bounds(self):
        task = self.core.get_peaks(resource("test1.wav"), start=-0.5, end=0.5)
        result = self.loop.run_until_complete(task)
        expected_peaks = np.expand_dims(np.clip(np.arange(-5, 5) / INT16_MAX, 0., 1.), 0)
        assert (np.array_equiv(result, expected_peaks))

    def test_peak_values_over_positive_bounds(self):
        task = self.core.get_peaks(resource("test1.wav"), start=0.5, end=1.5)
        result = self.loop.run_until_complete(task)
        expected_peaks = np.zeros((1,10))
        expected_peaks[0,0:5] = np.arange(5, 10) / INT16_MAX
        assert (np.array_equiv(result, expected_peaks))

    def test_peak_values_negative_and_over_positive_bounds(self):
        task = self.core.get_peaks(resource("test1.wav"), start=-0.5, end=1.5)
        result = self.loop.run_until_complete(task)
        expected_peaks = np.zeros((1,20))
        expected_peaks[0,5:15] = np.arange(0, 10) / INT16_MAX
        assert (np.array_equiv(result, expected_peaks))

    @pytest.mark.parametrize("i", range(1,20))
    def test_irregular_num_peaks(self, i):
        task = self.core.get_peaks(resource("test1.wav"), num_peaks=i)
        result = self.loop.run_until_complete(task)
        assert (result.shape == (1,i))

    def test_stereo(self):
        task = self.core.get_peaks(resource("test3.wav"), num_peaks=10)
        result = self.loop.run_until_complete(task)
        expected_peaks = np.tile(np.arange(10) / INT16_MAX, (2,1))
        assert (np.array_equiv(result, expected_peaks))


class TestSamples(BaseTest):

    def test_all_samples(self):
        task = self.core.get_samples(resource("test1.wav"))
        result = self.loop.run_until_complete(task)
        expected_samples = np.expand_dims(np.repeat(np.arange(0, 10, dtype=np.int16), 1024), 0) / INT16_MAX
        assert (np.array_equiv(result, expected_samples))

    def test_partial_file(self):
        start = 1020 / 10240
        end = 1028 / 10240
        task = self.core.get_samples(resource("test1.wav"), start=start, end=end)
        result = self.loop.run_until_complete(task)
        expected_samples = np.heaviside(np.linspace(-1, 1, num=8), 0) / INT16_MAX
        assert (np.array_equiv(result, expected_samples))

    def test_all_samples_resampled(self):
        task = self.core.get_samples(resource("test1.wav"), num_samps=100)
        result = self.loop.run_until_complete(task)
        assert (result.shape == (1,100))

    def test_partial_downsampled(self):
        end = 2048 / 10240
        task = self.core.get_samples(resource("test1.wav"), start=0, end=end, num_samps=100)
        result = self.loop.run_until_complete(task)
        assert (result.shape == (1,100))

    def test_partial_upsampled(self):
        start = 1020 / 10240
        end = 1028 / 10240
        task = self.core.get_samples(resource("test1.wav"), start=start, end=end, num_samps=100)
        result = self.loop.run_until_complete(task)
        assert (result.shape == (1,100))

    def test_stereo(self):
        start = 1020 / 10240
        end = 1028 / 10240
        task = self.core.get_samples(resource("test3.wav"), start=start, end=end)
        result = self.loop.run_until_complete(task)
        expected_left = np.heaviside(np.linspace(-1, 1, num=8), 0) / INT16_MAX
        expected_right = expected_left * -1
        expected = np.array([expected_left, expected_right])
        assert (np.array_equiv(result, expected))


class TestGetWav(BaseTest):

    def _assertCloseEnough(self, arr, expected, thresh=INT16_ONE):
        mse = ((arr - expected) ** 2).mean()
        assert (arr.shape == expected.shape)
        assert (mse <= thresh)

    def test_whole_file_no_specified_points(self):
        task = self.core.get_wav(resource("test1.wav"), num_points=None)
        point_type, points = self.loop.run_until_complete(task)
        assert (point_type == WavType.SAMPLES)
        assert (np.array_equiv(points, self.test1_samples()))

    def test_whole_file_with_specified_points(self):
        task = self.core.get_wav(resource("test1.wav"), num_points=10)
        point_type, points = self.loop.run_until_complete(task)
        expected_peaks = np.expand_dims(np.arange(10) / INT16_MAX, 0)
        assert (point_type == WavType.PEAKS)
        assert (np.array_equiv(points, expected_peaks))

    def test_small_section(self):
        start = 1000 / 10240
        end = 1048 / 10240
        task = self.core.get_wav(resource("test1.wav"), start=start, end=end, num_points=48)
        point_type, points = self.loop.run_until_complete(task)
        expected_points = np.expand_dims(step(48), 0) / INT16_MAX
        assert (point_type == WavType.SAMPLES)
        assert (np.array_equiv(points, expected_points))

    def test_small_section_downsampled(self):
        start = 1000 / 10240
        end = 1048 / 10240
        task = self.core.get_wav(resource("test1.wav"), start=start, end=end, num_points=20)
        point_type, points = self.loop.run_until_complete(task)
        expected_points = np.expand_dims(step(20), 0) / INT16_MAX
        assert (point_type == WavType.SAMPLES)
        self._assertCloseEnough(points, expected_points)

    def test_small_section_upsampled(self):
        start = 1000 / 10240
        end = 1048 / 10240
        task = self.core.get_wav(resource("test1.wav"), start=start, end=end, num_points=100)
        point_type, points = self.loop.run_until_complete(task)
        expected_points = np.expand_dims(step(100), 0) / INT16_MAX
        assert (point_type == WavType.SAMPLES)
        self._assertCloseEnough(points, expected_points)

    @pytest.mark.parametrize("i", range(5,200))
    def test_num_points_no_crashes(self, i):
        task = self.core.get_wav(resource("test1.wav"), num_points=i)
        point_type, points = self.loop.run_until_complete(task)

    @pytest.mark.parametrize("end", np.linspace(0.01, 1., num=100))
    def test_range_no_crashes(self, end):
        task = self.core.get_wav(resource("test1.wav"), start=0., end=end, num_points=100)
        point_type, points = self.loop.run_until_complete(task)


class TestWaview:

    def setup(self):
        self.loop = asyncio.get_event_loop()
        self.app = WaviewApp()

    def test_shift_left_full(self):
        self.loop.run_until_complete(self.app.shift_left())
        expected_start = 0 - self.app.delta_shift
        expected_end = 1 - self.app.delta_shift
        assert (math.isclose(self.app.start, expected_start))
        assert (math.isclose(self.app.end, expected_end))

    def test_shift_right_full(self):
        self.loop.run_until_complete(self.app.shift_right())
        expected_start = 0 + self.app.delta_shift
        expected_end = 1 + self.app.delta_shift
        assert (math.isclose(self.app.start, expected_start))
        assert (math.isclose(self.app.end, expected_end))

    def test_shift_left_partial(self):
        self.app.range = 0.1
        self.loop.run_until_complete(self.app.shift_left())
        expected_start = 0.43
        expected_end = 0.53
        assert (math.isclose(self.app.start, expected_start))
        assert (math.isclose(self.app.end, expected_end))

    def test_shift_right_partial(self):
        self.app.range = 0.1
        self.loop.run_until_complete(self.app.shift_right())
        expected_start = 0.47
        expected_end = 0.57
        assert (math.isclose(self.app.start, expected_start))
        assert (math.isclose(self.app.end, expected_end))

    def test_zoom_in(self):
        self.loop.run_until_complete(self.app.zoom_in())
        assert (math.isclose(self.app.start, 0.1))
        assert (math.isclose(self.app.end, 0.9))

        self.loop.run_until_complete(self.app.zoom_in())
        assert (math.isclose(self.app.start, 0.18))
        assert (math.isclose(self.app.end, 0.82))

    def test_zoom_out(self):
        self.loop.run_until_complete(self.app.zoom_out())
        assert (math.isclose(self.app.range, 1.2))
        assert (math.isclose(self.app.start, -0.1))
        assert (math.isclose(self.app.end, 1.1))

        self.loop.run_until_complete(self.app.zoom_out())
        assert (math.isclose(self.app.range, 1.44))
        assert (math.isclose(self.app.start, -0.22))
        assert (math.isclose(self.app.end, 1.22))
