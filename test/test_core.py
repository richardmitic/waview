from waview.core import WaviewCore, INT16_MAX, WavType
import asyncio
import os
import wave
import pytest
from scipy.io import wavfile
import numpy as np

core = None
loop = None

def resource(filename):
    return os.path.join(os.path.dirname(__file__), "..", "resources", filename)

def write_test_wav_file_1():
    wav = np.repeat(np.arange(0, 10, dtype=np.int16), 1024)
    wavfile.write(resource("test1.wav"), 8000, wav)

def write_test_wav_file_2():
    wav = np.repeat(np.arange(0, -10, -1, dtype=np.int16), 1024)
    wavfile.write(resource("test2.wav"), 8000, wav)

def setup_module(module):
    write_test_wav_file_1()
    write_test_wav_file_2()

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


class TestGetWav(BaseTest):
    
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
