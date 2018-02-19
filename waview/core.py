import logging
import sys
import asyncio
import wave
import array
import math
import numpy as np
from enum import Enum
from scipy.io import wavfile
from scipy.signal import resample, resample_poly

LOG = logging.getLogger(__name__)

INT16_MAX = int(2**15)-1

class WavType(Enum):
    SAMPLES=1
    PEAKS=2

def rms(arr):
    big_arr = np.array(arr, dtype=np.float32) / INT16_MAX
    return np.sqrt(np.sum(big_arr**2, axis=1))

async def _default_callback(event):
    LOG.error(f"{event}")


class LoggerWriter():
    """ Python logger that looks and acts like a file.
    """
    def __init__(self, level):
        self.level = level

    def write(self, message):
        msg = message.strip()
        if msg and msg != "\n":
            self.level(msg)

    def flush(self):
        pass


class WaviewCache():
    def __init__(self, key):
        self.key = key
        self.peaks = []


class WaviewCore():
    """ The core processor of waview. Handles all audio processing and anaylsis.
    """
    
    def __init__(self, on_event=_default_callback):
        self._on_event = on_event
        self._cache = None
        self._sample_cache = None

    def kill(self):
        pass

    @staticmethod
    def ensure_dimensions(samples):
        if isinstance(samples[0], np.ndarray):
            return samples.T
        else:
            return np.expand_dims(samples, 0) # So we can still index the only channel

    def load_samples(self, wav, start, end):
        "Very slow. Run in an async executor."
        if self._sample_cache is not None:
            samples = self._sample_cache
        else:
            sample_rate, samples = wavfile.read(wav)
            self._sample_cache = samples
        samples = self.ensure_dimensions(np.divide(samples, float(INT16_MAX)))
        start_index = int(np.clip(start, 0, 1) * samples.shape[1])
        end_index = int(np.clip(end, 0, 1) * samples.shape[1])
        head_size = -start if start < 0 else 0
        tail_size = end - 1 if end > 1 else 0
        head_samples = np.zeros((samples.shape[0], int(head_size * samples.shape[1])))
        tail_samples = np.zeros((samples.shape[0], int(tail_size * samples.shape[1])))
        # print(head_size, tail_size, head_samples.shape, tail_samples.shape, samples.shape, start_index, end_index)
        samples = np.concatenate((head_samples, samples[:,start_index:end_index], tail_samples), axis=1)
        return samples

    async def get_peaks(self, wav, start=0., end=1., num_peaks=None):
        """ Get the wave peaks for <wav>
            [param] wav Either a file path or file handle of a wav file
            [param] start Start position of wav file [0.0 - 1.0]
            [param] end End position of wav file [0.0 - 1.0]
            [param] num_peaks The number of peaks wanted in the result. If None,
                    waview will choose automatically.
            [return] Numpy array of peaks
        """

        def peaks(samples, num_peaks):
            """ Split into chunks and take the mean absolute value of each.
                Drop incomplete chunk from the end if present.
            """
            chunk_size = samples.shape[1] // num_peaks if num_peaks is not None else 1024
            n_splits = num_peaks if num_peaks is not None else samples.shape[1] // chunk_size
            last_index = (samples.shape[1] // chunk_size) * chunk_size
            chunks = np.array(np.split(samples[:,:last_index], n_splits, axis=1))
            avg = np.sum(np.abs(chunks), axis=2) / chunk_size
            return avg

        def perform(wav, start, end, num_peaks):
            "Fucntion is slow. Run in an async executor."
            samples = self.load_samples(wav, start, end)
            p = peaks(samples, num_peaks).T
            return p
    
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, perform, wav, start, end, num_peaks)

    async def get_samples(self, wav, start=0., end=1., num_samps=None):
        """ Get raw samples for <wav>, resampled to to <num_samps> values.
            [param] wav Either a file path or file handle of a wav file
            [param] start Start position of wav file [0.0 - 1.0]
            [param] end End position of wav file [0.0 - 1.0]
            [param] num_samps The number of sample wanted in the result. If None,
                    waview will choose automatically.
            [return] Numpy array of samples
        """
        
        def perform(wav, start, end, num_samps):
            "Fucntion is slow. Run in an async executor."
            samples = self.load_samples(wav, start, end)
            # Only resample if we've been asked for a specific number of samples.
            # Otherwise just return the samples without alteration.
            s = resample_poly(samples, num_samps, samples.shape[1], axis=1) if num_samps is not None else samples
            return s
    
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, perform, wav, start, end, num_samps)

    async def get_wav(self, wav, start=0., end=1., num_points=None):
        """ Get a view of <wav> with the given parameters, letting Waview choose
            whether to give peaks or samples.
        """
        samples = self.load_samples(wav, start, end)
        samples_per_point = samples.shape[1] / num_points if num_points else -1
        print(samples.shape, samples_per_point)
        if samples_per_point > 4:
            data = await self.get_peaks(wav, start, end, num_points)
            data_type = WavType.PEAKS
        else:
            data_type = WavType.SAMPLES
            data = await self.get_samples(wav, start, end, num_points)
        return data_type, data