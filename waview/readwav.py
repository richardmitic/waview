import audioread
import array
import threading
import sys
from waview import INT16_MAX

class AudioReader():
    def __init__(self, filepath, progress_granularity=1000):
        self.filepath = filepath
        self.progress_granularity = progress_granularity
        self.duration = 0
        self.samplerate = 0
        self.channels = 0
        self.total_samples = 0
        self.__progress = 0
        self.__thread = None
        self.__thread_kill = threading.Event()

    def __read_thread(self):
        with audioread.audio_open(self.filepath) as f:
            self.duration = f.duration
            self.samplerate = f.samplerate
            self.channels = f.channels
            self.__progress = 0
            self.total_samples = int(f.duration * f.samplerate)
            for i, buf in enumerate(f):
                if self.__thread_kill.is_set():
                    return
                self.on_buffer(i, buf, False)
            self.on_buffer(i+1, b"", True)

    def __maybe_report_progress(self, i, final):
        if self.on_progress:
            if (i > 0 and i % self.progress_granularity == 0) or final:
                self.on_progress(self.__progress / self.total_samples, final)

    def read(self):
        self.__thread = threading.Thread(target=self.__read_thread)
        self.__thread.start()

    def flush(self):
        try:
            self.__thread_kill.set()
            self.__thread.join()
        except KeyError as e:
            print(repr(e))

    def on_buffer(self, i, buf, final):
        arr = [samp/INT16_MAX for samp in array.array("h", buf)]
        self.__progress += len(arr)
        self.__maybe_report_progress(i, final)

    def on_progress(self, progress, final):
        pass


if __name__ == "__main__":
    import sys

    class TestAudioReader(AudioReader):
        def __init__(self, filepath):
            super().__init__(filepath)
            self.finished = False

        def on_progress(self, progress, final):
            print(f"\rprogress: {progress:f}", end="\n" if final else "")
            self.finished = final

    filename = sys.argv[1]
    af = TestAudioReader(filename)
    af.read()
    try:
        while not af.finished:
            pass
    except KeyboardInterrupt:
        af.flush()
        print("")
