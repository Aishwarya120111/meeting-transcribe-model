import pyaudio
import numpy as np

class AudioRecorder:
    def __init__(self, rate=16000, chunk_size=1024):
        self.rate = rate
        self.chunk_size = chunk_size
        self.audio_interface = pyaudio.PyAudio()
        self.stream = self.audio_interface.open(format=pyaudio.paInt16,
                                                channels=1,
                                                rate=self.rate,
                                                input=True,
                                                frames_per_buffer=self.chunk_size)
        self.frames = []

    def start_recording(self):
        self.frames = []
        while True:
            data = self.stream.read(self.chunk_size)
            self.frames.append(np.frombuffer(data, dtype=np.int16))
            if len(self.frames) > self.rate // self.chunk_size * 10:  # Keep only last 10 seconds
                self.frames.pop(0)

    def get_audio_data(self):
        if self.frames:
            return np.hstack(self.frames)
        else:
            return np.array([])

    def stop_recording(self):
        self.stream.stop_stream()
        self.stream.close()
        self.audio_interface.terminate()
