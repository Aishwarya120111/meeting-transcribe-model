import sys
import os
import time

# Ensure the current directory is in the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import threading
import numpy as np  # Explicitly import numpy before whisper
import whisper
from audio_recorder import AudioRecorder

class LiveTranscription:
    def __init__(self, model, audio_recorder):
        self.model = model
        self.audio_recorder = audio_recorder
        self.stop_event = threading.Event()

    def transcribe(self):
        # Wait for a few seconds to ensure audio data is captured
        time.sleep(5)
        while not self.stop_event.is_set():
            audio_data = self.audio_recorder.get_audio_data().astype(np.float32) / 32768.0
            if audio_data.size > 0:
                result = self.model.transcribe(audio_data, fp16=False)
                print(result["text"])
            else:
                print("No audio data captured yet.")

    def start(self):
        self.stop_event.clear()
        threading.Thread(target=self.audio_recorder.start_recording).start()
        threading.Thread(target=self.transcribe).start()

    def stop(self):
        self.stop_event.set()
        self.audio_recorder.stop_recording()

# Example usage
if __name__ == "__main__":
    model = whisper.load_model("base")
    audio_recorder = AudioRecorder()
    live_transcription = LiveTranscription(model, audio_recorder)
    live_transcription.start()

    # To stop, use the following line:
    # live_transcription.stop()
