import streamlit as st
import numpy as np
import whisper
from audio_recorder import AudioRecorder
import time

class LiveTranscription:
    def __init__(self, model, audio_recorder):
        self.model = model
        self.audio_recorder = audio_recorder
        self.transcriptions = []

    def transcribe(self):
        try:
            self.audio_recorder.record_chunk()
            audio_data = self.audio_recorder.get_audio_data().astype(np.float32) / 32768.0
            if audio_data.size > 0:
                result = self.model.transcribe(audio_data, fp16=False)
                self.transcriptions.append(result["text"])
            st.session_state.transcriptions = self.transcriptions
        except Exception as e:
            st.error(f"Error during transcription: {e}")

    def stop(self):
        self.audio_recorder.stop_recording()

@st.cache_resource
def get_model():
    return whisper.load_model("base")

@st.cache_resource
def get_audio_recorder(device_index=0):
    return AudioRecorder(device_index=device_index)

def start_transcription():
    st.session_state.transcribing = True
    audio_recorder.start_recording()

def stop_transcription():
    st.session_state.transcribing = False
    live_transcription.stop()

# Streamlit app
st.title("Live Meeting Transcription")

if "transcriptions" not in st.session_state:
    st.session_state.transcriptions = []

if "transcribing" not in st.session_state:
    st.session_state.transcribing = False

model = get_model()
audio_recorder = get_audio_recorder(device_index=0)
live_transcription = LiveTranscription(model, audio_recorder)

if st.button("Start Transcription"):
    start_transcription()

if st.button("Stop Transcription"):
    stop_transcription()

transcription_placeholder = st.empty()

if st.session_state.transcribing:
    live_transcription.transcribe()
    time.sleep(1)

with transcription_placeholder.container():
    st.write("Transcriptions:")
    for transcription in st.session_state.transcriptions:
        st.write(transcription)

st.button("Refresh Transcriptions")
