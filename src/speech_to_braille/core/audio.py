"""Audio capture module."""

import pyaudio
import wave
from typing import List

from ..config import RATE, CHUNK, CHANNELS, FORMAT, INPUT_DEVICE_INDEX, WAV_FILE


class AudioRecorder:
    """Handles audio recording from PyAudio input device."""

    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.frames: List[bytes] = []
        self.stream = None

    def start_recording(self) -> None:
        """Start audio recording from the specified input device."""
        format_code = getattr(pyaudio, FORMAT)
        self.stream = self.audio.open(
            format=format_code,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
            input_device_index=INPUT_DEVICE_INDEX
        )
        print("Recording... Press Ctrl+C to stop.")

    def record(self) -> None:
        """Continuously record audio until interrupted."""
        try:
            while True:
                data = self.stream.read(CHUNK, exception_on_overflow=False)
                self.frames.append(data)
        except KeyboardInterrupt:
            print("\nRecording stopped.")

    def stop_recording(self) -> None:
        """Stop recording and close the audio stream."""
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()

    def save_to_file(self, filename: str = WAV_FILE) -> None:
        """Save recorded frames to a WAV file."""
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            format_code = getattr(pyaudio, FORMAT)
            wf.setsampwidth(self.audio.get_sample_size(format_code))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(self.frames))
