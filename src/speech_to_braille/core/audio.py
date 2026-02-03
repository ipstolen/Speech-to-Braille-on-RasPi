"""Audio capture module with streaming support."""

import pyaudio
import numpy as np
from typing import Generator
import queue
import threading

from ..config import (
    RATE, CHUNK, CHANNELS, FORMAT, INPUT_DEVICE_INDEX, STREAM_CHUNK_SIZE
)


class AudioStreamer:
    """Handles streaming audio capture from PyAudio input device."""

    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.record_thread = None

    def start_stream(self) -> None:
        """Start audio stream from the specified input device."""
        format_code = getattr(pyaudio, FORMAT)
        self.stream = self.audio.open(
            format=format_code,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
            input_device_index=INPUT_DEVICE_INDEX,
            stream_callback=self._audio_callback
        )
        self.is_recording = True
        print("Audio stream started. Speak into the microphone...")
        print("Press Ctrl+C to stop.")

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback function for audio stream."""
        if self.is_recording:
            self.audio_queue.put(in_data)
        return (in_data, pyaudio.paContinue)

    def stream_chunks(self) -> Generator[np.ndarray, None, None]:
        """
        Generator that yields audio chunks as numpy arrays.
        Each chunk contains STREAM_CHUNK_SIZE samples.
        """
        buffer = []
        samples_collected = 0

        try:
            while self.is_recording:
                try:
                    # Get audio data from queue with timeout
                    data = self.audio_queue.get(timeout=0.1)

                    # Convert bytes to numpy array
                    audio_array = np.frombuffer(data, dtype=np.int16)
                    buffer.append(audio_array)
                    samples_collected += len(audio_array)

                    # Yield chunk when we have enough samples
                    if samples_collected >= STREAM_CHUNK_SIZE:
                        chunk = np.concatenate(buffer)
                        # Convert to float32 and normalize to [-1, 1]
                        chunk = chunk.astype(np.float32) / 32768.0
                        yield chunk[:STREAM_CHUNK_SIZE]

                        # Keep remaining samples for next chunk
                        remainder = chunk[STREAM_CHUNK_SIZE:]
                        buffer = [remainder] if len(remainder) > 0 else []
                        samples_collected = len(remainder)

                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"Error in stream_chunks: {e}")
                    continue

        except KeyboardInterrupt:
            print("\nStopping audio stream...")
            self.stop_stream()

    def stop_stream(self) -> None:
        """Stop the audio stream."""
        self.is_recording = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()
        print("Audio stream stopped.")
