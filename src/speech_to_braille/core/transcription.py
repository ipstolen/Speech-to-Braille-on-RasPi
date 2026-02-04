"""Transcription module using whisper.cpp for streaming."""

import numpy as np
from typing import Optional
from pathlib import Path
import urllib.request
import tempfile
import wave
from whisper_cpp_python import Whisper

from ..config import WHISPER_MODEL, WHISPER_MODEL_PATH, RATE


# Model download URLs from whisper.cpp repository
MODEL_URLS = {
    "tiny": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin",
    "tiny.en": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin",
    "base": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin",
    "base.en": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin",
    "small": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin",
    "small.en": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.en.bin",
    "medium": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin",
    "medium.en": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.en.bin",
    "large": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin",
}


class StreamingTranscriber:
    """Handles streaming transcription using whisper.cpp."""

    def __init__(self, model_name: str = WHISPER_MODEL):
        self.model_name = model_name
        self.model_path = WHISPER_MODEL_PATH
        self.whisper = None

        # Set default model cache directory
        self.cache_dir = Path.home() / ".cache" / "whisper-cpp"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _download_model(self, model_name: str) -> Path:
        """Download whisper.cpp model if not already cached."""
        model_file = self.cache_dir / f"ggml-{model_name}.bin"

        if model_file.exists():
            print(f"Model already cached at {model_file}")
            return model_file

        if model_name not in MODEL_URLS:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_URLS.keys())}")

        url = MODEL_URLS[model_name]
        print(f"Downloading {model_name} model from {url}...")
        print("This may take a few minutes...")

        try:
            urllib.request.urlretrieve(url, model_file)
            print(f"Model downloaded to {model_file}")
            return model_file
        except Exception as e:
            raise RuntimeError(f"Failed to download model: {e}")

    def load_model(self) -> None:
        """Load the whisper.cpp model."""
        print(f"Loading Whisper.cpp model: {self.model_name}")

        # Get or download model file
        if self.model_path:
            model_file = Path(self.model_path)
            if not model_file.exists():
                raise FileNotFoundError(f"Model file not found: {model_file}")
        else:
            model_file = self._download_model(self.model_name)

        # Initialize Whisper with model path
        self.whisper = Whisper(model_path=str(model_file), n_threads=4)
        print("Model loaded successfully.")

    def transcribe_chunk(self, audio_chunk: np.ndarray) -> Optional[str]:
        """
        Transcribe a single audio chunk.

        Args:
            audio_chunk: numpy array of audio samples (float32, normalized to [-1, 1])

        Returns:
            Transcribed text or None if transcription fails
        """
        if self.whisper is None:
            self.load_model()

        try:
            # Ensure audio is float32
            if audio_chunk.dtype != np.float32:
                audio_chunk = audio_chunk.astype(np.float32)

            # Convert to int16 for WAV file
            audio_int16 = (audio_chunk * 32767).astype(np.int16)

            # Save to temporary WAV file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
                temp_path = temp_wav.name
                with wave.open(temp_path, 'wb') as wf:
                    wf.setnchannels(1)  # Mono
                    wf.setsampwidth(2)  # 16-bit
                    wf.setframerate(RATE)
                    wf.writeframes(audio_int16.tobytes())

            # Transcribe the file
            result = self.whisper.transcribe(temp_path, language='en')

            # Clean up temp file
            Path(temp_path).unlink()

            # Extract text from result
            if result and hasattr(result, 'text'):
                return result.text.strip()
            elif result and isinstance(result, str):
                return result.strip()
            elif result and isinstance(result, dict) and 'text' in result:
                return result['text'].strip()
            else:
                return None

        except Exception as e:
            print(f"Transcription error: {e}")
            return None

    def stream_transcribe(self, audio_generator):
        """
        Process streaming audio chunks and yield transcriptions.

        Args:
            audio_generator: Generator yielding audio chunks (numpy arrays)

        Yields:
            Transcribed text for each chunk
        """
        if self.whisper is None:
            self.load_model()

        chunk_num = 0
        for audio_chunk in audio_generator:
            chunk_num += 1
            print(f"\n[Chunk {chunk_num}] Processing {len(audio_chunk)/RATE:.1f}s of audio...")

            text = self.transcribe_chunk(audio_chunk)

            if text:
                yield text
            else:
                print(f"[Chunk {chunk_num}] No transcription")
