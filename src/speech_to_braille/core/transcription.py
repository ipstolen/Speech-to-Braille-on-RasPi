"""Transcription module using OpenAI Whisper."""

import whisper
from typing import Dict, Any

from ..config import WHISPER_MODEL, WAV_FILE


class WhisperTranscriber:
    """Handles transcription using OpenAI Whisper model."""
    def __init__(self, model_name: str = WHISPER_MODEL):
        self.model_name = model_name
        self.model = None

    def load_model(self) -> None:
        """Load the Whisper model."""
        print(f"Loading Whisper model: {self.model_name}")
        self.model = whisper.load_model(self.model_name)

    def transcribe(self, audio_file: str = WAV_FILE) -> Dict[str, Any]:
        """Transcribe audio file to text."""
        if self.model is None:
            self.load_model()

        print("Transcribing...")
        result = self.model.transcribe(audio_file)
        return result

    def get_text(self, audio_file: str = WAV_FILE) -> str:
        """Transcribe and return just the text."""
        result = self.transcribe(audio_file)
        return result.get("text", "")
