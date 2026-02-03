"""Configuration and constants for Speech-to-Braille application."""

import os
from pathlib import Path

# Load .env file if it exists
def load_env():
    """Load environment variables from .env file."""
    env_file = Path(__file__).parent.parent.parent / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

load_env()

# Audio Configuration
RATE = 16000  # Sample rate in Hz (required for Whisper)
CHUNK = 1024  # Frame buffer size
CHANNELS = 1  # Mono audio
FORMAT = "paInt16"  # 16-bit PCM format

# Input device index - load from environment or use default
# Run 'python scripts/detect_audio.py' to configure
_device_from_env = os.environ.get('AUDIO_DEVICE_INDEX')
if _device_from_env:
    INPUT_DEVICE_INDEX = int(_device_from_env)
else:
    INPUT_DEVICE_INDEX = None  # None = use system default

# Streaming Configuration
STREAM_CHUNK_DURATION = 2.0  # Duration in seconds for each transcription chunk
STREAM_CHUNK_SIZE = int(RATE * STREAM_CHUNK_DURATION)  # Number of samples per chunk
VAD_THRESHOLD = 0.5  # Voice activity detection threshold (not implemented yet)

# File Configuration (for fallback/debugging)
WAV_FILE = "outputTest.wav"

# Whisper.cpp Model Configuration
WHISPER_MODEL = "base.en"  # Model size: tiny, base, small, medium, large
WHISPER_MODEL_PATH = None  # Auto-download if None

# Braille Configuration
BRAILLE_TABLE = "en-us-g2.ctb"
BRAILLE_LANGUAGE = "english"
