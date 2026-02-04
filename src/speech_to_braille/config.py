"""Configuration and constants for Speech-to-Braille application."""

# Audio Configuration
RATE = 16000  # Sample rate in Hz (required for Whisper)
CHUNK = 1024  # Frame buffer size
CHANNELS = 1  # Mono audio
FORMAT = "paInt16"  # 16-bit PCM format
INPUT_DEVICE_INDEX = 0  # Microphone device index (0 for MacBook, 2 for Pi5)

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
