"""Configuration and constants for Speech-to-Braille application."""

# Audio Configuration
RATE = 16000  # Sample rate in Hz (required for Whisper)
CHUNK = 1024  # Frame buffer size
CHANNELS = 1  # Mono audio
FORMAT = "paInt16"  # 16-bit PCM format
INPUT_DEVICE_INDEX = 2  # Pi5-specific device index

# File Configuration
WAV_FILE = "outputTest.wav"

# Model Configuration
WHISPER_MODEL = "base.en"

# Braille Configuration
BRAILLE_TABLE = "en-us-g2.ctb"
BRAILLE_LANGUAGE = "english"
