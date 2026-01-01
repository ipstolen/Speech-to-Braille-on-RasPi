# Dockerfile for Speech-to-Braille-on-RasPi
FROM python:3.12.2

# Set working directory
WORKDIR /app

# Copy all project files
COPY . /app

# Install system dependencies for audio and vosk
RUN apt-get update && apt-get install -y \
    portaudio19-dev \
    alsa-utils \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    pyaudio \
    librosa \
    numpy \
    sounddevice \
    vosk \
    whisper \
    torch \
    transformers \
    keyboard \
    pybrl \
    asciimathml \
    pdfminer.six \
    six

# Download Vosk model (small English model)
RUN mkdir -p /app/models && \
    cd /app/models && \
    wget -O vosk-model-small-en-us-0.15.zip https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip && \
    unzip vosk-model-small-en-us-0.15.zip && \
    rm vosk-model-small-en-us-0.15.zip

# Set environment variable for Vosk model path
ENV VOSK_MODEL_PATH=/app/models/vosk-model-small-en-us-0.15

# Default command
CMD ["python", "test.py"]


