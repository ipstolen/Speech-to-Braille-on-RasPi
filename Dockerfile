FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y \
    portaudio19-dev \
    alsa-utils \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt
COPY . /app
ENV VOSK_MODEL_PATH=/app/vosk-model-small-en-us-0.15



