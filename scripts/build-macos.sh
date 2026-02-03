#!/bin/bash
set -e

echo "Installing system dependencies via Homebrew..."
# Install Homebrew if not present
if ! command -v brew &> /dev/null; then
    echo "Homebrew not found. Install it from https://brew.sh"
    exit 1
fi

# Install dependencies
brew install portaudio ffmpeg

echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip wheel setuptools

echo "Installing Python packages..."
pip install \
    torch \
    torchvision \
    torchaudio \
    openai-whisper \
    pyaudio

chmod +x scripts/build-macos.sh
chmod +x scripts/run.sh

echo "Setup complete! Run './scripts/run.sh' to start the app."
