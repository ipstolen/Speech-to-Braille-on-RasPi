#!/bin/bash
set -e

echo "Update"
sudo apt update
sudo apt upgrade -y

echo "Install system dependencies"
sudo apt install -y \
    python3 \
    python3-venv \
    python3-pip \
    portaudio19-dev \
    libasound2-dev \
    ffmpeg \
    git

echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "Upgrade pip"
pip install --upgrade pip wheel setuptools

pip install \
    torch \
    torchvision \
    torchaudio \
    openai-whisper \
    pyaudio \

chmod +x build.sh
