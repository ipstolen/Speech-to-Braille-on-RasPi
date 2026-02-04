#!/bin/bash
set -e

#echo "Update"
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
    git \
    build-essential \
    cmake

#echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

#echo "Upgrade pip"
pip install --upgrade pip wheel setuptools

pip install \
    pyaudio \
    numpy

# Install whisper-cpp-python with CMAKE_ARGS to bypass version check
echo "Installing whisper-cpp-python (this may take several minutes on Pi5)..."
CMAKE_ARGS="-DCMAKE_POLICY_VERSION_MINIMUM=3.5" pip install whisper-cpp-python

chmod +x scripts/build.sh
chmod +x scripts/run.sh
