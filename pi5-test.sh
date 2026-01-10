#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Starting Whisplay..."

source "$SCRIPT_DIR/venv/bin/activate"

python3 "$SCRIPT_DIR/whisplay.py"