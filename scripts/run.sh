#!/bin/bash
# Runtime entry point for Speech-to-Braille application

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Activate virtual environment
source "$PROJECT_ROOT/venv/bin/activate"

# Add src to Python path and run the application
cd "$PROJECT_ROOT"
PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH" python3 -m speech_to_braille.main
