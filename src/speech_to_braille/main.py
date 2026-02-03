"""Main entry point for Speech-to-Braille application."""

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from .core import AudioRecorder, WhisperTranscriber, BrailleTranslator
from .config import WAV_FILE


def main():
    """Run the speech-to-braille pipeline."""
    # Audio capture
    recorder = AudioRecorder()
    recorder.start_recording()
    recorder.record()
    recorder.stop_recording()
    recorder.save_to_file(WAV_FILE)

    # Transcription
    transcriber = WhisperTranscriber()
    text = transcriber.get_text(WAV_FILE)

    print("Transcribed text:")
    print(text)

    # Braille translation (currently disabled)
    # translator = BrailleTranslator()
    # braille_output = translator.translate(text)
    # if braille_output:
    #     print("Braille output:")
    #     print(braille_output)


if __name__ == "__main__":
    main()
