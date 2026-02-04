"""Main entry point for Speech-to-Braille streaming application."""

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from .core import AudioStreamer, StreamingTranscriber, BrailleTranslator
from .config import WAV_FILE


def main():
    """Run the streaming speech-to-braille pipeline."""
    print("=" * 60)
    print("Speech-to-Braille Streaming System")
    print("=" * 60)

    # Initialize components
    audio_streamer = AudioStreamer()
    transcriber = StreamingTranscriber()

    # Pre-load the model before starting audio stream
    print("\nInitializing transcription model...")
    transcriber.load_model()

    # Start audio streaming
    audio_streamer.start_stream()

    # Process streaming audio and transcribe in real-time
    print("\nStarting real-time transcription...\n")
    try:
        audio_generator = audio_streamer.stream_chunks()
        text_generator = transcriber.stream_transcribe(audio_generator)

        for transcribed_text in text_generator:
            print("=" * 60)
            print("TRANSCRIPTION:")
            print(transcribed_text)
            print("=" * 60)

            # TODO: Braille translation (currently disabled)
            # translator = BrailleTranslator()
            # braille_output = translator.translate(transcribed_text)
            # if braille_output:
            #     print("\nBRAILLE:")
            #     print(braille_output)
            #     print("=" * 60)

    except KeyboardInterrupt:
        print("\n\nShutting down...")
    finally:
        audio_streamer.stop_stream()
        print("Done.")


if __name__ == "__main__":
    main()
