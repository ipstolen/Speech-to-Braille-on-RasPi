import time
import librosa
import torch
from transformers import AutoProcessor, AutoModel


class LiteASRTest:
    def __init__(self):
        self.transcription_time = 0
        self.dtype = torch.float32
        self.device = "cpu" 
        self.model = AutoModel.from_pretrained(
            "efficient-speech/lite-whisper-large-v3-turbo",
            trust_remote_code=True,
        ).to(self.dtype).to(self.device)

        self.processor = AutoProcessor.from_pretrained(
            "openai/whisper-large-v3"
        )

        # Chunking parameters
        self.sample_rate = 16000
        self.chunk_seconds = 20
        self.chunk_samples = self.sample_rate * self.chunk_seconds

    def split_audio(self, audio):
        """Split audio into fixed-length chunks"""
        return [
            audio[i:i + self.chunk_samples]
            for i in range(0, len(audio), self.chunk_samples)
        ]

    def transcribe(self, audio_path):
        audio, _ = librosa.load(audio_path, sr=self.sample_rate)
        chunks = self.split_audio(audio)

        full_transcription = []
        total_time = 0

        for chunk in chunks:
            input_features = self.processor(
                chunk,
                sampling_rate=self.sample_rate,
                return_tensors="pt"
            ).input_features.to(self.dtype).to(self.device)

            start_time = time.time() * 1000
            predicted_ids = self.model.generate(input_features)
            end_time = time.time() * 1000

            text = self.processor.batch_decode(
                predicted_ids,
                skip_special_tokens=True
            )[0]

            full_transcription.append(text)
            total_time += (end_time - start_time)

        self.transcription_time = total_time
        print(f"Total transcription time: {total_time:.2f} ms")

        return " ".join(full_transcription)

    def getTime(self):
        return self.transcription_time


def main():
    lite_test = LiteASRTest()
    audio_file = "testAudio/large.wav"  # path to your audio file

    transcription = lite_test.transcribe(audio_file)
    print("\n--- TRANSCRIPTION ---")
    print(transcription)


if __name__ == "__main__":
    main()
