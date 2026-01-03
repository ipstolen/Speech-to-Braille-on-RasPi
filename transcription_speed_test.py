import time
import vosk
import json
import wave
import whisper
import librosa
import torch
from transformers import AutoProcessor, AutoModel

class VoskTest:
    def __init__(self, model_path="vosk-model-small-en-us-0.15"):
        self.model = vosk.Model(model_path)
        self.rec = vosk.KaldiRecognizer(self.model, 16000)

    def transcribe(self, audio_path):
        wf = wave.open(audio_path, "rb")
        start_time = time.time() * 1000 #starting in ms
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            self.rec.AcceptWaveform(data)
        result = json.loads(self.rec.FinalResult())
        end_time = time.time() * 1000
        transcription_time = end_time - start_time
        return result["text"], transcription_time

class LiteASRTest:
    def __init__(self):
        self.dtype = torch.float32  # Changed from float16 to float32 for CPU compatibility
        self.device = "cuda" if torch.cuda.is_available() else "cpu" # for raspi, no gpu
        self.model = AutoModel.from_pretrained(
            "efficient-speech/lite-whisper-large-v3-turbo",
            trust_remote_code=True,
        )
        self.model.to(self.dtype).to(self.device)
        self.processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")

    def transcribe(self, audio_path):
        audio, _ = librosa.load(audio_path, sr=16000)
        input_features = self.processor(audio, sampling_rate=16000, return_tensors="pt").input_features
        input_features = input_features.to(self.dtype).to(self.device)
        start_time = time.time() * 1000
        predicted_ids = self.model.generate(input_features, max_new_tokens=447)  # Increased max length
        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        end_time = time.time() * 1000
        transcription_time = end_time - start_time
        return transcription, transcription_time

class WhisperTest:
    def __init__(self):
        self.model = whisper.load_model("base.en")

    def transcribe(self, audio_path):
        start_time = time.time() * 1000
        result = self.model.transcribe(audio_path)
        end_time = time.time() * 1000
        transcription_time = end_time - start_time
        return result["text"], transcription_time

def main():
    # Assume audio files are in the same directory
    small_audio = "testAudio/small.wav"
    medium_audio = "testAudio/medium.wav"
    large_audio = "testAudio/large.wav"

    vosk_test = VoskTest()
    lite_test = LiteASRTest()
    whisper_test = WhisperTest()

    tests = [
        ("Vosk", vosk_test),
        ("liteasr", lite_test),
        ("Whisper", whisper_test)
    ]

    audios = [
        ("Small", small_audio),
        ("Medium", medium_audio),
        ("Large", large_audio)
    ]

    for test_name, test_obj in tests:
        print(f"\n{test_name} Tests:")
        for audio_name, audio_path in audios:
            try:
                text, time_ms = test_obj.transcribe(audio_path)
                print("Time: {:.2f} ms | Audio: {} | Transcription: {}".format(time_ms, audio_name, text))
            except Exception as e:
                print(f"{audio_name}: Error - {e}")

if __name__ == "__main__":
    main()