import time
import vosk
import json
import wave
import whisper
import librosa
import torch
from transformers import AutoProcessor, AutoModel
import jiwer

class VoskTest:
    def __init__(self, model_path="vosk-model-small-en-us-0.15"):
        self.model = vosk.Model(model_path)
        self.rec = vosk.KaldiRecognizer(self.model, 16000)

    def transcribe(self, audio_path):
        wf = wave.open(audio_path, "rb")
        start_time = time.time() * 1000  # starting in ms
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
        self.dtype = torch.float32  
        self.device = "cpu"  # for raspi, no gpu
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
        predicted_ids = self.model.generate(input_features)
        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        end_time = time.time() * 1000
        transcription_time = end_time - start_time
        return transcription, transcription_time

class WhisperTest:
    def __init__(self):
        self.model = whisper.load_model("tiny.en")

    def transcribe(self, audio_path):
        start_time = time.time() * 1000
        result = self.model.transcribe(audio_path)
        end_time = time.time() * 1000
        transcription_time = end_time - start_time
        return result["text"], transcription_time

def load_ground_truth(audio_name):
    # Load ground truth transcription from file
    filename = f"testAudio/{audio_name.lower()}Transcription.txt"
    try:
        with open(filename, 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        return None

def calculate_wer(hypothesis, reference):
    return jiwer.wer(reference, hypothesis)

def main():
    # Assume audio files are in testAudio directory
    audios = [
        ("Small", "testAudio/small.wav"),
        ("Medium", "testAudio/medium.wav"),
        ("Large", "testAudio/large.wav")
    ]

    vosk_test = VoskTest()
    lite_test = LiteASRTest()
    whisper_test = WhisperTest()

    tests = [
        ("Vosk", vosk_test),
        ("LiteASR", lite_test),
        ("Whisper", whisper_test)
    ]

    results = {}

    for test_name, test_obj in tests:
        print(f"\n{test_name} Results:")
        results[test_name] = {}
        for audio_name, audio_path in audios:
            try:
                text, time_ms = test_obj.transcribe(audio_path)
                print(f"Audio: {audio_name}")
                print(f"Time: {time_ms:.2f} ms")

                # Load ground truth
                ground_truth = load_ground_truth(audio_name)
                if ground_truth:
                    wer = calculate_wer(text, ground_truth)
                    print(f"WER: {wer:.4f}")
                    print(f"Ground Truth: {ground_truth}")
                else:
                    print("Ground truth not found.")
                print("-" * 50)

                results[test_name][audio_name] = {
                    'time': time_ms,
                    'wer': wer if ground_truth else None,
                    'ground_truth': ground_truth
                }

            except Exception as e:
                print(f"{audio_name}: Error - {e}")
                results[test_name][audio_name] = {'error': str(e)}

    # Summary
    print("\n\nSUMMARY:")
    for audio_name, _ in audios:
        print(f"\n{audio_name} Audio:")
        ground_truth = load_ground_truth(audio_name)
        if ground_truth:
            print(f"Ground Truth: {ground_truth[:100]}...")
            for test_name in results:
                if audio_name in results[test_name] and 'wer' in results[test_name][audio_name]:
                    wer = results[test_name][audio_name]['wer']
                    time_ms = results[test_name][audio_name]['time']
                    print(f"{test_name}: WER={wer:.4f}, Time={time_ms:.2f}ms")
        else:
            print("No ground truth available for comparison.")

if __name__ == "__main__":
    main()