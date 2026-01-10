import whisper
import pyaudio
import wave
##import pybrl

RATE = 16000
CHUNK = 1024
CHANNELS = 1
FORMAT = pyaudio.paInt16
WAV_FILE = "outputTest.wav"

audio = pyaudio.PyAudio()

stream = audio.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK,
    input_device_index=2
)

frames = []

print("Recording... Press Ctrl+C to stop.")

try:
    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)
except KeyboardInterrupt:
    print("\nRecording stopped.")

stream.stop_stream()
stream.close()
audio.terminate()

with wave.open(WAV_FILE, 'wb') as wf:
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(pyaudio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))

print("Transcribing...")

model = whisper.load_model("base.en")
result = model.transcribe(WAV_FILE)

print("Transcribed text:")
print(result["text"])

##output_braille = pybrl.translate(result["text"], main_language="english")
##print("Braille output:")
##print(pybrl.toUnicodeSymbols(output_braille, flatten=True))
