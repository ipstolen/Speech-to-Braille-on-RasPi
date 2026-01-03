import whisper
import pyaudio
import wave
import pybrl
audio = pyaudio.PyAudio()

stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)

frames = []

try:
    while True: #change this later so it works with button on ras pi 5
        data = stream.read(1024)
        frames.append(data)
except KeyboardInterrupt:
    pass
stream.stop_stream()
stream.close()
audio.terminate()

sounds_file = wave.open("large.wav", 'wb')
sounds_file.setnchannels(1)
sounds_file.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
sounds_file.setframerate(16000)
sounds_file.writeframes(b''.join(frames))
sounds_file.close()


model = whisper.load_model("base.en") 
result = model.transcribe("large.wav")

print("Transcribed text:", result["text"])
output_braille = pybrl.translate(result["text"], main_language="english")
print("Braille output:", pybrl.toUnicodeSymbols(output_braille, flatten=True))
 
