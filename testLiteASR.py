import librosa 
import torch
from transformers import AutoProcessor, AutoModel

dtype = torch.float16

# load the compressed Whisper model
model = AutoModel.from_pretrained(
    "efficient-speech/lite-whisper-large-v3-turbo", 
    trust_remote_code=True, 
)
model.to(dtype).to(device)

# we use the same processor as the original model
processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")

# set the path to your audio file
path = "path/to/audio.wav"
audio, _ = librosa.load(path, sr=16000)

input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features
input_features = input_features.to(dtype).to(device)

predicted_ids = model.generate(input_features)
transcription = processor.batch_decode(
    predicted_ids, 
    skip_special_tokens=True
)[0]

print(transcription)

import sounddevice as sd
import numpy as np

samplerate = 16000
chunk = 1024

buffer = []

def callback(indata, frames, time, status):
    buffer.append(indata.copy())

stream = sd.InputStream(
    channels=1,
    samplerate=samplerate,
    blocksize=chunk,
    callback=callback
)

stream.start()

while True:
    if len(buffer) > 10:  # ~0.6 seconds of audio
        audio = np.concatenate(buffer, axis=0).flatten()
        buffer.clear()

        inputs = processor(audio, sampling_rate=16000, return_tensors="pt").input_features
        ids = model.generate(inputs)
        text = processor.batch_decode(ids, skip_special_tokens=True)[0]

        print(text)
