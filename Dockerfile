# Dockerfile, Image, Container
FROM python:3.12.2

ADD main.py .
ADD liveaudio.py .

RUN pip install pyaudio librosa numpy sounddevice

CMD ["python", "./main.py"]


