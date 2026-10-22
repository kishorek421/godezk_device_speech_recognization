FROM python:3.11-slim

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends libsndfile1 libportaudio2 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY walker_audio ./walker_audio
RUN pip install ./walker_audio

COPY audio_api ./audio_api
COPY word_test_server.py word_test_ui.html ./
COPY config ./config
COPY models ./models
RUN mkdir -p recordings

EXPOSE 8050

CMD ["uvicorn", "word_test_server:app", "--host", "0.0.0.0", "--port", "8050"]
