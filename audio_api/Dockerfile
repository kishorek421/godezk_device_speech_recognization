FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/app/data \
    HF_HOME=/app/data/.cache/huggingface \
    MPLCONFIGDIR=/app/data/.config/matplotlib

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --uid 10001 --home-dir /app/data --create-home app \
    && mkdir -p /app/data/.cache/huggingface /app/data/.config/matplotlib \
    && chown -R app:app /app/data

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY packages ./packages
RUN pip install ./packages/walker_audio-0.1.0-py3-none-any.whl

COPY --chown=app:app app ./app

USER app
EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8001/health', timeout=4)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
