# Walker Audio Transcription API

The audio equivalent of `recognition_api`: a FastAPI service that accepts a recorded audio window, runs Walker denoising and VAD (`walker_audio`), transcribes speech with Canary-Qwen (`walker_audio.models.CanaryTranscriber`, NeMo SALM), and returns the transcript.

## API

- `GET /health` — ok once the Canary model is loaded
- `POST /v1/transcribe` — multipart upload field `audio` with a WAV/FLAC/OGG clip (mono or stereo, any sample rate; resampled to 16 kHz)

Response:

```json
{
  "transcript": "walker can you hear me",
  "speech_seconds": 1.5,
  "peak_db": -6.0,
  "rms_db": -24.0,
  "reason": null,
  "processing_ms": 120.0
}
```

`transcript` is `null` with `reason` set to `no_speech` or `empty_audio` when nothing was said.

## Deploy on the GPU server (Docker)

From this directory on the server:

```bash
cd audio_api
docker compose up -d --build
```

The compose file binds port `8001` via `network_mode: host`, reserves GPU 0 (`device_ids: ["0"]`), and caches Hugging Face models in the `hf-cache` volume. Adjust `.env` for model, dtype, `CUDA_VISIBLE_DEVICES`, and limits.

`walker_audio` ships as a prebuilt wheel in `packages/` so this directory is self-contained. After changing `walker_audio`, rebuild it before deploying:

```bash
cd ../walker_audio
python -m pip wheel . --no-deps -w ../audio_api/packages
```

First start downloads ~5 GB of Canary weights and loads the model; `/health` returns 503 until ready.

## Run without Docker (venv on the server)

```bash
cd audio_api
python3.11 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
pip install ./packages/walker_audio-0.1.0-py3-none-any.whl
uvicorn app.main:app --host 0.0.0.0 --port 8001 --workers 1
```

Use a PyTorch build with CUDA 12.8+ for the RTX 50-series (Blackwell) GPUs.

## Client

`live_audio_test` talks to this API. From the laptop:

```bash
walker-live-audio --server-url http://<server>:8001/v1/transcribe
```

## Mock server for client development

```bash
pip install -r mock_requirements.txt
uvicorn mock_server:app --port 8001
```

Force scenarios with the `X-Mock-Scenario` header: `transcribe` (default), `no_speech`, `empty_audio`, `invalid`, `too_large`.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```
