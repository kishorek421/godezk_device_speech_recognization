# Walker Audio System Guide

This document describes how the three audio projects work together:

- `live_audio_test`: laptop microphone capture and end-to-end test client.
- `walker_audio`: reusable Walker DSP, spatial filtering, speaker verification, and ASR library.
- `audio_api`: GPU-hosted HTTP transcription service built on the mono-audio parts of `walker_audio`.

## 1. System architecture

### Laptop-to-server testing

```text
Laptop microphone
  -> sounddevice mono capture
  -> rolling audio window
  -> save local WAV
  -> multipart HTTP POST
  -> audio_api
       -> decode and downmix
       -> resample to configured rate (16 kHz by default)
       -> spectral denoising
       -> energy VAD
       -> concatenate speech segments
       -> Canary-Qwen transcription
  <- JSON metrics and transcript
```

### Real Walker microphone-array pipeline

```text
Synchronized multichannel microphone array
  -> direction-of-arrival scan
  -> target-direction check
  -> delay-and-sum beamforming
  -> optional NLMS echo cancellation
  -> optional spectral denoising
  -> energy VAD
  -> optional TitaNet speaker verification
  -> Canary-Qwen transcription
```

These are related but not identical paths. The current `audio_api` accepts an uploaded mono or stereo clip and downmixes it to mono. It does **not** run microphone-array direction estimation, beamforming, echo cancellation, camera targeting, or speaker verification. Those features are implemented by `WalkerAudioPipeline` in `walker_audio` and must run where synchronized array channels and robot context are available.

## 2. Project responsibilities

| Project | Purpose | Main entry point |
|---|---|---|
| `live_audio_test` | Capture laptop audio, show levels, save windows, and analyze locally or remotely | `walker-live-audio` |
| `walker_audio` | Reusable DSP and real Walker target-speaker pipeline | `walker-audio` / `WalkerAudioPipeline` |
| `audio_api` | Load Canary on a GPU server and expose mono transcription over HTTP | `uvicorn app.main:app` |

## 3. `live_audio_test`: laptop live-audio testing

### What it validates

The client validates microphone permissions, input device selection, levels, WAV recording, denoising, VAD, local ASR, and remote API communication. A laptop's single microphone cannot validate direction-of-arrival or beamforming.

### Capture and threading model

`LiveTester` uses three queues and two background threads:

1. A `sounddevice.InputStream` callback receives mono `float32` blocks and puts them in `audio_queue`.
2. The main loop drains those blocks into a bounded rolling window and updates the terminal peak meter every 100 ms.
3. Automatic mode submits one window every `window_seconds`; manual mode submits on `r` or a blank line.
4. `work_queue` has capacity one, preventing slow ASR or server requests from building an unbounded backlog.
5. A worker saves the window as a timestamped WAV and performs local or remote analysis.
6. A command-reader thread handles `r`, `m`, and `q` without blocking capture.

Defaults are 16 kHz, mono, 1,600-sample blocks, a five-second window, automatic analysis, and output under `recordings/`.

### Install

Use Python 3.11 or 3.12:

```bash
cd /Users/bell/Desktop/ai_services/live_audio_test
python3.11 -m venv .venv
. .venv/bin/activate
pip install -e ../walker_audio
pip install -e '.[dev]'
```

For local Canary inference:

```bash
pip install -e '.[asr]'
```

Canary-Qwen is loaded through NeMo's `SALM`, not Transformers `AutoProcessor`. On macOS, grant the terminal application microphone access under **System Settings -> Privacy & Security -> Microphone**.

### Basic tests

```bash
walker-live-audio --list-devices
walker-live-audio --no-asr
walker-live-audio --no-asr --input-device "MacBook Pro Microphone"
```

Local CPU ASR is the default when neither `--no-asr` nor `--server-url` is supplied, but a 2.5B model will be slow:

```bash
walker-live-audio
walker-live-audio --model-device cuda --dtype bfloat16
walker-live-audio --model-device mps --dtype float16
```

MPS support depends on the installed PyTorch/NeMo stack; use CPU when an operation is unsupported.

### Remote API mode

```bash
walker-live-audio --server-url http://127.0.0.1:8001/v1/transcribe
```

Supplying `--server-url` without a value uses `AUDIO_API_URL`; if unset, the code uses its compiled deployment URL. Prefer explicitly configuring `AUDIO_API_URL` rather than relying on that fallback.

The client encodes each window as a floating-point WAV and sends multipart field `audio` with a configurable timeout (120 seconds by default). The server performs denoising, VAD, and ASR. Local WAV saving still occurs. `--server-url` and `--no-asr` are mutually exclusive.

Useful controls and options:

```text
r or blank line   analyze now
m                 toggle automatic/manual mode
q or Ctrl+C       stop
```

```bash
walker-live-audio \
  --window-seconds 8 \
  --vad-threshold-db 6 \
  --output recordings
```

`--no-denoise` only affects local analysis. In remote mode, server-side `DENOISE` controls denoising.

## 4. `walker_audio`: real Walker audio processing

### Input contract

`WalkerAudioPipeline.process()` expects a NumPy `float32`-compatible array shaped:

```text
(samples, microphones)
```

The microphone count and channel order must exactly match `array.microphone_positions_m`. Capture must be synchronized at the configured sample rate, normally 16 kHz. Canary explicitly rejects sample rates other than 16 kHz.

### Processing stages

1. **Direction estimation**: scans configured azimuth angles. For each angle, delay-and-sum energy is calculated; the maximum-energy angle is selected.
2. **Target gate**: the detected angle must be within `azimuth_tolerance_deg` of the requested target. If no target is provided, the detected angle becomes the target.
3. **Beamforming**: fractional delays are applied by interpolation and aligned channels are averaged.
4. **Echo cancellation**: if `playback_reference` is supplied, a stateful normalized least-mean-square filter removes correlated robot-speaker playback. Its length must equal the beamformed microphone window.
5. **Denoising**: an STFT spectral suppressor estimates noise from the beginning of the window and applies a gain floor.
6. **VAD**: 30 ms frames are compared against the 20th-percentile noise floor plus the threshold. Minimum speech and release durations control segment creation.
7. **Speaker verification**: optional TitaNet scoring drops segments below `speaker_threshold`.
8. **ASR**: accepted segments are transcribed by lazily loaded Canary-Qwen/NeMo SALM.

The spectral denoiser is the current implemented enhancement stage. Although configuration includes `enhancement_model: nvidia/se_den_sb_16k_small`, that NVIDIA enhancement model is not loaded by the pipeline.

### Camera targeting

A face center can be converted to an azimuth:

```text
azimuth = (face_x / frame_width - 0.5) * horizontal_camera_fov
```

Negative angles are camera-left, zero is forward, and positive angles are camera-right.

### Configuration

`config/walker.yaml` controls:

- `array`: sample rate, speed of sound, physical microphone coordinates, and scan range/step.
- `enhancement`: enable flag, spectral suppression strength, and initial noise duration.
- `vad`: frame size, threshold above noise, minimum speech, and release duration.
- `target`: angular tolerance, speaker threshold, and whether speaker matching is mandatory.
- `models`: Canary and TitaNet names, device, and dtype.

The provided geometry describes a four-microphone linear array at x positions `-0.09`, `-0.03`, `0.03`, and `0.09` metres. Replace it with measured robot geometry before real use.

If `require_speaker_match` is true, the pipeline fails closed unless a verifier is configured and enrolled. Canary and TitaNet are lazy-loaded so DSP tests do not require GPU packages.

### Install and test

```bash
cd /Users/bell/Desktop/ai_services/walker_audio
python3.11 -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
pytest
```

On the NVIDIA robot, first install a CUDA/JetPack-compatible PyTorch build, then:

```bash
pip install -e '.[gpu]'
```

### File-based CLI

```bash
walker-audio input-4ch.wav --target-azimuth 20
walker-audio input-4ch.wav --face-x 1030 --frame-width 1920 --camera-fov 78
walker-audio input-4ch.wav \
  --face-x 1030 \
  --frame-width 1920 \
  --speaker-enrollment student.wav
```

The CLI emits JSON with detected/target azimuths, acceptance, segments, speaker scores, timestamps, and transcript. It processes existing multichannel files; continuous robot capture must call the Python pipeline from the robot runtime.

### Runtime integration example

```python
from walker_audio.config import WalkerAudioConfig
from walker_audio.pipeline import WalkerAudioPipeline

config = WalkerAudioConfig.from_yaml("config/walker.yaml")
pipeline = WalkerAudioPipeline(config)
result = pipeline.process(
    multichannel_audio,
    target_azimuth_deg=current_face_azimuth,
    playback_reference=robot_playback_pcm,
)
print(result.transcript)
```

Use utterance-sized, synchronized windows. Preserve the same sample clock for the microphone array and align the playback reference with each capture window.

## 5. `audio_api`: GPU transcription API

### Startup and model lifecycle

FastAPI's lifespan creates one `Services` instance and synchronously loads Canary before startup completes. The intended deployment uses one Uvicorn worker so one model copy occupies the GPU. `/health` depends on initialized services and a loaded model.

Processing is currently invoked directly from an async endpoint. There is no explicit inference lock, queue, authentication, or rate limit in the application. Put access control, TLS, request throttling, and concurrency limits in a trusted reverse proxy or private network for production use.

### Endpoints

#### `GET /health`

Successful response:

```json
{
  "status": "ok",
  "model": "nvidia/canary-qwen-2.5b",
  "device": "cuda"
}
```

Returns HTTP 503 while services/model are unavailable.

#### `POST /v1/transcribe`

Content type: `multipart/form-data`

Required field:

- `audio`: WAV, FLAC, or OGG readable by libsndfile; mono or multichannel; any source rate.

The API reads at most `MAX_AUDIO_BYTES + 1`, decodes to `float32`, averages all channels to mono, polyphase-resamples to `SAMPLE_RATE`, checks the duration limit, then denoises, applies VAD, concatenates speech segments, and transcribes.

Example:

```bash
curl -X POST http://127.0.0.1:8001/v1/transcribe \
  -F 'audio=@recordings/sample.wav'
```

Successful speech response:

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

No-speech and empty-audio outcomes are HTTP 200 with `transcript: null` and `reason` set to `no_speech` or `empty_audio`. Metrics are measured on decoded/downmixed audio before denoising. `speech_seconds` is the total retained VAD audio duration, including VAD release padding.

Errors:

| Status | Meaning |
|---|---|
| 400 | Audio cannot be decoded (`Invalid audio`) |
| 413 | Byte limit exceeded (`Audio is too large`) |
| 413 | Duration limit exceeded (`Audio is too long`) |
| 503 | Service/model unavailable or transcription raised an error |
| 422 | Multipart field is missing or malformed (FastAPI validation) |

### Environment configuration

| Variable | Default | Purpose |
|---|---:|---|
| `CANARY_MODEL` | `nvidia/canary-qwen-2.5b` | NeMo model identifier |
| `MODEL_DEVICE` | `cuda` | Torch device |
| `MODEL_DTYPE` | `bfloat16` | Torch dtype name |
| `SAMPLE_RATE` | `16000` | Internal processing rate; keep 16 kHz for Canary |
| `MAX_AUDIO_BYTES` | `10485760` | Maximum upload bytes |
| `MAX_AUDIO_SECONDS` | `60` | Maximum decoded duration |
| `VAD_THRESHOLD_DB` | `8` | VAD threshold above estimated noise floor |
| `DENOISE` | `true` | Enable spectral denoising |

Settings are evaluated when the module is imported, so restart the process after changing environment variables.

### Docker deployment

```bash
cd /Users/bell/Desktop/ai_services/audio_api
docker compose up -d --build
curl http://127.0.0.1:8001/health
```

The image uses PyTorch 2.8 with CUDA 12.8/cuDNN 9, runs as a non-root user, installs `libsndfile` and FFmpeg, and starts one worker on port 8001. Compose uses host networking, reserves NVIDIA GPU 0, and persists Hugging Face cache data. Initial startup downloads roughly 5 GB of model weights; the Docker health check allows a 600-second startup period.

The image installs the prebuilt `packages/walker_audio-0.1.0-py3-none-any.whl`. Rebuild that wheel after changing `walker_audio`:

```bash
cd /Users/bell/Desktop/ai_services/walker_audio
python -m pip wheel . --no-deps -w ../audio_api/packages
cd ../audio_api
docker compose up -d --build
```

### Non-Docker server

```bash
cd /Users/bell/Desktop/ai_services/audio_api
python3.11 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
pip install ./packages/walker_audio-0.1.0-py3-none-any.whl
uvicorn app.main:app --host 0.0.0.0 --port 8001 --workers 1
```

## 6. Mocked end-to-end testing

The mock server has the same routes and response model but no DSP or ML dependencies:

```bash
cd /Users/bell/Desktop/ai_services/audio_api
python3.11 -m venv .venv-mock
. .venv-mock/bin/activate
pip install -r mock_requirements.txt
uvicorn mock_server:app --host 127.0.0.1 --port 8001
```

In another shell:

```bash
cd /Users/bell/Desktop/ai_services/live_audio_test
. .venv/bin/activate
walker-live-audio --server-url http://127.0.0.1:8001/v1/transcribe
```

Mock scenarios can be forced with `X-Mock-Scenario`: `transcribe`, `no_speech`, `empty_audio`, `invalid`, or `too_large`.

```bash
curl -X POST http://127.0.0.1:8001/v1/transcribe \
  -H 'X-Mock-Scenario: no_speech' \
  -F 'audio=@recordings/sample.wav'
```

The live client itself does not expose a CLI option for this test header; use curl or modify/inject the request during automated testing.

## 7. Verification commands

```bash
cd /Users/bell/Desktop/ai_services/walker_audio
. .venv/bin/activate
pytest
ruff check .

cd /Users/bell/Desktop/ai_services/live_audio_test
. .venv/bin/activate
pytest
ruff check .

cd /Users/bell/Desktop/ai_services/audio_api
. .venv/bin/activate
pytest
```

The tests use fake transcribers and mocked HTTP requests where possible, so core DSP and client behavior can be validated without loading Canary.

## 8. Production integration checklist

1. Measure and configure the real microphone coordinates and channel order.
2. Guarantee synchronized multichannel 16 kHz capture.
3. Convert the selected camera face center to Walker azimuth and pass it to the pipeline.
4. Align robot playback PCM with the microphone window before enabling echo cancellation.
5. Record representative robot noise and tune VAD, denoising, angular tolerance, and speaker threshold.
6. Decide where spatial processing runs. The current HTTP API cannot receive or process Walker array context.
7. Use one controlled inference queue per loaded model/GPU or add safe concurrency management.
8. Keep the API private or add reverse-proxy authentication, TLS, rate limits, and request logging that does not retain sensitive audio.
9. Monitor latency, GPU memory, no-speech rate, upload rejection rate, and model failures.
10. Rebuild the packaged `walker_audio` wheel whenever library behavior changes.
