# Walker Live Laptop Audio Test

This is the microphone equivalent of `recognition_api/camera_client.py`. It captures the laptop microphone continuously, displays a live level meter, saves test windows, runs Walker denoising and VAD, and optionally transcribes detected speech with Canary-Qwen.

A laptop's single microphone cannot test direction-of-arrival or beamforming. Use this project to validate microphone permissions, audio levels, denoising, VAD, recordings, and ASR before moving to Walker's synchronized microphone array.

## Install

Use Python 3.11 or 3.12 for Canary/NeMo compatibility:

```bash
cd /Users/bell/Desktop/rmk_robotics/v2/live_audio_test
python3.11 -m venv .venv
. .venv/bin/activate
pip install -e ../walker_audio
pip install -e '.[dev]'
```

For local Canary inference, install NeMo and its ASR dependencies:

```bash
pip install -e '.[asr]'
```

Canary-Qwen is a NeMo `SALM` model and cannot be loaded with Transformers `AutoProcessor`.

On macOS, allow Terminal or Devin access under **System Settings → Privacy & Security → Microphone**.

## First microphone test

List audio devices:

```bash
walker-live-audio --list-devices
```

Test capture, recordings, noise suppression, and VAD without loading Canary:

```bash
walker-live-audio --no-asr
```

The level meter should move while speaking. Every five seconds, auto mode analyzes and saves the latest window under `recordings/`.

Use a particular input device:

```bash
walker-live-audio --no-asr --input-device "MacBook Pro Microphone"
```

## Canary test

CPU mode is the default and will be slow for a 2.5B model:

```bash
walker-live-audio
```

NVIDIA laptop:

```bash
walker-live-audio --model-device cuda --dtype bfloat16
```

Apple Silicon can be attempted with:

```bash
walker-live-audio --model-device mps --dtype float16
```

Model support on MPS depends on the installed PyTorch and Transformers versions. Use CPU if model loading or generation reports unsupported MPS operations.

## Remote transcription

Send each recorded window to the Walker Audio Transcription API (`../audio_api`) instead of loading Canary on the laptop:

```bash
walker-live-audio --server-url
```

This defaults to `AUDIO_API_URL` or the deployed server (`http://124.123.89.162:8001/v1/transcribe`). Point at another host or the local mock explicitly:

```bash
walker-live-audio --server-url http://127.0.0.1:8001/v1/transcribe
```

The laptop still captures the microphone, shows the level meter, and saves every analyzed window under `recordings/`; denoising, VAD, and Canary run on the server. `--server-url` cannot be combined with `--no-asr`. Use `--request-timeout` (default 120 s) if the server is slow.

For offline client testing, run `../audio_api/mock_server.py` and force scenarios with its `X-Mock-Scenario` header.

## Controls

Type a command and press Enter:

- `r` or a blank line: analyze the latest window immediately
- `m`: switch between automatic and manual analysis
- `q`: quit
- `Ctrl+C`: quit

Useful options:

```bash
walker-live-audio \
  --window-seconds 8 \
  --vad-threshold-db 6 \
  --output recordings
```

Use `--no-denoise` to compare raw microphone VAD/ASR against the CPU spectral denoiser.
