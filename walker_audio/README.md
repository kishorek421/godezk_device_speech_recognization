# Walker Audio

A configurable target-speaker pipeline for Walker:

`microphone array -> direction estimation -> beamforming -> echo cancellation -> denoising -> VAD -> speaker verification -> Canary-Qwen`

The camera can provide a face x-coordinate, which is converted to an azimuth and used as the beamforming target. TitaNet verification is optional; when enabled, non-matching speech is discarded before ASR.

## Robot requirements

- Python 3.11 or 3.12
- NVIDIA GPU and a CUDA-compatible PyTorch build for Canary-Qwen, NeMo, and TitaNet
- A synchronized multichannel microphone array recording at 16 kHz
- Known microphone positions in metres, ordered exactly like the input WAV channels
- A playback reference signal when acoustic echo cancellation is needed

## Install

Base DSP and tests:

```bash
python3.11 -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
pytest
```

On the NVIDIA robot after installing the correct PyTorch build for its CUDA/JetPack release:

```bash
pip install -e '.[gpu]'
```

## Configure

Edit `config/walker.yaml` to match the physical array geometry. Angles are relative to the robot: negative is camera-left, zero is forward, and positive is camera-right.

Speaker matching is fail-closed when `target.require_speaker_match` is `true`: processing raises an error unless a verifier and enrollment recording are supplied.

## Run on a multichannel recording

Target a known direction:

```bash
walker-audio input-4ch.wav --target-azimuth 20
```

Target the face at pixel x=1030 in a 1920-pixel camera frame:

```bash
walker-audio input-4ch.wav --face-x 1030 --frame-width 1920 --camera-fov 78
```

Require an enrolled voice by setting `require_speaker_match: true`, then run:

```bash
walker-audio input-4ch.wav --face-x 1030 --frame-width 1920 --speaker-enrollment student.wav
```

The command emits JSON containing detected and target azimuths, acceptance state, speaker scores, segment timestamps, and transcript.

## Runtime integration

Call `WalkerAudioPipeline.process()` with a NumPy array shaped `(samples, microphones)`. Pass the current face azimuth as `target_azimuth_deg`. Pass the robot speaker PCM as `playback_reference` for NLMS echo cancellation. Capture should be chunked into utterance-sized windows; the VAD separates speech inside each window.

The built-in spectral suppressor provides a deterministic CPU fallback. For deployment with NVIDIA's `nvidia/se_den_sb_16k_small`, inject its enhanced waveform at the same point before `EnergyVad`; model loading depends on the NeMo release and robot CUDA image. Canary and TitaNet are loaded lazily so DSP tests can run without GPU packages.
