# Godezk Device Speech Recognition & Voice Studio

Real-time speech-to-text (ASR), target-speaker verification, and domain terminology correction powered by NVIDIA Canary-Qwen 2.5B, CAM++ VoxCeleb neural embeddings, and audio DSP preprocessing.

---

## Quickstart (Laptop Setup)

### 1. Prerequisites
- Python 3.11 or 3.12
- Git
- FFmpeg (recommended for audio decoding)

### 2. Clone the Repository
```bash
git clone -b dev https://github.com/kishorek421/godezk_device_speech_recognization.git
cd godezk_device_speech_recognization
```

### 3. Create & Activate Virtual Environment
On Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

On Linux / macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -e walker_audio
pip install -e live_audio_test
```

### 5. Start the Voice Studio & Server
```bash
python word_test_server.py
```
Then open **[http://127.0.0.1:8050](http://127.0.0.1:8050)** in your browser.

---

## Running Automated Tests
```bash
pytest walker_audio/tests/ tests/
```

---

## Repository Structure
- `word_test_server.py`: FastAPI server for Voice Studio web interface and live transcription API.
- `word_test_ui.html`: Real-time studio frontend for live microphone recording, file upload, speaker identification, and domain correction.
- `walker_audio/`: Audio preprocessing pipeline (65Hz high-pass filter, adaptive quiet-frame spectral denoising, 3.5dB Energy VAD, CAM++ speaker recognition, domain dictionary).
- `audio_api/`: Microservice for remote GPU inference and LLM-assisted context review.
- `live_audio_test/`: Laptop microphone test utility and CLI.
- `config/`: Speaker registry configuration template (`speaker_registry.example.json`).
- `recordings/`: Local audio recordings directory (audio files are ignored by git).
