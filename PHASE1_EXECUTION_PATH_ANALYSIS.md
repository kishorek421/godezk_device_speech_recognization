# Phase 1: Execution Path Analysis

## Complete Execution Paths

### Path 1: Browser UI via word_test_server.py
**Entry:** Browser microphone → POST /api/transcribe

1. **Audio Ingestion** (word_test_server.py:304)
   - `prepare_audio_for_canary(content)` processes audio
   - 65Hz Butterworth high-pass filter (line 86)
   - Normalization to -1.2 dBFS (line 93)
   - Resample to 16kHz if needed (line 74)

2. **Speaker Identification** (word_test_server.py:333)
   - CAM++ VoxCeleb via speaker_recognizer.py
   - Threshold: 0.62 for <1.2s clips, 0.68 for longer clips

3. **Remote ASR Call** (word_test_server.py:340)
   - POST to http://124.123.89.162:8001/v1/transcribe
   - Sends cleaned audio bytes

4. **Remote API Processing** (audio_api/app/main.py)
   - 180Hz Butterworth high-pass filter (line 82) - **DIFFERENT FROM WORD_TEST_SERVER**
   - SpectralDenoiser (line 135) - assumes first 250ms is noise
   - EnergyVad (line 148)
   - Canary-Qwen 2.5B transcription (line 176)
   - Dictionary correction (line 188)

5. **Post-Processing** (word_test_server.py:370)
   - Local dictionary correction (re-applied)
   - Optional LLM context review (line 382)

### Path 2: Live Microphone via live_audio_test CLI
**Entry:** sounddevice → LiveTester

1. **Audio Capture** (cli.py:40-45)
   - sounddevice.InputStream callback
   - Rolling window (default 5s)

2. **Local Analysis** (analyzer.py:35)
   - SpectralDenoiser (line 41)
   - EnergyVad (line 42)
   - CanaryTranscriber (local) OR RemoteAnalyzer

3. **If RemoteAnalyzer** (remote.py:31)
   - Sends to http://124.123.89.162:8001/v1/transcribe
   - Same as Path 1 remote processing

### Path 3: Offline File via walker_audio Pipeline
**Entry:** WalkerAudioPipeline.process()

1. **Spatial Processing** (pipeline.py:91-99)
   - Direction estimation
   - Delay-and-sum beamforming
   - Echo cancellation (if reference provided)

2. **Audio Processing** (pipeline.py:102-104)
   - SpectralDenoiser
   - EnergyVad

3. **Speaker Verification** (pipeline.py:108-110)
   - TitaNetVerifier (if configured)

4. **Transcription** (pipeline.py:111)
   - CanaryTranscriber

## Identified Issues

### CRITICAL: Inconsistent High-Pass Filter Cutoffs
- word_test_server.py: **65Hz** (line 86)
- audio_api/app/main.py: **180Hz** (line 82)
- walker_audio/processing.py: **180Hz** default (line 11)
- Impact: Audio processed differently depending on entry point

### CRITICAL: SpectralDenoiser Assumes First 250ms is Noise
- processing.py:73-74
- `noise_frames = max(1, int(self.noise_seconds * self.sample_rate / (nperseg // 2)))`
- Default: 0.25 seconds of audio assumed to be noise-only
- Impact: Speech starting immediately will be suppressed

### HIGH: EnergyVAD Lacks Pre-roll/Post-roll Padding
- processing.py:134
- Adds release_ms padding only at end
- No pre-roll padding at segment start
- Impact: Beginning of words may be cut off

### MEDIUM: Canary-Qwen Prompts Not Strictly Verbatim
- models.py:12-15
- Prompts: "Transcribe the following clinical/medical speech accurately:"
- Impact: Model may paraphrase instead of exact transcription

### MEDIUM: Hardcoded Remote GPU Server IP
- word_test_server.py:23 - http://124.123.89.162:8001
- live_audio_test/remote.py:10 - http://124.123.89.162:8001
- Impact: Not configurable via environment variable

### LOW: Inconsistent Logging
- print() statements in audio_api/app/main.py (lines 71, 77, 79, 96, 102, 105, 172, 184, 262)
- logging module used in speaker_recognizer.py and context_engine.py
- Impact: Difficult to debug production issues

### LOW: Generic Exception Handling
- Multiple locations catch bare `Exception`
- word_test_server.py:38, 54, 65, 128, 152, 192, 197, 205, 237, 239, 319
- Impact: Loss of specific error information

## Next Steps

1. **Fix high-pass filter inconsistency** - standardize on 65Hz (preserves speech better)
2. **Improve SpectralDenoiser** - add speech-at-start detection or configurable noise window
3. **Add VAD padding** - pre-roll and post-roll to preserve word boundaries
4. **Make remote GPU URL configurable** - use environment variable
5. **Replace print() with logging** - structured logging throughout
6. **Improve exception handling** - catch specific exceptions
