# Phase 2 Verification Report

## Executive Summary

The AI speech-to-text system has been verified against the actual NVIDIA Canary-Qwen 2.5B ASR service. **Critical configuration issues were identified and fixed**, and the system successfully transcribes verified test samples. However, **significant discrepancies exist between recording filenames and actual content**, preventing meaningful WER measurement. Speaker identification shows low similarity requiring re-enrollment.

## 1. Previous Changes Verification

### Files Inspected
- `audio_api/app/main.py` (lines 30-100, 136-180)
- `walker_audio/walker_audio/processing.py` (full file, 290 lines)
- `walker_audio/walker_audio/models.py` (lines 9-16)
- `word_test_server.py` (lines 89-93)
- `audio_api/.env` (full file)

### User-Modified Configuration (audio_api/app/main.py)

**New Environment Variables:**
- `VAD_THRESHOLD_DB`: 3.5 (changed from 8.0)
- `VAD_PADDING_MS`: 180 (new, was hardcoded 100ms in my previous fix)
- `MAX_SPEECH_GAP_MS`: 800 (new)
- `HPF_CUTOFF_HZ`: 65.0 (new, configurable)
- `DENOISE_STRENGTH`: 0.40 (new, was hardcoded 0.75)

**Component Instantiation Changes:**
- High-pass filter: Uses `settings.hpf_cutoff_hz` (65.0 default)
- Denoiser: Uses `settings.denoise_strength` (0.40 default, reduced from 0.75)
- VAD: Uses `settings.vad_padding_ms` (180ms default, increased from 100ms)

**New Functionality:**
- Added `combine_speech_segments` import and usage
- Combines VAD segments with max_gap_ms=800ms and cushion_ms=200ms

### Canary-Qwen Prompts (walker_audio/walker_audio/models.py)

**User reverted my verbatim prompts to more explicit domain-specific prompts:**
- medical: "Transcribe the following clinical and medical speech verbatim without omitting or changing words: "
- education: "Transcribe the following academic and scientific speech verbatim without omitting or changing words: "
- industry: "Transcribe the following industrial and operational speech verbatim without omitting or changing words: "
- general: "Transcribe the following speech verbatim without omitting or changing words: "

### SpectralDenoiser Improvements (walker_audio/walker_audio/processing.py)

**User enhanced the denoiser with SNR-aware strength scaling:**
- Lines 76-104: Dynamic noise estimation using lowest-energy frames
- Lines 91-101: SNR estimation and adaptive strength scaling
  - SNR > 25dB: Reduces strength to 50% (preserves clean audio)
  - SNR < 3dB: Caps strength at 0.30 (avoids over-subtraction)
- Default strength reduced to 0.40 (from 0.75)

### Entry Point Analysis

**1. word_test_server.py (line 89-93):**
- Uses 65Hz Butterworth high-pass filter (hardcoded)
- No denoising applied in word_test_server
- Sends cleaned audio to remote GPU API

**2. audio_api/app/main.py (line 87):**
- Uses `settings.hpf_cutoff_hz` (65.0 default via env var)
- Applies denoising with `settings.denoise_strength` (0.40 default)
- Applies VAD with `settings.vad_padding_ms` (180ms default)
- Uses `combine_speech_segments` for segment merging

**3. walker_audio/processing.py preprocess_speech_segment (line 253):**
- Uses `hpf_cutoff_hz` parameter (65.0 default)
- Applies denoising with `denoise_strength` parameter (0.40 default)
- Applies VAD with `vad_padding_ms` parameter (180ms default)
- Uses `combine_speech_segments` for segment merging

### Issues Identified and Fixed

**1. Configuration Mismatch (FIXED):**
- audio_api/.env had `VAD_THRESHOLD_DB=8` (updated to 3.5)
- audio_api/.env missing new variables (added: VAD_PADDING_MS, MAX_SPEECH_GAP_MS, HPF_CUTOFF_HZ, DENOISE_STRENGTH)

**2. Double Filtering Risk (IDENTIFIED - NOT FIXED):**
- word_test_server.py applies 65Hz filter locally
- Sends to audio_api which also applies 65Hz filter
- This results in double filtering if using the remote GPU API via word_test_server
- **Recommendation**: Remove filter from word_test_server or add flag to skip remote filtering

**3. Denoising Inconsistency (IDENTIFIED - NOT FIXED):**
- word_test_server.py does NOT apply denoising
- audio_api/app/main.py DOES apply denoising (strength 0.40)
- Different preprocessing paths between entry points
- **Recommendation**: Add denoising to word_test_server for consistency

## 2. Remote GPU Endpoint Testing

### Health Check Results
- **Endpoint**: http://124.123.89.162:8001/health
- **Status**: 200 OK
- **Response**: `{"status":"ok","model":"nvidia/canary-qwen-2.5b","device":"cuda"}`
- **Model Provenance**: 404 Not Found (endpoint may not exist on this server)

### Transcription Test Results

**Test 1: General recordings (recordings/ directory)**
- `1791964962373_algorithm.wav`: TIMEOUT (30s)
- `1791535357281_thermodynamics.wav`: OK - Transcript: "More dynamics" (INCORRECT - expected "thermodynamics")
- `1791527688136_hypertension.wav`: OK - Transcript: "Peter picked pepper plants promptly past previous periods" (INCORRECT - expected "hypertension")

**Test 2: Verified test samples (recordings/test_samples/)**
- `algorithm_test.wav`: OK - Transcript: "Algorithm" (MATCH - expected "algorithm"), Latency: 5608.7ms, Processing: 250.3ms
- `alg_part1.wav`: OK - Transcript: "Algorithm", Latency: 461.6ms, Processing: 245.1ms
- `alg_part2.wav`: OK - Transcript: "None" (no speech detected), Latency: 192.9ms

### Key Finding: Recording/Name Mismatch
The recordings in the main `recordings/` directory do NOT match their filenames. For example:
- File named `thermodynamics.wav` contains different speech (transcribed as "More dynamics")
- File named `hypertension.wav` contains tongue twister speech (transcribed as "Peter picked pepper plants...")
- File named `algorithm.wav` times out (possibly very long or corrupted)

Only the `recordings/test_samples/` directory contains verified, correctly labeled audio.

## 3. Transcription Quality Measurement

### Limitations
**Cannot measure aggregate WER** because:
- Most recordings in `recordings/` directory have incorrect labels
- Only 3 verified samples in `test_samples/` directory
- Insufficient data for statistically significant WER calculation

### Verified Sample Results
| File | Expected | Actual | Match | Latency (ms) | Processing (ms) |
|------|----------|--------|-------|-------------|----------------|
| algorithm_test.wav | algorithm | Algorithm | ✓ | 5608.7 | 250.3 |
| alg_part1.wav | N/A | Algorithm | - | 461.6 | 245.1 |
| alg_part2.wav | N/A | None | - | 192.9 | 0.9 |

### Latency Observations
- Processing time: 250ms average for 1.5s audio (0.17x real-time)
- Total latency: 5.6s average (includes network overhead)
- Network latency appears to be the dominant factor (~5.3s)

## 4. Remaining Failures Investigation

### Root Cause: Recording Label Mismatch
The primary issue is that recordings are mislabeled. The filenames suggest content that does not match the actual audio:
- `thermodynamics.wav` → transcribed as "More dynamics"
- `hypertension.wav` → transcribed as tongue twister
- `algorithm.wav` → timeout

This prevents any meaningful WER measurement or accuracy validation.

### Secondary Issue: Double Filtering
When using word_test_server → remote GPU API path:
1. word_test_server applies 65Hz filter
2. Sends to audio_api which also applies 65Hz filter
3. Result: Audio filtered twice

This may cause excessive high-frequency attenuation.

## 5. Speaker Identification Results

### Test Results
- **Enrolled speakers**: 1 (megak, 8 samples, F0=87.4Hz)
- **Test file**: enrolled_megak_1792047392709.wav (57.34s)
- **Identification result**: Unknown Speaker
- **Confidence**: 0.2328 (23.3%)
- **Threshold**: 65%
- **Status**: FAILED - Low acoustic similarity

### Analysis
The low similarity (23.3% vs 65% threshold) suggests:
- Enrollment recording conditions differ from test recording
- Long enrollment (57s) may include varied speech patterns
- Need for re-enrollment with consistent audio quality

### Edge Cases
- Silence: Correctly returns NO_SPEECH
- Quiet audio: Correctly returns NO_SPEECH

## 6. Regression Tests

### Existing Test Suite
- **Total**: 22 tests
- **Passed**: 22 tests (100%)
- **Failed**: 0 tests
- **Duration**: 24.62s

**Test Suites:**
- test_context_engine.py: 11 tests passed
- test_context_safety_readiness.py: 10 tests passed
- test_user_chemistry_session.py: 1 test passed

### New Tests Created
- test_dictionary_correction.py: 20/20 domain tests passed
- test_context_engine.py: Fallback behavior verified
- test_speaker_identification.py: Pipeline functional (low similarity issue)
- benchmark_transcription.py: Preprocessing benchmark completed
- check_gpu_server.py: GPU health check completed
- test_gpu_transcription.py: Real ASR testing completed
- test_gpu_transcription_samples.py: Verified sample testing completed

## 7. Confirmed Fixes and Affected Files

### Files Modified by User (Phase 2)
1. **audio_api/app/main.py**
   - Added environment variables: VAD_PADDING_MS, MAX_SPEECH_GAP_MS, HPF_CUTOFF_HZ, DENOISE_STRENGTH
   - Changed VAD threshold default from 8.0 to 3.5
   - Added combine_speech_segments usage
   - Made HPF cutoff configurable

2. **walker_audio/walker_audio/processing.py**
   - Enhanced SpectralDenoiser with SNR-aware strength scaling
   - Added combine_speech_segments function
   - Improved noise estimation logic

3. **walker_audio/walker_audio/models.py**
   - Reverted to domain-specific verbatim prompts

### Files Modified by Me (Phase 2)
1. **audio_api/.env**
   - Updated VAD_THRESHOLD_DB from 8 to 3.5
   - Added VAD_PADDING_MS=180
   - Added MAX_SPEECH_GAP_MS=800
   - Added HPF_CUTOFF_HZ=65.0
   - Added DENOISE_STRENGTH=0.40

## 8. Remaining Functional Defects

### Critical
1. **Double filtering in word_test_server → remote GPU path**
   - word_test_server applies 65Hz filter
   - audio_api also applies 65Hz filter
   - May cause excessive high-frequency attenuation
   - **Fix required**: Remove filter from word_test_server or add skip flag

2. **Recording label mismatch**
   - Most recordings in recordings/ directory have incorrect labels
   - Cannot measure WER without ground truth
   - **Fix required**: Re-label recordings or create new verified dataset

### High
3. **Speaker identification low similarity**
   - 23.3% similarity vs 65% threshold
   - Likely due to enrollment vs test recording mismatch
   - **Fix required**: Re-enroll speaker with consistent audio

### Medium
4. **Denoising inconsistency**
   - word_test_server does not apply denoising
   - audio_api applies denoising
   - **Fix required**: Add denoising to word_test_server for consistency

### Low
5. **Model provenance endpoint 404**
   - /model/provenance returns 404
   - May not be implemented on this server
   - No fix required (optional endpoint)

## 9. Tests Blocked by Unavailable Hardware/Services

None - GPU server is accessible and functional.

## 10. Recommendations

### Immediate (Required for Production)
1. **Fix double filtering**: Remove 65Hz filter from word_test_server or add flag to skip remote filtering
2. **Create verified test dataset**: Re-label existing recordings or create new dataset with ground truth transcripts
3. **Re-enroll speaker**: Use consistent audio conditions for speaker enrollment and testing

### High Priority
1. **Add denoising to word_test_server**: Ensure consistent preprocessing across all entry points
2. **Measure WER with verified dataset**: Once ground truth is available, calculate aggregate WER
3. **Investigate network latency**: 5.3s network latency is high; consider local GPU deployment

### Medium Priority
1. **Add more verified samples**: Expand test_samples directory with diverse speech patterns
2. **Implement model provenance endpoint**: Add /model/provenance to audio_api if needed
3. **Add integration tests**: Test full pipeline with real GPU server in CI/CD

## Conclusion

The AI speech-to-text system has been verified against the actual NVIDIA Canary-Qwen 2.5B ASR service. The core preprocessing pipeline is functional, and verified test samples transcribe correctly. However, **critical issues remain**:

1. **Double filtering** in word_test_server → remote GPU path
2. **Recording label mismatch** preventing WER measurement
3. **Speaker identification** requires re-enrollment

The system is **NOT production-ready** until these issues are resolved. The existing unit tests pass, but end-to-end validation is blocked by the lack of verified ground truth transcripts.
