"""
Diagnostic tool to detect electrical noise, ground hum, and coil whine
directly from the user's laptop microphone.
"""

import sys
import time
from pathlib import Path
import numpy as np
import scipy.signal as sp
import sounddevice as sd
import soundfile as sf

DURATION = 4.0  # seconds
SR = 16000

print("=" * 70)
print("LAPTOP MICROPHONE ELECTRICAL NOISE & HARDWARE DIAGNOSTIC")
print("=" * 70)
print(f"Target recording duration: {DURATION}s at {SR} Hz mono")
print("PLEASE REMAIN SILENT for 4 seconds while we measure ambient noise...")

try:
    recording = sd.rec(int(DURATION * SR), samplerate=SR, channels=1, dtype='float32')
    for i in range(int(DURATION), 0, -1):
        print(f"Recording... {i}s remaining", flush=True)
        time.sleep(1.0)
    sd.wait()
    print("Recording complete! Analyzing spectrum...")
except Exception as e:
    print(f"Error accessing sound device: {e}")
    sys.exit(1)

audio = recording.flatten()

# Save raw recording
raw_path = Path("laptop_electric_noise_raw.wav")
sf.write(raw_path, audio, SR, format="WAV", subtype="PCM_16")

# Also copy to recordings folder for browser playback
recordings_dir = Path("recordings")
recordings_dir.mkdir(exist_ok=True)
sf.write(recordings_dir / "laptop_electric_noise_raw.wav", audio, SR, format="WAV", subtype="PCM_16")

# 1. Overall Metrics
raw_peak = float(np.max(np.abs(audio)))
raw_rms = float(np.sqrt(np.mean(audio**2)))
raw_peak_db = 20 * np.log10(raw_peak + 1e-9)
raw_rms_db = 20 * np.log10(raw_rms + 1e-9)

# 2. FFT Spectral Analysis
fft_vals = np.abs(np.fft.rfft(audio))
freqs = np.fft.rfftfreq(len(audio), 1.0 / SR)

# Frequencies of interest
def band_power(f_min, f_max):
    mask = (freqs >= f_min) & (freqs <= f_max)
    if not np.any(mask):
        return 0.0, 0.0
    idx_max = np.argmax(fft_vals[mask])
    peak_freq = freqs[mask][idx_max]
    power_db = 20 * np.log10(np.mean(fft_vals[mask]) + 1e-9)
    return peak_freq, power_db

peak_sub, db_sub = band_power(1, 40)       # Sub-audible / DC rumble
peak_50hz, db_50hz = band_power(45, 55)    # 50 Hz mains hum (India / UK / Europe)
peak_60hz, db_60hz = band_power(55, 65)    # 60 Hz mains hum (US)
peak_100hz, db_100hz = band_power(95, 105) # 100 Hz 2nd harmonic
peak_120hz, db_120hz = band_power(115, 125)# 120 Hz 2nd harmonic
peak_coil, db_coil = band_power(1000, 8000)# High-frequency coil whine / switching

# 3. Apply DSP Filter (180 Hz Butterworth HPF + Spectral Denoiser)
# 180 Hz HPF
sos = sp.butter(5, 180, 'hp', fs=SR, output='sos')
filtered_audio = sp.sosfilt(sos, audio).astype(np.float32)

# Spectral suppression
if len(filtered_audio) > 256:
    nperseg = min(512, len(filtered_audio))
    _, _, stft_mat = sp.stft(filtered_audio, fs=SR, nperseg=nperseg)
    powers = np.sum(np.abs(stft_mat) ** 2, axis=0)
    quiet_idx = np.argsort(powers)[: max(1, len(powers) // 5)]
    noise_profile = np.median(np.abs(stft_mat[:, quiet_idx]), axis=1, keepdims=True)
    mag = np.abs(stft_mat)
    gain = np.maximum(1.0 - 0.75 * (noise_profile / np.maximum(mag, 1e-8)), 0.1)
    _, cleaned = sp.istft(stft_mat * gain, fs=SR, nperseg=nperseg)
    filtered_audio = cleaned[: len(filtered_audio)].astype(np.float32)

filt_peak = float(np.max(np.abs(filtered_audio)))
filt_rms = float(np.sqrt(np.mean(filtered_audio**2)))
filt_peak_db = 20 * np.log10(filt_peak + 1e-9)
filt_rms_db = 20 * np.log10(filt_rms + 1e-9)

noise_reduction_db = raw_rms_db - filt_rms_db

# Save filtered version
filt_path = Path("laptop_electric_noise_filtered.wav")
sf.write(filt_path, filtered_audio, SR, format="WAV", subtype="PCM_16")
sf.write(recordings_dir / "laptop_electric_noise_filtered.wav", filtered_audio, SR, format="WAV", subtype="PCM_16")

print("\n" + "=" * 70)
print("DIAGNOSTIC FINDINGS:")
print("=" * 70)
print(f"1. Total Microphone Noise Floor:")
print(f"   - Raw Peak Volume: {raw_peak_db:.1f} dBFS")
print(f"   - Raw RMS Power:   {raw_rms_db:.1f} dBFS")

print(f"\n2. Detected Electrical / Acoustic Signatures:")
print(f"   - 50 Hz Mains Power Hum:       {db_50hz:5.1f} dB  (Peak near {peak_50hz:.1f} Hz)")
print(f"   - 100 Hz Rectified Harmonic:   {db_100hz:5.1f} dB  (Peak near {peak_100hz:.1f} Hz)")
print(f"   - 60 Hz / 120 Hz Mains Band:   {db_60hz:5.1f} dB / {db_120hz:5.1f} dB")
print(f"   - Sub-bass Chassis/Fan Rumble: {db_sub:5.1f} dB  (Peak near {peak_sub:.1f} Hz)")
print(f"   - High-Freq Switching Noise:   {db_coil:5.1f} dB  (Peak near {peak_coil:.1f} Hz)")

# Diagnosis
diagnoses = []
if db_50hz > -65 or db_100hz > -65:
    diagnoses.append("50 Hz / 100 Hz AC Mains Hum detected (Laptop charger ground loop or ungrounded AC outlet)")
if db_sub > -50:
    diagnoses.append("Sub-audible chassis vibration / fan motor rumble detected")
if db_coil > -60:
    diagnoses.append("High-frequency electromagnetic interference / coil whine / switching power supply ripple")

if not diagnoses:
    diagnoses.append("Microphone floor is exceptionally quiet (no severe electrical fault detected)")

print("\n3. Hardware Diagnosis:")
for d in diagnoses:
    print(f"   [!] {d}")

print(f"\n4. DSP Filter Performance:")
print(f"   - Filtered RMS Power: {filt_rms_db:.1f} dBFS")
print(f"   - Electrical Noise Cut: {noise_reduction_db:.1f} dB of noise suppressed!")

print("\n5. Audible Proof Files Generated:")
print(f"   - Raw laptop sound:     http://127.0.0.1:8050/api/audio/laptop_electric_noise_raw.wav")
print(f"   - Clean filtered sound: http://127.0.0.1:8050/api/audio/laptop_electric_noise_filtered.wav")
print("=" * 70)
