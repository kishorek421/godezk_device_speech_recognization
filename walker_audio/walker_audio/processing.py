from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from scipy.signal import butter, istft, sosfilt, stft



class ButterworthHighPassFilter:
    """Removes electrical hum (50/60 Hz) and low-frequency rumble while preserving speech fundamentals (85Hz+)."""

    def __init__(self, cutoff_hz: float = 65.0, sample_rate: int = 16000, order: int = 5):
        self.cutoff_hz = cutoff_hz
        self.sample_rate = sample_rate
        self.order = order
        # Ensure cutoff is below Nyquist
        nyquist = sample_rate / 2.0
        cutoff = min(cutoff_hz, nyquist - 10.0)
        self.sos = butter(order, cutoff, "hp", fs=sample_rate, output="sos")

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        if len(audio) < 16:
            return audio.astype(np.float32)
        filtered = sosfilt(self.sos, audio)
        return filtered.astype(np.float32)


def normalize_audio(audio: np.ndarray, target_peak: float = 0.85) -> np.ndarray:
    """Normalizes clean speech to avoid clipping while ensuring optimal signal level."""
    if len(audio) == 0:
        return audio.astype(np.float32)
    peak = float(np.percentile(np.abs(audio), 99.8))
    if peak > 1e-4:
        scaled = audio * (target_peak / peak)
        return np.clip(scaled, -0.98, 0.98).astype(np.float32)
    return audio.astype(np.float32)


class NlmsEchoCanceller:
    def __init__(self, filter_length: int = 1024, step_size: float = 0.2):
        self.filter_length = filter_length
        self.step_size = step_size
        self.weights = np.zeros(filter_length, dtype=np.float64)
        self.history = np.zeros(filter_length, dtype=np.float64)

    def __call__(self, microphone: np.ndarray, playback_reference: np.ndarray) -> np.ndarray:
        if len(microphone) != len(playback_reference):
            raise ValueError("microphone and playback reference must have equal lengths")
        output = np.empty(len(microphone), dtype=np.float32)
        for index, (sample, reference) in enumerate(
            zip(microphone, playback_reference, strict=True)
        ):
            self.history[1:] = self.history[:-1]
            self.history[0] = reference
            estimate = float(self.weights @ self.history)
            error = float(sample - estimate)
            power = float(self.history @ self.history) + 1e-8
            self.weights += self.step_size * error * self.history / power
            output[index] = error
        return output


class SpectralDenoiser:
    """Spectral subtraction denoiser with dynamic quiet-frame noise estimation."""

    def __init__(self, sample_rate: int, strength: float = 0.40, noise_seconds: float = 0.25):
        self.sample_rate = sample_rate
        self.strength = strength
        self.noise_seconds = noise_seconds

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        if len(audio) < 2:
            return audio.astype(np.float32)
        nperseg = min(512, len(audio))
        _, _, spectrum = stft(audio, fs=self.sample_rate, nperseg=nperseg)
        
        noise_frames = max(2, int(self.noise_seconds * self.sample_rate / (nperseg // 2)))
        frame_energies = np.mean(np.abs(spectrum)**2, axis=0)
        
        # Guard against zero or near-silent input
        if len(frame_energies) == 0 or np.max(frame_energies) < 1e-9:
            return audio.astype(np.float32)

        # Select frames with lowest energy (likely ambient silence / inter-word pauses)
        # Avoid assuming speech-at-start is noise
        if len(frame_energies) > noise_frames:
            quiet_count = min(noise_frames, max(2, int(0.20 * len(frame_energies))))
            lowest_indices = np.argpartition(frame_energies, quiet_count)[:quiet_count]
            noise = np.median(np.abs(spectrum[:, lowest_indices]), axis=1, keepdims=True)
            
            # Check SNR: If peak energy is very high relative to quiet floor, gentle attenuation
            peak_energy = float(np.percentile(frame_energies, 95))
            quiet_energy = float(np.median(frame_energies[lowest_indices])) + 1e-9
            snr_est_db = 10 * np.log10(peak_energy / quiet_energy)
            
            # Clean recording (SNR > 25 dB): scale down subtraction strength to preserve pristine acoustics
            eff_strength = self.strength
            if snr_est_db > 25.0:
                eff_strength = max(0.15, self.strength * 0.5)
            elif snr_est_db < 3.0:
                # Almost pure noise or unvoiced: do not over-subtract
                eff_strength = min(0.30, self.strength)
        else:
            eff_strength = min(0.30, self.strength)
            noise = np.median(np.abs(spectrum[:, :noise_frames]), axis=1, keepdims=True)
        
        magnitude = np.abs(spectrum)
        gain = np.maximum(1.0 - eff_strength * noise / np.maximum(magnitude, 1e-8), 0.12)
        _, clean = istft(spectrum * gain, fs=self.sample_rate, nperseg=nperseg)
        return clean[: len(audio)].astype(np.float32)


@dataclass(frozen=True)
class SpeechSegment:
    start_sample: int
    end_sample: int
    audio: np.ndarray


class EnergyVad:
    """Frame-energy Voice Activity Detector with configurable threshold and padding."""

    def __init__(
        self,
        sample_rate: int,
        frame_ms: int = 30,
        threshold_db: float = 3.5,
        minimum_speech_ms: int = 180,
        release_ms: int = 350,
        pre_roll_ms: int = 0,
    ):

        self.sample_rate = sample_rate
        self.frame_samples = sample_rate * frame_ms // 1000
        self.threshold_db = threshold_db
        self.minimum_frames = max(1, minimum_speech_ms // frame_ms)
        self.release_frames = max(1, release_ms // frame_ms)
        self.pre_roll_frames = max(0, pre_roll_ms // frame_ms)

    def split(self, audio: np.ndarray) -> list[SpeechSegment]:
        frame_count = len(audio) // self.frame_samples
        if frame_count == 0:
            return []
        frames = audio[: frame_count * self.frame_samples].reshape(frame_count, self.frame_samples)
        levels = 20 * np.log10(np.sqrt(np.mean(frames**2, axis=1)) + 1e-8)
        noise_floor = float(np.percentile(levels, 20))
        active = levels >= noise_floor + self.threshold_db
        segments: list[SpeechSegment] = []
        start = None
        silence = 0
        for index, is_active in enumerate(active):
            if is_active:
                start = index if start is None else start
                silence = 0
            elif start is not None:
                silence += 1
                if silence >= self.release_frames:
                    self._append(segments, audio, start, index - silence + 1)
                    start = None
                    silence = 0
        if start is not None:
            self._append(segments, audio, start, frame_count)
        return segments

    def _append(
        self, segments: list[SpeechSegment], audio: np.ndarray, start_frame: int, end_frame: int
    ) -> None:
        if end_frame - start_frame < self.minimum_frames:
            return
        # Add pre-roll padding to preserve consonant onsets and word beginnings
        start = max(0, (start_frame - self.pre_roll_frames) * self.frame_samples)
        end = min(len(audio), (end_frame + self.release_frames) * self.frame_samples)
        segments.append(SpeechSegment(start, end, audio[start:end]))


def combine_speech_segments(
    audio: np.ndarray,
    segments: list[SpeechSegment],
    sample_rate: int = 16000,
    max_gap_ms: int = 800,
    cushion_ms: int = 200,
) -> np.ndarray:
    """Combines VAD speech segments while preserving natural intra-sentence pauses.
    
    Prevents harsh 0 ms phase discontinuities by retaining inter-segment audio for
    gaps <= max_gap_ms, and inserting gentle silence cushions for long pauses.
    """
    if not segments:
        return np.array([], dtype=np.float32)
    if len(segments) == 1:
        return segments[0].audio.astype(np.float32)

    max_gap_samples = int(sample_rate * max_gap_ms / 1000)
    cushion_samples = int(sample_rate * cushion_ms / 1000)
    cushion = np.zeros(cushion_samples, dtype=np.float32)

    sorted_segs = sorted(segments, key=lambda s: s.start_sample)
    out_parts: list[np.ndarray] = []
    prev_end = sorted_segs[0].end_sample
    curr_start = sorted_segs[0].start_sample

    for seg in sorted_segs[1:]:
        gap = seg.start_sample - prev_end
        if 0 <= gap <= max_gap_samples:
            # Intra-sentence pause: bridge continuously without cutting speech prosody
            prev_end = max(prev_end, seg.end_sample)
        else:
            # Long pause: record previous continuous block and insert a silence cushion
            chunk = audio[curr_start:prev_end]
            out_parts.append(chunk)
            out_parts.append(cushion)
            curr_start = seg.start_sample
            prev_end = seg.end_sample

    out_parts.append(audio[curr_start:prev_end])
    return np.concatenate(out_parts).astype(np.float32)


def preprocess_speech_audio(
    audio: np.ndarray,
    sample_rate: int = 16000,
    target_sample_rate: int = 16000,
    hpf_cutoff_hz: float = 65.0,
    denoise: bool = True,
    denoise_strength: float = 0.40,
    target_peak: float = 0.85,
    vad_threshold_db: float = 3.5,
    vad_padding_ms: int = 180,
    max_gap_ms: int = 800,
    apply_vad_trim: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Shared, reproducible speech preprocessing pipeline for live mic and offline audio."""
    # 1. Mono conversion
    mono = np.asarray(audio, dtype=np.float32)
    if mono.ndim > 1:
        mono = mono.mean(axis=1)
    mono = mono.reshape(-1)

    if len(mono) == 0:
        return mono, {"status": "empty_audio", "duration_sec": 0.0}

    # 2. Resample to target rate if needed
    if sample_rate != target_sample_rate:
        from scipy.signal import resample_poly
        divisor = math.gcd(sample_rate, target_sample_rate)
        mono = resample_poly(mono, target_sample_rate // divisor, sample_rate // divisor).astype(np.float32)
        sample_rate = target_sample_rate

    raw_peak = float(np.max(np.abs(mono))) if len(mono) > 0 else 0.0
    raw_peak_db = float(20 * np.log10(raw_peak + 1e-8))
    duration_sec = len(mono) / sample_rate

    # 3. 65 Hz High-pass filter (removes 50/60Hz hum while preserving vocal pitch)
    if hpf_cutoff_hz > 0:
        hpf = ButterworthHighPassFilter(hpf_cutoff_hz, sample_rate, order=4)
        mono = hpf(mono)

    # 4. Speech-safe Spectral Denoising
    if denoise and denoise_strength > 0:
        denoiser = SpectralDenoiser(sample_rate, strength=denoise_strength)
        mono = denoiser(mono)

    # 5. Peak Normalization
    mono = normalize_audio(mono, target_peak=target_peak)

    # 6. Speech-Aware VAD and Boundary Cushioning
    if apply_vad_trim:
        vad = EnergyVad(sample_rate, threshold_db=vad_threshold_db, pre_roll_ms=vad_padding_ms)
        segments = vad.split(mono)
        if not segments:
            return mono, {
                "status": "no_speech",
                "raw_peak_db": round(raw_peak_db, 1),
                "duration_sec": round(duration_sec, 2),
                "speech_sec": 0.0,
            }
        speech = combine_speech_segments(mono, segments, sample_rate, max_gap_ms=max_gap_ms, cushion_ms=200)
    else:
        speech = mono

    speech_sec = len(speech) / sample_rate
    meta = {
        "status": "ok",
        "raw_peak_db": round(raw_peak_db, 1),
        "duration_sec": round(duration_sec, 2),
        "speech_sec": round(speech_sec, 2),
        "hpf_cutoff_hz": hpf_cutoff_hz,
        "denoise_strength": denoise_strength,
    }
    return speech, meta

