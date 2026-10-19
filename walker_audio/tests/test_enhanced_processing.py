import numpy as np
import pytest
from walker_audio.processing import (
    ButterworthHighPassFilter,
    SpectralDenoiser,
    EnergyVad,
    combine_speech_segments,
    preprocess_speech_audio,
    SpeechSegment,
)
from walker_audio.dictionary_corrector import correct_transcript


def test_hpf_preserves_male_pitch_and_attenuates_hum():
    sr = 16000
    t = np.arange(sr) / float(sr)
    
    # 50 Hz electrical hum
    hum_50hz = np.sin(2 * np.pi * 50 * t).astype(np.float32)
    # 100 Hz male fundamental pitch
    pitch_100hz = np.sin(2 * np.pi * 100 * t).astype(np.float32)
    
    hpf = ButterworthHighPassFilter(cutoff_hz=65.0, sample_rate=sr, order=4)
    filtered_hum = hpf(hum_50hz)
    filtered_pitch = hpf(pitch_100hz)
    
    hum_attenuation = float(np.mean(filtered_hum[2000:]**2) / np.mean(hum_50hz[2000:]**2))
    pitch_transmission = float(np.mean(filtered_pitch[2000:]**2) / np.mean(pitch_100hz[2000:]**2))
    
    # 50 Hz must be attenuated significantly (< 0.25 of power, > 6 dB down)
    assert hum_attenuation < 0.25
    # 100 Hz vocal pitch must be preserved (> 0.70 of power, < 1.5 dB loss)
    assert pitch_transmission > 0.70


def test_spectral_denoiser_does_not_mutilate_speech_at_onset():
    sr = 16000
    t = np.arange(sr * 2) / float(sr)
    # Speech starting immediately at t=0
    immediate_speech = (0.5 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    
    denoiser = SpectralDenoiser(sample_rate=sr, strength=0.40)
    cleaned = denoiser(immediate_speech)
    
    # First 0.25s must not be wiped out or silenced
    first_quarter_sec = cleaned[:int(sr * 0.25)]
    assert np.max(np.abs(first_quarter_sec)) > 0.20


def test_combine_speech_segments_bridges_intra_sentence_pauses():
    sr = 16000
    audio = np.ones(sr * 4, dtype=np.float32) * 0.5
    # Two segments separated by 400ms (intra-sentence pause)
    seg1 = SpeechSegment(0, int(sr * 1.0), audio[0:int(sr * 1.0)])
    seg2 = SpeechSegment(int(sr * 1.4), int(sr * 2.5), audio[int(sr * 1.4):int(sr * 2.5)])
    
    combined = combine_speech_segments(audio, [seg1, seg2], sample_rate=sr, max_gap_ms=800)
    # The pause should be preserved continuously: 0 to 2.5s = 2.5s
    assert len(combined) == int(sr * 2.5)


def test_preprocess_speech_audio_end_to_end():
    sr = 16000
    t = np.arange(sr * 3) / float(sr)
    speech = (0.4 * np.sin(2 * np.pi * 250 * t)).astype(np.float32)
    # Add quiet prefix and suffix
    full_audio = np.concatenate([np.zeros(sr//2, dtype=np.float32), speech, np.zeros(sr//2, dtype=np.float32)])
    
    processed, meta = preprocess_speech_audio(full_audio, sample_rate=sr)
    assert meta["status"] == "ok"
    assert meta["speech_sec"] > 2.0
    assert abs(np.percentile(np.abs(processed), 99.8) - 0.85) < 0.05
    assert np.max(np.abs(processed)) <= 0.98



def test_technical_abbreviations_normalized_correctly():
    # Chemistry abbreviations
    res_chem = correct_transcript("add dilute h c l to n a o h until the p h reaches 7", domain="education")
    assert "HCl" in res_chem["corrected_transcript"]
    assert "NaOH" in res_chem["corrected_transcript"]
    assert "pH" in res_chem["corrected_transcript"]
    
    # Hospital abbreviations
    res_hosp = correct_transcript("check b p and s p o 2 before medication", domain="hospital")
    assert "BP" in res_hosp["corrected_transcript"]
    assert "SpO2" in res_hosp["corrected_transcript"]
    
    # Industry abbreviations
    res_ind = correct_transcript("emergency stop on p l c conveyor", domain="industry")
    assert "PLC" in res_ind["corrected_transcript"]
