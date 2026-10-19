import numpy as np

from walker_audio.processing import EnergyVad, NlmsEchoCanceller


def test_vad_returns_speech_region() -> None:
    rate = 16_000
    silence = np.zeros(rate // 2, dtype=np.float32)
    speech = 0.4 * np.sin(2 * np.pi * 220 * np.arange(rate) / rate).astype(np.float32)
    segments = EnergyVad(rate).split(np.concatenate([silence, speech, silence]))
    assert len(segments) == 1
    assert segments[0].start_sample >= rate * 0.45
    assert len(segments[0].audio) >= rate


def test_echo_canceller_reduces_correlated_playback() -> None:
    rng = np.random.default_rng(7)
    reference = rng.normal(0, 0.1, 4000).astype(np.float32)
    microphone = np.concatenate([np.zeros(4), reference[:-4]])
    clean = NlmsEchoCanceller(filter_length=32, step_size=0.4)(microphone, reference)
    assert np.mean(clean[-1000:] ** 2) < np.mean(microphone[-1000:] ** 2) * 0.2
