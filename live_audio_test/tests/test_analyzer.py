import numpy as np

from live_audio_test.analyzer import LiveAudioAnalyzer


class FakeTranscriber:
    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        return "live microphone works"


def test_analyzer_detects_and_transcribes_speech() -> None:
    rate = 16_000
    silence = np.zeros(rate // 2, dtype=np.float32)
    speech = 0.4 * np.sin(2 * np.pi * 220 * np.arange(rate) / rate).astype(np.float32)
    audio = np.concatenate([silence, speech, silence])
    result = LiveAudioAnalyzer(rate, FakeTranscriber(), denoise=False).analyze(audio)
    assert result.transcript == "live microphone works"
    assert result.speech_seconds >= 1.0
    assert result.reason is None


def test_analyzer_rejects_silence() -> None:
    result = LiveAudioAnalyzer(16_000, denoise=False).analyze(np.zeros(16_000, dtype=np.float32))
    assert result.transcript is None
    assert result.reason == "no_speech"


def test_analyzer_handles_empty_audio() -> None:
    result = LiveAudioAnalyzer(16_000, denoise=False).analyze(np.array([], dtype=np.float32))
    assert result.reason == "empty_audio"
