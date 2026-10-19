import numpy as np

from app.main import Services, Settings


class FakeTranscriber:
    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        return "walker can you hear me"

    def transcribe_advanced(
        self, audio: np.ndarray, sample_rate: int, mode: str = "salm", domain: str = "general"
    ) -> dict[str, str]:
        text = self.transcribe(audio, sample_rate)
        return {"transcript": text, "raw_asr_text": text}


def make_service(denoise=False, threshold=8.0):
    service = object.__new__(Services)
    service.settings = Settings(denoise=denoise, vad_threshold_db=threshold)
    service.transcriber = FakeTranscriber()
    from walker_audio.processing import ButterworthHighPassFilter, EnergyVad, SpectralDenoiser

    service.high_pass = ButterworthHighPassFilter(180.0, service.settings.sample_rate)
    service.denoiser = SpectralDenoiser(service.settings.sample_rate)
    service.echo_canceller = None
    service.verifier = None
    service.vad = EnergyVad(
        service.settings.sample_rate, threshold_db=service.settings.vad_threshold_db
    )
    return service


def test_transcribes_speech_segment():
    rate = 16_000
    silence = np.zeros(rate // 2, dtype=np.float32)
    speech = 0.4 * np.sin(2 * np.pi * 220 * np.arange(rate) / rate).astype(np.float32)
    audio = np.concatenate([silence, speech, silence])

    result = make_service().transcribe(audio)

    assert result.transcript == "walker can you hear me"
    assert result.speech_seconds >= 1.0
    assert result.reason is None
    assert result.processing_ms is not None


def test_rejects_silence():
    result = make_service().transcribe(np.zeros(16_000, dtype=np.float32))

    assert result.transcript is None
    assert result.reason == "no_speech"


def test_handles_empty_audio():
    result = make_service().transcribe(np.array([], dtype=np.float32))

    assert result.reason == "empty_audio"
