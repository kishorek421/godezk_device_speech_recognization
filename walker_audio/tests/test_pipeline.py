import numpy as np

from walker_audio.config import ArrayConfig, EnhancementConfig, WalkerAudioConfig
from walker_audio.pipeline import WalkerAudioPipeline


class FakeTranscriber:
    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        return "hello walker"


def test_pipeline_transcribes_target_direction() -> None:
    rate = 16_000
    config = WalkerAudioConfig(
        array=ArrayConfig(
            sample_rate=rate,
            microphone_positions_m=((0, 0, 0), (0, 0, 0)),
            scan_min_deg=-10,
            scan_max_deg=10,
            scan_step_deg=10,
        ),
        enhancement=EnhancementConfig(enabled=False),
    )
    silence = np.zeros(rate // 2, dtype=np.float32)
    speech = 0.4 * np.sin(2 * np.pi * 220 * np.arange(rate) / rate).astype(np.float32)
    mono = np.concatenate([silence, speech, silence])
    audio = np.column_stack([mono, mono])
    result = WalkerAudioPipeline(config, transcriber=FakeTranscriber()).process(audio, 0)
    assert result.accepted
    assert result.transcript == "hello walker"


def test_pipeline_rejects_wrong_direction() -> None:
    rate = 16_000
    config = WalkerAudioConfig(
        array=ArrayConfig(
            sample_rate=rate,
            microphone_positions_m=((0, 0, 0), (0, 0, 0)),
            scan_min_deg=40,
            scan_max_deg=40,
            scan_step_deg=5,
        )
    )
    audio = np.zeros((rate, 2), dtype=np.float32)
    result = WalkerAudioPipeline(config, transcriber=FakeTranscriber()).process(audio, -40)
    assert not result.accepted
    assert not result.segments
