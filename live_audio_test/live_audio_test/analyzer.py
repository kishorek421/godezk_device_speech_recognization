from dataclasses import dataclass
from typing import Protocol

import numpy as np
from walker_audio.processing import EnergyVad, SpectralDenoiser


class Transcriber(Protocol):
    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str: ...


@dataclass(frozen=True)
class AnalysisResult:
    peak_db: float
    rms_db: float
    speech_seconds: float
    transcript: str | None
    reason: str | None


class LiveAudioAnalyzer:
    def __init__(
        self,
        sample_rate: int = 16_000,
        transcriber: Transcriber | None = None,
        denoise: bool = True,
        vad_threshold_db: float = 8.0,
    ):
        self.sample_rate = sample_rate
        self.transcriber = transcriber
        self.denoise = denoise
        self.denoiser = SpectralDenoiser(sample_rate)
        self.vad = EnergyVad(sample_rate, threshold_db=vad_threshold_db)

    def analyze(self, audio: np.ndarray) -> AnalysisResult:
        mono = np.asarray(audio, dtype=np.float32).reshape(-1)
        if not len(mono):
            return AnalysisResult(-160.0, -160.0, 0.0, None, "empty_audio")
        peak_db = float(20 * np.log10(np.max(np.abs(mono)) + 1e-8))
        rms_db = float(20 * np.log10(np.sqrt(np.mean(mono**2)) + 1e-8))
        processed = self.denoiser(mono) if self.denoise else mono
        segments = self.vad.split(processed)
        if not segments:
            return AnalysisResult(peak_db, rms_db, 0.0, None, "no_speech")
        speech = np.concatenate([segment.audio for segment in segments])
        transcript = (
            self.transcriber.transcribe(speech, self.sample_rate) if self.transcriber else None
        )
        return AnalysisResult(
            peak_db,
            rms_db,
            len(speech) / self.sample_rate,
            transcript,
            None if self.transcriber else "asr_disabled",
        )
