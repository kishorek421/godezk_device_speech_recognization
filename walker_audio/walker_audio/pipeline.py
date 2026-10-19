from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Protocol

import numpy as np

from walker_audio.config import WalkerAudioConfig
from walker_audio.models import CanaryTranscriber, TitaNetVerifier
from walker_audio.processing import EnergyVad, NlmsEchoCanceller, SpectralDenoiser
from walker_audio.spatial import delay_and_sum, estimate_azimuth


class Transcriber(Protocol):
    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str: ...


class SpeakerVerifier(Protocol):
    def score_audio(self, audio: np.ndarray, sample_rate: int, temporary_path: Path) -> float: ...


@dataclass(frozen=True)
class SegmentResult:
    start_seconds: float
    end_seconds: float
    speaker_score: float | None
    transcript: str


@dataclass(frozen=True)
class PipelineResult:
    detected_azimuth_deg: float
    target_azimuth_deg: float
    accepted: bool
    segments: tuple[SegmentResult, ...]

    @property
    def transcript(self) -> str:
        return " ".join(segment.transcript for segment in self.segments)


class WalkerAudioPipeline:
    def __init__(
        self,
        config: WalkerAudioConfig,
        transcriber: Transcriber | None = None,
        speaker_verifier: SpeakerVerifier | None = None,
    ):
        self.config = config
        rate = config.array.sample_rate
        self.transcriber = transcriber or CanaryTranscriber(
            config.models.canary_model, config.models.device, config.models.dtype
        )
        self.speaker_verifier = speaker_verifier
        self.echo_canceller = NlmsEchoCanceller()
        self.denoiser = SpectralDenoiser(
            rate, config.enhancement.strength, config.enhancement.noise_seconds
        )
        self.vad = EnergyVad(
            rate,
            config.vad.frame_ms,
            config.vad.threshold_db,
            config.vad.minimum_speech_ms,
            config.vad.release_ms,
            pre_roll_ms=config.vad.pre_roll_ms,
        )


    @classmethod
    def with_titanet(cls, config: WalkerAudioConfig) -> "WalkerAudioPipeline":
        verifier = TitaNetVerifier(config.models.speaker_model, config.models.device)
        return cls(config, speaker_verifier=verifier)

    def enroll_speaker(self, audio_path: str | Path) -> None:
        if not isinstance(self.speaker_verifier, TitaNetVerifier):
            raise TypeError("the configured verifier does not support enrollment")
        self.speaker_verifier.enroll(audio_path)

    def process(
        self,
        multichannel_audio: np.ndarray,
        target_azimuth_deg: float | None = None,
        playback_reference: np.ndarray | None = None,
    ) -> PipelineResult:
        config = self.config
        rate = config.array.sample_rate
        audio = np.asarray(multichannel_audio, dtype=np.float32)
        scan = np.arange(
            config.array.scan_min_deg,
            config.array.scan_max_deg + config.array.scan_step_deg / 2,
            config.array.scan_step_deg,
        )
        detected, _ = estimate_azimuth(
            audio, rate, config.array.positions, scan, config.array.speed_of_sound
        )
        target = detected if target_azimuth_deg is None else target_azimuth_deg
        if abs(detected - target) > config.target.azimuth_tolerance_deg:
            return PipelineResult(detected, target, False, ())
        clean = delay_and_sum(
            audio, rate, config.array.positions, target, config.array.speed_of_sound
        )
        if playback_reference is not None:
            clean = self.echo_canceller(clean, playback_reference)
        if config.enhancement.enabled:
            clean = self.denoiser(clean)
        segments = self.vad.split(clean)
        results: list[SegmentResult] = []
        with TemporaryDirectory(prefix="walker-audio-") as directory:
            for index, segment in enumerate(segments):
                score = self._speaker_score(segment.audio, rate, Path(directory) / f"{index}.wav")
                if score is not None and score < config.target.speaker_threshold:
                    continue
                transcript = self.transcriber.transcribe(segment.audio, rate)
                results.append(
                    SegmentResult(
                        segment.start_sample / rate,
                        segment.end_sample / rate,
                        score,
                        transcript,
                    )
                )
        accepted = bool(results) or (not segments and not config.target.require_speaker_match)
        return PipelineResult(detected, target, accepted, tuple(results))

    def _speaker_score(self, audio: np.ndarray, rate: int, path: Path) -> float | None:
        if self.speaker_verifier is None:
            if self.config.target.require_speaker_match:
                raise RuntimeError("speaker matching is required but no verifier is configured")
            return None
        return self.speaker_verifier.score_audio(audio, rate, path)
