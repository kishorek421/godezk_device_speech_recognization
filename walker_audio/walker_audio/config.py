from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml


@dataclass(frozen=True)
class ArrayConfig:
    sample_rate: int = 16_000
    speed_of_sound: float = 343.0
    microphone_positions_m: tuple[tuple[float, float, float], ...] = (
        (-0.09, 0.0, 0.0),
        (-0.03, 0.0, 0.0),
        (0.03, 0.0, 0.0),
        (0.09, 0.0, 0.0),
    )
    scan_min_deg: float = -75.0
    scan_max_deg: float = 75.0
    scan_step_deg: float = 2.5

    @property
    def positions(self) -> np.ndarray:
        return np.asarray(self.microphone_positions_m, dtype=np.float64)


@dataclass(frozen=True)
class EnhancementConfig:
    enabled: bool = True
    strength: float = 0.40
    noise_seconds: float = 0.25


@dataclass(frozen=True)
class VadConfig:
    frame_ms: int = 30
    threshold_db: float = 3.5
    minimum_speech_ms: int = 180
    release_ms: int = 350
    pre_roll_ms: int = 180



@dataclass(frozen=True)
class TargetConfig:
    azimuth_tolerance_deg: float = 20.0
    speaker_threshold: float = 0.7
    require_speaker_match: bool = False


@dataclass(frozen=True)
class ModelConfig:
    canary_model: str = "nvidia/canary-qwen-2.5b"
    enhancement_model: str = "nvidia/se_den_sb_16k_small"
    speaker_model: str = "titanet_large"
    device: str = "cuda"
    dtype: str = "bfloat16"


@dataclass(frozen=True)
class WalkerAudioConfig:
    array: ArrayConfig = field(default_factory=ArrayConfig)
    enhancement: EnhancementConfig = field(default_factory=EnhancementConfig)
    vad: VadConfig = field(default_factory=VadConfig)
    target: TargetConfig = field(default_factory=TargetConfig)
    models: ModelConfig = field(default_factory=ModelConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "WalkerAudioConfig":
        data = yaml.safe_load(Path(path).read_text()) or {}
        array = data.get("array", {})
        if "microphone_positions_m" in array:
            array["microphone_positions_m"] = tuple(map(tuple, array["microphone_positions_m"]))
        return cls(
            array=ArrayConfig(**array),
            enhancement=EnhancementConfig(**data.get("enhancement", {})),
            vad=VadConfig(**data.get("vad", {})),
            target=TargetConfig(**data.get("target", {})),
            models=ModelConfig(**data.get("models", {})),
        )
