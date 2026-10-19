import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import soundfile as sf

from walker_audio.config import WalkerAudioConfig
from walker_audio.pipeline import WalkerAudioPipeline
from walker_audio.spatial import camera_x_to_azimuth


def load_audio(path: Path, sample_rate: int) -> np.ndarray:
    audio, rate = sf.read(path, always_2d=True, dtype="float32")
    if rate != sample_rate:
        raise ValueError(f"expected {sample_rate} Hz audio, received {rate} Hz")
    return audio


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="walker-audio")
    parser.add_argument("audio", type=Path)
    parser.add_argument("--config", type=Path, default=Path("config/walker.yaml"))
    parser.add_argument("--target-azimuth", type=float)
    parser.add_argument("--face-x", type=float)
    parser.add_argument("--frame-width", type=int)
    parser.add_argument("--camera-fov", type=float, default=78.0)
    parser.add_argument("--speaker-enrollment", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = WalkerAudioConfig.from_yaml(args.config)
    pipeline = (
        WalkerAudioPipeline.with_titanet(config)
        if args.speaker_enrollment
        else WalkerAudioPipeline(config)
    )
    if args.speaker_enrollment:
        pipeline.enroll_speaker(args.speaker_enrollment)
    target = args.target_azimuth
    if args.face_x is not None:
        if args.frame_width is None:
            raise ValueError("--frame-width is required with --face-x")
        target = camera_x_to_azimuth(args.face_x, args.frame_width, args.camera_fov)
    audio = load_audio(args.audio, config.array.sample_rate)
    result = pipeline.process(audio, target)
    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()
