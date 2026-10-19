import numpy as np


def direction_vector(azimuth_deg: float) -> np.ndarray:
    angle = np.deg2rad(azimuth_deg)
    return np.asarray([np.sin(angle), np.cos(angle), 0.0])


def channel_delays_seconds(
    positions_m: np.ndarray, azimuth_deg: float, speed_of_sound: float = 343.0
) -> np.ndarray:
    delays = positions_m @ direction_vector(azimuth_deg) / speed_of_sound
    return delays - delays.mean()


def delay_and_sum(
    audio: np.ndarray,
    sample_rate: int,
    positions_m: np.ndarray,
    azimuth_deg: float,
    speed_of_sound: float = 343.0,
) -> np.ndarray:
    if audio.ndim != 2 or audio.shape[1] != len(positions_m):
        raise ValueError("audio must have shape (samples, configured microphones)")
    samples = np.arange(len(audio), dtype=np.float64)
    delays = channel_delays_seconds(positions_m, azimuth_deg, speed_of_sound) * sample_rate
    aligned = [
        np.interp(samples + delay, samples, audio[:, channel], left=0.0, right=0.0)
        for channel, delay in enumerate(delays)
    ]
    return np.mean(aligned, axis=0).astype(np.float32)


def estimate_azimuth(
    audio: np.ndarray,
    sample_rate: int,
    positions_m: np.ndarray,
    scan_angles_deg: np.ndarray,
    speed_of_sound: float = 343.0,
) -> tuple[float, np.ndarray]:
    if len(audio) == 0:
        raise ValueError("audio cannot be empty")
    scores = np.asarray(
        [
            np.mean(delay_and_sum(audio, sample_rate, positions_m, angle, speed_of_sound) ** 2)
            for angle in scan_angles_deg
        ]
    )
    return float(scan_angles_deg[int(np.argmax(scores))]), scores


def camera_x_to_azimuth(x: float, frame_width: int, horizontal_fov_deg: float) -> float:
    if frame_width <= 0 or not 0 <= x <= frame_width:
        raise ValueError("face x-coordinate must be inside a non-empty frame")
    return (x / frame_width - 0.5) * horizontal_fov_deg
