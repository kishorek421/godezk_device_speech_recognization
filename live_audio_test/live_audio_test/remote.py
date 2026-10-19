import io
import os

import numpy as np
import requests
import soundfile as sf

from live_audio_test.analyzer import AnalysisResult

DEFAULT_SERVER_URL = os.getenv("REMOTE_GPU_API", "http://124.123.89.162:8001") + "/v1/transcribe"


def default_server_url() -> str:
    return os.getenv("AUDIO_API_URL", DEFAULT_SERVER_URL)


class RemoteAnalyzer:
    """Sends each recorded window to the Walker Audio Transcription API.

    The server performs denoising, VAD, and Canary-Qwen transcription, so this
    client needs no ML dependencies beyond soundfile.
    """

    def __init__(
        self, server_url: str, sample_rate: int = 16_000, timeout: float = 120.0
    ) -> None:
        self.server_url = server_url
        self.sample_rate = sample_rate
        self.timeout = timeout

    def analyze(self, audio: np.ndarray) -> AnalysisResult:
        mono = np.asarray(audio, dtype=np.float32).reshape(-1)
        buffer = io.BytesIO()
        sf.write(buffer, mono, self.sample_rate, format="WAV", subtype="FLOAT")
        try:
            response = requests.post(
                self.server_url,
                files={"audio": ("window.wav", buffer.getvalue(), "audio/wav")},
                timeout=self.timeout,
            )
        except requests.Timeout:
            return AnalysisResult(-160.0, -160.0, 0.0, None, "api_timeout")
        except requests.RequestException as exc:
            return AnalysisResult(-160.0, -160.0, 0.0, None, f"network_error: {exc}")
        if response.status_code != 200:
            detail = response.text[:200]
            return AnalysisResult(
                -160.0, -160.0, 0.0, None, f"http_{response.status_code}: {detail}"
            )
        data = response.json()
        return AnalysisResult(
            peak_db=float(data.get("peak_db") or -160.0),
            rms_db=float(data.get("rms_db") or -160.0),
            speech_seconds=float(data.get("speech_seconds") or 0.0),
            transcript=data.get("transcript"),
            reason=data.get("reason"),
        )
