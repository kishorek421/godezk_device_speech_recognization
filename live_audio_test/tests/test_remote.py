import io

import numpy as np
import pytest
import requests
import soundfile as sf

from live_audio_test.remote import RemoteAnalyzer


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


def test_remote_analyzer_posts_wav_and_maps_response(monkeypatch) -> None:
    captured = {}

    def fake_post(url, files, timeout):
        captured["url"] = url
        captured["timeout"] = timeout
        filename, payload, content_type = files["audio"]
        captured["filename"] = filename
        captured["content_type"] = content_type
        audio, rate = sf.read(io.BytesIO(payload), dtype="float32")
        captured["rate"] = rate
        captured["length"] = len(audio)
        return FakeResponse(
            payload={
                "transcript": "walker can you hear me",
                "speech_seconds": 1.2,
                "peak_db": -5.0,
                "rms_db": -20.0,
            }
        )

    monkeypatch.setattr(requests, "post", fake_post)
    analyzer = RemoteAnalyzer("http://server:8001/v1/transcribe", timeout=30)

    result = analyzer.analyze(np.zeros(8_000, dtype=np.float32))

    assert captured["url"] == "http://server:8001/v1/transcribe"
    assert captured["timeout"] == 30
    assert captured["content_type"] == "audio/wav"
    assert captured["rate"] == 16_000
    assert captured["length"] == 8_000
    assert result.transcript == "walker can you hear me"
    assert result.speech_seconds == 1.2
    assert result.peak_db == -5.0
    assert result.reason is None


def test_remote_analyzer_maps_no_speech(monkeypatch) -> None:
    monkeypatch.setattr(
        requests,
        "post",
        lambda *args, **kwargs: FakeResponse(
            payload={"reason": "no_speech", "peak_db": -60.0, "rms_db": -70.0}
        ),
    )
    result = RemoteAnalyzer("http://server/v1/transcribe").analyze(np.zeros(100, np.float32))
    assert result.transcript is None
    assert result.reason == "no_speech"
    assert result.rms_db == -70.0


def test_remote_analyzer_handles_http_error(monkeypatch) -> None:
    monkeypatch.setattr(
        requests,
        "post",
        lambda *args, **kwargs: FakeResponse(status_code=413, text="Audio is too large"),
    )
    result = RemoteAnalyzer("http://server/v1/transcribe").analyze(np.zeros(100, np.float32))
    assert result.transcript is None
    assert result.reason.startswith("http_413")


def test_remote_analyzer_handles_timeout(monkeypatch) -> None:
    def raise_timeout(*args, **kwargs):
        raise requests.Timeout()

    monkeypatch.setattr(requests, "post", raise_timeout)
    result = RemoteAnalyzer("http://server/v1/transcribe").analyze(np.zeros(100, np.float32))
    assert result.reason == "api_timeout"


def test_remote_analyzer_handles_network_error(monkeypatch) -> None:
    def raise_error(*args, **kwargs):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(requests, "post", raise_error)
    result = RemoteAnalyzer("http://server/v1/transcribe").analyze(np.zeros(100, np.float32))
    assert result.transcript is None
    assert "network_error" in result.reason


@pytest.mark.parametrize("rate", [8_000, 44_100, 48_000])
def test_remote_analyzer_preserves_sample_rate(monkeypatch, rate) -> None:
    rates = []

    def fake_post(url, files, timeout):
        _, payload, _ = files["audio"]
        _, wav_rate = sf.read(io.BytesIO(payload), dtype="float32")
        rates.append(wav_rate)
        return FakeResponse(payload={"transcript": "ok"})

    monkeypatch.setattr(requests, "post", fake_post)
    RemoteAnalyzer("http://server/v1/transcribe", sample_rate=rate).analyze(
        np.zeros(rate // 2, dtype=np.float32)
    )
    assert rates == [rate]
