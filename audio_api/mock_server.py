"""Lightweight mock of the Walker Audio Transcription API.

Use this for local end-to-end testing without installing heavy ML dependencies
(torch / nemo_toolkit / walker_audio).

Install deps:
    pip install -r mock_requirements.txt

Run:
    uvicorn mock_server:app --host 0.0.0.0 --port 8001 --reload
    # or
    python -m uvicorn mock_server:app --host 0.0.0.0 --port 8001

The mock exposes the same routes and response schema as app/main.py:
    GET  /health
    POST /v1/transcribe

You can force the response with the `X-Mock-Scenario` header:
    transcribe   -> transcript: "mock transcript" (default)
    no_speech    -> transcript: null, reason: "no_speech"
    empty_audio  -> transcript: null, reason: "empty_audio"
    invalid      -> HTTP 400 Invalid audio
    too_large    -> HTTP 413 Audio is too large

If no header is sent, the mock guesses from the uploaded filename:
    *silence*, *quiet* -> no_speech
    *empty*, *bad*     -> invalid
"""

import os
from typing import Annotated, Any

from fastapi import FastAPI, File, Header, HTTPException, UploadFile, status
from pydantic import BaseModel


class TranscriptionResult(BaseModel):
    transcript: str | None = None
    speech_seconds: float = 0.0
    peak_db: float = -160.0
    rms_db: float = -160.0
    reason: str | None = None
    processing_ms: float | None = None


app = FastAPI(title="Walker Audio Transcription API (MOCK)", version="mock-1.0.0")

MAX_AUDIO_BYTES = int(os.getenv("MAX_AUDIO_BYTES", str(10 * 1024 * 1024)))
MOCK_TRANSCRIPT = os.getenv("MOCK_TRANSCRIPT", "mock transcript")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _guess_scenario(filename: str) -> str:
    lowered = filename.lower()
    if "silence" in lowered or "quiet" in lowered:
        return "no_speech"
    if "empty" in lowered or "bad" in lowered or "invalid" in lowered:
        return "invalid"
    return "transcribe"


def _make_result(scenario: str) -> dict[str, Any]:
    if scenario == "transcribe":
        return {
            "transcript": MOCK_TRANSCRIPT,
            "speech_seconds": 1.5,
            "peak_db": -6.0,
            "rms_db": -24.0,
            "processing_ms": 120.0,
        }
    if scenario == "no_speech":
        return {"peak_db": -60.0, "rms_db": -70.0, "reason": "no_speech"}
    if scenario == "empty_audio":
        return {"reason": "empty_audio"}
    return {"peak_db": -60.0, "rms_db": -70.0, "reason": "no_speech"}


@app.post("/v1/transcribe", response_model=TranscriptionResult)
async def transcribe(
    audio: Annotated[UploadFile, File(description="A WAV clip of microphone audio")],
    x_mock_scenario: Annotated[str | None, Header()] = None,
) -> TranscriptionResult:
    content = await audio.read(MAX_AUDIO_BYTES + 1)

    if len(content) > MAX_AUDIO_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Audio is too large",
        )

    scenario = (x_mock_scenario or _guess_scenario(audio.filename or "")).lower()

    if scenario == "too_large":
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Audio is too large",
        )

    if scenario == "invalid" or not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid audio",
        )

    return TranscriptionResult(**_make_result(scenario))
