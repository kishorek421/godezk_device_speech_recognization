import io
import json
import math
import os
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import soundfile as sf
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel
from scipy.signal import resample_poly
from walker_audio.dictionary_corrector import correct_transcript
from walker_audio.models import CanaryTranscriber, TitaNetVerifier
from walker_audio.processing import (
    ButterworthHighPassFilter,
    EnergyVad,
    NlmsEchoCanceller,
    SpectralDenoiser,
    combine_speech_segments,
    normalize_audio,
)


@dataclass(frozen=True)
class Settings:
    canary_model: str = os.getenv("CANARY_MODEL", "nvidia/canary-qwen-2.5b")
    speaker_model: str = os.getenv("SPEAKER_MODEL", "titanet_large")
    model_device: str = os.getenv("MODEL_DEVICE", "cuda")
    model_dtype: str = os.getenv("MODEL_DTYPE", "bfloat16")
    sample_rate: int = int(os.getenv("SAMPLE_RATE", "16000"))
    max_audio_bytes: int = int(os.getenv("MAX_AUDIO_BYTES", str(10 * 1024 * 1024)))
    max_audio_seconds: float = float(os.getenv("MAX_AUDIO_SECONDS", "60"))
    vad_threshold_db: float = float(os.getenv("VAD_THRESHOLD_DB", "3.5"))
    vad_padding_ms: int = int(os.getenv("VAD_PADDING_MS", "180"))
    max_speech_gap_ms: int = int(os.getenv("MAX_SPEECH_GAP_MS", "800"))
    hpf_cutoff_hz: float = float(os.getenv("HPF_CUTOFF_HZ", "65.0"))
    denoise_strength: float = float(os.getenv("DENOISE_STRENGTH", "0.40"))
    denoise: bool = os.getenv("DENOISE", "true").strip().lower() not in {"0", "false", "no"}
    enable_hpf: bool = os.getenv("ENABLE_HPF", "true").strip().lower() not in {"0", "false", "no"}
    enable_titanet: bool = os.getenv("ENABLE_TITANET", "true").strip().lower() not in {"0", "false", "no"}
    default_processing_mode: str = os.getenv("PROCESSING_MODE", "salm").strip().lower()
    default_domain: str = os.getenv("DEFAULT_DOMAIN", "general").strip().lower()
    enable_harvesting: bool = os.getenv("ENABLE_HARVESTING", "true").strip().lower() not in {"0", "false", "no"}
    harvest_dir: str = os.getenv("HARVEST_DIR", "dataset/harvested")
    speaker_registry_path: str = os.getenv("SPEAKER_REGISTRY_PATH", "config/speaker_registry.json")


class TranscriptionResult(BaseModel):
    transcript: str | None = None
    original_transcript: str | None = None
    corrected_transcript: str | None = None
    corrections: list[dict[str, str]] = []
    dictionary_enabled: bool = True
    speaker_id: str | None = "unknown"
    speaker_confidence: float = 0.0
    processing_mode: str = "salm"
    domain: str = "general"
    raw_asr_text: str | None = None
    speech_seconds: float = 0.0
    peak_db: float = -160.0
    rms_db: float = -160.0
    reason: str | None = None
    processing_ms: float | None = None
    model_version: str = "nvidia/canary-qwen-2.5b"


class Services:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        print(f"[STARTUP] Initializing CanaryTranscriber ({settings.canary_model}) on {settings.model_device}...")
        self.transcriber = CanaryTranscriber(
            settings.canary_model, settings.model_device, settings.model_dtype
        )
        try:
            self.transcriber.load()
            print("[STARTUP] CanaryTranscriber loaded successfully.")
        except Exception as exc:
            print(f"[STARTUP] CanaryTranscriber lazy load or fallback: {exc}")

        # Layer 2: DSP Filtering (65Hz to remove 50/60Hz hum while preserving speech fundamentals)
        self.high_pass = ButterworthHighPassFilter(settings.hpf_cutoff_hz, settings.sample_rate)

        # Layer 3: Noise Suppression with gentle strength
        self.denoiser = SpectralDenoiser(settings.sample_rate, strength=settings.denoise_strength)

        # Layer 4: Echo / Dereverb
        self.echo_canceller = NlmsEchoCanceller()

        # Layer 5: VAD with speech onset padding
        self.vad = EnergyVad(
            settings.sample_rate,
            threshold_db=settings.vad_threshold_db,
            pre_roll_ms=settings.vad_padding_ms,
        )

        # Layer 6: TitaNet Speaker Recognition
        self.verifier: TitaNetVerifier | None = None
        if settings.enable_titanet:
            print(f"[STARTUP] Initializing TitaNetVerifier ({settings.speaker_model})...")
            try:
                self.verifier = TitaNetVerifier(settings.speaker_model, settings.model_device)
                self.verifier.load()
                if Path(settings.speaker_registry_path).exists():
                    self.verifier.load_registry(settings.speaker_registry_path)
                    print(f"[STARTUP] Loaded {len(self.verifier.registry)} enrolled speakers from registry.")
                print("[STARTUP] TitaNetVerifier loaded successfully.")
            except Exception as exc:
                print(f"[STARTUP] TitaNetVerifier load error (will proceed with unknown speaker): {exc}")
                self.verifier = None

        # Setup harvested samples directory
        if settings.enable_harvesting:
            Path(settings.harvest_dir).mkdir(parents=True, exist_ok=True)
            Path(settings.harvest_dir, "wavs").mkdir(parents=True, exist_ok=True)

    def transcribe(
        self,
        audio: np.ndarray,
        mode: str | None = None,
        domain: str | None = None,
        dictionary_enabled: bool = True,
        playback_reference: np.ndarray | None = None,
    ) -> TranscriptionResult:
        started = time.perf_counter()
        active_mode = (mode or self.settings.default_processing_mode).lower()
        active_domain = (domain or self.settings.default_domain).lower()

        # LAYER 1: Hardware / Acoustic format checks
        mono = np.asarray(audio, dtype=np.float32).reshape(-1)
        if not len(mono):
            return TranscriptionResult(reason="empty_audio")

        # LAYER 2: DSP 65 Hz High-Pass Filter (removes rumble / electrical hum without clipping pitch)
        if self.settings.enable_hpf:
            mono = self.high_pass(mono)

        # LAYER 3: Neural / Spectral Noise Suppression
        processed = self.denoiser(mono) if self.settings.denoise else mono

        # LAYER 4: Echo / Dereverb (if reference supplied)
        if playback_reference is not None and len(playback_reference) == len(processed):
            processed = self.echo_canceller(processed, playback_reference)

        # LAYER 1 Normalization: Clean peak level
        processed = normalize_audio(processed, target_peak=0.85)

        peak_db = float(20 * np.log10(np.max(np.abs(processed)) + 1e-8))
        rms_db = float(20 * np.log10(np.sqrt(np.mean(processed**2)) + 1e-8))

        # LAYER 5: Voice Activity Detection (VAD)
        segments = self.vad.split(processed)
        if not segments:
            processing_ms = (time.perf_counter() - started) * 1000
            return TranscriptionResult(
                peak_db=peak_db,
                rms_db=rms_db,
                reason="no_speech",
                processing_ms=processing_ms,
                processing_mode=active_mode,
                domain=active_domain,
                dictionary_enabled=dictionary_enabled,
            )

        speech = combine_speech_segments(
            processed,
            segments,
            self.settings.sample_rate,
            max_gap_ms=self.settings.max_speech_gap_ms,
            cushion_ms=200,
        )


        # LAYER 6: TitaNet Speaker Recognition (WHO is speaking?)
        speaker_id = "unknown"
        speaker_confidence = 0.0
        if self.verifier is not None:
            try:
                speaker_id, speaker_confidence = self.verifier.identify(
                    speech, self.settings.sample_rate, threshold=0.65
                )
            except Exception as exc:
                print(f"[TITANET] Speaker identification failed: {exc}")

        # DUAL PATH: Mode A (ASR -> LLM) vs Mode B (SALM Speech-Language)
        try:
            adv_result = self.transcriber.transcribe_advanced(
                speech, self.settings.sample_rate, mode=active_mode, domain=active_domain
            )
            transcript = adv_result["transcript"]
            raw_asr_text = adv_result.get("raw_asr_text")
        except Exception as exc:
            transcript = None
            raw_asr_text = None
            print(f"[TRANSCRIBER] Execution failed: {exc}")
            raise exc

        # Dictionary-based domain terminology correction
        dict_result = correct_transcript(
            transcript=transcript or "",
            domain=active_domain,
            enabled=dictionary_enabled,
        )
        corrected_transcript = dict_result["corrected_transcript"] if transcript else None
        corrections = dict_result["corrections"] if transcript else []

        processing_ms = (time.perf_counter() - started) * 1000
        speech_seconds = len(speech) / self.settings.sample_rate

        # Background Data Harvesting (status: 'predicted')
        if self.settings.enable_harvesting and transcript:
            self._harvest_sample(
                speech=speech,
                transcript=corrected_transcript or transcript,
                raw_asr=raw_asr_text,
                speaker_id=speaker_id,
                domain=active_domain,
                mode=active_mode,
                duration=speech_seconds,
            )

        return TranscriptionResult(
            transcript=corrected_transcript if dictionary_enabled else transcript,
            original_transcript=transcript,
            corrected_transcript=corrected_transcript,
            corrections=corrections,
            dictionary_enabled=dictionary_enabled,
            speaker_id=speaker_id,
            speaker_confidence=speaker_confidence,
            processing_mode=active_mode,
            domain=active_domain,
            raw_asr_text=raw_asr_text,
            speech_seconds=speech_seconds,
            peak_db=peak_db,
            rms_db=rms_db,
            processing_ms=processing_ms,
            model_version=self.settings.canary_model,
        )

    def _harvest_sample(
        self,
        speech: np.ndarray,
        transcript: str,
        raw_asr: str | None,
        speaker_id: str,
        domain: str,
        mode: str,
        duration: float,
    ) -> None:
        try:
            sample_id = f"sample_{uuid.uuid4().hex[:10]}"
            wav_filename = f"{sample_id}.wav"
            wav_path = Path(self.settings.harvest_dir, "wavs", wav_filename)
            sf.write(str(wav_path), speech, self.settings.sample_rate)

            manifest_entry = {
                "sample_id": sample_id,
                "audio_filepath": str(wav_path.as_posix()),
                "duration": round(duration, 3),
                "predicted_transcript": transcript,
                "raw_asr_text": raw_asr,
                "speaker_id": speaker_id,
                "domain": domain,
                "processing_mode": mode,
                "status": "predicted",  # Strictly unverified until human review
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            manifest_file = Path(self.settings.harvest_dir, "samples.jsonl")
            with manifest_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(manifest_entry) + "\n")
        except Exception as exc:
            print(f"[HARVEST] Failed to harvest sample: {exc}")


settings = Settings()
services: Services | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global services
    services = Services(settings)
    yield
    services = None


app = FastAPI(
    title="Walker Audio Speech-Language & ASR Service",
    version="2.0.0",
    lifespan=lifespan,
)


def get_services() -> Services:
    if services is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Service is starting"
        )
    return services


@app.get("/health")
def health(service: Annotated[Services, Depends(get_services)]) -> dict[str, Any]:
    canary_loaded = service.transcriber.runtime_info.get("is_loaded", False)
    return {
        "status": "ok",
        "model": settings.canary_model,
        "device": service.transcriber.runtime_info.get("actual_device", settings.model_device),
        "canary_loaded": canary_loaded,
        "titanet_loaded": service.verifier is not None and service.verifier.runtime_info.get("is_loaded", False),
    }


@app.get("/model/provenance")
def provenance(service: Annotated[Services, Depends(get_services)]) -> dict[str, Any]:
    runtime_info = service.transcriber.runtime_info
    titanet_info = service.verifier.runtime_info if service.verifier else {}
    enrolled_speakers = list(service.verifier.registry.keys()) if service.verifier else []

    return {
        "base_model": settings.canary_model,
        "architecture": "NeMo SALM (FastConformer Audio Encoder + Qwen-2.5 LLM Decoder)",
        "active_checkpoint": settings.canary_model,
        "runtime_device": runtime_info.get("actual_device", settings.model_device),
        "runtime_dtype": runtime_info.get("actual_dtype", settings.model_dtype),
        "model_loaded": runtime_info.get("is_loaded", False),
        "titanet_model": settings.speaker_model,
        "titanet_loaded": titanet_info.get("is_loaded", False),
        "enrolled_speakers": enrolled_speakers,
        "default_mode": settings.default_processing_mode,
        "default_domain": settings.default_domain,
        "supported_modes": ["salm", "asr"],
        "supported_domains": ["medical", "education", "industry", "general"],
        "layers": [
            "Layer 1: Hardware Acoustic Normalization (-1.2 dBFS)",
            "Layer 2: DSP 180 Hz Butterworth High-Pass Filtering",
            "Layer 3: STFT Spectral Noise Suppression",
            "Layer 4: NLMS Echo Cancellation & Dereverberation",
            "Layer 5: Frame-Energy Voice Activity Detection",
            "Layer 6: TitaNet Speaker Recognition",
            "Layer 7: Domain-Conditioned Multimodal SALM or ASR->LLM Processing",
        ],
        "harvesting_enabled": settings.enable_harvesting,
    }


@app.get("/v1/speakers")
def list_speakers(service: Annotated[Services, Depends(get_services)]) -> dict[str, Any]:
    if service.verifier is None:
        return {"enrolled_speakers": [], "count": 0}
    return {
        "enrolled_speakers": list(service.verifier.registry.keys()),
        "count": len(service.verifier.registry),
    }


@app.post("/v1/speakers/enroll")
async def enroll_speaker(
    speaker_id: Annotated[str, Form(description="Unique speaker ID e.g. speaker_001")],
    audio: Annotated[UploadFile, File(description="Clean audio sample of target speaker")],
    service: Annotated[Services, Depends(get_services)],
) -> dict[str, Any]:
    if service.verifier is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="TitaNet is not enabled"
        )
    content = await audio.read()
    try:
        data, sample_rate = sf.read(io.BytesIO(content), dtype="float32", always_2d=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid audio") from exc

    mono = data.mean(axis=1)
    mono = _resample(mono, sample_rate, settings.sample_rate)
    service.verifier.enroll_speaker(speaker_id, mono, settings.sample_rate)

    # Persist registry
    Path(settings.speaker_registry_path).parent.mkdir(parents=True, exist_ok=True)
    service.verifier.save_registry(settings.speaker_registry_path)

    return {
        "status": "enrolled",
        "speaker_id": speaker_id,
        "total_enrolled": len(service.verifier.registry),
    }


def _resample(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return audio
    divisor = math.gcd(source_rate, target_rate)
    return resample_poly(audio, target_rate // divisor, source_rate // divisor).astype(
        np.float32
    )


@app.post("/v1/transcribe", response_model=TranscriptionResult)
async def transcribe(
    audio: Annotated[
        UploadFile, File(description="A WAV/FLAC/OGG clip, mono or stereo, any sample rate")
    ],
    service: Annotated[Services, Depends(get_services)],
    mode: Annotated[str | None, Form(description="Processing mode: 'salm' or 'asr'")] = None,
    domain: Annotated[str | None, Form(description="Domain: medical, education, industry, general")] = None,
    dictionary_enabled: Annotated[
        str | bool, Form(description="Enable dictionary-based transcript correction")
    ] = True,
) -> TranscriptionResult:
    content = await audio.read(settings.max_audio_bytes + 1)
    if len(content) > settings.max_audio_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Audio is too large",
        )
    try:
        data, source_rate = sf.read(io.BytesIO(content), dtype="float32", always_2d=True)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid audio"
        ) from exc
    mono = data.mean(axis=1)
    mono = _resample(mono, source_rate, settings.sample_rate)
    if len(mono) > int(settings.max_audio_seconds * settings.sample_rate):
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Audio is too long",
        )
    dict_enabled = (
        dictionary_enabled
        if isinstance(dictionary_enabled, bool)
        else str(dictionary_enabled).strip().lower() not in {"0", "false", "no", "off", "disable", "disabled"}
    )
    try:
        return service.transcribe(mono, mode=mode, domain=domain, dictionary_enabled=dict_enabled)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Transcription failed: {exc}"
        ) from exc

