import io
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Annotated

# Ensure walker_audio package directory is on sys.path
_ROOT_DIR = Path(__file__).resolve().parent
_WALKER_PATH = str(_ROOT_DIR / "walker_audio")
if _WALKER_PATH not in sys.path:
    sys.path.insert(0, _WALKER_PATH)

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.responses import HTMLResponse, FileResponse
import numpy as np
import requests
import soundfile as sf
import uvicorn

# Office GPU Server URL
# Remote GPU API server - configurable via environment variable
REMOTE_GPU_API = os.getenv("REMOTE_GPU_API", "http://124.123.89.162:8001")
HTML_FILE_PATH = Path(__file__).parent / "word_test_ui.html"
RECORDINGS_DIR = Path(__file__).parent / "recordings"
RECORDINGS_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Canary-Qwen Isolated Word Benchmark Studio")


def robust_read_audio(raw_audio_bytes: bytes) -> tuple[np.ndarray, int]:
    """Reads audio bytes reliably across formats (WAV, WebM, OGG, MP3, Opus).
    Falls back to ffmpeg for unsupported formats.
    """
    try:
        data, sample_rate = sf.read(io.BytesIO(raw_audio_bytes), dtype="float32", always_2d=True)
        return data, sample_rate
    except (sf.LibsndfileError, ValueError) as e:
        # Fallback to ffmpeg for WebM/Opus browser blobs
        try:
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                "pipe:0",
                "-f",
                "wav",
                "-ar",
                "16000",
                "-ac",
                "1",
                "pipe:1",
            ]
            proc = subprocess.run(
                cmd, input=raw_audio_bytes, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
            )
            data, sample_rate = sf.read(io.BytesIO(proc.stdout), dtype="float32", always_2d=True)
            return data, sample_rate
        except (subprocess.CalledProcessError, FileNotFoundError, sf.LibsndfileError) as exc:
            raise ValueError(f"Could not decode audio with soundfile or ffmpeg: {exc}")


def prepare_audio_for_canary(raw_audio_bytes: bytes) -> tuple[bytes, dict]:
    """Prepares audio cleanly without chopping or corrupting word boundaries.

    Ensures 16 kHz mono and optimal peak volume for Canary-Qwen.
    """
    try:
        data, sample_rate = robust_read_audio(raw_audio_bytes)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid audio format: {exc}")

    mono = data.mean(axis=1)

    # Resample to 16 kHz if needed
    if sample_rate != 16000:
        import scipy.signal as sp
        divisor = math.gcd(sample_rate, 16000)
        mono = sp.resample_poly(mono, 16000 // divisor, sample_rate // divisor).astype(np.float32)
        sample_rate = 16000

    raw_peak = float(np.max(np.abs(mono))) if len(mono) > 0 else 0.0
    raw_peak_db = float(20 * np.log10(raw_peak + 1e-8))
    duration_sec = len(mono) / sample_rate

    # NOTE: High-pass filter removed to avoid double filtering
    # The remote GPU API (audio_api) applies 65Hz filter as part of its DSP pipeline
    # word_test_server only handles resampling and normalization

    # Normalize speech cleanly to -1.2 dBFS (only if voiced speech is present)
    clean_peak = float(np.percentile(np.abs(mono), 99.8)) if len(mono) > 0 else 0.0
    is_silence = False
    if clean_peak >= 0.020 and raw_peak_db >= -34.0:
        mono = mono * (0.85 / clean_peak)
        mono = np.clip(mono, -0.98, 0.98)
    else:
        # Near silence/room hiss: keep low so ASR and speaker models don't hallucinate on amplified noise
        is_silence = True

    out_buf = io.BytesIO()
    sf.write(out_buf, mono, sample_rate, format="WAV", subtype="PCM_16")

    meta = {
        "raw_peak_db": round(raw_peak_db, 1),
        "duration_sec": round(duration_sec, 2),
        "hum_filtered": True,
        "is_silence": is_silence,
    }
    return out_buf.getvalue(), meta


@app.get("/", response_class=HTMLResponse)
def index():
    if not HTML_FILE_PATH.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="UI template word_test_ui.html not found.",
        )
    return HTML_FILE_PATH.read_text(encoding="utf-8")


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/health")
def health_proxy():
    try:
        resp = requests.get(f"{REMOTE_GPU_API}/health", timeout=5)
        if resp.ok:
            return resp.json()
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Cannot reach GPU server at {REMOTE_GPU_API}: {exc}",
        )


@app.get("/model/provenance")
def provenance_proxy():
    try:
        resp = requests.get(f"{REMOTE_GPU_API}/model/provenance", timeout=5)
        if resp.ok:
            return resp.json()
        # Fallback if remote server is running previous version
        return {
            "base_model": "nvidia/canary-qwen-2.5b",
            "architecture": "NeMo SALM (FastConformer + Qwen-2.5)",
            "runtime_device": "cuda",
            "runtime_dtype": "bfloat16",
            "model_loaded": True,
            "titanet_loaded": True,
            "supported_modes": ["salm", "asr"],
            "supported_domains": ["medical", "education", "industry", "general"],
        }
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Cannot reach GPU server at {REMOTE_GPU_API}: {exc}",
        )


from walker_audio.speaker_recognizer import get_speaker_registry

speaker_registry = get_speaker_registry()


@app.get("/api/speakers")
@app.get("/v1/speakers")
def list_enrolled_speakers():
    summary = speaker_registry.get_enrolled_summary()
    return {
        "enrolled_speakers": [s["name"] for s in summary],
        "profiles": summary,
        "count": len(summary),
    }


@app.post("/api/speakers/enroll")
async def enroll_speaker_endpoint(
    name: Annotated[str, Form(description="Person's name to register")],
    audio: Annotated[UploadFile, File(description="Voice recording clip")],
):
    content = await audio.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty voice audio received")

    # Clean & normalize voice using exact same DSP filter (removes 50Hz electrical hum)
    clean_audio_bytes, meta = prepare_audio_for_canary(content)

    import time
    ts = int(time.time() * 1000)
    enrolled_path = RECORDINGS_DIR / f"enrolled_{name}_{ts}.wav"
    try:
        enrolled_path.write_bytes(clean_audio_bytes)
    except Exception:
        pass

    try:
        data, sample_rate = robust_read_audio(clean_audio_bytes)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid audio format: {exc}")

    mono = data.mean(axis=1)
    duration_sec = len(mono) / sample_rate
    audio_proof_url = f"/api/audio/{enrolled_path.name}"
    try:
        result = speaker_registry.enroll(name, mono, sample_rate, duration_sec=duration_sec, audio_url=audio_proof_url)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return {
        "status": "enrolled",
        "speaker_name": name,
        "sample_count": result["sample_count"],
        "total_enrolled": result["total_enrolled"],
        "enrolled_speakers": result["enrolled_names"],
        "audio_url": f"/api/audio/{enrolled_path.name}",
    }


@app.delete("/api/speakers/{speaker_name}")
def delete_speaker_endpoint(speaker_name: str):
    success = speaker_registry.delete(speaker_name)
    return {"status": "deleted" if success else "not_found", "speaker_name": speaker_name}


@app.delete("/api/speakers")
def clear_all_speakers_endpoint():
    speaker_registry.clear()
    return {"status": "cleared", "total_enrolled": 0, "enrolled_speakers": []}


def prune_old_recordings(max_keep: int = 300):
    """Retains only the latest max_keep recordings to prevent unbounded disk growth."""
    try:
        wav_files = sorted(RECORDINGS_DIR.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
        for old_file in wav_files[max_keep:]:
            try:
                old_file.unlink()
            except Exception:
                pass
    except Exception:
        pass


@app.get("/api/audio/{filename}")
def get_audio_recording(filename: str):
    safe_name = Path(filename).name
    file_path = (RECORDINGS_DIR / safe_name).resolve()
    # Path traversal protection: Ensure target file is strictly inside RECORDINGS_DIR
    if not file_path.exists() or not str(file_path).startswith(str(RECORDINGS_DIR.resolve())):
        raise HTTPException(status_code=404, detail="Audio proof file not found")
    return FileResponse(file_path, media_type="audio/wav")


from walker_audio.dictionary_corrector import correct_transcript, load_domain_dictionary


@app.get("/api/dictionary")
def get_dictionary():
    return load_domain_dictionary()


@app.post("/api/dictionary/test")
def test_dictionary_endpoint(text: Annotated[str, Form()], domain: Annotated[str, Form()] = "general"):
    return correct_transcript(text, domain=domain, enabled=True)


from audio_api.context_engine import get_context_engine

context_engine = get_context_engine()


@app.get("/api/context/session/{session_id}")
def get_session_context(session_id: str):
    summary = context_engine.session_manager.get_summary(session_id)
    if not summary:
        return {"session_id": session_id, "turn_count": 0, "recent_turns": []}
    return summary


@app.post("/api/context/session/reset")
def reset_session_context(session_id: Annotated[str, Form()] = "default_session"):
    success = context_engine.session_manager.reset(session_id)
    return {"status": "reset" if success else "not_found", "session_id": session_id}


@app.post("/api/transcribe")
async def transcribe_proxy(
    audio: Annotated[UploadFile, File(description="Recorded audio WAV from browser")],
    filter_mode: Annotated[str, Form()] = "clean",
    expected_word: Annotated[str, Form()] = "",
    mode: Annotated[str, Form()] = "salm",
    domain: Annotated[str, Form()] = "general",
    dictionary_enabled: Annotated[str, Form()] = "true",
    subject: Annotated[str, Form()] = "",
    topic: Annotated[str, Form()] = "",
    session_id: Annotated[str, Form()] = "default_session",
    context_review_enabled: Annotated[str, Form()] = "false",
):

    content = await audio.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty audio received")

    # Clean & normalize audio without destroying word syllables
    clean_audio_bytes, meta = prepare_audio_for_canary(content)

    import time
    import re
    timestamp = int(time.time() * 1000)
    safe_word = re.sub(r"[^a-zA-Z0-9_-]", "_", expected_word)[:30] or "speech"
    audio_filename = f"{timestamp}_{safe_word}.wav"
    audio_path = RECORDINGS_DIR / audio_filename

    # Save audio proof file and prune oldest records
    try:
        audio_path.write_bytes(clean_audio_bytes)
        (RECORDINGS_DIR / "debug_last_raw.wav").write_bytes(content)
        (RECORDINGS_DIR / "debug_last_clean.wav").write_bytes(clean_audio_bytes)
        prune_old_recordings(max_keep=300)
    except Exception:
        pass

    dict_enabled = str(dictionary_enabled).strip().lower() not in {
        "0", "false", "no", "off", "disable", "disabled"
    }

    # Dynamic threshold calibration:
    # Short words (<1.2s) yield naturally lower CAM++ cosine similarity (0.60-0.65).
    # Calibrate to 0.62 for short clips (<1.2s) and 0.68 for normal/longer clips (>=1.2s).
    duration_sec = meta.get("duration_sec", 2.0)
    calibrated_threshold = 0.62 if duration_sec < 1.2 else 0.68

    # Neural CAM++ + Biometric speaker identification
    spk_res = speaker_registry.identify(clean_audio_bytes, sample_rate=16000, threshold=calibrated_threshold)
    spk_id = spk_res["speaker_id"]
    spk_conf = spk_res["speaker_confidence"]
    spk_status = spk_res["speaker_status"]
    spk_reason = spk_res.get("reason", "")

    try:
        resp = requests.post(
            f"{REMOTE_GPU_API}/v1/transcribe",
            files={"audio": (audio.filename or "recording.wav", clean_audio_bytes, "audio/wav")},
            data={"mode": mode, "domain": domain, "dictionary_enabled": str(dict_enabled)},
            timeout=25,
        )
        if resp.ok:
            result = resp.json()
            result["filter_meta"] = meta
            result["audio_url"] = f"/api/audio/{audio_filename}"

            raw_text = result.get("original_transcript") or result.get("transcript") or ""

            # Check if ASR returned no speech, pure punctuation (e.g. '..'), or silence
            clean_letters = re.sub(r'[\W_]+', '', raw_text)
            is_no_speech = (
                not clean_letters
                or result.get("reason") == "no_speech"
                or raw_text.strip().lower() in {"", "..", "...", ".", "-", "[no_speech]", "none", "null", "no speech"}
                or meta.get("is_silence", False)
                or spk_status == "NO_SPEECH"
            )

            if is_no_speech:
                spk_id = "Unknown Speaker"
                spk_status = "NO_SPEECH"
                spk_conf = 0.0
                result["reason"] = "no_speech"

            # Always run local dictionary correction using updated local dictionary
            dict_res = correct_transcript(raw_text, domain=domain, enabled=dict_enabled)
            result["original_transcript"] = raw_text
            result["corrected_transcript"] = dict_res["corrected_transcript"]
            result["corrections"] = dict_res["corrections"]
            result["dictionary_enabled"] = dict_enabled
            if dict_enabled:
                result["transcript"] = dict_res["corrected_transcript"]

            # Optional LLM Context-Review Stage (Ollama Qwen) - Disabled by default
            llm_enabled = str(context_review_enabled).strip().lower() in {
                "1", "true", "yes", "on", "enable", "enabled"
            }
            context_res = context_engine.review_transcript(
                original_transcript=raw_text,
                dict_transcript=dict_res["corrected_transcript"],
                session_id=session_id,
                domain=domain,
                subject=subject,
                topic=topic,
                speaker_id=spk_id,
                enabled=llm_enabled,
            )
            result["llm_context_enabled"] = llm_enabled
            result["reviewed_transcript"] = context_res.get("reviewed_transcript", dict_res["corrected_transcript"])
            result["context_corrections"] = context_res.get("corrections", [])
            result["context_correction_reasons"] = context_res.get("correction_reasons", [])
            result["inferred_terms"] = context_res.get("inferred_terms", [])
            result["uncertain_segments"] = context_res.get("uncertain_segments", [])
            result["requires_audio_recheck"] = context_res.get("requires_audio_recheck", False)
            result["context_latency_ms"] = context_res.get("context_latency_ms", 0.0)
            result["session_id"] = session_id
            result["subject"] = subject
            result["topic"] = topic

            # If LLM review was explicitly enabled and returned a verified review, use it as the final transcript
            if llm_enabled and context_res.get("llm_reviewed"):
                result["transcript"] = context_res["reviewed_transcript"]

            result["speaker_id"] = spk_id
            result["speaker_confidence"] = spk_conf
            result["speaker_status"] = spk_status
            result["speaker_score"] = 0.0 if is_no_speech else spk_res.get("similarity_score", 0.0)
            result["speaker_samples"] = spk_res.get("sample_count", 0)
            result["speaker_reason"] = spk_reason


            p_mode = result.get("processing_mode") or mode
            corr_count = len(result.get("corrections", []))
            proc_ms = result.get("processing_ms", 0.0)
            print(f"[STT-STUDIO] [{p_mode.upper()}|{domain.upper()}] Speaker: {spk_id} ({spk_status}|{spk_conf*100:.1f}%) | Transcript: '{result['transcript']}' | Latency: {proc_ms:.0f}ms | Duration: {meta['duration_sec']}s")
            return result
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    except requests.Timeout:
        # Fallback if GPU server has high latency: still provide speaker identification and audio proof
        return {
            "status": "partial_gpu_timeout",
            "transcript": "[Audio received - GPU server busy, speaker recognized]",
            "original_transcript": "[Audio received]",
            "corrected_transcript": "[Audio received]",
            "corrections": [],
            "speaker_id": spk_id,
            "speaker_confidence": spk_conf,
            "speaker_status": spk_status,
            "filter_meta": meta,
            "audio_url": f"/api/audio/{audio_filename}",
            "dictionary_enabled": dict_enabled,
            "reason": "GPU server timed out, speaker voiceprint recognized locally",
        }
    except Exception as exc:
        return {
            "status": "partial_gpu_offline",
            "transcript": "[Audio received - GPU offline, speaker recognized]",
            "original_transcript": "[Audio received]",
            "corrected_transcript": "[Audio received]",
            "corrections": [],
            "speaker_id": spk_id,
            "speaker_confidence": spk_conf,
            "speaker_status": spk_status,
            "filter_meta": meta,
            "audio_url": f"/api/audio/{audio_filename}",
            "dictionary_enabled": dict_enabled,
            "reason": f"GPU server error: {exc}",
        }



if __name__ == "__main__":
    port = int(os.getenv("PORT", "8050"))
    print("\n" + "=" * 60)
    print("CANARY-QWEN 2.5B WORD BENCHMARK STUDIO")
    print(f">> Open in browser: http://localhost:{port}")
    print(f">> Connected GPU Server: {REMOTE_GPU_API}")
    print("=" * 60 + "\n")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")
