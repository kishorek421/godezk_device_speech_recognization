"""Integration Test for Existing GPU ASR API and Standalone Dictionary Engine.

Target API: POST http://124.123.89.162:8001/v1/transcribe
Model: NeMo Canary-Qwen 2.5B

Contract:
- Multipart Form:
    - 'audio': File binary (WAV, FLAC, OGG)
    - 'domain': (optional) Form string ('education', 'hospital'/'medical', 'industry', 'general')
    - 'mode': (optional) 'salm' or 'asr'
- Response: JSON with 'transcript' or 'original_transcript'

Features:
- Accepts an audio file path via CLI argument (--audio <path>)
- Sends the audio to the GPU ASR API
- Records API latency and response code
- Applies local DictionaryCorrectionEngine to the received transcript
- Compares original ASR output and corrected output side-by-side
- Handles connection failures, timeouts, and missing files gracefully
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import time
from pathlib import Path

import requests

# Ensure audio_api is on sys.path for dictionary engine
audio_api_path = str(Path(__file__).parent / "audio_api")
if audio_api_path not in sys.path:
    sys.path.insert(0, audio_api_path)

from dictionary_engine import correct_text

DEFAULT_API_URL = "http://124.123.89.162:8001/v1/transcribe"
HEALTH_URL = "http://124.123.89.162:8001/health"


def check_api_health(health_url: str = HEALTH_URL, timeout: float = 3.0) -> dict | None:
    try:
        resp = requests.get(health_url, timeout=timeout)
        if resp.status_code == 200:
            return resp.json()
    except Exception as exc:
        print(f"[WARN] Health check failed for {health_url}: {exc}")
    return None


def run_gpu_integration(
    audio_path: str | None,
    api_url: str = DEFAULT_API_URL,
    domain: str = "education",
    timeout: float = 15.0,
):
    print("=" * 80)
    print("GPU ASR API + LOCAL DICTIONARY CORRECTION INTEGRATION TEST")
    print(f"Target API Endpoint : {api_url}")
    print(f"Evaluation Domain   : {domain}")
    print("=" * 80)

    # 1. Health check
    print("\n[STEP 1] Checking remote GPU API health...")
    health_info = check_api_health(timeout=3.0)
    if health_info:
        print(f"  API Status : {health_info.get('status', 'ok')}")
        print(f"  Model      : {health_info.get('model', 'unknown')}")
        print(f"  Device     : {health_info.get('device', 'unknown')}")
    else:
        print("  [WARNING] Remote API did not respond to health check. Proceeding with caution...")

    # 2. Check audio file
    print("\n[STEP 2] Verifying input audio file...")
    if not audio_path or not Path(audio_path).exists():
        print(f"  [SKIPPED] Audio file '{audio_path}' not found or not provided.")
        print("  NOTE: Per specification Section 8, audio integration test is skipped")
        print("  when no suitable audio file is provided. Text-based tests remain authoritative.")
        return {
            "status": "skipped",
            "reason": "audio_file_not_found",
            "audio_path": audio_path,
        }

    audio_file = Path(audio_path)
    print(f"  Found audio file: {audio_file.resolve()} ({audio_file.stat().st_size} bytes)")

    # 3. Transcribe via GPU API
    print("\n[STEP 3] Sending audio to GPU ASR API...")
    started = time.perf_counter()
    try:
        with open(audio_file, "rb") as f:
            files = {"audio": (audio_file.name, f, "audio/wav")}
            data = {"domain": domain, "mode": "salm"}
            response = requests.post(api_url, files=files, data=data, timeout=timeout)

        latency_ms = (time.perf_counter() - started) * 1000
        print(f"  HTTP Status Code : {response.status_code}")
        print(f"  Round-trip Latency: {latency_ms:.1f} ms")

        if response.status_code != 200:
            print(f"  [ERROR] API returned error response: {response.text}")
            return {
                "status": "failed",
                "http_status": response.status_code,
                "error": response.text,
                "latency_ms": latency_ms,
            }

        result_json = response.json()
        raw_transcript = (
            result_json.get("raw_asr_text")
            or result_json.get("transcript")
            or result_json.get("original_transcript")
            or ""
        )
        print(f"  ASR Raw Transcript: \"{raw_transcript}\"")

    except requests.exceptions.Timeout:
        print(f"  [ERROR] Request timed out after {timeout} seconds.")
        return {"status": "failed", "reason": "timeout"}
    except requests.exceptions.ConnectionError as exc:
        print(f"  [ERROR] Connection failed to {api_url}: {exc}")
        return {"status": "failed", "reason": "connection_error"}
    except Exception as exc:
        print(f"  [ERROR] Unexpected error during ASR request: {exc}")
        return {"status": "failed", "reason": str(exc)}

    # 4. Apply local DictionaryCorrectionEngine
    print("\n[STEP 4] Applying local DictionaryCorrectionEngine...")
    corrected_transcript, corrections = correct_text(raw_transcript, domain=domain)
    has_changed = raw_transcript != corrected_transcript

    # 5. Display comparison
    print("\n" + "=" * 80)
    print("RESULTS COMPARISON")
    print(f"  Domain          : {domain}")
    print(f"  Raw ASR Output  : \"{raw_transcript}\"")
    print(f"  Corrected Text  : \"{corrected_transcript}\"")
    print(f"  Changed         : {'YES' if has_changed else 'NO'}")
    print(f"  Corrections (n) : {len(corrections)}")
    for idx, c in enumerate(corrections, 1):
        print(f"    [{idx}] '{c.get('original')}' -> '{c.get('replacement')}' (applied={c.get('applied')}, reason={c.get('reason')})")
    print(f"  API Latency     : {latency_ms:.1f} ms")
    print("=" * 80)

    return {
        "status": "success",
        "raw_transcript": raw_transcript,
        "corrected_transcript": corrected_transcript,
        "corrections": corrections,
        "has_changed": has_changed,
        "latency_ms": latency_ms,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test GPU ASR API with Local Dictionary Engine")
    parser.add_argument("--audio", type=str, default=None, help="Path to WAV audio file to test")
    parser.add_argument("--url", type=str, default=DEFAULT_API_URL, help="ASR API endpoint URL")
    parser.add_argument("--domain", type=str, default="education", help="Domain: education, hospital, industry")
    args = parser.parse_args()

    run_gpu_integration(audio_path=args.audio, api_url=args.url, domain=args.domain)
