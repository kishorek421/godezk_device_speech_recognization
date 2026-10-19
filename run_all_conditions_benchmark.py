"""Automated All-Conditions Test Matrix Runner.

Evaluates the speech system across all operating conditions:
1. Domains: Education, Hospital, Industry, General
2. Architectures: Mode B (SALM Speech-Language) vs Mode A (ASR -> LLM)
3. Noise Conditions: Clean, Low SNR, Simulated Hum/Rumble
4. Speaker Identification: Enrolled vs Unknown (TitaNet)
5. Distance Attenuation: Near-field (1.0m) vs Simulated Far-field (5.0m)

Outputs a comprehensive CSV report and summary breakdown.
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import requests
import soundfile as sf


TARGET_WORDS = {
    "education": ["algorithm", "curriculum", "derivation", "acceleration", "velocity"],
    "hospital": ["tachycardia", "hypertension", "diabetes", "anesthesia", "diagnosis"],
    "industry": ["maintenance", "machinery", "conveyor", "calibration", "vibration"],
}


def simulate_far_field(audio: np.ndarray, distance_m: float = 5.0) -> np.ndarray:
    """Simulates distance attenuation and slight room reverberation."""
    attenuation = 1.0 / max(1.0, (distance_m / 1.0) ** 0.8)
    damped = audio * attenuation
    # Add subtle reverberation tail
    tail_len = int(16000 * 0.08)
    reverb = np.zeros(len(damped) + tail_len, dtype=np.float32)
    reverb[:len(damped)] += damped
    reverb[tail_len // 2:len(damped) + tail_len // 2] += damped * 0.22
    reverb[tail_len:] += damped * 0.08
    return reverb[:len(damped)].astype(np.float32)


def simulate_noise(audio: np.ndarray, noise_type: str = "clean", snr_db: float = 12.0) -> np.ndarray:
    """Simulates realistic acoustic noise conditions."""
    if noise_type == "clean":
        return audio.astype(np.float32)

    signal_power = np.mean(audio ** 2) + 1e-8
    noise_power = signal_power / (10 ** (snr_db / 10))

    if noise_type == "hum":
        # 50 Hz electrical hum + harmonics
        t = np.arange(len(audio)) / 16000.0
        hum = np.sin(2 * np.pi * 50 * t) + 0.3 * np.sin(2 * np.pi * 100 * t)
        hum = hum * np.sqrt(noise_power)
        noisy = audio + hum
    elif noise_type == "machine":
        # Low frequency rumble + random noise
        t = np.arange(len(audio)) / 16000.0
        drone = np.sin(2 * np.pi * 120 * t) + np.sin(2 * np.pi * 240 * t)
        noise = np.random.normal(0, np.sqrt(noise_power * 0.5), len(audio)) + drone * np.sqrt(noise_power * 0.5)
        noisy = audio + noise
    else:
        # Gaussian / AC room noise
        noise = np.random.normal(0, np.sqrt(noise_power), len(audio))
        noisy = audio + noise

    return noisy.astype(np.float32)


def run_matrix_benchmark(
    api_url: str = "http://127.0.0.1:8050",
    base_wav: str = "algorithm_test.wav",
    output_csv: str = "all_conditions_benchmark_report.csv",
):
    print("\n" + "=" * 80)
    print("ALL-CONDITIONS ASR & SALM COMPREHENSIVE BENCHMARK MATRIX")
    print(f">> Target API: {api_url}")
    print(f">> Base Audio Sample: {base_wav}")
    print("=" * 80 + "\n")

    if not Path(base_wav).exists():
        wav_files = list(Path(".").glob("*.wav")) + list(Path("recordings").glob("*.wav"))
        if not wav_files:
            print(">> No WAV file available for testing.")
            return
        base_wav = str(wav_files[0])

    data, sr = sf.read(base_wav, dtype="float32", always_2d=True)
    raw_mono = data.mean(axis=1)

    domains = ["education", "hospital", "industry", "general"]
    modes = ["salm", "asr"]
    noise_conditions = ["clean", "fan_ac", "hum", "machine"]
    distances = [1.0, 5.0]

    test_records = []
    total_tests = len(domains) * len(modes) * len(noise_conditions) * len(distances)
    current = 0

    print(f">> Executing {total_tests} condition permutations...\n")
    print(f"{'#':<4} | {'Domain':<10} | {'Mode':<6} | {'Noise':<10} | {'Dist':<6} | {'Transcript':<20} | {'Speaker':<10} | {'Lat (ms)'}")
    print("-" * 88)

    for domain in domains:
        for dist in distances:
            dist_audio = simulate_far_field(raw_mono, distance_m=dist) if dist > 1.0 else raw_mono

            for noise in noise_conditions:
                noisy_audio = simulate_noise(dist_audio, noise_type=noise, snr_db=15.0 if dist == 1.0 else 8.0)

                # Export temp wav
                temp_wav_path = Path("temp_condition_test.wav")
                sf.write(str(temp_wav_path), noisy_audio, 16000)

                for mode in modes:
                    current += 1
                    t0 = time.perf_counter()
                    try:
                        resp = requests.post(
                            f"{api_url}/api/transcribe",
                            files={"audio": (temp_wav_path.name, temp_wav_path.read_bytes(), "audio/wav")},
                            data={"mode": mode, "domain": domain, "expected_word": "algorithm"},
                            timeout=45,
                        )
                        res_json = resp.json() if resp.ok else {}
                        transcript = res_json.get("transcript") or "[No Speech]"
                        latency = round(res_json.get("processing_ms") or (time.perf_counter() - t0) * 1000, 1)
                        speaker = res_json.get("speaker_id") or "unknown"
                    except Exception as exc:
                        transcript = f"[Error: {exc}]"
                        latency = 0.0
                        speaker = "unknown"

                    print(f"{current:<4} | {domain:<10} | {mode.upper():<6} | {noise:<10} | {dist}m   | {transcript[:20]:<20} | {speaker:<10} | {latency:<7}")

                    test_records.append({
                        "test_id": f"COND_{current:03d}",
                        "domain": domain,
                        "processing_mode": mode,
                        "noise_condition": noise,
                        "distance_m": dist,
                        "transcript": transcript,
                        "speaker_id": speaker,
                        "latency_ms": latency,
                    })

    # Cleanup temp wav
    if temp_wav_path.exists():
        temp_wav_path.unlink()

    # Save to CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(test_records[0].keys()))
        writer.writeheader()
        writer.writerows(test_records)

    # Calculate statistics
    salm_latencies = [r["latency_ms"] for r in test_records if r["processing_mode"] == "salm" and r["latency_ms"] > 0]
    asr_latencies = [r["latency_ms"] for r in test_records if r["processing_mode"] == "asr" and r["latency_ms"] > 0]

    avg_salm_lat = np.mean(salm_latencies) if salm_latencies else 0.0
    avg_asr_lat = np.mean(asr_latencies) if asr_latencies else 0.0

    print("\n" + "=" * 80)
    print("ALL-CONDITIONS BENCHMARK SUMMARY:")
    print(f"Total Conditions Tested: {total_tests}")
    print(f"SALM Average Latency:    {avg_salm_lat:.1f} ms")
    print(f"ASR->LLM Average Latency:{avg_asr_lat:.1f} ms")
    print(f"Speedup with SALM:       {avg_asr_lat / max(1.0, avg_salm_lat):.2f}x faster")
    print(f">> Comprehensive Report Saved: {output_csv}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="All Conditions Benchmark Runner")
    parser.add_argument("--api-url", type=str, default="http://127.0.0.1:8050")
    parser.add_argument("--base-wav", type=str, default="algorithm_test.wav")
    parser.add_argument("--output-csv", type=str, default="all_conditions_benchmark_report.csv")
    args = parser.parse_args()

    run_matrix_benchmark(api_url=args.api_url, base_wav=args.base_wav, output_csv=args.output_csv)
