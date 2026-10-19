"""
Multi-Layer Noise Pipeline Verification & Benchmarking Script
Tests Layer 1, Layer 2, Layer 3, and Layer 4 with real audio proof generation.
"""

import io
import math
from pathlib import Path
import numpy as np
import requests
import soundfile as sf

SERVER_URL = "http://127.0.0.1:8050"
SR = 16000

TEST_CONDITIONS = [
    {
        "layer": "Layer 1: Hardware / Acoustic",
        "noise_type": "Breath Pop ('P' Plosives)",
        "domain": "Hospital",
        "sentence": "Patient prescribed potassium penicillin post procedural pain",
        "distance": "0.2m",
        "noise_sim": "plosive",
    },
    {
        "layer": "Layer 1: Hardware / Acoustic",
        "noise_type": "Wind / Air Draft",
        "domain": "Education",
        "sentence": "Principals praised professors proposing profound philosophical principles",
        "distance": "0.2m",
        "noise_sim": "wind",
    },
    {
        "layer": "Layer 2: DSP Frequency Filtering",
        "noise_type": "50-Hz Electrical Hum",
        "domain": "Industry",
        "sentence": "High voltage transformers operate at fifty hertz electrical frequency",
        "distance": "1.0m",
        "noise_sim": "50hz_hum",
    },
    {
        "layer": "Layer 2: DSP Frequency Filtering",
        "noise_type": "AC Low Rumble (<180 Hz)",
        "domain": "General",
        "sentence": "Air conditioning systems reduce ambient room temperature consistently",
        "distance": "1.0m",
        "noise_sim": "ac_rumble",
    },
    {
        "layer": "Layer 3: Neural Noise Suppression",
        "noise_type": "Fan Running Drone",
        "domain": "Hospital",
        "sentence": "The patient presented with sinus tachycardia and pulmonary edema",
        "distance": "1.0m",
        "noise_sim": "fan",
    },
    {
        "layer": "Layer 3: Neural Noise Suppression",
        "noise_type": "Machinery Hum",
        "domain": "Industry",
        "sentence": "Emergency stop activated on the automated conveyor assembly line",
        "distance": "1.0m",
        "noise_sim": "machine",
    },
    {
        "layer": "Layer 4: Echo / Dereverberation",
        "noise_type": "Hall Echo (3m-5m)",
        "domain": "Education",
        "sentence": "The auditorium lecture on quantum electrodynamics will begin shortly",
        "distance": "3.0m",
        "noise_sim": "hall_echo",
    },
    {
        "layer": "Layer 4: Echo / Dereverberation",
        "noise_type": "Room Reverberation (3m-5m)",
        "domain": "General",
        "sentence": "Can you hear me clearly from the other side of the conference hall",
        "distance": "5.0m",
        "noise_sim": "room_reverb",
    },
]


def generate_test_signal(noise_sim: str, duration_sec: float = 3.2) -> np.ndarray:
    """Generates synthetic speech-like harmonic carrier corrupted by specific noise."""
    t = np.linspace(0, duration_sec, int(SR * duration_sec), endpoint=False)
    
    # Fundamental voice formant structure (vowel simulation: F0=130Hz, F1=600Hz, F2=1200Hz, F3=2400Hz)
    voice = (
        0.35 * np.sin(2 * np.pi * 130 * t) +
        0.25 * np.sin(2 * np.pi * 260 * t) +
        0.20 * np.sin(2 * np.pi * 600 * t) +
        0.15 * np.sin(2 * np.pi * 1200 * t) +
        0.10 * np.sin(2 * np.pi * 2400 * t)
    )
    # Natural speech syllabic amplitude envelope (4-5 syllabic bursts)
    envelope = (np.sin(2 * np.pi * 1.8 * t) ** 2) * (t > 0.3) * (t < duration_sec - 0.3)
    speech = voice * envelope * 0.45

    if noise_sim == "plosive":
        # Massive low-frequency acoustic puff/pop at syllable onsets (10 Hz - 40 Hz burst)
        puff = np.zeros_like(t)
        for burst_time in [0.5, 1.2, 1.8, 2.4]:
            idx = int(burst_time * SR)
            puff_len = int(0.12 * SR)
            puff_t = np.linspace(0, 0.12, puff_len)
            puff[idx:idx + puff_len] += 0.8 * np.sin(2 * np.pi * 25 * puff_t) * np.hanning(puff_len)
        return speech + puff

    elif noise_sim == "wind":
        # Modulated low frequency turbulent pink/brown noise
        wind = np.cumsum(np.random.randn(len(t)))
        wind = (wind / np.max(np.abs(wind))) * 0.4 * (1.0 + 0.5 * np.sin(2 * np.pi * 0.8 * t))
        return speech + wind

    elif noise_sim == "50hz_hum":
        # High-amplitude 50 Hz power hum + 100 Hz harmonic
        hum = 0.65 * np.sin(2 * np.pi * 50 * t) + 0.25 * np.sin(2 * np.pi * 100 * t)
        return speech + hum

    elif noise_sim == "ac_rumble":
        # Low sub-bass vibration (30 Hz - 90 Hz)
        rumble = 0.5 * np.sin(2 * np.pi * 42 * t) + 0.3 * np.sin(2 * np.pi * 84 * t)
        return speech + rumble

    elif noise_sim == "fan":
        # Continuous stationary pink/white hiss
        fan = 0.25 * np.random.randn(len(t))
        return speech + fan

    elif noise_sim == "machine":
        # Industrial 120 Hz motor drone + random vibration
        drone = 0.35 * np.sin(2 * np.pi * 120 * t) + 0.15 * np.sin(2 * np.pi * 360 * t) + 0.1 * np.random.randn(len(t))
        return speech + drone

    elif noise_sim in ("hall_echo", "room_reverb"):
        # Attenuate direct signal for distance (3m - 5m) + add reflections at 60ms, 120ms, 220ms
        far_speech = speech * 0.25
        echo1 = np.roll(speech * 0.15, int(0.07 * SR))
        echo2 = np.roll(speech * 0.09, int(0.15 * SR))
        echo3 = np.roll(speech * 0.05, int(0.24 * SR))
        return far_speech + echo1 + echo2 + echo3

    return speech


def run_benchmark():
    print("=" * 80)
    print("CANARY-QWEN MULTI-LAYER NOISE BENCHMARK WITH AUDIO PROOF")
    print(f"Target Server: {SERVER_URL}")
    print("=" * 80)

    results = []
    for idx, test in enumerate(TEST_CONDITIONS, 1):
        raw_audio = generate_test_signal(test["noise_sim"])
        buf = io.BytesIO()
        sf.write(buf, raw_audio.astype(np.float32), SR, format="WAV", subtype="PCM_16")
        wav_bytes = buf.getvalue()

        print(f"\n[{idx}/{len(TEST_CONDITIONS)}] Testing: {test['layer']} | {test['noise_type']}")
        print(f"    Domain: {test['domain']} | Dist: {test['distance']}")
        print(f"    Sentence: '{test['sentence']}'")

        resp = requests.post(
            f"{SERVER_URL}/api/transcribe",
            files={"audio": (f"test_layer_{idx}.wav", wav_bytes, "audio/wav")},
            data={
                "expected_word": test["sentence"],
                "mode": "salm",
                "domain": test["domain"].lower(),
            },
            timeout=45,
        )

        if resp.ok:
            data = resp.json()
            audio_url = data.get("audio_url", "")
            transcript = data.get("transcript", "")
            proc_ms = data.get("processing_ms", 0.0)
            meta = data.get("filter_meta", {})
            print(f"    -> Status: SUCCESS ({proc_ms:.1f} ms)")
            print(f"    -> Transcript: '{transcript}'")
            print(f"    -> Audio Proof URL: {SERVER_URL}{audio_url}")
            print(f"    -> Hum Filtered: {meta.get('hum_filtered')} | Peak: {meta.get('raw_peak_db')} dBFS")
            results.append({**test, "transcript": transcript, "latency_ms": proc_ms, "audio_url": audio_url})
        else:
            print(f"    -> FAILED: HTTP {resp.status_code}: {resp.text}")

    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY & PROOF TABLE:")
    print(f"{'Layer':<28} | {'Noise Type':<22} | {'Dist':<6} | {'Latency':<9} | Audio Proof File")
    print("-" * 95)
    for r in results:
        print(f"{r['layer']:<28} | {r['noise_type']:<22} | {r['distance']:<6} | {r['latency_ms']:<7.1f}ms | {r['audio_url']}")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmark()
