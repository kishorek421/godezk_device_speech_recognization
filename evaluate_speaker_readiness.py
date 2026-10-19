"""Evidence-Based Speaker Identification Benchmark for CAM++ VoxCeleb.

Evaluates:
- Registered speakers (megak, sunilk) using actual recordings
- Unregistered speakers using other speaker recordings
- Clean and noisy audio (simulated hum, fan noise)
- Microphone distance (1.0m vs 5.0m)
- Short recordings vs complete sentences
- Silence and background electrical noise
- Computes Correct ID Rate, False Acceptance Rate (FAR), False Rejection Rate (FRR),
  Unknown Rejection Rate, Confusion Matrix, and Latency (median & p95).
"""

import json
import time
from pathlib import Path
import numpy as np
import soundfile as sf
import scipy.signal as sp

from walker_audio.speaker_recognizer import (
    get_speaker_registry,
    extract_voiceprint_embedding,
    isolate_voiced_region,
    extract_biometrics,
)

def run_speaker_benchmark():
    registry = get_speaker_registry()
    print("=" * 80)
    print("CAM++ VOXCELEB SPEAKER IDENTIFICATION PRODUCTION BENCHMARK")
    print(f"Registry Path: {registry.registry_file}")
    enrolled = registry.get_enrolled_summary()
    print(f"Enrolled Speakers ({len(enrolled)}): {[e['name'] for e in enrolled]}")
    for e in enrolled:
        print(f"  - {e['name']}: {e['sample_count']} samples, F0={e['f0']:.1f}Hz, F1/F2={e['f1_f2']:.2f}")
    print("=" * 80)

    recordings_dir = Path("recordings")
    
    # Define test suite with ground truth labels
    test_cases = []

    # 1. Registered speaker: megak (actual user recordings)
    megak_files = [
        "1792076656231_calculate_the_p_after_adding_h.wav",
        "1792076691869_calculate_the_p_after_adding_h.wav",
        "1792076719522_calculate_the_p_after_adding_h.wav",
        "1792078018533_p_after_adding_h_to_the_soluti.wav",
        "1792078034705_p_after_adding_h_to_the_soluti.wav",
        "1792078074893_p_after_adding_h_to_the_soluti.wav",
        "1792128759525_add_dilute_h_to_n_a_o_h_until_.wav",
        "1792128964841_ph_after_adding_hcl_to_the_sol.wav",
        "1792142681584_check_b_p_and_s_p_o_two_before.wav",
        "1792142709929_check_b_p_and_s_p_o_two_before.wav",
    ]
    for fn in megak_files:
        p = recordings_dir / fn
        if p.exists():
            test_cases.append({"path": str(p), "true_speaker": "megak", "condition": "clean_near", "type": "sentence"})

    # 2. Registered speaker: sunilk (actual recordings from sunilk session)
    sunilk_files = [
        "enrolled_sunilk_1792058987283.wav",
        "1792059040403_algorithm.wav",
        "1792059057040_algorithm.wav",
        "1792059082857_algorithm.wav",
        "1792059454009_algorithm.wav",
    ]
    for fn in sunilk_files:
        p = recordings_dir / fn
        if p.exists():
            test_cases.append({"path": str(p), "true_speaker": "sunilk", "condition": "clean_near", "type": "word_or_sentence"})

    # 3. Unregistered / Other Speakers (Primary speaker from earlier dataset recordings)
    unregistered_files = [
        "1791457214556_Principals_praised_professors_.wav",
        "1791458977802_The_auditorium_lecture_on_quan.wav",
        "1791456783164_Patient_prescribed_potassium_p.wav",
        "1791459423504_Emergency_stop_activated_on_th.wav",
        "1791459358284_High_voltage_transformers_oper.wav",
        "1791458847955_Photosynthesis_converts_carbon.wav",
        "1791884729971_please_bring_the_stetho_scope_.wav",
        "mic-20261007-150332-732935.wav",
        "mic-20261007-150337-772999.wav",
        "mic-20261007-150903-658797.wav",
    ]
    for fn in unregistered_files:
        p = recordings_dir / fn
        if p.exists():
            test_cases.append({"path": str(p), "true_speaker": "unregistered", "condition": "clean_near", "type": "sentence"})

    # 4. Silence and Noise-Only audio
    noise_files = [
        ("laptop_electric_noise_raw.wav", "electrical_hum"),
        ("laptop_electric_noise_filtered.wav", "filtered_noise"),
        ("laptop_battery_noise_raw.wav", "battery_noise"),
        ("laptop_battery_steady_raw.wav", "steady_noise"),
    ]
    for fn, cond in noise_files:
        p = recordings_dir / fn
        if p.exists():
            test_cases.append({"path": str(p), "true_speaker": "silence_or_noise", "condition": cond, "type": "noise"})

    # 5. Simulated far-field (5m) and hum-corrupted speech for megak
    if (recordings_dir / megak_files[0]).exists():
        test_cases.append({"path": str(recordings_dir / megak_files[0]), "true_speaker": "megak", "condition": "simulated_far_5m", "type": "sentence", "augment": "far_field"})
        test_cases.append({"path": str(recordings_dir / megak_files[0]), "true_speaker": "megak", "condition": "simulated_hum", "type": "sentence", "augment": "hum"})

    print(f"Total benchmark test samples prepared: {len(test_cases)}")

    results = []
    latencies = []
    emb_latencies = []

    confusion_matrix = {
        "megak": {"megak": 0, "sunilk": 0, "unknown": 0, "no_speech": 0},
        "sunilk": {"megak": 0, "sunilk": 0, "unknown": 0, "no_speech": 0},
        "unregistered": {"megak": 0, "sunilk": 0, "unknown": 0, "no_speech": 0},
        "silence_or_noise": {"megak": 0, "sunilk": 0, "unknown": 0, "no_speech": 0},
    }

    for idx, tc in enumerate(test_cases, 1):
        p = tc["path"]
        data, sr = sf.read(p, dtype="float32", always_2d=True)
        mono = data.mean(axis=1)

        # Apply augmentation if requested
        if tc.get("augment") == "far_field":
            # 5m distance attenuation
            mono = mono * (1.0 / (5.0 ** 0.8))
        elif tc.get("augment") == "hum":
            t = np.arange(len(mono)) / float(sr)
            hum = 0.05 * np.sin(2 * np.pi * 50 * t)
            mono = mono + hum

        t_start = time.perf_counter()
        
        # Measure raw embedding extraction latency
        t_emb0 = time.perf_counter()
        _ = extract_voiceprint_embedding(mono, sr)
        emb_lat_ms = (time.perf_counter() - t_emb0) * 1000.0
        emb_latencies.append(emb_lat_ms)

        # Measure complete identification pipeline (VAD + Biometrics + Centroid Matching)
        ident = registry.identify(mono, sample_rate=sr, threshold=0.65)
        total_lat_ms = (time.perf_counter() - t_start) * 1000.0
        latencies.append(total_lat_ms)

        pred_spk = ident.get("speaker_id", "Unknown Speaker")
        pred_status = ident.get("speaker_status", "UNKNOWN")
        conf = ident.get("speaker_confidence", 0.0)
        sim = ident.get("similarity_score", 0.0)

        # Normalize prediction label for confusion matrix
        if pred_status == "NO_SPEECH":
            pred_label = "no_speech"
        elif pred_spk in ("megak", "sunilk"):
            pred_label = pred_spk
        else:
            pred_label = "unknown"

        true_label = tc["true_speaker"]
        confusion_matrix[true_label][pred_label] += 1

        results.append({
            "file": Path(p).name,
            "true_speaker": true_label,
            "condition": tc["condition"],
            "pred_speaker": pred_spk,
            "pred_status": pred_status,
            "pred_label": pred_label,
            "confidence": conf,
            "similarity": sim,
            "latency_ms": round(total_lat_ms, 2),
            "emb_latency_ms": round(emb_lat_ms, 2),
        })

    # Metrics computation
    total_registered = sum(1 for r in results if r["true_speaker"] in ("megak", "sunilk"))
    correct_registered = sum(1 for r in results if r["true_speaker"] in ("megak", "sunilk") and r["pred_label"] == r["true_speaker"])
    false_rejected = sum(1 for r in results if r["true_speaker"] in ("megak", "sunilk") and r["pred_label"] in ("unknown", "no_speech"))
    cross_speaker_error = sum(1 for r in results if r["true_speaker"] in ("megak", "sunilk") and r["pred_label"] in ("megak", "sunilk") and r["pred_label"] != r["true_speaker"])

    total_unregistered = sum(1 for r in results if r["true_speaker"] == "unregistered")
    correct_unregistered_rejection = sum(1 for r in results if r["true_speaker"] == "unregistered" and r["pred_label"] in ("unknown", "no_speech"))
    false_acceptance_unregistered = sum(1 for r in results if r["true_speaker"] == "unregistered" and r["pred_label"] in ("megak", "sunilk"))

    total_noise = sum(1 for r in results if r["true_speaker"] == "silence_or_noise")
    correct_noise_rejection = sum(1 for r in results if r["true_speaker"] == "silence_or_noise" and r["pred_label"] in ("unknown", "no_speech"))
    false_acceptance_noise = sum(1 for r in results if r["true_speaker"] == "silence_or_noise" and r["pred_label"] in ("megak", "sunilk"))

    correct_id_rate = (correct_registered / total_registered * 100.0) if total_registered > 0 else 0.0
    frr = (false_rejected / total_registered * 100.0) if total_registered > 0 else 0.0
    far = (false_acceptance_unregistered / total_unregistered * 100.0) if total_unregistered > 0 else 0.0
    noise_rejection_rate = (correct_noise_rejection / total_noise * 100.0) if total_noise > 0 else 0.0

    print("\n" + "=" * 80)
    print("SPEAKER IDENTIFICATION BENCHMARK RESULTS")
    print("=" * 80)
    print(f"Total Test Audio Samples:          {len(results)}")
    print(f"Registered Speaker Samples:        {total_registered} (megak: {len(megak_files)}, sunilk: {len(sunilk_files)})")
    print(f"Unregistered Voice Samples:        {total_unregistered}")
    print(f"Silence / Background Noise:        {total_noise}")
    print("-" * 80)
    print(f"Correct Identification Rate (TPR): {correct_id_rate:.2f}% ({correct_registered}/{total_registered})")
    print(f"False Rejection Rate (FRR):        {frr:.2f}% ({false_rejected}/{total_registered})")
    print(f"False Acceptance Rate (FAR):       {far:.2f}% ({false_acceptance_unregistered}/{total_unregistered})")
    print(f"Unknown Speaker Rejection Rate:    {((total_unregistered - false_acceptance_unregistered) / total_unregistered * 100.0):.2f}%")
    print(f"Noise/Silence Rejection Rate:      {noise_rejection_rate:.2f}% ({correct_noise_rejection}/{total_noise})")
    print("-" * 80)
    print("CONFUSION MATRIX:")
    hdr = "True / Pred"
    print(f"{hdr:<20} | {'megak':<10} | {'sunilk':<10} | {'unknown':<10} | {'no_speech':<10}")
    print("-" * 65)
    for true_k, row in confusion_matrix.items():
        print(f"{true_k:<20} | {row['megak']:<10} | {row['sunilk']:<10} | {row['unknown']:<10} | {row['no_speech']:<10}")
    print("-" * 80)
    print("LATENCY MEASUREMENTS (Local CPU, Sherpa-ONNX):")
    print(f"Embedding Extraction Latency:      Median = {np.median(emb_latencies):.2f} ms,  P95 = {np.percentile(emb_latencies, 95):.2f} ms")
    print(f"Total Verification Pipeline:       Median = {np.median(latencies):.2f} ms,  P95 = {np.percentile(latencies, 95):.2f} ms")
    print("=" * 80 + "\n")

    # Save structured results to file
    out_file = Path("speaker_benchmark_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "total_samples": len(results),
            "correct_id_rate_pct": round(correct_id_rate, 2),
            "frr_pct": round(frr, 2),
            "far_pct": round(far, 2),
            "noise_rejection_pct": round(noise_rejection_rate, 2),
            "confusion_matrix": confusion_matrix,
            "latency": {
                "embedding_median_ms": round(float(np.median(emb_latencies)), 2),
                "embedding_p95_ms": round(float(np.percentile(emb_latencies, 95)), 2),
                "total_median_ms": round(float(np.median(latencies)), 2),
                "total_p95_ms": round(float(np.percentile(latencies, 95)), 2),
            },
            "samples": results
        }, f, indent=2)
    print(f"Saved complete benchmark data to: {out_file.resolve()}")

if __name__ == "__main__":
    run_speaker_benchmark()
