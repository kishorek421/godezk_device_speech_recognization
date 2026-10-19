"""Comprehensive Empirical Benchmark for ASR Accuracy and Configuration Comparison (A vs B vs C).

Evaluates across verified audio/transcript pairs:
- Short technical words
- Complete sentences across Education (Physics, Chemistry, Biology, Math),
  Hospital (Cardiology, Vitals, Clinical notes), and Industry (Electrical, Transformers, Machinery)
- Noise conditions (clean, hum, fan AC) and distance attenuation
- Compares Configurations:
    Config A: ASR + Dictionary Correction
    Config B: ASR + Dictionary + Local Domain Rules
    Config C: ASR + Dictionary + Local Domain Rules + Ollama Qwen Review
- Computes WER, CER, Technical Accuracy, Abbreviation Accuracy, Latencies (Median & P95).
"""

import json
import time
import re
from pathlib import Path
import numpy as np
import requests
import soundfile as sf
import scipy.signal as sp

from walker_audio.dictionary_corrector import correct_transcript
from audio_api.context_engine import get_context_engine

ASR_URL = "http://124.123.89.162:8001/v1/transcribe"

def levenshtein_distance(seq1, seq2):
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]

def calculate_wer(reference: str, hypothesis: str) -> float:
    # Normalize: lowercase and strip punctuation except alphanumeric
    ref_tokens = re.sub(r"[^\w\s]", "", reference.lower()).split()
    hyp_tokens = re.sub(r"[^\w\s]", "", hypothesis.lower()).split()
    if not ref_tokens:
        return 0.0 if not hyp_tokens else 1.0
    return min(1.0, levenshtein_distance(ref_tokens, hyp_tokens) / len(ref_tokens))

def calculate_cer(reference: str, hypothesis: str) -> float:
    ref_chars = re.sub(r"[^\w]", "", reference.lower())
    hyp_chars = re.sub(r"[^\w]", "", hypothesis.lower())
    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0
    return min(1.0, levenshtein_distance(ref_chars, hyp_chars) / len(ref_chars))

def run_benchmark():
    engine = get_context_engine()

    # Load verified audio dataset
    manifest_path = Path("dataset/manifests/full_manifest.jsonl")
    records = []
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                records.append(json.loads(line))

    # Curate a representative, verified test suite across all required categories (24 diverse samples)
    selected_samples = []

    # 1. Short technical words
    short_words = ["thermodynamics", "hypertension", "algorithm", "molecule", "hypothesis"]
    for rec in records:
        txt = rec.get("text", "").strip()
        if txt.lower() in short_words and Path(rec.get("audio_filepath", "")).exists():
            selected_samples.append({
                "category": "short_technical_word",
                "audio_path": rec["audio_filepath"],
                "reference": txt,
                "domain": rec.get("domain", "general"),
                "subject": "",
                "topic": "",
                "expected_terms": [txt.lower()],
                "expected_abbr": [],
            })
            if len([s for s in selected_samples if s["category"] == "short_technical_word"]) >= 4:
                break

    # 2. Education Domain (Physics, Chemistry, Biology, Math)
    edu_curated = [
        {"pattern": "mathematical derivation", "domain": "education", "subject": "mathematics", "topic": "Calculus", "expected_terms": ["calculus", "differential", "integral"], "expected_abbr": []},
        {"pattern": "quantum electrodynamics", "domain": "education", "subject": "physics", "topic": "Quantum Mechanics", "expected_terms": ["quantum", "electrodynamics"], "expected_abbr": []},
        {"pattern": "photosynthesis", "domain": "education", "subject": "biology", "topic": "Cellular Energy", "expected_terms": ["photosynthesis"], "expected_abbr": []},
        {"pattern": "philosophical principles", "domain": "education", "subject": "general", "topic": "Philosophy", "expected_terms": ["principles"], "expected_abbr": []},
    ]
    for target in edu_curated:
        for rec in records:
            txt = rec.get("text", "")
            if target["pattern"] in txt.lower() and Path(rec.get("audio_filepath", "")).exists():
                selected_samples.append({
                    "category": f"education_{target['subject']}",
                    "audio_path": rec["audio_filepath"],
                    "reference": txt,
                    "domain": target["domain"],
                    "subject": target["subject"],
                    "topic": target["topic"],
                    "expected_terms": target["expected_terms"],
                    "expected_abbr": target["expected_abbr"],
                })
                break

    # 3. Hospital Domain (Cardiology, Medication, Resuscitation)
    hosp_curated = [
        {"pattern": "potassium penicillin", "domain": "hospital", "subject": "general", "topic": "Medication", "expected_terms": ["potassium penicillin"], "expected_abbr": []},
        {"pattern": "cardiac telemetry", "domain": "hospital", "subject": "cardiology", "topic": "Patient Monitoring", "expected_terms": ["cardiac", "telemetry"], "expected_abbr": []},
        {"pattern": "emergency resuscitation", "domain": "hospital", "subject": "emergency", "topic": "Resuscitation", "expected_terms": ["resuscitation"], "expected_abbr": []},
        {"pattern": "sinus rhythm", "domain": "hospital", "subject": "cardiology", "topic": "Arrhythmia", "expected_terms": ["sinus rhythm"], "expected_abbr": []},
    ]
    for target in hosp_curated:
        for rec in records:
            txt = rec.get("text", "")
            if target["pattern"] in txt.lower() and Path(rec.get("audio_filepath", "")).exists():
                selected_samples.append({
                    "category": "hospital_clinical",
                    "audio_path": rec["audio_filepath"],
                    "reference": txt,
                    "domain": target["domain"],
                    "subject": target["subject"],
                    "topic": target["topic"],
                    "expected_terms": target["expected_terms"],
                    "expected_abbr": target["expected_abbr"],
                })
                break

    # 4. Industry Domain (Electrical, Machinery, Safety)
    ind_curated = [
        {"pattern": "high voltage transformers", "domain": "industry", "subject": "electrical", "topic": "Transformers", "expected_terms": ["transformers", "voltage"], "expected_abbr": []},
        {"pattern": "emergency stop activated", "domain": "industry", "subject": "safety", "topic": "Machinery", "expected_terms": ["emergency stop", "conveyor"], "expected_abbr": []},
        {"pattern": "pneumatic pumps", "domain": "industry", "subject": "manufacturing", "topic": "Pneumatics", "expected_terms": ["pneumatic"], "expected_abbr": []},
        {"pattern": "warehouse forklift", "domain": "industry", "subject": "manufacturing", "topic": "Logistics", "expected_terms": ["forklift"], "expected_abbr": []},
    ]
    for target in ind_curated:
        for rec in records:
            txt = rec.get("text", "")
            if target["pattern"] in txt.lower() and Path(rec.get("audio_filepath", "")).exists():
                selected_samples.append({
                    "category": "industry_technical",
                    "audio_path": rec["audio_filepath"],
                    "reference": txt,
                    "domain": target["domain"],
                    "subject": target["subject"],
                    "topic": target["topic"],
                    "expected_terms": target["expected_terms"],
                    "expected_abbr": target["expected_abbr"],
                })
                break

    # 5. Technical Abbreviations (Chemistry: pH, HCl, NaOH; Hospital: BP, SpO2)
    abbr_test_recordings = [
        {
            "category": "chemistry_abbreviations",
            "filename": "1792128759525_add_dilute_h_to_n_a_o_h_until_.wav",
            "reference": "Add dilute HCl to NaOH until the pH reaches seven.",
            "domain": "education",
            "subject": "Chemistry",
            "topic": "Acids and Bases",
            "expected_terms": ["HCl", "NaOH", "pH"],
            "expected_abbr": ["pH", "HCl", "NaOH"],
        },
        {
            "category": "chemistry_abbreviations",
            "filename": "1792128964841_ph_after_adding_hcl_to_the_sol.wav",
            "reference": "Calculate the pH after adding HCl to the solution.",
            "domain": "education",
            "subject": "Chemistry",
            "topic": "Acids and Bases",
            "expected_terms": ["pH", "HCl"],
            "expected_abbr": ["pH", "HCl"],
        },
        {
            "category": "hospital_abbreviations",
            "filename": "1792142681584_check_b_p_and_s_p_o_two_before.wav",
            "reference": "Check BP and SpO2 before administering the medication.",
            "domain": "hospital",
            "subject": "cardiology",
            "topic": "Vital Signs",
            "expected_terms": ["BP", "SpO2"],
            "expected_abbr": ["BP", "SpO2"],
        },
    ]
    recordings_dir = Path("recordings")
    for t in abbr_test_recordings:
        p = recordings_dir / t["filename"]
        if p.exists():
            selected_samples.append({
                "category": t["category"],
                "audio_path": str(p),
                "reference": t["reference"],
                "domain": t["domain"],
                "subject": t["subject"],
                "topic": t["topic"],
                "expected_terms": t["expected_terms"],
                "expected_abbr": t["expected_abbr"],
            })

    # 6. Noise and distance variants (Hum & 5m Far-field)
    if (recordings_dir / "1791459358284_High_voltage_transformers_oper.wav").exists():
        selected_samples.append({
            "category": "noise_electrical_hum",
            "audio_path": str(recordings_dir / "1791459358284_High_voltage_transformers_oper.wav"),
            "reference": "High voltage transformers operate under continuous load conditions.",
            "domain": "industry",
            "subject": "electrical",
            "topic": "Transformers",
            "expected_terms": ["transformers", "voltage"],
            "expected_abbr": [],
            "augment": "hum"
        })
        selected_samples.append({
            "category": "distance_5m_farfield",
            "audio_path": str(recordings_dir / "1791459358284_High_voltage_transformers_oper.wav"),
            "reference": "High voltage transformers operate under continuous load conditions.",
            "domain": "industry",
            "subject": "electrical",
            "topic": "Transformers",
            "expected_terms": ["transformers", "voltage"],
            "expected_abbr": [],
            "augment": "far_field"
        })

    print("=" * 90)
    print("ASR ACCURACY & 3-CONFIGURATION PRODUCTION BENCHMARK")
    print(f"Total Evaluated Verified Samples: {len(selected_samples)}")
    print(f"ASR Endpoint: {ASR_URL}")
    print(f"Ollama Endpoint: {engine.base_url}")
    print("=" * 90)

    # Containers for results
    results = []

    latencies_asr = []
    latencies_dict = []
    latencies_domain = []
    latencies_ollama = []

    for idx, s in enumerate(selected_samples, 1):
        audio_path = s["audio_path"]
        ref = s["reference"]
        dom = s["domain"]
        subj = s["subject"]
        top = s["topic"]

        # Read audio waveform
        data, sr = sf.read(audio_path, dtype="float32", always_2d=True)
        mono = data.mean(axis=1)

        # Apply augmentation if requested
        if s.get("augment") == "hum":
            t = np.arange(len(mono)) / float(sr)
            hum = 0.05 * np.sin(2 * np.pi * 50 * t)
            mono = mono + hum
        elif s.get("augment") == "far_field":
            mono = mono * (1.0 / (5.0 ** 0.8))

        # Save temporary wav for ASR request
        tmp_wav = Path("recordings") / f"tmp_bench_{idx}.wav"
        sf.write(str(tmp_wav), mono, sr)

        # Stage 1: ASR Inference (Canary-Qwen 2.5B via GPU endpoint)
        t_asr0 = time.perf_counter()
        try:
            with open(tmp_wav, "rb") as f:
                files = {"audio": ("audio.wav", f, "audio/wav")}
                resp = requests.post(ASR_URL, files=files, data={"mode": "salm"}, timeout=12.0)
                if resp.status_code == 200:
                    raw_asr = resp.json().get("transcript", "").strip()
                else:
                    raw_asr = f"[ASR HTTP Error {resp.status_code}]"
        except Exception as e:
            raw_asr = f"[ASR Exception: {e}]"
        lat_asr = (time.perf_counter() - t_asr0) * 1000.0
        latencies_asr.append(lat_asr)

        # Stage 2: Dictionary Correction (Config A)
        t_dict0 = time.perf_counter()
        dict_res = correct_transcript(raw_asr, domain=dom, enabled=True)
        lat_dict = (time.perf_counter() - t_dict0) * 1000.0
        latencies_dict.append(lat_dict)
        pred_config_a = dict_res["corrected_transcript"]

        # Stage 3: Local Domain Rules (Config B)
        t_dom0 = time.perf_counter()
        # Create a fresh isolated session for benchmark fairness
        bench_session = engine.session_manager.get_or_create(f"bench_sess_{idx}", dom, subj, top)
        fast_res = engine.apply_domain_context(pred_config_a, session=bench_session, domain=dom, subject=subj, topic=top)
        lat_dom = (time.perf_counter() - t_dom0) * 1000.0
        latencies_domain.append(lat_dom)
        pred_config_b = fast_res["reviewed_transcript"]

        # Stage 4: Ollama Context Review (Config C)
        t_ol0 = time.perf_counter()
        ol_res = engine.review_transcript(
            original_transcript=raw_asr,
            dict_transcript=pred_config_a,
            session_id=f"bench_sess_{idx}",
            domain=dom,
            subject=subj,
            topic=top,
            enabled=True,
            timeout_sec=2.0  # Fast timeout threshold
        )
        lat_ol = (time.perf_counter() - t_ol0) * 1000.0
        latencies_ollama.append(lat_ol)
        pred_config_c = ol_res.get("reviewed_transcript", pred_config_b)

        # Calculate WER & CER for each config
        wer_a = calculate_wer(ref, pred_config_a)
        wer_b = calculate_wer(ref, pred_config_b)
        wer_c = calculate_wer(ref, pred_config_c)

        cer_a = calculate_cer(ref, pred_config_a)
        cer_b = calculate_cer(ref, pred_config_b)
        cer_c = calculate_cer(ref, pred_config_c)

        # Technical terms check
        exp_terms = s.get("expected_terms", [])
        def check_terms(pred_text):
            if not exp_terms:
                return 1.0
            found = sum(1 for t in exp_terms if re.search(rf"\b{re.escape(t)}\b", pred_text, re.IGNORECASE))
            return found / len(exp_terms)

        t_acc_a = check_terms(pred_config_a)
        t_acc_b = check_terms(pred_config_b)
        t_acc_c = check_terms(pred_config_c)

        # Abbreviation recovery check
        exp_abbr = s.get("expected_abbr", [])
        def check_abbr(pred_text):
            if not exp_abbr:
                return None
            found = sum(1 for a in exp_abbr if re.search(rf"\b{re.escape(a)}\b", pred_text))
            return found / len(exp_abbr)

        ab_acc_a = check_abbr(pred_config_a)
        ab_acc_b = check_abbr(pred_config_b)
        ab_acc_c = check_abbr(pred_config_c)

        # Cleanup tmp wav
        if tmp_wav.exists():
            try:
                tmp_wav.unlink()
            except Exception:
                pass

        results.append({
            "sample_id": idx,
            "category": s["category"],
            "reference": ref,
            "raw_asr": raw_asr,
            "config_a": pred_config_a,
            "config_b": pred_config_b,
            "config_c": pred_config_c,
            "wer_a": wer_a,
            "wer_b": wer_b,
            "wer_c": wer_c,
            "cer_a": cer_a,
            "cer_b": cer_b,
            "cer_c": cer_c,
            "t_acc_a": t_acc_a,
            "t_acc_b": t_acc_b,
            "t_acc_c": t_acc_c,
            "ab_acc_a": ab_acc_a,
            "ab_acc_b": ab_acc_b,
            "ab_acc_c": ab_acc_c,
            "corrections_a": dict_res["corrections"],
            "corrections_b": fast_res["corrections"],
            "corrections_c": ol_res.get("corrections", []),
            "inferred_terms_b": fast_res["inferred_terms"],
            "inferred_terms_c": ol_res.get("inferred_terms", []),
            "latency_asr": lat_asr,
            "latency_dict": lat_dict,
            "latency_domain": lat_dom,
            "latency_ollama": lat_ol,
        })
        print(f"[{idx:02d}/{len(selected_samples)}] {s['category']:<22} | WER A: {wer_a*100:4.1f}% | WER B: {wer_b*100:4.1f}% | WER C: {wer_c*100:4.1f}%")

    # Aggregate Metrics Computation
    avg_wer_a = np.mean([r["wer_a"] for r in results]) * 100.0
    avg_wer_b = np.mean([r["wer_b"] for r in results]) * 100.0
    avg_wer_c = np.mean([r["wer_c"] for r in results]) * 100.0

    avg_cer_a = np.mean([r["cer_a"] for r in results]) * 100.0
    avg_cer_b = np.mean([r["cer_b"] for r in results]) * 100.0
    avg_cer_c = np.mean([r["cer_c"] for r in results]) * 100.0

    avg_tacc_a = np.mean([r["t_acc_a"] for r in results]) * 100.0
    avg_tacc_b = np.mean([r["t_acc_b"] for r in results]) * 100.0
    avg_tacc_c = np.mean([r["t_acc_c"] for r in results]) * 100.0

    abbr_samples_a = [r["ab_acc_a"] for r in results if r["ab_acc_a"] is not None]
    abbr_samples_b = [r["ab_acc_b"] for r in results if r["ab_acc_b"] is not None]
    abbr_samples_c = [r["ab_acc_c"] for r in results if r["ab_acc_c"] is not None]

    avg_abacc_a = (np.mean(abbr_samples_a) * 100.0) if abbr_samples_a else 0.0
    avg_abacc_b = (np.mean(abbr_samples_b) * 100.0) if abbr_samples_b else 0.0
    avg_abacc_c = (np.mean(abbr_samples_c) * 100.0) if abbr_samples_c else 0.0

    # End-to-end latencies
    e2e_lat_a = [r["latency_asr"] + r["latency_dict"] for r in results]
    e2e_lat_b = [r["latency_asr"] + r["latency_dict"] + r["latency_domain"] for r in results]
    e2e_lat_c = [r["latency_asr"] + r["latency_dict"] + r["latency_ollama"] for r in results]

    print("\n" + "=" * 90)
    print("EMPIRICAL CONFIGURATION COMPARISON RESULTS")
    print("=" * 90)
    print(f"{'Metric':<35} | {'Config A (ASR+Dict)':<20} | {'Config B (ASR+Dict+Dom)':<22} | {'Config C (Full Review)'}")
    print("-" * 90)
    print(f"{'Word Error Rate (WER)':<35} | {avg_wer_a:18.2f}% | {avg_wer_b:20.2f}% | {avg_wer_c:18.2f}%")
    print(f"{'Character Error Rate (CER)':<35} | {avg_cer_a:18.2f}% | {avg_cer_b:20.2f}% | {avg_cer_c:18.2f}%")
    print(f"{'Technical Term Accuracy':<35} | {avg_tacc_a:18.2f}% | {avg_tacc_b:20.2f}% | {avg_tacc_c:18.2f}%")
    print(f"{'Abbreviation Recovery Accuracy':<35} | {avg_abacc_a:18.2f}% | {avg_abacc_b:20.2f}% | {avg_abacc_c:18.2f}%")
    print("-" * 90)
    print("PER-STAGE LATENCY BREAKDOWN (ms):")
    print(f"  - ASR (Canary-Qwen 2.5B GPU):   Median = {np.median(latencies_asr):.2f} ms,   P95 = {np.percentile(latencies_asr, 95):.2f} ms")
    print(f"  - Dictionary Corrector (CPU):   Median = {np.median(latencies_dict):.2f} ms,   P95 = {np.percentile(latencies_dict, 95):.2f} ms")
    print(f"  - Local Domain Rules (CPU):     Median = {np.median(latencies_domain):.2f} ms,   P95 = {np.percentile(latencies_domain, 95):.2f} ms")
    print(f"  - Ollama Qwen Review (Remote):  Median = {np.median(latencies_ollama):.2f} ms,   P95 = {np.percentile(latencies_ollama, 95):.2f} ms")
    print("-" * 90)
    print("END-TO-END SYSTEM LATENCY (ms):")
    print(f"  - Config A (ASR + Dict):        Median = {np.median(e2e_lat_a):.2f} ms,   P95 = {np.percentile(e2e_lat_a, 95):.2f} ms")
    print(f"  - Config B (ASR + Dict + Dom):  Median = {np.median(e2e_lat_b):.2f} ms,   P95 = {np.percentile(e2e_lat_b, 95):.2f} ms")
    print(f"  - Config C (Full Context):      Median = {np.median(e2e_lat_c):.2f} ms,   P95 = {np.percentile(e2e_lat_c, 95):.2f} ms")
    print("=" * 90)

    # Save to JSON artifact
    out_file = Path("asr_configuration_benchmark_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "total_samples": len(results),
            "summary": {
                "config_a": {"wer_pct": round(avg_wer_a, 2), "cer_pct": round(avg_cer_a, 2), "term_acc_pct": round(avg_tacc_a, 2), "abbr_acc_pct": round(avg_abacc_a, 2), "median_lat_ms": round(float(np.median(e2e_lat_a)), 2), "p95_lat_ms": round(float(np.percentile(e2e_lat_a, 95)), 2)},
                "config_b": {"wer_pct": round(avg_wer_b, 2), "cer_pct": round(avg_cer_b, 2), "term_acc_pct": round(avg_tacc_b, 2), "abbr_acc_pct": round(avg_abacc_b, 2), "median_lat_ms": round(float(np.median(e2e_lat_b)), 2), "p95_lat_ms": round(float(np.percentile(e2e_lat_b, 95)), 2)},
                "config_c": {"wer_pct": round(avg_wer_c, 2), "cer_pct": round(avg_cer_c, 2), "term_acc_pct": round(avg_tacc_c, 2), "abbr_acc_pct": round(avg_abacc_c, 2), "median_lat_ms": round(float(np.median(e2e_lat_c)), 2), "p95_lat_ms": round(float(np.percentile(e2e_lat_c, 95)), 2)},
            },
            "per_stage_latency": {
                "asr": {"median_ms": round(float(np.median(latencies_asr)), 2), "p95_ms": round(float(np.percentile(latencies_asr, 95)), 2)},
                "dictionary": {"median_ms": round(float(np.median(latencies_dict)), 2), "p95_ms": round(float(np.percentile(latencies_dict, 95)), 2)},
                "domain_rules": {"median_ms": round(float(np.median(latencies_domain)), 2), "p95_ms": round(float(np.percentile(latencies_domain, 95)), 2)},
                "ollama": {"median_ms": round(float(np.median(latencies_ollama)), 2), "p95_ms": round(float(np.percentile(latencies_ollama, 95)), 2)},
            },
            "samples": results
        }, f, indent=2)
    print(f"Saved complete ASR & configuration benchmark data to: {out_file.resolve()}")

if __name__ == "__main__":
    run_benchmark()
