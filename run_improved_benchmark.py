import io
import json
import os
import re
import time
from pathlib import Path
import numpy as np
import requests
import soundfile as sf

from walker_audio.processing import preprocess_speech_audio
from walker_audio.dictionary_corrector import correct_transcript

ASR_URL = "http://124.123.89.162:8001/v1/transcribe"

def compute_detailed_wer(ref: str, hyp: str):
    ref_tokens = re.sub(r"[^\w\s]", "", ref.lower()).split()
    hyp_tokens = re.sub(r"[^\w\s]", "", (hyp or "").lower()).split()
    n = len(ref_tokens)
    m = len(hyp_tokens)
    
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    ops = [[None] * (m + 1) for _ in range(n + 1)]
    
    for i in range(n + 1):
        dp[i][0] = i
        ops[i][0] = 'D'
    for j in range(m + 1):
        dp[0][j] = j
        ops[0][j] = 'I'
    ops[0][0] = 'O'
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                ops[i][j] = 'M'
            else:
                sub_cost = dp[i - 1][j - 1] + 1
                del_cost = dp[i - 1][j] + 1
                ins_cost = dp[i][j - 1] + 1
                
                min_cost = min(sub_cost, del_cost, ins_cost)
                dp[i][j] = min_cost
                if min_cost == sub_cost:
                    ops[i][j] = 'S'
                elif min_cost == del_cost:
                    ops[i][j] = 'D'
                else:
                    ops[i][j] = 'I'
                    
    i, j = n, m
    subs = 0
    dels = 0
    inss = 0
    while i > 0 or j > 0:
        op = ops[i][j]
        if op == 'M':
            i -= 1
            j -= 1
        elif op == 'S':
            subs += 1
            i -= 1
            j -= 1
        elif op == 'D':
            dels += 1
            i -= 1
        elif op == 'I':
            inss += 1
            j -= 1
            
    wer = (subs + dels + inss) / max(1, n)
    return {
        "wer": round(wer, 4),
        "subs": subs,
        "dels": dels,
        "inss": inss,
        "ref_tokens": len(ref_tokens),
        "hyp_tokens": len(hyp_tokens),
    }

def categorize_error(ref: str, hyp: str, wer_data: dict, duration: float, speech_sec: float) -> str:
    if wer_data["wer"] == 0.0:
        return "None (Exact Match)"
    if not hyp or hyp.strip() == "":
        return "Complete Deletion / VAD Drop"
    if speech_sec < duration * 0.50:
        return "Severe VAD Word Truncation / Silence Stripping"
    if wer_data["dels"] > wer_data["subs"] and wer_data["dels"] >= 2:
        return "Word Truncation / Dropout"
    if any(term in ref for term in ["pH", "HCl", "NaOH", "BP", "SpO2"]):
        return "Domain Abbreviation / Normalization Error"
    if wer_data["subs"] > 0:
        return "Acoustic / Phoneme Substitution (Accent/Denoise Distortion)"
    return "Insertion / Hallucination"

def run_improved_benchmark():
    test_cases = [
        # Education Domain
        {
            "filename": "1791458942441_The_auditorium_lecture_on_quan.wav",
            "ref": "The auditorium lecture on quantum electrodynamics will begin shortly.",
            "domain": "education",
            "display_domain": "Education (Physics)",
            "condition": "Clean Mic (Normal Speed)",
            "speaker": "Speaker 1 (Indian Accent)",
        },
        {
            "filename": "1791458977802_The_auditorium_lecture_on_quan.wav",
            "ref": "The auditorium lecture on quantum electrodynamics will begin shortly.",
            "domain": "education",
            "display_domain": "Education (Physics)",
            "condition": "Clean Mic (Slower Enunciation)",
            "speaker": "Speaker 1 (Indian Accent)",
        },
        {
            "filename": "1791535374463_thermodynamics.wav",
            "ref": "thermodynamics",
            "domain": "education",
            "display_domain": "Education (Single Word)",
            "condition": "Clean Mic (Isolated Term)",
            "speaker": "Speaker 1 (Indian Accent)",
        },
        # Hospital Domain
        {
            "filename": "1791459061564_Patient_prescribed_potassium_p.wav",
            "ref": "Patient prescribed potassium penicillin post procedural pain.",
            "domain": "hospital",
            "display_domain": "Hospital (Medication/Clinical)",
            "condition": "Clean Mic (Deliberate Speech)",
            "speaker": "Speaker 1 (Indian Accent)",
        },
        {
            "filename": "1791459126252_Cardiac_telemetry_monitors_oxy.wav",
            "ref": "Cardiac telemetry monitors oxygen saturation and vital signs.",
            "domain": "hospital",
            "display_domain": "Hospital (Cardiology)",
            "condition": "Clean Mic (Normal Speed)",
            "speaker": "Speaker 1 (Indian Accent)",
        },
        {
            "filename": "1791459211230_The_patient_presented_with_sin.wav",
            "ref": "The patient presented with sinus rhythm arrhythmia.",
            "domain": "hospital",
            "display_domain": "Hospital (Cardiology)",
            "condition": "Clean Mic (Soft Voice)",
            "speaker": "Speaker 1 (Indian Accent)",
        },
        {
            "filename": "1791527688136_hypertension.wav",
            "ref": "hypertension",
            "domain": "hospital",
            "display_domain": "Hospital (Single Word)",
            "condition": "Clean Mic (Isolated Term)",
            "speaker": "Speaker 1 (Indian Accent)",
        },
        # Industry Domain
        {
            "filename": "1791459423504_Emergency_stop_activated_on_th.wav",
            "ref": "Emergency stop activated on the automated conveyor assembly line.",
            "domain": "industry",
            "display_domain": "Industry (Safety/Machinery)",
            "condition": "Clean Mic (Normal Speed)",
            "speaker": "Speaker 1 (Indian Accent)",
        },
        {
            "filename": "1791459311104_Pressurized_pneumatic_pumps_pr.wav",
            "ref": "Pressurized pneumatic pumps provide compressed air.",
            "domain": "industry",
            "display_domain": "Industry (Mechanical)",
            "condition": "Clean Mic (Normal Speed)",
            "speaker": "Speaker 1 (Indian Accent)",
        },
        {
            "filename": "1791459358284_High_voltage_transformers_oper.wav",
            "ref": "High voltage transformers operate under continuous load conditions.",
            "domain": "industry",
            "display_domain": "Industry (Electrical)",
            "condition": "Clean Mic (Normal Speed)",
            "speaker": "Speaker 1 (Indian Accent)",
        },
        # Domain Abbreviations
        {
            "filename": "1792128759525_add_dilute_h_to_n_a_o_h_until_.wav",
            "ref": "Add dilute HCl to NaOH until the pH reaches seven.",
            "domain": "education",
            "display_domain": "Education (Chemistry Abbr)",
            "condition": "Clean Mic (Technical Abbr)",
            "speaker": "Speaker 1 (Indian Accent)",
        },
        {
            "filename": "1792128964841_ph_after_adding_hcl_to_the_sol.wav",
            "ref": "Calculate the pH after adding HCl to the solution.",
            "domain": "education",
            "display_domain": "Education (Chemistry Abbr)",
            "condition": "Clean Mic (Technical Abbr)",
            "speaker": "Speaker 1 (Indian Accent)",
        },
        {
            "filename": "1792142681584_check_b_p_and_s_p_o_two_before.wav",
            "ref": "Check BP and SpO2 before administering the medication.",
            "domain": "hospital",
            "display_domain": "Hospital (Vitals Abbr)",
            "condition": "Clean Mic (Vitals Abbr)",
            "speaker": "Speaker 1 (Indian Accent)",
        },
    ]

    recordings_dir = Path("recordings")
    results = []

    print(f"\n=======================================================")
    print(f"EXECUTING POST-IMPROVEMENT ASR ACCURACY BENCHMARK")
    print(f"Total Evaluated Samples: {len(test_cases)}")
    print(f"=======================================================\n")

    for tc in test_cases:
        wav_path = recordings_dir / tc["filename"]
        if not wav_path.exists():
            print(f"Skipping missing: {tc['filename']}")
            continue
            
        data, sr = sf.read(str(wav_path))
        raw_duration = len(data) / float(sr)
        
        t0 = time.perf_counter()
        
        # 1. Improved Preprocessing: 65Hz HPF, safe spectral denoising, 3.5dB VAD with 180ms padding & pause preservation
        processed_audio, meta = preprocess_speech_audio(
            data,
            sample_rate=sr,
            target_sample_rate=16000,
            hpf_cutoff_hz=65.0,
            denoise=True,
            denoise_strength=0.35,
            vad_threshold_db=3.5,
            vad_padding_ms=180,
            max_gap_ms=800,
            apply_vad_trim=False,  # Let Canary-Qwen receive full continuous prosody with restored 65Hz fundamentals
        )
        t_prep = (time.perf_counter() - t0) * 1000.0
        
        # Write clean WAV buffer for dispatch
        out_buf = io.BytesIO()
        sf.write(out_buf, processed_audio, 16000, format="WAV", subtype="PCM_16")
        clean_bytes = out_buf.getvalue()
        
        # 2. Canary-Qwen ASR dispatch
        t_asr0 = time.perf_counter()
        try:
            resp = requests.post(
                ASR_URL,
                files={"audio": (tc["filename"], clean_bytes, "audio/wav")},
                data={"mode": "salm", "domain": tc["domain"]},
                timeout=15,
            )
            if resp.status_code == 200:
                json_data = resp.json()
                raw_pred = json_data.get("transcript", "").strip()
                speech_sec = json_data.get("speech_seconds", raw_duration)
            else:
                raw_pred = f"[HTTP Error {resp.status_code}]"
                speech_sec = 0.0
        except Exception as e:
            raw_pred = f"[Exception: {e}]"
            speech_sec = 0.0
        t_asr = (time.perf_counter() - t_asr0) * 1000.0
        
        # 3. Enhanced Domain Dictionary Correction
        t_dict0 = time.perf_counter()
        dict_res = correct_transcript(raw_pred, domain=tc["domain"], enabled=True)
        final_pred = dict_res["corrected_transcript"]
        t_dict = (time.perf_counter() - t_dict0) * 1000.0
        
        total_latency_ms = (time.perf_counter() - t0) * 1000.0
        
        wer_stats = compute_detailed_wer(tc["ref"], final_pred)
        err_cat = categorize_error(tc["ref"], final_pred, wer_stats, raw_duration, speech_sec)
        
        row = {
            "filename": tc["filename"],
            "domain": tc["display_domain"],
            "condition": tc["condition"],
            "speaker": tc["speaker"],
            "reference": tc["ref"],
            "raw_asr": raw_pred,
            "predicted": final_pred,
            "corrections": dict_res["corrections"],
            "wer": wer_stats["wer"],
            "subs": wer_stats["subs"],
            "dels": wer_stats["dels"],
            "inss": wer_stats["inss"],
            "duration_s": round(raw_duration, 2),
            "speech_s": round(speech_sec, 2),
            "latency_ms": round(total_latency_ms, 1),
            "latency_prep_ms": round(t_prep, 1),
            "latency_asr_ms": round(t_asr, 1),
            "latency_dict_ms": round(t_dict, 1),
            "error_category": err_cat,
        }
        results.append(row)
        print(f"[{tc['display_domain']}] WER: {wer_stats['wer']*100:.1f}% | Ref: '{tc['ref']}' -> Pred: '{final_pred}' ({total_latency_ms:.0f}ms)")

    out_file = Path("improved_benchmark_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} improved benchmark results to {out_file}")

if __name__ == "__main__":
    run_improved_benchmark()
