import json
import os
import re
import time
from pathlib import Path
import numpy as np
import requests
import soundfile as sf

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
    if speech_sec < duration * 0.65:
        return "Severe VAD Word Truncation / Silence Stripping"
    if wer_data["dels"] > wer_data["subs"] and wer_data["dels"] >= 2:
        return "Word Truncation / Dropout"
    if any(term in ref for term in ["pH", "HCl", "NaOH", "BP", "SpO2"]):
        return "Domain Abbreviation / Normalization Error"
    if wer_data["subs"] > 0:
        return "Acoustic / Phoneme Substitution (Accent/Denoise Distortion)"
    return "Insertion / Hallucination"

def run_baseline_benchmark():
    test_cases = [
        # Education Domain
        {
            "filename": "1791458942441_The_auditorium_lecture_on_quan.wav",
            "ref": "The auditorium lecture on quantum electrodynamics will begin shortly.",
            "domain": "Education (Physics)",
            "condition": "Clean Mic (Normal Speed)",
            "speaker": "Speaker 1 (Indian Accent)",
        },
        {
            "filename": "1791458977802_The_auditorium_lecture_on_quan.wav",
            "ref": "The auditorium lecture on quantum electrodynamics will begin shortly.",
            "domain": "Education (Physics)",
            "condition": "Clean Mic (Slower Enunciation)",
            "speaker": "Speaker 1 (Indian Accent)",
        },
        {
            "filename": "1791457214556_Principals_praised_professors_.wav",
            "ref": "Principals praised professors proposing profound philosophical principles.",
            "domain": "Education (Alliteration/Complex)",
            "condition": "Clean Mic (Rapid Connected Speech)",
            "speaker": "Speaker 1 (Indian Accent)",
        },
        {
            "filename": "1791535374463_thermodynamics.wav",
            "ref": "thermodynamics",
            "domain": "Education (Single Word)",
            "condition": "Clean Mic (Isolated Term)",
            "speaker": "Speaker 1 (Indian Accent)",
        },
        # Hospital Domain
        {
            "filename": "1791456783164_Patient_prescribed_potassium_p.wav",
            "ref": "Patient prescribed potassium penicillin post procedural pain.",
            "domain": "Hospital (Medication/Clinical)",
            "condition": "Clean Mic (Normal Speed)",
            "speaker": "Speaker 1 (Indian Accent)",
        },
        {
            "filename": "1791459061564_Patient_prescribed_potassium_p.wav",
            "ref": "Patient prescribed potassium penicillin post procedural pain.",
            "domain": "Hospital (Medication/Clinical)",
            "condition": "Clean Mic (Deliberate Speech)",
            "speaker": "Speaker 1 (Indian Accent)",
        },
        {
            "filename": "1791459126252_Cardiac_telemetry_monitors_oxy.wav",
            "ref": "Cardiac telemetry monitors oxygen saturation and vital signs.",
            "domain": "Hospital (Cardiology)",
            "condition": "Clean Mic (Normal Speed)",
            "speaker": "Speaker 1 (Indian Accent)",
        },
        {
            "filename": "1791459211230_The_patient_presented_with_sin.wav",
            "ref": "The patient presented with sinus rhythm arrhythmia.",
            "domain": "Hospital (Cardiology)",
            "condition": "Clean Mic (Soft Voice)",
            "speaker": "Speaker 1 (Indian Accent)",
        },
        {
            "filename": "1791527688136_hypertension.wav",
            "ref": "hypertension",
            "domain": "Hospital (Single Word)",
            "condition": "Clean Mic (Isolated Term)",
            "speaker": "Speaker 1 (Indian Accent)",
        },
        # Industry Domain
        {
            "filename": "1791459423504_Emergency_stop_activated_on_th.wav",
            "ref": "Emergency stop activated on the automated conveyor assembly line.",
            "domain": "Industry (Safety/Machinery)",
            "condition": "Clean Mic (Normal Speed)",
            "speaker": "Speaker 1 (Indian Accent)",
        },
        {
            "filename": "1791459311104_Pressurized_pneumatic_pumps_pr.wav",
            "ref": "Pressurized pneumatic pumps provide compressed air.",
            "domain": "Industry (Mechanical)",
            "condition": "Clean Mic (Normal Speed)",
            "speaker": "Speaker 1 (Indian Accent)",
        },
        {
            "filename": "1791459358284_High_voltage_transformers_oper.wav",
            "ref": "High voltage transformers operate under continuous load conditions.",
            "domain": "Industry (Electrical)",
            "condition": "Clean Mic (Normal Speed)",
            "speaker": "Speaker 1 (Indian Accent)",
        },
        # Domain Abbreviations
        {
            "filename": "1792128759525_add_dilute_h_to_n_a_o_h_until_.wav",
            "ref": "Add dilute HCl to NaOH until the pH reaches seven.",
            "domain": "Education (Chemistry Abbr)",
            "condition": "Clean Mic (Technical Abbr)",
            "speaker": "Speaker 1 (Indian Accent)",
        },
        {
            "filename": "1792128964841_ph_after_adding_hcl_to_the_sol.wav",
            "ref": "Calculate the pH after adding HCl to the solution.",
            "domain": "Education (Chemistry Abbr)",
            "condition": "Clean Mic (Technical Abbr)",
            "speaker": "Speaker 1 (Indian Accent)",
        },
        {
            "filename": "1792142681584_check_b_p_and_s_p_o_two_before.wav",
            "ref": "Check BP and SpO2 before administering the medication.",
            "domain": "Hospital (Vitals Abbr)",
            "condition": "Clean Mic (Vitals Abbr)",
            "speaker": "Speaker 1 (Indian Accent)",
        },
    ]

    recordings_dir = Path("recordings")
    results = []

    print(f"Executing baseline benchmark on {len(test_cases)} audio samples across conditions...")

    for tc in test_cases:
        wav_path = recordings_dir / tc["filename"]
        if not wav_path.exists():
            print(f"Skipping missing: {tc['filename']}")
            continue
            
        data, sr = sf.read(str(wav_path))
        duration = len(data) / float(sr)
        
        t0 = time.perf_counter()
        try:
            with open(wav_path, "rb") as f:
                resp = requests.post(ASR_URL, files={"audio": (tc["filename"], f, "audio/wav")}, timeout=15)
                if resp.status_code == 200:
                    json_data = resp.json()
                    pred = json_data.get("transcript", "").strip()
                    speech_sec = json_data.get("speech_seconds", duration)
                else:
                    pred = f"[HTTP Error {resp.status_code}]"
                    speech_sec = 0.0
        except Exception as e:
            pred = f"[Exception: {e}]"
            speech_sec = 0.0
        latency_ms = (time.perf_counter() - t0) * 1000.0
        
        wer_stats = compute_detailed_wer(tc["ref"], pred)
        err_cat = categorize_error(tc["ref"], pred, wer_stats, duration, speech_sec)
        
        row = {
            "filename": tc["filename"],
            "domain": tc["domain"],
            "condition": tc["condition"],
            "speaker": tc["speaker"],
            "reference": tc["ref"],
            "predicted": pred,
            "wer": wer_stats["wer"],
            "subs": wer_stats["subs"],
            "dels": wer_stats["dels"],
            "inss": wer_stats["inss"],
            "duration_s": round(duration, 2),
            "speech_s": round(speech_sec, 2),
            "latency_ms": round(latency_ms, 1),
            "error_category": err_cat,
        }
        results.append(row)
        print(f"[{tc['domain']}] WER: {wer_stats['wer']*100:.1f}% | Ref: '{tc['ref']}' -> Pred: '{pred}' ({latency_ms:.0f}ms)")

    out_file = Path("baseline_benchmark_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} baseline benchmark results to {out_file}")

if __name__ == "__main__":
    run_baseline_benchmark()
