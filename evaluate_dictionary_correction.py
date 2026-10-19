"""
Comparative Evaluation: Dictionary-Based Transcript Correction
Evaluates speech transcripts across domains with dictionary_enabled=False vs dictionary_enabled=True.
Computes Word Error Rate (WER) before and after against verified reference texts.
"""

import csv
import json
import re
from pathlib import Path
from typing import List, Dict, Any
from walker_audio.dictionary_corrector import correct_transcript, load_domain_dictionary


def calculate_wer(reference: str, hypothesis: str) -> float:
    """Computes Word Error Rate using dynamic programming Levenshtein distance."""
    ref_words = re.findall(r"\b\w+\b", reference.lower())
    hyp_words = re.findall(r"\b\w+\b", hypothesis.lower())

    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j

    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                substitution = d[i - 1][j - 1] + 1
                insertion = d[i][j - 1] + 1
                deletion = d[i - 1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    return d[len(ref_words)][len(hyp_words)] / len(ref_words)


# Benchmark evaluation corpus with realistic speech recognition acoustic outputs
# representing challenging domain terminology before dictionary post-processing
TEST_CORPUS = [
    {
        "id": "MED_01",
        "domain": "medical",
        "reference": "The patient has hypertension and takes metformin.",
        "asr_output": "The patient has hyper tension and takes metforman.",
        "notes": "Phonetic split & vowel substitution in clinical terms",
    },
    {
        "id": "MED_02",
        "domain": "medical",
        "reference": "Doctor administered paracetamol for acute fever and severe hypertension.",
        "asr_output": "Doctor administered paracetemol for acute fever and severe hyper tension.",
        "notes": "Common brand/generic spelling and multi-word split",
    },
    {
        "id": "EDU_01",
        "domain": "education",
        "reference": "Photosynthesis converts light into chemical energy while thermodynamics governs heat.",
        "asr_output": "photosynthe sis converts light into chemical energy while thermodynamiks governs heat.",
        "notes": "Syllabic space split and phonetic k substitution",
    },
    {
        "id": "EDU_02",
        "domain": "education",
        "reference": "The modern semiconductor revolutionized computer science education.",
        "asr_output": "The modern semiconducter revolutionized computer science education.",
        "notes": "Vowel ending confusion (-er vs -or)",
    },
    {
        "id": "IND_01",
        "domain": "industry",
        "reference": "The plant manager inspected the centrifuge and the PLC in the control room.",
        "asr_output": "The plant manager inspected the centrifuj and the programmable logic controller in the control room.",
        "notes": "Colloquial expansion to acronym & phonetic spelling",
    },
    {
        "id": "GEN_01",
        "domain": "general",
        "reference": "The weather forecast predicts light rain and clear skies tomorrow afternoon.",
        "asr_output": "The weather forecast predicts light rain and clear skies tomorrow afternoon.",
        "notes": "General baseline: no domain terms should be falsely corrupted",
    },
]


def run_evaluation():
    print("=" * 85)
    print("DICTIONARY-BASED TRANSCRIPT CORRECTION: COMPARATIVE BENCHMARK")
    print("=" * 85)

    results: List[Dict[str, Any]] = []

    for item in TEST_CORPUS:
        # Run 1: Dictionary Disabled (Raw ASR Output)
        res_disabled = correct_transcript(item["asr_output"], domain=item["domain"], enabled=False)
        wer_before = calculate_wer(item["reference"], res_disabled["corrected_transcript"])

        # Run 2: Dictionary Enabled
        res_enabled = correct_transcript(item["asr_output"], domain=item["domain"], enabled=True)
        wer_after = calculate_wer(item["reference"], res_enabled["corrected_transcript"])

        corrections_str = "; ".join(
            [f"'{c['original']}' -> '{c['replacement']}'" for c in res_enabled["corrections"]]
        ) or "None"

        wer_reduction = (wer_before - wer_after) * 100

        results.append({
            "id": item["id"],
            "domain": item["domain"],
            "reference": item["reference"],
            "original_transcript": res_disabled["corrected_transcript"],
            "corrected_transcript": res_enabled["corrected_transcript"],
            "corrections": corrections_str,
            "wer_before": wer_before,
            "wer_after": wer_after,
            "wer_reduction": wer_reduction,
            "notes": item["notes"],
        })

    # Display Table
    print(f"{'ID':<7} | {'Domain':<10} | {'WER Raw':<8} | {'WER Dict':<8} | {'WER Reduc':<9} | Corrections Applied")
    print("-" * 85)
    for r in results:
        print(f"{r['id']:<7} | {r['domain']:<10} | {r['wer_before']*100:6.1f}%  | {r['wer_after']*100:6.1f}%  | {r['wer_reduction']:+6.1f}%   | {r['corrections']}")

    # Overall Summary
    avg_wer_before = sum(r["wer_before"] for r in results) / len(results) * 100
    avg_wer_after = sum(r["wer_after"] for r in results) / len(results) * 100
    print("-" * 85)
    print(f"OVERALL AVERAGE WER (Raw ASR):          {avg_wer_before:5.1f}%")
    print(f"OVERALL AVERAGE WER (With Dictionary):  {avg_wer_after:5.1f}%")
    print(f"NET WER RELATIVE IMPROVEMENT:           {(avg_wer_before - avg_wer_after):+5.1f}% absolute reduction")
    print("=" * 85)

    # Save to CSV
    csv_path = Path("comparison_dictionary_correction.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "id", "domain", "reference", "original_transcript", "corrected_transcript",
            "corrections", "wer_before", "wer_after", "wer_reduction", "notes"
        ])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nDetailed comparison report saved to: {csv_path.resolve()}")


if __name__ == "__main__":
    run_evaluation()
