"""Standalone Comparison Script: Dictionary-Based Correction Engine Evaluation.

NOTE: The input transcripts in this evaluation script are SYNTHETIC test samples,
designed to verify the correction engine's behavior across domains and safety constraints.
They are NOT evaluated against a verified real-world speech dataset.

Outputs:
- Console table comparing before and after transcripts
- Summary of proposed, applied, and rejected corrections
- Exported CSV file: dictionary_comparison_results.csv
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

# Add audio_api to sys.path
audio_api_path = str(Path(__file__).parent / "audio_api")
if audio_api_path not in sys.path:
    sys.path.insert(0, audio_api_path)

from dictionary_engine import correct_text

# Synthetic test cases covering various domains, known errors, correct text, and edge cases
SYNTHETIC_SAMPLES = [
    {
        "id": "SYNTH-001",
        "domain": "education",
        "asr_output": "The curiculum is updated.",
        "description": "Known misspelling (education)",
    },
    {
        "id": "SYNTH-002",
        "domain": "education",
        "asr_output": "The curriculum is updated.",
        "description": "Already correct text (education)",
    },
    {
        "id": "SYNTH-003",
        "domain": "education",
        "asr_output": "Please explain the derivasion carefully.",
        "description": "Known ASR error (derivation)",
    },
    {
        "id": "SYNTH-004",
        "domain": "education",
        "asr_output": "We observed photo synthesis under sunlight.",
        "description": "Multi-word split error (photo synthesis)",
    },
    {
        "id": "SYNTH-005",
        "domain": "hospital",
        "asr_output": "The doctor wrote a perscription for hyper tension.",
        "description": "Multiple errors in medical/hospital domain",
    },
    {
        "id": "SYNTH-006",
        "domain": "hospital",
        "asr_output": "The patient diagnosis was confirmed by radiology.",
        "description": "Already correct medical terminology",
    },
    {
        "id": "SYNTH-007",
        "domain": "industry",
        "asr_output": "Scheduled maintainance of the machinary is ongoing.",
        "description": "Industry domain spelling variants",
    },
    {
        "id": "SYNTH-008",
        "domain": "industry",
        "asr_output": "Routine inspection and calibration completed.",
        "description": "Already correct industrial terms",
    },
    {
        "id": "SYNTH-009",
        "domain": "general",
        "asr_output": "The meeting is tomorrow at nine.",
        "description": "Unrelated general domain text",
    },
    {
        "id": "SYNTH-010",
        "domain": "education",
        "asr_output": "CURICULUM renewal policy announced!",
        "description": "Uppercase and punctuation preservation",
    },
    {
        "id": "SYNTH-011",
        "domain": "education",
        "asr_output": "Recuriculum testing does not apply here.",
        "description": "Word boundary protection (substring recurrence)",
    },
    {
        "id": "SYNTH-012",
        "domain": "astronomy",
        "asr_output": "The curiculum of the observatory is set.",
        "description": "Unsupported domain error policy",
    },
]


def run_comparison(samples: list[dict[str, str]], output_csv: str = "dictionary_comparison_results.csv"):
    print("=" * 90)
    print("DICTIONARY-BASED TRANSCRIPT CORRECTION ENGINE — SYNTHETIC BENCHMARK EVALUATION")
    print("NOTE: All samples below are SYNTHETIC test cases for algorithm verification.")
    print("=" * 90)

    results = []
    total_transcripts = len(samples)
    changed_count = 0
    total_proposed = 0
    total_applied = 0
    total_rejected = 0

    csv_rows = []

    for item in samples:
        sample_id = item["id"]
        domain = item["domain"]
        original = item["asr_output"]
        desc = item["description"]

        corrected, records = correct_text(original, domain=domain)
        has_changed = original != corrected

        if has_changed:
            changed_count += 1

        proposed_count = len(records)
        applied_in_sample = sum(1 for r in records if r["applied"])
        rejected_in_sample = sum(1 for r in records if not r["applied"])

        total_proposed += proposed_count
        total_applied += applied_in_sample
        total_rejected += rejected_in_sample

        corrections_summary = "; ".join(
            f"'{r['original']}' -> '{r['replacement']}' (applied={r['applied']}, reason={r['reason']})"
            for r in records
        ) if records else "None"

        print(f"\n[{sample_id}] Domain: {domain} | {desc}")
        print(f"  Original : \"{original}\"")
        print(f"  Corrected: \"{corrected}\"")
        print(f"  Changed  : {'YES' if has_changed else 'NO'}")
        if records:
            print(f"  Details  : {corrections_summary}")

        csv_rows.append({
            "sample_id": sample_id,
            "sample_type": "SYNTHETIC",
            "domain": domain,
            "description": desc,
            "original_asr_transcript": original,
            "corrected_transcript": corrected,
            "is_changed": has_changed,
            "num_proposed_corrections": proposed_count,
            "num_applied_corrections": applied_in_sample,
            "num_rejected_corrections": rejected_in_sample,
            "correction_details": corrections_summary,
        })

    # Write CSV
    csv_path = Path(output_csv)
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "sample_id",
            "sample_type",
            "domain",
            "description",
            "original_asr_transcript",
            "corrected_transcript",
            "is_changed",
            "num_proposed_corrections",
            "num_applied_corrections",
            "num_rejected_corrections",
            "correction_details",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print("\n" + "=" * 90)
    print("EVALUATION SUMMARY")
    print(f"- Total synthetic input transcripts: {total_transcripts}")
    print(f"- Transcripts changed               : {changed_count} ({changed_count/total_transcripts*100:.1f}%)")
    print(f"- Transcripts unchanged             : {total_transcripts - changed_count}")
    print(f"- Total proposed corrections        : {total_proposed}")
    print(f"- Total applied corrections         : {total_applied}")
    print(f"- Total rejected/ambiguous/unknown  : {total_rejected}")
    print(f"- CSV export path                   : {csv_path.resolve()}")
    print("=" * 90)

    return {
        "total_transcripts": total_transcripts,
        "changed_count": changed_count,
        "total_proposed": total_proposed,
        "total_applied": total_applied,
        "total_rejected": total_rejected,
        "csv_path": str(csv_path.resolve()),
    }


if __name__ == "__main__":
    run_comparison(SYNTHETIC_SAMPLES)
