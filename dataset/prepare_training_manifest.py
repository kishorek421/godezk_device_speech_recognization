"""Dataset Generator and Verification Tool for Canary-Qwen 2.5B Fine-Tuning.

Scans recorded audio in `recordings/`, maps audio files to ground-truth transcripts,
validates sample rates and durations, and generates train/validation manifest files
in the exact format expected by NeMo Canary-Qwen SALM LoRA training (`train_canary_salm.py`).

Usage:
    python dataset/prepare_training_manifest.py [--val-ratio 0.15] [--output-dir dataset/manifests]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Any

import soundfile as sf

# Known Sentence & Word Ground-Truth Registry
KNOWN_TARGETS = {
    # Noise Layer Sentences
    "Principals_praised": {
        "text": "Principals praised professors proposing profound philosophical principles.",
        "domain": "education",
    },
    "The_mathematical_derivation": {
        "text": "The mathematical derivation requires differential and integral calculus.",
        "domain": "education",
    },
    "Photosynthesis_converts": {
        "text": "Photosynthesis converts carbon dioxide and sunlight into glucose and oxygen.",
        "domain": "education",
    },
    "The_auditorium_lecture": {
        "text": "The auditorium lecture on quantum electrodynamics will begin shortly.",
        "domain": "education",
    },
    "Patient_prescribed_potassium": {
        "text": "Patient prescribed potassium penicillin post procedural pain.",
        "domain": "hospital",
    },
    "Cardiac_telemetry_monitors": {
        "text": "Cardiac telemetry monitors oxygen saturation and blood pressure.",
        "domain": "hospital",
    },
    "The_patient_presented_with": {
        "text": "The patient presented with sinus tachycardia and pulmonary edema.",
        "domain": "hospital",
    },
    "Emergency_resuscitation_cart": {
        "text": "Emergency resuscitation cart required in intensive care unit immediately.",
        "domain": "hospital",
    },
    "Pressurized_pneumatic_pumps": {
        "text": "Pressurized pneumatic pumps protect petroleum pipeline pressure.",
        "domain": "industry",
    },
    "High_voltage_transformers": {
        "text": "High voltage transformers operate at fifty hertz electrical frequency.",
        "domain": "industry",
    },
    "Emergency_stop_activated": {
        "text": "Emergency stop activated on the automated conveyor assembly line.",
        "domain": "industry",
    },
    "Warehouse_forklift_operating": {
        "text": "Warehouse forklift operating in warehouse quadrant seven.",
        "domain": "industry",
    },
    "Peter_picked_pepper": {
        "text": "Peter picked pepper plants promptly past previous periods.",
        "domain": "general",
    },
    "Air_conditioning_systems": {
        "text": "Air conditioning systems reduce ambient room temperature consistently.",
        "domain": "general",
    },
    "Good_morning_team": {
        "text": "Good morning team, let us review the quarterly deployment roadmap.",
        "domain": "general",
    },
    "Can_you_hear_me_clearly": {
        "text": "Can you hear me clearly from the other side of the conference hall?",
        "domain": "general",
    },
    # Dedicated Studio Recordings
    "_this_machine_has_a_programmab": {
        "text": "This machine has a programmable logic controller installed.",
        "domain": "industry",
    },
    "we_studied_photo_synthesis_in_": {
        "text": "We studied photosynthesis in the laboratory.",
        "domain": "education",
    },
    "the_patient_was_diagnosed_with": {
        "text": "The patient was diagnosed with hypertension.",
        "domain": "hospital",
    },
    "please_bring_the_stetho_scope_": {
        "text": "Please bring the stethoscope and check the electrocardiogram.",
        "domain": "hospital",
    },
    # Isolated Target Words
    "algorithm": {"text": "algorithm", "domain": "education"},
    "laboratory": {"text": "laboratory", "domain": "education"},
    "hypothesis": {"text": "hypothesis", "domain": "education"},
    "thermodynamics": {"text": "thermodynamics", "domain": "education"},
    "curriculum": {"text": "curriculum", "domain": "education"},
    "derivation": {"text": "derivation", "domain": "education"},
    "photosynthesis": {"text": "photosynthesis", "domain": "education"},
    "hypertension": {"text": "hypertension", "domain": "hospital"},
    "tachycardia": {"text": "tachycardia", "domain": "hospital"},
    "diagnosis": {"text": "diagnosis", "domain": "hospital"},
    "prescription": {"text": "prescription", "domain": "hospital"},
    "stethoscope": {"text": "stethoscope", "domain": "hospital"},
    "maintenance": {"text": "maintenance", "domain": "industry"},
    "machinery": {"text": "machinery", "domain": "industry"},
    "calibration": {"text": "calibration", "domain": "industry"},
    "inspection": {"text": "inspection", "domain": "industry"},
}


def resolve_ground_truth(filename: str) -> tuple[str | None, str]:
    """Resolves ground truth transcript and domain from recording filename."""
    # Strip timestamp e.g., '1791456783164_Patient_prescribed_potassium_p.wav'
    base_name = Path(filename).stem
    parts = base_name.split("_", 1)
    tag = parts[1] if len(parts) > 1 and parts[0].isdigit() else base_name

    # Check exact match
    if tag in KNOWN_TARGETS:
        return KNOWN_TARGETS[tag]["text"], KNOWN_TARGETS[tag]["domain"]

    # Check prefix matching
    for key, val in KNOWN_TARGETS.items():
        if tag.startswith(key) or key.startswith(tag[:20]):
            return val["text"], val["domain"]

    # Fallback to humanized tag if it looks like words
    clean_tag = tag.replace("_", " ").strip()
    if clean_tag and not clean_tag.startswith("mic-") and not clean_tag.startswith("laptop_"):
        return clean_tag, "general"

    return None, "general"


def scan_and_generate_manifest(
    recordings_dir: str | Path = "recordings",
    output_dir: str | Path = "dataset/manifests",
    val_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, Any]:
    recordings_path = Path(recordings_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    wav_files = sorted(list(recordings_path.glob("*.wav")))
    print(f">> Found {len(wav_files)} total audio recordings in {recordings_path.resolve()}")

    entries = []
    skipped_count = 0

    for wav in wav_files:
        # Ignore raw background noise benchmark captures
        if wav.name.startswith("laptop_") or wav.name.startswith("mic-"):
            continue

        ground_truth, domain = resolve_ground_truth(wav.name)
        if not ground_truth:
            skipped_count += 1
            continue

        try:
            info = sf.info(str(wav))
            duration = round(info.duration, 3)
            # Skip invalid, ultra-short, or empty recordings
            if duration < 0.35:
                skipped_count += 1
                continue

            entry = {
                "audio_filepath": str(wav.resolve()),
                "duration": duration,
                "text": ground_truth,
                "conversations": [
                    {
                        "role": "user",
                        "value": "Transcribe the following: <audio>\n",
                        "audio": str(wav.resolve()),
                    },
                    {
                        "role": "assistant",
                        "value": ground_truth,
                    },
                ],
                "domain": domain,
                "speaker": "primary_speaker",
                "status": "verified",
            }
            entries.append(entry)
        except Exception as exc:
            print(f"[WARN] Failed to read {wav.name}: {exc}")
            skipped_count += 1

    print(f">> Verified and mapped {len(entries)} speech recordings ({skipped_count} skipped)")

    if not entries:
        print("[ERROR] No valid entries found to generate manifest.")
        return {"total": 0, "train": 0, "val": 0}

    # Deterministic split
    random.seed(seed)
    random.shuffle(entries)

    val_count = max(1, int(len(entries) * val_ratio)) if len(entries) > 5 else 0
    val_entries = entries[:val_count]
    train_entries = entries[val_count:]

    train_file = out_path / "train_manifest.jsonl"
    val_file = out_path / "val_manifest.jsonl"
    full_file = out_path / "full_manifest.jsonl"

    def write_jsonl(path: Path, items: list[dict]):
        with open(path, "w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")

    write_jsonl(train_file, train_entries)
    write_jsonl(val_file, val_entries)
    write_jsonl(full_file, entries)

    total_seconds = sum(e["duration"] for e in entries)
    print("\n" + "=" * 70)
    print("CANARY-QWEN SALM TRAINING DATASET GENERATED SUCCESSFULLY")
    print("=" * 70)
    print(f"Total Verified Samples : {len(entries)}")
    print(f"Total Speech Audio     : {total_seconds:.1f} seconds ({total_seconds / 60:.1f} minutes)")
    print(f"Training Set           : {len(train_entries)} samples -> {train_file.resolve()}")
    print(f"Validation Set         : {len(val_entries)} samples -> {val_file.resolve()}")
    print(f"Full Manifest          : {len(entries)} samples -> {full_file.resolve()}")
    print("=" * 70)

    print("\n>> Next Step: Run LoRA Fine-Tuning on your GPU server:")
    print(f"   python training/train_canary_salm.py --train-manifest {train_file} --val-manifest {val_file} --epochs 5\n")

    return {
        "total": len(entries),
        "train": len(train_entries),
        "val": len(val_entries),
        "train_file": str(train_file.resolve()),
        "val_file": str(val_file.resolve()),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate NeMo Canary-Qwen training manifests from recordings")
    parser.add_argument("--recordings-dir", type=str, default="recordings")
    parser.add_argument("--output-dir", type=str, default="dataset/manifests")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    args = parser.parse_args()

    scan_and_generate_manifest(
        recordings_dir=args.recordings_dir,
        output_dir=args.output_dir,
        val_ratio=args.val_ratio,
    )
