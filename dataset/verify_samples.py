"""Dataset Sample Harvester & Verification Tool.

Separates PREDICTED / UNVERIFIED live inference data from VERIFIED training data.

Workflow:
    LIVE INFERENCE -> dataset/harvested/samples.jsonl (status: 'predicted')
         |
    dataset/verify_samples.py review / verify
         |
    dataset/manifests/manifest_verified.json (status: 'verified')
         |
    Periodic NeMo LoRA Fine-Tuning
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any


def load_harvested_samples(harvest_file: Path) -> list[dict[str, Any]]:
    if not harvest_file.exists():
        return []
    samples = []
    with harvest_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def verify_sample(
    sample: dict[str, Any],
    ground_truth: str,
    reviewer: str = "human_curator",
) -> dict[str, Any]:
    """Transitions a sample from status='predicted' to status='verified'."""
    verified = dict(sample)
    verified["verified_transcript"] = ground_truth.strip()
    verified["status"] = "verified"
    verified["reviewer"] = reviewer
    return verified


def export_verified_manifest(
    verified_samples: list[dict[str, Any]],
    output_manifest: Path,
) -> int:
    """Exports verified samples in exact NeMo SALM training manifest format."""
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_manifest.open("w", encoding="utf-8") as out:
        for s in verified_samples:
            if s.get("status") != "verified":
                continue
            wav_path = s.get("audio_filepath")
            duration = s.get("duration", 1.0)
            text = s.get("verified_transcript") or s.get("expected") or s.get("predicted_transcript")
            domain = s.get("domain", "general")
            speaker_id = s.get("speaker_id", "unknown")

            # NeMo SALM conversational multi-modal entry
            entry = {
                "audio_filepath": wav_path,
                "duration": duration,
                "text": text,
                "conversations": [
                    {
                        "role": "user",
                        "value": f"Transcribe the following: <audio>\n",
                        "audio": wav_path,
                    },
                    {
                        "role": "assistant",
                        "value": text,
                    },
                ],
                "domain": domain,
                "speaker": speaker_id,
                "status": "verified",
            }
            out.write(json.dumps(entry) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify harvested audio samples for training.")
    parser.add_argument("--harvest-dir", type=str, default="dataset/harvested", help="Harvest directory")
    parser.add_argument("--output-manifest", type=str, default="dataset/manifests/manifest_verified.json", help="Output verified manifest")
    parser.add_argument("--auto-verify-expected", action="store_true", help="Auto-verify when expected ground truth is present in metadata")
    args = parser.parse_args()

    harvest_file = Path(args.harvest_dir, "samples.jsonl")
    samples = load_harvested_samples(harvest_file)
    print(f">> Loaded {len(samples)} harvested samples from {harvest_file}")

    verified_list = []
    for s in samples:
        if s.get("status") == "verified":
            verified_list.append(s)
        elif args.auto_verify_expected and s.get("expected"):
            v = verify_sample(s, s["expected"], reviewer="benchmark_ground_truth")
            verified_list.append(v)

    exported_count = export_verified_manifest(verified_list, Path(args.output_manifest))
    print(f">> Successfully exported {exported_count} verified samples to {args.output_manifest}")


if __name__ == "__main__":
    main()
