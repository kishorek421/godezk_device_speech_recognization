"""Analyze speaker identification issues and improve enrollment."""
import soundfile as sf
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "walker_audio"))
from walker_audio.speaker_recognizer import SpeakerRegistry, extract_voiceprint_embedding, extract_biometrics

RECORDINGS_DIR = Path("recordings")

def analyze_audio(file_path):
    """Analyze audio characteristics."""
    data, sr = sf.read(file_path, dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    duration = len(mono) / sr
    peak = float(np.max(np.abs(mono)))
    rms = float(np.sqrt(np.mean(mono**2)))
    
    # Extract biometrics
    f0, f1_f2 = extract_biometrics(mono, sr)
    
    # Extract embedding
    emb = extract_voiceprint_embedding(mono, sr)
    emb_norm = np.linalg.norm(emb)
    
    return {
        "duration_sec": duration,
        "peak": peak,
        "rms": rms,
        "peak_db": 20 * np.log10(peak + 1e-8),
        "rms_db": 20 * np.log10(rms + 1e-8),
        "f0": f0,
        "f1_f2": f1_f2,
        "embedding_norm": emb_norm,
        "sample_rate": sr,
    }

def main():
    print("=" * 60)
    print("SPEAKER IDENTIFICATION ANALYSIS")
    print("=" * 60)
    
    # Load current registry
    registry_path = Path("config/speaker_registry.json")
    registry = SpeakerRegistry(registry_file=registry_path)
    if registry_path.exists():
        registry.load()
    
    print(f"\nCurrent enrollment: {len(registry.profiles)} speaker(s)")
    for name, prof in registry.profiles.items():
        print(f"\n  {name}:")
        print(f"    Samples: {len(prof['samples'])}")
        print(f"    F0: {prof.get('f0', 'N/A')}Hz")
        print(f"    F1/F2: {prof.get('f1_f2', 'N/A')}")
        print(f"    Centroid norm: {np.linalg.norm(prof['centroid']):.4f}")
    
    # Analyze available enrollment files
    print(f"\nAvailable enrollment files:")
    enrollment_files = [
        "enrolled_mega_short.wav",
        "enrolled_mega_sec5_15.wav",
        "enrolled_mega_1791971449667.wav",
        "enrolled_mega_1791972644153.wav",
    ]
    
    for fname in enrollment_files:
        fpath = RECORDINGS_DIR / fname
        if fpath.exists():
            meta = analyze_audio(fpath)
            print(f"\n  {fname}:")
            print(f"    Duration: {meta['duration_sec']:.2f}s")
            print(f"    F0: {meta['f0']:.1f}Hz")
            print(f"    Peak: {meta['peak']:.3f}")
            print(f"    RMS: {meta['rms']:.3f}")
            print(f"    Embedding norm: {meta['embedding_norm']:.4f}")
    
    # Recommendation: Multi-sample enrollment
    print("\n" + "=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)
    print("Current enrollment uses only 1 sample (weak).")
    print("Recommended: Enroll with 3-5 samples from the same speaker.")
    print("This improves robustness and reduces false positives.")
    
    # Re-enroll with multiple samples
    print("\nRe-enrolling with multiple samples...")
    
    # Clear existing enrollment
    if "megak" in registry.profiles:
        registry.delete("megak")
    
    # Enroll with multiple samples
    samples_to_use = [
        "enrolled_mega_short.wav",
        "enrolled_mega_sec5_15.wav",
        "enrolled_mega_1791971449667.wav",
    ]
    
    for fname in samples_to_use:
        fpath = RECORDINGS_DIR / fname
        if fpath.exists():
            print(f"\n  Enrolling: {fname}")
            meta = analyze_audio(fpath)
            data, sr = sf.read(fpath, dtype="float32", always_2d=True)
            mono = data.mean(axis=1)
            result = registry.enroll("megak", mono, sr, duration_sec=meta["duration_sec"])
            print(f"    Sample count: {result['sample_count']}")
            print(f"    Total enrolled: {result['total_enrolled']}")
            print(f"    F0: {result.get('f0', 'N/A')}Hz")
    
    registry.save()
    
    print("\n" + "=" * 60)
    print("FINAL ENROLLMENT")
    print("=" * 60)
    registry.load()
    for name, prof in registry.profiles.items():
        print(f"\n  {name}:")
        print(f"    Samples: {len(prof['samples'])}")
        print(f"    F0: {prof.get('f0', 'N/A')}Hz")
        print(f"    F1/F2: {prof.get('f1_f2', 'N/A')}")
        print(f"    Centroid norm: {np.linalg.norm(prof['centroid']):.4f}")

if __name__ == "__main__":
    main()
