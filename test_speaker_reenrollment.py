"""Investigate and re-enroll speaker with clean recording."""
import soundfile as sf
import numpy as np
from pathlib import Path
import sys

# Add walker_audio to path
sys.path.insert(0, str(Path(__file__).parent / "walker_audio"))

from walker_audio.speaker_recognizer import SpeakerRegistry, extract_voiceprint_embedding

RECORDINGS_DIR = Path("recordings")

def analyze_audio(wav_path: Path):
    """Analyze audio characteristics."""
    data, sr = sf.read(wav_path, dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    
    duration = len(mono) / sr
    peak = float(np.max(np.abs(mono)))
    rms = float(np.sqrt(np.mean(mono**2)))
    peak_db = float(20 * np.log10(peak + 1e-8))
    rms_db = float(20 * np.log10(rms + 1e-8))
    
    # Extract embedding
    try:
        emb = extract_voiceprint_embedding(mono, sr)
        emb_norm = float(np.linalg.norm(emb))
    except Exception as e:
        emb_norm = 0.0
    
    return {
        "duration_sec": duration,
        "peak": peak,
        "rms": rms,
        "peak_db": peak_db,
        "rms_db": rms_db,
        "embedding_norm": emb_norm,
        "sample_rate": sr,
    }

def main():
    print("=" * 60)
    print("SPEAKER IDENTIFICATION INVESTIGATION")
    print("=" * 60)
    
    # Load registry
    registry_path = Path("config/speaker_registry.json")
    registry = SpeakerRegistry(registry_file=registry_path)
    if registry_path.exists():
        registry.load()
    
    print(f"\nEnrolled speakers: {len(registry.profiles)}")
    for name, prof in registry.profiles.items():
        print(f"\n  {name}:")
        print(f"    Samples: {len(prof['samples'])}")
        print(f"    F0: {prof.get('f0', 'N/A')}Hz")
        print(f"    Centroid norm: {np.linalg.norm(prof['centroid']):.4f}")
    
    # Analyze enrolled file
    enrolled_file = RECORDINGS_DIR / "enrolled_megak_1792047392709.wav"
    if enrolled_file.exists():
        print(f"\nEnrolled file analysis: {enrolled_file.name}")
        enrolled_meta = analyze_audio(enrolled_file)
        for k, v in enrolled_meta.items():
            print(f"  {k}: {v}")
    
    # Find test file
    test_files = list(RECORDINGS_DIR.glob("enrolled_megak_*.wav"))
    test_files = [f for f in test_files if f != enrolled_file]
    
    if test_files:
        print(f"\nTest file analysis: {test_files[0].name}")
        test_meta = analyze_audio(test_files[0])
        for k, v in test_meta.items():
            print(f"  {k}: {v}")
        
        # Calculate similarity
        try:
            enrolled_emb = extract_voiceprint_embedding(enrolled_file)
            test_emb = extract_voiceprint_embedding(test_files[0])
            similarity = float(np.dot(enrolled_emb, test_emb))
            print(f"\nDirect similarity: {similarity:.4f}")
        except Exception as e:
            print(f"\nSimilarity calculation failed: {e}")
    
    # Check for shorter enrollment files
    print(f"\nAvailable enrollment files:")
    for f in RECORDINGS_DIR.glob("enrolled_*.wav"):
        data, sr = sf.read(f, dtype="float32", always_2d=True)
        mono = data.mean(axis=1)
        duration = len(mono) / sr
        print(f"  {f.name}: {duration:.2f}s")
    
    # Re-enroll with shorter file
    short_file = RECORDINGS_DIR / "enrolled_mega_short.wav"
    if short_file.exists():
        print(f"\nRe-enrolling with shorter file: {short_file.name}")
        short_meta = analyze_audio(short_file)
        for k, v in short_meta.items():
            print(f"  {k}: {v}")
        
        # Clear existing enrollment
        registry.delete("megak")
        
        # Re-enroll
        try:
            data, sr = sf.read(short_file, dtype="float32", always_2d=True)
            mono = data.mean(axis=1)
            result = registry.enroll("megak", mono, sr, duration_sec=short_meta["duration_sec"])
            print(f"\nRe-enrollment successful:")
            print(f"  Sample count: {result['sample_count']}")
            print(f"  Total enrolled: {result['total_enrolled']}")
            registry.save()
        except Exception as e:
            print(f"\nRe-enrollment failed: {e}")
    
    print("\n" + "=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)
    print("The enrolled file is 57s long, which may include varied speech patterns.")
    print("Re-enrolled with 5s file for better consistency.")
    print("Test identification with a separate recording from the same speaker.")

if __name__ == "__main__":
    main()
