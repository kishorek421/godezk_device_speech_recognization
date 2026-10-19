"""Test speaker identification pipeline with existing recordings."""
import sys
from pathlib import Path

# Add walker_audio to path
sys.path.insert(0, str(Path(__file__).parent / "walker_audio"))

import numpy as np
import soundfile as sf
from walker_audio.speaker_recognizer import get_speaker_registry, extract_voiceprint_embedding

def test_speaker_identification():
    """Test speaker identification with existing enrolled speakers."""
    
    registry = get_speaker_registry()
    
    print("=" * 80)
    print("SPEAKER IDENTIFICATION TEST")
    print("=" * 80)
    
    # Check enrolled speakers
    summary = registry.get_enrolled_summary()
    print(f"\nEnrolled speakers: {len(summary)}")
    for profile in summary:
        print(f"  - {profile['name']}: {profile['sample_count']} samples, F0={profile['f0']:.1f}Hz")
    
    if not summary:
        print("\nNo enrolled speakers found. Testing with empty registry...")
        # Test with empty registry
        test_audio = np.random.randn(16000).astype(np.float32) * 0.1
        result = registry.identify(test_audio, sample_rate=16000)
        print(f"Empty registry result: {result}")
        return
    
    # Test identification with one of the enrolled speaker recordings
    recordings_dir = Path("recordings")
    if not recordings_dir.exists():
        print(f"\nRecordings directory not found: {recordings_dir}")
        return
    
    # Find enrolled speaker recordings
    enrolled_files = []
    for profile in summary:
        name = profile['name']
        # Look for enrolled files
        for wav_file in recordings_dir.glob(f"enrolled_{name}_*.wav"):
            enrolled_files.append((name, wav_file))
    
    if not enrolled_files:
        print("\nNo enrolled speaker recordings found in recordings directory")
        return
    
    print(f"\nFound {len(enrolled_files)} enrolled speaker recordings")
    
    # Test identification on enrolled files
    print("\n" + "=" * 80)
    print("IDENTIFICATION TESTS")
    print("=" * 80)
    
    for name, wav_file in enrolled_files[:3]:  # Test first 3
        try:
            audio, sr = sf.read(wav_file, dtype="float32", always_2d=True)
            mono = audio.mean(axis=1)
            
            print(f"\nTesting: {wav_file.name}")
            print(f"  Expected speaker: {name}")
            print(f"  Duration: {len(mono)/sr:.2f}s")
            print(f"  Peak: {np.max(np.abs(mono)):.4f}")
            print(f"  RMS: {np.sqrt(np.mean(mono**2)):.4f}")
            
            # Test identification
            result = registry.identify(mono, sample_rate=sr)
            print(f"  Identified: {result['speaker_id']}")
            print(f"  Confidence: {result['speaker_confidence']:.4f}")
            print(f"  Status: {result['speaker_status']}")
            if 'reason' in result:
                print(f"  Reason: {result['reason']}")
            
            # Test embedding extraction
            emb = extract_voiceprint_embedding(mono, sr)
            print(f"  Embedding norm: {np.linalg.norm(emb):.4f}")
            
        except Exception as e:
            print(f"  Error: {e}")
    
    # Test with silence
    print("\n" + "=" * 80)
    print("SILENCE TEST")
    print("=" * 80)
    silence = np.zeros(16000, dtype=np.float32)
    result = registry.identify(silence, sample_rate=16000)
    print(f"Silence result: {result['speaker_status']}")
    if 'reason' in result:
        print(f"Reason: {result['reason']}")
    
    # Test with very quiet audio
    print("\n" + "=" * 80)
    print("QUIET AUDIO TEST")
    print("=" * 80)
    quiet = np.random.randn(16000).astype(np.float32) * 0.001
    result = registry.identify(quiet, sample_rate=16000)
    print(f"Quiet audio result: {result['speaker_status']}")
    if 'reason' in result:
        print(f"Reason: {result['reason']}")

if __name__ == "__main__":
    test_speaker_identification()
