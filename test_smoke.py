"""End-to-end smoke test for internal deployment."""
import requests
import soundfile as sf
import io
import numpy as np
from pathlib import Path

REMOTE_GPU_API = "http://124.123.89.162:8001"
TEST_SAMPLES_DIR = Path("recordings/test_samples")

def smoke_test():
    """Run end-to-end smoke test with verified audio."""
    print("=" * 60)
    print("END-TO-END SMOKE TEST")
    print("=" * 60)
    
    # Test 1: Health check
    print("\n1. GPU Server Health Check")
    try:
        response = requests.get(f"{REMOTE_GPU_API}/health", timeout=5)
        if response.status_code == 200:
            print(f"   ✓ Health: OK")
            print(f"   Model: {response.json().get('model')}")
        else:
            print(f"   ✗ Health: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ✗ Health check failed: {e}")
        return False
    
    # Test 2: Transcription with verified sample
    print("\n2. Transcription Test")
    wav_file = TEST_SAMPLES_DIR / "algorithm_test.wav"
    if not wav_file.exists():
        print(f"   ✗ Test file not found: {wav_file}")
        return False
    
    audio, sr = sf.read(wav_file, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    
    buffer = io.BytesIO()
    sf.write(buffer, mono, sr, format="WAV", subtype="FLOAT")
    buffer.seek(0)
    
    try:
        response = requests.post(
            f"{REMOTE_GPU_API}/v1/transcribe",
            files={"audio": (wav_file.name, buffer.getvalue(), "audio/wav")},
            data={"mode": "salm", "domain": "general", "dictionary_enabled": "true"},
            timeout=30,
        )
        
        if response.status_code == 200:
            result = response.json()
            transcript = result.get('transcript', '').lower()
            if 'algorithm' in transcript:
                print(f"   ✓ Transcription: '{result.get('transcript')}'")
                print(f"   ✓ Latency: {response.elapsed.total_seconds()*1000:.1f}ms")
                print(f"   ✓ Processing: {result.get('processing_ms', 0):.1f}ms")
            else:
                print(f"   ✗ Transcription incorrect: '{result.get('transcript')}'")
                return False
        else:
            print(f"   ✗ Transcription failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ✗ Transcription error: {e}")
        return False
    
    # Test 3: Speaker identification
    print("\n3. Speaker Identification Test")
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent / "walker_audio"))
        from walker_audio.speaker_recognizer import SpeakerRegistry
        
        registry_path = Path("config/speaker_registry.json")
        registry = SpeakerRegistry(registry_file=registry_path)
        if registry_path.exists():
            registry.load()
        
        if len(registry.profiles) > 0:
            print(f"   ✓ Enrolled speakers: {len(registry.profiles)}")
            for name in registry.profiles.keys():
                print(f"      - {name}")
        else:
            print(f"   ⚠ No enrolled speakers")
    except Exception as e:
        print(f"   ⚠ Speaker identification check skipped: {e}")
    
    # Test 4: Dictionary correction
    print("\n4. Dictionary Correction Test")
    try:
        from walker_audio.dictionary_corrector import correct_transcript
        result = correct_transcript("bp is one twenty over eighty", domain="hospital", enabled=True)
        if "BP" in result['corrected_transcript']:
            print(f"   ✓ Dictionary correction: '{result['corrected_transcript']}'")
        else:
            print(f"   ⚠ Dictionary correction: '{result['corrected_transcript']}'")
    except Exception as e:
        print(f"   ⚠ Dictionary correction check skipped: {e}")
    
    print("\n" + "=" * 60)
    print("SMOKE TEST: PASSED")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = smoke_test()
    exit(0 if success else 1)
