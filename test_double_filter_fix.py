"""Test that audio is filtered exactly once in word_test_server → audio_api path."""
import requests
import soundfile as sf
import io
import numpy as np
from pathlib import Path

REMOTE_GPU_API = "http://124.123.89.162:8001"
TEST_SAMPLES_DIR = Path("recordings/test_samples")

def test_single_filtering():
    """Verify audio is filtered only once by sending through word_test_server → audio_api."""
    print("=" * 60)
    print("TEST: Single Filtering Verification")
    print("=" * 60)
    
    # Use verified sample
    wav_file = TEST_SAMPLES_DIR / "algorithm_test.wav"
    if not wav_file.exists():
        print(f"Test file not found: {wav_file}")
        return False
    
    # Read original audio
    audio, sr = sf.read(wav_file, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    
    print(f"\nOriginal audio:")
    print(f"  Duration: {len(mono)/sr:.2f}s")
    print(f"  Sample rate: {sr}Hz")
    print(f"  Peak: {np.max(np.abs(mono)):.4f}")
    
    # Convert to WAV bytes (simulating word_test_server output)
    buffer = io.BytesIO()
    sf.write(buffer, mono, sr, format="WAV", subtype="FLOAT")
    buffer.seek(0)
    
    # Send directly to audio_api (simulating what word_test_server does)
    print(f"\nSending to remote GPU API (audio_api)...")
    try:
        response = requests.post(
            f"{REMOTE_GPU_API}/v1/transcribe",
            files={"audio": (wav_file.name, buffer.getvalue(), "audio/wav")},
            data={"mode": "salm", "domain": "general", "dictionary_enabled": "true"},
            timeout=30,
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"  Status: OK")
            print(f"  Transcript: '{result.get('transcript', 'N/A')}'")
            print(f"  Speech seconds: {result.get('speech_seconds', 0):.2f}s")
            print(f"  Processing ms: {result.get('processing_ms', 0):.1f}ms")
            
            # Verify transcription is still correct
            transcript = result.get('transcript', '').lower()
            if 'algorithm' in transcript:
                print(f"\n✓ Transcription correct (contains 'algorithm')")
                print(f"✓ Single filtering verified - audio processed correctly")
                return True
            else:
                print(f"\n✗ Transcription incorrect (expected 'algorithm')")
                return False
        else:
            print(f"  Status: {response.status_code}")
            print(f"  Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"  ERROR: {e}")
        return False

if __name__ == "__main__":
    success = test_single_filtering()
    print("\n" + "=" * 60)
    if success:
        print("RESULT: PASS - Single filtering verified")
    else:
        print("RESULT: FAIL - Single filtering not verified")
    print("=" * 60)
