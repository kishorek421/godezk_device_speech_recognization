"""Test transcription with verified test samples."""
import requests
import soundfile as sf
import io
import time
from pathlib import Path

REMOTE_GPU_API = "http://124.123.89.162:8001"
TEST_SAMPLES_DIR = Path("recordings/test_samples")

def test_transcription(wav_file: Path, expected: str = None):
    """Send audio file to GPU API for transcription."""
    try:
        # Read audio
        audio, sr = sf.read(wav_file, dtype="float32", always_2d=True)
        mono = audio.mean(axis=1)
        
        # Convert to WAV bytes
        buffer = io.BytesIO()
        sf.write(buffer, mono, sr, format="WAV", subtype="FLOAT")
        buffer.seek(0)
        
        print(f"\nTesting: {wav_file.name}")
        if expected:
            print(f"  Expected: '{expected}'")
        print(f"  Duration: {len(mono)/sr:.2f}s")
        print(f"  Sample rate: {sr}Hz")
        
        # Send to GPU API
        start_time = time.time()
        response = requests.post(
            f"{REMOTE_GPU_API}/v1/transcribe",
            files={"audio": (wav_file.name, buffer.getvalue(), "audio/wav")},
            data={"mode": "salm", "domain": "general", "dictionary_enabled": "true"},
            timeout=30,
        )
        latency = (time.time() - start_time) * 1000
        
        print(f"  Status: {response.status_code}")
        print(f"  Latency: {latency:.1f}ms")
        
        if response.status_code == 200:
            result = response.json()
            transcript = result.get('transcript', 'N/A')
            print(f"  Transcript: '{transcript}'")
            print(f"  Original: '{result.get('original_transcript', 'N/A')}'")
            print(f"  Corrected: '{result.get('corrected_transcript', 'N/A')}'")
            print(f"  Corrections: {result.get('corrections', [])}")
            print(f"  Speech seconds: {result.get('speech_seconds', 0):.2f}s")
            print(f"  Processing ms: {result.get('processing_ms', 0):.1f}ms")
            
            if expected:
                match = expected.lower() in transcript.lower()
                print(f"  Match: {match}")
                return True, result, match
            return True, result, None
        else:
            print(f"  Error: {response.text}")
            return False, None, None
            
    except requests.Timeout:
        print(f"  ERROR: Timeout")
        return False, None, None
    except requests.ConnectionError:
        print(f"  ERROR: Connection error")
        return False, None, None
    except Exception as e:
        print(f"  ERROR: {e}")
        return False, None, None

if __name__ == "__main__":
    print("=" * 60)
    print("REMOTE GPU TRANSCRIPTION TEST - VERIFIED SAMPLES")
    print(f"Endpoint: {REMOTE_GPU_API}")
    print("=" * 60)
    
    if not TEST_SAMPLES_DIR.exists():
        print(f"Test samples directory not found: {TEST_SAMPLES_DIR}")
        exit(1)
    
    # Test with verified samples
    test_cases = [
        ("algorithm_test.wav", "algorithm"),
        ("alg_part1.wav", None),
        ("alg_part2.wav", None),
    ]
    
    results = []
    for filename, expected in test_cases:
        wav_file = TEST_SAMPLES_DIR / filename
        if wav_file.exists():
            success, result, match = test_transcription(wav_file, expected)
            results.append((filename, success, result, match))
        else:
            print(f"\nFile not found: {filename}")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, success, result, match in results:
        status = "OK" if success else "FAILED"
        match_str = f" (MATCH: {match})" if match is not None else ""
        print(f"{name}: {status}{match_str}")
