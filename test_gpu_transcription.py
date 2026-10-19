"""Test actual transcription through remote GPU API."""
import requests
import soundfile as sf
import io
import time
from pathlib import Path

REMOTE_GPU_API = "http://124.123.89.162:8001"
RECORDINGS_DIR = Path("recordings")

def test_transcription(wav_file: Path):
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
            print(f"  Transcript: '{result.get('transcript', 'N/A')}'")
            print(f"  Original: '{result.get('original_transcript', 'N/A')}'")
            print(f"  Corrected: '{result.get('corrected_transcript', 'N/A')}'")
            print(f"  Corrections: {result.get('corrections', [])}")
            print(f"  Speech seconds: {result.get('speech_seconds', 0):.2f}s")
            print(f"  Processing ms: {result.get('processing_ms', 0):.1f}ms")
            return True, result
        else:
            print(f"  Error: {response.text}")
            return False, None
            
    except requests.Timeout:
        print(f"  ERROR: Timeout")
        return False, None
    except requests.ConnectionError:
        print(f"  ERROR: Connection error")
        return False, None
    except Exception as e:
        print(f"  ERROR: {e}")
        return False, None

if __name__ == "__main__":
    print("=" * 60)
    print("REMOTE GPU TRANSCRIPTION TEST")
    print(f"Endpoint: {REMOTE_GPU_API}")
    print("=" * 60)
    
    if not RECORDINGS_DIR.exists():
        print(f"Recordings directory not found: {RECORDINGS_DIR}")
        exit(1)
    
    # Test with a few short recordings
    test_files = [
        "algorithm",
        "thermodynamics",
        "hypertension",
    ]
    
    results = []
    for pattern in test_files:
        matching = list(RECORDINGS_DIR.glob(f"*{pattern}*.wav"))
        if matching:
            success, result = test_transcription(matching[0])
            results.append((matching[0].name, success, result))
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, success, result in results:
        status = "OK" if success else "FAILED"
        print(f"{name}: {status}")
