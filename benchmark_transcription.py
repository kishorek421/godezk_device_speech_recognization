"""Reproducible benchmark for transcription accuracy testing."""
import sys
from pathlib import Path
import time
import numpy as np
import soundfile as sf

# Add walker_audio to path
sys.path.insert(0, str(Path(__file__).parent / "walker_audio"))

from walker_audio.processing import ButterworthHighPassFilter, SpectralDenoiser, EnergyVad
from walker_audio.dictionary_corrector import correct_transcript

def calculate_wer(reference: str, hypothesis: str) -> tuple[int, int, int, float]:
    """Calculate Word Error Rate: (substitutions + deletions + insertions) / reference_words."""
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()
    
    # Simple Levenshtein-based WER calculation
    d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1), dtype=int)
    
    for i in range(len(ref_words) + 1):
        d[i, 0] = i
    for j in range(len(hyp_words) + 1):
        d[0, j] = j
    
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                cost = 0
            else:
                cost = 1
            d[i, j] = min(
                d[i-1, j] + 1,      # deletion
                d[i, j-1] + 1,      # insertion
                d[i-1, j-1] + cost  # substitution
            )
    
    wer = d[len(ref_words), len(hyp_words)] / max(len(ref_words), 1)
    return d[len(ref_words), len(hyp_words)], len(ref_words), len(hyp_words), wer

def run_benchmark():
    """Run benchmark on existing recordings with reference transcripts."""
    
    print("=" * 80)
    print("TRANSCRIPTION BENCHMARK")
    print("=" * 80)
    
    # Define test cases with reference transcripts
    # Using filenames from recordings directory
    test_cases = [
        {
            "pattern": "algorithm",
            "reference": "algorithm",
            "domain": "general",
        },
        {
            "pattern": "thermodynamics",
            "reference": "thermodynamics",
            "domain": "education",
        },
        {
            "pattern": "hypertension",
            "reference": "hypertension",
            "domain": "hospital",
        },
        {
            "pattern": "molecule",
            "reference": "molecule",
            "domain": "education",
        },
    ]
    
    recordings_dir = Path("recordings")
    if not recordings_dir.exists():
        print(f"Recordings directory not found: {recordings_dir}")
        return
    
    # Initialize processing components
    sample_rate = 16000
    high_pass = ButterworthHighPassFilter(65.0, sample_rate)
    denoiser = SpectralDenoiser(sample_rate)
    vad = EnergyVad(sample_rate, pre_roll_ms=100)
    
    print("\nProcessing components initialized:")
    print(f"  High-pass filter: 65Hz")
    print(f"  Denoiser: Improved (lowest-energy frames)")
    print(f"  VAD: Pre-roll 100ms, Release 300ms")
    
    results = []
    
    for test_case in test_cases:
        pattern = test_case["pattern"]
        reference = test_case["reference"]
        domain = test_case["domain"]
        
        # Find matching recordings
        matching_files = list(recordings_dir.glob(f"*{pattern}*.wav"))
        
        if not matching_files:
            print(f"\nNo recordings found for pattern: {pattern}")
            continue
        
        # Use the first matching file
        wav_file = matching_files[0]
        
        try:
            # Load audio
            audio, sr = sf.read(wav_file, dtype="float32", always_2d=True)
            mono = audio.mean(axis=1)
            
            if sr != sample_rate:
                print(f"\nWarning: {wav_file.name} has sample rate {sr}, expected {sample_rate}")
                continue
            
            print(f"\n--- Testing: {wav_file.name} ---")
            print(f"Reference: '{reference}'")
            print(f"Duration: {len(mono)/sr:.2f}s")
            
            # Apply processing pipeline
            start_time = time.perf_counter()
            
            # Step 1: High-pass filter
            filtered = high_pass(mono)
            filter_time = time.perf_counter() - start_time
            
            # Step 2: Denoising
            denoised = denoiser(filtered)
            denoise_time = time.perf_counter() - filter_time - start_time
            
            # Step 3: VAD
            segments = vad.split(denoised)
            vad_time = time.perf_counter() - denoise_time - filter_time - start_time
            
            if not segments:
                print("  No speech detected by VAD")
                continue
            
            # Concatenate segments
            speech = np.concatenate([seg.audio for seg in segments])
            speech_duration = len(speech) / sr
            
            processing_time = time.perf_counter() - start_time
            
            print(f"  Speech detected: {speech_duration:.2f}s")
            print(f"  Processing time: {processing_time*1000:.1f}ms")
            print(f"    Filter: {filter_time*1000:.1f}ms")
            print(f"    Denoise: {denoise_time*1000:.1f}ms")
            print(f"    VAD: {vad_time*1000:.1f}ms")
            
            # Note: We cannot test actual ASR without the GPU server
            # This benchmark tests the preprocessing pipeline only
            print(f"  ASR: Skipped (requires GPU server)")
            
            # Test dictionary correction on reference
            dict_result = correct_transcript(reference, domain=domain, enabled=True)
            dict_time = time.perf_counter() - processing_time - start_time
            
            print(f"  Dictionary correction: {dict_time*1000:.1f}ms")
            print(f"  Original: '{reference}'")
            print(f"  Corrected: '{dict_result['corrected_transcript']}'")
            print(f"  Corrections: {dict_result['corrections']}")
            
            results.append({
                "file": wav_file.name,
                "reference": reference,
                "corrected": dict_result["corrected_transcript"],
                "domain": domain,
                "speech_duration": speech_duration,
                "processing_time_ms": processing_time * 1000,
                "corrections": dict_result["corrections"],
            })
            
        except Exception as e:
            print(f"  Error processing {wav_file.name}: {e}")
    
    # Summary
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    
    if results:
        print(f"\nProcessed {len(results)} recordings")
        
        total_duration = sum(r["speech_duration"] for r in results)
        total_processing = sum(r["processing_time_ms"] for r in results)
        
        print(f"Total speech duration: {total_duration:.2f}s")
        print(f"Total processing time: {total_processing:.1f}ms")
        print(f"Average processing time: {total_processing/len(results):.1f}ms")
        print(f"Real-time factor: {total_processing/1000/total_duration:.2f}x")
        
        print("\nDetailed results:")
        for r in results:
            print(f"  {r['file']}: {r['speech_duration']:.2f}s speech, {r['processing_time_ms']:.1f}ms processing")
    else:
        print("No results to report")
    
    print("\nNote: This benchmark tests preprocessing pipeline only.")
    print("Full end-to-end ASR testing requires GPU server access.")

if __name__ == "__main__":
    run_benchmark()
