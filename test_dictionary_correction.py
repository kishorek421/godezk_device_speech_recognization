"""Test dictionary correction across all domains."""
import sys
from pathlib import Path

# Add walker_audio to path
sys.path.insert(0, str(Path(__file__).parent / "walker_audio"))

from walker_audio.dictionary_corrector import correct_transcript, load_domain_dictionary

def test_dictionary_correction():
    """Test dictionary correction with domain-specific terminology."""
    
    # Load dictionary
    dictionary = load_domain_dictionary()
    print(f"Loaded dictionary with domains: {list(dictionary.keys())}")
    
    # Test cases for each domain
    test_cases = [
        # General domain
        ("general", "the patient has hyper tension", "hypertension"),
        ("general", "use the seismoscope", "stethoscope"),
        ("general", "math derivation is complex", "mathematical derivation"),
        ("general", "logic controller failed", "PLC"),
        ("general", "tachy cardia detected", "tachycardia"),
        
        # Education domain (uses general dictionary)
        ("education", "photo synthesis process", "photosynthesis"),
        ("education", "thermo dynamics laws", "thermodynamics"),
        ("education", "semi conductor device", "semiconductor"),
        ("education", "curiculum review", "curriculum"),
        ("education", "derivasion formula", "derivation"),
        
        # Hospital/Medical domain (uses general dictionary)
        ("hospital", "diagonsis confirmed", "diagnosis"),
        ("hospital", "perscription written", "prescription"),
        ("hospital", "radiologi department", "radiology"),
        ("hospital", "arrythmia treatment", "arrhythmia"),
        ("hospital", "stetho scope check", "stethoscope"),
        
        # Industry domain (uses general dictionary)
        ("industry", "manufaturing process", "manufacturing"),
        ("industry", "maintainance schedule", "maintenance"),
        ("industry", "inspeksion required", "inspection"),
        ("industry", "calibaration needed", "calibration"),
        ("industry", "machinary repair", "machinery"),
    ]
    
    print("\n" + "=" * 80)
    print("DICTIONARY CORRECTION TEST RESULTS")
    print("=" * 80)
    
    passed = 0
    failed = 0
    
    for domain, input_text, expected_term in test_cases:
        result = correct_transcript(input_text, domain=domain, enabled=True)
        corrected = result["corrected_transcript"]
        corrections = result["corrections"]
        
        # Check if expected term appears in corrected text
        if expected_term.lower() in corrected.lower():
            status = "✓ PASS"
            passed += 1
        else:
            status = "✗ FAIL"
            failed += 1
        
        print(f"\n{status} [{domain.upper()}]")
        print(f"  Input:    '{input_text}'")
        print(f"  Output:   '{corrected}'")
        print(f"  Expected: '{expected_term}'")
        print(f"  Corrections: {corrections}")
    
    print("\n" + "=" * 80)
    print(f"SUMMARY: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    print("=" * 80)
    
    # Test edge cases
    print("\n" + "=" * 80)
    print("EDGE CASE TESTS")
    print("=" * 80)
    
    edge_cases = [
        ("general", "", "empty input"),
        ("general", "hello world", "no corrections needed"),
        ("general", "The patient has Hyper Tension", "case preservation"),
        ("general", "algorithm", "already correct"),
        ("invalid_domain", "hyper tension", "invalid domain"),
    ]
    
    for domain, input_text, description in edge_cases:
        result = correct_transcript(input_text, domain=domain, enabled=True)
        print(f"\nTest: {description}")
        print(f"  Input:  '{input_text}'")
        print(f"  Output: '{result['corrected_transcript']}'")
        print(f"  Domain: {result['domain']}")
        print(f"  Enabled: {result['dictionary_enabled']}")

if __name__ == "__main__":
    test_dictionary_correction()
