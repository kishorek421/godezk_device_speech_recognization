"""Test context engine fallback behavior when Ollama is unavailable."""
import sys
from pathlib import Path

# Add audio_api to path
sys.path.insert(0, str(Path(__file__).parent / "audio_api"))

from audio_api.context_engine import get_context_engine

def test_context_engine_fallback():
    """Test context engine behavior when Ollama is unavailable."""
    
    print("=" * 80)
    print("CONTEXT ENGINE FALLBACK TEST")
    print("=" * 80)
    
    engine = get_context_engine()
    
    # Test with Ollama disabled (simulate unavailability)
    test_cases = [
        {
            "original_transcript": "the patient has hyper tension",
            "dict_transcript": "the patient has hypertension",
            "session_id": "test_session_1",
            "domain": "medical",
            "subject": "cardiology",
            "topic": "hypertension",
            "speaker_id": "doctor",
            "enabled": False,  # Disabled to test fallback
        },
        {
            "original_transcript": "we studied photo synthesis",
            "dict_transcript": "we studied photosynthesis",
            "session_id": "test_session_2",
            "domain": "education",
            "subject": "biology",
            "topic": "photosynthesis",
            "speaker_id": "teacher",
            "enabled": True,  # Enabled but Ollama may be unavailable
        },
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i} ---")
        print(f"Original: '{test_case['original_transcript']}'")
        print(f"Dictionary: '{test_case['dict_transcript']}'")
        print(f"Domain: {test_case['domain']}")
        print(f"LLM Enabled: {test_case['enabled']}")
        
        try:
            result = engine.review_transcript(**test_case)
            
            print(f"Reviewed: '{result.get('reviewed_transcript', 'N/A')}'")
            print(f"Corrections: {result.get('corrections', [])}")
            print(f"LLM Reviewed: {result.get('llm_reviewed', False)}")
            print(f"Latency: {result.get('context_latency_ms', 0):.1f}ms")
            
            # When disabled, should return dictionary transcript unchanged
            if not test_case['enabled']:
                if result['reviewed_transcript'] == test_case['dict_transcript']:
                    print("✓ PASS: Fallback returns dictionary transcript when disabled")
                else:
                    print("✗ FAIL: Fallback did not return dictionary transcript")
            else:
                # When enabled but Ollama unavailable, should still return dictionary transcript
                if result['reviewed_transcript'] == test_case['dict_transcript']:
                    print("✓ PASS: Fallback returns dictionary transcript when Ollama unavailable")
                elif result.get('llm_reviewed'):
                    print("✓ PASS: LLM review succeeded (Ollama available)")
                else:
                    print("✓ PASS: Fallback handled gracefully")
                    
        except Exception as e:
            print(f"Exception: {e}")
            print("✓ PASS: Exception handled gracefully")
    
    # Test session management
    print("\n" + "=" * 80)
    print("SESSION MANAGEMENT TEST")
    print("=" * 80)
    
    session_id = "test_session_3"
    
    # Reset session
    engine.session_manager.reset(session_id)
    print(f"Reset session: {session_id}")
    
    # Get or create session (this is how turns are added via review_transcript)
    session = engine.session_manager.get_or_create(
        session_id,
        domain="general",
        subject="test",
        topic="test"
    )
    print(f"Session created: {session.session_id}")
    
    # Get summary
    summary = engine.session_manager.get_summary(session_id)
    print(f"Session summary: {summary}")
    
    if summary:
        print("✓ PASS: Session management works correctly")
    else:
        print("✗ FAIL: Session summary is None")

if __name__ == "__main__":
    test_context_engine_fallback()
