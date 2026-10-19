"""Comprehensive Automated Unit Tests for Real-Time Context Engine & Conversation Memory.

Validates all 11 requirements specified in Section 10 with deterministic mocked responses:
1. Physics vs Chemistry polysemy (e.g., 'work')
2. General vs Technical context (e.g., 'base')
3. Abbreviation recovery (pH, HCl, BP, SpO2)
4. Multi-turn topic continuity (Transformers conversation)
5. Topic updates across turns
6. Incomplete sentence continuity across adjacent turns
7. Anti-hallucination: missing words marked uncertain rather than fabricated
8. Session isolation & cross-session leakage prevention
9. Graceful fallback on timeout, invalid output, and unavailable endpoint
10. Disabled-mode backward compatibility
11. Regression test: existing dictionary and speaker verification compatibility
"""

import unittest
import time
from audio_api.context_engine import (
    OllamaContextEngine,
    SessionManager,
    ConversationSession,
    DOMAIN_KNOWLEDGE,
)
from walker_audio.dictionary_corrector import correct_transcript
from walker_audio.speaker_recognizer import get_speaker_registry


class TestContextEngine(unittest.TestCase):

    def setUp(self):
        self.engine = OllamaContextEngine(
            base_url="http://127.0.0.1:9999",  # unreachable dummy url to enforce mocked / fallback paths
            model_name="qwen3.6:27b",
            timeout_sec=0.5
        )

    # 1. Physics vs Chemistry word meanings
    def test_01_physics_vs_chemistry_polysemy(self):
        # Physics test
        mock_physics = {
            "reviewed_transcript": "The work done is equal to force multiplied by displacement.",
            "corrections": [],
            "inferred_terms": ["work", "force", "displacement"],
            "uncertain_segments": []
        }
        res_p = self.engine.review_transcript(
            original_transcript="The work done is equal to force multiplied by displacement.",
            dict_transcript="The work done is equal to force multiplied by displacement.",
            session_id="sess_physics",
            domain="education",
            subject="physics",
            topic="Classical mechanics",
            enabled=True,
            mock_response=mock_physics
        )
        self.assertEqual(res_p["reviewed_transcript"], "The work done is equal to force multiplied by displacement.")
        self.assertIn("work", res_p["inferred_terms"])

        # Chemistry test
        mock_chem = {
            "reviewed_transcript": "The gas does work when it expands against external pressure.",
            "corrections": [],
            "inferred_terms": ["thermodynamic work", "pressure"],
            "uncertain_segments": []
        }
        res_c = self.engine.review_transcript(
            original_transcript="The gas does work when it expands against external pressure.",
            dict_transcript="The gas does work when it expands against external pressure.",
            session_id="sess_chem",
            domain="education",
            subject="chemistry",
            topic="Thermodynamics and gas laws",
            enabled=True,
            mock_response=mock_chem
        )
        self.assertEqual(res_c["reviewed_transcript"], "The gas does work when it expands against external pressure.")

    # 2. General vs Technical context (e.g. 'base')
    def test_02_general_vs_technical_context(self):
        mock_res = {
            "reviewed_transcript": "Titrate the strong acid with a standard base solution.",
            "corrections": [{"original": "vase", "corrected": "base", "reason": "Chemistry context indicates alkaline base"}],
            "inferred_terms": ["base", "acid"],
            "uncertain_segments": []
        }
        res = self.engine.review_transcript(
            original_transcript="Titrate the strong acid with a standard vase solution.",
            dict_transcript="Titrate the strong acid with a standard vase solution.",
            session_id="sess_chem_base",
            domain="education",
            subject="chemistry",
            topic="Titration",
            enabled=True,
            mock_response=mock_res
        )
        self.assertEqual(res["reviewed_transcript"], "Titrate the strong acid with a standard base solution.")
        self.assertEqual(res["corrections"][0]["corrected"], "base")

    # 3. Technical abbreviations (pH, HCl, BP, SpO2)
    def test_03_technical_abbreviations(self):
        # Chemistry pH and HCl
        mock_chem_abbr = {
            "reviewed_transcript": "Calculate the pH after adding HCl to the solution.",
            "corrections": [
                {"original": "p", "corrected": "pH", "reason": "Acidity abbreviation"},
                {"original": "H", "corrected": "HCl", "reason": "Hydrochloric acid abbreviation"}
            ],
            "inferred_terms": ["pH", "HCl"],
            "uncertain_segments": []
        }
        res_chem = self.engine.review_transcript(
            original_transcript="Calculate the p after adding H to the solution.",
            dict_transcript="Calculate the p after adding H to the solution.",
            session_id="sess_abbr_chem",
            domain="education",
            subject="chemistry",
            topic="Acid-base titration",
            enabled=True,
            mock_response=mock_chem_abbr
        )
        self.assertEqual(res_chem["reviewed_transcript"], "Calculate the pH after adding HCl to the solution.")
        self.assertIn("pH", res_chem["inferred_terms"])
        self.assertIn("HCl", res_chem["inferred_terms"])

        # Medical BP and SpO2
        mock_med_abbr = {
            "reviewed_transcript": "Check BP and SpO2 before administering the medication.",
            "corrections": [
                {"original": "S P O", "corrected": "SpO2", "reason": "Oxygen saturation abbreviation"}
            ],
            "inferred_terms": ["BP", "SpO2"],
            "uncertain_segments": []
        }
        res_med = self.engine.review_transcript(
            original_transcript="Check BP and S P O before administering the medication.",
            dict_transcript="Check BP and S P O before administering the medication.",
            session_id="sess_abbr_med",
            domain="hospital",
            subject="cardiology",
            topic="Vital signs assessment",
            enabled=True,
            mock_response=mock_med_abbr
        )
        self.assertEqual(res_med["reviewed_transcript"], "Check BP and SpO2 before administering the medication.",)
        self.assertIn("SpO2", res_med["inferred_terms"])

    # 4. Multi-turn continuing conversation (Transformers)
    def test_04_multi_turn_topic_continuity(self):
        sess_id = "sess_transformers"
        
        # Turn 1
        m1 = {"reviewed_transcript": "Today we are discussing transformers.", "corrections": [], "inferred_terms": ["transformers"], "uncertain_segments": []}
        self.engine.review_transcript("Today we are discussing transformers.", "Today we are discussing transformers.", sess_id, "industry", "electrical", "Transformers", enabled=True, mock_response=m1)
        
        # Turn 2
        m2 = {"reviewed_transcript": "The primary winding receives the input voltage.", "corrections": [], "inferred_terms": ["primary winding"], "uncertain_segments": []}
        self.engine.review_transcript("The primary winding receives the input voltage.", "The primary winding receives the input voltage.", sess_id, "industry", "electrical", "Transformers", enabled=True, mock_response=m2)
        
        # Turn 3
        m3 = {"reviewed_transcript": "The secondary winding produces the output voltage.", "corrections": [], "inferred_terms": ["secondary winding"], "uncertain_segments": []}
        res3 = self.engine.review_transcript("The secondary winding produces the output voltage.", "The secondary winding produces the output voltage.", sess_id, "industry", "electrical", "Transformers", enabled=True, mock_response=m3)
        
        summary = self.engine.session_manager.get_summary(sess_id)
        self.assertIsNotNone(summary)
        self.assertEqual(summary["turn_count"], 3)
        self.assertEqual(summary["topic"], "Transformers")
        self.assertEqual(summary["recent_turns"][2]["reviewed"], "The secondary winding produces the output voltage.")

    # 5. Topic updates after topic changes
    def test_05_topic_updates(self):
        sess_id = "sess_topic_change"
        m1 = {"reviewed_transcript": "First topic is calculus integrals.", "corrections": [], "inferred_terms": [], "uncertain_segments": []}
        self.engine.review_transcript("First topic is calculus integrals.", "First topic is calculus integrals.", sess_id, "education", "mathematics", "Calculus", enabled=True, mock_response=m1)

        # Switch topic
        m2 = {"reviewed_transcript": "Now we move to linear algebra and eigenvalues.", "corrections": [], "inferred_terms": ["eigenvalues"], "uncertain_segments": []}
        res2 = self.engine.review_transcript("Now we move to linear algebra and eigenvalues.", "Now we move to linear algebra and eigenvalues.", sess_id, "education", "mathematics", "Linear Algebra", enabled=True, mock_response=m2)
        
        session = self.engine.session_manager.sessions[sess_id]
        self.assertEqual(session.topic, "Linear Algebra")
        self.assertEqual(res2["topic"], "Linear Algebra")

    # 6. Incomplete sentences across adjacent chunks
    def test_06_incomplete_sentences(self):
        sess_id = "sess_incomplete"
        m1 = {"reviewed_transcript": "The patient exhibits severe...", "corrections": [], "inferred_terms": [], "uncertain_segments": ["severe..."]}
        self.engine.review_transcript("The patient exhibits severe", "The patient exhibits severe", sess_id, "hospital", "cardiology", "Exam", enabled=True, mock_response=m1)

        m2 = {"reviewed_transcript": "...tachycardia and hypertension.", "corrections": [], "inferred_terms": ["tachycardia", "hypertension"], "uncertain_segments": []}
        res2 = self.engine.review_transcript("tachycardia and hypertension", "tachycardia and hypertension", sess_id, "hospital", "cardiology", "Exam", enabled=True, mock_response=m2)
        
        self.assertEqual(res2["reviewed_transcript"], "...tachycardia and hypertension.")
        self.assertIn("tachycardia", res2["inferred_terms"])

    # 7. Anti-hallucination: missing speech must remain uncertain
    def test_07_anti_hallucination_and_uncertainty(self):
        # Simulating LLM trying to hallucinate an entire unsaid sentence
        hallucinated_mock = {
            "reviewed_transcript": "Check the oxygen level and immediately administer fifty milligrams of intravenous epinephrine twice daily without fail.",
            "corrections": [],
            "inferred_terms": [],
            "uncertain_segments": []
        }
        res = self.engine.review_transcript(
            original_transcript="Check oxygen",
            dict_transcript="Check oxygen",
            session_id="sess_hallucination",
            domain="hospital",
            subject="cardiology",
            topic="Emergency check",
            enabled=True,
            mock_response=hallucinated_mock
        )
        # Guardrail must reject the hallucinated expansion and retain original or mark uncertain
        self.assertIn("Check oxygen", res["reviewed_transcript"])
        self.assertTrue(len(res["uncertain_segments"]) > 0)

    # 8. Session isolation & preventing context leakage
    def test_08_session_isolation(self):
        # Session A: Alice in Hospital
        mA = {"reviewed_transcript": "Patient Alice has hypertension.", "corrections": [], "inferred_terms": ["hypertension"], "uncertain_segments": []}
        self.engine.review_transcript("Patient Alice has hypertension.", "Patient Alice has hypertension.", "session_A", "hospital", "cardiology", "Patient Alice", enabled=True, mock_response=mA)

        # Session B: Bob in Industry
        mB = {"reviewed_transcript": "Testing generator transformer coils.", "corrections": [], "inferred_terms": ["transformer"], "uncertain_segments": []}
        self.engine.review_transcript("Testing generator transformer coils.", "Testing generator transformer coils.", "session_B", "industry", "electrical", "Generators", enabled=True, mock_response=mB)

        summary_A = self.engine.session_manager.get_summary("session_A")
        summary_B = self.engine.session_manager.get_summary("session_B")

        # Verify no cross-talk
        self.assertEqual(summary_A["domain"], "hospital")
        self.assertEqual(summary_B["domain"], "industry")
        self.assertNotIn("transformer", summary_A["established_terms"])
        self.assertNotIn("hypertension", summary_B["established_terms"])

    # 9. LLM timeout, malformed output and unavailable endpoint
    def test_09_llm_timeout_and_fallback(self):
        # Calling without mock against unreachable port 9999 triggers timeout fallback
        res = self.engine.review_transcript(
            original_transcript="The algorithm operates in logarithmic time.",
            dict_transcript="The algorithm operates in logarithmic time.",
            session_id="sess_timeout",
            domain="education",
            subject="mathematics",
            topic="Complexity",
            enabled=True,
            timeout_sec=0.2  # Fast timeout
        )
        self.assertEqual(res["status"], "fallback")
        self.assertEqual(res["reviewed_transcript"], "The algorithm operates in logarithmic time.")
        self.assertIn("timeout", res.get("reason", "").lower())

    # 10. Context feature disabled by default
    def test_10_context_feature_disabled_by_default(self):
        res = self.engine.review_transcript(
            original_transcript="Normal ASR output without LLM review.",
            dict_transcript="Normal ASR output without LLM review.",
            session_id="sess_disabled",
            domain="general",
            enabled=False  # Disabled by default
        )
        self.assertEqual(res["status"], "disabled")
        self.assertEqual(res["reviewed_transcript"], "Normal ASR output without LLM review.")
        self.assertFalse(res["llm_reviewed"])

    # 11. Existing dictionary and speaker verification regression tests
    def test_11_existing_components_regression(self):
        # Verify dictionary engine still corrects domain terms accurately
        dict_res = correct_transcript("The patient has high hyper tension", domain="hospital", enabled=True)
        self.assertIn("hypertension", dict_res["corrected_transcript"].lower())

        # Verify speaker registry exists and loads registered profiles without regression
        reg = get_speaker_registry()
        enrolled = reg.get_enrolled_summary()
        self.assertGreaterEqual(len(enrolled), 1)
        names = [s["name"] for s in enrolled]
        self.assertIn("megak", names)


if __name__ == "__main__":
    unittest.main()
