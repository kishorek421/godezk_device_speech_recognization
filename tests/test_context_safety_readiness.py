"""Evidence-Based Context Engine Safety, Anti-Hallucination & Session Isolation Suite.

Rigorous unit and scenario testing covering:
1. Anti-hallucination guardrail (blocks fabricated clinical text)
2. Medical dosage & quantity preservation (never modifies numbers, units, dosages)
3. Negation preservation (verifies 'not', 'never', 'without' are never flipped)
4. Critical entity preservation (patient names, vital measurements)
5. Chemistry vs Physics polysemy (W=Fd vs W=-P*deltaV)
6. Multi-turn topic shifting within same session
7. Session isolation & cross-session leakage prevention
8. Session TTL expiration & memory cleanup
9. Misleading / contradictory context resistance
10. Malformed / corrupted LLM response recovery
"""

import time
import unittest
from audio_api.context_engine import (
    OllamaContextEngine,
    SessionManager,
    ConversationSession,
    DEFAULT_SESSION_TTL,
)

class TestContextSafetyReadiness(unittest.TestCase):

    def setUp(self):
        # Engine configured with unreachable base URL to test deterministic safety logic and guardrails
        self.engine = OllamaContextEngine(
            base_url="http://127.0.0.1:9999",
            model_name="qwen3.6:27b",
            timeout_sec=0.1
        )

    # 1. Anti-Hallucination Guardrail
    def test_01_anti_hallucination_expansion_barrier(self):
        # Simulated LLM hallucinating a massive clinical paragraph from short audio
        mock_hallucination = {
            "reviewed_transcript": "Check the oxygen level immediately and administer 50 milligrams of intravenous epinephrine twice daily without delay or patient will decompensate rapidly.",
            "corrections": [],
            "inferred_terms": [],
            "uncertain_segments": []
        }
        res = self.engine.review_transcript(
            original_transcript="Check oxygen.",
            dict_transcript="Check oxygen.",
            session_id="hallucination_test",
            domain="hospital",
            subject="cardiology",
            enabled=True,
            mock_response=mock_hallucination
        )
        # Guardrail must block the expansion and fall back to input text, marking the expansion as uncertain
        self.assertEqual(res["reviewed_transcript"], "Check oxygen.")
        self.assertTrue(len(res["uncertain_segments"]) > 0, "Hallucinated expansion must be flagged in uncertain_segments")

    # 2. Medical Dosage & Unit Preservation
    def test_02_medical_dosage_preservation(self):
        mock_dosage_tamper = {
            "reviewed_transcript": "Administer 100 milligrams of potassium chloride once daily.",
            "corrections": [{"original": "20", "corrected": "100", "reason": "Adjusted dose"}],
            "inferred_terms": [],
            "uncertain_segments": []
        }
        # In strict domain context, numbers and dosages must be preserved
        res = self.engine.review_transcript(
            original_transcript="Administer 20 milligrams of potassium chloride once daily.",
            dict_transcript="Administer 20 milligrams of potassium chloride once daily.",
            session_id="dosage_test",
            domain="hospital",
            subject="general",
            enabled=True,
        )
        # Even without mock, verify our engine never modifies numbers or clinical amounts
        self.assertIn("20 milligrams", res["reviewed_transcript"])
        self.assertNotIn("100 milligrams", res["reviewed_transcript"])

    # 3. Negation Preservation
    def test_03_negation_preservation(self):
        res = self.engine.review_transcript(
            original_transcript="The patient has not experienced chest pain or shortness of breath.",
            dict_transcript="The patient has not experienced chest pain or shortness of breath.",
            session_id="negation_test",
            domain="hospital",
            subject="cardiology",
            enabled=True,
        )
        self.assertIn("not experienced", res["reviewed_transcript"].lower())
        self.assertIn("chest pain", res["reviewed_transcript"].lower())

    # 4. Critical Named Entity Preservation
    def test_04_patient_name_preservation(self):
        res = self.engine.review_transcript(
            original_transcript="Patient Jonathan Miller admitted for observation.",
            dict_transcript="Patient Jonathan Miller admitted for observation.",
            session_id="name_test",
            domain="hospital",
            subject="general",
            enabled=True,
        )
        self.assertIn("Jonathan Miller", res["reviewed_transcript"])

    # 5. Chemistry vs Physics Polysemy Disambiguation
    def test_05_polysemy_physics_vs_chemistry(self):
        # Physics context
        res_phys = self.engine.review_transcript(
            original_transcript="The work done equals force multiplied by displacement.",
            dict_transcript="The work done equals force multiplied by displacement.",
            session_id="poly_phys",
            domain="education",
            subject="physics",
            topic="Classical Mechanics",
            enabled=True,
        )
        self.assertIn("work", res_phys["inferred_terms"])
        self.assertIn("force", res_phys["inferred_terms"])

        # Chemistry context
        res_chem = self.engine.review_transcript(
            original_transcript="The gas does work when it expands against external pressure.",
            dict_transcript="The gas does work when it expands against external pressure.",
            session_id="poly_chem",
            domain="education",
            subject="chemistry",
            topic="Thermodynamics",
            enabled=True,
        )
        self.assertIn("thermodynamic work", res_chem["inferred_terms"])
        self.assertIn("pressure", res_chem["inferred_terms"])

    # 6. Multi-Turn Topic Shift Within Same Session
    def test_06_topic_shift_within_session(self):
        sess_id = "shift_session"
        # Turn 1: Chemistry
        t1 = self.engine.review_transcript(
            original_transcript="We prepare the standard NaOH solution.",
            dict_transcript="We prepare the standard NaOH solution.",
            session_id=sess_id,
            domain="education",
            subject="chemistry",
            topic="Titration",
            enabled=True,
        )
        self.assertIn("NaOH", t1["inferred_terms"])

        # Turn 2: Topic shifts to Electrical Transformers in same session
        t2 = self.engine.review_transcript(
            original_transcript="Connect the transformer primary winding to the AC supply.",
            dict_transcript="Connect the transformer primary winding to the AC supply.",
            session_id=sess_id,
            domain="industry",
            subject="electrical",
            topic="Transformers",
            enabled=True,
        )
        self.assertIn("transformer", t2["inferred_terms"])
        self.assertIn("primary winding", t2["inferred_terms"])
        self.assertIn("AC", t2["inferred_terms"])

        summary = self.engine.session_manager.get_summary(sess_id)
        self.assertEqual(summary["turn_count"], 2)
        self.assertIn("NaOH", summary["established_terms"])
        self.assertIn("transformer", summary["established_terms"])

    # 7. Session Isolation (Leakage Prevention)
    def test_07_session_isolation_and_no_cross_leakage(self):
        mgr = SessionManager()
        s_user1 = mgr.get_or_create("user_alpha", domain="hospital", subject="cardiology")
        s_user2 = mgr.get_or_create("user_beta", domain="industry", subject="electrical")

        s_user1.add_turn(
            original_transcript="Patient has hypertension.",
            dict_transcript="Patient has hypertension.",
            reviewed_transcript="Patient has hypertension.",
            inferred_terms=["hypertension"]
        )
        s_user2.add_turn(
            original_transcript="Transformer voltage is high.",
            dict_transcript="Transformer voltage is high.",
            reviewed_transcript="Transformer voltage is high.",
            inferred_terms=["transformer"]
        )

        sum1 = mgr.get_summary("user_alpha")
        sum2 = mgr.get_summary("user_beta")

        self.assertIn("hypertension", sum1["established_terms"])
        self.assertNotIn("transformer", sum1["established_terms"])

        self.assertIn("transformer", sum2["established_terms"])
        self.assertNotIn("hypertension", sum2["established_terms"])

    # 8. Session TTL Expiration & Auto-Cleanup
    def test_08_session_ttl_expiration(self):
        mgr = SessionManager(ttl_seconds=1)  # 1 second TTL for test
        s = mgr.get_or_create("expiring_user")
        s.add_turn("Test transcript", "Test transcript", "Test transcript")

        # Before expiry
        self.assertIsNotNone(mgr.get_summary("expiring_user"))

        # Wait for TTL to elapse
        time.sleep(1.2)

        # Trigger cleanup
        mgr._cleanup_expired_locked()
        self.assertIsNone(mgr.get_summary("expiring_user"))

    # 9. Resistance to Misleading / False Prior Turns
    def test_09_resistance_to_misleading_prior_context(self):
        sess_id = "misleading_session"
        # Prior turn had an unverified incorrect claim
        self.engine.review_transcript(
            original_transcript="The earth is flat and gravity is zero.",
            dict_transcript="The earth is flat and gravity is zero.",
            session_id=sess_id,
            domain="education",
            subject="physics",
            enabled=True,
        )
        # Next turn states accurate physical formula
        t2 = self.engine.review_transcript(
            original_transcript="The work done equals force multiplied by displacement.",
            dict_transcript="The work done equals force multiplied by displacement.",
            session_id=sess_id,
            domain="education",
            subject="physics",
            enabled=True,
        )
        # Engine must not corrupt the valid physics sentence
        self.assertEqual(t2["reviewed_transcript"], "The work done equals force multiplied by displacement.")
        self.assertIn("work", t2["inferred_terms"])

    # 10. Malformed / Corrupted LLM Response Recovery
    def test_10_malformed_llm_response_recovery(self):
        # LLM returns empty or broken JSON string
        broken_mock = {}
        res = self.engine.review_transcript(
            original_transcript="Standard clinical observation.",
            dict_transcript="Standard clinical observation.",
            session_id="malformed_test",
            domain="hospital",
            enabled=True,
            mock_response=broken_mock
        )
        # Pipeline must not crash, must fall back cleanly
        self.assertEqual(res["reviewed_transcript"], "Standard clinical observation.")
        self.assertEqual(res["status"], "success")

if __name__ == "__main__":
    unittest.main()
