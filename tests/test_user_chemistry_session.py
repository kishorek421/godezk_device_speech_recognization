import unittest
from audio_api.context_engine import OllamaContextEngine

class TestUserChemistrySession(unittest.TestCase):
    def setUp(self):
        # Point to unreachable URL so it tests the fast domain context engine fallback
        self.engine = OllamaContextEngine(
            base_url="http://127.0.0.1:9999",
            model_name="qwen3.6:27b",
            timeout_sec=0.1
        )

    def test_user_chemistry_sequence(self):
        session_id = "test_user_browser_session"
        domain = "education"
        subject = "Chemistry"
        topic = "Acids and Bases"

        # Turn 1
        res1 = self.engine.review_transcript(
            original_transcript="After adding H to the solution",
            dict_transcript="After adding H to the solution",
            session_id=session_id,
            domain=domain,
            subject=subject,
            topic=topic,
            speaker_id="Unknown Speaker",
            enabled=True,
        )
        self.assertEqual(res1["reviewed_transcript"], "After adding HCl to the solution")
        self.assertIn("HCl", res1["inferred_terms"])

        # Turn 2
        res2 = self.engine.review_transcript(
            original_transcript="B after adding it to the solution",
            dict_transcript="B after adding it to the solution",
            session_id=session_id,
            domain=domain,
            subject=subject,
            topic=topic,
            speaker_id="megak",
            enabled=True,
        )
        self.assertEqual(res2["reviewed_transcript"], "Base after adding it to the solution")
        self.assertIn("Base", res2["inferred_terms"])

        # Turn 3
        res3 = self.engine.review_transcript(
            original_transcript="P after adding H to the solution.",
            dict_transcript="P after adding H to the solution.",
            session_id=session_id,
            domain=domain,
            subject=subject,
            topic=topic,
            speaker_id="megak",
            enabled=True,
        )
        self.assertEqual(res3["reviewed_transcript"], "pH after adding HCl to the solution.")
        self.assertIn("pH", res3["inferred_terms"])
        self.assertIn("HCl", res3["inferred_terms"])

        # Inspect session summary
        summary = self.engine.session_manager.get_summary(session_id)
        self.assertEqual(summary["turn_count"], 3)
        self.assertIn("HCl", summary["established_terms"])
        self.assertIn("Base", summary["established_terms"])
        self.assertIn("pH", summary["established_terms"])
        self.assertIn("pH", summary["established_abbreviations"])

        print("User Chemistry Session verified perfectly:")
        import json
        print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    unittest.main()
