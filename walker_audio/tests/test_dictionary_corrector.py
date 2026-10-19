"""
Unit tests for Dictionary-Based Transcript Correction.
Verifies all 10 requirements and edge cases specified in the requirements.
"""

import json
from pathlib import Path
import pytest
from walker_audio.dictionary_corrector import correct_transcript, load_domain_dictionary


@pytest.fixture
def sample_dict_path(tmp_path: Path) -> Path:
    data = {
        "medical": {
            "hyper tension": "hypertension",
            "metforman": "metformin",
            "paracetemol": "paracetamol",
        },
        "education": {
            "photosynthe sis": "photosynthesis",
            "thermodynamiks": "thermodynamics",
            "semiconducter": "semiconductor",
        },
        "industry": {
            "centrifuj": "centrifuge",
            "programmable logic controller": "PLC",
        },
        "general": {},
    }
    p = tmp_path / "test_domain_dictionary.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def test_1_exact_term_replacement(sample_dict_path: Path):
    """Test 1: Exact term replacement in target domain."""
    text = "The doctor prescribed metforman for the condition."
    res = correct_transcript(text, domain="medical", dictionary_path=sample_dict_path)
    assert res["corrected_transcript"] == "The doctor prescribed metformin for the condition."
    assert res["original_transcript"] == text
    assert len(res["corrections"]) == 1
    assert res["corrections"][0]["original"] == "metforman"
    assert res["corrections"][0]["replacement"] == "metformin"


def test_2_case_insensitive_matching(sample_dict_path: Path):
    """Test 2: Case-insensitive matching with case preservation."""
    # Title Case
    res_title = correct_transcript("Patient had Metforman prescribed.", domain="medical", dictionary_path=sample_dict_path)
    assert res_title["corrected_transcript"] == "Patient had Metformin prescribed."

    # UPPERCASE
    res_upper = correct_transcript("PATIENT HAD METFORMAN PRESCRIBED.", domain="medical", dictionary_path=sample_dict_path)
    assert res_upper["corrected_transcript"] == "PATIENT HAD METFORMIN PRESCRIBED."

    # Multi-word Title Case
    res_multi_title = correct_transcript("Diagnosis was Hyper Tension.", domain="medical", dictionary_path=sample_dict_path)
    assert res_multi_title["corrected_transcript"] == "Diagnosis was Hypertension."


def test_3_multi_word_correction(sample_dict_path: Path):
    """Test 3: Multi-word phrases with single or multiple spaces."""
    text = "The factory installed a programmable logic controller today."
    res = correct_transcript(text, domain="industry", dictionary_path=sample_dict_path)
    assert res["corrected_transcript"] == "The factory installed a PLC today."
    assert res["corrections"][0]["replacement"] == "PLC"

    # Multi-word with multiple spaces
    text_spaces = "The plant studied photosynthe    sis in class."
    res_spaces = correct_transcript(text_spaces, domain="education", dictionary_path=sample_dict_path)
    assert res_spaces["corrected_transcript"] == "The plant studied photosynthesis in class."


def test_4_punctuation_preservation(sample_dict_path: Path):
    """Test 4: Surrounding punctuation (commas, quotes, periods, exclamation)."""
    text = "Symptoms include: hyper tension, fatigue; then take metforman!"
    res = correct_transcript(text, domain="medical", dictionary_path=sample_dict_path)
    assert res["corrected_transcript"] == "Symptoms include: hypertension, fatigue; then take metformin!"
    assert res["original_transcript"] == text


def test_5_embedded_words_unrelated(sample_dict_path: Path):
    """Test 5: Words embedded inside larger unrelated words must NOT be modified."""
    text = "The unmetformanx compound caused hyperextension of the knee."
    res = correct_transcript(text, domain="medical", dictionary_path=sample_dict_path)
    assert res["corrected_transcript"] == text
    assert len(res["corrections"]) == 0


def test_6_empty_transcript(sample_dict_path: Path):
    """Test 6: Empty or whitespace-only transcript handled gracefully."""
    assert correct_transcript("", domain="medical", dictionary_path=sample_dict_path)["corrected_transcript"] == ""
    assert correct_transcript("   ", domain="medical", dictionary_path=sample_dict_path)["corrected_transcript"] == "   "
    assert correct_transcript(None, domain="medical", dictionary_path=sample_dict_path)["corrected_transcript"] == ""


def test_7_unknown_domain(sample_dict_path: Path):
    """Test 7: Unknown domain leaves text unchanged."""
    text = "The patient has hyper tension."
    res = correct_transcript(text, domain="finance", dictionary_path=sample_dict_path)
    assert res["corrected_transcript"] == text
    assert len(res["corrections"]) == 0


def test_8_dictionary_disabled(sample_dict_path: Path):
    """Test 8: When enabled=False, text remains unchanged."""
    text = "The patient has hyper tension and takes metforman."
    res = correct_transcript(text, domain="medical", enabled=False, dictionary_path=sample_dict_path)
    assert res["corrected_transcript"] == text
    assert res["dictionary_enabled"] is False
    assert len(res["corrections"]) == 0


def test_9_missing_or_invalid_dictionary_file(tmp_path: Path):
    """Test 9: Missing or malformed JSON file handled safely without crashing."""
    text = "The patient has hyper tension."
    # Nonexistent file
    res_missing = correct_transcript(text, domain="medical", dictionary_path=tmp_path / "nonexistent.json")
    assert res_missing["corrected_transcript"] == text
    assert len(res_missing["corrections"]) == 0

    # Invalid JSON
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{invalid json", encoding="utf-8")
    res_bad = correct_transcript(text, domain="medical", dictionary_path=bad_json)
    assert res_bad["corrected_transcript"] == text
    assert len(res_bad["corrections"]) == 0


def test_10_no_matching_terms(sample_dict_path: Path):
    """Test 10: Clean text with no dictionary entries needed."""
    text = "The quick brown fox jumps over the lazy dog."
    res = correct_transcript(text, domain="medical", dictionary_path=sample_dict_path)
    assert res["corrected_transcript"] == text
    assert len(res["corrections"]) == 0


def test_prompt_example(sample_dict_path: Path):
    """Explicit verification of user prompt requirement:
    Input: 'The patient has hyper tension and takes metforman.'
    Domain: 'medical'
    Expected: 'The patient has hypertension and takes metformin.'
    """
    prompt_input = "The patient has hyper tension and takes metforman."
    res = correct_transcript(prompt_input, domain="medical", dictionary_path=sample_dict_path)
    assert res["corrected_transcript"] == "The patient has hypertension and takes metformin."
    assert res["original_transcript"] == prompt_input
    assert len(res["corrections"]) == 2
    assert res["corrections"][0]["original"] == "hyper tension"
    assert res["corrections"][0]["replacement"] == "hypertension"
    assert res["corrections"][1]["original"] == "metforman"
    assert res["corrections"][1]["replacement"] == "metformin"


def test_hospital_alias_mapping(sample_dict_path: Path):
    """Verifies that domain alias 'hospital' maps to 'medical' dictionary."""
    text = "Patient prescribed paracetemol."
    res = correct_transcript(text, domain="hospital", dictionary_path=sample_dict_path)
    assert res["corrected_transcript"] == "Patient prescribed paracetamol."
    assert res["domain"] == "medical"
