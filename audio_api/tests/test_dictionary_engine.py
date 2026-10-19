"""Unit tests for the Standalone Dictionary-Based Correction Engine.

Tests cover:
- Test A: Known error (curiculum -> curriculum)
- Test B: Known error (derivasion -> derivation)
- Test C: Already-correct text
- Test D: Unrelated text
- Test E: Case preservation (uppercase, titlecase)
- Test F: Punctuation preservation (!, ., ?, commas, quotes)
- Test G: Word-boundary protection (nested substrings)
- Test H: Unknown domain error-handling policy
- Test I: Ambiguous match handling (rejection without modification)
- Test J: Empty input
- Additional tests: Multi-word alias, numbers, Unicode, multiple occurrences.
"""

import pytest
from dictionary_engine import (
    DOMAIN_DICTIONARY,
    DictionaryCorrectionEngine,
    correct_text,
    levenshtein_distance,
    string_similarity,
)


def test_a_known_error_curriculum():
    text = "The curiculum is updated."
    corrected, records = correct_text(text, domain="education")
    assert corrected == "The curriculum is updated."
    assert len(records) == 1
    assert records[0]["original"] == "curiculum"
    assert records[0]["replacement"] == "curriculum"
    assert records[0]["applied"] is True
    assert records[0]["reason"] == "alias_match"


def test_b_known_error_derivation():
    text = "Explain the derivasion."
    corrected, records = correct_text(text, domain="education")
    assert corrected == "Explain the derivation."
    assert len(records) == 1
    assert records[0]["original"] == "derivasion"
    assert records[0]["replacement"] == "derivation"
    assert records[0]["applied"] is True


def test_c_already_correct_text():
    text = "The curriculum is updated."
    corrected, records = correct_text(text, domain="education")
    assert corrected == "The curriculum is updated."
    assert len(records) == 0


def test_d_unrelated_text():
    text = "The meeting is tomorrow."
    corrected, records = correct_text(text, domain="education")
    assert corrected == "The meeting is tomorrow."
    assert len(records) == 0


def test_e_case_preservation():
    # UPPERCASE
    text_upper = "CURICULUM"
    corrected_upper, records_upper = correct_text(text_upper, domain="education")
    assert corrected_upper == "CURRICULUM"
    assert records_upper[0]["applied"] is True

    # TitleCase
    text_title = "Curiculum is essential."
    corrected_title, records_title = correct_text(text_title, domain="education")
    assert corrected_title == "Curriculum is essential."
    assert records_title[0]["applied"] is True


def test_f_punctuation_preservation():
    text = "Please explain the curiculum!"
    corrected, records = correct_text(text, domain="education")
    assert corrected == "Please explain the curriculum!"
    assert len(records) == 1

    text_complex = "Did you say (curiculum), or 'derivasion'?"
    corrected_complex, _ = correct_text(text_complex, domain="education")
    assert corrected_complex == "Did you say (curriculum), or 'derivation'?"


def test_g_word_boundary_protection():
    # 'curiculum' should not match inside 'recuriculum' or 'curiculumsystem'
    text = "The recuriculum was not affected by curiculumsystem."
    corrected, records = correct_text(text, domain="education")
    assert corrected == text
    assert len(records) == 0


def test_h_unknown_domain():
    text = "The curiculum is updated."
    # Pass an unsupported domain 'astronomy'
    corrected, records = correct_text(text, domain="astronomy")
    # Verify engine follows documented error-handling policy without silently using the wrong dictionary
    assert corrected == text
    assert len(records) == 1
    assert records[0]["applied"] is False
    assert "unknown_domain" in records[0]["reason"]


def test_i_ambiguous_match():
    # Construct a custom dictionary where an alias is ambiguous between two canonical terms
    ambiguous_dict = {
        "education": {
            "course": ["unit_item"],
            "module": ["unit_item"],
        }
    }
    engine = DictionaryCorrectionEngine(dictionary=ambiguous_dict)
    text = "Please review unit_item carefully."
    corrected, records = engine.correct_text(text, domain="education")
    # Ambiguous candidate must NOT be modified
    assert corrected == text
    assert len(records) == 1
    assert records[0]["applied"] is False
    assert "ambiguous_match" in records[0]["reason"]


def test_j_empty_input():
    corrected, records = correct_text("", domain="education")
    assert corrected == ""
    assert records == []


def test_multi_word_alias():
    text = "We studied photo synthesis today."
    corrected, records = correct_text(text, domain="education")
    assert corrected == "We studied photosynthesis today."
    assert len(records) == 1
    assert records[0]["applied"] is True
    assert records[0]["original"].lower() == "photo synthesis"
    assert records[0]["replacement"].lower() == "photosynthesis"


def test_hospital_domain():
    text = "The patient has hyper tension and needs perscription."
    corrected, records = correct_text(text, domain="hospital")
    assert corrected == "The patient has hypertension and needs prescription."
    assert len(records) == 2


def test_industry_domain():
    text = "Regular maintainance of machinary is required."
    corrected, records = correct_text(text, domain="industry")
    assert corrected == "Regular maintenance of machinery is required."
    assert len(records) == 2


def test_general_domain_leaves_text_intact():
    text = "The curiculum is updated."
    corrected, records = correct_text(text, domain="general")
    assert corrected == text
    assert records == []


def test_numbers_and_unicode():
    text = "Formula 42: derivasion of π ≈ 3.1415."
    corrected, records = correct_text(text, domain="education")
    assert corrected == "Formula 42: derivation of π ≈ 3.1415."
    assert len(records) == 1


def test_fuzzy_matching_optional_conservative():
    # With fuzzy disabled by default, unknown spelling typo is not touched
    text = "The curriculm was reviewed."
    corrected_default, records_default = correct_text(text, domain="education", enable_fuzzy=False)
    assert corrected_default == text
    assert records_default == []

    # With conservative fuzzy matching enabled (threshold 0.88)
    corrected_fuzzy, records_fuzzy = correct_text(
        text, domain="education", enable_fuzzy=True, similarity_threshold=0.88
    )
    assert corrected_fuzzy == "The curriculum was reviewed."
    assert len(records_fuzzy) == 1
    assert records_fuzzy[0]["applied"] is True
    assert records_fuzzy[0]["reason"] == "fuzzy_match"
