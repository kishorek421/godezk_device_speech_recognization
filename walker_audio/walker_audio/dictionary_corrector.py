"""
Dictionary-based transcript post-processing and correction module.
Provides fast, deterministic domain-specific terminology correction without modifying
audio processing or underlying model weights.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Standard domain aliases (e.g., Hospital is an alias for Medical domain)
DOMAIN_ALIASES = {
    "hospital": "medical",
    "healthcare": "medical",
    "med": "medical",
    "edu": "education",
    "industrial": "industry",
    "gen": "general",
}


def find_dictionary_path(custom_path: str | Path | None = None) -> Path | None:
    """Locates the domain_dictionary.json file across known potential roots."""
    if custom_path is not None:
        p = Path(custom_path)
        return p if p.is_file() else None

    env_path = os.getenv("DOMAIN_DICTIONARY_PATH")
    if env_path:
        p = Path(env_path)
        return p if p.is_file() else None

    # Look relative to package source
    pkg_dir = Path(__file__).resolve().parent
    candidates = [
        pkg_dir.parent / "dictionaries" / "domain_dictionary.json",
        pkg_dir / "dictionaries" / "domain_dictionary.json",
        Path.cwd() / "walker_audio" / "dictionaries" / "domain_dictionary.json",
        Path.cwd() / "dictionaries" / "domain_dictionary.json",
        Path.cwd().parent / "walker_audio" / "dictionaries" / "domain_dictionary.json",
    ]

    for cand in candidates:
        if cand.is_file():
            return cand

    return None


def load_domain_dictionary(dictionary_path: str | Path | None = None) -> dict[str, dict[str, str]]:
    """Loads and validates the domain dictionary from a JSON file.

    Returns an empty dict if the file is missing or invalid.
    """
    resolved_path = find_dictionary_path(dictionary_path)
    if not resolved_path:
        logger.warning("Domain dictionary file not found; proceeding with empty dictionary.")
        return {}

    try:
        with resolved_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            logger.warning("Domain dictionary JSON is not an object; returning empty dictionary.")
            return {}
        # Normalize keys
        normalized: dict[str, dict[str, str]] = {}
        for domain_name, terms in data.items():
            if isinstance(terms, dict):
                normalized[str(domain_name).strip().lower()] = {
                    str(k).strip(): str(v).strip() for k, v in terms.items()
                }
            elif isinstance(terms, str):
                if "general" not in normalized:
                    normalized["general"] = {}
                normalized["general"][str(domain_name).strip()] = str(terms).strip()
        return normalized
    except Exception as exc:
        logger.warning("Failed to load domain dictionary from %s: %s", resolved_path, exc)
        return {}


def _match_case(original_match: str, replacement: str) -> str:
    """Preserves capitalization context where appropriate."""
    # If replacement explicitly contains uppercase acronyms (e.g., "PLC"), preserve replacement
    if any(c.isupper() for c in replacement) and not replacement.islower():
        return replacement
    # If original match was entirely UPPERCASE, match it
    if original_match.isupper():
        return replacement.upper()
    # If original match was Title Case / Capitalized, capitalize first letter
    if original_match and original_match[0].isupper():
        if len(replacement) > 1:
            return replacement[0].upper() + replacement[1:]
        return replacement.upper()
    return replacement


def correct_transcript(
    transcript: str,
    domain: str = "general",
    enabled: bool = True,
    dictionary_path: str | Path | None = None,
) -> dict[str, Any]:
    """Corrects known domain terminology in speech transcripts using a configured dictionary.

    Args:
        transcript: Raw predicted transcript from ASR/SALM.
        domain: Target domain (medical, education, industry, general).
        enabled: Flag to toggle dictionary correction on or off.
        dictionary_path: Optional explicit path to dictionary JSON.

    Returns:
        Structured dictionary containing original and corrected transcripts,
        the domain, corrections applied, and enabled status.
    """
    original_transcript = transcript if transcript is not None else ""
    canonical_domain = str(domain).strip().lower() if domain else "general"
    active_domain = DOMAIN_ALIASES.get(canonical_domain, canonical_domain)

    # Base result if disabled or empty
    if not enabled:
        return {
            "original_transcript": original_transcript,
            "corrected_transcript": original_transcript,
            "domain": active_domain,
            "corrections": [],
            "dictionary_enabled": False,
        }

    if not isinstance(original_transcript, str) or not original_transcript.strip():
        return {
            "original_transcript": original_transcript,
            "corrected_transcript": original_transcript,
            "domain": active_domain,
            "corrections": [],
            "dictionary_enabled": True,
        }

    dictionaries = load_domain_dictionary(dictionary_path)
    merged_terms: dict[str, str] = {}
    if "general" in dictionaries:
        merged_terms.update(dictionaries["general"])
    if active_domain in dictionaries:
        merged_terms.update(dictionaries[active_domain])

    if not merged_terms:
        # Unknown domain or no entries for this domain
        return {
            "original_transcript": original_transcript,
            "corrected_transcript": original_transcript,
            "domain": active_domain,
            "corrections": [],
            "dictionary_enabled": True,
        }

    # Sort terms by length (descending) to avoid partial replacement of sub-phrases
    sorted_items = sorted(merged_terms.items(), key=lambda kv: len(kv[0]), reverse=True)

    corrected_text = original_transcript
    corrections_applied: list[dict[str, str]] = []

    for term, replacement in sorted_items:
        # Tokenize term to allow flexible whitespace between words in multi-word phrases
        tokens = [re.escape(tok) for tok in term.split() if tok]
        if not tokens:
            continue
        # Enforce word boundaries around entire phrase
        pattern_str = r"\b" + r"\s+".join(tokens) + r"\b"
        regex = re.compile(pattern_str, flags=re.IGNORECASE)

        # Track occurrences
        matches = list(regex.finditer(corrected_text))
        if not matches:
            continue

        for m in matches:
            matched_text = m.group(0)
            applied_rep = _match_case(matched_text, replacement)
            # Only record if it actually changes the text
            if matched_text != applied_rep:
                corrections_applied.append({
                    "original": matched_text,
                    "replacement": applied_rep,
                })

        # Apply replacement using matching function
        def _replace_callback(m: re.Match[str]) -> str:
            return _match_case(m.group(0), replacement)

        corrected_text = regex.sub(_replace_callback, corrected_text)

    return {
        "original_transcript": original_transcript,
        "corrected_transcript": corrected_text,
        "domain": active_domain,
        "corrections": corrections_applied,
        "dictionary_enabled": True,
    }
