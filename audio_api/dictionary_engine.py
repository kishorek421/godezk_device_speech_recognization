"""Standalone Dictionary-Based Text Correction Engine for Speech-to-Text.

This module provides post-processing correction for domain-specific speech recognition
errors. It does not train or fine-tune ASR models, nor does it use LLMs or vector databases.
All corrections are deterministic, explainable, and governed by strict safety rules:
- Already-correct text is left unchanged.
- Unrelated text is left unchanged.
- Case and punctuation are preserved.
- Word boundaries are strictly enforced.
- Ambiguous candidate matches are rejected (applied=False).
- Unsupported domains follow documented error policy without fallback substitution.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class CorrectionRecord:
    """Detailed record of a single proposed or evaluated correction."""

    original: str
    replacement: str
    applied: bool
    reason: str
    similarity_score: float | None = None
    start_pos: int | None = None
    end_pos: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Default Domain Dictionary
# Mapping: domain -> canonical_term -> list of observed incorrect aliases / ASR errors
DOMAIN_DICTIONARY: dict[str, dict[str, list[str]]] = {
    "education": {
        "curriculum": ["curiculum", "curriculam"],
        "derivation": ["derivasion"],
        "photosynthesis": ["photo synthesis"],
        "pedagogy": ["pedagadgy", "pedagagy"],
        "syllabus": ["syllubus", "sylabus"],
    },
    "hospital": {
        "diagnosis": ["diagonsis", "diagnosys"],
        "prescription": ["perscription", "prescripcion"],
        "radiology": ["radiologi", "radio logy"],
        "hypertension": ["hyper tension", "hyper tenshon"],
        "arrhythmia": ["arrythmia", "arrhytmia"],
    },
    "industry": {
        "manufacturing": ["manufaturing", "manufactoring"],
        "maintenance": ["maintainance", "maintenence"],
        "inspection": ["inspeksion", "in spection"],
        "calibration": ["calibaration", "calibrasion"],
        "machinery": ["machinary"],
    },
}

# Domain aliases for user convenience (e.g., 'medical' maps to 'hospital')
DOMAIN_ALIASES: dict[str, str] = {
    "medical": "hospital",
    "healthcare": "hospital",
    "ed": "education",
    "industrial": "industry",
}


def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate the Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def string_similarity(s1: str, s2: str) -> float:
    """Compute normalized similarity metric in [0.0, 1.0] based on Levenshtein distance.

    NOTE: This is a string edit-distance similarity metric, NOT a calibrated
    probability.
    """
    s1_lower, s2_lower = s1.lower(), s2.lower()
    if s1_lower == s2_lower:
        return 1.0
    max_len = max(len(s1_lower), len(s2_lower))
    if max_len == 0:
        return 1.0
    dist = levenshtein_distance(s1_lower, s2_lower)
    return round(max(0.0, 1.0 - (dist / max_len)), 4)


def preserve_case(source_sample: str, target: str) -> str:
    """Preserve case pattern (UPPERCASE, Title Case, lowercase) from source into target."""
    if not source_sample:
        return target

    # All uppercase: CURICULUM -> CURRICULUM
    if source_sample.isupper():
        return target.upper()

    # Title Case: Curiculum -> Curriculum or Photo Synthesis -> Photosynthesis
    if source_sample.istitle():
        words = target.split(" ")
        return " ".join(w.capitalize() for w in words)

    # All lowercase: curiculum -> curriculum
    if source_sample.islower():
        return target.lower()

    # Default to target as defined
    return target


class DictionaryCorrectionEngine:
    """Configurable, rule-based text correction engine with strict safety guarantees."""

    def __init__(
        self,
        dictionary: dict[str, dict[str, list[str]]] | None = None,
        similarity_threshold: float = 0.88,
        enable_fuzzy: bool = False,
    ) -> None:
        self.dictionary = dictionary if dictionary is not None else DOMAIN_DICTIONARY
        self.similarity_threshold = similarity_threshold
        self.enable_fuzzy = enable_fuzzy

    def resolve_domain(self, domain: str) -> str | None:
        """Resolve a domain string against configured domains and aliases.

        Returns canonical domain name if supported, else None.
        """
        if not domain:
            return None
        d_lower = domain.strip().lower()
        if d_lower in self.dictionary:
            return d_lower
        if d_lower in DOMAIN_ALIASES and DOMAIN_ALIASES[d_lower] in self.dictionary:
            return DOMAIN_ALIASES[d_lower]
        return None

    def _build_alias_index(
        self, domain_dict: dict[str, list[str]]
    ) -> tuple[dict[str, list[str]], set[str]]:
        """Build alias index mapping lowercase alias -> list of canonical terms.

        Also returns the set of lowercase canonical terms to prevent overriding
        already-correct terms.
        """
        alias_to_canonical: dict[str, list[str]] = {}
        canonical_terms_lower: set[str] = set()

        for canonical, aliases in domain_dict.items():
            c_low = canonical.lower()
            canonical_terms_lower.add(c_low)
            for alias in aliases:
                a_low = alias.lower()
                if a_low not in alias_to_canonical:
                    alias_to_canonical[a_low] = []
                if canonical not in alias_to_canonical[a_low]:
                    alias_to_canonical[a_low].append(canonical)

        return alias_to_canonical, canonical_terms_lower

    def correct_text(
        self,
        text: str,
        domain: str = "general",
        enable_fuzzy: bool | None = None,
        similarity_threshold: float | None = None,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Correct text using domain dictionary.

        Returns (corrected_text, list_of_correction_records).
        """
        if not text:
            return "", []

        active_fuzzy = self.enable_fuzzy if enable_fuzzy is None else enable_fuzzy
        active_threshold = (
            self.similarity_threshold if similarity_threshold is None else similarity_threshold
        )

        canonical_domain = self.resolve_domain(domain)
        # If domain is 'general' or unrecognized domain
        if canonical_domain is None:
            if domain.lower() in ("general", "none", "all", ""):
                # General domain: no domain-specific dictionary active; return original text cleanly
                return text, []

            # Unknown domain: follow documented error policy without fallback substitution
            record = CorrectionRecord(
                original=text,
                replacement=text,
                applied=False,
                reason=f"unknown_domain: '{domain}' is not configured",
                similarity_score=None,
            )
            return text, [record.to_dict()]

        domain_dict = self.dictionary[canonical_domain]
        alias_index, canonical_set = self._build_alias_index(domain_dict)

        records: list[CorrectionRecord] = []

        # Sort aliases by length descending so multi-word aliases match before single words
        # e.g., "photo synthesis" before "photo"
        sorted_aliases = sorted(alias_index.keys(), key=len, reverse=True)

        current_text = text

        # 1. Exact alias replacement with word-boundary awareness and case preservation
        for alias_low in sorted_aliases:
            canonicals = alias_index[alias_low]

            # Construct word-boundary-aware regex: \b...\b
            # Handle potential multi-word spacing flexibility
            pattern_parts = [re.escape(word) for word in alias_low.split()]
            regex_str = r"\b" + r"\s+".join(pattern_parts) + r"\b"
            pattern = re.compile(regex_str, re.IGNORECASE)

            # Find all occurrences
            matches = list(pattern.finditer(current_text))
            if not matches:
                continue

            # Check for ambiguity: multiple canonical terms configured for the same alias
            if len(canonicals) > 1:
                for match in matches:
                    matched_str = match.group(0)
                    records.append(
                        CorrectionRecord(
                            original=matched_str,
                            replacement=matched_str,
                            applied=False,
                            reason=f"ambiguous_match: alias '{alias_low}' maps to multiple candidates {canonicals}",
                            similarity_score=1.0,
                            start_pos=match.start(),
                            end_pos=match.end(),
                        )
                    )
                # Do NOT replace ambiguous aliases
                continue

            canonical_target = canonicals[0]

            # If the alias is already identical to the canonical term, do nothing
            if alias_low == canonical_target.lower():
                continue

            # Perform replacement from right to left to preserve offsets if multiple
            for match in reversed(matches):
                matched_str = match.group(0)

                # Safety rule: Do not replace if it is already a known canonical term
                if matched_str.lower() in canonical_set:
                    continue

                replacement_str = preserve_case(matched_str, canonical_target)

                # Record successful correction
                records.append(
                    CorrectionRecord(
                        original=matched_str,
                        replacement=replacement_str,
                        applied=True,
                        reason="alias_match",
                        similarity_score=1.0,
                        start_pos=match.start(),
                        end_pos=match.end(),
                    )
                )

                # Apply replacement
                current_text = (
                    current_text[: match.start()] + replacement_str + current_text[match.end() :]
                )

        # 2. Conservative Fuzzy Matching (optional)
        if active_fuzzy:
            # Tokenize words while preserving punctuation and whitespace
            tokens = re.finditer(r"\b[A-Za-z0-9_-]+\b", current_text)
            fuzzy_replacements = []

            for token_match in tokens:
                token_str = token_match.group(0)
                token_low = token_str.lower()

                # Rule 1: Never modify already-canonical words
                if token_low in canonical_set:
                    continue

                # Rule 4 & 5: Check candidate similarity against canonical terms and aliases
                candidates: list[tuple[str, float]] = []
                for canonical in domain_dict.keys():
                    sim = string_similarity(token_low, canonical.lower())
                    if sim >= active_threshold:
                        candidates.append((canonical, sim))

                if not candidates:
                    continue

                # Sort by similarity descending
                candidates.sort(key=lambda x: x[1], reverse=True)
                top_candidate, top_score = candidates[0]

                # Ambiguity check: if top 2 candidates have equal score, reject
                if len(candidates) > 1 and candidates[0][1] == candidates[1][1]:
                    records.append(
                        CorrectionRecord(
                            original=token_str,
                            replacement=token_str,
                            applied=False,
                            reason=f"ambiguous_match: equal similarity score for '{candidates[0][0]}' and '{candidates[1][0]}'",
                            similarity_score=top_score,
                            start_pos=token_match.start(),
                            end_pos=token_match.end(),
                        )
                    )
                    continue

                replacement_str = preserve_case(token_str, top_candidate)
                fuzzy_replacements.append((token_match.start(), token_match.end(), token_str, replacement_str, top_score))

            # Apply fuzzy replacements from right to left
            for start, end, orig, repl, score in reversed(fuzzy_replacements):
                records.append(
                    CorrectionRecord(
                        original=orig,
                        replacement=repl,
                        applied=True,
                        reason="fuzzy_match",
                        similarity_score=score,
                        start_pos=start,
                        end_pos=end,
                    )
                )
                current_text = current_text[:start] + repl + current_text[end:]

        # Sort records chronologically by start_pos where available
        records_dict = [r.to_dict() for r in records]
        return current_text, records_dict


# Module-level convenience function matching the requested interface
_DEFAULT_ENGINE = DictionaryCorrectionEngine()


def correct_text(
    text: str,
    domain: str = "general",
    enable_fuzzy: bool = False,
    similarity_threshold: float = 0.88,
    dictionary: dict[str, dict[str, list[str]]] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Correct text using domain dictionary.

    Args:
        text: The input transcript or sentence.
        domain: Domain key (e.g., 'education', 'hospital', 'industry', 'general').
        enable_fuzzy: Whether to enable conservative fuzzy matching. Default is False.
        similarity_threshold: Minimum string similarity threshold for fuzzy matching.
        dictionary: Optional custom dictionary mapping domain -> canonical -> aliases.

    Returns:
        tuple of (corrected_text, list_of_correction_records)
    """
    if dictionary is not None:
        engine = DictionaryCorrectionEngine(
            dictionary=dictionary,
            similarity_threshold=similarity_threshold,
            enable_fuzzy=enable_fuzzy,
        )
        return engine.correct_text(text, domain=domain)

    return _DEFAULT_ENGINE.correct_text(
        text,
        domain=domain,
        enable_fuzzy=enable_fuzzy,
        similarity_threshold=similarity_threshold,
    )
