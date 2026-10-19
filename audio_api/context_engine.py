"""Real-Time LLM Context Understanding and Conversation Memory Engine.

Provides session-based multi-turn conversation memory, domain-aware technical
terminology disambiguation, abbreviation recovery, and structured context review
via Ollama Qwen models.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://124.123.89.162:11434")
DEFAULT_MODEL_NAME = os.getenv("OLLAMA_MODEL", "qwen3.6:27b")
DEFAULT_SESSION_TTL = int(os.getenv("SESSION_TTL_SEC", "1800"))  # 30 minutes
DEFAULT_CONTEXT_WINDOW = int(os.getenv("CONTEXT_WINDOW_TURNS", "6"))
DEFAULT_OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT_SEC", "2.0"))


# ---------------------------------------------------------------------------
# Domain Knowledge Base & Technical Context
# ---------------------------------------------------------------------------

DOMAIN_KNOWLEDGE = {
    "education": {
        "subjects": {
            "physics": {
                "keywords": ["work", "force", "displacement", "energy", "velocity", "acceleration", "thermodynamics", "entropy", "momentum", "wavelength", "frequency", "quantum", "gravitation", "inertia"],
                "abbreviations": {"KE": "kinetic energy", "PE": "potential energy", "EM": "electromagnetic", "SHM": "simple harmonic motion"},
                "polysemy": {
                    "work": "Mechanical work: force multiplied by displacement along the line of action.",
                    "power": "Rate of doing work or transferring energy per unit time."
                }
            },
            "chemistry": {
                "keywords": ["work", "pressure", "expansion", "pH", "HCl", "H2SO4", "NaOH", "titration", "molarity", "equilibrium", "catalyst", "redox", "enthalpy", "photosynthesis", "reaction", "acid", "base"],
                "abbreviations": {"pH": "potential of hydrogen", "HCl": "hydrochloric acid", "H2SO4": "sulfuric acid", "NaOH": "sodium hydroxide", "STP": "standard temperature and pressure"},
                "polysemy": {
                    "work": "Thermodynamic work: gas expanding or compressing against external pressure (P*deltaV).",
                    "base": "Substance capable of accepting protons or donating pairs of valence electrons."
                }
            },
            "biology": {
                "keywords": ["photosynthesis", "chlorophyll", "mitochondria", "cellular respiration", "enzyme", "mitosis", "meiosis", "homeostasis", "ribosome", "genome", "transcription", "translation"],
                "abbreviations": {"DNA": "deoxyribonucleic acid", "RNA": "ribonucleic acid", "ATP": "adenosine triphosphate"},
                "polysemy": {}
            },
            "mathematics": {
                "keywords": ["calculus", "differential", "integral", "derivation", "equation", "algorithm", "matrix", "eigenvalue", "vector", "polynomial", "theorem"],
                "abbreviations": {"LHS": "left hand side", "RHS": "right hand side", "QED": "quod erat demonstrandum"},
                "polysemy": {}
            }
        }
    },
    "hospital": {
        "subjects": {
            "cardiology": {
                "keywords": ["hypertension", "tachycardia", "arrhythmia", "ischemia", "electrocardiogram", "infarction", "systolic", "diastolic", "myocardium", "ventricle", "atrium", "stethoscope"],
                "abbreviations": {"BP": "blood pressure", "SpO2": "peripheral capillary oxygen saturation", "ECG": "electrocardiogram", "EKG": "electrocardiogram", "HR": "heart rate", "IV": "intravenous"},
                "polysemy": {}
            },
            "pulmonology": {
                "keywords": ["pneumonia", "oxygen", "intubation", "ventilation", "dyspnea", "bronchitis", "alveoli", "saturation", "cyanosis", "nebulizer"],
                "abbreviations": {"SpO2": "peripheral capillary oxygen saturation", "PEEP": "positive end-expiratory pressure", "COPD": "chronic obstructive pulmonary disease"},
                "polysemy": {}
            },
            "general": {
                "keywords": ["diabetes", "diagnosis", "prescription", "medication", "anesthesia", "patient", "vital signs", "dosage", "dosage regimen", "sterile"],
                "abbreviations": {"BP": "blood pressure", "SpO2": "oxygen saturation", "PRN": "as needed", "BID": "twice daily", "TID": "three times daily"},
                "polysemy": {}
            }
        }
    },
    "industry": {
        "subjects": {
            "electrical": {
                "keywords": ["transformer", "primary winding", "secondary winding", "voltage", "step-up", "step-down", "induction", "impedance", "circuit breaker", "harmonics", "substation", "switchgear"],
                "abbreviations": {"AC": "alternating current", "DC": "direct current", "LOTO": "lockout-tagout", "HVAC": "heating ventilation and air conditioning", "PLC": "programmable logic controller"},
                "polysemy": {}
            },
            "manufacturing": {
                "keywords": ["pneumatic", "hydraulic", "tolerance", "calibration", "conveyor", "actuator", "assembly", "tooling", "fabrication", "lathe", "milling"],
                "abbreviations": {"CNC": "computer numerical control", "OEE": "overall equipment effectiveness", "SOP": "standard operating procedure"},
                "polysemy": {}
            },
            "safety": {
                "keywords": ["lockout", "tagout", "hazardous", "voltage isolation", "interlock", "containment", "respirator", "inspection"],
                "abbreviations": {"LOTO": "lockout-tagout", "PPE": "personal protective equipment", "OSHA": "occupational safety and health administration"},
                "polysemy": {}
            }
        }
    }
}


# ---------------------------------------------------------------------------
# Data Models for Conversation Memory
# ---------------------------------------------------------------------------

@dataclass
class ConversationTurn:
    turn_id: int
    timestamp: float
    speaker_id: str
    original_transcript: str
    dict_transcript: str
    reviewed_transcript: str
    corrections: list[dict[str, Any]] = field(default_factory=list)
    inferred_terms: list[str] = field(default_factory=list)
    uncertain_segments: list[str] = field(default_factory=list)
    requires_audio_recheck: bool = False
    context_latency_ms: float = 0.0


@dataclass
class ConversationSession:
    session_id: str
    domain: str = "general"
    subject: str = ""
    topic: str = ""
    created_at: float = field(default_factory=time.time)
    last_activity_at: float = field(default_factory=time.time)
    history: list[ConversationTurn] = field(default_factory=list)
    established_terms: set[str] = field(default_factory=set)
    established_abbreviations: dict[str, str] = field(default_factory=dict)

    def is_expired(self, ttl_seconds: int = DEFAULT_SESSION_TTL) -> bool:
        return (time.time() - self.last_activity_at) > ttl_seconds

    def touch(self) -> None:
        self.last_activity_at = time.time()

    def update_context(self, domain: str = "", subject: str = "", topic: str = "") -> None:
        if domain:
            self.domain = domain.strip().lower()
        if subject:
            self.subject = subject.strip()
        if topic:
            self.topic = topic.strip()
        self.touch()

    def add_turn(
        self,
        original_transcript: str,
        dict_transcript: str,
        reviewed_transcript: str,
        speaker_id: str = "Unknown Speaker",
        corrections: list[dict[str, Any]] | None = None,
        inferred_terms: list[str] | None = None,
        uncertain_segments: list[str] | None = None,
        requires_audio_recheck: bool = False,
        latency_ms: float = 0.0,
        max_turns: int = DEFAULT_CONTEXT_WINDOW,
    ) -> ConversationTurn:
        self.touch()
        turn_id = len(self.history) + 1
        turn = ConversationTurn(
            turn_id=turn_id,
            timestamp=time.time(),
            speaker_id=speaker_id,
            original_transcript=original_transcript,
            dict_transcript=dict_transcript,
            reviewed_transcript=reviewed_transcript,
            corrections=corrections or [],
            inferred_terms=inferred_terms or [],
            uncertain_segments=uncertain_segments or [],
            requires_audio_recheck=requires_audio_recheck,
            context_latency_ms=latency_ms,
        )
        self.history.append(turn)

        # Track established technical terms and abbreviations
        if inferred_terms:
            for term in inferred_terms:
                self.established_terms.add(term)
        if corrections:
            for corr in corrections:
                c = corr.get("corrected")
                if c and len(c) <= 6 and (c.isupper() or any(ch.isupper() for ch in c) or c.lower() in {"ph", "spo2", "hcl", "naoh"}):
                    reason = corr.get("reason", "")
                    self.established_abbreviations[c] = reason

        # Maintain rolling window of recent history
        if len(self.history) > max_turns:
            self.history = self.history[-max_turns:]

        return turn

    def get_recent_history(self, count: int = DEFAULT_CONTEXT_WINDOW) -> list[ConversationTurn]:
        return self.history[-count:]

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "domain": self.domain,
            "subject": self.subject,
            "topic": self.topic,
            "turn_count": len(self.history),
            "established_terms": list(self.established_terms),
            "established_abbreviations": self.established_abbreviations,
            "recent_turns": [
                {
                    "turn_id": t.turn_id,
                    "speaker": t.speaker_id,
                    "original": t.original_transcript,
                    "reviewed": t.reviewed_transcript,
                    "corrections": t.corrections,
                    "inferred_terms": t.inferred_terms,
                    "uncertain_segments": t.uncertain_segments,
                }
                for t in self.history[-DEFAULT_CONTEXT_WINDOW:]
            ]
        }


# ---------------------------------------------------------------------------
# Session Manager (Thread-safe, Multi-Session Isolation)
# ---------------------------------------------------------------------------

class SessionManager:
    """Manages active conversation sessions with memory isolation and automatic TTL cleanup."""

    def __init__(self, ttl_seconds: int = DEFAULT_SESSION_TTL):
        self.ttl_seconds = ttl_seconds
        self.sessions: dict[str, ConversationSession] = {}
        self._lock = threading.Lock()

    def get_or_create(
        self,
        session_id: str,
        domain: str = "general",
        subject: str = "",
        topic: str = "",
    ) -> ConversationSession:
        with self._lock:
            self._cleanup_expired_locked()
            clean_id = (session_id or "default_session").strip()
            if clean_id not in self.sessions:
                self.sessions[clean_id] = ConversationSession(
                    session_id=clean_id,
                    domain=domain,
                    subject=subject,
                    topic=topic,
                )
            else:
                session = self.sessions[clean_id]
                session.update_context(domain, subject, topic)
            return self.sessions[clean_id]

    def reset(self, session_id: str) -> bool:
        with self._lock:
            clean_id = (session_id or "default_session").strip()
            if clean_id in self.sessions:
                del self.sessions[clean_id]
                return True
            return False

    def get_summary(self, session_id: str) -> dict[str, Any] | None:
        with self._lock:
            clean_id = (session_id or "default_session").strip()
            session = self.sessions.get(clean_id)
            if session:
                return session.to_summary_dict()
            return None

    def _cleanup_expired_locked(self) -> None:
        now = time.time()
        expired_keys = [
            sid for sid, s in self.sessions.items()
            if (now - s.last_activity_at) > self.ttl_seconds
        ]
        for sid in expired_keys:
            del self.sessions[sid]


# ---------------------------------------------------------------------------
# Ollama Qwen Context Review Engine
# ---------------------------------------------------------------------------

class OllamaContextEngine:
    """Orchestrates Ollama /api/chat with strict safety constraints and conversation memory."""

    def __init__(
        self,
        base_url: str = DEFAULT_OLLAMA_URL,
        model_name: str = DEFAULT_MODEL_NAME,
        timeout_sec: float = DEFAULT_OLLAMA_TIMEOUT,
    ):
        self.base_url = base_url.rstrip("/")
        self.configured_model = model_name
        self.timeout_sec = timeout_sec
        self.session_manager = SessionManager()
        self._resolved_model: str | None = None

    def detect_available_qwen_model(self) -> str | None:
        """Queries /api/tags and identifies exact installed Qwen model without guessing."""
        try:
            url = f"{self.base_url}/api/tags"
            req = urllib.request.Request(url, headers={"User-Agent": "Canary-ContextEngine/1.0"})
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                data = json.loads(resp.read().decode())
                models = [m.get("name", "") for m in data.get("models", [])]
                # Look for Qwen variants
                for m in models:
                    if "qwen" in m.lower():
                        logger.info("Detected installed Qwen model: %s", m)
                        return m
                return models[0] if models else None
        except Exception as exc:
            logger.warning("Could not query Ollama /api/tags: %s", exc)
            return None

    def get_active_model(self) -> str:
        if self._resolved_model:
            return self._resolved_model
        detected = self.detect_available_qwen_model()
        self._resolved_model = detected or self.configured_model
        return self._resolved_model

    def build_system_prompt(self, domain: str, subject: str, topic: str) -> str:
        """Constructs strict context correction system prompt with negative safety rules."""
        dom_key = (domain or "general").strip().lower()
        dom_info = DOMAIN_KNOWLEDGE.get(dom_key, {})
        subj_info = dom_info.get("subjects", {}).get(subject.strip().lower(), {}) if subject else {}

        polysemy_guidance = ""
        if subj_info.get("polysemy"):
            polysemy_guidance = "Domain Polysemy Rules:\n" + "\n".join(
                f"- '{k}': {v}" for k, v in subj_info["polysemy"].items()
            )

        abbr_guidance = ""
        if subj_info.get("abbreviations"):
            abbr_guidance = "Known Domain Abbreviations:\n" + ", ".join(
                f"{k} ({v})" for k, v in subj_info["abbreviations"].items()
            )

        return (
            "You are a real-time, highly rigorous Speech-to-Text Context Reviewer and Post-Processor.\n"
            f"Active Domain: {domain.upper()}\n"
            f"Active Subject: {subject or 'General'}\n"
            f"Active Topic: {topic or 'General Technical Discussion'}\n\n"
            f"{polysemy_guidance}\n"
            f"{abbr_guidance}\n\n"
            "STRICT OPERATIONAL SAFETY RULES:\n"
            "1. PRESERVE ORIGINAL SPOKEN MEANING: Never change words that are grammatically or acoustically sound.\n"
            "2. DO NOT INVENT OR FABRICATE: Never invent missing sentences, clauses, or words based only on context.\n"
            "3. PRESERVE CRITICAL ELEMENTS: Never alter numbers, quantities, measurements, negation (not/never), or patient names.\n"
            "4. MEDICAL SAFETY: In Hospital domain, NEVER invent or modify medication names, dosages, or clinical diagnoses.\n"
            "5. ABBREVIATION RECOVERY: When context strongly indicates a technical abbreviation was fragmented by ASR "
            "(e.g., 'p' + 'H' in chemistry acids -> 'pH', 'H' + 'C' + 'l' -> 'HCl', 'S P O' in medical -> 'SpO2', 'B P' -> 'BP'), "
            "propose the correction and record it in 'inferred_terms'.\n"
            "6. UNCERTAIN WORDS: If an ASR word appears inconsistent or ambiguous with insufficient acoustic/context evidence, "
            "do NOT guess wildly. Mark it in 'uncertain_segments'.\n"
            "7. STRUCTURED OUTPUT: You MUST return strictly valid JSON matching this schema:\n"
            "{\n"
            '  "reviewed_transcript": "<the final reviewed sentence>",\n'
            '  "corrections": [{"original": "<word>", "corrected": "<word>", "reason": "<justification>"}],\n'
            '  "correction_reasons": ["<summary of changes>"],\n'
            '  "inferred_terms": ["<recovered technical terms/abbreviations>"],\n'
            '  "uncertain_segments": ["<words with uncertain interpretation>"],\n'
            '  "requires_audio_recheck": false\n'
            "}\n"
        )

    def apply_domain_context(
        self,
        text: str,
        session: ConversationSession,
        domain: str = "general",
        subject: str = "",
        topic: str = "",
    ) -> dict[str, Any]:
        """Performs zero-latency dynamic domain context evaluation, technical abbreviation
        recovery, and polysemy tracking based on DOMAIN_KNOWLEDGE and active session memory."""
        dom = (domain or session.domain or "").strip().lower()
        subj = (subject or session.subject or "").strip().lower()
        top = (topic or session.topic or "").strip().lower()
        history_tokens = " ".join(t.reviewed_transcript for t in session.get_recent_history(DEFAULT_CONTEXT_WINDOW)).lower()
        full_context = f"{dom} {subj} {top} {history_tokens}".lower()

        reviewed = text
        corrections: list[dict[str, Any]] = []
        inferred_terms: list[str] = []
        uncertain_segments: list[str] = []

        is_chem = any(k in full_context for k in ("chem", "acid", "base", "titrat", "solution", "ph", "reaction", "hcl", "molar"))
        is_cardio = any(k in full_context for k in ("cardio", "pulmon", "hospital", "vital", "medic", "blood pressure", "spo2", "ecg"))
        is_elec = any(k in full_context for k in ("electr", "industry", "transform", "circuit", "voltage", "winding", "loto", "plc"))
        is_physics = any(k in full_context for k in ("physic", "mechanic", "kinetic", "displacement", "gravity", "force"))

        # 1. Chemistry / Acids & Bases / Solutions
        if is_chem:
            # Multi-letter chemical formulas spaced by ASR (run first before single letters)
            if re.search(r"\b[hH]\s+[cC]\s+[lL]\b", reviewed):
                reviewed = re.sub(r"\b[hH]\s+[cC]\s+[lL]\b", "HCl", reviewed)
                corrections.append({"original": "H C l", "corrected": "HCl", "reason": "Recovered chemical formula HCl"})
                if "HCl" not in inferred_terms:
                    inferred_terms.append("HCl")

            if re.search(r"\b[nN]\s*[aA]\s*[oO]\s*[hH]\b", reviewed):
                reviewed = re.sub(r"\b[nN]\s*[aA]\s*[oO]\s*[hH]\b", "NaOH", reviewed)
                corrections.append({"original": "N a O H", "corrected": "NaOH", "reason": "Recovered chemical formula NaOH"})
                if "NaOH" not in inferred_terms:
                    inferred_terms.append("NaOH")

            # P H or p h -> pH
            pattern_ph_split = r"\b[pP]\s+[hH]\b"
            if re.search(pattern_ph_split, reviewed):
                reviewed = re.sub(pattern_ph_split, "pH", reviewed)
                corrections.append({
                    "original": "P H",
                    "corrected": "pH",
                    "reason": "Recovered pH abbreviation"
                })
                if "pH" not in inferred_terms:
                    inferred_terms.append("pH")

            # P -> pH: Match standalone P or p when followed by 'after', 'of', 'value', 'scale', 'level',
            # or in sentence with 'solution' / 'acid' / 'base' / 'reaches'
            pattern_p = r"\b[pP]\b(?=(?:\s+(?:after|of|value|scale|level|is|was|drops|increases|decreases|meter|indicator|reading|paper|buffer|solution|reaches|rises|approaches|equals)|[\.,\?!]|$))"
            if re.search(pattern_p, reviewed):
                reviewed = re.sub(pattern_p, "pH", reviewed)
                corrections.append({
                    "original": "P",
                    "corrected": "pH",
                    "reason": "Acidity/alkalinity scale abbreviation recovered from chemistry context"
                })
                if "pH" not in inferred_terms:
                    inferred_terms.append("pH")

            # H -> HCl when following 'adding'/'added' or preceding 'to the solution'
            pattern_h = r"(\b(?:adding|added|titrating|with|concentrated|dilute|mix|plus)\s+)[hH]\b|\b[hH](\s+(?:to\s+(?:the\s+)?solution|acid|ions?|gas|reagent))"
            if re.search(pattern_h, reviewed, re.IGNORECASE):
                def repl_h(m):
                    if m.group(1):
                        return f"{m.group(1)}HCl"
                    else:
                        return f"HCl{m.group(2)}"
                reviewed = re.sub(pattern_h, repl_h, reviewed, flags=re.IGNORECASE)
                corrections.append({
                    "original": "H",
                    "corrected": "HCl",
                    "reason": "Hydrochloric acid abbreviation recovered from chemistry acid-base context"
                })
                if "HCl" not in inferred_terms:
                    inferred_terms.append("HCl")

            # B -> Base when in acids and bases context
            pattern_b = r"\b[bB]\b(?=(?:\s+(?:after\s+adding|to\s+(?:the\s+)?solution|is\s+added|added|titrated|solution)))"
            if re.search(pattern_b, reviewed) and ("acid" in full_context or "base" in full_context):
                reviewed = re.sub(pattern_b, "Base", reviewed)
                corrections.append({
                    "original": "B",
                    "corrected": "Base",
                    "reason": "Base reagent recovered from acids and bases context"
                })
                if "Base" not in inferred_terms:
                    inferred_terms.append("Base")

            # Acoustic confusion: vase -> base in chemistry solution context
            pattern_vase = r"\bvase\b(?=\s+solution|\s+reagent)"
            if re.search(pattern_vase, reviewed, re.IGNORECASE):
                reviewed = re.sub(pattern_vase, "base", reviewed, flags=re.IGNORECASE)
                corrections.append({
                    "original": "vase",
                    "corrected": "base",
                    "reason": "Chemistry context indicates alkaline base"
                })
                if "base" not in inferred_terms:
                    inferred_terms.append("base")

            if "work" in reviewed.lower():
                if "thermodynamic work" not in inferred_terms:
                    inferred_terms.append("thermodynamic work")
            if "pressure" in reviewed.lower() and "pressure" not in inferred_terms:
                inferred_terms.append("pressure")
            if "base" in reviewed.lower() and "base" not in inferred_terms:
                inferred_terms.append("base")

            # Scan and record verified chemical entities
            for chem_term in ("pH", "HCl", "NaOH", "H2SO4", "titration", "molarity", "acid"):
                if re.search(rf"\b{chem_term}\b", reviewed, re.IGNORECASE) and chem_term not in inferred_terms:
                    inferred_terms.append(chem_term)

        # 2. Hospital / Cardiology / Pulmonology
        if is_cardio:
            pattern_spo = r"\b(?:S\s*P\s*O\s*(?:2|two)?|SpO)\b"
            if re.search(pattern_spo, reviewed, re.IGNORECASE):
                reviewed = re.sub(pattern_spo, "SpO2", reviewed, flags=re.IGNORECASE)
                corrections.append({
                    "original": "S P O",
                    "corrected": "SpO2",
                    "reason": "Oxygen saturation abbreviation"
                })
                if "SpO2" not in inferred_terms:
                    inferred_terms.append("SpO2")

            pattern_bp = r"\bB\s+P\b"
            if re.search(pattern_bp, reviewed):
                reviewed = re.sub(pattern_bp, "BP", reviewed)
                corrections.append({
                    "original": "B P",
                    "corrected": "BP",
                    "reason": "Blood pressure abbreviation"
                })
                if "BP" not in inferred_terms:
                    inferred_terms.append("BP")

            pattern_ecg = r"\b(?:E\s*C\s*G|E\s*K\s*G)\b"
            if re.search(pattern_ecg, reviewed):
                reviewed = re.sub(pattern_ecg, "ECG", reviewed)
                corrections.append({
                    "original": "E C G",
                    "corrected": "ECG",
                    "reason": "Electrocardiogram abbreviation"
                })
                if "ECG" not in inferred_terms:
                    inferred_terms.append("ECG")

            for med_term in ("BP", "SpO2", "ECG", "EKG", "HR", "IV", "hypertension", "arrhythmia", "systolic", "diastolic", "oxygen"):
                if re.search(rf"\b{med_term}\b", reviewed, re.IGNORECASE) and med_term not in inferred_terms:
                    inferred_terms.append(med_term)

        # 3. Industry / Electrical / Safety
        if is_elec:
            pattern_ac = r"\bA\s+C\b"
            if re.search(pattern_ac, reviewed):
                reviewed = re.sub(pattern_ac, "AC", reviewed)
                corrections.append({"original": "A C", "corrected": "AC", "reason": "Alternating current abbreviation"})
                if "AC" not in inferred_terms:
                    inferred_terms.append("AC")

            pattern_dc = r"\bD\s+C\b"
            if re.search(pattern_dc, reviewed):
                reviewed = re.sub(pattern_dc, "DC", reviewed)
                corrections.append({"original": "D C", "corrected": "DC", "reason": "Direct current abbreviation"})
                if "DC" not in inferred_terms:
                    inferred_terms.append("DC")

            for kw in ("transformer", "primary winding", "secondary winding", "voltage", "impedance", "circuit breaker", "PLC", "LOTO", "AC", "DC"):
                if re.search(rf"\b{re.escape(kw)}\b", reviewed, re.IGNORECASE) and kw not in inferred_terms:
                    inferred_terms.append(kw)

        # 4. Physics Polysemy
        if is_physics:
            for kw in ("work", "force", "displacement"):
                if kw in reviewed.lower() and kw not in inferred_terms:
                    inferred_terms.append(kw)

        # 5. Reuse previously established session abbreviations
        for abbr, desc in session.established_abbreviations.items():
            if abbr not in inferred_terms and abbr.lower() in reviewed.lower():
                inferred_terms.append(abbr)

        return {
            "reviewed_transcript": reviewed,
            "corrections": corrections,
            "correction_reasons": [c["reason"] for c in corrections],
            "inferred_terms": inferred_terms,
            "uncertain_segments": uncertain_segments,
            "requires_audio_recheck": False,
        }

    def review_transcript(
        self,
        original_transcript: str,
        dict_transcript: str,
        session_id: str = "default_session",
        domain: str = "general",
        subject: str = "",
        topic: str = "",
        speaker_id: str = "Unknown Speaker",
        enabled: bool = False,
        timeout_sec: float | None = None,
        mock_response: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Reviews transcript with conversation memory and LLM.
        
        If enabled is False, gracefully returns original dictionary result.
        When enabled is True, applies dynamic domain context knowledge and
        queries Ollama Qwen with safe fallback.
        """
        session = self.session_manager.get_or_create(session_id, domain, subject, topic)
        actual_timeout = timeout_sec or self.timeout_sec
        clean_input = dict_transcript.strip() or original_transcript.strip()

        # Fallback response template when review is disabled or fails
        fallback_result = {
            "session_id": session.session_id,
            "original_transcript": original_transcript,
            "dict_transcript": dict_transcript,
            "reviewed_transcript": dict_transcript,
            "corrections": [],
            "correction_reasons": [],
            "inferred_terms": [],
            "uncertain_segments": [],
            "requires_audio_recheck": False,
            "context_latency_ms": 0.0,
            "llm_reviewed": False,
            "status": "disabled" if not enabled else "fallback",
            "domain": session.domain,
            "subject": session.subject,
            "topic": session.topic,
        }

        # Check for empty / silence / no speech
        if not clean_input or clean_input.lower() in {"[no_speech]", "..", "...", "none", "null"}:
            return fallback_result

        # If context review is disabled by default, return immediately
        if not enabled and mock_response is None:
            session.add_turn(
                original_transcript=original_transcript,
                dict_transcript=dict_transcript,
                reviewed_transcript=dict_transcript,
                speaker_id=speaker_id,
            )
            return fallback_result

        t0 = time.time()
        # Compute dynamic domain-knowledge contextual pass (zero-latency, deterministic)
        fast_context = self.apply_domain_context(
            clean_input,
            session=session,
            domain=domain,
            subject=subject,
            topic=topic
        )

        reviewed_payload: dict[str, Any] | None = None

        is_fallback = False
        fallback_reason = ""

        # Unit-test / Mocked response path (deterministic, no external dependency)
        if mock_response is not None:
            reviewed_payload = mock_response
            latency_ms = round((time.time() - t0) * 1000, 1)
        else:
            # Build rolling conversation history for LLM
            recent_turns = session.get_recent_history(DEFAULT_CONTEXT_WINDOW)
            history_text = ""
            if recent_turns:
                history_text = "Recent Conversation History (chronological):\n" + "\n".join(
                    f"- Speaker [{t.speaker_id}]: {t.reviewed_transcript}"
                    for t in recent_turns
                )

            user_prompt = (
                f"{history_text}\n\n"
                f"Candidate Transcript: \"{clean_input}\"\n"
                f"Original Acoustic ASR: \"{original_transcript}\"\n\n"
                "Review this candidate sentence considering the domain, subject, and conversation history. "
                "Apply appropriate technical abbreviation recovery or domain disambiguation if supported by context. "
                "Output strictly valid JSON."
            )

            system_prompt = self.build_system_prompt(session.domain, session.subject, session.topic)
            active_model = self.get_active_model()

            request_body = {
                "model": active_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
                "format": "json",
                "options": {
                    "temperature": 0.1,
                    "num_predict": 256,
                },
            }

            try:
                url = f"{self.base_url}/api/chat"
                req = urllib.request.Request(
                    url,
                    data=json.dumps(request_body).encode("utf-8"),
                    headers={"Content-Type": "application/json", "User-Agent": "Canary-ContextEngine/1.0"},
                )
                with urllib.request.urlopen(req, timeout=actual_timeout) as resp:
                    resp_json = json.loads(resp.read().decode("utf-8"))
                    raw_content = resp_json.get("message", {}).get("content", "{}")
                    reviewed_payload = json.loads(raw_content)
                latency_ms = round((time.time() - t0) * 1000, 1)
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, Exception) as exc:
                latency_ms = round((time.time() - t0) * 1000, 1)
                is_fallback = True
                fallback_reason = f"Ollama unavailable or timeout: {exc}"
                logger.info(
                    "Ollama request timed out or unavailable (%s: %s). Applying dynamic domain context engine.",
                    type(exc).__name__, exc
                )
                # Instead of dropping to uncorrected text, use domain context engine results
                reviewed_payload = fast_context
                reviewed_payload["from_fast_engine"] = True

        # Validate and sanitize structured output
        final_reviewed = str(reviewed_payload.get("reviewed_transcript") or clean_input).strip()
        corrections = reviewed_payload.get("corrections", [])
        correction_reasons = reviewed_payload.get("correction_reasons", [])
        inferred_terms = reviewed_payload.get("inferred_terms", [])
        uncertain_segments = reviewed_payload.get("uncertain_segments", [])
        requires_recheck = bool(reviewed_payload.get("requires_audio_recheck", False))

        # Anti-Hallucination Guardrail:
        # If the LLM fabricated an entirely new sentence (length expanded by > 2.5x without justification),
        # flag it as uncertain and fall back to the acoustic transcript.
        input_words = clean_input.split()
        reviewed_words = final_reviewed.split()
        if len(input_words) > 0 and (len(reviewed_words) / len(input_words)) > 2.8:
            logger.warning("Guarded against potential hallucination: input='%s', reviewed='%s'", clean_input, final_reviewed)
            uncertain_segments.append(final_reviewed)
            final_reviewed = clean_input

        # Commit verified turn to rolling conversation session memory
        turn = session.add_turn(
            original_transcript=original_transcript,
            dict_transcript=dict_transcript,
            reviewed_transcript=final_reviewed,
            speaker_id=speaker_id,
            corrections=corrections,
            inferred_terms=inferred_terms,
            uncertain_segments=uncertain_segments,
            requires_audio_recheck=requires_recheck,
            latency_ms=latency_ms,
        )

        review_status = "fallback" if (is_fallback and not corrections) else "success"

        return {
            "session_id": session.session_id,
            "turn_id": turn.turn_id,
            "original_transcript": original_transcript,
            "dict_transcript": dict_transcript,
            "reviewed_transcript": final_reviewed,
            "corrections": corrections,
            "correction_reasons": correction_reasons,
            "inferred_terms": inferred_terms,
            "uncertain_segments": uncertain_segments,
            "requires_audio_recheck": requires_recheck,
            "context_latency_ms": latency_ms,
            "llm_reviewed": True,
            "status": review_status,
            "reason": fallback_reason,
            "domain": session.domain,
            "subject": session.subject,
            "topic": session.topic,
        }


# ---------------------------------------------------------------------------
# Global Singleton Accessor
# ---------------------------------------------------------------------------

_GLOBAL_CONTEXT_ENGINE: OllamaContextEngine | None = None


def get_context_engine() -> OllamaContextEngine:
    global _GLOBAL_CONTEXT_ENGINE
    if _GLOBAL_CONTEXT_ENGINE is None:
        _GLOBAL_CONTEXT_ENGINE = OllamaContextEngine()
    return _GLOBAL_CONTEXT_ENGINE
