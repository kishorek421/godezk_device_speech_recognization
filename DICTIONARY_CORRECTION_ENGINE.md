# Standalone Dictionary-Based Correction Engine

## 1. Overview and Scope

The Dictionary Correction Engine is a standalone, deterministic post-processing module for domain-specific transcript error correction in the Walker Audio Speech AI pipeline.

### What the Engine Does:
- **Domain-Specific Normalization**: Replaces known, verified ASR recognition errors and phonetic/spelling misrecognitions with canonical terminology across domains (e.g., `education`, `hospital`, `industry`).
- **Word-Boundary Awareness**: Employs strict word-boundary matching (`\b`) to prevent corrupting larger unrelated words that happen to contain matching substrings (e.g., `curiculum` vs `recuriculum`).
- **Case Preservation**: Automatically preserves letter casing:
  - `CURICULUM` $\rightarrow$ `CURRICULUM`
  - `Curiculum` $\rightarrow$ `Curriculum`
  - `curiculum` $\rightarrow$ `curriculum`
- **Punctuation Preservation**: Preserves surrounding punctuation (commas, exclamation marks, parentheses, periods, quotes).
- **Ambiguity Guard**: When an alias maps to multiple canonical terms, or when fuzzy candidate scores tie, the engine **rejects** automatic replacement (`applied=False`, `reason="ambiguous_match"`) to prevent silent semantic distortion.
- **Explainability and Auditing**: Every evaluated token or phrase generates a detailed structured record:
  - `original`: original token or phrase
  - `replacement`: suggested replacement
  - `applied`: boolean flag
  - `reason`: explanatory decision reason (`alias_match`, `fuzzy_match`, `ambiguous_match`, `already_canonical`, etc.)
  - `similarity_score`: string edit similarity metric (Levenshtein-based), clearly documented as not a calibrated probability.

### What the Engine Does NOT Do:
- **No Model Training or Fine-Tuning**: Does not retrain or alter acoustic weights, language models, or NeMo Canary-Qwen 2.5B checkpoints.
- **No Vector DB / LLM Overhead**: Operates entirely in memory with $O(N)$ string and regex operations without requiring LLMs or embeddings.
- **No Unsafe Guessing**: Does not modify words solely because they look somewhat similar to domain terms unless explicitly configured or conservatively matched above strict thresholds.
- **No Cross-Domain Pollution**: Unsupported or unknown domains do not fall back to arbitrary dictionaries; they report `unknown_domain` and leave transcripts intact.

---

## 2. Why Dictionary Correction is NOT Model Training or Fine-Tuning

| Feature | Dictionary Correction | Model Fine-Tuning (LoRA / SALM) |
| :--- | :--- | :--- |
| **Stage** | Post-processing on output text | Acoustic/Language model training |
| **GPU / Compute** | Negligible CPU (<1 ms per sentence) | High GPU memory & hours/days of compute |
| **Dataset Requirement**| Curated list of verified error pairs | Thousands of paired audio-text clips |
| **Risk of Catastrophic Forgetting** | Zero risk | Significant risk for general domain |
| **Determinism** | 100% deterministic & reversible | Probabilistic generation |
| **Auditability** | Full trace of every token change | Black-box weights |

Dictionary correction does not change what the speech model "hears" or predicts; it acts as a deterministic safety filter on the resulting string.

---

## 3. Configuration & Adding Verified Errors

### Dictionary Format
The dictionary maps `domain` $\rightarrow$ `canonical_term` $\rightarrow$ list of observed error aliases:

```python
DOMAIN_DICTIONARY = {
    "education": {
        "curriculum": ["curiculum", "curriculam"],
        "derivation": ["derivasion"],
        "photosynthesis": ["photo synthesis"],
    },
    "hospital": {
        "diagnosis": ["diagonsis", "diagnosys"],
        "prescription": ["perscription", "prescripcion"],
        "radiology": ["radiologi", "radio logy"],
        "hypertension": ["hyper tension", "hyper tenshon"],
    },
    "industry": {
        "manufacturing": ["manufaturing", "manufactoring"],
        "maintenance": ["maintainance", "maintenence"],
        "inspection": ["inspeksion", "in spection"],
        "calibration": ["calibaration", "calibrasion"],
        "machinery": ["machinary"],
    },
}
```

### Procedure to Add a Verified ASR Error
1. Observe an actual transcription mistake from the Canary-Qwen model (e.g., ASR consistently transcribes "photosynthesis" as "photo synthesis").
2. Identify the target domain (`education`, `hospital`, or `industry`).
3. Add the exact observed misspelling/token into the alias list for the canonical word.
4. Run the unit tests to verify no regressions or ambiguous conflicts are introduced.

---

## 4. Usage & API

```python
from dictionary_engine import correct_text

# Example 1: Known error
corrected, records = correct_text(
    text="The curiculum is updated.",
    domain="education"
)
# Result: "The curriculum is updated."

# Example 2: Medical domain
corrected, records = correct_text(
    text="The patient was diagnosed with hyper tension.",
    domain="hospital"
)
# Result: "The patient was diagnosed with hypertension."

# Example 3: Ambiguous or unknown domain
corrected, records = correct_text(
    text="The curiculum is updated.",
    domain="unknown_field"
)
# Result: "The curiculum is updated." (records note unknown_domain, applied=False)
```

---

## 5. Verification Commands

### A. Run Unit Tests (Tests A through J and safety checks)
```bash
.venv\Scripts\python.exe -m pytest audio_api/tests/ -o pythonpath=audio_api -v
```

### B. Run Standalone Synthetic Evaluation Script (Exports CSV)
```bash
.venv\Scripts\python.exe test_dictionary_comparison.py
```
Output:
- Detailed console table showing Original, Corrected, and Change status
- Exported CSV at `dictionary_comparison_results.csv`

### C. Run GPU API Integration Test
```bash
# Health check and audio test against http://124.123.89.162:8001
.venv\Scripts\python.exe test_gpu_api_integration.py --audio <path_to_wav> --domain education
```

---

## 6. Interpreting Results
- **`applied=True`**: The correction satisfied all safety rules and was safely substituted.
- **`applied=False`, `reason="ambiguous_match"`**: Multiple dictionary entries matched equally. Correction was safely skipped.
- **`applied=False`, `reason="unknown_domain"`**: The requested domain has no configured dictionary. Original text was preserved without modification.
- **WER Reporting**: Word Error Rate improvements measured on synthetic sentences must always be labeled as synthetic benchmarks, not real-world ASR accuracy gains.
