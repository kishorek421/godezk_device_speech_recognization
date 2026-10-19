import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np
import soundfile as sf


class CanaryTranscriber:
    SALM_DOMAIN_PROMPTS = {
        "medical": "Transcribe the following clinical and medical speech verbatim without omitting or changing words: ",
        "education": "Transcribe the following academic and scientific speech verbatim without omitting or changing words: ",
        "industry": "Transcribe the following industrial and operational speech verbatim without omitting or changing words: ",
        "general": "Transcribe the following speech verbatim without omitting or changing words: ",
    }


    def __init__(self, model_name: str, device: str = "cuda", dtype: str = "bfloat16"):
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self._model: Any = None
        self.runtime_info: dict[str, Any] = {
            "model_name": model_name,
            "device": device,
            "dtype": dtype,
            "is_loaded": False,
        }

    def load(self) -> None:
        import importlib

        try:
            torch = importlib.import_module("torch")
            slm_mod = importlib.import_module("nemo.collections.speechlm2.models")
            SALM = getattr(slm_mod, "SALM")
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError(
                f"CanaryTranscriber requires PyTorch and NeMo SpeechLM2 (SALM): {exc}"
            ) from exc

        torch_dtype = getattr(torch, self.dtype, torch.float32)
        device_target = torch.device(self.device if torch.cuda.is_available() or self.device == "cpu" else "cpu")
        self._model = (
            SALM.from_pretrained(self.model_name)
            .to(dtype=torch_dtype)
            .eval()
            .to(device_target)
        )
        self.runtime_info["is_loaded"] = True
        self.runtime_info["actual_device"] = str(device_target)
        self.runtime_info["actual_dtype"] = str(torch_dtype)


    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int,
        mode: str = "salm",
        domain: str = "general",
    ) -> str:
        """Unified entry point returning the final transcript string for backward compatibility."""
        result = self.transcribe_advanced(audio, sample_rate, mode=mode, domain=domain)
        return result["transcript"]

    def transcribe_advanced(
        self,
        audio: np.ndarray,
        sample_rate: int,
        mode: str = "salm",
        domain: str = "general",
    ) -> dict[str, Any]:
        """Runs either Mode A (ASR -> LLM) or Mode B (SALM Speech-Language) with domain context."""
        import importlib
        torch = importlib.import_module("torch")

        if self._model is None:
            self.load()
        if sample_rate != 16_000:
            raise ValueError("Canary-Qwen requires 16 kHz audio")

        mode = (mode or "salm").lower()
        domain = (domain or "general").lower()
        if domain not in self.SALM_DOMAIN_PROMPTS:
            domain = "general"

        with NamedTemporaryFile(suffix=".wav") as temporary:
            sf.write(temporary.name, audio, sample_rate)

            with torch.inference_mode():
                if mode == "asr":
                    # Mode A: Step 1: Plain ASR (audio -> raw acoustic text)
                    base_prompt = f"Transcribe the following: {self._model.audio_locator_tag}"
                    asr_ids = self._model.generate(
                        prompts=[[{"role": "user", "content": base_prompt, "audio": [temporary.name]}]],
                        max_new_tokens=128,
                    )
                    raw_asr_text = self._model.tokenizer.ids_to_text(asr_ids[0].cpu()).strip()

                    # Mode A: Step 2: Pass raw ASR text to Qwen text LLM for domain understanding
                    llm_text_prompt = (
                        f"Domain Context: {domain.capitalize()}.\n"
                        f"Raw speech transcription: \"{raw_asr_text}\".\n"
                        f"Understand and output the exact intended terminology or sentence. "
                        f"Output only the final transcript."
                    )
                    llm_ids = self._model.generate(
                        prompts=[[{"role": "user", "content": llm_text_prompt}]],
                        max_new_tokens=64,
                    )
                    final_text = self._model.tokenizer.ids_to_text(llm_ids[0].cpu()).strip()
                    if not final_text:
                        final_text = raw_asr_text

                    return {
                        "transcript": final_text,
                        "raw_asr_text": raw_asr_text,
                        "processing_mode": "asr",
                        "domain": domain,
                    }

                else:
                    # Mode B: Direct SALM (Audio -> Speech representation into Qwen with domain prompt)
                    salm_prompt_prefix = self.SALM_DOMAIN_PROMPTS[domain]
                    full_prompt = f"{salm_prompt_prefix}{self._model.audio_locator_tag}"
                    answer_ids = self._model.generate(
                        prompts=[[{"role": "user", "content": full_prompt, "audio": [temporary.name]}]],
                        max_new_tokens=128,
                    )
                    final_text = self._model.tokenizer.ids_to_text(answer_ids[0].cpu()).strip()

                    return {
                        "transcript": final_text,
                        "raw_asr_text": None,
                        "processing_mode": "salm",
                        "domain": domain,
                    }


class TitaNetVerifier:
    def __init__(self, model_name: str = "titanet_large", device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self._model: Any = None
        self._reference: np.ndarray | None = None
        self.registry: dict[str, np.ndarray] = {}  # speaker_id -> normalized embedding
        self.runtime_info: dict[str, Any] = {
            "model_name": model_name,
            "device": device,
            "is_loaded": False,
        }

    def load(self) -> None:
        import importlib

        try:
            torch = importlib.import_module("torch")
            asr_mod = importlib.import_module("nemo.collections.asr.models")
            EncDecSpeakerLabelModel = getattr(asr_mod, "EncDecSpeakerLabelModel")
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError(f"TitaNetVerifier requires PyTorch and NeMo ASR: {exc}") from exc

        device_target = self.device if torch.cuda.is_available() or self.device == "cpu" else "cpu"
        self._model = EncDecSpeakerLabelModel.from_pretrained(self.model_name).to(device_target)
        self._model.eval()
        self.runtime_info["is_loaded"] = True
        self.runtime_info["actual_device"] = str(device_target)

    def identify(
        self,
        audio: np.ndarray,
        sample_rate: int,
        threshold: float = 0.65,
    ) -> tuple[str, float]:
        """Identifies speaker against the enrolled registry. Returns (speaker_id, confidence).

        If no enrolled speaker matches above threshold, returns ('unknown', max_score).
        """
        if not self.registry:
            return "unknown", 0.0

        with NamedTemporaryFile(suffix=".wav") as temp_file:
            sf.write(temp_file.name, audio, sample_rate)
            embedding = self._embed(temp_file.name)

        norm = np.linalg.norm(embedding)
        if norm > 1e-8:
            embedding = embedding / norm

        best_speaker = "unknown"
        best_score = -1.0

        for speaker_id, ref_emb in self.registry.items():
            ref_norm = np.linalg.norm(ref_emb)
            if ref_norm > 1e-8:
                ref_normed = ref_emb / ref_norm
            else:
                ref_normed = ref_emb
            score = float(np.dot(embedding, ref_normed))
            if score > best_score:
                best_score = score
                if score >= threshold:
                    best_speaker = speaker_id

        confidence = max(0.0, min(1.0, (best_score + 1.0) / 2.0))
        return best_speaker, round(confidence, 4)

    def enroll_speaker(self, speaker_id: str, audio: np.ndarray | str | Path, sample_rate: int = 16000) -> None:
        """Enrolls a speaker into the active registry."""
        if isinstance(audio, np.ndarray):
            with NamedTemporaryFile(suffix=".wav") as temp_file:
                sf.write(temp_file.name, audio, sample_rate)
                emb = self._embed(temp_file.name)
        else:
            emb = self._embed(audio)

        norm = np.linalg.norm(emb)
        if norm > 1e-8:
            emb = emb / norm
        self.registry[speaker_id] = emb

    def save_registry(self, file_path: str | Path) -> None:
        """Saves registry embeddings as a JSON-serializable mapping."""
        serialized = {k: v.tolist() for k, v in self.registry.items()}
        Path(file_path).write_text(json.dumps(serialized), encoding="utf-8")

    def load_registry(self, file_path: str | Path) -> None:
        """Loads registry embeddings from file."""
        path = Path(file_path)
        if not path.exists():
            return
        data = json.loads(path.read_text(encoding="utf-8"))
        self.registry = {k: np.array(v, dtype=np.float32) for k, v in data.items()}

    def enroll(self, audio_path: str | Path) -> None:
        self._reference = self._embed(audio_path)

    def score_audio(self, audio: np.ndarray, sample_rate: int, temporary_path: Path) -> float:
        if self._reference is None:
            raise RuntimeError("enroll a target speaker before requesting speaker verification")
        sf.write(temporary_path, audio, sample_rate)
        embedding = self._embed(temporary_path)
        denominator = np.linalg.norm(self._reference) * np.linalg.norm(embedding)
        return float(np.dot(self._reference, embedding) / max(denominator, 1e-8))

    def _embed(self, audio_path: str | Path) -> np.ndarray:
        if self._model is None:
            self.load()
        embedding = self._model.get_embedding(str(audio_path)).detach().cpu().numpy().squeeze()
        return np.asarray(embedding, dtype=np.float32)

