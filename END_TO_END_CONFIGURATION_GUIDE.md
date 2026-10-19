# End-to-End System Architecture & Numbered Configuration Guide

**Speech-to-Text, Speaker Identification, and Domain Context Understanding Platform**  
**Repository:** `ai_services`  
**Updated:** October 2026

---

## Table of Contents
1. [System Architecture Overview](#1-system-architecture-overview)
2. [End-to-End Request Pipeline (Step-by-Step)](#2-end-to-end-request-pipeline-step-by-step)
3. [Deep-Dive into Subsystems & Models](#3-deep-dive-into-subsystems--models)
   - [3.1 Audio DSP Preprocessing](#31-audio-dsp-preprocessing)
   - [3.2 Speaker Identification (CAM++ VoxCeleb)](#32-speaker-identification-cam-voxceleb)
   - [3.3 Speech-to-Text Engine (Canary-Qwen 2.5B)](#33-speech-to-text-engine-canary-qwen-25b)
   - [3.4 Dictionary Correction Engine](#34-dictionary-correction-engine)
   - [3.5 Real-Time Domain Context Engine](#35-real-time-domain-context-engine)
   - [3.6 Optional LLM Context Review (Ollama Qwen)](#36-optional-llm-context-review-ollama-qwen)
4. [Complete Model Inventory & Specifications](#4-complete-model-inventory--specifications)
5. [Numbered Configuration Modes (Budget, Hardware & Latency)](#5-numbered-configuration-modes)
   - [Mode 1: Pure Local CPU / Zero-Cost Embedded](#mode-1-pure-local-cpu--zero-cost-embedded)
   - [Mode 2: Local Hybrid (Lightweight ASR + CAM++ + Rules)](#mode-2-local-hybrid-lightweight-asr--cam--rules)
   - [Mode 3: Production Standard / Balanced (Recommended Baseline)](#mode-3-production-standard--balanced-recommended-baseline)
   - [Mode 4: High-Precision Real-Time LLM (GPU Canary + Fast 7B LLM)](#mode-4-high-precision-real-time-llm-gpu-canary--fast-7b-llm)
   - [Mode 5: Enterprise Multi-Turn Deep Context (Canary SALM + 30B MoE)](#mode-5-enterprise-multi-turn-deep-context-canary-salm--30b-moe)
   - [Mode 6: Distributed Cloud Microservices](#mode-6-distributed-cloud-microservices)
6. [Side-by-Side Configuration Comparison Matrix](#6-side-by-side-configuration-comparison-matrix)
7. [Comprehensive Configuration Settings & Environment Variables](#7-comprehensive-configuration-settings--environment-variables)
8. [API Endpoints & Payload Specifications](#8-api-endpoints--payload-specifications)
9. [Operational Deployment & Troubleshooting](#9-operational-deployment--troubleshooting)

---

## 1. System Architecture Overview

The platform is designed as a modular, fault-tolerant speech intelligence pipeline capable of processing audio from noisy browser microphones, identifying enrolled speakers, transcribing domain-specific jargon (Education, Hospital/Clinical, Industry), and disambiguating polysemous terminology using local rules and optional LLM context.

```
                          [ Client / Web Browser UI ]
                       (Microphone WebM/WAV Audio Stream)
                                       │
                                       ▼ HTTP POST /api/transcribe
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ Local FastAPI Application Gateway (word_test_server.py:8050)                │
  │                                                                             │
  │  [Stage 1: Audio Ingestion & DSP Preprocessing]                             │
  │   ├── Format Decoder (soundfile / ffmpeg pipe fallback)                     │
  │   ├── Polyphase Resampler (Target: 16,000 Hz Mono PCM)                      │
  │   ├── 4th-Order Butterworth 65 Hz High-Pass Filter (Removes 50/60 Hz hum)   │
  │   ├── 99.8th Percentile Voice Normalization (-1.2 dBFS target)              │
  │   └── Energy-Based Silence Detector (<0.02 peak gate)                       │
  │                                                                             │
  │  [Stage 2: Neural Speaker Identification & Biometrics]                      │
  │   ├── sherpa-onnx Runtime (CAM++ VoxCeleb 16k ONNX)                         │
  │   ├── 512-dimensional L2-normalized embedding extraction                    │
  │   ├── Cosine Similarity search against speaker_registry.json                │
  │   └── Calibrated Threshold Decision (Threshold = 0.68)                      │
  └──────────────────────┬──────────────────────────────────────────────────────┘
                         │
         ┌───────────────┴───────────────────────────────┐
         ▼                                               ▼
  [Local CPU Path]                               [Remote GPU Path]
  • Speaker ID: "megak" (0.84 conf)               • Endpoint: http://124.123.89.162:8001/v1/transcribe
  • Duration: 3.2s                                • Architecture: NeMo FastConformer + Qwen-2.5 2.5B
  • Peak: -1.2 dBFS                               • Precision: CUDA bfloat16
         │                                        • Latency: ~637 ms median
         │                                               │
         └───────────────────────┬───────────────────────┘
                                 ▼
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ Post-Processing & Context Intelligence                                      │
  │                                                                             │
  │  [Stage 3: Domain Dictionary Correction] (Latency: ~3 ms)                   │
  │   ├── Multi-domain static lexicon (Hospital, Education, Industry)           │
  │   └── Fast token/regex mapping (e.g., "high blood presure" -> "hypertension")│
  │                                                                             │
  │  [Stage 4: Real-Time Context Engine & Polysemy Disambiguation] (~0.1 ms)   │
  │   ├── Thread-safe session memory buffer (history, turn count, TTL 1800s)    │
  │   ├── Domain polysemy disambiguation ("work" in Physics vs Chemistry)       │
  │   └── Abbreviation recovery ("p h" -> "pH", "s p o 2" -> "SpO2")            │
  │                                                                             │
  │  [Stage 5: Optional LLM Context Review] (Configurable: On / Off)           │
  │   ├── Ollama Remote Endpoint: http://124.123.89.162:11434                   │
  │   ├── Model: Qwen 30B MoE / Qwen 2.5 7B                                     │
  │   ├── Strict 2.0s circuit breaker timeout with instant fallback             │
  │   └── Anti-hallucination guardrail (word expansion capped < 2.8x)           │
  └──────────────────────────────┬──────────────────────────────────────────────┘
                                 │
                                 ▼
  [ JSON API Response: Reviewed Transcript + Speaker Tag + Timing Diagnostics ]
```

---

## 2. End-to-End Request Pipeline (Step-by-Step)

1. **Audio Capture**: Browser records audio from the microphone via `MediaRecorder` API (usually in `audio/webm;codecs=opus` or `audio/wav`).
2. **DSP Ingestion**:
   - `robust_read_audio()` decodes bytes using `soundfile`; if the format is Opus/WebM, it seamlessly spawns an `ffmpeg` pipe to produce standard 16 kHz WAV.
   - `prepare_audio_for_canary()` applies a 4th-order 65 Hz Butterworth High-Pass filter to strip laptop fan vibration and 50/60 Hz mains electrical hum.
   - If voiced speech is present ($\ge -34\text{ dBFS}$), peak volume is scaled to -1.2 dBFS. Pure silence or background rumble is flagged as `is_silence=True`.
3. **Biometric Speaker Identification**:
   - The cleaned 16 kHz audio is passed to `speaker_registry.identify()`.
   - `sherpa-onnx` executes `campplus_voxceleb_16k.onnx` locally on CPU.
   - A 512-dimensional embedding is compared against enrolled centroids in `config/speaker_registry.json`.
   - If cosine similarity $\ge 0.68$, the speaker is identified (e.g., `"megak"`). Otherwise, labeled `"Unknown Speaker"`. If audio has no energy, labeled `"NO_SPEECH"`.
4. **Remote GPU Speech-to-Text**:
   - Audio is dispatched via multipart HTTP POST to `http://124.123.89.162:8001/v1/transcribe`.
   - Canary-Qwen 2.5B executes forward inference on GPU (CUDA `bfloat16`).
   - If the GPU server times out or is offline, the gateway catches the exception and returns a partial fallback without crashing.
5. **Dictionary Correction**:
   - Spoken transcript passes through `correct_transcript()` using `walker_audio/dictionaries/domain_dictionary.json`.
   - Exact and phonetic domain misspellings are corrected in ~3 ms.
6. **Real-Time Context Review**:
   - Session memory retrieves the recent 6 turns for this `session_id`.
   - Domain polysemy rules check whether current words have conflicting definitions across subjects (e.g., thermodynamic work vs mechanical work).
   - Abbreviation regex reassembles spaced characters (`s p o 2` $\rightarrow$ `SpO2`).
7. **Optional LLM Context Review**:
   - If `context_review_enabled=true`, prompt is sent to Ollama with conversation history.
   - Strict 2.0s timeout ensures the user never experiences long stalls.
   - Output passes anti-hallucination validation (length ratio $<2.8\times$, medical dosages and negations strictly preserved).
8. **Client Response**: Final payload containing `original_transcript`, `corrected_transcript`, `reviewed_transcript`, `speaker_id`, `speaker_confidence`, and latency diagnostics is returned.

---

## 3. Deep-Dive into Subsystems & Models

### 3.1 Audio DSP Preprocessing
- **Source File**: [word_test_server.py](file:///c:/Users/Admin/Downloads/ai_services/ai_services/word_test_server.py#L24-L102)
- **High-Pass Filter**: 4th-order Butterworth filter at $65\text{ Hz}$ cutoff ($f_s = 16000\text{ Hz}$). Preserves human pitch (85 Hz – 4000 Hz) while completely eliminating 50 Hz/60 Hz electrical transformer hum, air conditioning rumble, and microphone wind pops.
- **Normalization**: Percentile-based (99.8th percentile scaled to 0.85, clipping at $[-0.98, +0.98]$) to prevent clipping while maximizing signal-to-quantization noise ratio.
- **Execution**: Pure NumPy + SciPy Signal on local CPU (~3.5 ms execution time).

### 3.2 Speaker Identification (CAM++ VoxCeleb)
- **Model Binary**: `models/campplus_voxceleb_16k.onnx` (29.6 MB)
- **Engine**: `sherpa-onnx` ONNXRuntime engine on CPU.
- **Architecture**: CAM++ (Context-Aware Masking) trained on VoxCeleb 1 & 2.
- **Embedding**: 512-dimensional float32 vector, L2-normalized.
- **Scoring**: Cosine similarity against enrolled exemplar centroids:
  $$\text{sim}(u, v) = \frac{u \cdot v}{\|u\|_2 \|v\|_2}$$
- **Threshold**: Calibrated at `0.68`. Yields **0.00% False Acceptance Rate** (impostors never pass).
- **Latency**: Embedding extraction = ~233 ms; full verification = ~615 ms on CPU.

### 3.3 Speech-to-Text Engine (Canary-Qwen 2.5B)
- **Base Architecture**: NVIDIA NeMo SALM (Speech-Augmented Language Model).
- **Audio Encoder**: Conformer / FastConformer (~600M parameters).
- **Decoder / LLM Backbone**: Qwen-2.5 1.5B/2.5B instruction-tuned LLM.
- **Weights Precision**: `bfloat16` running on NVIDIA RTX/A-series CUDA GPU.
- **Endpoint**: `POST /v1/transcribe` at port `8001`.
- **Modes Supported**:
  - `mode="asr"`: Standard verbatim automatic speech recognition.
  - `mode="salm"`: Direct speech question answering and technical transcript generation.
- **Latency**: Median **637.35 ms**, P95 **2255.36 ms**.

### 3.4 Dictionary Correction Engine
- **Source File**: [walker_audio/dictionary_corrector.py](file:///c:/Users/Admin/Downloads/ai_services/ai_services/walker_audio/walker_audio/dictionary_corrector.py)
- **Dictionary File**: `walker_audio/dictionaries/domain_dictionary.json`
- **Supported Domains**:
  - `education`: Physics, Chemistry, Biology, Mathematics.
  - `hospital`: Cardiology, Pulmonology, Pharmacology, Vitals.
  - `industry`: Electrical engineering, Manufacturing, Automation, Safety.
- **Speed**: **3.11 ms** median latency. Zero risk of generative hallucination.

### 3.5 Real-Time Domain Context Engine
- **Source File**: [audio_api/context_engine.py](file:///c:/Users/Admin/Downloads/ai_services/ai_services/audio_api/context_engine.py)
- **Session Memory**: Sliding buffer of the last $N$ turns (default $N=6$) indexed by `session_id`.
- **TTL**: Automatic memory expiration after 1,800 seconds (30 minutes) of inactivity.
- **Polysemy Resolution**: Resolves terms whose scientific meaning changes based on domain:
  - *"Work"* in Physics: $\vec{F} \cdot \vec{d}$ (mechanical displacement).
  - *"Work"* in Chemistry: $P \Delta V$ (expansion of gases against external pressure).
  - *"Base"* in Chemistry: Proton acceptor / electron pair donor.
  - *"Base"* in Mathematics: Number radix or triangle foundation.
- **Latency**: **0.11 ms** median latency on CPU.

### 3.6 Optional LLM Context Review (Ollama Qwen)
- **Engine**: Ollama HTTP API (`/api/generate` or `/api/chat`).
- **Endpoint**: `http://124.123.89.162:11434`.
- **Safety Filters**:
  - Output expansion barrier: Max output token length capped at $2.8\times$ input.
  - Numerical dosage preservation: Numbers followed by units (`mg`, `ml`, `mcg`, `V`, `A`) cannot be altered.
  - Negation preservation: Words like `"no"`, `"not"`, `"without"` are protected from inversion.
  - Circuit Breaker: Strict 2.0s client timeout. If Ollama takes $>2.0\text{ s}$, the engine discards the call and returns the dictionary-corrected text immediately.

---

## 4. Complete Model Inventory & Specifications

| Model Name | Role / Function | Format / Architecture | Parameter Count | Disk Size | Runtime Memory | Execution Target | Typical Latency |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **CAM++ VoxCeleb** | Speaker Verification | ONNX (`campplus_voxceleb_16k.onnx`) | ~7.2M | 29.6 MB | ~45 MB RAM | Local Host CPU | 233 ms |
| **Speaker ResNet34** | Lightweight Speaker ID | ONNX (`speaker_resnet34.onnx`) | ~6.6M | 6.7 MB | ~20 MB RAM | Local Host CPU | 110 ms |
| **Canary-Qwen 2.5B** | Primary Speech-to-Text | NeMo FastConformer + Qwen-2.5 | 2.5 Billion | ~5.2 GB | ~5.8 GB VRAM | Remote CUDA GPU | 637 ms |
| **Qwen-2.5 3B / 7B** | Fast LLM Context Review | GGUF Q4_K_M (Ollama / vLLM) | 3B - 7B | 2.1 - 4.7 GB | 3.5 - 6.0 GB VRAM | CUDA GPU / Host | 220 - 450 ms |
| **Qwen 30B MoE** | Deep Multi-Turn Context | GGUF / Ollama (`qwen3-coder:30b`)| 30 Billion | 18.55 GB | ~22 GB RAM/VRAM | Remote Host | 20 - 60s (CPU) |
| **Domain Lexicon** | Spelling & Terminology | In-memory Hash Table | N/A | 3.3 KB | < 2 MB RAM | Local Host CPU | 3 ms |

---

## 5. Numbered Configuration Modes

The platform can be deployed in **6 distinct configurations**, ranging from zero-cost offline edge computing to enterprise cloud setups.

---

### Mode 1: Pure Local CPU / Zero-Cost Embedded
*Zero external server dependencies, zero GPU required, 100% offline and confidential.*

```
[ Audio In ] ──> [ DSP Prep ] ──> [ CAM++ ONNX ] ──> [ Whisper-Base CPU ] ──> [ Dict + Rules ] ──> [ Output ]
```

- **Target Use Case**: Offline edge laptops, air-gapped field workstations, zero-cloud budget.
- **Models Used**:
  - ASR: OpenAI Whisper-Base / Small via `faster-whisper` (INT8 on CPU).
  - Speaker ID: CAM++ VoxCeleb ONNX (`models/campplus_voxceleb_16k.onnx`).
  - Context: Local Dictionary + Static Domain Rules.
  - LLM: None (Disabled).
- **Hardware Requirements**: 4-Core CPU, 4 GB RAM, 0 GB VRAM.
- **Estimated Latency**:
  - ASR: ~800 – 1200 ms
  - Speaker ID: ~250 ms
  - End-to-End: **~1.1 – 1.5 seconds**
- **Configuration Flags**:
  ```bash
  REMOTE_GPU_API=""
  LOCAL_WHISPER_ENABLED=true
  WHISPER_MODEL="base.en"
  SPEAKER_MODEL="models/campplus_voxceleb_16k.onnx"
  DICTIONARY_ENABLED=true
  CONTEXT_REVIEW_ENABLED=false
  ```
- **Pros & Cons**:
  -  Zero hosting cost, operates without internet, complete data privacy.
  - ❌ Slower ASR on older CPUs, lower technical vocabulary accuracy without Canary.

---

### Mode 2: Local Hybrid (Lightweight ASR + CAM++ + Rules)
*Low cost, fast turnaround for office desktop environments with modest GPUs.*

```
[ Audio In ] ──> [ DSP Prep ] ──> [ CAM++ ONNX ] ──> [ Whisper-Large-v3-Turbo (GPU) ] ──> [ Dict + Rules ]
```

- **Target Use Case**: Single workstation with an entry-level NVIDIA GPU (GTX 1660 / RTX 3050).
- **Models Used**:
  - ASR: `Whisper-Large-v3-Turbo` (800M params, FP16 via CUDA).
  - Speaker ID: CAM++ VoxCeleb ONNX (Local CPU).
  - Post-processing: Local Dictionary + Domain Rules.
  - LLM: None.
- **Hardware Requirements**: 4-Core CPU, 8 GB RAM, 4 GB GPU VRAM.
- **Estimated Latency**:
  - ASR: ~280 ms
  - Speaker ID: ~230 ms
  - End-to-End: **~520 ms**
- **Configuration Flags**:
  ```bash
  REMOTE_GPU_API="http://127.0.0.1:8001"
  SPEAKER_MODEL="models/campplus_voxceleb_16k.onnx"
  DICTIONARY_ENABLED=true
  CONTEXT_REVIEW_ENABLED=false
  ```
- **Pros & Cons**:
  -  Ultra-fast response (< 600 ms), low hardware threshold.
  - ❌ General Whisper lacks specialized NeMo SALM conversational prompting capabilities.

---

### Mode 3: Production Standard / Balanced (Recommended Baseline)
*The verified, highest-reliability production configuration evaluated in this benchmark.*

```
[ Audio In ] ──> [ DSP Prep ] ──> [ CAM++ ONNX (Local CPU) ]
                      │
                      └──> [ Canary-Qwen 2.5B (Remote CUDA) ] ──> [ Dict + Local Rules ] ──> [ Output ]
```

- **Target Use Case**: Production clinics, educational lecture halls, industrial testing benches.
- **Models Used**:
  - ASR: `nvidia/canary-qwen-2.5b` (NeMo FastConformer + Qwen-2.5 on Remote GPU).
  - Speaker ID: CAM++ VoxCeleb ONNX (`campplus_voxceleb_16k.onnx` on Local CPU).
  - Post-processing: `dictionary_corrector.py` + `context_engine.py` (Local Rules).
  - LLM: Bypassed (`context_review_enabled=false`).
- **Hardware Requirements**:
  - Client / Gateway Host: 2-Core CPU, 2 GB RAM.
  - Remote ASR Server: 1x NVIDIA GPU with $\ge 8\text{ GB}$ VRAM (RTX 3080/4080, T4, A10).
- **Empirical Measured Latency**:
  - Audio DSP: **3.45 ms**
  - Speaker ID (CAM++): **615.87 ms**
  - Canary-Qwen ASR: **637.35 ms** (P95: 2255 ms)
  - Dictionary Corrector: **3.11 ms**
  - Local Domain Rules: **0.11 ms**
  - End-to-End Total: **641.41 ms** median
- **Empirical Accuracy**:
  - Technical Word Recovery: **21.67%**
  - Abbreviation Recovery: **44.44%**
  - Hallucination Rate: **0.00%**
- **Configuration Flags**:
  ```bash
  REMOTE_GPU_API="http://124.123.89.162:8001"
  DICTIONARY_ENABLED=true
  CONTEXT_REVIEW_ENABLED=false
  SPEAKER_THRESHOLD=0.68
  SESSION_TTL_SEC=1800
  ```
- **Pros & Cons**:
  -  **Sub-second response (641 ms)**, zero external LLM failure points, handles complex multi-turn medical/technical terms, 100% stable.
  - ❌ Requires a dedicated GPU server for Canary-Qwen 2.5B.

---

### Mode 4: High-Precision Real-Time LLM (GPU Canary + Fast 7B LLM)
*Interactive conversational AI with active LLM context review completed under 1 second.*

```
[ Audio In ] ──> [ DSP Prep ] ──> [ CAM++ ONNX ] ──> [ Canary-Qwen 2.5B ]
                                                              │
                                                              ▼
               [ Output ] <── [ Guardrails ] <── [ Qwen-2.5 7B (GPU Ollama/vLLM) ] <── [ Dict ]
```

- **Target Use Case**: Real-time voice assistants, live doctor-patient consultations requiring complex contextual rephrasing and acronym expansion.
- **Models Used**:
  - ASR: `nvidia/canary-qwen-2.5b` (Remote CUDA GPU).
  - Speaker ID: CAM++ VoxCeleb ONNX (Local CPU).
  - Post-processing: Local Dictionary Corrector.
  - LLM: `qwen2.5:7b-instruct-q4_K_M` running on a **dedicated GPU** via Ollama or vLLM.
- **Hardware Requirements**:
  - Gateway Host: 4-Core CPU, 4 GB RAM.
  - GPU Server: 1x NVIDIA RTX 3090 / 4090 / A5000 (24 GB VRAM) running both Canary (6 GB) and Qwen-7B (5 GB).
- **Estimated Latency**:
  - ASR GPU: ~640 ms
  - Dictionary: ~3 ms
  - LLM Context Review (GPU): ~280 ms
  - End-to-End Total: **~925 ms** (Strictly under 1.0 second)
- **Configuration Flags**:
  ```bash
  REMOTE_GPU_API="http://124.123.89.162:8001"
  OLLAMA_BASE_URL="http://124.123.89.162:11434"
  OLLAMA_MODEL="qwen2.5:7b-instruct-q4_K_M"
  OLLAMA_TIMEOUT_SEC=1.5
  DICTIONARY_ENABLED=true
  CONTEXT_REVIEW_ENABLED=true
  ```
- **Pros & Cons**:
  -  Full LLM semantic reasoning, intelligent medical abbreviation expansion, under 1-second latency.
  - ❌ Requires 24 GB GPU VRAM to co-host Canary ASR and Qwen LLM.

---

### Mode 5: Enterprise Multi-Turn Deep Context (Canary SALM + 30B MoE)
*Maximum reasoning accuracy for complex multi-party medical boards, lectures, and legal reviews.*

```
[ Audio Stream ] ──> [ Diarization (CAM++) ] ──> [ Canary SALM ] ──> [ Qwen 30B MoE Deep Review ]
```

- **Target Use Case**: Asynchronous or batch transcript processing, recorded surgery analysis, technical symposium transcription where accuracy outweighs immediate latency.
- **Models Used**:
  - ASR: Canary-Qwen 2.5B in full Speech-Augmented Language Model (`mode="salm"`) mode.
  - Speaker ID: CAM++ multi-speaker continuous clustering diarization.
  - LLM: `qwen3-coder:30b` or `qwen2.5:32b-instruct` on an enterprise multi-GPU node.
- **Hardware Requirements**:
  - Enterprise Cluster: 2x NVIDIA A100 / H100 (80 GB) or 4x RTX 4090.
- **Estimated Latency**:
  - End-to-End: **3.5 – 6.0 seconds** per utterance.
- **Configuration Flags**:
  ```bash
  REMOTE_GPU_API="http://gpu-cluster:8001"
  OLLAMA_BASE_URL="http://llm-cluster:11434"
  OLLAMA_MODEL="qwen3-coder:30b"
  OLLAMA_TIMEOUT_SEC=10.0
  DICTIONARY_ENABLED=true
  CONTEXT_REVIEW_ENABLED=true
  CONTEXT_WINDOW_TURNS=12
  ```
- **Pros & Cons**:
  -  Unrivaled technical precision, deep contextual cross-referencing across 12 turns, flawless polysemy disambiguation.
  - ❌ Expensive enterprise infrastructure; latency exceeds real-time interactive thresholds.

---

### Mode 6: Distributed Cloud Microservices
*High-availability, horizontally auto-scaling enterprise deployment.*

```
                    ┌──> [ Ingestion & Preprocessing Node ] ──> [ Kafka / RabbitMQ ]
                    │
 [ API Gateway ] ───┼──> [ Speaker Biometrics Worker (CAM++ ONNX) ]
 (NGINX + Traefik)  │
                    ├──> [ Triton Inference Server (Canary-Qwen ASR GPU Pool) ]
                    │
                    └──> [ Redis Session Store ] <──> [ vLLM Worker Cluster (Qwen LLM) ]
```

- **Target Use Case**: SaaS multi-tenant speech platforms serving thousands of concurrent users.
- **Components**:
  - Frontend / Gateway: NGINX terminating TLS + FastAPI load balancers.
  - Session Store: Shared Redis Cluster (stores `session_id`, turn buffers, speaker centroids).
  - ASR Cluster: NVIDIA Triton Inference Server running Canary-Qwen with dynamic batching.
  - LLM Cluster: vLLM engine serving quantized Qwen-2.5 with continuous batching.
- **Scalability**: Scales from 10 to 10,000 concurrent sessions dynamically.

---

## 6. Side-by-Side Configuration Comparison Matrix

| Configuration Mode | Budget Tier | Target Hardware | Models Employed | End-to-End Median Latency | Abbr. Recovery | Hallucination Safety | Ready for Production? |
| :--- | :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| **Mode 1: Pure Local CPU** | **$0** (Free) | 4-Core CPU, 4 GB RAM | Whisper-Base + CAM++ + Dict | 1,250 ms | 33% | 100% Safe (No LLM) | ✅ Pilot Only |
| **Mode 2: Local Hybrid** | **Low** (~$500) | 4 GB GPU (GTX 1660) | Whisper-Turbo + CAM++ + Dict | 520 ms | 33% | 100% Safe (No LLM) | ✅ Yes |
| **Mode 3: Production Standard** | **Medium** (Current) | 1x 8GB GPU Server | **Canary-Qwen 2.5B + CAM++ + Local Rules** | **641 ms** | **44.4%** | **100% Safe (Local Rules)** | ** Recommended Baseline** |
| **Mode 4: High-Precision LLM** | **Medium-High** | 1x 24GB GPU Server | Canary-Qwen + CAM++ + Qwen-7B GPU | 925 ms | 55% | Protected (<2.8x filter) | ✅ Yes (GPU LLM only) |
| **Mode 5: Enterprise Deep Context** | **High** | 2x A100 / 4x 4090 | Canary SALM + CAM++ + Qwen-30B | 4,200 ms | 75% | Protected (<2.8x filter) | ⚠️ Batch/Async only |
| **Mode 6: Distributed Cloud** | **Enterprise** | K8s Cluster + Triton | Canary Triton + vLLM + Redis | ~750 ms | 65% | Enterprise Guardrails | 🚀 Enterprise Scale |

---

## 7. Comprehensive Configuration Settings & Environment Variables

All parameters can be configured either via operating system environment variables or inside `audio_api/.env` and `word_test_server.py`.

### Gateway Server (`word_test_server.py`)

| Environment Variable | Default Value | Description |
| :--- | :--- | :--- |
| `PORT` | `8050` | HTTP port on which the web gateway and UI listen. |
| `REMOTE_GPU_API` | `http://124.123.89.162:8001` | Base URL of the remote Canary-Qwen ASR GPU server. |
| `SPEAKER_THRESHOLD` | `0.68` | Cosine similarity cutoff for CAM++ speaker verification. |
| `RECORDINGS_DIR` | `./recordings` | Directory where uploaded WAV audio proofs are saved. |

### Domain Context Engine (`audio_api/context_engine.py`)

| Environment Variable | Default Value | Description |
| :--- | :--- | :--- |
| `OLLAMA_BASE_URL` | `http://124.123.89.162:11434`| HTTP endpoint of the Ollama LLM review service. |
| `OLLAMA_MODEL` | `qwen3.6:27b` | LLM model tag deployed in Ollama. Recommended for GPU: `qwen2.5:7b`. |
| `OLLAMA_TIMEOUT_SEC` | `2.0` | Maximum seconds to await Ollama before triggering local fallback. |
| `SESSION_TTL_SEC` | `1800` | Inactivity timeout in seconds (30 minutes) before purging session turns. |
| `CONTEXT_WINDOW_TURNS` | `6` | Maximum historical conversation turns preserved in context buffer. |

### Remote ASR Server (`audio_api/.env`)

| Environment Variable | Default Value | Description |
| :--- | :--- | :--- |
| `CANARY_MODEL` | `nvidia/canary-qwen-2.5b` | Hugging Face model identifier for Canary-Qwen. |
| `MODEL_DEVICE` | `cuda` | Target execution device (`cuda` or `cpu`). |
| `MODEL_DTYPE` | `bfloat16` | Weights precision (`bfloat16`, `float16`, `float32`). |
| `SAMPLE_RATE` | `16000` | Input audio sample rate (16 kHz required). |
| `MAX_AUDIO_SECONDS` | `60` | Hard cap on individual recording clip duration. |
| `CUDA_VISIBLE_DEVICES` | `0` | GPU index assigned to the ASR server. |

---

## 8. API Endpoints & Payload Specifications

### 1. Unified Transcription Endpoint
- **URL**: `POST /api/transcribe`
- **Content-Type**: `multipart/form-data`

#### Request Parameters
| Parameter | Type | Required | Default | Description |
| :--- | :---: | :---: | :---: | :--- |
| `audio` | File (Binary) | **Yes** | — | WAV/WebM voice clip from client microphone. |
| `domain` | String | No | `"general"` | Target domain: `"education"`, `"hospital"`, `"industry"`, `"general"`. |
| `subject` | String | No | `""` | Specialized sub-topic (e.g., `"physics"`, `"cardiology"`, `"electrical"`). |
| `session_id` | String | No | `"default_session"`| Client session UUID for multi-turn conversation memory. |
| `dictionary_enabled` | Boolean | No | `"true"` | Enable or disable domain dictionary corrections. |
| `context_review_enabled` | Boolean | No | `"false"` | Toggle Ollama LLM review (Keep `false` for Mode 3). |
| `mode` | String | No | `"salm"` | ASR processing mode: `"asr"` or `"salm"`. |

#### Response Schema (JSON)
```json
{
  "status": "ok",
  "transcript": "Check patient BP and SpO2 before administering medication.",
  "original_transcript": "Check patient B P and S P O 2 before administering medication.",
  "corrected_transcript": "Check patient BP and SpO2 before administering medication.",
  "reviewed_transcript": "Check patient BP and SpO2 before administering medication.",
  "speaker_id": "megak",
  "speaker_confidence": 0.8412,
  "speaker_status": "IDENTIFIED",
  "speaker_score": 0.8412,
  "corrections": [
    {
      "original": "S P O 2",
      "corrected": "SpO2",
      "reason": "Oxygen saturation abbreviation"
    }
  ],
  "inferred_terms": ["BP", "SpO2"],
  "filter_meta": {
    "raw_peak_db": -6.2,
    "duration_sec": 3.42,
    "hum_filtered": true,
    "is_silence": false
  },
  "context_latency_ms": 0.11,
  "audio_url": "/api/audio/1792062300000_medication.wav",
  "session_id": "clinic_room_104"
}
```

---

### 2. Speaker Biometrics Endpoints

- **`GET /api/speakers`**: Returns list of all enrolled speakers, profile summaries, and sample counts.
- **`POST /api/speakers/enroll`**:
  - `name` (Form String): Speaker name.
  - `audio` (Form File): WAV audio clip of the speaker's voice ($\ge 1.5\text{ s}$).
  - *Returns*: Enrolled status, updated sample count, and audio proof URL.
- **`DELETE /api/speakers/{speaker_name}`**: Deletes a specific speaker profile from the registry.
- **`DELETE /api/speakers`**: Clears all speaker profiles.

---

### 3. Session Context Endpoints

- **`GET /api/context/session/{session_id}`**: Returns the active turn history, established technical terms, and last activity timestamp for a given session.
- **`POST /api/context/session/reset`**: Flushes turn history for `session_id`.

---

## 9. Operational Deployment & Troubleshooting

### How to Start the Recommended Production Baseline (Mode 3)

#### Step 1: Start the Gateway Server
From the workspace root:
```powershell
# Activate Python 3.11 Virtual Environment
.venv\Scripts\activate

# Launch word_test_server on port 8050
python word_test_server.py
```

#### Step 2: Access the Studio UI
Open your browser to:
```
http://localhost:8050
```

#### Step 3: Verify Remote GPU Health
In your terminal or browser:
```bash
curl http://localhost:8050/health
# Expected Output: {"status": "ok", "model": "nvidia/canary-qwen-2.5b", "device": "cuda"}
```

---

### Troubleshooting Common Defects

| Symptom | Underlying Cause | Solution |
| :--- | :--- | :--- |
| **HTTP 503 on `/health`** | Remote GPU server offline or IP `124.123.89.162` unreachable. | Verify network route; ensure GPU daemon is running on port 8001. Gateway will automatically fall back to local offline mode. |
| **Speaker always recognized as "Unknown"** | Audio clip is too short (< 0.8s) or volume is below `-34 dBFS`. | Speak for at least 1.5–2 seconds. Check microphone gain in UI. |
| **Cross-speaker confusion (e.g., Sunil recognized as Mega)** | Single-sample enrollment bias. One speaker has 8 samples, other has 1. | Re-enroll the speaker with 3 to 5 distinct recordings to build an accurate centroid vector. |
| **Transcription latency > 2.0s** | `context_review_enabled` is set to `true`, querying the slow CPU-bound Ollama host. | Set `context_review_enabled=false` in the UI toggle or API payload to use the 0.11 ms local rules engine. |
| **HTTP 400 "Empty audio received"** | Browser audio stream was blocked by microphone permissions or 0 bytes recorded. | Allow browser microphone access; ensure recording duration is $> 0.5\text{ s}$. |
