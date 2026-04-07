# Outreach Report Generator

An end-to-end pipeline that converts multi-speaker agricultural outreach audio recordings into structured, translated, insight-rich reports. Designed for large-scale rural interactions - where farmers are introduced to an AI agri-chatbot and openly discuss their challenges, practices, and feedback - the system automates everything from raw audio to polished outputs (PDF, Excel, Word) for analysis, documentation, and decision-making.

---

## Overview

The pipeline takes a directory of audio files in a supported Indic language, diarizes speakers, transcribes speech using OpenAI Whisper, translates to English, extracts structured insights, and assembles a PDF report.

```
Audio files (wav / mp3 / m4a / flac)
    └─► Combine & normalize
        └─► Speaker diarization       (pyannote.audio 3.1)
            └─► ASR transcription     (OpenAI Whisper large-v3)
                └─► Translation       (sarvam-translate)
                    └─► Extraction    (LLM-based: insights, participants, terminology, metadata, conclusion)
                        └─► PDF Report (ReportLab)
```

---

## What Changed from the Original

The original pipeline used **IndicConformer** for ASR and **IndicTrans2** for translation. Both have been replaced:

| Component | Original | Current |
|-----------|----------|---------|
| ASR Model | `IndicConformer` (gated HuggingFace, complex setup) | `OpenAI Whisper large-v3` (pip install, no gated access) |
| Translation | `IndicTrans2` | `sarvamai/sarvam-translate` (loaded locally via HuggingFace) |
| Audio loading | `torchaudio` (broken with PyTorch 2.9.1+) | `pydub` + `soundfile` |
| Audio saving | `torchaudio.save` (torchcodec incompatible) | `soundfile.write` |
| Diarization input | File path | Preloaded waveform tensor (bypasses torchcodec) |
| LLM Extraction | `Mistral-3B-Instruct` (HuggingFace ID) | Local path to globally cached `Mistral-7B-Instruct-v0.3` |

---

## Features

- **Multi-speaker diarization** - identifies and separates speakers using `pyannote.audio 3.1`
- **Whisper ASR** - transcribes speech in 10 Indian languages via `openai-whisper large-v3`
- **Neural machine translation** - translates Indic-language transcripts to English using `sarvam-translate`
- **LLM-based extraction** - concurrently extracts farmer questions, challenges, participant info, domain terminology, meeting metadata, and a narrative conclusion
- **PDF report generation** - assembles all extracted content into a formatted report using ReportLab
- **Resumable pipeline** - skip any completed stage (`--skip_combine`, `--skip_asr`, `--skip_translation`) to avoid re-running expensive steps
- **GPU-accelerated** - automatically uses CUDA when available, with explicit memory management between stages

---

## Supported Languages

| Code | Language   | IndicTrans2 Tag | ASR Tag  |
|------|------------|-----------------|----------|
| `pa` | Punjabi    | `pan_Guru`      | `punjabi`|
| `hi` | Hindi      | `hin_Deva`      | `hindi`  |
| `ta` | Tamil      | `tam_Taml`      | `tamil`  |
| `te` | Telugu     | `tel_Telu`      | `telugu` |
| `mr` | Marathi    | `mar_Deva`      | `marathi`|
| `kn` | Kannada    | `kan_Knda`      | `kannada`|
| `gu` | Gujarati   | `guj_Gujr`      | `gujarati`|
| `bn` | Bengali    | `ben_Beng`      | `bengali`|
| `or` | Odia       | `ory_Orya`      | `odia`   |
| `ml` | Malayalam  | `mal_Mlym`      | `malayalam`|

---

## Project Structure

```
outreach-report-generator/
├── main.py                        # Pipeline entry point & CLI
├── pipeline/
│   ├── ingestion/
│   │   └── audio_utils.py         # Audio combining & normalization (pydub + soundfile)
│   ├── diarization/
│   │   └── pyannote_diarizer.py   # Speaker diarization (pyannote 3.1, waveform input)
│   ├── asr/
│   │   ├── indic_conformer.py     # Legacy IndicConformer ASR (not used)
│   │   └── whisper_asr.py         # OpenAI Whisper ASR (active)
│   ├── transcript/
│   │   └── builder.py             # Builds & serializes structured transcripts
│   ├── translation/
│   │   ├── indictrans2.py         # Legacy IndicTrans2 translation (not used)
│   │   └── sarvam_translate.py    # sarvam-translate translation (active)
│   ├── extraction/
│   │   ├── base_llm.py            # Shared LLM model loader (Mistral-7B local path)
│   │   ├── insights.py            # Farmer questions & challenges extractor
│   │   ├── narration.py           # Narrative text generator
│   │   ├── conclusion.py          # Conclusion generator
│   │   ├── metadata.py            # Meeting metadata extractor
│   │   ├── participants.py        # Participant extractor
│   │   └── terminology.py         # Domain terminology extractor
│   └── report/
│       └── assembler.py           # Assembles & saves final report (JSON + PDF)
├── config/                        # Configuration files
├── scripts/
│   └── batch_process.py           # Batch processing utility
├── tests/                         # Test suite
├── assets/
│   └── fonts/                     # Fonts for PDF generation
├── .env.example                   # Environment variable template
├── requirements.txt               # Python dependencies
└── pyproject.toml                 # Project metadata
```

---

## Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (strongly recommended; CPU fallback supported but slow)
- `pip`
- FFmpeg installed on system

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/ShauryaMehraa/Outreach-Report.git
cd Outreach-Report

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install additional dependencies not in requirements.txt
pip install openai-whisper pydub soundfile scipy pyannote.audio indic-transliteration

# 5. Set up environment variables
cp .env.example .env
# Edit .env and fill in your HuggingFace token
```

---

## Configuration

Copy `.env.example` to `.env` and populate:

```env
HF_TOKEN=hf_your_token_here
```

| Service | Purpose | Where to obtain |
|---------|---------|-----------------|
| **HuggingFace Token** | pyannote diarization model (gated) + model downloads | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |

You must also accept the model license at: [huggingface.co/pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)

No Sarvam API key is needed — `sarvam-translate` is loaded locally via HuggingFace transformers.

---

## Usage

```bash
python main.py --input_dir <path_to_audio> --language <lang_code> [options]
```

### Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--input_dir` | ✅ | — | Directory containing audio files (wav, mp3, m4a, flac) |
| `--language` | ✅ | — | Source language code (see Supported Languages table) |
| `--output_dir` | — | `./outputs` | Directory for all outputs |
| `--skip_combine` | — | `false` | Reuse existing `combined.wav` |
| `--skip_asr` | — | `false` | Reuse existing `transcript_raw.json` |
| `--skip_translation` | — | `false` | Reuse existing `transcript_translated.json` |
| `--no_pdf` | — | `false` | Skip PDF generation; save JSON outputs only |

### Examples

```bash
# Basic run - Punjabi audio
python main.py --input_dir ./audio --language pa

# Custom output directory - Hindi audio
python main.py --input_dir ./audio --language hi --output_dir ./outputs/meeting_001

# Skip combine only (reuse existing combined.wav)
python main.py --input_dir ./audio --language pa --skip_combine

# Skip ASR (reuse existing transcript_raw.json)
python main.py --input_dir ./audio --language pa --skip_combine --skip_asr

# Skip ASR and translation (jump straight to extraction)
python main.py --input_dir ./audio --language pa --skip_combine --skip_asr --skip_translation

# Skip PDF generation
python main.py --input_dir ./audio --language hi --no_pdf
```

### Running on JupyterHub (shared server)

If models are cached globally on the server (e.g. at `/models/huggingface`), set these environment variables before running to avoid downloading and to save disk space:

```bash
export HF_HOME=/tmp/hf_cache_yourname
export TRANSFORMERS_CACHE=/tmp/hf_cache_yourname
export HF_HUB_CACHE=/models/huggingface
python main.py --input_dir ./audio --language pa --output_dir ./outputs/test1
```

---

## Pipeline Stages

### Stage 1 — Audio Ingestion
All audio files in `--input_dir` are loaded using `pydub`, converted to 16kHz mono float32, combined into a single normalized WAV file (`combined.wav`), and saved using `soundfile`.

### Stage 2 — Speaker Diarization
`pyannote/speaker-diarization-3.1` identifies speaker turns. Audio is preloaded as a waveform tensor and passed directly to the pipeline to avoid torchcodec compatibility issues with PyTorch 2.9+.

### Stage 3 — ASR Transcription
`openai-whisper large-v3` transcribes each speaker segment in the source Indic language, producing a structured raw transcript (`transcript_raw.json`). Speaker turns shorter than 1.5 seconds are skipped.

### Stage 4 — Translation
`sarvamai/sarvam-translate` translates each transcript segment from the source Indic language to English in batches of 32, producing `transcript_translated.json`.

### Stage 5 — Extraction (Concurrent)
An LLM (`Mistral-7B-Instruct-v0.3`) processes the translated transcript to concurrently extract:
- **Terminology** - domain-specific vocabulary from the session
- **Narration** - a flowing narrative summary of the session
- **Insights** - farmer questions and challenges raised during the session
- **Participants** - count and roles of attendees
- **Metadata** - meeting date, location, topic, and other structured fields
- **Conclusion** - a synthesized conclusion drawn from all extracted content

### Stage 6 — Report Assembly
All extracted data is assembled into a structured JSON and a formatted PDF report (`outreach_report.pdf`) via ReportLab.

---

## Outputs

All outputs are saved to `--output_dir` (default: `./outputs`):

| File | Description |
|------|-------------|
| `combined.wav` | Merged and normalized input audio |
| `transcript_raw.json` | Speaker-diarized Whisper transcript in source language |
| `transcript_translated.json` | English-translated transcript |
| `outreach_report.pdf` | Final assembled PDF report (unless `--no_pdf`) |

---

## GPU Memory Requirements

The pipeline loads models sequentially with explicit cleanup between stages:

| Stage | Model | Approx. VRAM |
|-------|-------|-------------|
| Diarization | pyannote 3.1 | ~1 GB |
| ASR | Whisper large-v3 | ~5 GB |
| Translation | sarvam-translate (bfloat16) | ~7 GB |
| Extraction | Mistral-7B (bfloat16) | ~14 GB |

A GPU with at least 16GB VRAM (e.g. NVIDIA A100, H100, H200) is recommended for smooth end-to-end runs.

---

## Key Dependencies

| Package | Purpose |
|---------|---------|
| `torch` | Deep learning backend |
| `openai-whisper` | ASR transcription |
| `transformers` | sarvam-translate + Mistral extraction |
| `pyannote-audio` | Speaker diarization |
| `pydub` | Audio file loading (mp3, m4a, flac) |
| `soundfile` | Audio file saving (wav) |
| `scipy` | Audio resampling |
| `reportlab` | PDF generation |
| `accelerate` | HuggingFace model acceleration |
| `indic-transliteration` | Indic script handling in terminology extraction |
| `python-dotenv` | Environment variable management |
| `rapidfuzz` | Fuzzy deduplication in extraction |

---

## Known Issues & Workarounds

| Issue | Workaround Applied |
|-------|--------------------|
| `torchaudio` broken with PyTorch 2.9.1+ (torchcodec) | Replaced with `pydub` + `soundfile` |
| `pyannote` fails to read WAV files via torchcodec | Audio preloaded as waveform tensor dict |
| `torchaudio.set_audio_backend` removed in new versions | Removed entirely, using soundfile directly |
| Disk space exhaustion on shared JupyterHub | Set `HF_HUB_CACHE` to globally cached model path |
| Mistral model shard missing from global cache | Use full local path to available snapshot |

---

## Future Improvements

* **ASR robustness** — performance drops for low-resource dialects, long/noisy field recordings, and domain-specific agricultural vocabulary
* **Translation fidelity** — some loss of nuance when converting from Indic languages to English
* **LLM extraction consistency** — variability across runs in structured outputs
* **Hallucinations & accuracy** — occasional generation of unsupported facts
* **Insight quality** — irrelevant or noisy outputs in question/challenge extraction

---

## License

See [LICENSE](LICENSE) for details.